"""
Perceiver MIL Model for TissueClock Age Prediction.
Based on the Perceiver architecture from Prism, adapted as a standalone MIL model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel


# ============================================================================
# Perceiver components (extracted from Prism, no external dependencies)
# ============================================================================

class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: int,
        head_dim: int,
        heads: int,
        c_norm: bool = True,
    ) -> None:
        super().__init__()

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.head_dim = head_dim
        self.heads = heads
        self.inner_dim = self.head_dim * self.heads

        self.x_norm = nn.LayerNorm(self.query_dim)
        self.c_norm = nn.LayerNorm(self.context_dim) if c_norm else None

        self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(self.context_dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, self.query_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        c: Tensor | None = None,
        kvt: tuple[Tensor, Tensor] | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Args:
            x: queries (B, Nx, query_dim)
            c: context (B, Nc, context_dim)
            kvt: key-value cache
            attn_mask: mask for context padding
        Returns:
            output, kv-cache
        """
        Bx, Nx, _ = x.shape

        x = self.x_norm(x)
        c = self.c_norm(c) if self.c_norm is not None else c

        q: Tensor = self.to_q(x)
        q = q.reshape(Bx, Nx, self.heads, self.head_dim)

        if c is not None and kvt is None:
            Bc, Nc, _ = c.shape
            kv: Tensor = self.to_kv(c)
            kv = kv.reshape(Bc, Nc, 2, self.heads, self.head_dim)
            k, v = kv.unbind(2)
            kvt = (k, v)
        elif kvt is not None and c is None:
            k, v = kvt
            Bc, Nc, _, _ = k.shape
        else:
            raise Exception(f'XOR(c, kvt) but got: {type(c)} and {type(kvt)}.')

        if attn_mask is not None:
            attn_mask = attn_mask.reshape(Bc, 1, Nx, Nc).expand(-1, self.heads, -1, -1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        with sdpa_kernel(SDPBackend.MATH):
            a: Tensor = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        a = a.transpose(1, 2)

        c_out = a.reshape(Bx, Nx, self.inner_dim)
        o = self.to_out(c_out)

        return o, kvt


class MHSA(nn.Module):
    def __init__(self, *, dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=dim,
            vdim=dim,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        with sdpa_kernel([SDPBackend.MATH]):
            x, _ = self.mha(x, x, x, need_weights=False)
        return x


class GEGLU(nn.Module):
    def forward(self, x: Tensor):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, *, dim: int, mult: int = 1, dropout: float = 0.0, activation: str = 'geglu'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        extra_dim = 1
        if activation == 'geglu':
            actfn = GEGLU
            extra_dim = 2
        elif activation == 'gelu':
            actfn = nn.GELU
        else:
            raise ValueError(f'{activation=} not supported.')

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * extra_dim),
            actfn(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        return self.net(self.norm(x))


class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        latent_seq: int = 64,
        latent_dim: int = 512,
        context_dim: int = 1536,
        mhsa_heads: int = 8,
        perceiver_depth: int = 4,
        transformer_depth: int = 2,
        share_xattn_start_layer: int = 1,
        share_tf_start_layer: int = 0,
        xattn_heads: int = 8,
        mlp_mult: int = 1,
        mlp_activation: str = 'geglu',
    ):
        super().__init__()

        assert perceiver_depth > 0
        assert share_xattn_start_layer >= 0
        assert share_tf_start_layer >= 0

        self.share_xattn_start_layer = share_xattn_start_layer
        self.share_tf_start_layer = share_tf_start_layer
        self.latent_seq = latent_seq
        self.mhsa_heads = mhsa_heads

        latent_weights = torch.randn(latent_seq, latent_dim)
        self.latents = nn.Parameter(latent_weights)

        get_xattn = lambda: nn.ModuleDict({
            'xattn': CrossAttention(
                query_dim=latent_dim,
                context_dim=context_dim,
                head_dim=latent_dim // xattn_heads,
                heads=xattn_heads,
                c_norm=False,  # norming large contexts explodes memory
            ),
            'ff': FeedForward(
                dim=latent_dim,
                mult=mlp_mult,
                activation=mlp_activation,
            ),
        })

        get_mhsa = lambda: nn.ModuleDict({
            'mhsa': MHSA(dim=latent_dim, num_heads=mhsa_heads),
            'ff': FeedForward(dim=latent_dim, mult=mlp_mult, activation=mlp_activation),
        })

        get_transformer = lambda: nn.ModuleList([get_mhsa() for _ in range(transformer_depth)])

        layers = []
        for i in range(perceiver_depth):
            layer = nn.ModuleDict({
                'xattn': get_xattn() if i <= self.share_xattn_start_layer else layers[-1]['xattn'],
                'tf': get_transformer() if i <= self.share_tf_start_layer else layers[-1]['tf'],
            })
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, context: Tensor, attn_mask: Tensor) -> Tensor:
        """
        Args:
            context: (batch, seq_len, feature_dim)
            attn_mask: (batch, seq_len) bool mask for padding

        Returns:
            processed latent queries (batch, latent_seq, latent_dim)
        """
        B, N, _ = context.shape

        assert len(attn_mask.shape) == 2
        assert attn_mask.shape[:1] == context.shape[:1]

        attn_mask = attn_mask.reshape(B, 1, N).expand(-1, self.latent_seq, -1)

        x = self.latents.unsqueeze(0).expand(B, -1, -1)

        kvt = None
        for i, l in enumerate(self.layers):
            xattn = l['xattn']
            if i <= self.share_xattn_start_layer:
                xattn_out, kvt = xattn['xattn'](x, c=context, kvt=None, attn_mask=attn_mask)
            else:
                xattn_out, kvt = xattn['xattn'](x, c=None, kvt=kvt, attn_mask=attn_mask)

            x = xattn_out + x
            x = xattn['ff'](x) + x

            tf = l['tf']
            for mhsa in tf:
                x = mhsa['mhsa'](x) + x
                x = mhsa['ff'](x) + x

        return x


# ============================================================================
# TissuePerceiverMIL — MIL model for age regression
# ============================================================================

class TissuePerceiverMIL(nn.Module):
    """
    Perceiver-based MIL model for Age Prediction (Regression).
    Uses learnable latent queries to cross-attend to patch embeddings,
    then pools the first latent as a CLS embedding for regression.
    """
    def __init__(
        self,
        num_tissues: int,
        in_dim: int = 1536,
        latent_seq: int = 64,
        latent_dim: int = 512,
        tissue_embed: bool = False,
        perceiver_depth: int = 4,
        transformer_depth: int = 2,
        xattn_heads: int = 8,
        mhsa_heads: int = 8,
        share_xattn_start_layer: int = 1,
        share_tf_start_layer: int = 0,
        mlp_mult: int = 1,
        mlp_activation: str = 'geglu',
    ):
        super().__init__()

        self.tissue_embed_flag = tissue_embed
        self.latent_dim = latent_dim

        # Input projection
        self._fc1 = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ReLU())

        # Optional tissue embedding
        if tissue_embed:
            self.tissue_embedding = nn.Embedding(
                num_embeddings=num_tissues, embedding_dim=latent_dim
            )

        # Perceiver core
        self.perceiver = Perceiver(
            latent_seq=latent_seq + 1,  # +1 for CLS latent query
            latent_dim=latent_dim,
            context_dim=latent_dim,     # after projection
            mhsa_heads=mhsa_heads,
            perceiver_depth=perceiver_depth,
            transformer_depth=transformer_depth,
            share_xattn_start_layer=share_xattn_start_layer,
            share_tf_start_layer=share_tf_start_layer,
            xattn_heads=xattn_heads,
            mlp_mult=mlp_mult,
            mlp_activation=mlp_activation,
        )

        self.cls_norm = nn.LayerNorm(latent_dim)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )

        self.apply(self._init_weights)

    def forward(self, features, attn_mask=None, tissue_id=None):
        """
        Args:
            features: (batch_size, num_patches, in_dim)
            attn_mask: (batch_size, num_patches) bool mask
            tissue_id: (batch_size,) tissue index
        Returns:
            Y_pred: (batch_size,)
            head_attentions: None (Perceiver uses implicit attention)
        """
        B, N, _ = features.shape
        h = features.float()

        # Project input features
        h = self._fc1(h)  # [B, N, latent_dim]

        # Optional: prepend tissue token
        if self.tissue_embed_flag and tissue_id is not None:
            tissue_tokens = self.tissue_embedding(tissue_id).unsqueeze(1)  # [B, 1, latent_dim]
            h = torch.cat((tissue_tokens, h), dim=1)  # [B, N+1, latent_dim]
            N = N + 1

        # Build attention mask
        if attn_mask is None:
            attn_mask = torch.ones(B, N, device=h.device, dtype=torch.bool)
        elif self.tissue_embed_flag and tissue_id is not None:
            # Extend mask for tissue token (always valid)
            tissue_mask = torch.ones(B, 1, device=h.device, dtype=torch.bool)
            attn_mask = torch.cat((tissue_mask, attn_mask), dim=1)

        # Perceiver: cross-attend latent queries to patch embeddings
        x = self.perceiver(context=h, attn_mask=attn_mask)  # [B, latent_seq+1, latent_dim]

        # Use first latent as CLS embedding
        cls_emb = x[:, 0]  # [B, latent_dim]
        cls_emb = self.cls_norm(cls_emb)

        # Predict age
        Y_pred = self.regressor(cls_emb).squeeze(-1)  # [B]

        head_attentions = None

        return Y_pred, head_attentions

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            init.trunc_normal_(m.weight, std=0.02)
