import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class TransposeTokenReEmbedding:
    @staticmethod
    def transpose_normal_padding(x, rate):
        x = rearrange(x, "b c l -> b l c")
        B, N, C = x.shape
        value = N // rate
        if N % rate != 0:
            padding_length = (value + 1) * rate - N
            padded_x = F.pad(x, (0, 0, 0, padding_length))
        else:
            padded_x = x
        x_ = rearrange(padded_x, "b (k w) d -> b (w k) d", w = rate)
        x_ = rearrange(x_, "b l c -> b c l")
        return x_

    @staticmethod
    def transpose_remove_padding(x, rate, length):
        x = rearrange(x, "b c l -> b l c")
        x = rearrange(x, "b (w k) d -> b (k w) d", w = rate)
        x = x[:,:length,:]
        x = rearrange(x, "b l c -> b c l")
        return x

class SRMambaLayer(nn.Module):
    """
    Sequence Reordering Mamba Layer (SR-Mamba)
    Requires mamba_ssm installed. If not, falls back to a dummy pass or error depending on usage.
    """
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        
        if not HAS_MAMBA:
            raise ImportError("mamba_ssm is required for MambaMIL. Please install it using `pip install mamba-ssm causal-conv1d`.")
            
        # We need two Mamba blocks: one for normal sequence, one for transposed sequence
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mamba_b = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.out_proj = nn.Linear(d_model * expand, d_model) # Assuming Mamba expands and then we project back, but standard Mamba usually handles this internally.
        # Actually, standard Mamba block output is same as d_model. We'll follow the SR-Mamba logic where we add the outputs.

    def forward(self, hidden_states, rate=10):
        # hidden_states: [B, L, D]
        # In official implementation, they use mamba_inner_fn directly to optimize.
        # For simplicity and robustness across mamba-ssm versions, we use two separate Mamba blocks.
        
        # Branch 1: Normal sequence
        out_f = self.mamba(hidden_states)
        
        # Branch 2: Transposed sequence for 2D structure awareness
        # Transpose expects [B, D, L] shape, but hidden_states is [B, L, D]
        x_transpose = rearrange(hidden_states, "b l d -> b d l")
        B, D, L = x_transpose.shape
        x_transpose_padded = TransposeTokenReEmbedding.transpose_normal_padding(x_transpose, rate)
        
        # Mamba expects [B, L, D]
        x_transpose_padded_blh = rearrange(x_transpose_padded, "b d l -> b l d")
        out_b_padded = self.mamba_b(x_transpose_padded_blh)
        
        # Convert back to [B, D, L] to unpad
        out_b_padded_bdh = rearrange(out_b_padded, "b l d -> b d l")
        out_b = TransposeTokenReEmbedding.transpose_remove_padding(out_b_padded_bdh, rate, L)
        
        # Back to [B, L, D]
        out_b = rearrange(out_b, "b d l -> b l d")
        
        return out_f + out_b


class TissueMambaMIL(nn.Module):
    """
    MambaMIL adapted for Age Prediction (Regression) and optional Tissue Embedding.
    """
    def __init__(self, num_tissues, in_dim=1536, dim=512, layer=2, rate=10, 
                 tissue_embed=False, tissue_embed_dim=16, 
                 d_state=16, d_conv=4, expand=2):
        super(TissueMambaMIL, self).__init__()
        self.tissue_embed = tissue_embed
        self.rate = rate
        
        if tissue_embed:
            self.tissue_embedding = nn.Embedding(num_embeddings=num_tissues, embedding_dim=dim)
            self.num_extra_tokens = 2 # CLS + Tissue
        else:
            self.num_extra_tokens = 1 # CLS
            
        self._fc1 = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU())
        
        self.layers = nn.ModuleList()
        for _ in range(layer):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    SRMambaLayer(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
                )
            )
        
        self.norm = nn.LayerNorm(dim)
        
        # MambaMIL aggregation (Attention-based pooling similar to ABMIL)
        self.attention = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Regressor layer for Age Prediction
        self._fc2 = nn.Linear(dim, 1)

    def forward(self, features, attn_mask=None, tissue_id=None):
        """
        features shape: (batch_size, num_images(patches), in_dim)
        tissue_id shape: (batch_size)
        """
        B, N_patches, _ = features.shape
        h = features.float()
        
        h = self._fc1(h) # [B, N, dim]
        
        # Optional: Append Tissue Token if provided
        if self.tissue_embed and tissue_id is not None:
            tissue_tokens = self.tissue_embedding(tissue_id).unsqueeze(1) # [B, 1, dim]
            h = torch.cat((tissue_tokens, h), dim=1)

        # SR-Mamba Layers
        for layer in self.layers:
            h_residual = h
            h = layer[0](h) # LayerNorm
            h = layer[1](h, rate=self.rate) # SRMambaLayer
            h = h + h_residual

        h = self.norm(h)
        
        # Attention Pooling (MambaMIL uses this instead of just taking the CLS token)
        # We pool over all tokens (including CLS and Tissue tokens if present)
        A = self.attention(h) # [B, n, 1]
        A = torch.transpose(A, 1, 2) # [B, 1, n]
        A = F.softmax(A, dim=-1)
        
        # Aggregated representation
        h_agg = torch.bmm(A, h).squeeze(1) # [B, dim]
        
        # Predict Age
        Y_pred = self._fc2(h_agg).squeeze(-1) # [B]
        
        head_attentions = A.squeeze(1) # [B, n]
        
        return Y_pred, head_attentions
