import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, 
                 nystrom_dim_head=64, nystrom_heads=8, 
                 nystrom_num_landmarks=256, nystrom_pinv_iterations=6, 
                 nystrom_residual=True):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = nystrom_dim_head,
            heads = nystrom_heads,
            num_landmarks = nystrom_num_landmarks,
            pinv_iterations = nystrom_pinv_iterations,
            residual = nystrom_residual,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class PPEG(nn.Module):
    def __init__(self, dim=512, num_extra_tokens=1, ppeg_kernel_sizes=[7, 5, 3]):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, ppeg_kernel_sizes[0], 1, ppeg_kernel_sizes[0]//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, ppeg_kernel_sizes[1], 1, ppeg_kernel_sizes[1]//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, ppeg_kernel_sizes[2], 1, ppeg_kernel_sizes[2]//2, groups=dim)
        self.num_extra_tokens = num_extra_tokens

    def forward(self, x, H, W):
        B, _, C = x.shape
        extra_tokens, feat_token = x[:, :self.num_extra_tokens], x[:, self.num_extra_tokens:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((extra_tokens, x), dim=1)
        return x

class TissueTransMIL(nn.Module):
    def __init__(self, num_tissues, in_dim=1536, dim=512, tissue_embed=False, tissue_embed_dim=16,
                 nystrom_dim_head=64, nystrom_heads=8, nystrom_num_landmarks=256, 
                 nystrom_pinv_iterations=6, nystrom_residual=True, ppeg_kernel_sizes=[7, 5, 3]):
        super(TissueTransMIL, self).__init__()
        
        self.tissue_embed = tissue_embed
        if tissue_embed:
            # We treat the tissue embedding as a special token, so it must match 'dim'
            self.tissue_embedding = nn.Embedding(num_embeddings=num_tissues, embedding_dim=dim)
            num_extra_tokens = 2 # CLS token + Tissue token
        else:
            num_extra_tokens = 1 # CLS token only
            
        self.pos_layer = PPEG(dim=dim, num_extra_tokens=num_extra_tokens, ppeg_kernel_sizes=ppeg_kernel_sizes)
        self._fc1 = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.layer1 = TransLayer(dim=dim, nystrom_dim_head=nystrom_dim_head, nystrom_heads=nystrom_heads, 
                                 nystrom_num_landmarks=nystrom_num_landmarks, nystrom_pinv_iterations=nystrom_pinv_iterations, 
                                 nystrom_residual=nystrom_residual)
        self.layer2 = TransLayer(dim=dim, nystrom_dim_head=nystrom_dim_head, nystrom_heads=nystrom_heads, 
                                 nystrom_num_landmarks=nystrom_num_landmarks, nystrom_pinv_iterations=nystrom_pinv_iterations, 
                                 nystrom_residual=nystrom_residual)
        self.norm = nn.LayerNorm(dim)
        
        # Regressor layer for Age Prediction
        self._fc2 = nn.Linear(dim, 1)

    def forward(self, features, attn_mask=None, tissue_id=None):
        """
        features shape: (batch_size, num_images(patches), in_dim)
        tissue_id shape: (batch_size)
        """
        h = features.float()
            
        h = self._fc1(h) #[B, n, dim]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            h = torch.cat([h, h[:,:add_length,:]], dim=1) #[B, N, dim]

        #---->cls_token & tissue_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        
        if self.tissue_embed and tissue_id is not None:
            tissue_tokens = self.tissue_embedding(tissue_id).unsqueeze(1) # [B, 1, dim]
            h = torch.cat((cls_tokens, tissue_tokens, h), dim=1)
        else:
            h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, dim]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, dim]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, dim]

        #---->cls_token (we use the updated CLS token for prediction)
        h = self.norm(h)[:,0]

        #---->predict
        Y_pred = self._fc2(h).squeeze(-1) #[B]
        
        head_attentions = None
        
        return Y_pred, head_attentions
