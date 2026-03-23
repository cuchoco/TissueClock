import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class ABMIL(nn.Module):
    """
    Multi-headed attention network with optional gating. Uses tanh-attention and sigmoid-gating as in ABMIL (https://arxiv.org/abs/1802.04712).
    Note that this is different from canonical attention in that the attention scores are computed directly by a linear layer rather than by a dot product between queries and keys.

    Args:
        feature_dim (int): Input feature dimension
        head_dim (int): Hidden layer dimension for each attention head. Defaults to 256.
        n_heads (int): Number of attention heads. Defaults to 8.
        dropout (float): Dropout probability. Defaults to 0.
        n_branches (int): Number of attention branches. Defaults to 1, but can be set to n_classes to generate one set of attention scores for each class.
        gated (bool): If True, sigmoid gating is applied. Otherwise, the simple attention mechanism is used.
    """

    def __init__(self, feature_dim = 1536, head_dim = 256, n_heads = 8, dropout = 0., n_branches = 1, gated = False):
        super().__init__()
        self.gated = gated
        self.n_heads = n_heads

        # Initialize attention head(s)
        self.attention_heads = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim, head_dim),
                                                               nn.Tanh(),
                                                               nn.Dropout(dropout)) for _ in range(n_heads)])
        
        # Initialize gating layers if gating is used
        if self.gated:
            self.gating_layers = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim, head_dim),
                                                                   nn.Sigmoid(),
                                                                   nn.Dropout(dropout)) for _ in range(n_heads)])
        
        # Initialize branching layers
        self.branching_layers = nn.ModuleList([nn.Linear(head_dim, n_branches) for _ in range(n_heads)])

        # Initialize condensing layer if multiple heads are used
        if n_heads > 1:
            self.condensing_layer = nn.Linear(n_heads * feature_dim, feature_dim)
        
    def forward(self, features, attn_mask = None):
        """
        Forward pass

        Args:
            features (torch.Tensor): Input features, acting as queries and values. Shape: batch_size x num_images x feature_dim
            attn_mask (torch.Tensor): Attention mask to enforce zero attention on empty images. Defaults to None. Shape: batch_size x num_images

        Returns:
            aggregated_features (torch.Tensor): Attention-weighted features aggregated across heads. Shape: batch_size x n_branches x feature_dim
        """

        assert features.dim() == 3, f'Input features must be 3-dimensional (batch_size x num_images x feature_dim). Got {features.shape} instead.'
        if attn_mask is not None:
            assert attn_mask.dim() == 2, f'Attention mask must be 2-dimensional (batch_size x num_images). Got {attn_mask.shape} instead.'
            assert features.shape[:2] == attn_mask.shape, f'Batch size and number of images must match between features and mask. Got {features.shape[:2]} and {attn_mask.shape} instead.'

        # Get attention scores for each head
        head_attentions = []
        head_features = []
        for i in range(len(self.attention_heads)):
            attention_vectors = self.attention_heads[i](features)        # Main attention vectors (shape: batch_size x num_images x head_dim)
            
            if self.gated:
                gating_vectors = self.gating_layers[i](features)                # Gating vectors (shape: batch_size x num_images x head_dim)
                attention_vectors = attention_vectors.mul(gating_vectors)       # Element-wise multiplication to apply gating vectors
                
            attention_scores = self.branching_layers[i](attention_vectors)       # Attention scores for each branch (shape: batch_size x num_images x n_branches)

            # Set attention scores for empty images to -inf
            if attn_mask is not None:
                attention_scores = attention_scores.masked_fill(~attn_mask.unsqueeze(-1), -1e9) # Mask is automatically broadcasted to shape: batch_size x num_images x n_branches

            # Softmax attention scores over num_images
            attention_scores_softmax = F.softmax(attention_scores, dim=1) # Shape: batch_size x num_images x n_branches

            # Multiply features by attention scores
            weighted_features = torch.einsum('bnr,bnf->brf', attention_scores_softmax, features) # Shape: batch_size x n_branches x feature_dim

            head_attentions.append(attention_scores)
            head_features.append(weighted_features)

        # Concatenate multi-head outputs and condense
        aggregated_features = torch.cat(head_features, dim=-1) # Shape: batch_size x n_branches x (n_heads * feature_dim)
        if self.n_heads > 1:
            aggregated_features = self.condensing_layer(aggregated_features) # Shape: batch_size x n_branches x feature_dim
        
        # Stack attention scores
        head_attentions = torch.stack(head_attentions, dim=-1) # Shape: batch_size x num_images x n_branches x n_heads
        head_attentions = rearrange(head_attentions, 'b n r h -> b r h n') # Shape: batch_size x n_branches x n_heads x num_images

        return aggregated_features, head_attentions
    

class TissueABMIL(nn.Module):
    def __init__(self, num_tissues, feature_dim=1536, head_dim=256, n_heads=4, gated=True, tissue_embed=False, tissue_embed_dim=16):
        super(TissueABMIL, self).__init__()

        self.tissue_embed = tissue_embed
        if tissue_embed:
            self.tissue_embedding = nn.Embedding(num_embeddings=num_tissues, embedding_dim=tissue_embed_dim)
            combined_dim = feature_dim + tissue_embed_dim
        else:
            tissue_embed_dim = 0
            combined_dim = feature_dim

        # Feature Aggregation
        self.abmil = ABMIL(
            feature_dim=combined_dim, 
            head_dim=head_dim, 
            n_heads=n_heads, 
            dropout=0.25, 
            n_branches=1, 
            gated=gated
        )
        
        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )
        # self.regressor = nn.Linear(combined_dim, 1)


    def forward(self, features, attn_mask=None, tissue_id=None):
        """
        features shape: (batch_size, num_images(patches), feature_dim)
        tissue_id shape: (batch_size)
        """

        if self.tissue_embed and tissue_id is not None:
            T_emb = self.tissue_embedding(tissue_id)
            T_emb = T_emb.unsqueeze(1).expand(-1, features.size(1), -1)
            features = torch.cat((features, T_emb), dim=-1)

        aggregated_features, head_attentions = self.abmil(features, attn_mask)
        
        # aggregated_features shape: (batch_size, n_branches, feature_dim)
        # (batch_size, feature_dim)
        M = aggregated_features.squeeze(1)
        Y_pred = self.regressor(M).squeeze(-1)
        
        return Y_pred, head_attentions

