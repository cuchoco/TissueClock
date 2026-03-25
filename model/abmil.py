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

# ============================================================
# Tissue Conditioning Modules
# ============================================================

class TissueFiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioned on tissue type.
    Generates scale (gamma) and shift (beta) to modulate features without
    changing the feature dimension.
    
    Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
    """
    def __init__(self, num_tissues, feature_dim, embed_dim=64):
        super().__init__()
        self.tissue_embedding = nn.Embedding(num_tissues, embed_dim)
        self.film_generator = nn.Sequential(
            nn.Linear(embed_dim, feature_dim * 2),  # generate gamma and beta together
        )
        # Initialize gamma close to 1 and beta close to 0
        nn.init.zeros_(self.film_generator[0].weight)
        nn.init.zeros_(self.film_generator[0].bias)
        # Set gamma bias to 1 (first half of output)
        with torch.no_grad():
            self.film_generator[0].bias[:feature_dim] = 1.0
    
    def forward(self, features, tissue_id):
        """
        Args:
            features: (batch_size, num_patches, feature_dim)
            tissue_id: (batch_size,)
        Returns:
            modulated features: (batch_size, num_patches, feature_dim)
        """
        emb = self.tissue_embedding(tissue_id)  # (B, embed_dim)
        film_params = self.film_generator(emb)   # (B, feature_dim * 2)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, feature_dim)
        
        # Broadcast over num_patches: (B, 1, feature_dim)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        return gamma * features + beta

class TissueCondRegressor(nn.Module):
    """
    Tissue-conditioned regressor. The base prediction is made from features,
    then tissue-specific scale and bias are applied.
    This allows different tissues to have different aging rates and baselines.
    """
    def __init__(self, num_tissues, feature_dim, embed_dim=64):
        super().__init__()
        self.tissue_embedding = nn.Embedding(num_tissues, embed_dim)
        
        # Base regressor (tissue-agnostic)
        self.base_regressor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )
        
        # Tissue-conditioned scale and bias for the prediction
        self.cond_generator = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # scale and bias
        )
        # Initialize so scale ≈ 1 and bias ≈ 0
        nn.init.zeros_(self.cond_generator[2].weight)
        nn.init.constant_(self.cond_generator[2].bias, 0.0)
        with torch.no_grad():
            self.cond_generator[2].bias[0] = 1.0  # scale = 1
    
    def forward(self, features, tissue_id):
        """
        Args:
            features: (batch_size, feature_dim)
            tissue_id: (batch_size,)
        Returns:
            prediction: (batch_size,)
        """
        base_pred = self.base_regressor(features).squeeze(-1)  # (B,)
        
        emb = self.tissue_embedding(tissue_id)  # (B, embed_dim)
        cond = self.cond_generator(emb)          # (B, 2)
        scale, bias = cond[:, 0], cond[:, 1]
        
        return scale * base_pred + bias


# ============================================================
# Main Model
# ============================================================

class TissueABMIL(nn.Module):
    """
    ABMIL model with flexible tissue conditioning strategies.
    
    Args:
        num_tissues: Number of tissue types
        feature_dim: Input feature dimension (default: 1536 for UNI v2)
        head_dim: Attention head hidden dimension
        n_heads: Number of attention heads
        gated: Whether to use gated attention
        tissue_cond_mode: Tissue conditioning strategy. One of:
            - 'none': No tissue conditioning
            - 'concat': Original concatenation approach (legacy)
            - 'film': FiLM conditioning on features (recommended)
            - 'attn_bias': Tissue-conditioned attention bias
            - 'cond_regressor': Tissue-conditioned prediction head
        tissue_embed_dim: Dimension of tissue embedding (for concat mode)
        tissue_cond_embed_dim: Dimension of tissue embedding (for new conditioning modes)
    """
    VALID_COND_MODES = ('none', 'concat', 'film', 'cond_regressor')

    def __init__(self, num_tissues, feature_dim=1536, head_dim=256, n_heads=4, gated=True,
                 tissue_cond_mode='none', tissue_cond_embed_dim=64,
                 # Legacy compatibility
                 tissue_embed=False):
        super(TissueABMIL, self).__init__()
        
        # Legacy compatibility: if tissue_embed=True and tissue_cond_mode='none', use 'concat'
        if tissue_embed and tissue_cond_mode == 'none':
            tissue_cond_mode = 'concat'
        
        assert tissue_cond_mode in self.VALID_COND_MODES, \
            f"Invalid tissue_cond_mode '{tissue_cond_mode}'. Must be one of {self.VALID_COND_MODES}"
        
        self.tissue_cond_mode = tissue_cond_mode
        self.feature_dim = feature_dim
        
        # Legacy concat mode
        self.tissue_embed = (tissue_cond_mode == 'concat')
        if self.tissue_embed:
            self.tissue_embedding = nn.Embedding(num_embeddings=num_tissues, embedding_dim=tissue_cond_embed_dim)
            combined_dim = feature_dim + tissue_cond_embed_dim
        else:
            combined_dim = feature_dim
        
        # New conditioning modules
        if tissue_cond_mode == 'film':
            self.tissue_film = TissueFiLM(num_tissues, feature_dim, embed_dim=tissue_cond_embed_dim)
        elif tissue_cond_mode == 'cond_regressor':
            self.tissue_cond_regressor = TissueCondRegressor(num_tissues, feature_dim, embed_dim=tissue_cond_embed_dim)

        # Feature Aggregation
        self.abmil = ABMIL(
            feature_dim=combined_dim, 
            head_dim=head_dim, 
            n_heads=n_heads, 
            dropout=0.25, 
            n_branches=1, 
            gated=gated
        )
        
        # Regressor (used as fallback when tissue_id=None in cond_regressor mode)
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )


    def forward(self, features, attn_mask=None, tissue_id=None, return_features=False):
        """
        features shape: (batch_size, num_images(patches), feature_dim)
        tissue_id shape: (batch_size)
        """

        # Apply tissue conditioning
        if self.tissue_cond_mode == 'concat' and tissue_id is not None:
            T_emb = self.tissue_embedding(tissue_id)
            T_emb = T_emb.unsqueeze(1).expand(-1, features.size(1), -1)
            features = torch.cat((features, T_emb), dim=-1)
        elif self.tissue_cond_mode == 'film' and tissue_id is not None:
            features = self.tissue_film(features, tissue_id)

        aggregated_features, head_attentions = self.abmil(features, attn_mask)
        
        # aggregated_features shape: (batch_size, n_branches, feature_dim)
        # (batch_size, feature_dim)
        M = aggregated_features.squeeze(1)
        
        if self.tissue_cond_mode == 'cond_regressor' and tissue_id is not None:
            Y_pred = self.tissue_cond_regressor(M, tissue_id)
        else:
            Y_pred = self.regressor(M).squeeze(-1)
        
        if return_features:
            return Y_pred, head_attentions, M
        
        return Y_pred, head_attentions
