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

class TissueHypernetwork(nn.Module):
    def __init__(self, num_tissues, in_features=256, embed_dim=64, sex_embed=False):
        super().__init__()
        self.tissue_embedding = nn.Embedding(num_tissues, embed_dim)
        self.sex_embed = sex_embed
        
        if sex_embed:
            self.sex_embedding = nn.Embedding(2, 8)
            input_dim = embed_dim + 8
        else:
            input_dim = embed_dim
            
        # 생성해야 할 최종 파라미터 개수: Weight(in_features * 1) + Bias(1)
        self.target_weight_dim = in_features
        self.target_bias_dim = 1
        num_target_params = self.target_weight_dim + self.target_bias_dim
        
        # Hypernetwork MLP
        self.hyper_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_target_params)
        )
        
        # 초기화: 생성되는 파라미터의 스케일을 줄여 초기 학습 폭발(Explosion) 방지
        nn.init.xavier_uniform_(self.hyper_net[-1].weight, gain=0.01)
        nn.init.zeros_(self.hyper_net[-1].bias)

    def forward(self, features, tissue_id, sex=None):
        """
        Args:
            features: (batch_size, in_features) - 공통 추출기를 통과한 Bag-level 피처
            tissue_id: (batch_size,)
            sex: (batch_size,)
        Returns:
            prediction: (batch_size,)
        """
        B = features.size(0)
        
        # 1. Condition Embedding 결합
        t_emb = self.tissue_embedding(tissue_id) # (B, embed_dim)
        if self.sex_embed and sex is not None:
            s_emb = self.sex_embedding(sex)
            combined_emb = torch.cat((t_emb, s_emb), dim=-1)
        else:
            combined_emb = t_emb
            
        # 2. 동적 파라미터 생성
        dynamic_params = self.hyper_net(combined_emb) # (B, in_features + 1)
        
        # Weight와 Bias 분리 (1차원 출력이므로 out_features=1)
        dynamic_weight = dynamic_params[:, :self.target_weight_dim].view(B, 1, self.target_weight_dim)
        dynamic_bias = dynamic_params[:, self.target_weight_dim:].view(B, 1)
        
        # 3. 예측 연산 (y = z * W^T + b)
        features_expanded = features.unsqueeze(1) # (B, 1, in_features)
        weight_transposed = dynamic_weight.transpose(1, 2) # (B, in_features, 1)
        
        y_pred = torch.bmm(features_expanded, weight_transposed).squeeze(1) # (B, 1)
        y_pred = y_pred + dynamic_bias # (B, 1)
        
        return y_pred.squeeze(-1) # (B,)


# ============================================================
# Main Model (수정됨)
# ============================================================

class TissueABMIL(nn.Module):
    # hypernetwork 모드 추가
    VALID_COND_MODES = ('none', 'concat', 'film', 'cond_regressor', 'hypernetwork')

    def __init__(self, num_tissues, feature_dim=1536, head_dim=256, n_heads=4, gated=True,
                 tissue_cond_mode='hypernetwork', tissue_cond_embed_dim=64,
                 tissue_embed=False, sex_embed=False):
        super(TissueABMIL, self).__init__()
        
        assert tissue_cond_mode in self.VALID_COND_MODES, \
            f"Invalid tissue_cond_mode '{tissue_cond_mode}'. Must be one of {self.VALID_COND_MODES}"
        
        self.feature_dim = feature_dim
        self.tissue_cond_mode = tissue_cond_mode
        self.sex_embed = sex_embed
        
        if tissue_embed and tissue_cond_mode == 'none':
            tissue_cond_mode = 'concat'

        if tissue_embed and tissue_cond_mode == 'concat':
            self.tissue_embedding = nn.Embedding(num_embeddings=num_tissues, embedding_dim=tissue_cond_embed_dim)
            combined_dim = feature_dim + tissue_cond_embed_dim
        else:
            combined_dim = feature_dim
        
        # New conditioning modules
        self.tissue_hypernetwork = TissueHypernetwork(num_tissues, in_features=256, embed_dim=tissue_cond_embed_dim, sex_embed=sex_embed)

        self.shared_feature_extractor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25)
        )
            
        # Feature Aggregation
        self.abmil = ABMIL(
            feature_dim=combined_dim, 
            head_dim=head_dim, 
            n_heads=n_heads, 
            dropout=0.25, 
            n_branches=1, 
            gated=gated
        )
        
        # Regressor (used as fallback or for standard modes)
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )


    def forward(self, features, attn_mask=None, tissue_id=None, sex=None, return_features=False):

        # Attention MIL Pooling
        aggregated_features, head_attentions = self.abmil(features, attn_mask)
        M = aggregated_features.squeeze(1) # (batch_size, feature_dim)
        
        # Prediction
        if self.tissue_cond_mode == 'hypernetwork' and tissue_id is not None:
            shared_features = self.shared_feature_extractor(M) 
            Y_pred = self.tissue_hypernetwork(shared_features, tissue_id, sex if self.sex_embed else None)
            
        else:
            Y_pred = self.regressor(M).squeeze(-1)
        
        if return_features:
            return Y_pred, head_attentions, M
        
        return Y_pred, head_attentions