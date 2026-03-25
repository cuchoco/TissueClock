import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.abmil import TissueABMIL, ABMIL, TissueFiLM, TissueCondRegressor


# ============================================================
# Test constants
# ============================================================
BATCH_SIZE = 2
NUM_PATCHES = 50
FEATURE_DIM = 1536
NUM_TISSUES = 29
N_HEADS = 4
HEAD_DIM = 256


# ============================================================
# Helper
# ============================================================
def make_dummy_data(batch_size=BATCH_SIZE, num_patches=NUM_PATCHES):
    features = torch.randn(batch_size, num_patches, FEATURE_DIM)
    tissue_ids = torch.randint(0, NUM_TISSUES, (batch_size,))
    attn_mask = torch.ones(batch_size, num_patches, dtype=torch.bool)
    # Make some patches masked
    attn_mask[:, -5:] = False
    return features, tissue_ids, attn_mask


# ============================================================
# Individual module tests
# ============================================================
def test_tissue_film():
    """Test TissueFiLM module."""
    film = TissueFiLM(NUM_TISSUES, FEATURE_DIM, embed_dim=64)
    features, tissue_ids, _ = make_dummy_data()
    
    out = film(features, tissue_ids)
    assert out.shape == features.shape, f"FiLM output shape mismatch: {out.shape} vs {features.shape}"
    
    # At init, gamma=1 and beta=0, so output should be approx equal to input
    assert torch.allclose(out, features, atol=1e-5), "FiLM at initialization should approximate identity"
    print("TissueFiLM test passed!")



def test_tissue_cond_regressor():
    """Test TissueCondRegressor module."""
    cond_reg = TissueCondRegressor(NUM_TISSUES, FEATURE_DIM, embed_dim=64)
    features = torch.randn(BATCH_SIZE, FEATURE_DIM)
    tissue_ids = torch.randint(0, NUM_TISSUES, (BATCH_SIZE,))
    
    pred = cond_reg(features, tissue_ids)
    assert pred.shape == (BATCH_SIZE,), f"CondRegressor output shape mismatch: {pred.shape}"
    print("TissueCondRegressor test passed!")


# ============================================================
# Full model tests for each conditioning mode
# ============================================================
def test_tissue_abmil_forward(cond_mode):
    """Test TissueABMIL forward pass for each conditioning mode."""
    kwargs = dict(
        num_tissues=NUM_TISSUES,
        feature_dim=FEATURE_DIM,
        head_dim=HEAD_DIM,
        n_heads=N_HEADS,
        gated=True,
        tissue_cond_mode=cond_mode,
    )
    if cond_mode == 'concat':
        kwargs['tissue_embed'] = True
        kwargs['tissue_embed_dim'] = 16
    
    model = TissueABMIL(**kwargs)
    features, tissue_ids, attn_mask = make_dummy_data()
    
    # Forward pass with tissue_id
    y_pred, head_attn = model(features, attn_mask=attn_mask, tissue_id=tissue_ids)
    assert y_pred.shape == (BATCH_SIZE,), f"[{cond_mode}] y_pred shape: {y_pred.shape}"
    print(f"[{cond_mode}] Forward pass OK: y_pred={y_pred.shape}, head_attn={head_attn.shape}")


def test_tissue_abmil_no_tissue_id(cond_mode):
    """Test that model works when tissue_id=None (graceful fallback)."""
    model = TissueABMIL(
        num_tissues=NUM_TISSUES,
        feature_dim=FEATURE_DIM,
        head_dim=HEAD_DIM,
        n_heads=N_HEADS,
        gated=True,
        tissue_cond_mode=cond_mode,
    )
    features, _, attn_mask = make_dummy_data()
    
    y_pred, head_attn = model(features, attn_mask=attn_mask, tissue_id=None)
    assert y_pred.shape == (BATCH_SIZE,), f"[{cond_mode}] y_pred shape without tissue: {y_pred.shape}"
    print(f"[{cond_mode}] No tissue_id OK: y_pred={y_pred.shape}")


def test_tissue_abmil_gradient_flow(cond_mode):
    """Test that gradients flow through tissue conditioning modules."""
    model = TissueABMIL(
        num_tissues=NUM_TISSUES,
        feature_dim=FEATURE_DIM,
        head_dim=HEAD_DIM,
        n_heads=N_HEADS,
        gated=True,
        tissue_cond_mode=cond_mode,
    )
    features, tissue_ids, attn_mask = make_dummy_data()
    
    y_pred, _ = model(features, attn_mask=attn_mask, tissue_id=tissue_ids)
    loss = y_pred.sum()
    loss.backward()
    
    # Check that tissue conditioning parameters have gradients
    tissue_params_with_grad = 0
    total_tissue_params = 0
    for name, param in model.named_parameters():
        if 'tissue' in name:
            total_tissue_params += 1
            if param.grad is not None:
                tissue_params_with_grad += 1
    
    assert tissue_params_with_grad == total_tissue_params, \
        f"[{cond_mode}] Not all tissue params have gradients: {tissue_params_with_grad}/{total_tissue_params}"
    print(f"[{cond_mode}] Gradient flow OK: {tissue_params_with_grad}/{total_tissue_params} tissue params have gradients")


def test_tissue_abmil_return_features():
    """Test return_features flag."""
    model = TissueABMIL(
        num_tissues=NUM_TISSUES,
        feature_dim=FEATURE_DIM,
        head_dim=HEAD_DIM,
        n_heads=N_HEADS,
        gated=True,
        tissue_cond_mode='film',
    )
    features, tissue_ids, attn_mask = make_dummy_data()
    
    y_pred, head_attn, M = model(features, attn_mask=attn_mask, tissue_id=tissue_ids, return_features=True)
    assert M.shape == (BATCH_SIZE, FEATURE_DIM), f"Feature shape mismatch: {M.shape}"
    print(f"return_features OK: M={M.shape}")


def test_legacy_tissue_embed_compat():
    """Test backward compatibility with tissue_embed=True (should map to concat mode)."""
    model = TissueABMIL(
        num_tissues=NUM_TISSUES,
        feature_dim=FEATURE_DIM,
        head_dim=HEAD_DIM,
        n_heads=N_HEADS,
        gated=True,
        tissue_embed=True,
        tissue_embed_dim=16,
    )
    assert model.tissue_cond_mode == 'concat', "tissue_embed=True should map to 'concat' mode"
    
    features, tissue_ids, attn_mask = make_dummy_data()
    y_pred, _ = model(features, attn_mask=attn_mask, tissue_id=tissue_ids)
    assert y_pred.shape == (BATCH_SIZE,)
    print("Legacy compatibility test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running ABMIL Tissue Conditioning Tests")
    print("=" * 60)
    
    test_tissue_film()
    test_tissue_cond_regressor()
    
    for mode in ['none', 'concat', 'film', 'cond_regressor']:
        test_tissue_abmil_forward(mode)
    
    for mode in ['none', 'film', 'cond_regressor']:
        test_tissue_abmil_no_tissue_id(mode)
    
    for mode in ['film', 'cond_regressor']:
        test_tissue_abmil_gradient_flow(mode)
    
    test_tissue_abmil_return_features()
    test_legacy_tissue_embed_compat()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
