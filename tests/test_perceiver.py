import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.perceiver_mil import TissuePerceiverMIL


def test_perceiver_no_tissue():
    """Test TissuePerceiverMIL without tissue embedding"""
    print("=" * 60)
    print("Test 1: TissuePerceiverMIL (no tissue embed)")
    print("=" * 60)

    model = TissuePerceiverMIL(
        num_tissues=40,
        in_dim=1536,
        latent_seq=64,
        latent_dim=512,
        tissue_embed=False,
        perceiver_depth=4,
        transformer_depth=2,
        xattn_heads=8,
        mhsa_heads=8,
    )

    # Mock data: batch_size=2, num_patches=100, feature_dim=1536
    batch_size = 2
    num_patches = 100
    features = torch.randn(batch_size, num_patches, 1536)
    attn_mask = torch.ones(batch_size, num_patches, dtype=torch.bool)

    y_pred, head_attentions = model(features, attn_mask=attn_mask)
    print(f"y_pred shape: {y_pred.shape} (expected: ({batch_size},))")
    assert y_pred.shape == (batch_size,), f"Shape mismatch: {y_pred.shape}"
    assert head_attentions is None
    print("Test 1 PASSED!\n")


def test_perceiver_with_tissue():
    """Test TissuePerceiverMIL with tissue embedding"""
    print("=" * 60)
    print("Test 2: TissuePerceiverMIL (with tissue embed)")
    print("=" * 60)

    model = TissuePerceiverMIL(
        num_tissues=40,
        in_dim=1536,
        latent_seq=64,
        latent_dim=512,
        tissue_embed=True,
        perceiver_depth=4,
        transformer_depth=2,
        xattn_heads=8,
        mhsa_heads=8,
    )

    batch_size = 2
    num_patches = 100
    features = torch.randn(batch_size, num_patches, 1536)
    attn_mask = torch.ones(batch_size, num_patches, dtype=torch.bool)
    tissue_ids = torch.randint(0, 40, (batch_size,))

    y_pred, head_attentions = model(features, attn_mask=attn_mask, tissue_id=tissue_ids)
    print(f"y_pred shape: {y_pred.shape} (expected: ({batch_size},))")
    assert y_pred.shape == (batch_size,), f"Shape mismatch: {y_pred.shape}"
    assert head_attentions is None
    print("Test 2 PASSED!\n")


def test_perceiver_no_mask():
    """Test TissuePerceiverMIL without attention mask"""
    print("=" * 60)
    print("Test 3: TissuePerceiverMIL (no attn_mask)")
    print("=" * 60)

    model = TissuePerceiverMIL(
        num_tissues=40,
        in_dim=1536,
        latent_seq=32,
        latent_dim=256,
        tissue_embed=False,
        perceiver_depth=2,
        transformer_depth=1,
        xattn_heads=4,
        mhsa_heads=4,
    )

    batch_size = 1
    num_patches = 50
    features = torch.randn(batch_size, num_patches, 1536)

    y_pred, _ = model(features)
    print(f"y_pred shape: {y_pred.shape} (expected: ({batch_size},))")
    assert y_pred.shape == (batch_size,), f"Shape mismatch: {y_pred.shape}"
    print("Test 3 PASSED!\n")


def test_registry():
    """Test that perceiver is registered in TRAINERS"""
    print("=" * 60)
    print("Test 4: TRAINERS registry")
    print("=" * 60)
    from trainers import TRAINERS
    assert 'perceiver' in TRAINERS, "'perceiver' not found in TRAINERS!"
    print(f"Available trainers: {list(TRAINERS.keys())}")
    print("Test 4 PASSED!\n")


if __name__ == "__main__":
    test_perceiver_no_tissue()
    test_perceiver_with_tissue()
    test_perceiver_no_mask()
    test_registry()
    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
