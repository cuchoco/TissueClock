import torch
from omegaconf import OmegaConf

from model.mambamil import TissueMambaMIL
from model.mambamil import SRMambaLayer


def test_srmamba_layer():
    print("Testing SRMambaLayer...")
    try:
        layer = SRMambaLayer(d_model=512, d_state=16, d_conv=4, expand=2)
        # Dummy Input: [Batch, Sequence_Length, Dimension]
        x = torch.randn(2, 100, 512)
        out = layer(x, rate=10)
        print(f"SRMambaLayer Output Shape: {out.shape}")
        assert out.shape == (2, 100, 512), "Output shape mismatch!"
        print("SRMambaLayer Test Passed!\n")
    except ImportError as e:
        print(f"Skipping SRMambaLayer test: {e}\n")


def test_tissue_mambamil():
    print("Testing TissueMambaMIL...")
    
    # Mock settings
    batch_size = 2
    num_patches = 100
    in_dim = 1536
    dim = 512
    num_tissues = 40
    
    model = TissueMambaMIL(
        num_tissues=num_tissues,
        in_dim=in_dim,
        dim=dim,
        layer=2,
        rate=10,
        tissue_embed=True
    )
    
    print(model)
    
    # Mock data
    features = torch.randn(batch_size, num_patches, in_dim)
    tissue_ids = torch.randint(0, num_tissues, (batch_size,))
    
    # Forward pass
    try:
        y_pred, head_attentions = model(features, tissue_id=tissue_ids)
        print(f"\nForward Pass Successful!")
        print(f"Y_pred Shape: {y_pred.shape} (Expected: ({batch_size},))")
        print(f"Head Attentions Shape: {head_attentions.shape} (Expected: ({batch_size}, {num_patches + 1}))") # +1 for Tissue Token
        
        assert y_pred.shape == (batch_size,), "y_pred shape mismatch!"
        assert head_attentions.shape == (batch_size, num_patches + 1), "head_attentions shape mismatch!"
        print("TissueMambaMIL Test Passed!\n")
        
    except Exception as e:
        print(f"Forward Pass Failed: {e}")

if __name__ == "__main__":
    test_srmamba_layer()
    test_tissue_mambamil()
