import torch
from omegaconf import OmegaConf
from model.transmil import TissueTransMIL
from trainers.transmil import TransMILTrainer

# Create a mock config
cfg_dict = {
    'fold': 0,
    'output_dir': './experiments/test',
    'batch_size': 1,
    'model_params': {
        'in_dim': 1536,
        'dim': 512,
        'tissue_embed': True,
        'tissue_embed_dim': 16
    },
    'num_tissues': 40,
    'use_wandb': False,
    'model': 'transmil'
}
cfg = OmegaConf.create(cfg_dict)

try:
    print("Building model...")
    model = TissueTransMIL(num_tissues=40, in_dim=1536, dim=512, tissue_embed=True)
    
    print("Testing forward pass...")
    # Mock data: batch_size=1, num_images=100, feature_dim=1536
    features = torch.randn(1, 100, 1536)
    attn_mask = torch.ones(1, 100, dtype=torch.bool)
    tissue_id = torch.tensor([5])
    
    y_pred, head_attentions = model(features, attn_mask, tissue_id=tissue_id)
    print(f"y_pred shape: {y_pred.shape}")
    print("Forward pass successful!")
    
except ImportError as e:
    print(f"Skipping structural test due to missing dependencies: {e}")
except Exception as e:
    print(f"Error during verification: {e}")
    raise e
