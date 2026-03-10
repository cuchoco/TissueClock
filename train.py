import argparse
import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from trainers import TRAINERS

@hydra.main(config_path='config', config_name='config_abmil')
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))
    if 'model' not in cfg:
        sys.exit('Error: Model name not specified in config.')

    model_name = cfg.model
    if model_name not in TRAINERS:
        sys.exit(f"Unknown model '{model_name}', available: {list(TRAINERS)}")

    trainer = TRAINERS[model_name]

    cfg.data_dir = to_absolute_path(cfg.get('data_dir', './data'))
    cfg.output_dir = to_absolute_path(cfg.get('output_dir', './experiments'))
    model_params = OmegaConf.to_container(cfg.get('model_params', {}), resolve=True) 

    print(f"Running model: {model_name}")
    print(f"Data directory: {cfg.data_dir}")
    print(f"Fold: {cfg.fold}, Output directory: {cfg.output_dir}")
    if model_params:
        print(f"Additional model parameters: {model_params}")

    result = trainer(cfg)

    # Result saving/summary logic (general processing)
    if isinstance(result, dict):
        # single-summary case
        out_file = os.path.join(cfg.output_dir, f'summary_fold{cfg.fold}.csv')
        os.makedirs(cfg.output_dir, exist_ok=True)
        import pandas as pd
        pd.DataFrame([result]).to_csv(out_file, index=False)
        print(f"Summary results saved to: {out_file}")
    else:
        # The trainer might have printed/saved its own summary
        pass


if __name__ == '__main__':
    main()