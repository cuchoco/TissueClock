import os
import re
import h5py
from pathlib import Path
import pandas as pd 
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from dataset.config import *

class AgePredictDataset(Dataset):
    """
    Dataset for Age Prediction from Whole Slide Images.
    Loads h5 features and normalizes age.
    """
    def __init__(self, fold: int = 0, train: bool = True, feature = "uni") -> None:
        self.df = pd.read_csv(CSV_PATH)
        if train:
            self.df = self.df[self.df['fold'] != fold]
        else:
            self.df = self.df[self.df['fold'] == fold]

        if feature == "uni":
            self.feature_root = Path(FEATURE_ROOT)
        elif feature == "conch":
            self.feature_root = Path(FEATURE_ROOT_CONCH)
            
        self.tissue_ids = self._map_tissues()

    def _map_tissues(self) -> List[int]:
        tissue_ids = []
        for idx in range(len(self.df)):
            tissue_name = self.df.iloc[idx]['Tissue']
            tissue_name = re.sub(r'[^a-zA-Z0-9]+', '_', str(tissue_name)).strip('_')
            organ_name = SUBTYPE_TO_ORGAN[tissue_name]
            tissue_ids.append(ORGAN_TO_ID[organ_name])
        return tissue_ids

    def get_sample_weights(self) -> torch.Tensor:
        class_counts = pd.Series(self.tissue_ids).value_counts().to_dict()
        sample_weights = [1.0 / (class_counts[t_id] ** 0.5) for t_id in self.tissue_ids]
        return torch.DoubleTensor(sample_weights)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.df.iloc[idx]
        sample_id = data['Tissue Sample ID']
        age = data['AGE']
        age = (age - AGE_MEAN) / AGE_STD  # Normalize age
        sex = data['SEX'] - 1 # 0: male, 1: female
        file = self.feature_root / f'{sample_id}.h5'
        tissue_id = self.tissue_ids[idx]

        with h5py.File(file, 'r') as f:
            features = f['features'][:]
        
        return torch.tensor(features, dtype=torch.float32), \
               torch.tensor(age, dtype=torch.float32), \
               torch.tensor(tissue_id, dtype=torch.long), \
               torch.tensor(sex, dtype=torch.long)
    
def mil_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to pad variable-length features.
    
    batch: [(features_1, tissue_id_1, label_1), (features_2, tissue_id_2, label_2), ...]
           features shape (num_images, feature_dim)
    """
    features_list = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    tissue_ids = [item[2] for item in batch]
    sexs = [item[3] for item in batch]
    
    # Features Padding
    padded_features = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    
    # Attention Mask 
    lengths = torch.tensor([len(f) for f in features_list])
    batch_size = len(features_list)
    max_len = padded_features.shape[1]
    attn_mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    
    labels = torch.tensor(labels, dtype=torch.float32)
    tissue_ids = torch.tensor(tissue_ids, dtype=torch.long)
    sexs = torch.tensor(sexs, dtype=torch.long)
    
    return padded_features, labels, tissue_ids, attn_mask, sexs

def get_abmil_dataloader(fold: int = 0, batch_size: int = 1, feature="uni", use_sampler: bool = False) -> Tuple[DataLoader, DataLoader]:
    
    tr_dataset = AgePredictDataset(fold=fold, train=True, feature=feature)
    val_dataset = AgePredictDataset(fold=fold, train=False, feature=feature)

    # 샘플 가중치 획득 및 Sampler 생성
    sample_weights = tr_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True 
    )

    tr_dataloader = DataLoader(tr_dataset, 
                               batch_size=batch_size, 
                               sampler=sampler if use_sampler else None,
                               shuffle=None if use_sampler else True,
                               pin_memory=True,
                               collate_fn=mil_collate_fn)
    
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False,
                                pin_memory=True,
                                collate_fn=mil_collate_fn)
    
    return tr_dataloader, val_dataloader