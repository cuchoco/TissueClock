import os
import re
import h5py
from pathlib import Path
import pandas as pd 

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataset.config import *

class AgePredictDataset(Dataset):
    def __init__(self, fold=0, train=True):
        self.df = pd.read_csv(CSV_PATH)
        if train:
            self.df = self.df[self.df['fold'] != fold]
        else:
            self.df = self.df[self.df['fold'] == fold]
        self.feature_root = Path(FEATURE_ROOT)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        sample_id = data['Tissue Sample ID']
        age = data['AGE']
        age = (age - AGE_MEAN) / AGE_STD  # Normalize age
        file = self.feature_root / f'{sample_id}.h5'

        tissue_name = data['Tissue']
        tissue_name = re.sub(r'[^a-zA-Z0-9]+', '_', str(tissue_name)).strip('_')
        tissue_id = TISSUE_TO_ID[tissue_name]

        with h5py.File(file, 'r') as f:
            features = f['features'][:]
        
        return torch.tensor(features, dtype=torch.float32), \
               torch.tensor(age, dtype=torch.float32), \
               torch.tensor(tissue_id, dtype=torch.long)
    
def mil_collate_fn(batch):
    """
    batch: Dataset에서 반환된 아이템들의 리스트. 
           예: [(features_1, tissue_id_1, label_1), (features_2, tissue_id_2, label_2), ...]
           features의 shape은 (num_images, feature_dim) 이라고 가정합니다.
    """
    features_list = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    tissue_ids = [item[2] for item in batch]
    
    # Features Padding
    padded_features = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    
    # Attention Mask 
    lengths = torch.tensor([len(f) for f in features_list])
    batch_size = len(features_list)
    max_len = padded_features.shape[1]
    attn_mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    
    labels = torch.tensor(labels, dtype=torch.float32)
    tissue_ids = torch.tensor(tissue_ids, dtype=torch.long)
    
    return padded_features, labels, tissue_ids, attn_mask

def get_abmil_dataloader(fold=0, batch_size=1):
    
    tr_dataset = AgePredictDataset(fold=fold, train=True)
    val_dataset = AgePredictDataset(fold=fold, train=False)

    tr_dataloader = DataLoader(tr_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=mil_collate_fn)
    
    val_dataloader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=mil_collate_fn)
    
    return tr_dataloader, val_dataloader