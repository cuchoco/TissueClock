import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from dataset.data import AGE_MEAN, AGE_STD


def train(cfg):
    """Function to train a Pan-tissue model using Hydra config

    Args:
        cfg (DictConfig): Hydra configuration object containing data_dir, fold, output_dir, and model_params.
    """
    
    data_dir = Path(cfg.data_dir)
    fold = cfg.fold
    output_dir = cfg.output_dir
    use_tissue_label = cfg.get('use_tissue_label', False)
    
    model_params = cfg.get('model_params', {})
    alphas = model_params.get('alphas', np.logspace(-1, 7, 20))

    print(f"\n{'='*60}")
    print(f"Starting Pan-Tissue model training (Fold {fold})")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    npz_files = list(data_dir.glob('age_predict_*.npz'))

    if not npz_files:
        print(f"Error: No .npz files found in the specified directory ({data_dir}).")
        return

    all_X, all_y, all_folds, all_tissues = [], [], [], []

    for npz_path in npz_files:
        tissue_name = npz_path.stem.replace('age_predict_', '')
        data = np.load(npz_path)

        X, y, folds = data['X'], data['y'], data['folds']

        all_X.append(X)
        all_y.append(y)
        all_folds.append(folds)
        all_tissues.extend([tissue_name] * len(y))

    X_stacked = np.vstack(all_X)
    y_stacked = np.concatenate(all_y)
    folds_stacked = np.concatenate(all_folds)
    tissues_array = np.array(all_tissues).reshape(-1, 1)

    print(f"Total number of samples: {len(y_stacked)}")
    print(f"Image feature dimensions: {X_stacked.shape[1]}")

    train_mask = (folds_stacked != fold)
    val_mask = (folds_stacked == fold)

    X_train_img, y_train_norm = X_stacked[train_mask], y_stacked[train_mask]
    X_val_img, y_val_norm = X_stacked[val_mask], y_stacked[val_mask]

    tissues_train = tissues_array[train_mask]
    tissues_val = tissues_array[val_mask]

    folds_train = folds_stacked[train_mask]

    if use_tissue_label:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        tissue_train_encoded = encoder.fit_transform(tissues_train)
        tissue_val_encoded = encoder.transform(tissues_val)

        print(f"Tissue category (One-Hot) dimensions: {tissue_train_encoded.shape[1]}")

        X_train = np.hstack((X_train_img, tissue_train_encoded))
        X_val = np.hstack((X_val_img, tissue_val_encoded))
        print(f"Final combined feature dimensions: {X_train.shape[1]}")

        encoder_save_path = os.path.join(output_dir, f'tissue_encoder_fold{fold}.pkl')
        joblib.dump(encoder, encoder_save_path)
    else:
        # 라벨을 사용하지 않을 경우 이미지 피처만 사용
        X_train = X_train_img
        X_val = X_val_img
        print("Skipping tissue label One-Hot Encoding.")
        print(f"Final feature dimensions: {X_train.shape[1]}")

    n_splits = len(np.unique(folds_train))
    gkf = GroupKFold(n_splits=n_splits)
    cv_splits = list(gkf.split(X_train, y_train_norm, groups=folds_train))

    model = RidgeCV(alphas=alphas, cv=cv_splits)
    model.fit(X_train, y_train_norm)

    y_pred_norm = model.predict(X_val)

    y_val_real = (y_val_norm * AGE_STD) + AGE_MEAN
    y_pred_real = (y_pred_norm * AGE_STD) + AGE_MEAN

    mae = mean_absolute_error(y_val_real, y_pred_real)
    r2 = r2_score(y_val_real, y_pred_real)

    print(f"Selected optimal alpha: {model.alpha_:.2e}")
    print(f"Overall MAE: {mae:.4f} years")
    print(f"Overall R2 Score: {r2:.4f}")

    model_save_path = os.path.join(output_dir, f'pan_tissue_model_fold{fold}.pkl')
    joblib.dump(model, model_save_path)

    # Visualization
    results_df = pd.DataFrame({
        'True_Age': y_val_real,
        'Predicted_Age': y_pred_real,
        'Age_Gap': y_pred_real - y_val_real,
        'Tissue': tissues_val.flatten()
    })

    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    scatter = ax1.scatter(results_df['True_Age'], results_df['Predicted_Age'],
                          c=results_df['Age_Gap'], cmap='coolwarm',
                          alpha=0.4, edgecolors='none', s=15)

    x_min, x_max = 20, 70
    y_min = int(results_df['Predicted_Age'].min() - 5)
    y_max = int(results_df['Predicted_Age'].max() + 5)

    ax1.plot([x_min, x_max], [y_min, y_max], color='gray', linestyle='--', lw=1.5, alpha=0.6)

    m, b = np.polyfit(results_df['True_Age'], results_df['Predicted_Age'], 1)
    reg_x = np.array([x_min, x_max])
    reg_y = m * reg_x + b
    ax1.plot(reg_x, reg_y, color='red', linestyle='-', lw=2.5, label=f'Regression Line ($y = {m:.2f}x + {b:.2f}$)')

    ax1.set_title(f'Tissue Chronological vs Predicted Age (Fold {fold})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Chronological Age (Years)', fontsize=12)
    ax1.set_ylabel('Predicted Age (Years)', fontsize=12)
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.legend()

    cbar = fig.colorbar(scatter, ax=ax1)
    cbar.set_label('Age Gap (Years)', rotation=270, labelpad=15)

    plt.tight_layout()
    plot_save_path = os.path.join(output_dir, f'pan_tissue_fold{fold}.png')
    plt.savefig(plot_save_path, dpi=300)
    plt.close(fig)

    if use_tissue_label:
        print(f"Saved successfully: Model ({os.path.basename(model_save_path)}),\
                Encoder ({os.path.basename(encoder_save_path)})")
    else:
        print(f"Saved successfully: Model ({os.path.basename(model_save_path)})")

    return {'mae': mae, 'r2': r2}