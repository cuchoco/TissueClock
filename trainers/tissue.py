import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from dataset.data import AGE_MEAN, AGE_STD


def train(cfg):
    """Function to train a model for a single tissue using Hydra config
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing data_dir, fold, output_dir, and model_params.
    """
    
    data_dir = Path(cfg.data_dir)
    fold = cfg.fold
    output_dir = cfg.output_dir
    model_params = cfg.get('model_params', {})

    npz_files = list(data_dir.glob('age_predict_*.npz'))

    if not npz_files:
        print(f"No .npz files found in the specified path ({data_dir}).")
        return

    print(f"Starting training and visualization for {len(npz_files)} tissue datasets...")

    results = []
    for npz_path in npz_files:
        res = _train_one(str(npz_path), fold, output_dir, **model_params)
        if res:
            results.append(res)

    # Output and save summary
    if results:
        _print_summary(results, fold, output_dir)
    else:
        print("No trainable tissues found, skipping summary file creation.")

    return results


def _train_one(npz_path, fold, output_dir, **kwargs):
    filename = os.path.basename(npz_path)
    tissue_name = filename.replace('age_predict_', '').replace('.npz', '')

    print(f"\n{'='*50}")
    print(f"Starting training for [{tissue_name}] tissue (Fold {fold})")
    print(f"{'='*50}")

    data = np.load(npz_path)
    X_all, y_all, folds_all = data['X'], data['y'], data['folds']

    train_mask = (folds_all != fold)
    val_mask = (folds_all == fold)

    X_train, y_train_norm = X_all[train_mask], y_all[train_mask]
    X_val, y_val_norm = X_all[val_mask], y_all[val_mask]

    folds_train = folds_all[train_mask]

    if len(X_train) == 0 or len(X_val) == 0:
        print(f"Warning: Insufficient data for [{tissue_name}], skipping training.")
        return None

    unique_folds = np.unique(folds_train)
    n_splits = len(unique_folds)

    if n_splits < 2:
        print(f"Warning: Train set for [{tissue_name}] has less than 2 fold types ({n_splits}), \
              cannot perform cross-validation. Skipping training.")
        return None

    gkf = GroupKFold(n_splits=n_splits)
    cv_splits = list(gkf.split(X_train, y_train_norm, groups=folds_train))

    alphas = kwargs.get('alphas', np.logspace(-1, 7, 20))
    model = RidgeCV(alphas=alphas, cv=cv_splits)
    model.fit(X_train, y_train_norm)

    y_pred_norm = model.predict(X_val)

    y_val_real = (y_val_norm * AGE_STD) + AGE_MEAN
    y_pred_real = (y_pred_norm * AGE_STD) + AGE_MEAN

    mae = mean_absolute_error(y_val_real, y_pred_real)
    r2 = r2_score(y_val_real, y_pred_real)

    print(f"Selected optimal alpha: {model.alpha_:.2e}")
    print(f"MAE: {mae:.4f} years")
    print(f"R2 Score: {r2:.4f}")

    model_save_path = os.path.join(output_dir, f'{tissue_name}/fold{fold}.pkl')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)

    # Visualization
    results_df = pd.DataFrame({
        'True_Age': y_val_real,
        'Predicted_Age': y_pred_real,
        'Age_Gap': y_pred_real - y_val_real
    })

    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    scatter = ax1.scatter(results_df['True_Age'], results_df['Predicted_Age'],
                          c=results_df['Age_Gap'], cmap='coolwarm',
                          alpha=0.7, edgecolors='w', linewidth=0.5)

    x_min, x_max = 20, 70
    y_min = int(results_df['Predicted_Age'].min() - 5)
    y_max = int(results_df['Predicted_Age'].max() + 5)

    ax1.plot([x_min, x_max], [y_min, y_max], color='gray', linestyle='--', lw=1.5, alpha=0.6)

    m, b = np.polyfit(results_df['True_Age'], results_df['Predicted_Age'], 1)
    reg_x = np.array([results_df['True_Age'].min(), results_df['True_Age'].max()])
    reg_y = m * reg_x + b

    ax1.plot(reg_x, reg_y, color='red', linestyle='-', lw=2.5, label=f'Regression Line ($y = {m:.2f}x + {b:.2f}$)')

    ax1.set_title(f'[{tissue_name}] Chronological vs Predicted Age (Fold {fold})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Chronological Age (Years)', fontsize=12)
    ax1.set_ylabel('Predicted Age (Years)', fontsize=12)
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.legend()

    cbar = fig.colorbar(scatter, ax=ax1)
    cbar.set_label('Age Gap (Years)', rotation=270, labelpad=15)

    plt.tight_layout()
    plot_save_path = os.path.join(output_dir, f'{tissue_name}/fold{fold}.png')
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path, dpi=300)
    plt.close(fig)

    print(f"Saved: Model ({os.path.basename(model_save_path)}), Plot ({os.path.basename(plot_save_path)})")

    return {'tissue': tissue_name, 'mae': mae, 'r2': r2, 'n_val': len(X_val)}


def _print_summary(results, fold, output_dir):
    print("\n" + "="*50)
    print(f"Summary of training and evaluation for all tissues (Fold {fold})")
    print("="*50)
    for r in results:
        print(f"- {r['tissue']:<20}: MAE = {r['mae']:.4f}, R2 = {r['r2']:.4f}, N = {r['n_val']}")

    summary_df = pd.DataFrame(results)
    total_samples = summary_df['n_val'].sum()
    weighted_mae = (summary_df['mae'] * summary_df['n_val']).sum() / total_samples
    weighted_r2 = (summary_df['r2'] * summary_df['n_val']).sum() / total_samples

    print("\n" + "="*50)
    print(f"Weighted average performance across all tissues (Fold {fold})")
    print("-"*50)
    print(f"Total Samples    : {total_samples}")
    print(f"Weighted Avg MAE : {weighted_mae:.4f}")
    print(f"Weighted Avg R2  : {weighted_r2:.4f}")
    print("="*50)

    new_row = pd.DataFrame([{  
        'tissue': 'Weighted Average',
        'mae': weighted_mae,
        'r2': weighted_r2,
        'n_val': total_samples
    }])
    summary_df = pd.concat([summary_df, new_row], ignore_index=True)
    summary_save_path = os.path.join(output_dir, f'summary_fold{fold}.csv')
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(summary_save_path, index=False)

    print(f"\nOverall summary saved to: {summary_save_path}")