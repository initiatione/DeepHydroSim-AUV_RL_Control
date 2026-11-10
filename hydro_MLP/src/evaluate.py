"""
Evaluate the trained DeepHydroMLP on the held-out test split.

Usage (from hydro_MLP/):
    python src/evaluate.py
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import HydroDataManager
from model import build_model
from paths import path as pkg_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get('MODEL_PATH', pkg_path('models/best_hydro_model.pth'))
SCALER_DIR = os.environ.get('SCALER_DIR', pkg_path('models'))
CSV_PATH = os.environ.get(
    'CSV_PATH', pkg_path('data/phy_processed/mission_log_processed_Physics.csv'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLOT_LEN = 500  # number of consecutive samples to visualise

with open(os.path.join(os.path.dirname(__file__), '..', 'config.yaml')) as _f:
    _CFG = yaml.safe_load(_f)
SPLIT_MODE = os.environ.get('SPLIT_MODE', _CFG['training'].get('split_mode', 'random'))
BLOCK_SIZE = int(os.environ.get('BLOCK_SIZE', _CFG['training'].get('block_size', 200)))
VAL_SPLIT = float(_CFG['training']['val_split'])
TEST_SPLIT = float(_CFG['training']['test_split'])

_meta_path = os.path.join(SCALER_DIR, 'model_meta.yaml')
MODEL_TYPE = 'baseline'
HIDDEN_DIMS = tuple(_CFG['model']['hidden_dims'])
if os.path.exists(_meta_path):
    with open(_meta_path) as _mf:
        _meta = yaml.safe_load(_mf)
    MODEL_TYPE = _meta.get('type', 'baseline')
    HIDDEN_DIMS = tuple(_meta.get('hidden_dims', HIDDEN_DIMS))
# ---------------------------------------------------------------------------


def evaluate():
    print(f"Evaluating on: {DEVICE}   model: {MODEL_TYPE}   split: {SPLIT_MODE}")

    model = build_model(MODEL_TYPE, hidden_dims=HIDDEN_DIMS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    data_manager = HydroDataManager(CSV_PATH, save_dir=SCALER_DIR)
    _, _, test_loader = data_manager.get_dataloaders(
        batch_size=1000,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        split_mode=SPLIT_MODE,
        block_size=BLOCK_SIZE,
    )

    # get_dataloaders() refits and rewrites the scalers for this CSV, so reload
    # them here to make sure the inverse transform matches the split used.
    scaler_X = joblib.load(os.path.join(SCALER_DIR, 'scaler_X.pkl'))
    scaler_Y = joblib.load(os.path.join(SCALER_DIR, 'scaler_Y.pkl'))

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(DEVICE))
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    pred_scaled = np.concatenate(all_preds, axis=0)
    true_scaled = np.concatenate(all_targets, axis=0)

    pred_real = scaler_Y.inverse_transform(pred_scaled)
    true_real = scaler_Y.inverse_transform(true_scaled)

    rmse_all = float(np.sqrt(np.mean((pred_real - true_real) ** 2)))
    mae_all = float(np.mean(np.abs(pred_real - true_real)))
    print(f"\nTest RMSE (pooled, physical scale): {rmse_all:.4f}")
    print(f"Test MAE  (pooled, physical scale): {mae_all:.4f}")

    labels = ['Fx (Surge)', 'Fy (Sway)', 'Fz (Heave)', 'Tx (Roll)', 'Ty (Pitch)', 'Tz (Yaw)']
    units  = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']
    print(f"\n{'DOF':<14} | {'R2':>8} | {'MAE':>12} | {'RMSE':>12} | {'Max Error':>12}")
    print('-' * 72)
    r2s = []
    for i, label in enumerate(labels):
        r2 = r2_score(true_real[:, i], pred_real[:, i])
        r2s.append(r2)
        mae = mean_absolute_error(true_real[:, i], pred_real[:, i])
        rmse = float(np.sqrt(mean_squared_error(true_real[:, i], pred_real[:, i])))
        max_err = float(np.max(np.abs(true_real[:, i] - pred_real[:, i])))
        print(f"{label:<14} | {r2:>8.4f} | {mae:>8.4f} {units[i]} | "
              f"{rmse:>8.4f} {units[i]} | {max_err:>8.4f} {units[i]}")
    print(f"\nMean R2 over 6 axes: {np.mean(r2s):.4f}")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(true_real[:PLOT_LEN, i], 'k-', label='Ground Truth', linewidth=1.5, alpha=0.7)
        ax.plot(pred_real[:PLOT_LEN, i], 'r--', label='Prediction', linewidth=1.5)
        ax.set_title(labels[i])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    os.makedirs(pkg_path('logs'), exist_ok=True)
    save_path = pkg_path('logs/evaluation_result.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


if __name__ == '__main__':
    evaluate()
