"""
Train DeepHydroMLP on the processed physics CSV.

Usage (from hydro_MLP/):
    python src/train.py

Override the CSV path via the CSV_PATH constant below or pass it as an
environment variable: CSV_PATH=/path/to/data.csv python src/train.py
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from dataset import HydroDataManager
from model import build_model
from paths import path as pkg_path

# ---------------------------------------------------------------------------
# Configuration (config.yaml next to this package; env vars take precedence)
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(_CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

CSV_PATH = os.environ.get('CSV_PATH', pkg_path(_CFG['data']['processed_csv']))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', _CFG['training']['batch_size']))
LR = float(os.environ.get('LR', _CFG['training']['learning_rate']))
EPOCHS = int(os.environ.get('EPOCHS', _CFG['training']['epochs']))
PATIENCE = int(_CFG['training'].get('early_stop_patience', 0))
GRAD_CLIP = float(_CFG['training'].get('grad_clip', 5.0))
MODEL_TYPE = os.environ.get('MODEL_TYPE', _CFG['model'].get('type', 'baseline'))
SPLIT_MODE = os.environ.get('SPLIT_MODE', _CFG['training'].get('split_mode', 'random'))
BLOCK_SIZE = int(os.environ.get('BLOCK_SIZE', _CFG['training'].get('block_size', 200)))
HIDDEN_DIMS = tuple(int(v) for v in os.environ.get(
    'HIDDEN_DIMS', ','.join(str(h) for h in _CFG['model']['hidden_dims'])).split(','))
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', _CFG['training'].get('weight_decay', 1e-5)))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_SAVE_ENV = os.environ.get('SAVE_DIR')
SAVE_DIR = os.path.abspath(_SAVE_ENV) if _SAVE_ENV else pkg_path(_CFG['paths']['save_dir'])
LOG_DIR = pkg_path(_CFG['paths']['log_dir'])
# ---------------------------------------------------------------------------


def main():
    print(f"Training on: {DEVICE}")
    print(f"Model type: {MODEL_TYPE}   split: {SPLIT_MODE}")

    data_manager = HydroDataManager(CSV_PATH, save_dir=SAVE_DIR)
    train_loader, val_loader, _ = data_manager.get_dataloaders(
        batch_size=BATCH_SIZE,
        val_split=float(_CFG['training']['val_split']),
        test_split=float(_CFG['training']['test_split']),
        split_mode=SPLIT_MODE,
        block_size=BLOCK_SIZE,
    )

    model = build_model(MODEL_TYPE, hidden_dims=HIDDEN_DIMS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}   hidden: {HIDDEN_DIMS}   wd: {WEIGHT_DECAY}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Sidecar so evaluate.py and the MuJoCo plugin rebuild the same architecture.
    with open(os.path.join(SAVE_DIR, 'model_meta.yaml'), 'w') as _mf:
        yaml.safe_dump({'type': MODEL_TYPE, 'input_dim': 12, 'output_dim': 6,
                        'hidden_dims': list(HIDDEN_DIMS), 'n_params': n_params}, _mf)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, f'run_{int(time.time())}'))

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                val_loss += criterion(model(inputs), targets).item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]  "
            f"Train: {avg_train_loss:.6f}  Val: {avg_val_loss:.6f}  LR: {current_lr:.6f}"
        )
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_hydro_model.pth'))
            print("  -> Best model saved.")
        else:
            epochs_without_improvement += 1
            if PATIENCE and epochs_without_improvement >= PATIENCE:
                print(f"Early stopping after {PATIENCE} epochs without improvement.")
                break

    print("Training complete.")
    writer.close()


if __name__ == '__main__':
    main()
