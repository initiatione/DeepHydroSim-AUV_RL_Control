import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class HydroDataset(Dataset):
    """Minimal PyTorch dataset wrapper around pre-processed numpy arrays."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class HydroDataManager:
    """
    Loads the processed physics CSV, splits it, fits StandardScalers,
    saves scalers to disk, and returns DataLoader objects.

    The scaler is fit only on the training split to prevent data leakage.

    Two split modes are available:

    ``random``
        Per-sample shuffle-and-split (the original behaviour). Adjacent 200 Hz
        samples land on both sides of the split, so temporally correlated rows
        leak into the test set and the reported metrics are optimistic.

    ``block``
        Contiguous blocks of ``block_size`` timesteps are shuffled and whole
        blocks are assigned to train/val/test. This is the protocol used in the
        paper (blocks of at least 150 timesteps) and is the only mode in which
        the reported test metrics are meaningful.
    """

    INPUT_COLS = [
        'u(m/s)', 'v(m/s)', 'w(m/s)', 'p(rad/s)', 'q(rad/s)', 'r(rad/s)',
        'Accel_u(m/s2)', 'Accel_v(m/s2)', 'Accel_w(m/s2)',
        'Accel_p(m/s2)', 'Accel_q(m/s2)', 'Accel_r(m/s2)',
    ]
    TARGET_COLS = [
        'F_Fluid_u(N)', 'F_Fluid_v(N)', 'F_Fluid_w(N)',
        'F_Fluid_p(N)', 'F_Fluid_q(N)', 'F_Fluid_r(N)',
    ]

    def __init__(self, csv_path: str, save_dir: str = '../models/'):
        self.csv_path = csv_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _block_indices(n: int, block_size: int, val_split: float, test_split: float,
                       seed: int = 42):
        """Return (train_idx, val_idx, test_idx) using contiguous blocks."""
        n_blocks = n // block_size
        if n_blocks < 10:
            raise ValueError(
                f"Only {n_blocks} blocks of {block_size} samples fit in {n} rows; "
                "reduce block_size or collect more data."
            )
        order = np.random.default_rng(seed).permutation(n_blocks)
        n_test = max(1, int(round(n_blocks * test_split)))
        n_val = max(1, int(round(n_blocks * val_split)))
        test_blocks = order[:n_test]
        val_blocks = order[n_test:n_test + n_val]
        train_blocks = order[n_test + n_val:]

        def gather(blocks):
            return np.concatenate([
                np.arange(b * block_size, (b + 1) * block_size) for b in blocks
            ])

        return gather(train_blocks), gather(val_blocks), gather(test_blocks)

    def get_dataloaders(
        self,
        batch_size: int = 256,
        val_split: float = 0.1,
        test_split: float = 0.05,
        split_mode: str = "random",
        block_size: int = 200,
    ):
        """
        Returns (train_loader, val_loader, test_loader).

        Pipeline: load CSV -> drop NaNs -> split -> fit scalers on train
                  -> apply to val/test -> wrap in DataLoaders.
        """
        print(f"[DataManager] Loading {self.csv_path} ...")
        df = pd.read_csv(self.csv_path)

        if df.isnull().values.any():
            print("[DataManager] Warning: NaN values detected — dropping rows.")
            df = df.dropna()

        X_raw = df[self.INPUT_COLS].values.astype(np.float32)
        Y_raw = df[self.TARGET_COLS].values.astype(np.float32)
        print(f"[DataManager] Total samples: {len(X_raw)}")

        if split_mode == "block":
            tr, va, te = self._block_indices(
                len(X_raw), block_size, val_split, test_split
            )
            X_train, Y_train = X_raw[tr], Y_raw[tr]
            X_val, Y_val = X_raw[va], Y_raw[va]
            X_test, Y_test = X_raw[te], Y_raw[te]
            print(f"[DataManager] Split: block ({block_size} samples/block)")
        elif split_mode == "random":
            # Split: first carve out the held-out test set, then split remainder
            X_temp, X_test, Y_temp, Y_test = train_test_split(
                X_raw, Y_raw, test_size=test_split, random_state=42, shuffle=True
            )
            adjusted_val = val_split / (1.0 - test_split)
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_temp, Y_temp, test_size=adjusted_val, random_state=42, shuffle=True
            )
            print("[DataManager] Split: per-sample random (metrics are optimistic)")
        else:
            raise ValueError(f"Unknown split_mode '{split_mode}'.")
        print(f"[DataManager] Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

        # Fit scalers on training data only
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X_train_s = scaler_X.fit_transform(X_train)
        Y_train_s = scaler_Y.fit_transform(Y_train)
        X_val_s = scaler_X.transform(X_val)
        Y_val_s = scaler_Y.transform(Y_val)
        X_test_s = scaler_X.transform(X_test)
        Y_test_s = scaler_Y.transform(Y_test)

        joblib.dump(scaler_X, os.path.join(self.save_dir, 'scaler_X.pkl'))
        joblib.dump(scaler_Y, os.path.join(self.save_dir, 'scaler_Y.pkl'))
        print(f"[DataManager] Scalers saved to {self.save_dir}")

        train_loader = DataLoader(
            HydroDataset(X_train_s, Y_train_s),
            batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            HydroDataset(X_val_s, Y_val_s),
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        )
        test_loader = DataLoader(
            HydroDataset(X_test_s, Y_test_s),
            batch_size=batch_size, shuffle=False, num_workers=2,
        )
        return train_loader, val_loader, test_loader
