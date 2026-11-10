"""
Downsample the processed physics CSV to the paper's 50 Hz sampling rate.

The Stonefish collector logs every physics step (200 Hz).  The paper reports
857,542 samples at 50 Hz, so the 200 Hz log is decimated by 4 *after* the
Savitzky-Golay differentiation, i.e. the acceleration labels keep the
0.255 s smoothing window used by ``data_process.py``.

Usage (from hydro_MLP/):
    python src/downsample.py --factor 4
    python src/downsample.py --input a.csv --output b.csv --factor 4
"""

import argparse
import os

import pandas as pd

from paths import path as pkg_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=pkg_path(
        'data/phy_processed/mission_log_processed_Physics.csv'))
    parser.add_argument('--output', default=pkg_path(
        'data/phy_processed/mission_log_processed_Physics_50hz.csv'))
    parser.add_argument('--factor', type=int, default=4,
                        help='Decimation factor (200 Hz / 4 = 50 Hz).')
    args = parser.parse_args()

    if args.factor < 1:
        raise SystemExit('[Downsample] --factor must be >= 1.')

    df = pd.read_csv(args.input)
    out = df.iloc[::args.factor].reset_index(drop=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False, float_format='%.6f')

    dt = out['Time(s)'].diff().dropna()
    print(f'[Downsample] {len(df)} -> {len(out)} rows '
          f'(factor {args.factor}, median dt {dt.median():.4f} s)')
    print(f'[Downsample] Wrote {args.output}')


if __name__ == '__main__':
    main()
