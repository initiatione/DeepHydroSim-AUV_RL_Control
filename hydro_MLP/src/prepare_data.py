"""
Merge raw Stonefish mission logs into a single processed mission log.

The teacher simulator writes one CSV per run
(``cpp_sim/logs/GeneralMission_<timestamp>.csv``).  This script concatenates
any number of those files, re-bases the time column so the result is
monotonic, drops rows truncated by an interrupted run, and writes
``data/phy_processed/mission_log_processed.csv`` — the input expected by
``data_process.py``.

Usage (from hydro_MLP/):
    python src/prepare_data.py
    python src/prepare_data.py --inputs path/a.csv path/b.csv
"""

import argparse
import glob
import os

import pandas as pd

from paths import path as pkg_path

RAW_DIR = pkg_path('data/raw')
FALLBACK_DIR = pkg_path('..', 'cpp_sim', 'logs')
OUTPUT_CSV = pkg_path('data/phy_processed/mission_log_processed.csv')

KEY_COLS = ['Time(s)', 'u(m/s)', 'v(m/s)', 'w(m/s)', 'p(rad/s)', 'q(rad/s)', 'r(rad/s)']


def _discover(raw_dir: str):
    files = sorted(glob.glob(os.path.join(raw_dir, 'GeneralMission_*.csv')))
    if not files:
        files = sorted(glob.glob(os.path.join(FALLBACK_DIR, 'GeneralMission_*.csv')))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='*', default=None,
                        help='Explicit list of raw CSVs (default: auto-discover).')
    parser.add_argument('--raw-dir', default=RAW_DIR,
                        help='Directory to scan for GeneralMission_*.csv.')
    parser.add_argument('--output', default=OUTPUT_CSV)
    args = parser.parse_args()

    files = args.inputs or _discover(args.raw_dir)
    if not files:
        raise SystemExit(
            f'[PrepareData] No raw logs found in {args.raw_dir} '
            f'or {FALLBACK_DIR}. Run the Stonefish collector first.'
        )

    frames = []
    time_offset = 0.0
    for path in files:
        df = pd.read_csv(path)
        df = df.dropna(subset=KEY_COLS)
        if df.empty:
            print(f'[PrepareData] Skipping empty log: {path}')
            continue
        # Re-base time so merged episodes form one monotonic timeline.
        t = df['Time(s)'].to_numpy()
        if len(t) > 1:
            dt = float(pd.Series(t).diff().median())
        else:
            dt = 0.005
        df['Time(s)'] = t - t[0] + time_offset
        time_offset = float(df['Time(s)'].iloc[-1]) + dt
        frames.append(df)
        print(f'[PrepareData] {os.path.basename(path)}: {len(df)} rows')

    if not frames:
        raise SystemExit('[PrepareData] All input logs were empty.')

    merged = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.to_csv(args.output, index=False, float_format='%.6f')
    print(f'[PrepareData] Wrote {len(merged)} rows -> {args.output}')


if __name__ == '__main__':
    main()
