"""Build a per-patient digital state from the raw (un-PCA'd) radiomics table.

Usage:
    python standardize_raw.py --input data/RADIOMICS_SELECTED_CASES.csv

Standardizing the raw 87-feature vector is critical: without it the agent
ignores small-scale features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standardize raw PRED_* features and emit a per-patient digital state."
    )
    p.add_argument('--input', type=Path, required=True,
                   help='CSV containing PRED_* radiomic features and a Case column.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df_raw = pd.read_csv(args.input)
    raw_feature_cols = [col for col in df_raw.columns if col.startswith('PRED_')]
    if not raw_feature_cols:
        raise ValueError(f"No PRED_* columns found in {args.input}")

    scaler_raw = StandardScaler()
    df_raw_scaled = df_raw.copy()
    df_raw_scaled[raw_feature_cols] = scaler_raw.fit_transform(df_raw[raw_feature_cols])

    raw_patient_db: dict = {}
    for _, row in df_raw_scaled.iterrows():
        p_id = row['Case']
        raw_patient_db[p_id] = {
            "initial_state": row[raw_feature_cols].values.astype(np.float32),
            "feature_names": raw_feature_cols,
        }

    print(
        f"Initialized raw digital states with {len(raw_feature_cols)} features "
        f"for {len(raw_patient_db)} patients."
    )


if __name__ == "__main__":
    main()
