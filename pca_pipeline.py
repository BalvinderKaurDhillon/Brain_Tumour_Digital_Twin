"""Reduce a radiomic feature table to its principal components (95% variance).

Usage:
    python pca_pipeline.py \
        --input data/RADIOMICS_SELECTED_CASES.csv \
        --pca-out data/RADIOMICS_PCA_DATA.csv \
        --features-out data/PCA_EXTRACTED_RADIOMIC_PHENOTYPES.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PCA on radiomic PRED_* features.")
    p.add_argument('--input', type=Path, required=True,
                   help='CSV containing PRED_* radiomic features and a Case column.')
    p.add_argument('--pca-out', type=Path, default=Path('RADIOMICS_PCA_DATA.csv'),
                   help='Where to write the PCA-reduced patient table.')
    p.add_argument('--features-out', type=Path,
                   default=Path('PCA_EXTRACTED_RADIOMIC_PHENOTYPES.csv'),
                   help='Where to write the per-component representative-feature mapping.')
    p.add_argument('--variance', type=float, default=0.95,
                   help='Target cumulative explained variance ratio.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    pred_cols = [col for col in df.columns if col.startswith('PRED_')]
    if not pred_cols:
        raise ValueError(f"No PRED_* columns found in {args.input}")

    X = df[pred_cols]
    X = X.dropna(axis=1, how='all')
    pred_cols = list(X.columns)
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=args.variance)
    X_pca = pca.fit_transform(X_scaled)

    pca_cols = [f'PC_{i + 1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    df_pca['Case'] = df['Case'].values

    args.pca_out.parent.mkdir(parents=True, exist_ok=True)
    df_pca.to_csv(args.pca_out, index=False)

    loadings = pca.components_
    important_features = []
    for i in range(len(pca_cols)):
        idx = int(np.abs(loadings[i]).argmax())
        important_features.append({
            'Principal_Component': pca_cols[i],
            'Representative_Feature': pred_cols[idx],
            'Explained_Variance_Ratio': float(pca.explained_variance_ratio_[i]),
        })

    df_features = pd.DataFrame(important_features)
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(args.features_out, index=False)

    print(f"Original feature count: {len(pred_cols)}")
    print(f"Components needed for {args.variance:.0%} variance: {len(pca_cols)}")
    print(df_features.head(10))


if __name__ == "__main__":
    main()
