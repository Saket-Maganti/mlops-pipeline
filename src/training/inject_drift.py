"""
Drift Injection Script
Injects synthetic data drift into the wine dataset to simulate production distribution shift.
Supports covariate shift, label shift, and concept drift modes.
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_reference_data():
    """Load base wine dataset as reference."""
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    return df


def inject_covariate_shift(df: pd.DataFrame, drift_magnitude: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """
    Inject covariate shift: feature distributions shift while P(Y|X) stays same.
    Simulates sensor calibration drift or population shift.
    """
    np.random.seed(seed)
    df_drifted = df.copy()
    feature_cols = [c for c in df.columns if c != "target"]
    n_features_to_drift = max(1, int(len(feature_cols) * 0.6))
    drift_features = np.random.choice(feature_cols, n_features_to_drift, replace=False)

    for col in drift_features:
        std = df[col].std()
        mean_shift = drift_magnitude * std * np.random.choice([-1, 1])
        scale_factor = 1 + drift_magnitude * np.random.uniform(0.1, 0.5)
        noise = np.random.normal(mean_shift, std * drift_magnitude * 0.1, len(df))
        df_drifted[col] = df_drifted[col] * scale_factor + noise

    logger.info(f"Covariate shift injected on {n_features_to_drift} features (magnitude={drift_magnitude})")
    return df_drifted


def inject_label_shift(df: pd.DataFrame, target_class: int = 0, multiplier: float = 3.0, seed: int = 42) -> pd.DataFrame:
    """
    Inject label shift: class proportions change.
    Simulates seasonal patterns or market shifts.
    """
    np.random.seed(seed)
    class_samples = df[df["target"] == target_class]
    extra_samples = class_samples.sample(
        n=int(len(class_samples) * multiplier),
        replace=True,
        random_state=seed,
    )
    df_drifted = pd.concat([df, extra_samples], ignore_index=True)
    logger.info(f"Label shift injected: class {target_class} oversampled by {multiplier}x")
    return df_drifted


def inject_concept_drift(df: pd.DataFrame, noise_level: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """
    Inject concept drift: P(Y|X) changes. 
    Simulates genuine change in underlying relationship.
    """
    np.random.seed(seed)
    df_drifted = df.copy()
    n_flip = int(len(df) * noise_level * 0.2)
    flip_indices = np.random.choice(df.index, n_flip, replace=False)
    n_classes = df["target"].nunique()
    for idx in flip_indices:
        current = df_drifted.loc[idx, "target"]
        new_label = np.random.choice([c for c in range(n_classes) if c != current])
        df_drifted.loc[idx, "target"] = new_label

    logger.info(f"Concept drift injected: {n_flip} labels flipped")
    return df_drifted


def generate_production_stream(
    n_batches: int = 10,
    batch_size: int = 50,
    drift_start_batch: int = 5,
    drift_type: str = "covariate",
    drift_magnitude: float = 0.4,
    output_dir: str = "data/production",
    seed: int = 42,
):
    """
    Generate a stream of production batches with drift starting at drift_start_batch.
    Saves each batch as a timestamped CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    reference_df = load_reference_data()
    batch_metadata = []

    for batch_idx in range(n_batches):
        batch_seed = seed + batch_idx
        batch = reference_df.sample(n=batch_size, replace=True, random_state=batch_seed)
        drifted = batch_idx >= drift_start_batch

        if drifted:
            # Gradually increase drift magnitude
            progressive_magnitude = drift_magnitude * (1 + 0.1 * (batch_idx - drift_start_batch))
            if drift_type == "covariate":
                batch = inject_covariate_shift(batch, progressive_magnitude, batch_seed)
            elif drift_type == "label":
                batch = inject_label_shift(batch, seed=batch_seed)
            elif drift_type == "concept":
                batch = inject_concept_drift(batch, progressive_magnitude, batch_seed)

        timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_{batch_idx:03d}")
        fname = f"batch_{timestamp}.csv"
        fpath = os.path.join(output_dir, fname)
        batch.to_csv(fpath, index=False)

        meta = {
            "batch_idx": batch_idx,
            "filename": fname,
            "size": len(batch),
            "is_drifted": drifted,
            "drift_type": drift_type if drifted else "none",
        }
        batch_metadata.append(meta)
        logger.info(f"Batch {batch_idx}: {fname} | drifted={drifted}")

    meta_path = os.path.join(output_dir, "batch_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(batch_metadata, f, indent=2)

    # Save reference data
    ref_path = os.path.join(output_dir, "reference.csv")
    reference_df.to_csv(ref_path, index=False)

    logger.info(f"Generated {n_batches} batches in {output_dir}")
    logger.info(f"Drift starts at batch {drift_start_batch} ({drift_type})")
    return batch_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--drift-start", type=int, default=5)
    parser.add_argument("--drift-type", choices=["covariate", "label", "concept"], default="covariate")
    parser.add_argument("--drift-magnitude", type=float, default=0.4)
    parser.add_argument("--output-dir", type=str, default="data/production")
    args = parser.parse_args()

    generate_production_stream(
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        drift_start_batch=args.drift_start,
        drift_type=args.drift_type,
        drift_magnitude=args.drift_magnitude,
        output_dir=args.output_dir,
    )
