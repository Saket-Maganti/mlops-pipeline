"""
Scenario generation for drift-response benchmark experiments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split


@dataclass
class ScenarioSpec:
    name: str
    drift_type: str
    batch_size: int = 48
    n_batches: int = 8
    drift_start_batch: int = 3
    base_magnitude: float = 0.35
    label_delay_batches: int = 0
    seed: int = 42
    train_size: float = 0.55
    reference_size: float = 0.20


DEFAULT_SCENARIOS: List[ScenarioSpec] = [
    ScenarioSpec(name="mild_covariate", drift_type="covariate", base_magnitude=0.20, seed=41),
    ScenarioSpec(name="severe_covariate", drift_type="covariate", base_magnitude=0.55, seed=42),
    ScenarioSpec(name="label_shift", drift_type="label", base_magnitude=0.30, seed=43),
    ScenarioSpec(name="concept_drift", drift_type="concept", base_magnitude=0.45, seed=44),
    ScenarioSpec(name="hybrid_regime_shift", drift_type="hybrid", base_magnitude=0.60, label_delay_batches=2, seed=45),
    ScenarioSpec(name="adaptive_recovery_window", drift_type="concept", n_batches=12, drift_start_batch=2, base_magnitude=0.95, label_delay_batches=1, seed=46),
    ScenarioSpec(name="late_breaking_regime_shift", drift_type="hybrid", n_batches=12, drift_start_batch=4, base_magnitude=0.90, label_delay_batches=1, seed=47),
]


def load_base_dataset(data_path: Optional[str] = None, dataset_name: str = "wine") -> pd.DataFrame:
    """Load a classification dataset with a target column."""
    if data_path:
        df = pd.read_csv(data_path)
        if "target" not in df.columns:
            raise ValueError("Custom dataset must include a 'target' column.")
        return df

    if dataset_name == "wine":
        source = load_wine()
        df = pd.DataFrame(source.data, columns=source.feature_names)
        df["target"] = source.target
    elif dataset_name == "breast_cancer":
        source = load_breast_cancer()
        df = pd.DataFrame(source.data, columns=source.feature_names)
        df["target"] = source.target
    elif dataset_name == "digits":
        source = load_digits()
        feature_names = [f"pixel_{index}" for index in range(source.data.shape[1])]
        df = pd.DataFrame(source.data, columns=feature_names)
        df["target"] = source.target
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    return df


def parse_scenario_specs(raw_specs: Optional[List[Dict[str, object]]]) -> List[ScenarioSpec]:
    """Parse scenario definitions from config data."""
    if not raw_specs:
        return DEFAULT_SCENARIOS
    specs = []
    for raw_spec in raw_specs:
        specs.append(ScenarioSpec(**raw_spec))
    return specs


def split_reference_data(
    df: pd.DataFrame,
    train_size: float = 0.55,
    reference_size: float = 0.20,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Split base dataset into training, reference, and holdout sets."""
    if train_size + reference_size >= 1.0:
        raise ValueError("train_size + reference_size must be < 1.0")

    train_df, remaining = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df["target"],
    )
    reference_fraction = reference_size / (1.0 - train_size)
    reference_df, holdout_df = train_test_split(
        remaining,
        train_size=reference_fraction,
        random_state=seed,
        stratify=remaining["target"],
    )
    return {
        "train": train_df.reset_index(drop=True),
        "reference": reference_df.reset_index(drop=True),
        "holdout": holdout_df.reset_index(drop=True),
    }


def apply_drift(df: pd.DataFrame, drift_type: str, magnitude: float, seed: int) -> pd.DataFrame:
    """Apply a specific type of synthetic drift to a batch."""
    rng = np.random.default_rng(seed)
    drifted = df.copy()
    feature_cols = [col for col in drifted.columns if col != "target"]

    if drift_type == "covariate":
        n_features = max(1, int(len(feature_cols) * 0.6))
        selected = rng.choice(feature_cols, size=n_features, replace=False)
        for col in selected:
            std = max(drifted[col].std(), 1e-6)
            mean_shift = rng.choice([-1.0, 1.0]) * magnitude * std
            scale = 1.0 + magnitude * rng.uniform(0.15, 0.75)
            noise = rng.normal(0.0, std * magnitude * 0.15, size=len(drifted))
            drifted[col] = drifted[col] * scale + mean_shift + noise

    elif drift_type == "label":
        target_class = int(rng.choice(sorted(drifted["target"].unique())))
        class_rows = drifted[drifted["target"] == target_class]
        extra_rows = class_rows.sample(
            n=max(1, int(len(class_rows) * (1.0 + magnitude))),
            replace=True,
            random_state=seed,
        )
        drifted = pd.concat([drifted, extra_rows], ignore_index=True)
        drifted = drifted.sample(n=len(df), replace=True, random_state=seed).reset_index(drop=True)

    elif drift_type == "concept":
        classes = sorted(int(value) for value in drifted["target"].unique())
        primary_feature = feature_cols[0]
        secondary_feature = feature_cols[min(1, len(feature_cols) - 1)]
        combined_signal = (
            drifted[primary_feature].rank(pct=True)
            + (1.0 + magnitude) * drifted[secondary_feature].rank(pct=True)
        )
        percent_rank = combined_signal.rank(method="first", pct=True).to_numpy()
        class_index = np.minimum((percent_rank * len(classes)).astype(int), len(classes) - 1)
        remapped = pd.Series([classes[idx] for idx in class_index], index=drifted.index)
        flip_mask = rng.random(len(drifted)) < min(0.25 + magnitude * 0.25, 0.65)
        drifted.loc[flip_mask, "target"] = remapped.loc[flip_mask].to_numpy()

    elif drift_type == "hybrid":
        drifted = apply_drift(drifted, "covariate", magnitude, seed)
        drifted = apply_drift(drifted, "concept", min(0.75, magnitude * 0.9), seed + 101)
    else:
        raise ValueError(f"Unsupported drift type: {drift_type}")

    return drifted.reset_index(drop=True)


def create_scenario(
    base_df: pd.DataFrame,
    spec: ScenarioSpec,
    output_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Create a scenario with reference data and sequential batches."""
    splits = split_reference_data(
        base_df,
        train_size=spec.train_size,
        reference_size=spec.reference_size,
        seed=spec.seed,
    )
    train_df = splits["train"]
    reference_df = splits["reference"]
    candidate_pool = pd.concat([reference_df, splits["holdout"]], ignore_index=True)
    batches: List[pd.DataFrame] = []
    batch_metadata: List[Dict[str, object]] = []

    for batch_idx in range(spec.n_batches):
        sampled = candidate_pool.sample(
            n=spec.batch_size,
            replace=True,
            random_state=spec.seed + batch_idx,
        ).reset_index(drop=True)
        drift_active = batch_idx >= spec.drift_start_batch
        severity = 0.0
        if drift_active:
            severity = spec.base_magnitude * (1.0 + 0.18 * (batch_idx - spec.drift_start_batch))
            sampled = apply_drift(
                sampled,
                drift_type=spec.drift_type,
                magnitude=severity,
                seed=spec.seed + batch_idx,
            )

        sampled["label_available_batch"] = batch_idx + spec.label_delay_batches
        sampled["scenario_name"] = spec.name
        batches.append(sampled)
        batch_metadata.append(
            {
                "scenario_name": spec.name,
                "batch_index": batch_idx,
                "rows": len(sampled),
                "drift_active": drift_active,
                "severity": round(float(severity), 4),
                "label_available_batch": batch_idx + spec.label_delay_batches,
            }
        )

    scenario = {
        "name": spec.name,
        "drift_type": spec.drift_type,
        "seed": spec.seed,
        "spec": asdict(spec),
        "train_df": train_df,
        "reference_df": reference_df,
        "holdout_df": splits["holdout"],
        "post_drift_holdout_df": apply_drift(
            splits["holdout"],
            drift_type=spec.drift_type,
            magnitude=spec.base_magnitude * (1.0 + 0.18 * max(spec.n_batches - spec.drift_start_batch - 1, 0)),
            seed=spec.seed + 999,
        ) if spec.drift_type != "label" or spec.base_magnitude > 0 else splits["holdout"].copy(),
        "batches": batches,
        "metadata": batch_metadata,
    }

    if output_dir:
        persist_scenario(scenario, output_dir)

    return scenario


def persist_scenario(scenario: Dict[str, object], output_dir: str) -> None:
    """Persist scenario artifacts for reproducibility."""
    scenario_dir = Path(output_dir) / scenario["name"]
    scenario_dir.mkdir(parents=True, exist_ok=True)

    scenario["train_df"].to_csv(scenario_dir / "train.csv", index=False)
    scenario["reference_df"].to_csv(scenario_dir / "reference.csv", index=False)
    scenario["holdout_df"].to_csv(scenario_dir / "holdout.csv", index=False)
    scenario["post_drift_holdout_df"].to_csv(scenario_dir / "post_drift_holdout.csv", index=False)

    metadata_rows = []
    for batch_df, meta in zip(scenario["batches"], scenario["metadata"]):
        batch_path = scenario_dir / f"batch_{meta['batch_index']:02d}.csv"
        batch_df.to_csv(batch_path, index=False)
        metadata_rows.append({**meta, "path": batch_path.name})

    pd.DataFrame(metadata_rows).to_csv(scenario_dir / "scenario_manifest.csv", index=False)
    (scenario_dir / "scenario_spec.json").write_text(json.dumps(scenario["spec"], indent=2))


def create_benchmark_suite(
    output_dir: Optional[str] = None,
    data_path: Optional[str] = None,
    dataset_name: str = "wine",
    scenario_specs: Optional[List[ScenarioSpec]] = None,
) -> List[Dict[str, object]]:
    """Create all configured benchmark scenarios."""
    base_df = load_base_dataset(data_path, dataset_name=dataset_name)
    specs = scenario_specs or DEFAULT_SCENARIOS
    return [create_scenario(base_df, spec, output_dir=output_dir) for spec in specs]
