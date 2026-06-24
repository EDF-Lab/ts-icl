#!/usr/bin/env python3
"""
Compute Overall Leaderboard from TIME evaluation results.

This script:
1. Downloads Seasonal Naive results from HuggingFace Hub (if not found locally)
2. Loads your model results from output/results/{model_name}
3. Computes Overall leaderboard metrics (normalized by Seasonal Naive)
4. Prints the results in a formatted table

Usage:
    python scripts/compute_overall_leaderboard.py

The script uses default values:
    - Results directory: output/results
    - Sorting metric: MASE

Requirements:
    - huggingface_hub
    - pandas
    - numpy
    - scipy
"""

import sys
from pathlib import Path
from typing import Optional
import argparse

import pandas as pd
import numpy as np
from scipy import stats
from huggingface_hub import snapshot_download


# HuggingFace repository for Seasonal Naive results
HF_OUTPUT_REPO_ID = "Real-TSF/TIME-Output"
SEASONAL_NAIVE_MODEL = "seasonal_naive"


def check_local_seasonal_naive(results_dir: Path) -> Optional[Path]:
    """
    Check if Seasonal Naive results exist locally.

    Args:
        results_dir: Path to results directory (e.g., output/results)

    Returns:
        Path to results directory if found, None otherwise
    """
    seasonal_naive_path = results_dir / SEASONAL_NAIVE_MODEL
    if seasonal_naive_path.exists() and seasonal_naive_path.is_dir():
        # Check if it contains at least one dataset result
        has_results = False
        for dataset_dir in seasonal_naive_path.iterdir():
            if dataset_dir.is_dir():
                # Check if it has freq subdirectories with horizon results
                for freq_dir in dataset_dir.iterdir():
                    if freq_dir.is_dir():
                        for horizon in ["short", "medium", "long"]:
                            config_path = freq_dir / horizon / "config.json"
                            if config_path.exists():
                                has_results = True
                                break
                        if has_results:
                            break
                if has_results:
                    break

        if has_results:
            return results_dir
    return None


def download_seasonal_naive_results(cache_dir: Optional[str] = None) -> Path:
    """
    Download Seasonal Naive results from HuggingFace Hub.

    Args:
        cache_dir: Optional cache directory. If None, uses default HF cache.

    Returns:
        Path to the downloaded results directory (containing seasonal_naive subdirectory)
    """
    print(f"📥 Downloading Seasonal Naive results from HuggingFace Hub...")
    print(f"   Repository: {HF_OUTPUT_REPO_ID}")

    try:
        local_dir = snapshot_download(
            repo_id=HF_OUTPUT_REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"results/{SEASONAL_NAIVE_MODEL}/**"],
            cache_dir=cache_dir,
        )

        results_path = Path(local_dir) / "results"
        seasonal_naive_path = results_path / SEASONAL_NAIVE_MODEL

        if not seasonal_naive_path.exists():
            raise FileNotFoundError(f"Seasonal Naive results not found at {seasonal_naive_path}")

        print(f"✅ Seasonal Naive results cached at: {seasonal_naive_path}")
        return results_path  # Return results directory, not just seasonal_naive
    except Exception as e:
        print(f"❌ Error downloading Seasonal Naive results: {e}")
        raise


def load_time_results(root_dir: Path, model_name: str, dataset_with_freq: str, horizon: str):
    """
    Load TIME results from NPZ files for a specific model, dataset, and horizon.

    Args:
        root_dir: Root directory containing TIME results
        model_name: Model name (e.g., "chronos2")
        dataset_with_freq: Dataset and freq combined (e.g., "Traffic/15T")
        horizon: Horizon name (e.g., "short", "medium", "long")

    Returns:
        tuple: (metrics_dict, config_dict) or (None, None) if not found
    """
    horizon_dir = root_dir / model_name / dataset_with_freq / horizon
    metrics_path = horizon_dir / "metrics.npz"
    config_path = horizon_dir / "config.json"

    if not metrics_path.exists():
        return None, None

    metrics = np.load(metrics_path)
    metrics_dict = {k: metrics[k] for k in metrics.files}

    config_dict = {}
    if config_path.exists():
        import json
        with open(config_path, "r") as f:
            config_dict = json.load(f)

    return metrics_dict, config_dict


def get_all_datasets_results(results_root: Path) -> pd.DataFrame:
    """
    Load dataset-level leaderboard by reading TIME NPZ files and aggregating.

    Args:
        results_root: Path to the TIME results root directory

    Returns:
        pd.DataFrame: DataFrame containing dataset-level results with columns
            ["model", "dataset", "freq", "dataset_id", "horizon", "MASE", "CRPS", "MAE", "MSE"]
    """
    rows = []

    if not results_root.exists():
        print(f"❌ Error: results_root={results_root} does not exist")
        return pd.DataFrame(columns=["model", "dataset", "freq", "dataset_id", "horizon", "MASE", "CRPS", "MAE", "MSE"])

    for model_dir in results_root.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        if model_name == ".hydra":
            continue

        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name

            # Nested structure: model/dataset/freq/horizon/
            for freq_dir in dataset_dir.iterdir():
                if not freq_dir.is_dir():
                    continue

                freq_name = freq_dir.name

                for horizon in ["short", "medium", "long"]:
                    dataset_with_freq = f"{dataset_name}/{freq_name}"
                    metrics_dict, _ = load_time_results(results_root, model_name, dataset_with_freq, horizon)

                    if metrics_dict is None:
                        continue

                    # Aggregate metrics across all series/windows/variates
                    mase = np.nanmean(metrics_dict.get("MASE", np.array([])))
                    crps = np.nanmean(metrics_dict.get("CRPS", np.array([])))
                    mae = np.nanmean(metrics_dict.get("MAE", np.array([])))
                    mse = np.nanmean(metrics_dict.get("MSE", np.array([])))

                    rows.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "freq": freq_name,
                        "dataset_id": dataset_with_freq,
                        "horizon": horizon,
                        "MASE": mase,
                        "CRPS": crps,
                        "MAE": mae,
                        "MSE": mse,
                    })

    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame(columns=["model", "dataset", "freq", "dataset_id", "horizon", "MASE", "CRPS", "MAE", "MSE"])


def compute_ranks(df: pd.DataFrame, groupby_cols: list) -> pd.DataFrame:
    """
    Compute ranks for models across datasets based on MASE and CRPS.

    Args:
        df: Dataset-level results with columns ["model", "dataset_id", "horizon", "MASE", "CRPS"]
        groupby_cols: Columns to group by for ranking

    Returns:
        DataFrame with added ["MASE_rank", "CRPS_rank"] columns
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df["MASE_rank"] = df.groupby(groupby_cols)["MASE"].rank(method="first", ascending=True)
    df["CRPS_rank"] = df.groupby(groupby_cols)["CRPS"].rank(method="first", ascending=True)

    return df


def normalize_by_seasonal_naive(
    df: pd.DataFrame,
    baseline_model: str = "seasonal_naive",
    metrics: list = None,
    groupby_cols: list = None,
) -> pd.DataFrame:
    """
    Normalize metrics by Seasonal Naive baseline for each (dataset_id, horizon) group.

    Args:
        df: Dataset-level results with columns including ["model", "dataset_id", "horizon", "MASE", "CRPS"]
        baseline_model: Name of the baseline model
        metrics: List of metric columns to normalize
        groupby_cols: Columns to group by for normalization

    Returns:
        DataFrame with normalized metric values
    """
    if metrics is None:
        metrics = ["MASE", "CRPS"]
    if groupby_cols is None:
        groupby_cols = ["dataset_id", "horizon"]

    if df.empty:
        return df.copy()

    # Check if baseline model exists
    if baseline_model not in df["model"].values:
        print(f"⚠️  Warning: baseline model '{baseline_model}' not found in data")
        return pd.DataFrame()

    # Work on a copy
    df_normalized = df.copy()

    # Get baseline values for each group
    baseline_df = df[df["model"] == baseline_model].copy()

    # Create a mapping: (dataset_id, horizon) -> {metric: baseline_value}
    baseline_values = {}
    for _, row in baseline_df.iterrows():
        key = tuple(row[col] for col in groupby_cols)
        baseline_values[key] = {metric: row[metric] for metric in metrics}

    # Normalize each row
    rows_to_keep = []
    for idx, row in df_normalized.iterrows():
        key = tuple(row[col] for col in groupby_cols)

        # Skip configurations without baseline results
        if key not in baseline_values:
            continue

        rows_to_keep.append(idx)

        # Normalize each metric
        for metric in metrics:
            baseline_val = baseline_values[key][metric]
            if baseline_val is not None and baseline_val != 0 and not np.isnan(baseline_val):
                df_normalized.at[idx, metric] = row[metric] / baseline_val
            else:
                df_normalized.at[idx, metric] = np.nan

    # Keep only rows with valid baseline
    df_normalized = df_normalized.loc[rows_to_keep].copy()

    # Handle any remaining inf values
    for metric in metrics:
        df_normalized[metric] = df_normalized[metric].replace([np.inf, -np.inf], np.nan)

    return df_normalized


def get_overall_leaderboard(df_datasets: pd.DataFrame, metric: str = "MASE") -> pd.DataFrame:
    """
    Compute overall leaderboard across datasets by normalizing metrics by Seasonal Naive
    and aggregating with geometric mean.

    Args:
        df_datasets: Dataset-level results, must include
            ["model", "dataset_id", "horizon", "MASE", "CRPS", "MASE_rank", "CRPS_rank"]
        metric: Metric to use for sorting. Defaults to "MASE"

    Returns:
        DataFrame: Leaderboard with:
            - MASE (norm.), CRPS (norm.): Geometric mean of Seasonal Naive-normalized values
            - MASE_rank, CRPS_rank: Average rank across configurations
            Sorted by the chosen metric.
    """
    if df_datasets.empty:
        return pd.DataFrame()

    if metric not in df_datasets.columns:
        return pd.DataFrame()

    # Step 1: Normalize MASE and CRPS by Seasonal Naive per (dataset_id, horizon)
    df_normalized = normalize_by_seasonal_naive(
        df_datasets,
        baseline_model=SEASONAL_NAIVE_MODEL,
        metrics=["MASE", "CRPS"],
        groupby_cols=["dataset_id", "horizon"],
    )

    if df_normalized.empty:
        print("❌ Error: Normalization failed. Make sure Seasonal Naive results are available.")
        return pd.DataFrame()

    # Step 2: Aggregate normalized MASE and CRPS with geometric mean
    def gmean_with_nan(x):
        """Compute geometric mean, ignoring NaN values."""
        valid = x.dropna()
        if len(valid) == 0:
            return np.nan
        return stats.gmean(valid)

    normalized_metrics = (
        df_normalized.groupby("model")[["MASE", "CRPS"]]
        .agg(gmean_with_nan)
        .reset_index()
    )

    # Rename columns
    normalized_metrics = normalized_metrics.rename(columns={
        "MASE": "MASE (norm.)",
        "CRPS": "CRPS (norm.)"
    })

    # Step 3: Compute average ranks from original data (pre-normalized)
    if "MASE_rank" in df_datasets.columns and "CRPS_rank" in df_datasets.columns:
        # Use the same configurations that were used in normalization
        df_with_baseline = df_datasets[
            df_datasets.set_index(["dataset_id", "horizon"]).index.isin(
                df_normalized.set_index(["dataset_id", "horizon"]).index.unique()
            )
        ]
        avg_ranks = (
            df_with_baseline.groupby("model")[["MASE_rank", "CRPS_rank"]]
            .mean()
            .reset_index()
        )
        # Merge normalized metrics with average ranks
        leaderboard = normalized_metrics.merge(avg_ranks, on="model", how="left")
    else:
        leaderboard = normalized_metrics

    # Step 4: Sort by chosen metric
    sort_metric = "MASE (norm.)" if metric == "MASE" else "CRPS (norm.)"

    if sort_metric in leaderboard.columns:
        leaderboard = leaderboard.sort_values(by=sort_metric, ascending=True).reset_index(drop=True)
    else:
        leaderboard = leaderboard.sort_values(by=leaderboard.columns[1], ascending=True).reset_index(drop=True)

    # Step 5: Select and order columns
    col_order = ["model", "MASE (norm.)", "CRPS (norm.)", "MASE_rank", "CRPS_rank"]
    col_order = [col for col in col_order if col in leaderboard.columns]
    leaderboard = leaderboard[col_order]
    leaderboard = leaderboard.round(3)

    return leaderboard


def main(
    results_dir,
    metric: str = "MASE"
):
    # Fixed configuration
    cache_dir = None

    print("=" * 80)
    print("TIME Overall Leaderboard Calculator")
    print("=" * 80)
    print()

    # Step 1: Check for local Seasonal Naive results or download
    results_root = Path(results_dir)
    print("Step 1: Checking for Seasonal Naive results...")

    # First check if seasonal_naive exists locally in the results directory
    local_seasonal_naive = check_local_seasonal_naive(results_root)

    if local_seasonal_naive:
        print(f"✅ Found local Seasonal Naive results at {results_root / SEASONAL_NAIVE_MODEL}")
        seasonal_naive_results_path = results_root
    else:
        print(f"   Local Seasonal Naive results not found at {results_root / SEASONAL_NAIVE_MODEL}")
        print("   Downloading from HuggingFace Hub...")
        try:
            downloaded_results_path = download_seasonal_naive_results(cache_dir=cache_dir)
            seasonal_naive_results_path = downloaded_results_path
        except Exception as e:
            print(f"❌ Failed to download Seasonal Naive results: {e}")
            sys.exit(1)

    # Step 2: Load user results
    print(f"\nStep 2: Loading results from {results_root}...")

    if not results_root.exists():
        print(f"❌ Error: Results directory does not exist: {results_root}")
        sys.exit(1)

    # Load user results
    user_results = get_all_datasets_results(results_root)

    if user_results.empty:
        print(f"❌ No results found in {results_root}")
        sys.exit(1)

    print(f"✅ Loaded {len(user_results)} user results")

    # Step 3: Load Seasonal Naive results
    print(f"\nStep 3: Loading Seasonal Naive results...")
    seasonal_naive_results = get_all_datasets_results(seasonal_naive_results_path)

    # Filter to only seasonal_naive model
    seasonal_naive_results = seasonal_naive_results[
        seasonal_naive_results["model"] == SEASONAL_NAIVE_MODEL
    ]

    if seasonal_naive_results.empty:
        print(f"❌ No Seasonal Naive results found")
        sys.exit(1)

    print(f"✅ Loaded {len(seasonal_naive_results)} Seasonal Naive results")

    # Step 4: Merge results
    print(f"\nStep 4: Merging results...")
    all_results = pd.concat([user_results, seasonal_naive_results], ignore_index=True)

    # Compute ranks
    all_results = compute_ranks(all_results, groupby_cols=["dataset_id", "horizon"])

    print(f"✅ Total configurations: {len(all_results)}")
    print(f"   Models: {sorted(all_results["model"].unique())}")
    print(f"   Datasets: {len(all_results["dataset_id"].unique())}")

    # Step 5: Compute Overall Leaderboard
    print(f"\nStep 5: Computing Overall Leaderboard...")
    leaderboard = get_overall_leaderboard(all_results, metric=metric)

    if leaderboard.empty:
        print("❌ Failed to compute leaderboard")
        sys.exit(1)

    # Step 6: Display results
    print("\n" + "=" * 80)
    print("Overall Leaderboard")
    print("=" * 80)
    print()
    print(leaderboard.to_string(index=False))
    print()

    print("\n" + "=" * 80)
    print("Note: Metrics are normalized by Seasonal Naive baseline.")
    print("      Lower values are better. Seasonal Naive = 1.0")
    print("=" * 80)

    leaderboard.to_csv(results_root / "leaderboard.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results_dir", type=str, default="outputs/Benchmarks/TIME")

    args = parser.parse_args()
    results_dir = args.results_dir
    main(results_dir)