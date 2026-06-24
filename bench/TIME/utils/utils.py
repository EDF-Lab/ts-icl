"""
Utility functions.
"""
import yaml
import numpy as np

def get_available_terms(dataset_name: str, config: dict) -> list[str]:
    """Get the terms that are actually defined in the config for a dataset."""
    datasets_config = config.get("datasets", {})
    if dataset_name not in datasets_config:
        return []
    dataset_config = datasets_config[dataset_name]
    available_terms = []
    for term in ["short", "medium", "long"]:
        if term in dataset_config and dataset_config[term].get("prediction_length") is not None:
            available_terms.append(term)
    return available_terms


def impute_nans_1d(series: np.ndarray) -> np.ndarray:
    series = series.astype(np.float32, copy=False)
    if not np.isnan(series).any():
        return series
    idx = np.arange(series.shape[0])
    mask = np.isfinite(series)
    if mask.sum() == 0:
        return np.nan_to_num(series, nan=0.0)
    series[~mask] = np.interp(idx[~mask], idx[mask], series[mask])
    return series


def clean_nan_target(series: np.ndarray) -> np.ndarray:
    if series.ndim == 1:
        return impute_nans_1d(series)
    if series.ndim == 2:
        cleaned = np.empty_like(series, dtype=np.float32)
        for i in range(series.shape[0]):
            cleaned[i] = impute_nans_1d(series[i])
        return cleaned
    return np.nan_to_num(series, nan=0.0)


def parse_dataset_key(dataset_key: str) -> tuple[str, str]:
    """
    Parse dataset key format '{dataset}/{freq}' in datasets.yaml

    Args:
        dataset_key: e.g., 'exchange_rate/D', 'ETTh1/H', 'bitbrains_rnd/5T'

    Returns:
        (dataset_name, freq): e.g., ('exchange_rate', 'D')
    """
    parts = dataset_key.split('/')
    if len(parts) != 2:
        raise ValueError(f"Invalid dataset key format: {dataset_key}. Expected 'dataset/freq'")
    return parts[0], parts[1]


def find_dataset_config(datasets_config: dict, dataset_key: str) -> tuple[str, str, dict]:
    """
    Find dataset configuration in datasets.yaml

    Args:
        datasets_config: 'datasets' dictionary in datasets.yaml
        dataset_key: dataset key, format as '{dataset_name}/{freq}' (e.g., 'IMOS/15T')

    Returns:
        (dataset_key, freq, config)
    """
    if dataset_key in datasets_config:
        name, freq = parse_dataset_key(dataset_key)
        return dataset_key, freq, datasets_config[dataset_key]

    # Compatible with old case of only passing dataset_name
    for key, config in datasets_config.items():
        name, freq = parse_dataset_key(key)
        if name == dataset_key:
            return key, freq, config

    raise ValueError(f"Dataset '{dataset_key}' not found in config")


def load_datasets_config(config_path: str) -> dict:
    """Load datasets.yaml configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_test_length(dataset_config: dict) -> int | None:
    """
    Get test_length value for a dataset

    Args:
        dataset_config: dataset specific configuration

    Returns:
        test_length value, if not configured, return None
    """
    if dataset_config and "test_length" in dataset_config:
        return dataset_config["test_length"]
    return None
