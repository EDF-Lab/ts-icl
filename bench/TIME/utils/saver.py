"""
Prediction saving utilities for per-window evaluation.

Output structure:
    output_dir/
        {dataset_config}/
            predictions.npz  # Contains predictions and ground truth
            metrics.npz      # Contains per-window metrics
            config.json    # Contains dataset config
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (
    compute_per_window_metrics_from_quantiles,
)


def save_window_predictions(
    dataset,
    fc_quantiles: np.ndarray,
    ds_config: str,
    output_base_dir: str | Path,
    seasonality: int = 1,
    model_hyperparams: dict | None = None,
    quantile_levels: list[float] | None = None,
) -> dict:
    """
    Save predictions and metrics for each test window.

    Args:
        dataset: Dataset object with attributes:
            - test_data: Test data iterator
            - windows: Number of test windows
            - target_dim: Number of variates
            - prediction_length: Forecast horizon
            - freq: Data frequency string
        fc_quantiles: Concatenated quantile prediction array.
            Shape:
                - (num_total_instances, num_quantiles, prediction_length) for univariate
                - (num_total_instances, num_quantiles, num_variates, prediction_length) for multivariate
            where num_total_instances = num_series_exp * num_windows
        ds_config: Dataset configuration string, e.g., "m4_weekly/W/short"
        output_base_dir: Base directory for output files
        seasonality: Seasonal period length for MASE computation
        model_hyperparams: Dictionary of model hyperparameters to save in config
        quantile_levels: Quantile levels for output (default: [0.1, 0.2, ..., 0.9])

    Output files:
        predictions.npz:
            - predictions_quantiles: (num_series, num_windows, num_quantiles, num_variates, prediction_length)
            - quantile_levels

        metrics.npz:
            - Each metric array with shape (num_series, num_windows, num_variates)

        config.json:
            - Dataset configuration and config (without shapes and metric_shape)

    Returns:
        config: Dictionary containing dataset config
    """
    # Setup quantile levels
    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantile_levels_array = np.array(quantile_levels, dtype=float)
    quantile_levels_list = [float(q) for q in quantile_levels_array.tolist()]
    num_quantiles = len(quantile_levels_list)

    # Create output directory for this dataset config
    ds_output_dir = os.path.join(output_base_dir, ds_config)
    os.makedirs(ds_output_dir, exist_ok=True)

    test_data = dataset.test_data
    num_windows = dataset.windows
    prediction_length = dataset.prediction_length
    original_num_variates = dataset.target_dim  # Original variates before to_univariate

    # Count number of series during experiment (test_data contains num_series_exp * num_windows instances)
    num_total_instances = fc_quantiles.shape[0]
    num_series_exp = num_total_instances // num_windows  # != num_series in original dataset if to_univariate is True

    print(f"    Total instances: {num_total_instances}, Series during experiment: {num_series_exp}, Windows: {num_windows}")

    # Collect ground truth labels and contexts for metrics computation
    print("    Collecting ground truth and context...")
    ground_truths = []
    contexts = []
    for inp, label in test_data:
        ground_truths.append(label["target"])
        contexts.append(inp["target"])

    # Determine if forecasts are univariate by checking the shape
    forecast_is_univariate = False
    if fc_quantiles.ndim == 3:
        # Shape: (num_total_instances, num_quantiles, prediction_length) -> univariate
        assert fc_quantiles.shape[1] == num_quantiles, \
            f"Mismatch in quantiles (dim 1). Expected {num_quantiles}, but got {fc_quantiles.shape[1]}. Full shape: {fc_quantiles.shape}"
        assert fc_quantiles.shape[2] == prediction_length, \
            f"Mismatch in prediction_length (dim 2). Expected {prediction_length}, but got {fc_quantiles.shape[2]}. Full shape: {fc_quantiles.shape}"
        forecast_is_univariate = True

    elif fc_quantiles.ndim == 4:
        # Shape: (num_total_instances, num_quantiles, num_variates, prediction_length)
        assert fc_quantiles.shape[1] == num_quantiles, \
            f"Mismatch in quantiles (dim 1). Expected {num_quantiles}, but got {fc_quantiles.shape[1]}. Full shape: {fc_quantiles.shape}"
        assert fc_quantiles.shape[2] == original_num_variates, \
            f"Mismatch in num_variates (dim 2). Expected {original_num_variates}, but got {fc_quantiles.shape[2]}. Full shape: {fc_quantiles.shape}"
        assert fc_quantiles.shape[3] == prediction_length, \
            f"Mismatch in prediction_length (dim 3). Expected {prediction_length}, but got {fc_quantiles.shape[3]}. Full shape: {fc_quantiles.shape}"
    else:
        raise ValueError(f"Expected fc_quantiles to have 3 or 4 dimensions, but got {fc_quantiles.ndim}. Shape: {fc_quantiles.shape}")


    # Detect to_univariate case: forecasts are univariate but original dataset has multiple variates
    # In this case, num_series_exp = original_num_series * original_num_variates
    is_to_univariate = forecast_is_univariate and original_num_variates > 1 and \
                       (num_series_exp % original_num_variates == 0)

    if is_to_univariate:
        # Reverse the to_univariate transformation
        # MultivariateToUnivariate expands as: series_0_dim0, series_0_dim1, ..., series_1_dim0, ...
        num_series = num_series_exp // original_num_variates
        num_variates = original_num_variates
        print(f"    Detected to_univariate case: {num_series_exp} flat series -> {num_series} series x {num_variates} variates")
    else:
        num_series = num_series_exp
        num_variates = original_num_variates if not forecast_is_univariate else 1

    # Initialize arrays
    predictions_quantiles = np.zeros((num_series, num_windows, num_quantiles, num_variates, prediction_length))
    ground_truth = np.zeros((num_series, num_windows, num_variates, prediction_length))

    # Find max context length for metrics computation
    context_len = max(ctx.shape[-1] for ctx in contexts)
    context_array = np.full((num_series, num_windows, num_variates, context_len), np.nan)

    print("    Organizing data into arrays...")
    for idx, (gt, ctx) in enumerate(zip(ground_truths, contexts)):
        # Extract fc_q from the concatenated array
        fc_q = fc_quantiles[idx]

        # Normalize fc_q to shape: (num_quantiles, num_variates, prediction_length)
        if fc_q.ndim == 2:
            # (num_quantiles, prediction_length) -> (num_quantiles, 1, prediction_length)
            fc_q = fc_q[:, np.newaxis, :]

        # Normalize ground truth
        if gt.ndim == 1:
            gt = gt[np.newaxis, :]

        # Normalize context
        if ctx.ndim == 1:
            ctx = ctx[np.newaxis, :]

        if is_to_univariate:
            # In to_univariate case:
            # idx iterates as: series_0_dim0_win0, series_0_dim0_win1, ..., series_0_dim1_win0, ...
            # We need to map flat_series_idx to (series_idx, variate_idx)
            flat_series_idx = idx // num_windows
            window_idx = idx % num_windows
            series_idx = flat_series_idx // num_variates
            variate_idx = flat_series_idx % num_variates

            predictions_quantiles[series_idx, window_idx, :, variate_idx, :] = fc_q[:, 0, :]
            ground_truth[series_idx, window_idx, variate_idx, :] = gt[0, :]

            ctx_len_actual = ctx.shape[-1]
            context_array[series_idx, window_idx, variate_idx, :ctx_len_actual] = ctx[0, :]
        else:
            series_idx = idx // num_windows
            window_idx = idx % num_windows

            predictions_quantiles[series_idx, window_idx] = fc_q
            ground_truth[series_idx, window_idx] = gt

            # Handle context shape mismatch
            if ctx.shape[0] != num_variates and ctx.shape[-1] == num_variates:
                ctx = ctx.T
            ctx_len_actual = ctx.shape[-1]
            context_array[series_idx, window_idx, :, :ctx_len_actual] = ctx

    # Save quantiles to npz file
    # Use float16 to reduce storage (sufficient for visualization purposes)
    # Apply dynamic scaling to prevent float16 overflow (max ~65504)
    FLOAT16_SAFE_MAX = 60000.0  # Leave margin below 65504
    max_abs_val = np.abs(predictions_quantiles).max()

    prediction_scale_factor = 1.0
    if max_abs_val > FLOAT16_SAFE_MAX:
        # Calculate scale factor as power of 10 for cleaner scaling
        prediction_scale_factor = float(10 ** np.ceil(np.log10(max_abs_val / FLOAT16_SAFE_MAX)))
        print(f"    ⚠️  Predictions max value ({max_abs_val:.2f}) exceeds float16 safe range.")
        print(f"    ⚠️  Applying scale factor: {prediction_scale_factor:.0f} (values will be divided)")
        predictions_quantiles_scaled = predictions_quantiles / prediction_scale_factor
    else:
        predictions_quantiles_scaled = predictions_quantiles

    npz_path = os.path.join(ds_output_dir, "predictions.npz")
    np.savez_compressed(
        npz_path,
        predictions_quantiles=predictions_quantiles_scaled.astype(np.float16),
        quantile_levels=quantile_levels_array.astype(np.float16),
    )
    print(f"    Saved predictions (quantiles) to {npz_path}")

    # Compute per-window metrics
    print("    Computing per-window metrics...")
    metrics = compute_per_window_metrics_from_quantiles(
        predictions_quantiles=predictions_quantiles,
        ground_truth=ground_truth,
        context=context_array,
        seasonality=seasonality,
        quantile_levels=quantile_levels_list,
    )

    # Save metrics to npz file
    metrics_path = os.path.join(ds_output_dir, "metrics.npz")
    np.savez_compressed(metrics_path, **metrics)
    print(f"    Saved metrics to {metrics_path}")

    # Save config
    config = {
        "dataset_config": ds_config,
        "num_series": num_series,
        "num_windows": num_windows,
        "num_variates": num_variates,
        "prediction_length": prediction_length,
        "num_quantiles": num_quantiles,
        "quantile_levels": quantile_levels_list,
        "freq": dataset.freq,
        "seasonality": seasonality,
        "context_length": context_len,
        "metric_names": list(metrics.keys()),
        "prediction_scale_factor": prediction_scale_factor,  # For float16 overflow prevention
    }

    # Add model hyperparameters if provided
    if model_hyperparams:
        config.update(model_hyperparams)

    config_path = os.path.join(ds_output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"    Saved config to {config_path}")

    # Print average metrics summary
    print("    Metrics summary (averaged over all series/windows/variates):")
    for metric_name, metric_values in metrics.items():
        mean_val = np.nanmean(metric_values)
        print(f"      {metric_name}: {mean_val:.4f}")
    
    metrics_summary = {
        'Dataset' : ds_config,
        'nb_chunks' : int( config['num_series'] * config['num_windows'] * config['num_variates'] )
    } | {
        metric_name: [np.nanmean(metric_values)] for metric_name, metric_values in metrics.items()
    }
    df = pd.DataFrame(metrics_summary)
    df.set_index('Dataset', inplace=True)
    df.to_csv(os.path.join(ds_output_dir, "summary.csv"))

    return config
