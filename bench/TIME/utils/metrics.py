"""
Per-window metrics computation for time series forecasting evaluation.
Aligned with GluonTS implementation.

Supported metrics:
- MSE: Mean Squared Error (using median forecast, aligned with GluonTS MSE[0.5])
- MAE: Mean Absolute Error (using median forecast)
- RMSE: Root Mean Squared Error (using median forecast)
- MAPE: Mean Absolute Percentage Error (using median forecast, returns fraction)
- sMAPE: Symmetric Mean Absolute Percentage Error (using median forecast, range [0, 2])
- MASE: Mean Absolute Scaled Error (using median forecast)
- ND: Normalized Deviation (using median forecast)
- CRPS: Continuous Ranked Probability Score (MeanWeightedSumQuantileLoss)

"""

import numpy as np

# Default quantile levels for CRPS computation (aligned with GluonTS)
DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def compute_per_window_metrics_from_quantiles(
    predictions_quantiles: np.ndarray,
    ground_truth: np.ndarray,
    context: np.ndarray,
    seasonality: int = 1,
    quantile_levels: list[float] = None,
) -> dict[str, np.ndarray]:
    """
    Compute evaluation metrics for each prediction window from quantile forecasts.

    Args:
        predictions_quantiles: Quantile forecasts with shape
            (num_series, num_windows, num_quantiles, num_variates, pred_len)
        ground_truth: Ground truth values with shape
            (num_series, num_windows, num_variates, pred_len)
        context: Historical context with shape
            (num_series, num_windows, num_variates, max_ctx_len)
            Note: Shorter contexts are NaN-padded
        seasonality: Seasonal period length for MASE computation
        quantile_levels: Quantile levels corresponding to predictions_quantiles.
            Defaults to [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Returns:
        Dictionary of metric arrays, each with shape (num_series, num_windows, num_variates)
        Keys: MSE, MAE, RMSE, MAPE, sMAPE, MASE, ND, CRPS
    """
    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILE_LEVELS

    (
        num_series,
        num_windows,
        num_quantiles,
        num_variates,
        pred_len,
    ) = predictions_quantiles.shape

    if len(quantile_levels) != num_quantiles:
        raise ValueError(
            "quantile_levels length must match predictions_quantiles' quantile dimension"
        )
    if 0.5 not in quantile_levels:
        raise ValueError("quantile_levels must include 0.5 for median-based metrics")
    median_idx = quantile_levels.index(0.5)

    # Initialize metric arrays: (num_series, num_windows, num_variates)
    mse = np.zeros((num_series, num_windows, num_variates))
    mae = np.zeros((num_series, num_windows, num_variates))
    rmse = np.zeros((num_series, num_windows, num_variates))
    mape = np.zeros((num_series, num_windows, num_variates))
    smape = np.zeros((num_series, num_windows, num_variates))
    mase = np.zeros((num_series, num_windows, num_variates))
    nd = np.zeros((num_series, num_windows, num_variates))

    # CRPS (Continuous Ranked Probability Score) - using weighted quantile loss
    crps = np.zeros((num_series, num_windows, num_variates))

    for s in range(num_series):
        for w in range(num_windows):
            for v in range(num_variates):
                q_preds = predictions_quantiles[s, w, :, v]  # (num_quantiles, pred_len)
                gt = ground_truth[s, w, v]  # (pred_len,)
                ctx = context[s, w, v]  # (max_ctx_len,)
                ctx = ctx[~np.isnan(ctx)]  # Remove NaN padding

                # Compute median (0.5 quantile) forecast - used for most metrics
                median_pred = q_preds[median_idx]  # (pred_len,)

                # Create valid mask to handle potential NaN padding in ground truth
                valid_mask = ~np.isnan(gt)
                if not np.any(valid_mask):
                    # No valid timesteps - set all metrics to NaN
                    mse[s, w, v] = mae[s, w, v] = rmse[s, w, v] = np.nan
                    mape[s, w, v] = smape[s, w, v] = mase[s, w, v] = np.nan
                    nd[s, w, v] = crps[s, w, v] = np.nan
                    continue

                # Filter to valid timesteps only
                gt = gt[valid_mask]
                median_pred = median_pred[valid_mask]
                q_preds = q_preds[:, valid_mask]  # (num_quantiles, valid_len)

                # Compute error using median forecast
                error = gt - median_pred
                abs_error = np.abs(error)

                # MSE (using median forecast, aligned with GluonTS MSE[0.5])
                mse[s, w, v] = np.mean(error ** 2)

                # MAE (using median forecast)
                mae[s, w, v] = np.mean(abs_error)

                # RMSE (sqrt of MSE)
                rmse[s, w, v] = np.sqrt(mse[s, w, v])

                # MAPE (using median forecast, returns fraction not percentage)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mape_vals = abs_error / np.abs(gt)
                    mape_vals = np.where(np.isfinite(mape_vals), mape_vals, np.nan)
                    mape[s, w, v] = np.nanmean(mape_vals)

                # sMAPE (using median forecast, range [0, 2])
                with np.errstate(divide="ignore", invalid="ignore"):
                    smape_vals = 2 * abs_error / (np.abs(gt) + np.abs(median_pred))
                    smape_vals = np.where(np.isfinite(smape_vals), smape_vals, np.nan)
                    smape[s, w, v] = np.nanmean(smape_vals)

                # MASE (Mean Absolute Scaled Error, using median forecast)
                if len(ctx) > seasonality:
                    naive_errors = np.abs(ctx[seasonality:] - ctx[:-seasonality])
                    seasonal_error = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
                    if seasonal_error > 0:
                        mase[s, w, v] = mae[s, w, v] / seasonal_error
                    else:
                        mase[s, w, v] = np.nan
                else:
                    mase[s, w, v] = np.nan

                # ND (Normalized Deviation, using median forecast)
                abs_label_sum = np.sum(np.abs(gt))
                if abs_label_sum > 0:
                    nd[s, w, v] = np.sum(abs_error) / abs_label_sum
                else:
                    nd[s, w, v] = np.nan

                # CRPS (MeanWeightedSumQuantileLoss) from provided quantiles
                if abs_label_sum > 0:
                    weighted_quantile_losses = []
                    for q, q_pred in zip(quantile_levels, q_preds):
                        q_error = gt - q_pred
                        indicator = (q_pred >= gt).astype(float)
                        q_loss = 2 * np.abs(q_error * (indicator - q))
                        weighted_ql = np.sum(q_loss) / abs_label_sum
                        weighted_quantile_losses.append(weighted_ql)
                    crps[s, w, v] = np.mean(weighted_quantile_losses)
                else:
                    crps[s, w, v] = np.nan

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "MASE": mase,
        "ND": nd,
        "CRPS": crps,
    }
