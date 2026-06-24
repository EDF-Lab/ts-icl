import warnings
from typing import Sequence, Tuple

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_sample_forecast(
    quantiles: np.ndarray | torch.Tensor,  # (t q)
    y_ctx: np.ndarray | torch.Tensor, # (t_ctx)
    y_true: np.ndarray | torch.Tensor | None = None, # (t)
    quantiles_univar: np.ndarray | torch.Tensor | None = None,
    quantile_levels: list | None = None,
    plot_iqr: bool = False,
    z_normalize: bool = False,
    model_name: str = 'TS-ICL',
    iqr_bands: Sequence = ((0.05, 0.95), (0.25, 0.75)),
    max_points: int = -1,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Forecasting plot util for a single sample.

    Args:
        quantiles (np.ndarray | torch.Tensor): array of predicted quantiles, shape `(H, q)`
        y_ctx (np.ndarray | torch.Tensor): available lookback as a 1D-array of len `L` (possibly with `NaN`s)
        y_true (np.ndarray | torch.Tensor | None): 1D-array of ground truth, len `H`
        quantiles_univar (np.ndarray | torch.Tensor | None):
            array of predicted quantiles when not using the covariates, shape `(H, q)`
        quantile_levels (list | None): list of the `q` quantile levels used in `quantiles`
        plot_iqr (bool): whether to plot InterQuantile Ranges, if available
        z_normalize (bool): whether to z-normalize all data before plotting
            (with stats computed from the lookback)
        model_name (str): model name to use in legend
        iqr_bands (Sequence): iterable sequence of IQRs bands (low, high) to plot
        max_points (int): how many time points to plot (will plot all if <0)
    
    Returns:
        The corresponding `matplotlib` figure and axes.
    """

    assert quantiles.ndim == 2

    if isinstance(quantiles, torch.Tensor):
        quantiles = quantiles.cpu().numpy()
    if isinstance(y_ctx, torch.Tensor):
        y_ctx = y_ctx.cpu().numpy()

    if y_ctx.ndim == 2:
        assert y_ctx.shape[-1] == 1
        y_ctx = y_ctx[:,0]
    
    if isinstance(quantiles_univar, torch.Tensor):
        quantiles_univar = quantiles_univar.cpu().numpy()

    if y_true is not None:
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if y_true.ndim == 2:
            assert y_true.shape[-1] == 1
            y_true = y_true[:,0]

    context_len = len(y_ctx)
    target_len  = len(quantiles)

    coords     = np.arange(context_len + target_len) # (L+H,)
    max_points = min(max_points, len(coords))

    # limit target to max_points samples for clarity:
    time_mask_t     = coords >= coords[-max_points]
    coords_t        = coords[time_mask_t]
    coords_f        = coords_t[-target_len:]

    # get pointwise estimate (median if available):
    if 0.5 in quantile_levels:
        forecast_values = quantiles[..., quantile_levels.index(0.5)] # (t, )
        if isinstance(quantiles_univar, np.ndarray):
            univar_values = quantiles_univar[..., quantile_levels.index(0.5)] # (t, )
    else:
        forecast_values = quantiles[time_mask_t].mean(axis=-1)
        if isinstance(quantiles_univar, np.ndarray):
            univar_values = quantiles_univar[time_mask_t].mean(axis=-1)
        
    if z_normalize:
        mu = np.nanmean(y_ctx, axis=0, keepdims=True)
        scale = np.nanstd(y_ctx, axis=0, keepdims=True)

        y_ctx  = (y_ctx - mu) / (scale + 1e-6)
        if y_true is not None:
            y_true = (y_true - mu) / (scale + 1e-6)
        forecast_values  = (forecast_values - mu) / (scale + 1e-6)
        quantiles        = (quantiles - mu[..., None]) / (scale[..., None] + 1e-6)

        if quantiles_univar is not None:
            univar_values    = (univar_values - mu) / (scale + 1e-6)
            quantiles_univar = (quantiles_univar - mu[..., None]) / (scale[..., None] + 1e-6)

    if y_true is None:
        y_true = np.nan * np.ones(target_len)
    gt = np.concatenate([y_ctx, y_true], axis=0) # T, 

    # set matplotlib params:
    matplotlib.rc('font', **{'size':12})
    matplotlib.rc('lines', **{'linewidth':2.5, 'linestyle': '-', 'markersize': 5})
    
    n_labels = 0

    # init figure:
    fig, ax = plt.subplots(1, 1, figsize=(12,4))

    # plot ground truth on lookback + horizon:
    ax.plot(
        coords_t,
        gt[time_mask_t],
        color="tab:green",
        lw=1.75,
        label='Ground Truth'
    )
    n_labels += 1

    # render forecast region as light gray:
    ax.axvline(coords_f[0], ls='--', color='k', lw=0.75, alpha=0.8)
    ax.axvspan(coords_f[0], ax.get_xlim()[1], facecolor='tab:gray', alpha=0.1)

    # plot IQR bands:
    if plot_iqr:
        alphas = np.linspace(0.3, 0.7, len(iqr_bands))
        for band, alpha in zip(sorted(iqr_bands), alphas):
            quantile_low, quantile_high = band
            if quantile_low not in quantile_levels:
                warnings.warn(f"Requested quantile {quantile_low} not in `quantile_levels`, skip IQR plot")
                continue
            if quantile_high not in quantile_levels:
                warnings.warn(f"Requested quantile {quantile_high} not in `quantile_levels`, skip IQR plot")
                continue

            ax.fill_between(
                coords_f,
                y1    = quantiles[...,quantile_levels.index(quantile_low)],
                y2    = quantiles[...,quantile_levels.index(quantile_high)],
                color = 'tab:blue',
                    label = 'IQR {:d}-{:d}'.format(int(100*quantile_low), int(100*quantile_high)),
                    lw    = 0,
                    alpha = alpha
            )
            n_labels += 1

    # plot univariate forecast:
    if quantiles_univar is not None:
        ax.plot(
            coords_f,
            univar_values,
            color = 'tab:blue',
            lw    = 1.25,
            ls    = ':',
            label = model_name + '(univar.)'
        )
        n_labels += 1

    # plot estimated median:
    ax.plot(
        coords_f,
        forecast_values, #quantiles[...,quantile_levels.index(0.5)],
        color = 'tab:blue',
        lw    = 1.5,
        label = model_name # + ('(covar)' if plot_covar else '')
    )
    n_labels += 1

    ax.legend(
        loc            = 'upper center', 
        bbox_to_anchor = (0.5, -0.075),
        fancybox       = True, 
        shadow         = True, 
        ncol           = n_labels
    )

    fig.tight_layout()
    
    return fig, ax