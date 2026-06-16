import warnings
from typing import Sequence, Tuple

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_sample_imputation(
    quantiles: np.ndarray | torch.Tensor,  # (t q)
    y_ctx: np.ndarray | torch.Tensor, # (t)
    y_true: np.ndarray | torch.Tensor | None = None, # (t)
    quantiles_univar: np.ndarray | torch.Tensor | None = None,
    quantile_levels: list | None = None,
    show_context_points: bool = False,
    plot_iqr: bool = False,
    z_normalize: bool = False,
    model_name: str = 'TS-ICL',
    iqr_bands: Sequence = ((0.05, 0.95), (0.25, 0.75)),
    max_points: int = -1,
    is_blockwise: bool = False
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:

    """Imputation plot util for a single sample.

    Args:
        quantiles (np.ndarray | torch.Tensor): array of predicted quantiles, shape `(t, q)`
        y_ctx (np.ndarray | torch.Tensor): available context as a 1D-array of len `t` with `NaN`s
        y_true (np.ndarray | torch.Tensor | None): 1D-array of ground truth, len `t`
        quantiles_univar (np.ndarray | torch.Tensor | None):
            if `quantiles` is predicted with covariates
            quantiles_univar is the estimated same target variable but without covariates.
            array of predicted quantiles, shape `(t, q)`
        quantile_levels (list | None): list of the `q` quantile levels used in `quantiles`
        show_context_points (bool): whether to show available observations as markers
        plot_iqr (bool): whether to plot InterQuantile Ranges, if available
        z_normalize (bool): whether to z-normalize all data before plotting
        model_name (str): model name to use in legend
        
        iqr_bands (Sequence): iterable sequence of IQRs bands (low, high) to plot
        max_points (int): how many time points to plot (will plot all if <0)
        is_blockwise (bool): if True, assumes that missing points are contiguous and will shade the corresponding regions
    
    Returns:
        The corresponding `matplotlib` figure and axes.
    """

    # check input format:

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

    # prepare coordinates:
    total_grid_len = len(quantiles)
    coords         = np.arange(total_grid_len) # (t,)
    max_points     = min(max_points, total_grid_len) if max_points > 0 else total_grid_len

    # limit target to max_points samples for clarity:
    time_mask_t     = coords >= coords[-max_points]
    coords_t        = coords[time_mask_t]

    # set matplotlib params:
    matplotlib.rc('font', **{'size':12})
    matplotlib.rc('lines', **{'linewidth':2.5, 'linestyle': '-', 'markersize': 5})
    
    # get pointwise estimate (median if available):
    if 0.5 in quantile_levels:
        forecast_values = quantiles[time_mask_t][..., quantile_levels.index(0.5)] # (t, )
        if isinstance(quantiles_univar, np.ndarray):
            univar_values = quantiles_univar[time_mask_t][..., quantile_levels.index(0.5)] # (t, )
    else:
        forecast_values = quantiles[time_mask_t].mean(axis=-1)
        if isinstance(quantiles_univar, np.ndarray):
            univar_values = quantiles_univar[time_mask_t].mean(axis=-1)

    # get target mask (True if value is missing, False if observed and part of context):
    mask = np.isnan(y_ctx)[time_mask_t] # (t, 1)

    # get start and end indices of NA blocks:
    block_indices = np.where(np.diff(mask))[0] if is_blockwise else []

    # do z-norm if required:
    if z_normalize:
        mu    = np.nanmean(y_ctx, axis=0, keepdims=True)
        scale = np.nanstd(y_ctx, axis=0, keepdims=True)

        y_ctx = (y_ctx - mu) / (scale + 1e-6)

        if y_true is not None:
            y_true = (y_true - mu) / (scale + 1e-6)
        forecast_values  = (forecast_values - mu) / (scale + 1e-6)
        quantiles        = (quantiles - mu[..., None]) / (scale[..., None] + 1e-6)

        if quantiles_univar is not None:
            quantiles_univar  = (quantiles_univar - mu) / (scale + 1e-6)
        
        # if covar_all is not None:
        #     covar_norm = {}
        #     for k,x in covar_all.items():
        #         mu = np.nanmean(x, axis=1, keepdims=True)
        #         scale = np.nanstd(x, axis=1, keepdims=True)
        #         covar_norm[k] =(x - mu) / (scale + 1e-6)

    # init figure:
    fig, ax = plt.subplots(1, 1, figsize=(12,4))

    n_labels = 0

    # plot context points (context = not is missing in target):
    if show_context_points:
        coords_context = coords_t[~mask]
        values_context = y_ctx[time_mask_t][~mask]

        ax.plot(
            coords_context,
            values_context,
            'o',
            markersize=2.,
            color="tab:red",
            label='Context'
        )
        n_labels += 1
    
    # plot Ground Truth:
    y_plot = y_ctx if y_true is None else y_true
    ax.plot(
        coords_t,
        y_plot[time_mask_t],
        color="tab:green",
        lw=1.75,
        label='Ground Truth'
    )
    n_labels += 1

    # plot IQR:
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
                coords_t,
                y1    = quantiles[time_mask_t][...,quantile_levels.index(quantile_low)],
                y2    = quantiles[time_mask_t][...,quantile_levels.index(quantile_high)],
                color = 'tab:blue',
                label = 'IQR {:d}-{:d}'.format(int(100*quantile_low), int(100*quantile_high)),
                lw    = 0,
                alpha = alpha
            )
            n_labels += 1
    
    # make interpo plot:

    # shaded block regions:
    for i,idx in enumerate(block_indices[::2]):
        plt.axvline(float(coords_t[idx]),ls='--',color='k',lw=.5)
        end_point = coords_t[block_indices[1::2][i]] if i < len(block_indices[1::2]) else coords_t[-1]
        plt.axvline(float(end_point),ls='--',color='k',lw=0.5)
        plt.axvspan(float(coords_t[idx]), float(end_point), facecolor='tab:gray',alpha=0.1)

    # plot forecast as a line:
    if quantiles_univar is not None:
        ax.plot(
            coords_t,
            univar_values,
            color = 'tab:blue',
            lw    = 1.25,
            ls    = ':',
            label = model_name + '(univar.)'
        )
        n_labels += 1

    # plot point forecast at all timesteps (as a line):
    ax.plot(
        coords_t,
        forecast_values,
        color = "tab:blue",
        lw    = 1.5,
        label = model_name
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