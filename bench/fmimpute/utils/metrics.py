from typing import Dict

import torch
from gluonts.ev.metrics import (
    MAE,
    MSE,
    NRMSE,
    RMSE,
    DirectMetric,
    MeanWeightedSumQuantileLoss,
)

# MAPE, MASE, MSIS, ND, SMAPE,

def initialize_gluonts_metrics(
    axis: int | None = 1
) -> Dict[str, DirectMetric]:
    """
    Initialize list of GluonTS metrics to compute at inference.
    """
    
    metrics = (
        MSE(forecast_type="mean"),
        MSE(forecast_type="0.5"),
        MAE(forecast_type="mean"),
        MAE(forecast_type="0.5"),
        RMSE(),
        NRMSE(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    )

    evaluators = {}
    for metric in metrics:
        evaluator = metric(axis=axis)
        evaluators[evaluator.name] = evaluator

    return evaluators

def update_gluonts_metrics(
    ytrue: torch.Tensor,
    yhat: torch.Tensor,
    evaluators: Dict[str, DirectMetric],
    is_target_mask: torch.Tensor | None = None
) -> Dict[str, DirectMetric]:
    """
    Update metrics with GluonTS submodule.

    Args:
        ytrue (torch.Tensor): ground truth of shape `(n, T, 1)`
        yhat (torch.Tensor): forecasts of shape `(n, T, q)`
        evaluators (Dict[str, DirectMetric]): dict of GluonTS metrics
        is_target_mask (torch.Tensor): boolean mask of shape `(n, T, 1)`, True for unobserved time indices
    
    Returns:
        Same `evaluators` with updated metrics
    """
    
    # target mask:
    mask = torch.ones(len(ytrue)).bool() if is_target_mask is None else is_target_mask
    mask.to(ytrue.device)

    # prepare groud truth and mean pred on target:
    label = ytrue[mask]
    mean  = yhat.mean(dim=-1, keepdim=True)[mask]

    # remove unobserved GT:
    nan_mask = ~label.isnan()
    label = label[nan_mask]
    mean  = mean[nan_mask]

    # instantiate batch dict:
    batch = {
        "label" : label.cpu().detach().numpy(),
        "mean"  : mean.cpu().detach().numpy(),
    }
    
    # get 10th, 20th, ... quantiles:
    if yhat.shape[2] > 1:
        inc  = int(0.1 * (yhat.shape[2] +1))
        yhat = yhat[...,inc-1::inc]
        assert yhat.shape[2] == 9

    # update batch dict:
    for q in range(9):
        idx = q * (yhat.shape[2]>1)
        yhat_q = yhat[...,idx:idx+1][mask]
        yhat_q = yhat_q[nan_mask]
        batch[str((q+1)/10)] = yhat_q.cpu().detach().numpy()

    # evaluate:
    for evaluator in evaluators.values():
        evaluator.update(batch)

    return evaluators
