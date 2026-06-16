from typing import Dict

import torch

from .utils import complete_nans


def prepare_context_tensors(
    grid: torch.Tensor,
    series_c: torch.Tensor,
    covariates: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Prepare context tensors for forward pass.

    Parameters
    ----------
    grid : torch.Tensor
        Raw time grid of shape `(bs, T, 1)`.

    series_c : torch.Tensor
        Tensor of past values of shape `(bs, T, 1)`.
    
    covariates : torch.Tensor, optional
        Tensor of covariates of shape `(bs, c, T, 1)`.
    
    Returns
    -------
    out : Dict[str, torch.Tensor]
        Dict with keys `coords_c` (context coordinates), 
        `series_c` (context values),
        `covar_c` (context covariates, if any).
    """
    
    assert series_c.ndim == 3
    assert series_c.shape[-1] == 1
    assert grid.shape[1] == series_c.shape[1]
    
    has_covar = covariates is not None

    # get the total number of missing points in each sample:
    nb_missing_points = torch.isnan(series_c).sum(1).squeeze() # (bs)
    if nb_missing_points.ndim == 0:
        nb_missing_points = nb_missing_points.unsqueeze(0)
    
    # compute the maximum context length:
    max_context_len   = int( series_c.shape[1] - min(nb_missing_points) )
    
    # prepare context tensors; handle each sample separately:
    list_coords_c, list_series_c, list_covar_c = [], [], []
    for idx in range(len(series_c)):
        
        # get sample:
        sample = series_c[idx].squeeze()

        # get missing mask:
        is_missing_mask = torch.isnan(sample)   # (T,)

        # handle series that have too many nans (= more than the min nb of nan in the batch)
        if is_missing_mask.sum() > min(nb_missing_points):

            # get number of values actually observed:
            cur_len = len(sample[~is_missing_mask])
            mis_len = max_context_len - cur_len

            # (i) pad context values with nans, up to length=max_context_len:
            # (nan values will be discared when we complete_nans)
            series_i = torch.cat([
                sample[~is_missing_mask],
                torch.nan * torch.ones(mis_len).to(sample.device)
            ]).unsqueeze(-1) # (max_context_len, 1)

            # (ii) same padding (with ones) for the grid:
            grid_i = torch.cat([
                grid[idx][~is_missing_mask].squeeze(),
                torch.ones(mis_len).to(sample.device) # (T, 1), (T, 1)
            ]).unsqueeze(-1) # (max_context_len, 1)

            # (iii) handle covariates (context part):
            if has_covar:

                # check time alignment:
                assert covariates.shape[-2] == series_c.shape[-2]
                
                # we keep only the covariates observed when the target series is observed
                # + then we must pad with nans:
                nb_covars = covariates.shape[1] # (bs C t 1)
                covar_c = torch.cat([
                    covariates[idx][...,~is_missing_mask,:], # (c t_obs 1)
                    torch.nan * torch.ones((nb_covars, mis_len, 1)).to(sample.device)
                ], dim = - 2) # (C, max_context_len, 1)

                # now, each covar may also contain NaN (and they may not be aligned between themselves)

            # (iv) complete nans:
            out = complete_nans(series_i, grid_i, is_test=True, X_cov=covar_c if has_covar else None)
            series_i, grid_i, covar_c = out['values'], out['coords'], out.get('covar')

        else:
            series_i, grid_i = series_c[idx][~is_missing_mask], grid[idx][~is_missing_mask] # (max_context_len, 1), (max_context_len, 1)
            if has_covar:
                covar_c = covariates[idx][...,~is_missing_mask,:] # (C, max_context_len, 1)
                assert covar_c.shape[-2] == series_i.shape[-2]
        
        # check that there is no nan in the context values:
        assert (not torch.isnan(series_i).any()) or torch.isnan(series_i).all()

        list_coords_c.append(grid_i.reshape(-1,1))   # (max_context_len, 1)
        list_series_c.append(series_i.reshape(-1,1)) # (max_context_len, 1)
        if covariates is not None:
            list_covar_c.append(covar_c)             # (C, max_context_len, 1)

    coords_c = torch.stack(list_coords_c) # (bs, max_context_len, 1)
    series_c = torch.stack(list_series_c) # (bs, max_context_len, 1)


    if has_covar:
        covariates_c = torch.stack(list_covar_c)    # (bs, C, max_context_len, 1)

    out = {
        'series_c' : series_c,
        'coords_c' : coords_c,
    }

    if has_covar:
        out['covar_c'] = covariates_c # (bs, C, max_context_len, 1)

    return out

