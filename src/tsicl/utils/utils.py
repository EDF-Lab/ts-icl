from typing import Dict, Generator, List

import torch


def batchify(
    x: torch.Tensor | List[torch.Tensor] | None,
    batch_size: int,
    device: torch.device | None = None,
    batch_len: int = 0
) -> Generator[torch.Tensor | None, None, None]:
    """Convert tensor or list into batches of desired size + move to device.
    
    Parameters
    ----------
    x : torch.Tensor | List[torch.Tensor] | None
        Tensor with first dimension as batch dim, or list of tensors.
    
    batch_size : int
        Batch size.
    
    device : torch.device, optinal
        Where to move the batch data to.
    
    batch_len : int
        If `x` is None, yield None `batch_len` times.

    Returns
    -------

    out : Generator
        Iterator over batches.
    """

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if isinstance(x, list):
        batch_size = 1
        for ii in range(len(x)):
            yield x[ii].to(device)
        
    elif isinstance(x, torch.Tensor):

        for ii in range(0, len(x), batch_size):
            yield x[ii : ii + batch_size].to(device)

    elif x is None:
        assert batch_len > 0
        for ii in range(batch_len):
            yield None

    else:
        raise NotImplementedError(
            f"`x` should be a `Tensor` or a list of `Tensor` or `None`, received {type(x)} instead"
        )


def make_grid(
    grid_length: int,
    num_samples: int,
    end_point: float = 1.0,
    start_point: float = 0.0
) -> torch.Tensor:
    """Create a grid of time coords in (0, 1)

    Args:
        grid_length (int): number of timesteps
        num_samples (int): number of samples
        end_point (float): end point of the grid
        start_point (float): start point of the grid
    
    Returns:
        Grid as torch.Tensor of shape `(num_samples, grid_length, 1)`.
    """
    
    grid = torch.linspace(start_point, end_point, grid_length).float().unsqueeze(0).repeat_interleave(repeats=num_samples, dim=0)
    grid = grid.unsqueeze(-1) # [N, T, 1]

    return grid


def complete_nans(
    X: torch.Tensor,
    grid: torch.Tensor,
    is_test: bool = False,
    X_cov: torch.Tensor | None = None
) -> Dict[str, torch.Tensor]:
    """
    Fills missing values (NaNs) in a target tensor
    (replaced by observed values at random instants of the same series).

    Args:
        X (torch.Tensor): value tensor of shape `(bs, seq_len, 1)`.
        grid (torch.Tensor): grid coord tensor of shape `(bs, seq_len, 1)`.
        is_test (bool): True if test mode.
        X_cov (torch.Tensor | None): optional tensor of covariates of shape `(bs, c, seq_len, 1)`.
            Will not remove all NaNs from `X_cov`, only make sure it remains aligned with `X`.
    
    Returns:
        Dict storing `'values'` and `'coords'` tensors of shape `(bs, seq_len, 1)`
        with duplicates instead of NaNs.
        Also contains `'covar'` tensor if `X_cov` was supplied.
    """

    has_nans = torch.isnan(X).any().item()
    
    if not has_nans:

        values_with_replacement = X.clone()
        grids_with_replacement  = grid.clone()
        cov_with_replacement    = X_cov

    else:
        
        if is_test:
            torch.manual_seed(42)

        # prepare tensors:
        has_batch_dim = X.ndim == 3
        X    = X.clone() if has_batch_dim else X.clone().unsqueeze(0)
        grid = grid.clone() if has_batch_dim else grid.clone().unsqueeze(0)

        mask_observed = ~torch.isnan(X[..., 0]) # [N, T]
        X_filled      = X[..., 0].clone()       # [N, T]
        partial_grid  = grid[..., 0].clone()    # [N, T]

        if X_cov is not None:
            X_cov = X_cov.clone() if has_batch_dim else X_cov.clone().unsqueeze(0)
            X_cov_filled = X_cov[..., 0].clone()

        # loop through samples:
        for i in range(X_filled.shape[0]):

            # time indices with no NaNs in raw data:
            valid_indices = torch.where(mask_observed[i])[0]

            # time indices corresponding to NaNs in raw data:
            invalid_indices = torch.where(~mask_observed[i])[0]
            
            # replace NaNs by any other observed value:
            if valid_indices.numel() > 0 and invalid_indices.numel() > 0:  
                replacements = valid_indices[torch.randint(0, valid_indices.numel(), (invalid_indices.numel(),))]
                X_filled[i, invalid_indices]     = X[i, replacements, 0]
                partial_grid[i, invalid_indices] = grid[i, replacements, 0]
                if X_cov is not None:
                    X_cov_filled[i, :, invalid_indices] = X_cov[i, :, replacements, 0]
        
        values_with_replacement = X_filled.unsqueeze(-1)      # [N, T, 1] (with duplicates)
        grids_with_replacement  = partial_grid.unsqueeze(-1)  # [N, T, 1] (with duplicates)
        if X_cov is not None:
            cov_with_replacement = X_cov_filled.unsqueeze(-1) # [N, T, 1] (with duplicates)

        if (not has_batch_dim):
            values_with_replacement = values_with_replacement.squeeze(0)
            grids_with_replacement  = grids_with_replacement.squeeze(0)
            if X_cov is not None:
                cov_with_replacement = cov_with_replacement.squeeze(0)

    out = {
        'values' : values_with_replacement,
        'coords' : grids_with_replacement,
    }
    if X_cov is not None:
        assert isinstance(cov_with_replacement, torch.Tensor)
        out['covar'] = cov_with_replacement
        
    return out