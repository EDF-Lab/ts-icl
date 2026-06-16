from typing import Tuple
from warnings import warn

import numpy as np
import pandas as pd
import torch
from einops import rearrange


def validate_target_inputs(
    inputs: list | torch.Tensor | np.ndarray | pd.DataFrame,
    batch_size: int
) -> Tuple[int, bool, int, list | torch.Tensor]:
    """Check dims and standardize all target tensor shapes.

    Ignore covariates.

    Returns
    -------
    num_batches : int
        The total number of mini batches to process.
    
    is_tensor : bool
        Whether the returned inputs are torch Tensor.
    
    num_var : int
        Number of target variables now hidden in batch dimension.
    
    out : torch.Tensor or list
        Preprocessed inputs as Tensors of shape `(batch, seq_len, 1)`.
        If multiple variables to predict, all are stacked in the batch dimension.
    """
    
    # convert list of tensors to stacked tensor, whenever possible:
    if isinstance(inputs, list):

        # for fevbench
        if isinstance(inputs[0], dict):
            # past_covars, future_covars = None, None
            # self._future_cov_keys = []
            if 'target' not in inputs[0].keys():
                raise ValueError(
                    f"Dict input should have a 'target' key, received keys {list(inputs[0].keys())} instead"
                )
            out, _ = _list_dict_inputs_utils(inputs, key = 'target', _future_cov_keys = []) # Tensor or list[Tensor]
            
        elif isinstance(inputs[0], np.ndarray):
            # convert to list of Tensors:
            out = [torch.Tensor(x) for x in inputs]
        
        elif isinstance(inputs[0], pd.DataFrame):
            # convert to list of DataFrames:
            out = [torch.Tensor(x.values) for x in inputs]
        
        else:
            out = inputs
            if not isinstance(inputs[0], torch.Tensor):
                raise ValueError(
                    f"inputs is a list of unsupported type {type(inputs[0])}, should be one of `torch.Tensor`, `np.ndarray`, `pd.DataFrame`, or `dict`"
                )

        # for time
        if isinstance(out, list) and isinstance(out[0], torch.Tensor):
            if all([val.shape == out[0].shape for val in out]):
                out = torch.stack(out, dim=0) # (bs, t, c)

    elif isinstance(inputs, np.ndarray):
        # convert to Tensor:
        out = torch.Tensor(inputs)
    
    elif isinstance(inputs, pd.DataFrame):
        # convert to Tensor
        out = torch.Tensor(inputs.values).unsqueeze(-1) # (b, t, 1)

    else:
        if not isinstance(inputs, torch.Tensor):
            raise ValueError(
                f"input type {type(inputs)} not supported, should be one of `torch.Tensor`, `np.ndarray`, `pd.DataFrame` or a `list`"
            )
        out = inputs

    # case 1: batched tensor:
    is_tensor = True
    if isinstance(out, torch.Tensor):
        # make sure the Tensor has shape (b, t, 1):
        # (b possibly includes the channel dim)
        out, num_var = _check_input_dim(out, has_batch_dim = True)      # (bs, t, 1)
        num_batches  = len(out) // batch_size + (len(out) % batch_size > 0)

    # case 2: list of 1D tensors (irregular lengths):
    elif isinstance(out, list):
        out = [_check_input_dim(x, has_batch_dim = False) for x in out] # [(c, t, 1), (c, t, 1), ...]
        num_var = out[0][1]
        out     = [x[0] for x in out]

        num_batches = len(out)
        is_tensor   = False

    else:
        raise NotImplementedError
            
    return num_batches, is_tensor, num_var, out


def _check_input_dim(
    x: torch.Tensor,
    has_batch_dim: bool = True
) -> Tuple[torch.Tensor, int]:
    """Convert arbitrary tensors of C channels and T timesteps into 3D univariate tensors.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to check
    has_batch_dim : bool
        Whether `x` has a first batch dimension
    
    Returns
    -------
    x : torch.Tensor
        3D Tensor of shape `(bs x C, T, 1)` (variable dim in the batch dim)
    
    num_var : int
        number of channels (now hidden in batch dim)

    Raises
    ------
    ValueError
        If a one of the samples has no observation (full of NaNs)
    """
    
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(-1) # (1, t, 1)
    elif x.ndim == 2:
        dim = -1 if has_batch_dim else 0
        x = x.unsqueeze(dim) # (b, t, 1)

    if x.ndim != 3:
        raise ValueError("Expected 3D tensor")

    # channel independence --> channel in batch dim:
    num_var = x.shape[-1]
    if num_var > 1:
        x = rearrange(x, 'b t c -> (b c) t 1')
    
    # raise error if a sample contains only NaNs:
    if ((~x.isnan()).sum(1) == 0).sum(0) > 0:
        raise ValueError(
            f"Input tensor of shape {x.shape} contains at least one sample full of NaNs, stop here"
        )
    
    return x, num_var


def _list_dict_inputs_utils(
    list_inputs: list[dict],
    key: str,
    _future_cov_keys: list
) -> Tuple[torch.Tensor | list | None, list]:

    first_input = list_inputs[0]
    assert isinstance(first_input, dict)
    assert isinstance(first_input[key], dict) or isinstance(first_input[key], torch.Tensor) or isinstance(first_input[key], np.ndarray)
    assert all([key in d for d in list_inputs])

    # handle 'target':
    if isinstance(first_input[key], torch.Tensor) or isinstance(first_input[key], np.ndarray):
        if all([d[key].shape == list_inputs[0][key].shape for d in list_inputs]):
            out = torch.stack([torch.Tensor(d[key]) for d in list_inputs]) # (bs, ...)
        else:
            out = [torch.Tensor(d[key]) for d in list_inputs]

    # handle 'past_covariates' and 'future_covariates'
    else:
        if not isinstance(first_input[key], dict):
            raise ValueError("Expected a Dict of Dict")
        
        list_keys = list(first_input[key].keys())
        
        if len(list_keys) == 0:
            return None, _future_cov_keys
        
        if len(_future_cov_keys) > 0:
            list_keys = _future_cov_keys.copy()
        if key == 'future_covariates':
            _future_cov_keys = list_keys
        
        if not all([all([k in d[key] for k in list_keys]) for d in list_inputs]):
            raise ValueError(
                f"Expected all samples in `covars` dicts to have identical covariate keys."
            )
        
        if not all([torch.Tensor(d[key][k]).squeeze().ndim == 1 for d in list_inputs for k in list_keys]):
            raise ValueError(
                "Expected covariates in a dict to be 1D Tensors."
            )
        
        # for a given sample, concatenate all covariates in a single tensor;
        try:
            out = [
                torch.stack(
                    [
                        torch.Tensor(d[key][k]).squeeze() for k in list_keys
                    ],
                    dim = 0 # (c t 1)
                ).unsqueeze(-1) for d in list_inputs
            ]
        except Exception:
            raise ValueError(
                "For a given sample, all covariates should be aligned and of the same length, \
                    not the case here: shapes. Check `covars` arg."
            )
        
        # if all samples have the same covariate length, concat along time dimension:
        if all([x.shape == out[0].shape for x in out]):
            out = torch.stack(out, dim = 0) # (bs, c, t, 1)

    return out, _future_cov_keys


def validate_covar_inputs(
    covars: list | torch.Tensor | np.ndarray | None
) -> Tuple[bool, list | torch.Tensor | None]:
    """Check dims and standardize all covariate tensor shapes.

    Parameters
    ----------
    covars : list | torch.Tensor | np.ndarray | None
        Covariates of shape `(T, C)` or `(C, T, 1)`
        (possibly with an extra batch dimension)

    Returns
    -------
    is_cov_tensor : bool
        True if returned covariate is a batch Tensor.
    covar : torch.Tensor or List[torch.Tensor] or None
        Tensor of covariates of shape `(bs, C, T, 1)` if all aligned,
        other a list of covariates of shape `(C, T, 1)`.
    """
    if covars is None:
        return True, None
    
    if isinstance(covars, np.ndarray):
        covars = torch.Tensor(covars)

    if isinstance(covars, list):

        if isinstance(covars[0], np.ndarray):
            covars = [torch.Tensor(x) for x in covars]
        
        elif isinstance(covars[0], pd.DataFrame):
            covars = [torch.Tensor(x.values) for x in covars]

        elif isinstance(covars[0], dict):

            _future_cov_keys = []

            if 'past_covariates' not in covars[0]:
                warn("no past covariates provided, skip all covariates for forecast")
                return True, None
            else:
            
                # convert covariates as Tensor or list[Tensor]:
                future_covars = None
                if 'future_covariates' in covars[0]:
                    future_covars, _future_cov_keys = _list_dict_inputs_utils(covars, key = 'future_covariates', _future_cov_keys = _future_cov_keys)

                past_covars, _future_cov_keys = _list_dict_inputs_utils(covars, key = 'past_covariates', _future_cov_keys = _future_cov_keys)
                # is_tensor = isinstance(past_covars, torch.Tensor)
                
                if isinstance(past_covars, torch.Tensor) and isinstance(future_covars, torch.Tensor):
                    if past_covars.shape[1] != future_covars.shape[1]:
                        raise ValueError(
                            f"Past and future covariates do not match, shapes {past_covars.shape} vs {future_covars.shape}\
                                Current version assumes the same set of covariates are observed in past-only and future.\
                                "
                        )
                    covars = torch.cat(
                        [past_covars, future_covars], dim = -2
                    ) # (b, c, t, 1)
                
                elif isinstance(past_covars, list) and isinstance(future_covars, list):
                    covars = [
                        torch.cat([x_c, x_t], dim=-2) for x_c, x_t in zip(past_covars, future_covars)
                    ] # (c, t, 1)

                else:
                    covars = past_covars

        else:
            if not isinstance(covars[0], torch.Tensor):
                raise RuntimeError(
                    f"Was given a covariate of unsupported type {type(covars[0])}"
                )

        if isinstance(covars, list) and all([val.shape == covars[0].shape for val in covars]):
            covars = torch.stack(covars, dim=0) # (bs, T, c) or (bs, c, T, 1)

    # case 1: batched tensor:
    is_cov_tensor = True
    if isinstance(covars, torch.Tensor):
        if covars.ndim == 1:
            covars = rearrange(covars, 't -> 1 1 t 1') # (bs, c, t, 1)
        elif covars.ndim == 2:
            covars = rearrange(covars, 'b t -> b 1 t 1') # (bs, c, t, 1)
        elif covars.ndim == 3:
            covars = rearrange(covars, 'b t c -> b c t 1') # (bs, c, t, 1)
        elif covars.ndim == 4:
            assert covars.shape[-1] == 1 # (bs, c, t, 1)

    # case 2: list of 1D tensors (irregular lengths):
    elif isinstance(covars, list):
        if covars[0].ndim == 1:
            covars = [rearrange(x, 't -> 1 1 t 1') for x in covars] # [(1, c, t, 1), (1, c, t, 1), ...]
        elif covars[0].ndim == 2:
            covars = [rearrange(x, 't c -> 1 c t 1') for x in covars] # [(1, c, t, 1), (1, c, t, 1), ...]
        elif covars[0].ndim == 3:
            assert covars[0].shape[-1] == 1
            covars = [rearrange(x, 'c t 1 -> 1 c t 1') for x in covars] # [(1, c, t, 1), (1, c, t, 1), ...]
        is_cov_tensor = False
        # assert isinstance(inputs, list)
        # assert len(covars) == len(inputs)

    else:
        raise ValueError(
            f"Unsupported type {type(covars)}"
        )
    
    return is_cov_tensor, covars

