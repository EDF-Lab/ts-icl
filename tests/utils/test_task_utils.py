from typing import Tuple

import pytest
import random
import torch

from tsicl.utils.task_utils import prepare_context_tensors


def add_nans(
    x: torch.Tensor,
    max_nans: int = 2,
    min_nans: int = 2,
    x_cov : torch.Tensor | None = None,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor | None]:
    # x (b t 1)
    # x_cov (b c t 1)

    ndims = x.ndim

    # iterate through the samples:
    for ii in range(len(x)):
        nb_nans = min_nans if ii == 0 else random.randint(min_nans, max_nans)
        idx = torch.randperm(x.shape[-2])[:nb_nans]
        if ndims == 3:
            x[ii,idx] = torch.nan
        elif ndims == 4:
            x[ii,:,idx] = torch.nan
        else:
            raise RuntimeError

        # apply nan at same time index to all covar channels:
        if x_cov is not None:
            x_cov[ii,:,idx] = torch.nan

    if x_cov is None:
        return x
    else:
        return x, x_cov


@pytest.mark.parametrize(
    "grid, series_c, covar, expected_output_shapes, covar_has_nans",
    [
        # batch univariate series no nan --> shape unchanged, no nan
        (torch.randn(4, 10, 1), torch.randn(4,10,1), None, (4,10, 1), False),

        # batch univariate same number of nans --> smaller seq_len, no nan
        (torch.randn(4, 10, 1), add_nans(torch.randn(4,10,1), 2, 2), None, (4,8,1), False),

        # batch univariate varying number of nans --> seq_len corresp to min number of nans, no nan
        (torch.randn(8, 10, 1), add_nans(torch.randn(8,10,1), 3, 1), None, (8,9,1), False),

        # batch with covariates no nan --> shape unchanged, no nan
        (torch.randn(4, 10, 1), torch.randn(4,10,1), torch.randn(4, 3, 10, 1), (4,10, 1), False),

        # batch with covariates nan only in target --> smaller seq len, no nan
        (torch.randn(4, 10, 1), add_nans(torch.randn(4,10,1), 3, 1), torch.randn(4, 3, 10, 1), (4,9,1), False),

        # batch with covariates nan only in covariate --> same seq_len, nan in covar
        (torch.randn(8, 10, 1), torch.randn(8,10,1), add_nans(torch.randn(8, 3, 10, 1),4,2), (8,10,1), True),

        # batch with covariates with same nan pattern to all variates --> smaller seq_len, no nan
        (torch.randn(4, 10, 1), add_nans(torch.randn(4,10,1), 2, 2, torch.randn(4, 3, 10, 1)), None, (4,8,1), False),

        # batch with constant number of nans but not aligned between target vs covar --> smaller seq_len, nan in covar
        (torch.randn(8, 10, 1), add_nans(torch.randn(8,10,1), 2, 2), add_nans(torch.randn(8, 3, 10, 1),4,2), (8,8,1), True)

    ]
)
def test_context_tensors_no_nans(
    grid,
    series_c,
    covar,
    expected_output_shapes,
    covar_has_nans
):

    if isinstance(series_c, tuple):
        series_c, covar = series_c

    out = prepare_context_tensors(grid, series_c, covar)
    # check output is a dict of tensors:
    assert isinstance(out, dict)
    assert isinstance(out.get('series_c'), torch.Tensor)
    assert isinstance(out.get('coords_c'), torch.Tensor)
    # check no nan in outputs:
    assert not out['series_c'].isnan().any()
    assert not out['coords_c'].isnan().any()
    # check shape ok:
    assert out['series_c'].shape == expected_output_shapes
    assert out['coords_c'].shape == expected_output_shapes

    if covar is not None:
        assert isinstance(out.get('covar_c'), torch.Tensor)
        assert out['covar_c'].ndim == 4
        # check covar dim ok:
        assert out['covar_c'].shape[1] == covar.shape[1]
        # check seq dims ok:
        assert out['covar_c'].shape[2:] == expected_output_shapes[1:]
        # (covariate may contain nan after testing)
        assert out['covar_c'].isnan().any() == covar_has_nans
    
