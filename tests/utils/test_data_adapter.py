import pytest

import numpy as np
import pandas as pd
import random
import torch

from tsicl.utils.data_adapter import (
    validate_target_inputs,
    validate_covar_inputs
)


SEQ_LEN=10


@pytest.mark.parametrize(
    "inputs, bs, expected_infos",
    [
        # 1D tensor inputs
        (torch.randn(10), 4, (1, True, 1)),

        # 1D np array inputs
        (np.random.rand(10), 4, (1, True, 1)),
        
        # 2D tensor inputs
        (torch.randn(18, 10), 4, (5, True, 1)),

        # 2D np array inputs
        (np.random.rand(18, 10), 4, (5, True, 1)),

        # 2D DataFrame inputs
        (pd.DataFrame(np.random.rand(18, 10)), 4, (5, True, 1)),

        # 3D tensor inputs, univariate
        (torch.randn(18, 10, 1), 4, (5, True, 1)),

        # 3D tensor inputs, multivariate
        (torch.randn(18, 10, 3), 4, (14, True, 3)),


        # list of regular 1D Tensors
        ([torch.randn(10) for _ in range(9)], 4, (3, True, 1)),

        # list of regular 1D np arrays
        ([np.random.rand(10) for _ in range(9)], 4, (3, True, 1)),

        # list of regular 1D DataFrame
        ([pd.DataFrame(np.random.rand(10)) for _ in range(9)], 4, (3, True, 1)),

        # list of regular 2D Tensors (seq_len, covars)
        ([torch.randn(10,2) for _ in range(9)], 4, (5, True, 2)),

        # list of regular 2D np arrays (seq_len, covars)
        ([np.random.rand(10,2) for _ in range(9)], 4, (5, True, 2)),

        # list of regular 2D DataFrame (seq_len, covars)
        ([pd.DataFrame(np.random.rand(10,2)) for _ in range(9)], 4, (5, True, 2)),


        # list of irregular 1D Tensors
        ([torch.randn(random.randint(8,12)) for _ in range(9)], 4, (9, False, 1)),

        # list of irregular 1D np arrays
        ([np.random.rand(random.randint(6,12)) for _ in range(9)], 4, (9, False, 1)),

        # list of irregular 1D DataFrame
        ([pd.DataFrame(np.random.rand(random.randint(6,12))) for _ in range(9)], 4, (9, False, 1)),

        # list of irregular 2D Tensors (seq_len, covars)
        ([torch.randn(random.randint(6,12),2) for _ in range(9)], 4, (9, False, 2)),

        # list of irregular 2D np arrays (seq_len, covars)
        ([np.random.rand(random.randint(6,12),2) for _ in range(9)], 4, (9, False, 2)),

        # list of irregular 2D DataFrame (seq_len, covars)
        ([pd.DataFrame(np.random.rand(random.randint(6,12),2)) for _ in range(9)], 4, (9, False, 2)),


        # list of dicts with 1D regular Tensor targets
        ([{'target': torch.randn(10)} for _ in range(10)], 4, (3, True, 1)),

        # list of dicts with 1D regular array targets
        ([{'target': np.random.rand(10)} for _ in range(10)], 4, (3, True, 1)),

        # list of dicts with 2D regular Tensor targets
        ([{'target': torch.randn(10,3)} for _ in range(10)], 4, (8, True, 3)),

        # list of dicts with 2D regular array targets
        ([{'target': np.random.rand(10,3)} for _ in range(10)], 4, (8, True, 3)),


        # list of dicts with 1D irregular Tensor targets
        ([{'target': torch.randn(random.randint(6,12))} for _ in range(10)], 4, (10, False, 1)),

        # list of dicts with 1D irregular array targets
        ([{'target': np.random.rand(random.randint(3,10))} for _ in range(10)], 4, (10, False, 1)),

        # list of dicts with 2D irregular Tensor targets
        ([{'target': torch.randn(random.randint(6,12), 2)} for _ in range(10)], 4, (10, False, 2)),

        # list of dicts with 2D irregular array targets
        ([{'target': np.random.rand(random.randint(3,10),4)} for _ in range(10)], 4, (10, False, 4)),
    ]
)
def test_validate_target_inputs_format_ok(
    inputs,
    bs,
    expected_infos
):
    
    num_batches, is_tensor, num_var, out = validate_target_inputs(inputs, batch_size=bs)

    # check all target tensors are 3D (bs, seq_len, 1)
    if isinstance(out, list):
        assert all([(x.ndim==3) and (x.shape[-1]==1) for x in out])
    else:
        assert out.ndim == 3 and out.shape[-1] == 1

    # check that all infos are correct
    assert (num_batches, is_tensor, num_var) == expected_infos


    return


@pytest.mark.parametrize(
   "inputs, bs",
   [
       # one sample contains nan only
       (torch.stack([torch.randn(10), torch.nan * torch.ones(10)], dim=0), 2),

       # Tensor  or array is 4D
       (torch.randn(8,1,10,1), 4),
       (np.random.randn(8,1,10,1), 4),

       # dict has no key 'target'
       ([{'tgt' : torch.randn(10)} for _ in range(3)], 4)
   ]
)
def test_invalid_target_inputs_format_raise_value_error(
    inputs,
    bs,
):
    with pytest.raises(ValueError):
        _ = validate_target_inputs(inputs, batch_size=bs)

    return


@pytest.mark.parametrize(
    "covars, is_tensor, tensor_shape, num_covars",
    [
        # no covar
        (None, True, None, 0),


        # 1D array-like (t)
        (torch.randn(SEQ_LEN), True, (1, 1, SEQ_LEN, 1), 1),
        (np.random.randn(SEQ_LEN), True, (1, 1, SEQ_LEN, 1), 1),
        ([torch.randn(SEQ_LEN) for _ in range(10)], True, (10, 1, SEQ_LEN, 1), 1),

        # 2D array-like (b t)
        (torch.randn(9, SEQ_LEN), True, (9, 1, SEQ_LEN, 1), 1),
        (np.random.randn(9, SEQ_LEN), True, (9, 1, SEQ_LEN, 1), 1),
        
        # list of 2D (t,c)
        ([np.random.randn(SEQ_LEN,3) for _ in range(5)], True, (5, 3, SEQ_LEN, 1), 3),
        
        # 3D array-like (b t c)
        (torch.randn(9, SEQ_LEN, 2), True, (9, 2, SEQ_LEN, 1), 2),
        (np.random.randn(9, SEQ_LEN, 3), True, (9, 3, SEQ_LEN, 1), 3),

        # 4D array-like (b c t 1)
        (torch.randn(9,2,SEQ_LEN,1), True, (9, 2, SEQ_LEN, 1), 2),
        (np.random.randn(9,2,SEQ_LEN,1), True, (9, 2, SEQ_LEN, 1), 2),


        # 1D array-like irregular len
        ([torch.randn(random.randint(6,10)) for _ in range(9)], False, None, 1),
        ([np.random.rand(random.randint(6,10)) for _ in range(9)], False, None, 1),

        # 2D array-like irregular len
        ([torch.randn(random.randint(6,10), 4) for _ in range(9)], False, None, 4),
        ([np.random.rand(random.randint(6,10), 4) for _ in range(9)], False, None, 4),
        ([pd.DataFrame(np.random.rand(random.randint(6,10), 4)) for _ in range(9)], False, None, 4),
        
        # 3D array-like irregular len
        # XXX assumes shape (c,t,1) --> why not (t,c,1)?
        ([torch.randn(4, random.randint(6,10), 1) for _ in range(9)], False, None, 4),
        ([np.random.rand(4, random.randint(6,10), 1) for _ in range(9)], False, None, 4),


        # past-only covariates
        ([{'past_covariates': {
            'covar_1': torch.randn(10), 'covar_2': torch.randn(10)
        }} for _ in range(3)], True, (3,2,10,1), 2),

        # irregular past-only covariates
        ([{'past_covariates': {
            'covar_1': torch.randn(random.randint(6,10))
        }} for _ in range(10)], False, None, 1),

        # past + future covariates
        ([{'past_covariates': {
            'covar_1': torch.randn(10), 'covar_2': torch.randn(10)
        }, 'future_covariates': {
            'covar_1': torch.randn(5), 'covar_2': torch.randn(5)
        }} for _ in range(3)], True, (3,2,15,1), 2),

        # past + future covariates future subset of past
        ([{'past_covariates': {
            'covar_1': torch.randn(10), 'covar_2': torch.randn(10)
        }, 'future_covariates': {
            'covar_2': torch.randn(5)
        }} for _ in range(3)], True, (3,1,15,1), 1),

        # past + future covariates future subset of past (numpy)
        ([{'past_covariates': {
            'covar_1': np.random.rand(10), 'covar_2': np.random.rand(10)
        }, 'future_covariates': {
            'covar_2': np.random.rand(5)
        }} for _ in range(3)], True, (3,1,15,1), 1),

    ]
)
def test_validate_covar_inputs_format_ok(
    covars,
    is_tensor,
    tensor_shape,
    num_covars
):
    
    is_cov_tensor, covars = validate_covar_inputs(covars)
        
    assert is_cov_tensor == is_tensor
    if covars is None:
        return
    
    if isinstance(covars, torch.Tensor):
    
        assert covars.shape == tensor_shape
    
    elif isinstance(covars, list):
    
        assert all([x.ndim == 4 for x in covars])
        assert all([x.shape[0] == 1 for x in covars]) # batch dim is always 1
        assert all([x.shape[1] == num_covars for x in covars])
        assert all([x.shape[3] == 1 for x in covars]) # last dim is always 1
    
    return


@pytest.mark.parametrize(
    "covars",
    [
        # not a 1D array-like
        ([{'past_covariates': {
            'covar_1': torch.randn(SEQ_LEN, 3)
        }} for _ in range(10)]),

        # unaligned covariates between themselves
        ([{'past_covariates': {
            'covar_1': torch.randn(SEQ_LEN), 'covar_2': torch.randn(2*SEQ_LEN)
        }} for _ in range(10)]),

        # raw DataFrame
        (pd.DataFrame(np.random.randn(9,SEQ_LEN)))

    ]
)
def test_invalid_covar_inputs_format_raise_value_error( covars ):
    with pytest.raises(ValueError):
        _ = validate_covar_inputs(covars)

    return