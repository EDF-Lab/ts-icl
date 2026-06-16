from typing import List, Tuple
import pytest

import random
import torch

from tsicl import TSICL

DUMMY_CKPT = 'tests/tsicl-dummy.ckpt'
BATCH_SIZE = 4
NUM_SAMPLES = 10
NUM_COVARS = 3
DEFAULT_Q_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GPU_DEVICE = torch.device('cuda')
CPU_DEVICE = torch.device('cpu')



def load_model() -> TSICL:
    return TSICL(model_path = DUMMY_CKPT)

def sparse_irregular_input_list(
    num_variates: int,
    horizon_len: int = 0
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    def add_nans(x:torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            nb_nans = random.randint(1, len(x) - 2)
            idx = torch.randperm(len(x))[:nb_nans]
            x[idx] = torch.nan
        else:
            assert x.ndim == 2
            for jj in range(num_variates):
                nb_nans = random.randint(1, len(x) - 2)
                idx = torch.randperm(len(x))[:nb_nans]
                x[idx,jj] = torch.nan
                
        return x

    if num_variates == 1:
        inputs = [add_nans(torch.randn(random.randint(6,12))) for _ in range(NUM_SAMPLES)]
    else:
        inputs = [add_nans(torch.randn(random.randint(6,12), num_variates)) for _ in range(NUM_SAMPLES)]
    covars = [add_nans(torch.randn(len(x)+horizon_len, NUM_COVARS)) for x in inputs]

    return inputs, covars

# load dummy model
def test_instantiate_pipeline():
    assert isinstance(load_model(), TSICL)


# check .impute
@pytest.mark.parametrize(
    "use_covariate, auto_complete, replace_by_gt, squeeze_out, num_variates",
    [
        # univariate
        (False, False, False, False, 1),

        # multivariate
        (False, False, False, False, 2),

        # univariate + squeeze
        (False, False, False, True, 1),

        # multivariate + squeeze
        (False, False, False, True, 2),

        # multivariate + replace by gt
        (False, False, True, False, 2),

        # multivariate + replace by gt + squeeze
        (False, False, True, True, 2),


        # with covar
        (True, False, False, False, 1),

        # with covar + autocomplete
        (True, True, False, False, 1),

        # with covar + autocomplete + squeeze
        (True, True, False, True, 1),

        # with covar + autocomplete + replace by gt + squeeze
        (True, True, True, True, 1),

        # multivariate with multi covar
        (True, False, False, False, 2),

        # multivariate with multi covar + autocomplete
        (True, True, False, False, 2),

        # multivariate with multi covar + autocomplete + squeeze + replace by gt
        (True, True, True, True, 2),
    ]
)
def test_impute_irregular(
    use_covariate,
    auto_complete,
    replace_by_gt,
    squeeze_out,
    num_variates
):

    # instantiate pipeline
    model = load_model()
    assert isinstance(model, TSICL)

    # load sparse irregular data
    inputs, covars = sparse_irregular_input_list(num_variates)

    if not use_covariate:
        covars = None
    # else:
    #     print([x.shape for x in inputs])
    #     print([x.shape for x in covars])
    
    q_levels = DEFAULT_Q_LEVELS

    mean, quantiles = model.impute(
        inputs              = inputs,
        covars              = covars,
        batch_size          = BATCH_SIZE,
        quantile_levels     = q_levels,
        device              = GPU_DEVICE,
        allow_auto_complete = auto_complete,
        replace_by_gt       = replace_by_gt,
        squeeze_output      = squeeze_out
    )

    # check output types
    assert isinstance(mean, list)
    assert isinstance(quantiles, list)
    assert isinstance(mean[0], torch.Tensor)
    assert isinstance(quantiles[0], torch.Tensor)

    # check output len
    assert len(mean) == len(quantiles) == len(inputs)
    
    # check expected output shapes
    expected_mean_shape = [
        (len(x),)
        if (squeeze_out and num_variates==1)
        else (num_variates, len(x),) if squeeze_out
        else (num_variates, len(x),1)
        for x in inputs
    ]
    assert all([x.shape == s for x,s in zip(mean, expected_mean_shape)])

    expected_quantile_shape = [
        (len(x),len(q_levels))
        if (squeeze_out and num_variates==1)
        else (num_variates, len(x),len(q_levels))
        for x in inputs
    ]
    assert all([x.shape == s for x,s in zip(quantiles, expected_quantile_shape)])

    # check no nan
    assert all([not x.isnan().any() for x in mean])
    assert all([not x.isnan().any() for x in quantiles])

@pytest.mark.parametrize(
    "inputs, covars, q_levels",
    [
        # wrong quantile
        (torch.randn(4,10,1), None, [0.1,0.123,0.9]),

        # covar not aligned with target
        (torch.randn(4,10,1), torch.randn(4,2,12,1), DEFAULT_Q_LEVELS),

    ]
)
def test_impute_raise_error(
    inputs,
    covars,
    q_levels
):
    # instantiate pipeline
    model = load_model()
    
    with pytest.raises(ValueError):
        _ = model.impute(
            inputs=inputs,
            covars=covars,
            quantile_levels=q_levels,
            point_estimator='median'
        )

    return


# check .forecast
@pytest.mark.parametrize(
    "pred_len, use_covariate, num_variates, auto_complete, squeeze_out, past_only, covar_forecast",
    [
        # univariate
        (5, False, 1, False, False, True, False),

        # multivariate
        (5, False, 2, False, False, True, False),

        # univariate + squeeze
        (5, False, 1, False, True, True, False),

        # multivariate + squeeze
        (5, False, 2, False, True, True, False),

        # univariate + autocomplete
        (5, False, 1, True, False, True, False),

        # multivariate + autocomplete
        (5, False, 2, True, False, True, False),


        # with past-only covar
        (5, True, 1, False, False, True, False),

        # past-only covar + multivariate
        (5, True, 2, False, False, True, False),

        # past-only covar + autocomplete
        (5, True, 1, True, False, True, False),

        # past-only covar + multivariate + autocomplete
        (5, True, 2, True, False, True, False),

        # past-only covar + multivariate + squeeze
        (5, True, 2, False, True, True, False),

        # past-only covar + autocomplete + covar forecast
        (5, True, 1, True, False, True, True),
        
        # past-only covar + multivariate + autocomplete + covar forecast
        (5, True, 2, True, False, True, True),


        # known covar + multivariate + autocomplete
        (5, True, 2, True, False, False, False),

        # known covar + multivariate + autocomplete + covar forecast
        (5, True, 2, True, False, False, True),

        # known covar + multivariate + autocomplete + covar forecast + squeeze
        (5, True, 2, True, True, False, True),

    ]
)
def test_forecast_irregular(
    pred_len,
    use_covariate,
    num_variates,
    auto_complete,
    squeeze_out,
    past_only,
    covar_forecast
):
    # instantiate pipeline
    model = load_model()
    assert isinstance(model, TSICL)

    # load sparse irregular data
    inputs, covars = sparse_irregular_input_list(
        num_variates,
        horizon_len = 0 if past_only else pred_len
    )

    if not use_covariate:
        covars = None
    
    q_levels = DEFAULT_Q_LEVELS

    mean, quantiles = model.forecast(
        inputs               = inputs,
        covars               = covars,
        prediction_length    = pred_len,
        batch_size           = BATCH_SIZE,
        quantile_levels      = q_levels,
        device               = GPU_DEVICE,
        allow_auto_complete  = auto_complete,
        allow_covar_forecast = covar_forecast,
        squeeze_output       = squeeze_out
    )

    # check output types
    assert isinstance(mean, list)
    assert isinstance(quantiles, list)
    assert isinstance(mean[0], torch.Tensor)
    assert isinstance(quantiles[0], torch.Tensor)

    # check output len
    assert len(mean) == len(quantiles) == len(inputs)

    # check expected output shapes
    expected_mean_shape = [
        (pred_len,)
        if (squeeze_out and num_variates==1)
        else (num_variates, pred_len,) if squeeze_out
        else (num_variates, pred_len,1)
        for _ in inputs
    ]
    assert all([x.shape == s for x,s in zip(mean, expected_mean_shape)])

    expected_quantile_shape = [
        (pred_len,len(q_levels))
        if (squeeze_out and num_variates==1)
        else (num_variates, pred_len, len(q_levels))
        for _ in inputs
    ]
    assert all([x.shape == s for x,s in zip(quantiles, expected_quantile_shape)])

    # check no nan
    assert all([not x.isnan().any() for x in mean])
    assert all([not x.isnan().any() for x in quantiles])
    
    return

@pytest.mark.parametrize(
    "inputs, covars, pred_len, q_levels",
    [
        # wrong quantile
        (torch.randn(4,10,1), None, 2, [0.1, 0.123]),

        # no pred len
        (torch.randn(4,10,1), None, 0, DEFAULT_Q_LEVELS),

        # covar not matching context+horizon len
        (torch.randn(4,10,1), torch.randn(4,2,11,1), 5, DEFAULT_Q_LEVELS)
    ]
)
def test_forecast_raise_error(
    inputs,
    covars,
    pred_len,
    q_levels
):

    # instantiate pipeline
    model = load_model()
    
    with pytest.raises(ValueError):
        _ = model.forecast(
            inputs = inputs,
            covars = covars,
            prediction_length = pred_len,
            quantile_levels = q_levels,
            point_estimator = 'median'
        )
    
    return

# check .extract latent
@pytest.mark.parametrize(
    "inputs, covars, setting",
    [   
        # imputation univariate
        (torch.randn(8,10,2), None, 'imputation'),

        # forecasting univariate
        (torch.randn(8,10,2), None, 'forecasting'),

        # imputation with covar
        (torch.randn(8,10,2), torch.randn(8,1,10,1), 'imputation'),

        # forecasting with past only covar
        (torch.randn(8,10,2), torch.randn(8,1,10,1), 'forecasting'),

        # forecasting with past known covar
        (torch.randn(8,10,2), torch.randn(8,3,15,1), 'forecasting'),        
    ]
)
def test_extract_latent_simple(
    inputs,
    covars,
    setting
):
    
    # instantiate pipeline
    model = load_model()
    assert isinstance(model, TSICL)

    pred_len = 0 if setting == 'imputation' else 5

    latents = model.extract_latent(
        inputs,
        covars,
        setting           = setting,
        batch_size        = BATCH_SIZE,
        device            = GPU_DEVICE,
        prediction_length = pred_len,
    )
    assert isinstance(latents, torch.Tensor)
    assert latents.ndim == 4
    assert len(latents) == len(inputs)
    assert latents.shape[1] == inputs.shape[-1]

    return