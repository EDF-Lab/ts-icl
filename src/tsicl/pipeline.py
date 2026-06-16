from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple
from warnings import warn

import numpy as np
import torch
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from hydra.utils import instantiate

from .model.network import TSICLNetwork
from .utils import batchify, complete_nans, make_grid
from .utils.data_adapter import validate_covar_inputs, validate_target_inputs
from .utils.scaler import CustomStandardScaler
from .utils.task_utils import prepare_context_tensors


class TSICL:
    """Main interface for TS-ICL (Time Series In-Context Learning) foundation model.

    Support both imputation and forecasting tasks.

    Parameters
    ----------
    model_path : str or Path, optional, default=None
        Path to the pre-trained model checkpoint file.

        - If provided and the file exists, it's loaded directly.
        - If provided but the file doesn't exist and `allow_auto_download` is true, the version
          specified by `checkpoint_version` is downloaded from Hugging Face Hub (repo: 'taharnbl/TS-ICL')
          to this path.
        - If `None` (default), the version specified by `checkpoint_version` is downloaded from
          Hugging Face Hub (repo: 'taharnbl/TS-ICL') and cached locally in the default
          Hugging Face cache directory (typically `~/.cache/huggingface/hub`).

    allow_auto_download : bool, default=True
        Whether to allow automatic download if the pretrained checkpoint cannot be found at the
        specified `model_path`.

    checkpoint_version : str, default='tsicl-v1.ckpt'
        Specifies which version of the pre-trained model checkpoint to use when `model_path`
        is `None` or points to a non-existent file (and `allow_auto_download` is true).
        Checkpoints are downloaded from https://huggingface.co/taharnbl/TS-ICL.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        checkpoint_version: str = 'tsicl-v1.ckpt',
        allow_auto_download: bool = True
    ):

        self.model_path          = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version  = checkpoint_version
        
        # load model:
        self._load_model()

    # ---------------
    # CORE UTILS
    # ---------------
    
    def _load_model(
        self
    ) -> None:
        """Load a model from a given path or download it if not available.

        Credits: TabICL repo, https://github.com/soda-inria/tabicl/blob/main/src/tabicl/_sklearn/regressor.py

        It uses `model_path` and `checkpoint_version` to determine the source.
         - If `model_path` is specified and exists, it's used directly.
         - If `model_path` is specified but doesn't exist (and auto-download is enabled),
           the version specified by `checkpoint_version` is downloaded to `model_path`.
         - If `model_path` is None, the version specified by `checkpoint_version` is downloaded
           from Hugging Face Hub and cached in the default Hugging Face cache directory.

        Raises
        ------
        AssertionError
            If the checkpoint doesn't contain the required 'config' or 'state_dict' keys.

        ValueError
            If a checkpoint cannot be found or downloaded based on the settings.
        """

        hf_repo_ID = 'taharnbl/TS-ICL'
        filename = self.checkpoint_version

        if self.model_path is None:
            
            try:
                model_path = Path( hf_hub_download(
                    repo_id          = hf_repo_ID,
                    filename         = filename,
                    local_files_only = True
                ) )
            except LocalEntryNotFoundError:
                if self.allow_auto_download:
                    print(f"Checkpoint '{filename}' not cached.\n Downloading from Hugging Face Hub ({hf_repo_ID}).\n")
                    model_path = Path( hf_hub_download(
                        repo_id  = hf_repo_ID,
                        filename = filename
                    ) )
                else:
                    raise ValueError(
                        f"Checkpoint '{filename}' not cached and automatic download is disabled.\n"
                        f"Set allow_auto_download=True to download the checkpoint from Hugging Face Hub ({hf_repo_ID})."
                    )
            if model_path:
                checkpoint = torch.load(model_path, map_location = "cpu", weights_only = True)
        
        else:
            model_path = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path

            if model_path.exists():
                checkpoint = torch.load(model_path, map_location = "cpu", weights_only = True)
            else:
                if self.allow_auto_download:
                    print(
                        f"Checkpoint not found at '{model_path}'.\n"
                        f"Downloading '{filename}' from Hugging Face Hub ({hf_repo_ID}) to this location.\n"
                    )
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path = hf_hub_download(
                        repo_id  = hf_repo_ID,
                        filename = filename,
                        local_dir = model_path.parent
                    )
                    Path(cache_path).rename(model_path)
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
                else:
                    raise ValueError(
                        f"Checkpoint not found at '{model_path}' and automatic download is disabled.\n"
                        f"Either provide a valid checkpoint path, or set allow_auto_download=True to download "
                        f"'{filename}' from Hugging Face Hub ({hf_repo_ID})."
                    )
        
        assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."

        # instantiate models:
        self.model_config = checkpoint['config']
        self.forecaster: TSICLNetwork = instantiate( self.model_config )
        self.imputer: TSICLNetwork = instantiate( self.model_config )

        # context / target lengths:
        self.max_context_length: int = self.model_config['max_context_len']
        self.max_target_length: int  = self.model_config['max_target_len']

        # grid lengths:
        self._grid_len_imputation  = self.max_context_length
        self._grid_len_forecasting = self.max_context_length + self.max_target_length

        # load weights:
        if 'forecaster' in checkpoint:
            self.forecaster.load_state_dict(checkpoint['forecaster'])
        else:
            print("The checkpoint doesn't contain the forecaster state, init as a random model")
        if 'imputer' in checkpoint:
            self.imputer.load_state_dict(checkpoint['imputer'])
        else:
            print("The checkpoint doesn't contain the imputer state, init as a random model")
        
        self.forecaster.eval()
        self.imputer.eval()

    def _get_quantile_indices(
        self,
        quantile_levels: List[float]
    ) -> List[int]:
        """Get the indices of selected quantiles in TS-ICL's output head.

        Parameters
        -----------
        quantile_levels : List[float]
            Selected quantile levels in (0, 1).
        
        Returns
        -------
        quantile_indices : List[int]
            List of quantile indices for slicing the output head.

        Raises
        ------
        ValueError
            If a quantile_level is not in the output distribution.
        """
        
        # get model quantiles from forecaster (same for imputer):
        start_quantile = self.forecaster.tf_icl.start_quantile
        end_quantile   = self.forecaster.tf_icl.end_quantile
        nb_quantiles   = self.forecaster.tf_icl.nb_quantiles

        self.training_quantile_levels = np.linspace(
            start_quantile * 100, 
            end_quantile * 100, 
            nb_quantiles
        ).tolist()

        quantile_levels = [100 * x for x in quantile_levels]

        if not set(quantile_levels).issubset(self.training_quantile_levels):
            raise ValueError(
                f"Misspecified arg `quantile_levels={quantile_levels}`, should be a subset of [0.01, 0.02, ..., 0.99]"
            )
        quantile_indices = [self.training_quantile_levels.index(q) for q in quantile_levels]
        
        return quantile_indices
    
    def _get_device(self, device: torch.device | None) -> torch.device:

        if not torch.cuda.is_available():
            device = torch.device('cpu')

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        return device
    
    # ---------------
    # ADAPTER UTILS
    # ---------------

    def input_adapter(
        self,
        inputs,
        covars: list | torch.Tensor | np.ndarray | None = None,
        batch_size: int = 32
    ):
        """Preprocess raw inputs of different formats to standardized TS-ICL inputs.
        
        Parameters
        ----------
        inputs
            See `TSICL.impute` or `TSICL.forecast`.
        
        covars : optional
            See `TSICL.impute` or `TSICL.forecast`.
        
        batch_size : int
            The mini batch size used for prediction.

        Returns
        -------
        inputs : List[torch.Tensor] or torch.Tensor
            Preprocessed inputs as Tensors of shape `(batch, seq_len, 1)`.
            If multiple variables to predict, all are stacked in the batch dimension.

        covars : List[torch.Tensor] or torch.Tensor or None
            Preprocessed covariates, if any, of shape `(batch, num_covariates, covar_seq_len, 1)`.

        num_var : int
            Number of target variables now hidden in batch dimension.
        
        num_batches : int
            The total number of mini batches to process.
        
        is_tensor : bool
            Whether the returned inputs are torch Tensor.

        has_covar : bool
            Whether the returned inputs have covariates.
        """
        
        # make sure covariates are in covars:
        if isinstance(inputs, list) and isinstance(inputs[0], dict):
            has_covar_keys = any([k in inputs[0] for k in ('past_covariates', 'future_covariates')])
            if has_covar_keys:
                covars = [{k:v for k,v in x.items() if k != 'target'} for x in inputs]

        # prepare inputs (targets):
        num_batches, is_tensor, num_var, inputs = validate_target_inputs(inputs, batch_size = batch_size)

        # prepare covariates:
        is_cov_tensor, covars = validate_covar_inputs(covars)

        # covar setting:
        has_covar = covars is not None

        # small check:
        if has_covar:
            # covars is now a tensor (bs, c, t, 1) or a list of tensors (c, t, 1)
            if isinstance(covars, torch.Tensor) and (len(covars) != len(inputs)):
                assert (len(inputs) % len(covars)) == 0, print('covars {} vs inputs {}'.format(covars.shape, inputs.shape)) # pyright: ignore[reportAttributeAccessIssue]
                covars = repeat(covars, 'n c t 1 -> (n b) c t 1', b = len(inputs) // len(covars))

            if isinstance(covars, list):
                assert len(covars) == len(inputs)
                if len(covars[0]) != len(inputs[0]):
                    assert all([len(x) != len(y) for x,y in zip(covars, inputs)])
                    assert all([len(x) == 1 for x in covars])
                    covars = [repeat(x, '1 c t 1 -> b c t 1', b = len(inputs[0])) for x in covars]

            # safety checks:
            assert len(covars) == len(inputs)
            assert is_cov_tensor == is_tensor
            

        return inputs, covars, num_var, num_batches, is_tensor, has_covar

    # ---------------
    # TASK UTILS
    # ---------------

    def _get_context_target_coords_f(
        self,
        grid: torch.Tensor,
        series_c: torch.Tensor,
        horizon_len: int,
        covariates: torch.Tensor | None = None,
        allow_auto_complete: bool = False,
        allow_covar_forecast: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Extract context and target tensors of the forecasting task.

        Parameters
        ----------
        grid : torch.Tensor
            Raw time grid of shape `(bs, T_grid, 1)`.

        series_c : torch.Tensor
            Tensor of past values of shape `(bs, T, 1)`.

        horizon_len : int
            Prediction length.

        covariates : torch.Tensor, optional
            Tensor of covariates of shape `(bs, C, T_cov, 1)`
            where T_cov=T (past-only) or T_cov=T+H (fully observed).

        allow_auto_complete : bool
            If True, will impute missing values in lookback and covariates before forecasting.

        allow_covar_forecast : bool
            If True, will forecast past-only covariates and use it as extended context.
        
        Returns
        -------
        out : Dict[str, torch.Tensor]
            Dict with keys `coords_c` (context coordinates), `coords_t` (target coordinates),
            `series_c` (context values), `series_t` (target values),
            `covar_c` (context covariates, if any), `covar_t` (target covariates, if any).
        """
        
        if allow_auto_complete:

            if series_c.isnan().any():
                imputed_lookback, _ = self.impute(
                    inputs              = series_c,
                    batch_size          = len(series_c),
                    device              = self._device,
                    denormalize         = True,
                    point_estimator     = 'median',
                    replace_by_gt       = True,
                    squeeze_output      = False
                ) # (b t 1)
                assert isinstance(imputed_lookback, torch.Tensor)
                series_c = imputed_lookback.to(series_c.device).squeeze(1)
                # imputed_lookback has shape (b c t 1) with c=1
                # (c is already in batch dim at this stage)
            if isinstance(covariates, torch.Tensor) and covariates.isnan().any():
                list_imputed_covar = []
                for c in range(covariates.shape[-3]):
                    imputed_covariates, _ = self.impute(
                        inputs              = covariates[...,c,:,:],
                        batch_size          = len(covariates),
                        device              = self._device,
                        denormalize         = True,
                        point_estimator     = 'median',
                        replace_by_gt       = True,
                        squeeze_output      = False
                    ) # (b 1 t 1)
                    list_imputed_covar.append(imputed_covariates)
                covariates = torch.cat(list_imputed_covar, dim=1).to(covariates.device)
        
        has_covar       = isinstance(covariates, torch.Tensor)
        covar_on_future = False

        lookback_len = min(series_c.shape[1], self.this_context_length)

        grid_threshold = self.max_context_length
        coords_c = grid[:,(grid_threshold-lookback_len):grid_threshold]
        coords_t = grid[:,grid_threshold:(grid_threshold+horizon_len)]

        # handle covar:
        if has_covar:

            # determine whether the covariates are past-only or fully-observed:
            covar_on_future = covariates.shape[-2] > series_c.shape[-2]

            # safety check:
            if covar_on_future:
                if covariates.shape[-2] != horizon_len + series_c.shape[-2]:
                    raise ValueError(
                        f"Covariate of seq len {covariates.shape[-2]} does not match context+horizon len {horizon_len + series_c.shape[-2]}"
                    )

            # get covariates on the horizon only:
            covar_t = covariates[..., -horizon_len:, :] if covar_on_future else None

            # get covariates on the lookback (max_context_len) only:
            covariates_c = covariates[..., :-horizon_len, :] if covar_on_future else covariates
            covariates_c = covariates_c[..., -lookback_len:,:] #
            assert covariates_c is not None

            # covariates_c: (b, c, t, 1)

            # optionally, forecast past-only covariates and use as GT:
            if (not covar_on_future) and (allow_covar_forecast):
                covar_t, _ = self.forecast(
                    inputs            = rearrange(covariates_c, 'b c t 1 -> b t c'),
                    prediction_length = horizon_len,
                    batch_size        = len(covariates_c),
                    device            = self._device,
                    denormalize       = True,
                    point_estimator   = 'median',
                    squeeze_output    = False
                )
                covar_on_future = True
                assert isinstance(covar_t, torch.Tensor)
                covar_t = covar_t.to(series_c.device)
        else:
            covariates_c, covar_t = None, None

        # make sure we don't exceed max context len:
        series_c = series_c[:,-lookback_len:]

        out = prepare_context_tensors(
            grid        = coords_c,
            series_c    = series_c,
            covariates  = covariates_c
            
        )
        out['coords_t'] = coords_t

        if covar_on_future:
            assert isinstance(covar_t, torch.Tensor)
            out['covar_t'] = covar_t # (bs, C, query_seq_len, 1)

        return out
    
    def _get_context_target_coords_i(
        self,
        grid: torch.Tensor,
        series_c: torch.Tensor,
        covariates: torch.Tensor | None = None,
        allow_auto_complete: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Extract context and target tensors of the imputation task.

        Parameters
        ----------
        grid : torch.Tensor
            Raw time grid of shape `(bs, T_grid, 1)`.

        series_c : torch.Tensor
            Tensor of context values of shape `(bs, T, 1)`.

        covariates : torch.Tensor, optional
            Tensor of covariates of shape `(bs, c, T, 1)`.

        allow_auto_complete : bool, default=`False`
            Allow imputation of the missing covariates, if any, and use the reconstructed covariates
            to impute the target time series.
        
        Returns
        -------
        out : Dict[str, torch.Tensor]
            Dict with keys `coords_c` (context coordinates), `coords_t` (target coordinates),
            `series_c` (context values), `series_t` (target values),
            `covar_c` (context covariates, if any), `covar_t` (target covariates, if any).
        """

        # grid (bs, T, 1)
        seq_len   = series_c.shape[1]
        grid      = grid[:,:seq_len,:]

        if isinstance(covariates, torch.Tensor) and covariates.shape[-2] != seq_len:
            raise ValueError(
                f"Target and covariate have different seq lengths {series_c.shape} vs {covariates.shape}"
            )

        # optionally, use TS-ICL to fill missing values in the covariates:
        if allow_auto_complete and isinstance(covariates, torch.Tensor) and covariates.isnan().any():
            list_imputed_covar = []
            for c in range(covariates.shape[-3]):
                imputed_covariates, _ = self.impute(
                    inputs              = covariates[...,c,:,:],
                    batch_size          = len(covariates),
                    device              = self._device,
                    denormalize         = True,
                    point_estimator     = 'median',
                    replace_by_gt       = True,
                    squeeze_output      = False
                ) # (b t 1)
                list_imputed_covar.append(imputed_covariates)
            covariates = torch.cat(list_imputed_covar, dim=1).to(covariates.device)
        
        out = prepare_context_tensors(
            grid       = grid,
            series_c   = series_c,
            covariates = covariates
        )

        # set targets:
        # (NB: we query the entire grid, context + missing points)
        out['coords_t'] = grid # (bs, seq_len, 1)
        if isinstance(covariates, torch.Tensor):
            out['covar_t'] = covariates   # (bs, C, query_seq_len, 1)

        return out

    # ---------------
    # MAIN UI
    # ---------------

    @torch.no_grad()
    def extract_latent(
        self,
        inputs,
        covars: List[torch.Tensor] | torch.Tensor | None = None,
        setting: Literal['imputation', 'forecasting'] = 'imputation',
        batch_size: int = 64,
        device: torch.device = torch.device('cuda'),
        prediction_length: int = 0
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Extract latent representation from the available context.

        Parameters
        ----------
        inputs
            See `TSICL.impute` or `TSICL.forecast`.
        
        covars : optional
            See `TSICL.impute` or `TSICL.forecast`.
        
        setting : Literal['imputation', 'forecasting']
            Whether the representation is computed from the imputation or the forecasting checkpoint.

        batch_size : int
            The batch size used for prediction.

        device : torch.device
            Set device for inference run.
            
        Returns
        -------
        all_latents : torch.Tensor or List[torch.Tensor]
            Extracted representations as a Tensor or a list of Tensor.
            If Tensor, the output shape is `(batch, num_variates, num_latents, hidden_dim)`, otherwise
            each element of the list has shape `(num_variates, context_length, 1)`.
        """

        if setting not in ['imputation', 'forecasting']:
            raise ValueError(
                f"Arg setting should either be 'imputation' or 'forecasting', received '{setting}' instead"
            )

        device = self._get_device(device)
        
        self.forecaster.eval()
        self.imputer.eval()

        self.forecaster.to(device)
        self.imputer.to(device)
        
        # define grid length:
        grid_len = self._grid_len_imputation if setting == 'imputation' else self._grid_len_forecasting

        self.this_context_length = self.max_context_length
        
        # 1/3 INPUT ADAPTER

        # prepare inputs:
        inputs, covars, num_var, num_batches, is_tensor, has_covar = self.input_adapter(
            inputs     = inputs,
            covars     = covars,
            batch_size = batch_size
        )
        # inputs is now a tensor (bs, t, 1) or a list of tensors (1, t, 1)

        # 2/3 RUN BATCH-WISE INFERENCE

        # allocate some space:
        all_latents = []

        # main loop:
        for idx in range(num_batches):

            # 
            series_c = inputs[idx*batch_size:(idx+1)*batch_size] if is_tensor else inputs[idx]
            assert isinstance(series_c, torch.Tensor)
            series_c = series_c.to(device)

            if covars is not None:
                covars_idx = covars[idx*batch_size:(idx+1)*batch_size] if is_tensor else covars[idx]
                assert isinstance(covars_idx, torch.Tensor)
                covars_idx = covars_idx.to(device)
            else:
                covars_idx = None

            # build context and target grids:
            grid = make_grid(grid_len, num_samples=len(series_c)).to(device)

            z_ = self._run_forward(
                grid               = grid,
                series_c           = series_c,
                covariates         = covars_idx,
                has_covar          = has_covar,
                prediction_length  = prediction_length,
                setting            = setting,
                denormalize        = False,
                save_scaler        = True,
                return_latent_only = True
            ) # (bs, num_tokens, hidden_dim)

            all_latents.append( z_.cpu().detach() )

        # 3/3 COLLECT FORECASTS

        # stack all preds:
        if isinstance(inputs, list):
            return all_latents
        else:
            all_latents  = torch.cat(all_latents, 0)
            all_latents  = rearrange(all_latents, '(b c) t d -> b c t d', c=num_var)

        return all_latents

    @torch.no_grad()
    def forecast(
        self,
        inputs,
        covars: List[torch.Tensor] | torch.Tensor | np.ndarray | None = None,
        prediction_length: int = 0,
        batch_size: int = 64,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        context_length: int | None = None,
        device: torch.device | None = None,
        denormalize: bool = True,
        point_estimator: Literal['mean', 'median'] = 'mean',
        allow_auto_complete: bool = False,
        allow_covar_forecast: bool = False,
        squeeze_output: bool = True
    ) -> Tuple[torch.Tensor | List[torch.Tensor], torch.Tensor | List[torch.Tensor]]:
        """
        Forecast future values for the given time series.

        Parameters
        ----------
        inputs
            The time series to forecast.
            May contain `NaN`s.
            Can be one of:

            * A multi-dimensional array-like (`torch.Tensor` or `np.ndarray`).
            If 1D, the array has shape `(context_length,)`.
            If 2D, the array has shape `(batch, context_length)`.
            If 3D, the array has shape `(batch, context_length, num_variates)` with `num_variates >= 1`.

            * A list of array-likes (`torch.Tensor` or `np.ndarray`), where each element is either 1D or 2D.
            If 1D, the array has shape `(context_length,)`.
            If 2D, the array has shape `(context_length, num_variates)`.
            *The `context_length`s may be different across elements of the list, not the `num_variates`*.
            
            * A list of dictionaries, where each dictionary may have the following keys.
                1. `target` (required): multi-dimensional array-likes (`torch.Tensor` or `np.ndarray`).
                If 1D, the array has shape `(context_length,)`.
                If 2D, the array has shape `(context_length, num_variates)`.
                Forecasts will be generated for items in `target`.
                2. `past_covariates` (optional): a dict of past-only covariates or past values of known future covariates.
                The keys of the dict
                must be names of the covariates and values must be 1-d `torch.Tensor` or `np.ndarray`
                with length equal to the `context_length` of `target`.
                3. `future_covariates` (optional): a dict of future values of known future covariates.
                The keys of the dict must be names of the
                covariates and values must be 1-d `torch.Tensor` or `np.ndarray` with length equal to the `prediction_length`.
                All keys in `future_covariates` must be a subset of the keys in `past_covariates`.

        covars (optional)
            The optional covariates available to help forecasting `inputs`.
            Each covariate has length `covar_length` where either
            `covar_length = context_length` (past-only) or
            `covar_length = context_length + prediction_length` (fuly-observed).
            Each covariate may contain `NaN` values observed at arbitrary timesteps.

            If `inputs` is a list of dicts, with keys `past_covariates` or `future_covariates`,
             `covars` will be overwritten by `inputs`.

            * A multi-dimensional array-like (`torch.Tensor` or `np.ndarray`).
            If 1D, the array has shape `(covar_length,)`.
            If 2D, the array has shape `(batch, covar_length)`.
            If 3D, the array has shape `(batch, covar_length, num_variates)` with `num_variates >= 1`.

            * A list of array-likes (`torch.Tensor` or `np.ndarray`), where each element is either 1D or 2D.
            If 1D, the array has shape `(covar_length,)`.
            If 2D, the array has shape `(covar_length, num_variates)`.
            *The `covar_length`s may be different across elements of the list, not the `num_variates`*.

        prediction_length : int
            Number of timesteps to forecast (horizon).
        
        batch_size : int
            The batch size used for prediction.

        quantile_levels : List[float]
            Quantile levels to compute, by default [0.1, 0.2, ..., 0.9].
            Must be a subset of [0.01, 0.02, ..., 0.99].
        
        context_length : int, optional
            Maximum length used for the lookback window at inference.
            Defaults to the model maximum length, 4096.
        
        device : torch.device
            Set device for inference run.
        
        denormalize : bool
            Whether to return z-normalized values (`False`) or denormalized values in data space (`True`).
        
        point_estimator : Literal['mean', 'median']:
            Set pointwise estimator as the `mean` of all quantiles or as the 0.5 quantile (`'median'`).
        
        allow_auto_complete : bool
            Allow imputation of both the lookback window and/or the missing covariates, if any,
            and use the reconstructed values as extended context for the target time series.

        allow_covar_forecast : bool
            Allow forecasting of past-only covariates and use forecasted values as
            extended context for the target time series.
        
        squeeze_output : bool
            If `True` squeeze all unit dims in the outputs.

        Returns
        -------
        mean : torch.Tensor or List[torch.Tensor]
            A batched torch Tensor or a list of torch Tensors containing containing the pointwise forecasts.
            If Tensor, the output shape is `(batch, num_variates, prediction_length, 1)`, otherwise
            each element of the list has shape `(num_variates, prediction_length, 1)`.

        quantiles : torch.Tensor or List[torch.Tensor]
            A batched torch Tensor or a list of torch Tensors containing quantile forecasts.
            If Tensor, the output shape is `(batch, num_variates, prediction_length, len(quantile_levels))`, otherwise
            each element of the list has shape `(num_variates, prediction_length, len(quantile_levels))`.
        """

        if prediction_length <= 0:
            raise ValueError(
                f"arg `prediction_length` should be a positive ìnt, received {prediction_length} instead"
            )
        
        device = self._get_device(device)

        pred_fn = self._rollout_f

        if context_length is None:
            self.this_context_length = self.max_context_length
        else:
            self.this_context_length = min(self.max_context_length, context_length)

        mean, quantiles = self._predict_main(
            pred_fn              = pred_fn,
            inputs               = inputs,
            covars               = covars,
            grid_len             = self._grid_len_forecasting,
            prediction_length    = prediction_length,
            batch_size           = batch_size,
            quantile_levels      = quantile_levels,
            device               = device,
            denormalize          = denormalize,
            point_estimator      = point_estimator,
            allow_auto_complete  = allow_auto_complete,
            allow_covar_forecast = allow_covar_forecast
        )
        if squeeze_output:
            mean      = mean.squeeze() if isinstance(mean, torch.Tensor) else [x.squeeze() for x in mean]
            quantiles = quantiles.squeeze() if isinstance(quantiles, torch.Tensor) else [x.squeeze() for x in quantiles]

        return mean, quantiles
    
    @torch.no_grad()
    def impute(
        self,
        inputs,
        covars: List[torch.Tensor] | torch.Tensor | None = None,
        batch_size: int = 64,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        device: torch.device | None = None,
        denormalize: bool = True,
        point_estimator: Literal['mean', 'median'] = 'mean',
        allow_auto_complete: bool = False,
        replace_by_gt: bool = False,
        squeeze_output: bool = True
    ) -> Tuple[torch.Tensor | List[torch.Tensor], torch.Tensor | List[torch.Tensor]]:
        """
        Impute missing values in the given time series.

        Parameters
        ----------
        inputs
            The time series with missing values (`NaN`) to fill in.
            Can be one of:

            * A multi-dimensional array-like (`torch.Tensor` or `np.ndarray`).
            If 1D, the array has shape `(context_length,)`.
            If 2D, the array has shape `(batch, context_length)`.
            If 3D, the array has shape `(batch, context_length, num_variates)` with `num_variates >= 1`.

            * A list of array-likes (`torch.Tensor` or `np.ndarray`), where each element is either 1D or 2D.
            If 1D, the array has shape `(context_length,)`.
            If 2D, the array has shape `(context_length, num_variates)`.
            *The `context_length`s may be different across elements of the list, not the `num_variates`*.
            
            * A 2-dimensional `pd.DataFrame` of shape `(batch, context_length)`.

        covars : optional
            The optional covariates available to help imputing `inputs`.
            The covariates must be aligned on the same grid as  `inputs` (sequence length is `context_length`),
            but may contain missing values (`NaN`s) at arbitrary instants.
            Can be one of:

            * A multi-dimensional array-like (`torch.Tensor` or `np.ndarray`).
            If 1D, the array has shape `(context_length,)` (single covariate).
            If 2D, the array has shape `(batch, context_length)` (single covariate).
            If 3D, the array has shape `(batch, context_length, num_covariates)` with `num_covariates >= 1`.
            If 4D, the array has shape `(batch, num_covariates, context_length, 1)`.
            
            * A list of array-likes (`torch.Tensor` or `np.ndarray`). Each element mutli-dimensional.
            If 1D, the array has shape `(context_length,)` (single covariate).
            If 2D, the array has shape `(context_length, num_covariates)` with `num_covariates >= 1`.
            If 3D, the array has shape `(context_length, num_covariates, 1)`.

        batch_size : int
            The batch size used for prediction.

        quantile_levels : List[float]
            Quantile levels to compute, by default [0.1, 0.2, ..., 0.9].
            Must be a subset of [0.01, 0.02, ..., 0.99].

        device : torch.device
            Set device for inference run.
        
        denormalize : bool
            Whether to return z-normalized values (`False`) or denormalized values in data space (`True`).
        
        point_estimator : Literal['mean', 'median']:
            Set pointwise estimator as the `mean` of all quantiles or as the 0.5 quantile (`'median'`).
        
        allow_auto_complete : bool
            Allow imputation of the missing covariates, if any, and use the reconstructed covariates
            to impute the target time series.

        replace_by_gt : bool
            The method always reconstructs the whole time series, incl. the available observation.
            However, if this arg is set to `True`, reconstructed values are replaced by the ground truth,
            when available.
        
        squeeze_output : bool
            If `True` squeeze all unit dims in the outputs.

        Returns
        -------
        mean : torch.Tensor or List[torch.Tensor]
            A batched torch Tensor or a list of torch Tensors containing containing the pointwise reconstructions.
            If Tensor, the output shape is `(batch, num_variates, context_length, 1)`, otherwise
            each element of the list has shape `(num_variates, context_length, 1)`.

        quantiles : torch.Tensor or List[torch.Tensor]
            A batched torch Tensor or a list of torch Tensors containing quantile forecasts.
            If Tensor, the output shape is `(batch, num_variates, context_length, len(quantile_levels))`, otherwise
            each element of the list has shape `(num_variates, context_length, len(quantile_levels))`.
        """

        device = self._get_device(device)
    
        pred_fn = self._rollout_i

        mean, quantiles = self._predict_main(
            pred_fn             = pred_fn,
            inputs              = inputs,
            covars              = covars,
            grid_len            = self._grid_len_imputation,
            batch_size          = batch_size,
            quantile_levels     = quantile_levels,
            device              = device,
            denormalize         = denormalize,
            point_estimator     = point_estimator,
            allow_auto_complete = allow_auto_complete,
            replace_by_gt       = replace_by_gt
        )

        if squeeze_output:
            mean      = mean.squeeze() if isinstance(mean, torch.Tensor) else [x.squeeze() for x in mean]
            quantiles = quantiles.squeeze() if isinstance(quantiles, torch.Tensor) else [x.squeeze() for x in quantiles]

        return mean, quantiles

    # ---------------
    # FORWARD PASS
    # ---------------

    def _predict_main(
        self,
        pred_fn: Callable,
        inputs,
        grid_len: int,
        covars: List[torch.Tensor] | torch.Tensor | np.ndarray | None = None,
        prediction_length: int = 0,
        batch_size: int = 64,
        quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        device: torch.device = torch.device('cuda'),
        denormalize: bool = False,
        point_estimator: Literal['mean', 'median'] = 'mean',
        **kwargs
    ) -> Tuple[torch.Tensor | List[torch.Tensor], torch.Tensor | List[torch.Tensor]]:
        """
        Inference pass of TS-ICL.

        Parameters
        ----------
        inputs
            Available context values either has a tensor or a list of tensors.
        
        grid_len : int
            Max grid length for imputation or forecasting.

        covar
            The optional covariates available to help imputing `inputs`.
        
        prediction_length : int
            Queried horizon length (forecasting only).
        
        batch_size : int
            Max batch size to process the inputs.
        
        quantiles_levels : list[float]
            List of quantiles to fetch.
        
        device : torch.device
            Device to map the model to.
        
        denormalize : bool
            Whether the undo z-normalization after predicting.
        
        point_estimator : Literal['mean', 'median']:
            Set pointwise estimator as the `mean` of all quantiles or as the 0.5 quantile (`'median'`).
            
        Returns
        -------
        point_forecast : torch.Tensor or List[torch.Tensor]
            Forecasts or imputed values as tensors of shape `(bs, num_variates, seq_len, 1)`.
        
        quantiles : torch.Tensor or List[torch.Tensor]
            Estimated quantiles as tensors of shape `(bs, num_variates, seq_len, num_quantiles)`.
        """

        if point_estimator not in ['mean', 'median']:
            raise ValueError(
                f"Point estimator should be `'mean'` or `'median'`, received `{point_estimator}` instead"
            )

        # 0/3 INITIALIZE PIPELINE

        self._device = device
        self.imputer.to(device)
        self.forecaster.to(device)

        # get quantile indices:
        quantile_indices = self._get_quantile_indices(quantile_levels)
        
        # get pointwise estimator fn:
        def _get_median(x: torch.Tensor) -> torch.Tensor:
            median_idx = quantile_levels.index(0.5)
            return x[..., median_idx:(median_idx+1)]
        
        def _get_mean(x: torch.Tensor) -> torch.Tensor:
            return x.mean(-1, keepdim = True)
        
        estimator = _get_median if point_estimator == 'median' else _get_mean
        if point_estimator == 'median' and 0.5 not in quantile_levels:
            warn(f"Point estimator is `median` but `0.5` not in `quantile_levels`={quantile_levels}, switch to `mean` estimator instead")
            estimator = _get_mean

        # 1/3 INPUT ADAPTER

        # prepare inputs:
        inputs, covars, num_var, num_batches, is_tensor, has_covar = self.input_adapter(
            inputs     = inputs,
            covars     = covars,
            batch_size = batch_size
        )
        # inputs is now a tensor (bs, t, 1) or a list of tensors (1, t, 1)
        # covars is now a tensor (bs, c, t, 1) or a list of tensors (1, c, t, 1) or None

        # 2/3 RUN BATCH-WISE INFERENCE

        # allocate some space:
        quantiles = []

        # create batches of data:
        batch_series_c = batchify(inputs, batch_size=batch_size, device=device)
        batch_covars   = batchify(covars, batch_size=batch_size, device=device, batch_len=len(inputs))

        # main loop:
        for series_ctx, covar_ctx in zip(batch_series_c, batch_covars):

            assert isinstance(series_ctx, torch.Tensor)

            # build context and target grids:
            grid = make_grid(grid_len, num_samples=len(series_ctx)).to(device)

            # predict:
            yhat = pred_fn(
                grid              = grid,
                series_c          = series_ctx,
                covariates        = covar_ctx,
                has_covar         = has_covar,
                prediction_length = prediction_length,
                denormalize       = denormalize,
                **kwargs
            ) # tensor of shape (b, t, num_quantiles)

            quantiles.append(yhat[...,quantile_indices].cpu().detach())


        # 3/3 COLLECT FORECASTS

        # compute pointwise estimator + stack all preds when possible:
        if isinstance(inputs, list):
            point_forecast = [estimator(x) for x in quantiles] # (c t 1)
        else:
            quantiles  = torch.cat(quantiles, 0)
            quantiles  = rearrange(quantiles, '(b c) t q -> b c t q', c=num_var)
            point_forecast = estimator(quantiles) # (b c t 1) 

        # check dims:
        if is_tensor:
            inputs = rearrange(inputs, '(b c) t 1 -> b c t 1', c=num_var)

        # safety check:
        assert len(quantiles) == len(inputs)
        
        return point_forecast, quantiles

    def _rollout_f(
        self,
        grid: torch.Tensor,
        series_c: torch.Tensor,
        covariates: torch.Tensor | None,
        has_covar: bool,
        prediction_length: int,
        denormalize: bool,
        allow_auto_complete: bool = False,
        allow_covar_forecast: bool = False
    ) -> torch.Tensor:
        """Forecasting wrapper.

        Returns
        -------
        yhat : torch.Tensor
            Estimated quantiles of shape `(bs, prediction_length, Q)`
        """
        
        if prediction_length > self.max_target_length:
            
            # series_c (bs, T, 1)
            # covariates (bs, C, T_cov, 1)

            n_rollouts = prediction_length // self.max_target_length + int(prediction_length % self.max_target_length > 0)
            yhat = []

            covar_on_future = False
            covariates_ii = None
            if isinstance(covariates, torch.Tensor):

                covar_len = covariates.shape[-2]
                # determine whether the covariates are past-only or fully-observed:

                covar_on_future = covar_len > series_c.shape[-2]
                # safety check:
                if covar_on_future:
                    assert covar_len == prediction_length + series_c.shape[-2]
                
                covar_past    = covariates[...,:series_c.shape[-2],:]
                covar_horizon = covariates[...,series_c.shape[-2]:,:] if covar_on_future else \
                    torch.nan * torch.ones(*covariates.shape[:-2], prediction_length, 1).to(covariates.device)
                
                covariates_ii = covariates[...,:series_c.shape[-2]+self.max_target_length,:] if covar_on_future else covariates

            for ii in range(n_rollouts):

                # get prediction length:
                pred_len = prediction_length % self.max_target_length if ii == (n_rollouts - 1) else self.max_target_length

                # update covariates:
                if isinstance(covariates, torch.Tensor):
                    covariates_ii = torch.cat([
                        covar_past,
                        covar_horizon[...,ii*pred_len:(ii+1)*pred_len,:]
                    ]) if covar_on_future else covar_past

                yhat_ii = self._run_forward(
                    grid                 = grid,
                    series_c             = series_c,
                    covariates           = covariates_ii,
                    has_covar            = has_covar,
                    prediction_length    = pred_len,
                    setting              = 'forecasting',
                    denormalize          = True,
                    save_scaler          = (ii==0),
                    allow_auto_complete  = allow_auto_complete,
                    allow_covar_forecast = allow_covar_forecast       
                )

                yhat.append(yhat_ii)

                # update context:
                series_c = torch.cat([
                    series_c,                          # (bs, t_in, 1)
                    yhat_ii.mean(dim=-1, keepdim=True) # (bs, t_mis, 1)
                ], dim = 1)

                # update covariates:
                if isinstance(covariates_ii, torch.Tensor):
                    covar_past = torch.cat([
                        covar_past,
                        covar_horizon[...,ii*pred_len:(ii+1)*pred_len,:]
                    ], dim = -2)

            
            # stack forecasts: 
            yhat = torch.cat(yhat, dim=1) # (bs, t, q)
            if not denormalize:
                yhat = self.scaler.transform(yhat)

        else:

            yhat = self._run_forward(
                grid                 = grid,
                series_c             = series_c,
                covariates           = covariates,
                has_covar            = has_covar,
                prediction_length    = prediction_length,
                setting              = 'forecasting',
                denormalize          = denormalize,
                save_scaler          = not denormalize,
                allow_auto_complete  = allow_auto_complete,
                allow_covar_forecast = allow_covar_forecast
            )
            
        return yhat

    def _rollout_i(
        self,
        grid: torch.Tensor,
        series_c: torch.Tensor,
        covariates: torch.Tensor | None,
        has_covar: bool,
        denormalize: bool,
        prediction_length: int = 0,
        allow_auto_complete: bool = False,
        replace_by_gt: bool = False
    ) -> torch.Tensor:
        """Imputation wrapper.

        Returns
        -------
        yhat : torch.Tensor
            Estimated quantiles of shape `(bs, seq_len, Q)`
        """
        
        context_length = series_c.shape[1]

        if context_length > self.max_context_length:
            
            raise NotImplementedError(
                f"Imputation supports context windows of maximum length {self.max_context_length}, \
                cannot handle series of len {context_length}. \
                Rollout will be implemented in a future version."
            )

            # n_rollouts = context_length // self.max_context_length + int(context_length % self.max_context_length > 0)
            # yhat = []

        else:

            yhat = self._run_forward(
                grid                = grid,
                series_c            = series_c,
                covariates          = covariates,
                has_covar           = has_covar,
                prediction_length   = prediction_length,
                setting             = 'imputation',
                denormalize         = denormalize,
                save_scaler         = not denormalize,
                allow_auto_complete = allow_auto_complete,
                replace_by_gt       = replace_by_gt
            )
            
        return yhat
            
    def _run_forward(
        self,
        grid: torch.Tensor,
        series_c: torch.Tensor,
        covariates: torch.Tensor | None,
        has_covar: bool,
        prediction_length: int | None,
        setting: Literal['imputation', 'forecasting'],
        denormalize: bool,
        save_scaler: bool,
        return_latent_only: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Create batch tensors and run TS-ICL forward pass.

        Parameters
        ----------
        grid : torch.Tensor
            Raw time coordinates, tensor of shape `(batch, max_grid_len, 1)`.

        series_c : torch.Tensor
            Preprocessed context series, of shape `(batch, seq_len, 1)`.

        covariates : torch.Tensor, optional
            Preprocessed covariates, if any, of shape `(batch, num_covariates, covar_seq_len, 1)`.

        has_covar : bool
            Whether `covariates` is not None.
        
        prediction_length : int
            Number of timesteps to forecast (forecasting-only).
        
        setting : Literal['imputation', 'forecasting']
            Task setting, imputation or forecasting.
        
        denormalize : bool
            Whether the undo z-normalization after predicting.
        
        save_scaler : bool
            Whether to save z-normalization stats.
        
        return_latent_only: bool
            If True, return latent representations instead of predicted quantiles.

        Returns
        -------
        yhat : torch.Tensor
            Predicted target quantiles of shape `(bs, T_mis, Q)` or
            latent representation of shape `(bs, num_variates, num_latents, hidden_dim)`.
        """
        
        if setting not in ['imputation', 'forecasting']:
            raise ValueError(
                f"Arg setting should either be 'imputation' or 'forecasting', received '{setting}' instead"
            )

        # check preprocessing went OK:
        assert grid.ndim == 3        
        assert series_c.ndim == 3
        assert series_c.shape[-1] == 1
        
        # prepare tensors:
        if setting == 'forecasting':
            if ( not isinstance(prediction_length, int) ) or prediction_length <= 0:
                raise ValueError(
                    f"Arg `prediction_length` should be a >0 `int`, received {prediction_length} instead"
                )
            processed_inputs = self._get_context_target_coords_f(
                grid = grid, series_c = series_c, horizon_len = prediction_length, covariates = covariates, **kwargs
            )
        else:
            processed_inputs = self._get_context_target_coords_i(
                grid = grid, series_c = series_c, covariates = covariates, **kwargs
            )
        
        # predict:
        yhat = self._predict_batch(
            series_c           = processed_inputs['series_c'], # (bs, max_context_len, 1)
            coords_c           = processed_inputs['coords_c'], # (bs, max_context_len, 1)
            coords_t           = processed_inputs['coords_t'], # (bs, query_seq_len, 1)
            covar_c            = processed_inputs['covar_c'] if has_covar else None,
            covar_t            = processed_inputs.get('covar_t') if has_covar else None,
            setting            = setting,
            denormalize        = denormalize,
            save_scaler        = save_scaler,
            return_latent_only = return_latent_only,
            **kwargs
        )

        return yhat

    def _predict_batch(
        self,
        series_c: torch.Tensor,
        coords_c: torch.Tensor,
        coords_t: torch.Tensor,
        covar_c: torch.Tensor | None = None,
        covar_t: torch.Tensor | None = None,
        setting: Literal['imputation', 'forecasting'] = 'imputation',
        denormalize: bool = False,
        save_scaler: bool = False,
        return_latent_only: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Run one forward of TS-ICL on batched data.

        Parameters
        ----------
        series_c : torch.Tensor
            `(bs, T_ctx, 1)` tensor of observed values.

        coords_c : torch.Tensor
            `(bs, T_ctx, 1)` tensor of observed coords.

        coords_t : torch.Tensor
            `(bs, T_mis, 1)` tensor of queried coords.

        covar_c : torch.Tensor, optional
            `(bs, C, T_ctx, 1)` tensor of covariates on context grid.

        covar_t : torch.Tensor, optional
            `(bs, C, T_mis, 1)` tensor of covariates on target grid.

        setting : Literal['imputation', 'forecasting']
            Task setting, imputation or forecasting.
        
        denormalize : bool
            Whether the undo z-normalization after predicting.
        
        save_scaler : bool
            Whether to save z-normalization stats.
        
        return_latent_only: bool
            If True, return latent representations instead of predicted quantiles.

        Returns
        -------
        out : torch.Tensor
            Predicted target quantiles of shape `(bs, T_mis, Q)` or
            latent representation of shape `(bs, num_variates, num_latents, hidden_dim)`.
        """

        # select model, imputer or forecaster:
        model = self.imputer if setting == 'imputation' else self.forecaster

        series_covar = None

        # replace nans:
        if setting == 'forecasting':
            out = complete_nans(series_c, coords_c, is_test=True)
            series_c, coords_c = out['values'], out['coords']

        # z-normalize context series:
        scaler   = CustomStandardScaler(dim=1,epsilon=1e-5)
        scaler.fit(series_c)
        series_c_norm = scaler.transform(series_c)

        # z-normalize covariate series:
        if (covar_c is not None) and (covar_t is not None):
            scaler_cov = CustomStandardScaler(dim=-2,epsilon=1e-7)
            scaler_cov.fit(covar_t) # type: ignore
            covar_t      = scaler_cov.transform(covar_t)         # (bs, C, query_seq_len, 1)
            covar_c      = scaler_cov.transform(covar_c)         # (bs, C, max_context_len, 1)
            series_covar = torch.cat([covar_c, covar_t], dim=-2) # (bs, C, max_context_len + query_seq_len, 1)

        # build query coords:
        query_coords = torch.cat([coords_c, coords_t], dim=1) # [N, T_in+T, 1]

        # get predictions:
        out = model(
            series               = series_c_norm,
            coords               = coords_c,
            target_coords        = query_coords,
            series_covar         = series_covar,
            coords_covar         = query_coords,
            undo_asinh_transform = True,
            return_latent_only   = return_latent_only
        )
        if return_latent_only:
            return out
        else:
            quantiles, _ = out

        if save_scaler:
            self.scaler = scaler

        # by default, available context is replaced by predited reconstructions
        # here, we allow forcing available context to be set to ground truth
        # in the predicted quantiles:
        # (arg used in imputation only)
        if kwargs.get('replace_by_gt', False):
            
            # if coords_t[i,j]==coords_c[i,k]
            # then we set quantiles[i,j,:] = series_c​[i,k,0]:

            # build match matrix of shape (bs, query_seq_len, max_seq_len):
            match = (coords_t == coords_c.transpose(1, 2))            

            # get non_zero indices:
            b_idx, t1_idx, t2_idx = match.nonzero(as_tuple=True)

            # replace values + repeat on all quantiles:
            quantiles[b_idx, t1_idx] = repeat(series_c_norm[b_idx, t2_idx], 'n 1 -> n q', q=quantiles.shape[-1])
            
        if denormalize:
            quantiles = scaler.inv_transform(quantiles)
            # if std_mask.sum() > 0:
            #     quantiles[std_mask] = repeat(series_c[std_mask].mean(dim=1,keepdim=True), 'b 1 q -> b h q', h = quantiles.shape[1])
    
        return quantiles

