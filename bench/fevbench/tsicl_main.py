from __future__ import annotations

from typing import cast, Dict
import os
import sys
import shutil
import warnings
from pathlib import Path
from time import time
from unittest.mock import patch

import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

import random
import numpy as np
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt

import datasets
import fev
from fev import leaderboard

import torch
from einops import rearrange

sys.path.insert(0, os.getcwd())

from tsicl.pipeline import TSICL
from tsicl.plot import plot_sample_forecast


# ---------------------------------------------------------------------------
# fev-bench data helpers borrowed from Chronos2 pipeline
# ---------------------------------------------------------------------------


def _cast_fev_features(
    past_data: "datasets.Dataset",
    future_data: "datasets.Dataset",
    target_columns: list[str],
    past_dynamic_columns: list[str],
    known_dynamic_columns: list[str],
) -> tuple["datasets.Dataset", "datasets.Dataset"]:
    import datasets

    dynamic_columns = [*past_dynamic_columns, *known_dynamic_columns]
    cat_cols = []
    for col in dynamic_columns:
        item = past_data[0][col]
        if not np.issubdtype(item.dtype, np.number):
            cat_cols.append(col)

    numeric_cols = target_columns + list(set(dynamic_columns) - set(cat_cols))
    past_feature_updates = {col: datasets.Sequence(datasets.Value("float64")) for col in numeric_cols} | {
        col: datasets.Sequence(datasets.Value("string")) for col in cat_cols
    }
    past_data_features = past_data.features
    past_data_features.update(past_feature_updates)
    past_data = past_data.cast(past_data_features)

    future_cat_cols = [k for k in cat_cols if k in known_dynamic_columns]
    future_numeric_cols = list(set(known_dynamic_columns) - set(future_cat_cols))
    future_feature_updates = {col: datasets.Sequence(datasets.Value("float64")) for col in future_numeric_cols} | {
        col: datasets.Sequence(datasets.Value("string")) for col in future_cat_cols
    }
    future_data_features = future_data.features
    future_data_features.update(future_feature_updates)
    future_data = future_data.cast(future_data_features)

    return past_data, future_data


def convert_fev_window_to_list_of_dicts_input(
    window: "fev.EvaluationWindow", as_univariate: bool
) -> tuple[list[dict[str, torch.Tensor | dict[str, torch.Tensor]]], list[str], list[str], list[str]]:

    if as_univariate:
        past_data, future_data = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        target_columns = ["target"]
        past_dynamic_columns = []
        known_dynamic_columns = []
    else:
        past_data, future_data = window.get_input_data()
        target_columns = window.target_columns
        past_dynamic_columns = window.past_dynamic_columns
        known_dynamic_columns = window.known_dynamic_columns

    past_data, future_data = _cast_fev_features(
        past_data=past_data,
        future_data=future_data,
        target_columns=target_columns,
        past_dynamic_columns=past_dynamic_columns,
        known_dynamic_columns=known_dynamic_columns,
    )

    num_series: int = len(past_data)
    num_past_covariates: int = len(past_dynamic_columns)
    num_future_covariates: int = len(known_dynamic_columns)

    # We use numpy format because torch does not support str covariates
    target_data = past_data.select_columns(target_columns).with_format("numpy")
    # past of past-only and known-future covariates
    dynamic_columns = [*past_dynamic_columns, *known_dynamic_columns]
    past_covariate_data = past_data.select_columns(dynamic_columns).with_format("numpy")
    future_known_data = future_data.select_columns(known_dynamic_columns).with_format("numpy")

    if num_past_covariates + num_future_covariates > 0:
        assert len(past_covariate_data) == num_series
    if num_future_covariates > 0:
        assert len(future_known_data) == num_series

    inputs: list[dict[str, torch.Tensor | dict[str, torch.Tensor]]] = []
    for idx, target_row in enumerate(target_data):
        target_row = cast(dict, target_row)
        # this assumes that the targets have the same length for multivariate tasks
        target_tensor_i = np.stack([target_row[col] for col in target_columns])
        entry: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "target": torch.Tensor(rearrange(target_tensor_i,"c t -> t c"))
        }

        if len(dynamic_columns) > 0:
            past_covariate_row = past_covariate_data[idx]
            entry["past_covariates"] = {
                col: torch.Tensor(rearrange(past_covariate_row[col], "t -> t 1"))
                for col in dynamic_columns if past_covariate_row[col].dtype.kind in "fuib"
            }

        if len(known_dynamic_columns) > 0:
            future_known_row = future_known_data[idx]
            entry["future_covariates"] = {
                col: torch.Tensor(rearrange(future_known_row[col], "t -> t 1"))
                for col in known_dynamic_columns if future_known_row[col].dtype.kind in "fuib"
            }

        inputs.append(entry)

    return inputs, target_columns, past_dynamic_columns, known_dynamic_columns


# ---------------------------------------------------------------------------
# Local-only data loading helpers
# ---------------------------------------------------------------------------


def _force_offline_mode() -> None:
    """Prevent any Hugging Face network call when loading FEV datasets."""
    for var in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(var, "1")


def _install_list_feature_compat() -> None:
    """Map the `List` feature type (datasets>=3.0) to `Sequence`.

    The FEV snapshots on disk were saved by a newer `datasets` that emits
    `{"_type": "List", ...}` for list columns. Older `datasets` versions
    (which the cluster env pins) raise `Feature type 'List' not found`.
    Aliasing `List` -> `Sequence` fixes decoding and keeps fev's
    `isinstance(feat, datasets.Sequence)` checks intact.
    """
    from datasets.features import features as _ff

    if "List" not in _ff._FEATURE_TYPES:
        _ff._FEATURE_TYPES["List"] = _ff.Sequence # type: ignore


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batchify(lst: list, batch_size: int):
    """Convert list into batches of desired size."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

# ---------------------------------------------------------------------------
# TS-ICL utils to run fev-bench
# ---------------------------------------------------------------------------


def _predict_window(
    model: TSICL,
    inputs: list,
    horizon: int,
    quantile_levels: list[float],
    max_context_length: int,
    batch_size: int = 64,
    device_map: str = "cuda"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (quantiles[N, T, Q], means[N, T]) for a single FEV window."""
    
    # alloc some space:
    list_quantiles, list_means = [],[]

    def _nan_var_to_num(
        x: Dict[str, torch.Tensor],
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if mask.sum() > 0:
            x["target"][:,mask] = 0.0
        return x
    
    # loop through all the inputs:
    for batch in batchify(inputs, batch_size=batch_size):

        mask = [((~torch.Tensor(x["target"]).isnan()).sum(0) == 0) for x in batch]
        batch_clean = [_nan_var_to_num(x,m) for x,m in zip(batch, mask)]

        try:
            # model inference:
            with torch.no_grad():
                _, batch_q = model.forecast(
                        inputs            = batch_clean,
                        prediction_length = horizon,
                        batch_size        = batch_size,
                        quantile_levels   = quantile_levels,
                        context_length    = max_context_length,
                        device            = torch.device(device_map),
                        denormalize       = True,
                        squeeze_output    = False
                )
                # since fev tasks are homogenous, we can safely stack the list of tensors into a single tensor
                if isinstance(batch_q, list):
                    batch_q = torch.stack(batch_q, dim=0) # (b c t q)
                assert isinstance(batch_q, torch.Tensor)

                # XXX handle nan outputs / no need?
                bs, c, t, _ = batch_q.shape
                batch_q = rearrange(batch_q, "b c t q -> (b c t) q")
                assert not batch_q.isnan().any()
                all_nans_mask = (~batch_q.isnan()).sum(-1) == 0 # (b,)
                batch_q[all_nans_mask] = 0.0
                batch_q = rearrange(batch_q, "(b c t) q -> b c t q", b=bs,t=t,c=c)

                quantiles_np = batch_q.cpu().numpy()                 # (bs, num_variates, horizon, num_quantiles)
                mean_np = quantiles_np.mean(axis=-1, keepdims=False) # (bs, num_variates, horizon)

        finally:
            sys.stderr = sys.stderr

        # append all forecasts:
        list_quantiles.append(quantiles_np)
        list_means.append(mean_np)

    # stack:
    quantiles_np = np.concatenate(list_quantiles, axis=0)
    mean_np      = np.concatenate(list_means, axis=0)
    
    return quantiles_np, mean_np


def run_tsicl_experiment(
    task,
    model_path: Path | str,
    output_dir: str | Path | None = None,
    as_univariate: bool = True,
    batch_size: int = 32,
    max_context_length: int = 2048,
    use_covariates: bool = False,
    use_static_covariates: bool = False,
    past_only: bool = False,
    quantile_levels: list[float] | None = None,
    make_plots: bool = False,
    max_nb_plots: int = 3
) -> tuple[list[datasets.DatasetDict], float, dict]:
    """ Run TS-ICL experiments."""

    # Set CUDA device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    
    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    assert output_dir is not None
    Path(output_dir).mkdir(exist_ok=True)
    
    # plot dir:
    task_name = str(task.task_name)

    # instantiate model:
    print("  Initializing model pipeline...")
    model = TSICL(
        model_path          = model_path,
        allow_auto_download = True,
    )

    if make_plots:
        plot_path = Path(output_dir) / "inference_plots"
        plot_path.mkdir(exist_ok=True, parents=True)
        all_plots_idx = np.random.permutation(task.num_windows)[:max_nb_plots]

    inference_time = 0.0
    predictions_per_window = []
    nb_samples = 0
    plot_idx = 0

    # loop through all windows:
    for window in task.iter_windows():
        
        # prepare data (chronos2 utils):
        inputs, target_columns, past_dynamic_columns, known_dynamic_columns = (
            convert_fev_window_to_list_of_dicts_input(window=window, as_univariate=as_univariate)
        )
        print(f"Infos: {target_columns=}, {past_dynamic_columns=}, {known_dynamic_columns=}")

        if past_only:
            past_dynamic_columns = [col for col in known_dynamic_columns if col in past_dynamic_columns]
            known_dynamic_columns = []

            inputs = [
                {
                    "target"          : x["target"],
                    "past_covariates" : x["past_covariates"]
                } for x in inputs
            ]
        
        num_variates: int = len(target_columns) + len(past_dynamic_columns) + len(known_dynamic_columns)
        if batch_size < num_variates:
            batch_size = num_variates

        t0 = time()
        # run forward pass on every input in the window:
        quantiles_np, means_np = _predict_window(
            model                 = model,
            inputs                = inputs,
            horizon               = task.horizon,
            quantile_levels       = quantile_levels,
            max_context_length    = max_context_length,
            batch_size            = batch_size,
            device_map            = device_map
        )
        inference_time += time() - t0
        nb_samples += len(quantiles_np)

        # store forecast in the right format (from chronos2 pipeline):
        multivariate_forecast: dict[str, dict[str, np.ndarray]] = {
            variate_name: {} for variate_name in target_columns
        }
        point_forecast = means_np  # [num_items, n_variates, horizon]

        for v_idx, variate_name in enumerate(target_columns):
            multivariate_forecast[variate_name]["predictions"] = point_forecast[:, v_idx]

        for q_idx, level in enumerate(quantile_levels):
            for v_idx, variate_name in enumerate(target_columns):
                multivariate_forecast[variate_name][str(level)] = quantiles_np[:, v_idx, :, q_idx]

        predictions_dict: dict = {}
        for variate_name in target_columns:
            predictions_dict[variate_name] = datasets.Dataset.from_dict(
                {
                    k: multivariate_forecast[variate_name][k]
                    for k in ["predictions"] + [str(q) for q in quantile_levels]
                }
            )
        predictions = datasets.DatasetDict(predictions_dict)
        predictions.set_format("numpy")

        if as_univariate:
            predictions = fev.utils.combine_univariate_predictions_to_multivariate(
                predictions, window.target_columns
            )

        predictions_per_window.append(predictions)

        if make_plots and plot_idx in all_plots_idx:

            if task.num_windows == 1:
                n_iters = max_nb_plots
            else:
                n_iters = 1
            
            for _ in range(n_iters):
                past_data, _, test_data = window._get_past_future_test_data()
                past_data.set_format("numpy")
                test_data.set_format("numpy")

                for variate_name in window.target_columns:
                    y_test = np.array(test_data[variate_name], dtype=np.float64)
                    # y_hat = predictions[variate_name]["predictions"]
                    quantiles = np.stack([predictions[variate_name][str(level)] for level in quantile_levels])
                    quantiles = rearrange(quantiles, "q n t -> n t q")
                    # (n t q) and (n, t) arrays
                    
                    if len(past_data) > 1:
                        sample_idx = int( np.random.permutation(len(past_data))[0] )
                        context    = past_data.with_format("numpy")[variate_name][sample_idx][None,:]
                        quantiles  = quantiles[sample_idx:sample_idx+1,...]
                        y_test     = y_test[sample_idx:sample_idx+1,...]
                    else:
                        context    = past_data.with_format("numpy")[variate_name]
                        sample_idx = int( np.random.permutation(len(y_test))[0] )

                    sample_idx = 0 if len(past_data) > 1 else sample_idx
                    fig, _ = plot_sample_forecast(
                        quantiles       = quantiles[sample_idx],
                        y_ctx           = context[sample_idx],
                        y_true          = y_test[sample_idx],
                        max_points      = 672,
                        quantile_levels = quantile_levels,
                        plot_iqr        = True,
                        iqr_bands       = ((0.1, 0.9), (0.3,0.7)),
                        model_name      = "TS-ICL",
                    )
                    fig.tight_layout()
                    fig.savefig(
                        plot_path / f"{task_name.replace("/","")}_{plot_idx}_{sample_idx}.pdf",
                        dpi=300,
                        bbox_inches="tight"
                    )
                    plt.close(fig)

        plot_idx += 1

    extra_info = {
        "model_config": {
            "model_name"            : "TS-ICL",
            "context_length"        : max_context_length,
            "quantile_levels"       : quantile_levels,
            "use_covariates"        : use_covariates,
            "use_static_covariates" : use_static_covariates,
        },
        "nb_samples" : nb_samples
    }
    return predictions_per_window, inference_time, extra_info


@hydra.main(version_base=None, config_path="config", config_name="fevbench")
def main(cfg : DictConfig):

    load_dotenv()

    _force_offline_mode()
    _install_list_feature_compat()
    datasets.disable_progress_bars()
    _set_seed(int(cfg.seed))

    # 
    this_file_dir = os.path.dirname(__file__)

    # data storage path:
    fev_repo = Path( os.environ.get("FEV_BENCH_REPO", "not_found") )
    assert fev_repo.exists()

    # get model ckpt path:
    model_path = Path( os.environ.get("TSICL_PATH", "not_found") )
    assert model_path.exists()

    # build output path:
    output_path = Path( this_file_dir ) / "results" / "tsicl"
    output_path.mkdir(exist_ok=True, parents=True)

    # name of the leaderboard path:
    leaderboard_path = Path( this_file_dir ) / "results" / "leaderboard"
    leaderboard_path.mkdir(exist_ok=True, parents=True)

    # get yaml:
    config_path = Path( this_file_dir ) / "config" / "tasks.yaml"
    assert config_path.exists()

    # instantiate benchmark:
    benchmark = fev.Benchmark.from_yaml(str(config_path))

    num_tasks = OmegaConf.select(cfg, "num_tasks", default=None)
    requested_tasks = (
        benchmark.tasks if num_tasks is None else benchmark.tasks[: int(num_tasks)]
    )

    # get list of tasks:
    tasks: list[fev.Task] = []
    for task in requested_tasks:
        task.dataset_path = str(fev_repo)
        try:
            task.load_full_dataset()
            tasks.append(task)
        except:            
            warnings.warn(
                f"[Data] Skipping task '{task.dataset_config}': no local snapshot at "
                f"{fev_repo / (task.dataset_config or "")}"
            )
    print(
        f"[Config] Running {len(tasks)} of {len(benchmark.tasks)} FEV tasks "
        f"(requested {len(requested_tasks)}, missing {len(requested_tasks) - len(tasks)})"
    )

    # set expe parameters:
    use_covariates     = bool(OmegaConf.select(cfg, "use_covariates", default=True))
    use_static_covariates = bool(
        OmegaConf.select(cfg, "use_static_covariates", default=False)
    )
    print(
        f"[Config] use_covariates={use_covariates} "
        f"use_static_covariates={use_static_covariates}"
    )

    out_csv = output_path / "fevbench_tsicl.csv"

    summaries = []

    # Iterate over all datasets with progress logging
    for task_idx, task in enumerate(tasks, 1):
        task_label = task.dataset_config or task.task_name or f"task_{task_idx}"
        
        print(f"\n{'#' * 60}")
        print(f"# Task {task_idx}/{len(tasks)}: {task_label} (horizon={task.horizon})")
        print(f"{'#' * 60}")

        make_plots = getattr(cfg, "make_plots", False)

        try:
            predictions_per_window, inference_time, extra_info = run_tsicl_experiment(
                task                  = task,
                model_path            = model_path,
                output_dir            = output_path,
                as_univariate         = bool(getattr(cfg, "to_univariate", True)),
                batch_size            = getattr(cfg, "batch_size", 64),
                max_context_length    = getattr(cfg, "context_length", 2048),
                use_covariates        = use_covariates,
                use_static_covariates = use_static_covariates,
                quantile_levels       = list(
                    getattr(cfg, "quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                ),
                make_plots            = getattr(cfg, "make_plots", False),
                max_nb_plots          = getattr(cfg, "nb_plots", 3)
            )

        except Exception as e:
            print(f"ERROR: Failed to run experiment for {task_label}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        evaluation_summary = task.evaluation_summary(
            predictions_per_window,
            model_name       = "TS-ICL",
            inference_time_s = inference_time,
            extra_info       = extra_info,
        )
        evaluation_summary["nb_chunks"] = extra_info["nb_samples"]
        summaries.append(evaluation_summary)
    
        # Show and save the results
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(output_path / "all_metrics.csv", index=False)
        
        metrics = summary_df[[
            "dataset_config",
            "horizon",
            "nb_chunks",
            "num_windows",
            "task_name",
            "SQL",
            "MASE",
            "WAPE",
            "WQL",
            "inference_time_s"
        ]]
        print(metrics.iloc[-1])

        metrics = metrics.sort_values(
            by=["task_name"],
            ascending=True,
            ignore_index=True,
            key=lambda col: col.str.lower()
        )

        is_int_cols   = metrics.dtypes[(metrics.dtypes == "int64")].index
        is_float_cols = metrics.dtypes[(metrics.dtypes == "float64")].index
        
        # compute mean and geometric mean:
        mean_scores  = metrics[is_float_cols].apply(np.mean, axis=0)
        gmean_scores = metrics[is_float_cols].apply(gmean, axis=0)
        
        metrics.loc["gmean", is_float_cols] = gmean_scores
        metrics.loc["mean", is_float_cols]  = mean_scores
        metrics.loc["mean", is_int_cols] = metrics[is_int_cols].apply(np.sum, axis=0)
        
        metrics.to_csv(out_csv)

    shutil.copy(output_path / "all_metrics.csv", leaderboard_path / "tsicl.csv")
    
    less_than_100_tasks = [
        "autoarima", "autoets", "deepar", "stat_ensemble", "tabpfn-ts"
    ]

    try:
        df_bench = leaderboard(
            summaries = [
                f for f in leaderboard_path.iterdir()
                if all([name not in str(f) for name in less_than_100_tasks])
            ],
            baseline_model = "Seasonal Naive"
        )
        df_bench.to_csv( output_path / "leaderboard.csv" )
        print(df_bench.head())

    finally:
        sys.stderr = sys.stderr

    print(f"\n{'#'*60}")
    print(f"# All {len(tasks)} task(s) completed!")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()