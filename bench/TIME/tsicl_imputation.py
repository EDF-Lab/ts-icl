import os
import sys
from pathlib import Path
from time import time

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

import numpy as np

import torch
from einops import rearrange, repeat
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.time_feature import get_seasonality

sys.path.insert(0, os.getcwd())

from bench.TIME.utils.utils import get_available_terms
from bench.TIME.utils.imputation import (
    build_default_imputation_scenarios,
    prepare_context,
    get_max_context_length
)
from bench.TIME.utils.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from bench.TIME.utils.metrics import compute_per_window_metrics_from_quantiles
from bench.TIME.utils.read_results import export_metrics

from tsicl.pipeline import TSICL
from tsicl.plot import plot_sample_imputation

DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]



def run_tsicl_experiment(
    dataset_name: str,
    model_path: Path | str,
    storage_path: Path | str,
    terms: list[str] | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = 32,
    to_univariate: bool = True,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
    missing_seed: int = 42,
    make_plots: bool = False,
    max_nb_plots: int = 3
):
    """Run TS_ICL imputation experiments over multiple masking scenarios."""

    # Set CUDA device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(f"No terms defined for {dataset_name=} in config")

    if quantile_levels is None:
        quantile_levels = DEFAULT_QUANTILES

    assert output_dir is not None
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"{'=' * 60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings["prediction_length"]
        test_length = settings["test_length"]
        val_length = settings["val_length"]

        print(
            f"  Config: {prediction_length=}, {test_length=}, {val_length=}"
        )

        # instantiate model
        print("  Initializing model pipeline...")
        model = TSICL(
            model_path          = model_path,
            allow_auto_download = False
        )

        # Dataset Initialization
        dataset = Dataset(
            name              = dataset_name,
            storage_path      = storage_path,
            term              = term,
            to_univariate     = to_univariate,
            prediction_length = prediction_length,
            test_length       = test_length,
            val_length        = val_length,
        )

        # Determine split
        eval_data = dataset.test_data
        eval_input_list = list(eval_data.input)
        total_items = len(eval_input_list)
        num_windows = dataset.windows

        # determine context length:
        context_length = get_max_context_length(term)

        # prepare imputation task:
        scenarios = build_default_imputation_scenarios(context_length)
        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(
            f"    - Series length: min={dataset._min_series_length}, "
            f"max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}"
        )
        print(f"    - Test split: {test_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")
        print("    - Scenarios:")

        for sc in scenarios:
            print(
                f"      * {sc["name"]}: "
                f"pointwise={sc.get("missing_pointwise_ratio", 0.0)}, "
                f"num_blocks={sc.get("num_blocks", 0)}, "
                f"block_size={sc.get("block_size", 0)}"
            )

        original_stderr = sys.stderr

        season_length = get_seasonality(dataset.freq)

        # results_rows = []

        # main loop, through all imputation tasks:
        for scenario in scenarios:

            # get task:
            scenario_name = scenario["name"]
            print(f"\n  >>> Running scenario: {scenario_name}")

            # fix seed:
            rng = np.random.default_rng(missing_seed)
            missing_count = 0

            pred_quantiles_instances = []
            fc_batch_masks = []
            gt_instances = []
            ctx_instances = []

            t0 = time()

            for start in range(0, total_items, batch_size):
                end = min(start + batch_size, total_items)
                # Load context only for current batch
                batch_items = [
                    prepare_context(
                        eval_input_list[i],
                        rng            = rng,
                        context_length = context_length,
                        scenario       = scenario,
                    )
                    for i in range(start, end)
                ]

                # list of segments to impute:
                batch_contexts = [x[0] for x in batch_items]

                # tensors of ground truths and target masks:
                batch_targets = [
                    rearrange(x[1], "t c -> c t")
                    for x in batch_items
                ]   # (c, t)
                batch_masks = [
                    rearrange(x[2], "t c -> c t").bool()
                    for x in batch_items
                ]   # (c, t)

                # predict:
                try:
                    with torch.no_grad():
                        _, batch_q = model.impute(
                            inputs            = batch_contexts,
                            batch_size        = batch_size,
                            quantile_levels   = quantile_levels,
                            device            = torch.device(device_map),
                            denormalize       = True,
                            squeeze_output    = False
                        )
                        if isinstance(batch_q, torch.Tensor):
                            batch_q = rearrange(batch_q, "b c t q -> b q c t").cpu().numpy()
                            batch_q = [
                                batch_q[idx] for idx in range(len(batch_q))
                            ] # (q c t)
                        else:
                            batch_q = [
                                rearrange(x, "c t q -> q c t").cpu().numpy() for x in batch_q
                            ]
                finally:
                    sys.stderr = original_stderr

                # store forecasts:
                pred_quantiles_instances.extend(batch_q)
                # (num_quantiles, num_variates, prediction_length)

                # store target missing masks:
                fc_batch_masks.extend(batch_masks)

                # store ground truths and ctx:
                gt_instances.extend(batch_targets)
                ctx_instances.extend(batch_contexts)

            t1 = time()

            # prepare shapes:
            num_total_instances = len(pred_quantiles_instances)
            num_variates        = dataset.target_dim
            num_series_exp      = num_total_instances // num_windows
            num_quantiles       = len(pred_quantiles_instances[0])
            max_ctx_len         = context_length

            # Initialize arrays
            quantiles_array = np.full(
                (num_series_exp, num_windows, num_quantiles, num_variates, max_ctx_len),
                np.nan,
                dtype=np.float32
            )
            gt_array = np.full(
                (num_series_exp, num_windows, num_variates, max_ctx_len),
                np.nan,
                dtype=np.float32
            )
            ctx_array = np.full(
                (num_series_exp, num_windows, num_variates, max_ctx_len),
                np.nan,
                dtype=np.float32
            )

            # prepare exports:
            ds_config = f"{dataset_name}/{term}"
            scenario_output_dir = Path(output_dir) / ds_config / scenario_name
            scenario_output_dir.mkdir(parents=True, exist_ok=True)

            permutation = np.random.permutation(len(pred_quantiles_instances))
            for idx in range(len(pred_quantiles_instances)):

                series_idx = idx // num_windows
                window_idx = idx % num_windows

                pred  = pred_quantiles_instances[idx] # (q, c, T) 9, 1, 672
                ytrue = gt_instances[idx]   # (c, T)
                mask  = fc_batch_masks[idx] # (c, T)

                # get max context len:
                cur_len = ytrue.shape[-1]

                # restrict to missing instants only:
                y_hat = pred[repeat(mask, "c T -> q c T", q=len(pred))]
                y_hat = rearrange(y_hat, "(q c T) -> q c T", q=len(pred), c=pred.shape[1])

                # get number of missing points:
                target_len = y_hat.shape[-1]

                # restrict GT to missing instants only:
                y_target = rearrange( ytrue[mask], "(c T) -> c T", c = len(ytrue) )

                # fill arrays up to target len only:
                quantiles_array[series_idx, window_idx, :, :, :target_len] = y_hat
                gt_array[series_idx, window_idx, :, :target_len]           = y_target.cpu().numpy()

                # ctx is for the seasonal (repeat) error only:
                ctx_array[series_idx, window_idx, :, :cur_len] = ytrue.cpu().numpy() # (c, T)

                if permutation[idx] < max_nb_plots and make_plots:
                    fig, ax = plot_sample_imputation(
                        quantiles           = rearrange(pred[:,0,:], "q t -> t q"),
                        y_ctx               = ctx_instances[idx][:,0], # (t)
                        y_true              = ytrue[0], # (t)
                        quantile_levels     = quantile_levels,
                        show_context_points = True,
                        plot_iqr            = True,
                        model_name          = "TS-ICL",
                        iqr_bands           = ((0.1, 0.9), (0.3,0.7)),
                        is_blockwise        = "block" in scenario_name
                    )
                    ax.set_title("{} - sample {}".format(ds_config, idx))
                    fig.tight_layout()
                    fig.savefig(
                        f"{scenario_output_dir}/sample_{idx}.pdf",
                        dpi=300,
                        bbox_inches="tight"
                    )
                    plt.close(fig)


            # Compute per-window metrics
            print("    Computing per-window metrics...")
            metrics = compute_per_window_metrics_from_quantiles(
                predictions_quantiles = quantiles_array,
                ground_truth          = gt_array,
                context               = ctx_array,
                seasonality           = season_length,
                quantile_levels       = quantile_levels,
            )
            metric_means = {
                metric_name: float(np.nanmean(metric_values))
                for metric_name, metric_values in metrics.items()
            }
            
            nb_chunks = num_series_exp * num_windows * (1 if to_univariate else num_variates)
            
            metrics_summary = {
                "model"     : ["TS-ICL"],
                "Dataset"   : [ds_config],
                "scenario"  : [scenario_name],
                "nb_chunks" : [int( nb_chunks )],
                "inference_time_s" : [t1 - t0]
            } | metric_means

            df = pd.DataFrame(metrics_summary)
            df.set_index("Dataset", inplace=True)
            df.to_csv( scenario_output_dir / "summary.csv")

            print(f"    Completed scenario {scenario_name}")
            print(f"    Masked points: {missing_count}")
            print(f"    Output: {scenario_output_dir}")
            print(f"    Inference time: {t1 - t0}s")

    print(f"\n{"=" * 60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


@hydra.main(version_base=None, config_path="config", config_name="time")
def main(cfg: DictConfig):

    load_dotenv()

    # 
    this_file_dir = os.path.dirname(__file__)

    # data storage path:
    time_repo = Path( os.environ.get("TIME_REPO", "not_found") )
    assert time_repo.exists()

    # list of datasets to eval:
    datasets = getattr(cfg, "datasets", None)

    # get model ckpt path:
    model_path = Path( os.environ.get("TSICL_PATH", "not_found") )
    assert model_path.exists()

    # build output path:
    output_path = Path( this_file_dir ) / "results" / "imputation"
    output_path.mkdir(exist_ok=True, parents=True)

    # Handle dataset list or "all_datasets"
    config_path = Path( this_file_dir ) / "config" / "datasets.yaml"
    assert config_path.exists()

    if datasets is None or datasets == "all_datasets":
        # Load all datasets from config
        config = load_dataset_config(config_path)
        datasets = list(config.get("datasets", {}).keys())
        print(f"Running all {len(datasets)} datasets from config:")
        for ds in datasets:
            print(f"  - {ds}")
    else:
        if isinstance(datasets, str):
            datasets = [datasets]
        assert isinstance(datasets, list)

    # Iterate over all datasets with progress logging
    total_datasets = len(datasets)
    for idx, dataset_name in enumerate(datasets, 1):
        print(f"\n{'#' * 60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#' * 60}")

        try:
            run_tsicl_experiment(
                dataset_name    = dataset_name,
                terms           = None,
                model_path      = model_path,
                storage_path    = time_repo,
                output_dir      = output_path,
                batch_size      = getattr(cfg, "batch_size", 64),
                to_univariate   = bool(getattr(cfg, "to_univariate", True)),
                config_path     = config_path,
                quantile_levels = list(
                    getattr(cfg, "quantiles", DEFAULT_QUANTILES)
                ),
                missing_seed    = getattr(cfg, "missing_seed", 42),
                make_plots      = getattr(cfg, "make_plots", False),
                max_nb_plots    = getattr(cfg, "nb_plots", 3)
            )
        except Exception as e:
            print(f"ERROR: Failed to run experiment for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'#' * 60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#' * 60}")

    export_metrics(output_path, "tsicl_imputation")


if __name__ == "__main__":
    main()
