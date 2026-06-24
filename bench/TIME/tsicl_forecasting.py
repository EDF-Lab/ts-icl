import os
import sys
from pathlib import Path
from time import time

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

import numpy as np

import torch
from einops import rearrange

from gluonts.time_feature import get_seasonality

sys.path.insert(0, os.getcwd())

from bench.TIME.utils.saver import save_window_predictions
from bench.TIME.utils.utils import get_available_terms
from bench.TIME.utils.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from bench.TIME.utils.read_results import export_metrics

from tsicl.pipeline import TSICL


def run_tsicl_experiment(
    dataset_name: str,
    model_path: Path | str,
    storage_path: Path | str,
    terms: list[str] | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = 32,
    context_length: int = 2048,
    config_path: Path | None = None,
    quantile_levels: list[float] | None = None,
):
    """ Run TS_ICL experiments."""

    # Set CUDA device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset configuration
    print("Loading configuration...")
    config = load_dataset_config(config_path)

    # Auto-detect available terms from config if not specified
    if terms is None:
        terms = get_available_terms(dataset_name, config)
        if not terms:
            raise ValueError(
                f"No terms defined for {dataset_name=} in config"
            )

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    assert output_dir is not None
    Path(output_dir).mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Terms: {terms}")
    print(f"{'='*60}")

    for term in terms:
        print(f"\n--- Term: {term} ---")

        # Get settings from config
        settings          = get_dataset_settings(dataset_name, term, config)
        prediction_length = settings["prediction_length"]
        test_length       = settings["test_length"]
        val_length        = settings["val_length"]
        print(f"  Config: {prediction_length=}, {test_length=}, {val_length=}")

        # instantiate model:
        print(f"  Initializing model pipeline...")
        model = TSICL(
            model_path          = model_path,
            allow_auto_download = False
        )

        # Dataset Initialization
        dataset = Dataset(
            name              = dataset_name,
            storage_path      = storage_path,
            term              = term,
            to_univariate     = False,
            prediction_length = prediction_length,
            test_length       = test_length,
            val_length        = val_length,
        )

        # Determine split
        data_length = test_length
        num_windows = dataset.windows
        split_name = "Test split"
        eval_data = dataset.test_data

        print("  Dataset info:")
        print(f"    - Frequency: {dataset.freq}")
        print(f"    - Num series: {len(dataset.hf_dataset)}")
        print(f"    - Target dim: {dataset.target_dim}")
        print(f"    - Series length: min={dataset._min_series_length}, \
              max={dataset._max_series_length}, avg={dataset._avg_series_length:.1f}")
        print(f"    - {split_name}: {data_length} steps")
        print(f"    - Prediction length: {dataset.prediction_length}")
        print(f"    - Windows: {num_windows}")

        season_length = get_seasonality(dataset.freq)

        # ---------------------------------------------------------
        # 1. Running Inference (Model Specific Logic)
        # ---------------------------------------------------------
        # Helper function to prepare a single context
        def _prepare_context(d):
            target = np.asarray(d["target"])

            # Manually truncate context
            seq_len = target.shape[-1]
            if seq_len > context_length:
                target = target[..., -context_length:]

            if target.ndim == 1:
                target = target[np.newaxis, :]
            
            return torch.tensor(target).permute(1, 0) # (seq_len, q)

        # Batch Inference with lazy loading
        fc_quantiles_batches = []
        eval_input_list = list(eval_data.input)  # Convert to list for indexing
        total_items = len(eval_input_list)

        original_stderr = sys.stderr

        t0 = time()

        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            # Load context only for current batch
            batch_contexts = [
                _prepare_context(eval_input_list[i])
                for i in range(start, end)
            ]
            
            try:
                with torch.no_grad():
                    _, batch_q = model.forecast(
                            inputs            = batch_contexts,
                            prediction_length = prediction_length,
                            batch_size        = batch_size,
                            quantile_levels   = quantile_levels,
                            context_length    = context_length,
                            device            = torch.device(device_map),
                            denormalize       = True,
                            squeeze_output    = False
                    )
                    batch_q = rearrange(batch_q, "b c t q -> b q c t").cpu().numpy()  # type: ignore
                    # (batch, num_quantiles, num_variates, prediction_length)

            finally:
                sys.stderr = original_stderr

            # Stack into batch: (batch_size, num_quantiles, num_variates, prediction_length)
            batch_q_array = batch_q
            fc_quantiles_batches.append(batch_q_array)

            # Optional progress logging
            if (start // batch_size + 1) % 10 == 0:
                print(f"    Processed {min(start + batch_size, total_items)}/{total_items}...")
            
        # Concatenate all batches into a single array
        # Shape: (num_total_instances, num_quantiles, num_variates, prediction_length)
        fc_quantiles = np.concatenate(fc_quantiles_batches, axis=0)
        t1 = time()
        # ---------------------------------------------------------
        # 3. Saving Results
        # ---------------------------------------------------------
        ds_config = f"{dataset_name}/{term}"
        model_hyperparams = {
            "model": "TS-ICL",
            "context_length": context_length,
            "quantile_levels": quantile_levels,
        }

        metadata = save_window_predictions(
            dataset           = dataset,
            fc_quantiles      = fc_quantiles,
            ds_config         = ds_config,
            output_base_dir   = output_dir,
            seasonality       = season_length,
            model_hyperparams = model_hyperparams,
            quantile_levels   = quantile_levels,
        )

        print(f"  Completed: {metadata["num_series"]} series × {metadata["num_windows"]} windows")
        print(f"  Output: {metadata.get("output_dir", output_dir)}")
        print(f"  Inference time: {t1-t0}s")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    return


@hydra.main(version_base=None, config_path="config", config_name="time")
def main(cfg : DictConfig):

    load_dotenv()

    # 
    this_file_dir = os.path.dirname(__file__)

    # data storage path:
    time_repo = Path( os.environ.get("TIME_REPO", "not_found") )
    assert time_repo.exists()

    # list of datasets to eval:
    datasets    = getattr(cfg, "datasets", None)

    # get model ckpt path:
    model_path = Path( os.environ.get("TSICL_PATH", "not_found") )
    assert model_path.exists()

    # build output path:
    output_path = Path( this_file_dir ) / "results" / "forecasting" / "ts-icl"
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
        print(f"\n{'#'*60}")
        print(f"# Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"{'#'*60}")

        try:
            run_tsicl_experiment(
                dataset_name    = dataset_name,
                terms           = None,
                model_path      = model_path,
                storage_path    = time_repo,
                output_dir      = output_path,
                batch_size      = getattr(cfg, "batch_size", 64),
                context_length  = getattr(cfg, "context_length", 2048),
                config_path     = config_path,
                quantile_levels = list(
                    getattr(cfg, "quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                ),
            )

        except Exception as e:
            print(f"ERROR: Failed to run experiment for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
    print(f"\n{'#'*60}")
    print(f"# All {total_datasets} dataset(s) completed!")
    print(f"{'#'*60}")

    # agg all csv files into one dataframe:
    export_metrics(output_path, "ts_icl")


if __name__ == "__main__":
    main()