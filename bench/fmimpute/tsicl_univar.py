import os
import sys
from pathlib import Path
from time import time

import hydra
from omegaconf import DictConfig
import yaml
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from einops import rearrange

sys.path.insert(0, os.getcwd())

from bench.fmimpute.utils.metrics import initialize_gluonts_metrics, update_gluonts_metrics
from bench.fmimpute.utils.read_results import make_table
from bench.fmimpute.utils.load import load_ts_data

from tsicl.pipeline import TSICL
from tsicl.plot import plot_sample_imputation


def batchify(x: torch.Tensor, batch_size: int):
    """Convert list into batches of desired size."""
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]


# ---------------------------------------------------------------------------

def run_tsicl_experiment(
    raw_values,
    raw_gt,
    model_path: Path | str,
    output_dir: str | Path,
    batch_size: int = 32,
    quantile_levels: list[float] | None = None,
    make_plots: bool = False,
    max_nb_plots: int = 3
):
    
    # Set CUDA device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    
    # instantiate model:
    model = TSICL(
        model_path          = model_path,
        allow_auto_download = False
    )

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    assert output_dir is not None
    Path(output_dir).mkdir(exist_ok=True)

    plot_path = Path(output_dir) / "inference_plots"
    if make_plots:
        plot_path.mkdir(exist_ok=True, parents=True)
        nb_batches = len(raw_values) // batch_size + (len(raw_values) % batch_size) > 0
        all_plots_idx = np.random.permutation(nb_batches)[:max_nb_plots]

    # initialize metrics:
    gluonts_metrics = initialize_gluonts_metrics(axis=None)

    t0 = time()
    nb_windows = 0
    plot_idx = 0
    for batch_x, batch_gt in zip(
        batchify(raw_values, batch_size=batch_size),
        batchify(raw_gt, batch_size=batch_size)
    ):

        mask = batch_x.isnan()

        _, batch_q = model.impute(
            inputs         = batch_x,
            batch_size     = len(batch_x),
            device         = torch.device( device_map ),
            denormalize    = False,
            squeeze_output = False
        )
        assert isinstance(batch_q, torch.Tensor) # b c t q
        assert batch_q.ndim == 4
        batch_q = batch_q.squeeze(1) # b t q

        # normalize gt:
        batch_gt = model.scaler.transform(batch_gt.to(device_map))

        gluonts_metrics = update_gluonts_metrics(
            ytrue           = batch_gt.to(device_map),
            yhat            = batch_q.to(device_map),
            evaluators      = gluonts_metrics,
            is_target_mask  = mask.to(device_map)
        )

        # make inference plot:
        if make_plots and plot_idx in all_plots_idx:

            n_iters = 1 if nb_batches > max_nb_plots else max_nb_plots

            for _ in range(n_iters):

                sample_idx = int( np.random.permutation(len(batch_q))[0] )
                y_ctx = batch_x[sample_idx].squeeze()
                mean = torch.nan_to_num(
                    torch.nanmean(y_ctx, keepdim=True), nan=0.0
                )
                scale= torch.nan_to_num(
                    (y_ctx - mean).square().nanmean(keepdim=True).sqrt(), nan=1.0
                )
                std = torch.where(scale==0, 1e-5, scale)
                fig, ax = plot_sample_imputation(
                    quantiles           = batch_q[sample_idx],
                    y_ctx               = (y_ctx-mean)/std,
                    y_true              = batch_gt[sample_idx].squeeze(),
                    quantile_levels     = quantile_levels,
                    show_context_points = True,
                    plot_iqr            = True,
                    z_normalize         = False,
                    iqr_bands           = ( (0.3, 0.7), (0.1, 0.9) ),
                    is_blockwise        = "block" in str(output_dir)
                )
                fig.tight_layout()
                fig.savefig(plot_path / f"sample_{plot_idx}_{sample_idx}.pdf", dpi=150)
                plt.close()
        
            plot_idx +=1

        nb_windows += len(batch_q)
    
    t1 = time()

    # save metrics:
    names = list( gluonts_metrics.keys() )
    metrics = pd.DataFrame( {
        "Model"  : ["TS-ICL"],
        "Chunks" : [nb_windows]
    } | {
        name:[gluonts_metrics[name].get().mean()] for name in names
    } | {
        "Time (s)" : [t1 - t0]
    } )
    metrics.set_index("Model", inplace=True)
    metrics.to_csv( Path(output_dir) / "gluonts_metrics.csv" )


    return

# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path= "config", config_name="fm_impute")
def main(cfg: DictConfig):

    load_dotenv()

    # 
    this_file_dir = os.path.dirname(__file__)

    # data storage path:
    fm_repo = Path( os.environ.get("FM_IMPUTE_REPO", "not_found") )
    assert fm_repo.exists()

    # get yaml:
    config_path = Path( this_file_dir ) / "config" / "tasks.yaml"
    assert config_path.exists()

    # get list of tasks:
    with open(config_path) as file:
        list_tasks = DictConfig( yaml.safe_load(file) )["tasks"]
    num_tasks  = getattr(cfg, "num_tasks", len(list_tasks))
    list_tasks = list_tasks[:num_tasks]

    list_imputation_tasks = [
        "pointwise_missing_1.pt",
        "pointwise_missing_2.pt",
        "blocks_missing_1.pt",
        "blocks_missing_2.pt"
    ]

    # output path:
    expe_path = Path( this_file_dir ) / "results" / "univariate"

    # loop through datasets:
    for task in list_tasks:

        # loop through imputation settings:
        for imputation_task in list_imputation_tasks:

            # 1/2 PREPARE DATA (LOAD, PREPROCESS, BUILD DATALOADERS)

            path_test = Path(fm_repo) / task["dataset_path"] / imputation_task
            assert path_test.exists()
            
            path_gt = Path(fm_repo) / task["dataset_path"] / "ground_truth.pt"
            assert path_gt.exists()

            # get sampling freq:
            sampling_freq = task["freq"]

            # get grid length:
            max_context_length = cfg.context_length
            
            # extract raw values:
            raw_values = load_ts_data( path_data = path_test ) # [N, T, 1]
            raw_gt     = load_ts_data( path_data = path_gt )   # [N, T, 1]
            assert len(raw_values) == len(raw_gt)

            num_samples, raw_grid_len, nvars = raw_values.size()

            window_len = max_context_length

            # handle imputation grids > 4096:
            if raw_grid_len > window_len:
                raw_values = rearrange(raw_values, "b (t s) 1 -> (b s) t 1", s=2)
                raw_gt     = rearrange(raw_gt, "b (t s) 1 -> (b s) t 1", s=2)

            # properly slice ground truth:
            assert isinstance(raw_gt, torch.Tensor)
            if window_len >= raw_grid_len:
                raw_gt = raw_gt[:,-window_len:]
            
            batch_size = 1 if sampling_freq == "5T" else 16 if (sampling_freq == "10T" or sampling_freq == "15T") else cfg.batch_size
            
            # 2/2 INFERENCE LOOP

            # get inference setting name:
            setting_name = Path(imputation_task).stem

            # prepare local output path:
            path_inference_setting = expe_path / task["name"] / setting_name
            path_inference_setting.mkdir(parents=True, exist_ok=True)
            
            # inference:
            run_tsicl_experiment(
                raw_values      = raw_values,
                raw_gt          = raw_gt,
                model_path      = Path( os.environ.get("TSICL_PATH", "not_found") ),
                output_dir      = path_inference_setting,
                batch_size      = batch_size,
                quantile_levels = getattr(cfg, "quantile_levels", None),
                make_plots      = getattr(cfg, "make_plots", False),
                max_nb_plots    = getattr(cfg, "nb_plots", 3)
            )

            # export results:
            all_metrics = make_table(
                output_path      = expe_path,
                list_settings    = "imputation",
                metrics_filename = "gluonts_metrics.csv"
            )
            all_metrics.to_csv( expe_path / "fmimpute_tsicl_univar.csv" )
    
    return


if __name__ == "__main__":
    main()

