from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import gmean

pretty_names = {
    "mean_weighted_sum_quantile_loss" : "WQL",
    "MSE[mean]": "MSE",
    "RMSE[mean]": "RMSE",
    "NRMSE[mean]": "NRMSE",
    "MAE[0.5]": "MAE"
}

def make_table(
    output_path: Path,
    list_datasets: str | Sequence[str] | None = None,
    list_settings: str | Sequence[str] | None = None,
    metrics_filename: str = "metrics/gluonts_metrics.csv",
) -> pd.DataFrame:
    """
    Scan `output_path` to read the csv files from `inference_mixture.py` and export to an aggregated file.
    """

    if list_datasets is None:
        list_datasets = [f.stem for f in Path(output_path).iterdir() if f.is_dir() and f.stem != ".hydra"]
    elif isinstance(list_datasets, str):
        list_datasets = [list_datasets]

    if (list_settings is None) or (list_settings == "imputation"):
        list_settings = ["pointwise_missing_1", "pointwise_missing_2", "blocks_missing_1", "blocks_missing_2"]
    elif isinstance(list_settings, str):
        list_settings = [list_settings]

    all_metrics = []
    for metric_name in ["MAE[0.5]", "MSE[mean]", "mean_weighted_sum_quantile_loss"]:
        
        results = defaultdict(list)
        list_dataset = []
        list_settting = []

        for dataset in list_datasets:

            for setting in list_settings:

                filename = output_path / dataset / setting / metrics_filename

                if filename.exists():
                    list_dataset.append(dataset)
                    list_settting.append(setting)
                    df = pd.read_csv(filename, index_col = 0)
                    results["nb_chunks"].append(df["Chunks"].iloc[0] if "Chunks" in df.columns else -1)
                    results["Time (s)"].append(df["Time (s)"].iloc[0] if "Time (s)" in df.columns else -1)
                    assert len(df) == 1
                    for key in df.index:
                        if ~df.loc[key].isna().any():
                            assert metric_name in df.loc[key].keys()
                            col_key = metric_name
                            results[pretty_names[col_key]].append(df.loc[key][col_key])
                    
        if len(list_dataset) == 0:
            return pd.DataFrame()
        
        df = pd.concat([pd.DataFrame({"Dataset": list_dataset, "Setting": list_settting}), pd.DataFrame(results)], axis=1)
        df = df.sort_values(by = ["Dataset", "Setting"], ascending=True, ignore_index=True, key=lambda col: col.str.lower())

        is_int_cols   = df.dtypes[(df.dtypes == "int64")].index    
        is_float_cols = df.dtypes[(df.dtypes == "float64")].index

        mean_scores  = df[is_float_cols].apply(np.mean, axis=0)
        gmean_scores = df[is_float_cols].apply(gmean, axis=0)

        df.loc["gmean", is_float_cols] = gmean_scores
        df.loc["mean", is_float_cols]  = mean_scores
        df.loc["mean", is_int_cols] = df[is_int_cols].apply(np.sum, axis=0)

        all_metrics.append(df if metric_name == "MAE[0.5]" else df.iloc[:,-1])
    
    return pd.concat(all_metrics, axis=1)
