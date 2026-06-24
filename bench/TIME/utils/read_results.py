
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gmean


def export_metrics(
    output_path: Path,
    model_name: str | None = None
) -> None:
    
    # get model name:
    model_name = model_name or output_path.stem


    # agg all csv files into one dataframe:
    list_datasets = [f.stem for f in output_path.iterdir() if f.is_dir() and f.stem != '.hydra']
    df = []
    for dataset in list_datasets:

        list_freqs = [f.stem for f in (output_path / dataset).iterdir()]

        for freq in list_freqs:

            for term in ['short', 'medium', 'long']:

                summary = output_path / dataset / freq / term / 'summary.csv'
                metrics_npz =  output_path / dataset / freq / term / 'metrics.npz'

                if summary.exists():
                    df.append(pd.read_csv(summary, index_col=0))
                
                elif metrics_npz.exists():
                    metrics = np.load(metrics_npz)
                    with open(output_path / dataset / freq / term / 'config.json', 'r') as f:
                        config  = json.load(f)
                    for metric_name, metric_values in metrics.items():
                        mean_val = np.nanmean(metric_values)
                        print(f"      {metric_name}: {mean_val:.4f}")
                    
                    metrics_summary = {
                        'Dataset' : config['dataset_config'],
                        'nb_chunks' : int( config['num_series'] * config['num_windows'] * config['num_variates'] )
                    } | {
                        metric_name: [np.nanmean(metric_values)] for metric_name, metric_values in metrics.items()
                    }
                    df.append(pd.DataFrame(metrics_summary))
                else:
                    for scenario in ['scenario_pointwise_missing_1', 'scenario_pointwise_missing_2', 'scenario_blocks_missing_1', 'scenario_blocks_missing_2']:

                        summary = output_path / dataset / freq / term / scenario / 'summary.csv'
                        if summary.exists():
                            df_i = pd.read_csv(summary, index_col=0)
                            # df_i.insert(1, "Setting",scenario)
                            df.append(df_i)

    df = pd.concat(df, axis=0)
    sort_by = ['Dataset', 'scenario'] if 'scenario' in df.columns else ['Dataset']
    df = df.sort_values(by = sort_by, ascending=True, ignore_index=False, key=lambda col: col.str.lower())
    # df = df.sort_values(by = ['Dataset'], ascending=True, ignore_index=False, key=lambda col: col.str.lower())

    is_int_cols   = df.dtypes[(df.dtypes == 'int64')].index
    is_float_cols = df.dtypes[(df.dtypes == 'float64')].index
    
    # compute mean and geometric mean:
    mean_scores  = df[is_float_cols].apply(np.mean, axis=0)
    gmean_scores = df[is_float_cols].apply(gmean, axis=0)
    
    df.loc['gmean', is_float_cols] = gmean_scores
    df.loc['mean', is_float_cols]  = mean_scores
    df.loc['mean', is_int_cols] = df[is_int_cols].apply(np.sum, axis=0)

    df.reset_index(inplace=True)
    
    # export:
    df.to_csv( output_path / '{}.csv'.format(model_name) )