from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange


def load_ts_data(
    path_data: Path
) -> torch.Tensor:
    """
    Load univariate or multivariate time series data. 
    
    Args:
        path_data (Path): path to data to load
        
    Returns:
        torch.Tensor of shape `[N, T, c]`
    """

    # get file extension:
    file_extension = path_data.suffix

    # load datasets:
    if file_extension == ".csv":

        df_X   = pd.read_csv(path_data, index_col=0)
        load_X = torch.Tensor(df_X.values).to(torch.float32)
    
    elif file_extension == ".tsv":

        df_X   = pd.read_csv(path_data, sep="\t", index_col=0)
        load_X = torch.Tensor(df_X.values).to(torch.float32)
        
    elif file_extension == ".npy":
        
        np_X   = np.load(path_data)
        load_X = torch.Tensor(np_X).to(torch.float32)
    
    elif file_extension == ".dat":

        load_X = np.memmap(path_data, dtype="float32", mode="r")
        load_X = torch.Tensor(load_X).to(torch.float32)
    
    elif file_extension == ".pt":            
        
        load_X = torch.load(path_data).to(torch.float32)  
    
    elif len(file_extension) == 0:

        load_X = torch.load(path_data.with_suffix(".pt")).to(torch.float32)  

    else:

        raise NotImplementedError("Data format should be one of: `.pt`, `.csv`, `.tsv`, `.npy`, `.dat`")
    
    if load_X.ndim == 2:

        load_X = load_X.unsqueeze(-1)

    elif load_X.ndim == 3:

        if load_X.shape[1] < load_X.shape[2]:
            load_X = rearrange(load_X, "n c T -> n T c")

    assert load_X.ndim == 3

    return load_X  # [N, T, c]
