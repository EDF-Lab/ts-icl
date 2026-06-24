"""
Dataset builder utilities for converting time series data to Arrow format.

Reference: https://github.com/SalesforceAIResearch/uni2ts/blob/main/example/prepare_data.ipynb

Arrow Dataset Schema:
- item_id: str - Unique identifier for the time series
- start: timestamp[s] - Start timestamp
- freq: str - Time frequency (e.g., "5T", "15T", "D")
- target: list[list[float]] - Shape [num_dims, seq_len]
- variate_names: list[str] - List of variate names (multivariate mode only)
- past_feat_dynamic_real: list[list[float]] - Shape [num_features, seq_len] (optional)
"""

import argparse
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value


def dataframes_to_generator(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    freq: Optional[str] = None,
    to_univariate: bool = False,
    num_dims: Optional[int] = None,
    include_past_feat: bool = False,
    num_past_feat: int = 0,
    csv_names: Optional[List[str]] = None,
) -> tuple[Callable[[], Generator[dict[str, Any], None, None]], Features]:
    """
    Convert pandas DataFrames to a HuggingFace-compatible generator and schema.

    Parameters
    ----------
    dfs : Union[pd.DataFrame, List[pd.DataFrame]]
        One or a list of DataFrames. Each DataFrame must have a datetime index
        (or first column as timestamp) and one or more value columns.
    freq : str, optional
        If provided, this frequency string is used directly for all items.
        If None, the frequency is inferred from the datetime index using pd.infer_freq;
        if inference fails, a ValueError is raised.
    to_univariate : bool
        If True: each column of each DataFrame becomes a separate item (UTS mode).
            - Multiple CSVs + 1 variate: item_id = csv_name
            - 1 CSV + multiple variates: item_id = variate_name
            - Multiple CSVs + multiple variates: item_id = "{csv_name}_{variate_name}"
        If False: each DataFrame becomes a single multivariate item (MTS mode).
            - item_id = csv_name
            - Includes variate_names field with column names
    num_dims : int, optional
        Fixed number of dimensions for target (MTS mode only).
        If None, inferred from the number of columns in the first DataFrame.
    include_past_feat : bool
        Whether to include past_feat_dynamic_real field in the schema (MTS mode only).
        If True, the field will be initialized with NaN values.
    num_past_feat : int
        Number of past dynamic features (only used if include_past_feat=True).
        Determines the length of the past_feat_dynamic_real sequence.
    csv_names : List[str], optional
        List of CSV file names (without extension) used for item_id generation.
        Must have the same length as dfs. If None, item_id will be generated
        using default naming logic.

    Returns
    -------
    tuple[Callable[[], Generator[dict[str, Any], None, None]], Features]
        A generator function that yields dictionaries with keys:
        - item_id: str - Unique identifier for the time series
        - start: timestamp[s] - Start timestamp of the series
        - freq: str - Time frequency string
        - target: list[float] (UTS) or list[list[float]] (MTS) - Time series data
        - variate_names: list[str] (MTS only) - Names of variates
        - past_feat_dynamic_real: list[list[float]] (MTS, optional) - Past features
        And the corresponding HuggingFace Features schema.
    """
    # Normalize input to list
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    # Infer num_dims from first DataFrame if not provided
    if num_dims is None and not to_univariate:
        first_df = dfs[0]
        if isinstance(first_df.index, pd.DatetimeIndex):
            num_dims = len(first_df.columns)
        else:
            num_dims = len(first_df.columns) - 1  # exclude timestamp column

    # Prepare schema
    if to_univariate:
        features = Features(dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(Value("float32")),
        ))
    else:
        feature_dict = dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(
                feature=Sequence(Value("float32")),
                length=num_dims
            ),
            variate_names=Sequence(Value("string"), length=num_dims),
        )
        if include_past_feat and num_past_feat > 0:
            feature_dict["past_feat_dynamic_real"] = Sequence(
                feature=Sequence(Value("float32")),
                length=num_past_feat
            )
        features = Features(feature_dict)

    # Determine naming strategy based on number of CSVs and variates
    num_csvs = len(dfs)
    # Check number of variates from first DataFrame (before processing)
    # We need to peek at the first DataFrame to determine variate count
    first_df_peek = dfs[0]
    if not isinstance(first_df_peek.index, pd.DatetimeIndex):
        # If timestamp is in first column, exclude it
        num_variates = len(first_df_peek.columns) - 1
    else:
        num_variates = len(first_df_peek.columns)

    # Build generator
    def gen_func() -> Generator[dict[str, Any], None, None]:
        for df_idx, df in enumerate(dfs):
            # Allow timestamp either as index or first column
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                time_col = df.columns[0]
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.sort_values(time_col)
                df = df.set_index(time_col)

            col_names = df.columns.tolist()
            csv_name = csv_names[df_idx]

            # Determine frequency:
            # - if freq is provided, use it directly (higher priority than infer)
            # - otherwise, try to infer; if inference fails, raise an error
            if freq is not None:
                effective_freq = freq
            else:
                inferred = pd.infer_freq(df.index)
                if inferred is None:
                    raise ValueError(
                        "Could not infer frequency from datetime index. "
                        "Please provide freq explicitly."
                    )
                effective_freq = inferred

            if to_univariate:
                # Each variate is one item
                # Naming logic:
                # a) Multiple CSVs + 1 variate: item_id = csv_name
                # b) 1 CSV + multiple variates: item_id = variate_name
                # c) Multiple CSVs + multiple variates: item_id = {csv_name}_{variate_name}
                for i, col in enumerate(col_names):
                    if num_csvs > 1 and num_variates == 1:
                        item_id = csv_name
                    elif num_csvs == 1:
                        item_id = col
                    else:
                        item_id = f"{csv_name}_{col}"

                    item = {
                        "item_id": item_id,
                        "start": df.index[0],
                        "freq": effective_freq,
                        "target": df[col].to_numpy(dtype=np.float32),
                    }
                    yield item
            else:
                # Each DataFrame is one multivariate item
                # item_id = csv_name
                target = df.to_numpy(dtype=np.float32).T  # shape: (var, time)
                item = {
                    "item_id": csv_name,
                    "start": df.index[0],
                    "freq": effective_freq,
                    "target": target,
                    "variate_names": col_names,
                }
                if include_past_feat and num_past_feat > 0:
                    seq_len = target.shape[1]
                    item["past_feat_dynamic_real"] = [
                        [np.nan] * seq_len for _ in range(num_past_feat)
                    ]
                yield item

    return gen_func, features


def build_dataset_from_csvs(
    csv_dir: Union[str, Path],
    output_path: Union[str, Path],
    pattern: str = "*.csv",
    freq: Optional[str] = None,
    to_univariate: bool = False,
    # item_prefix: str = "",
    include_past_feat: bool = False,
    num_past_feat: int = 0,
) -> Dataset:
    """
    Build Arrow dataset from a directory of CSV files.

    Parameters
    ----------
    csv_dir : str or Path
        Directory containing CSV files.
    output_path : str or Path
        Where to save the resulting dataset.
    pattern : str
        Glob pattern to match CSV files.
    freq : str, optional
        Time frequency. If provided, overrides automatic frequency inference.
        If None, the frequency is inferred from the datetime index; if
        inference fails, a ValueError is raised.
    to_univariate : bool
        Convert to univariate mode.
    include_past_feat : bool
        Include past_feat_dynamic_real field.
    num_past_feat : int
        Number of past features.

    Returns
    -------
    Dataset
        The created HuggingFace Dataset.
    """
    csv_dir = Path(csv_dir)
    output_path = Path(output_path)

    csv_paths = sorted(csv_dir.glob(pattern))
    if not csv_paths:
        raise ValueError(f"No CSV files found matching pattern '{pattern}' in {csv_dir}")

    print(f"Found {len(csv_paths)} CSV files")

    # Load all DataFrames
    dfs = [pd.read_csv(path, parse_dates=[0]) for path in csv_paths]

    # Extract CSV names (without extension) for item_id naming
    csv_names = [Path(path).stem for path in csv_paths]

    # Print statistics
    lengths = [len(df) for df in dfs]
    print(f"Time series lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}")

    # Create generator and features
    gen_func, features = dataframes_to_generator(
        dfs=dfs,
        freq=freq,
        to_univariate=to_univariate,
        include_past_feat=include_past_feat,
        num_past_feat=num_past_feat,
        csv_names=csv_names,
    )

    # Create and save dataset
    hf_dataset = Dataset.from_generator(gen_func, features=features)
    hf_dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")
    print(f"  - Number of series: {len(hf_dataset)}")
    print(f"  - Features: {list(hf_dataset.features.keys())}")

    return hf_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a HuggingFace Arrow dataset from a directory of CSV time series."
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        required=True,
        help="Directory containing input CSV files.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path (directory) where the dataset will be saved.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern to match CSV files (default: '*.csv').",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default=None,
        help=(
            "Time frequency string, e.g. 'D', '15T'. "
            "If omitted, the frequency will be inferred from the datetime index; "
            "if inference fails, an error is raised."
        ),
    )
    parser.add_argument(
        "--to-univariate",
        action="store_true",
        help="Convert each column to a separate univariate series (UTS mode).",
    )

    args = parser.parse_args()

    build_dataset_from_csvs(
        csv_dir=args.csv_dir,
        output_path=args.output_path,
        pattern=args.pattern,
        freq=args.freq,
        to_univariate=args.to_univariate,
    )
