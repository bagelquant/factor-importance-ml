"""
Defines data loading utilities for the project.

Input:

  - file_path: Path to the data file. 
Output:

  - df: A pandas DataFrame with:
    - multi-index: ["date", "permno"]
    - columns: ["feature1", "feature2", ..., "featureN", "target"]
"""

import json
from pathlib import Path

import pandas as pd

from .preprocessor import preprocess


# config path locate in root directory
CONFIG_PATH = Path(__file__).parent.parent / "configs.json"


def _get_raw_data_path() -> Path:
    """Get the path to the raw data directory."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)["file_path"]
    raw_data_path = Path(config["raw_data"])
    return raw_data_path


def load_raw(file_path: Path=_get_raw_data_path()) -> pd.DataFrame:
    """Load raw data from the specified file path.

    Parameters
    ----------
    file_path : Path
        Path to the data file.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with:
            - multi-index: ["date", "permno"]
            - columns: ["feature1", "feature2", ..., "featureN", "target"]
    """
    df = pd.read_parquet(file_path)

    # rename DateYM to date for consistency
    df = df.rename(columns={"DateYM": "date"})
    df = df.set_index(["date", "permno"]).sort_index()

    # drop duplicate indices if any
    df = df.loc[~df.index.duplicated(keep="first")]
    return df


def _get_processed_data_path() -> Path:
    """Get the path to the processed data directory."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)["file_path"]
    processed_data_path = Path(config["processed_data"])
    return processed_data_path


def load_processed(file_path: Path = _get_processed_data_path(),
                   reprocesse: bool = False) -> pd.DataFrame:
    """
    Load processed data from the specified file path.
    If not exist or reprocesse is True, load raw data and preprocess it.
    Save the processed data toe the processed data path.

    Parameters
    ----------
    file_path : Path
        Path to the processed data file.
    reprocesse : bool
        Whether to reprocess the data. If True, load raw data and preprocess it.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with:
            - multi-index: ["date", "permno"]
            - columns: ["feature1", "feature2", ..., "featureN", "target_next"]
    """
    if reprocesse or not file_path.exists():
        print("Processed data not found or reprocesse is True. Loading and preprocessing raw data...")
        raw_data_path = _get_raw_data_path()
        df = load_raw(raw_data_path)
        df = preprocess(df)
        # save processed data, make parent directories if not exist
        print("Saving processed data...")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path)
    else:
        df = pd.read_parquet(file_path)
    return df


if __name__ == "__main__":
    from time import perf_counter
    start = perf_counter()

    processed_df = load_processed(_get_processed_data_path())

    print(processed_df.head())
    print(processed_df["retadj_next"].describe())
    end = perf_counter()
    print(f"Data loaded in {end - start:.2f} seconds.")

