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
import numpy as np
from joblib import Parallel, delayed

from .preprocessor import preprocess


__all__ = ["load_raw", "load_processed", "load_train_ready_data", "add_peer_based_features", "load_train_ready_data_no_peer"]

# Locate `configs.json` by searching upward from this module so a repo-root
# `configs.json` (e.g. `/.../ml_project/configs.json`) is found even when the
# package lives under `src/`.
CURRENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = None
for p in (CURRENT_DIR, *CURRENT_DIR.parents):
    candidate = p / "configs.json"
    if candidate.exists():
        CONFIG_PATH = candidate
        break

if CONFIG_PATH is None:
    raise FileNotFoundError(
        f"configs.json not found when searching from {CURRENT_DIR}.\n"
        "Please place a `configs.json` in the repository root or set the "
        "environment to locate it."
    )


def _get_raw_data_path() -> Path:
    """Get the path to the raw data directory."""
    with open(CONFIG_PATH, "r") as f:  # type: ignore
        config = json.load(f)["file_path"]
    raw_data_path = Path(config["raw_data"])
    return raw_data_path


def _get_data_start() -> int:
    """Get the whole process start year, for all iterations."""
    with open(CONFIG_PATH, "r") as f:  # type: ignore
        config = json.load(f)["train_iteration"]
    train_start = config["data_start"]
    return train_start


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

    # select data from data_start year onwards
    data_start = _get_data_start()
    df = df.loc[df.index.get_level_values("date").year >= data_start]  # type: ignore
    return df


def _get_processed_data_path() -> Path:
    """Get the path to the processed data directory."""
    with open(CONFIG_PATH, "r") as f:  # type: ignore
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


def _cat_embeddings(
    base_data: pd.DataFrame,
    embeddings: pd.DataFrame
) -> pd.DataFrame:
    """Concatenate embeddings to base data on 'permno'."""
    base_data = base_data.copy()
    base_data = base_data.reset_index().merge(
        embeddings.reset_index(),
        on="permno",
        how="left"
    ).set_index(["date", "permno"])
    return base_data


def _compute_date_peer_features(
    index: pd.Index,
    X: np.ndarray,
    Z: np.ndarray,
    base_cols: list[str],
    suffix: str
) -> pd.DataFrame | None:
    """
    Compute peer features for a single date block.
    
    Args:
        index: The index (MultiIndex) for the resulting block.
        X: (N, K) Feature matrix for the date.
        Z: (N, D) Embedding matrix for the date.
        base_cols: List of feature names.
        suffix: Suffix for new columns.
        
    Returns:
        DataFrame with peer features or None if invalid.
    """
    N = Z.shape[0]
    
    # Use float32 for speed and memory efficiency
    Z = Z.astype(np.float32, copy=False)
    X = X.astype(np.float32, copy=False)

    # Pairwise squared Euclidean distances via (z z^T) trick
    # (N, 1)
    zz = np.sum(Z * Z, axis=1, keepdims=True)
    # (N, N)
    # sq_dists = zz + zz.T - 2.0 * (Z @ Z.T)
    # We calculate this into a new buffer, later reused for weights
    sq_dists = np.matmul(Z, Z.T)
    sq_dists *= -2.0
    sq_dists += zz
    sq_dists += zz.T
    
    # Numerical stability
    np.maximum(sq_dists, 0.0, out=sq_dists)
    
    # Bandwidth Calculation using Subsampling
    # Computing median of N^2 distances is too slow for large N.
    # We sample a subset of rows to estimate the median distance.
    SAMPLE_SIZE = 1000
    bw_sq = 1.0
    
    if N <= SAMPLE_SIZE:
        # Use full matrix if small
        triu_idx = np.triu_indices_from(sq_dists, k=1)
        nonzero_sq_d = sq_dists[triu_idx]
        # Filter small noise
        nonzero_sq_d = nonzero_sq_d[nonzero_sq_d > 1e-8]
        if nonzero_sq_d.size > 0:
            bw_sq = float(np.median(nonzero_sq_d))
    else:
        # Subsample rows for estimation
        # We need a new small distance matrix for the sample
        rng = np.random.default_rng(42)
        idx = rng.choice(N, SAMPLE_SIZE, replace=False)
        Z_sample = Z[idx]
        
        # Compute distances for sample
        zz_sample = np.sum(Z_sample * Z_sample, axis=1, keepdims=True)
        d_sample = zz_sample + zz_sample.T - 2.0 * (Z_sample @ Z_sample.T)
        np.maximum(d_sample, 0.0, out=d_sample)
        
        triu_idx = np.triu_indices_from(d_sample, k=1)
        sample_vals = d_sample[triu_idx]
        sample_vals = sample_vals[sample_vals > 1e-8]
        
        if sample_vals.size > 0:
            bw_sq = float(np.median(sample_vals))

    if bw_sq <= 0:
        return None

    # Gaussian similarities w_ij = exp( - d_ij^2 / (2 * bw^2) )
    # Reuse sq_dists memory for weights to save allocation
    # sq_dists currently holds d^2
    
    scale = -0.5 / bw_sq
    sq_dists *= scale
    np.exp(sq_dists, out=sq_dists) # In-place exp

    # Remove self-weights (diagonal was 0 distance -> exp(0)=1)
    np.fill_diagonal(sq_dists, 0.0)

    # Normalize each row to sum to 1
    row_sums = sq_dists.sum(axis=1, keepdims=True)
    
    # Identify valid rows (with at least one peer)
    # float32 precision check
    valid_mask = row_sums.squeeze() > 1e-8
    
    if not np.any(valid_mask):
        return None
        
    # Safe division in-place
    # Set zero sums to 1 to avoid NaN/Inf during division (result will be masked anyway)
    row_sums[row_sums == 0] = 1.0
    sq_dists /= row_sums

    # Compute peer-weighted features
    # peer_X = W_norm @ X
    peer_X = sq_dists @ X  # (N, K)
    
    # Mask invalid rows
    if not np.all(valid_mask):
         peer_X[~valid_mask, :] = np.nan

    return pd.DataFrame(
        peer_X,
        index=index,
        columns=[c + suffix for c in base_cols]
    )


def add_peer_based_features(
    data: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """Add peer-based features for numeric columns.

    For each date and firm, construct Gaussian-kernel peer weights in
    embedding space and use them to form peer-weighted averages.
    
    Assumes:
    - data has MultiIndex ["date", "permno"]
    - embeddings has either MultiIndex ["date", "permno"] or Index ["permno"]
    - embeddings and data are reasonably aligned.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with MultiIndex ["date", "permno"].
    embeddings : pd.DataFrame
        Embeddings DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of ``data`` with additional peer-based feature columns (suffixed with '_peer').
    """
    suffix = "_peer"
    
    # Pre-select base columns
    # Logic:
    # 1. Ignore target "retadj_next"
    # 2. Ignore categorical encoding: In preprocessor.py, categorical dummies are cast to int8,
    #    while continuous features are rank-transformed floats (float32).
    #    So we select only float columns to capture continuous features and exclude dummies.
    
    # Get all float columns (includes float32, float64)
    base_cols = list(data.select_dtypes(include=["float"]).columns)
    
    # Explicitly remove target if present
    if "retadj_next" in base_cols:
        base_cols.remove("retadj_next")
        
    if not base_cols:
        return data.copy()

    # Check embedding structure
    names = embeddings.index.names
    is_time_varying = "date" in names and "permno" in names
    
    # Sort data by date to ensure groupby is efficient
    # (Assuming users pass sorted data, but sorting index is fast enough to be safe)
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()

    # Prepare tasks for parallel execution
    tasks = []
    
    # Iterate over dates
    # We use groupby(level="date") which is efficient
    for date, group_df in data.groupby(level="date"):
        if len(group_df) <= 1:
            continue
            
        # 1. Get Embeddings Z for this date
        Z = None
        if is_time_varying:
            try:
                Z = embeddings.loc[date].to_numpy(dtype="float64")
            except KeyError:
                continue
        else:
            # Static embeddings
            current_permnos = group_df.index.get_level_values("permno")
            # Use reindex to handle missing keys safely; missing become NaN
            Z_df = embeddings.reindex(current_permnos)
            
            # Check for missing values (rows that became all NaNs or contained NaNs)
            # Assuming embeddings are dense, any NaN implies missing permno or corrupted embedding
            if Z_df.isna().any().any():
                 Z_df = Z_df.dropna()
                 if Z_df.empty:
                     continue
                 
                 # We have a subset of valid embeddings
                 # We must align group_df to this subset
                 valid_permnos = Z_df.index
                 if len(valid_permnos) <= 1:
                     continue
                     
                 # Align group_df
                 # group_df index is (date, permno). We select matching permnos.
                 # Using boolean mask is often safer/faster than .loc with tuples on MultiIndex
                 mask = group_df.index.get_level_values("permno").isin(valid_permnos)
                 group_df = group_df[mask]
                 
                 Z = Z_df.to_numpy(dtype="float64")
            else:
                 # All present
                 Z = Z_df.to_numpy(dtype="float64")
        
        # Check shapes
        if Z is None or Z.shape[0] != len(group_df):
             # Handle time-varying mismatch
             if is_time_varying and Z is not None:
                  permnos = group_df.index.get_level_values("permno")
                  # re-fetch aligned
                  try:
                      Z_aligned = embeddings.loc[date].reindex(permnos).to_numpy(dtype="float64")
                      # drop nans
                      mask = ~np.isnan(Z_aligned).any(axis=1)
                      if not mask.any(): continue
                      Z = Z_aligned[mask]
                      group_df = group_df.iloc[mask]
                  except KeyError:
                      continue
             elif not is_time_varying:
                 # Should have been handled above
                 continue
                 
        if len(group_df) <= 1:
            continue
            
        # 2. Extract X
        X = group_df[base_cols].to_numpy(dtype="float64")
        
        # 3. Schedule task
        tasks.append(delayed(_compute_date_peer_features)(
            group_df.index,
            X,
            Z,
            base_cols,
            suffix
        ))

    # Execute in parallel
    # prefer="threads" is good for numpy heavy workloads
    if tasks:
        results = Parallel(n_jobs=-1, prefer="threads")(tasks)
        # Filter None results
        peer_blocks = [res for res in results if res is not None]
        
        if peer_blocks:
            peer_df = pd.concat(peer_blocks)
            data = data.join(peer_df)

    return data
    

def load_train_ready_data(test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data set ready for training for a specific year.
    
    Output one: dataset with train only embeddings (used for validation)
    Output two: dataset with train + val embeddings (used for final training + prediction)
    """
    preprocessed_data = load_processed()
    # slice date index from data_start to test_year
    preprocessed_data = preprocessed_data.loc[
            preprocessed_data.index.get_level_values("date").year <= test_year  # type: ignore
    ]

    path_train_embed = Path(f"data/embeddings/train_embeddings_{test_year}.parquet.gzip")
    train_embeddings = pd.read_parquet(path_train_embed)

    # add peer-based features to preprocessed data using train embeddings
    train_with_peer = add_peer_based_features(
        data=preprocessed_data,
        embeddings=train_embeddings
    )
    train_embeddings = _cat_embeddings(
        base_data=train_with_peer,
        embeddings=train_embeddings
    )

    path_train_val_embed = Path(f"data/embeddings/train_val_embeddings_{test_year}.parquet.gzip")
    train_plus_val_embeddings = pd.read_parquet(path_train_val_embed)

    # add peer-based features to preprocessed data using train + val embeddings
    train_val_with_peer = add_peer_based_features(
        data=preprocessed_data,
        embeddings=train_plus_val_embeddings
    )
    train_plus_val_embeddings = _cat_embeddings(
        base_data=train_val_with_peer,
        embeddings=train_plus_val_embeddings
    )

    return train_embeddings, train_plus_val_embeddings
                          

def load_train_ready_data_no_peer(test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data set ready for training for a specific year (No Peer Features).
    
    Output one: dataset with train only embeddings (used for validation)
    Output two: dataset with train + val embeddings (used for final training + prediction)
    """
    preprocessed_data = load_processed()
    # slice date index from data_start to test_year
    preprocessed_data = preprocessed_data.loc[
            preprocessed_data.index.get_level_values("date").year <= test_year  # type: ignore
    ]

    path_train_embed = Path(f"data/embeddings/train_embeddings_{test_year}.parquet.gzip")
    train_embeddings = pd.read_parquet(path_train_embed)

    # concatenate embeddings to preprocessed data
    train_embeddings = _cat_embeddings(
        base_data=preprocessed_data,
        embeddings=train_embeddings
    )

    path_train_val_embed = Path(f"data/embeddings/train_val_embeddings_{test_year}.parquet.gzip")
    train_plus_val_embeddings = pd.read_parquet(path_train_val_embed)

    # concatenate embeddings to preprocessed data
    train_plus_val_embeddings = _cat_embeddings(
        base_data=preprocessed_data,
        embeddings=train_plus_val_embeddings
    )

    return train_embeddings, train_plus_val_embeddings


if __name__ == "__main__":
    from time import perf_counter
    start = perf_counter()

    data_with_train_embeddings, data_with_train_val_embeddings = load_train_ready_data(test_year=2009)

    print(data_with_train_embeddings.head())
    print(data_with_train_val_embeddings.head())
    end = perf_counter()
    print(f"Data loaded in {end - start:.2f} seconds.")
