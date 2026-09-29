import json
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
import pyarrow.parquet as pq

# Define target features to inspect
TARGET_FEATURES = ["vspd_fps", "p_dot", "q_dot", "r_dot"]

def extract_parquet_config(parquet_path: Path) -> dict:
    """Reads embedded pipeline_config metadata from Parquet header."""
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        schema_meta = parquet_file.schema.to_arrow_schema().metadata or {}
        file_meta = parquet_file.metadata.metadata if (parquet_file.metadata and parquet_file.metadata.metadata) else {}
        combined = {**file_meta, **schema_meta}
        
        if b"pipeline_config" in combined:
            return json.loads(combined[b"pipeline_config"].decode("utf-8"))
    except Exception:
        pass
    return {}

def evaluate_sweep_parquet_files(database_dir: Path) -> pd.DataFrame:
    """Scans all sweep Parquet files in directory and compiles kurtosis and noise metrics."""
    parquet_files = sorted(list(database_dir.glob("four_class_derived_dataset_savgol_*.parquet")))
    
    records = []
    for filepath in parquet_files:
        config = extract_parquet_config(filepath)
        n_state = config.get("savgol_state_window", "Unknown")
        n_body = config.get("savgol_body_rate_window", "Unknown")
        
        # Load only target derived feature columns to optimize I/O
        df = pd.read_parquet(filepath, columns=TARGET_FEATURES)
        
        row_entry = {
            "file_name": filepath.name,
            "N_state": n_state,
            "N_body": n_body,
        }
        
        for col in TARGET_FEATURES:
            # Flatten 1D array column if stored as list/array per row, or standard Series
            series = pd.Series(np.concatenate(df[col].values)) if isinstance(df[col].iloc[0], (np.ndarray, list)) else df[col]
            series = series.dropna()
            
            # Compute excess kurtosis (3.0 = Normal) and peak magnitude
            kurt_val = stats.kurtosis(series, fisher=False)
            max_abs = np.max(np.abs(series))
            
            row_entry[f"{col}_kurtosis"] = round(kurt_val, 2)
            row_entry[f"{col}_max_abs"] = round(max_abs, 4)
            
        records.append(row_entry)
        
    results_df = pd.DataFrame(records)
    return results_df.sort_values(by=["N_state", "N_body"]).reset_index(drop=True)