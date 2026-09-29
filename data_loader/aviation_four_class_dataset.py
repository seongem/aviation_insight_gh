import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Subset

logger = logging.getLogger("AviationInsight")

# Class label map for downstream metrics / visualization
CLASS_NAMES = {
    0: "Nominal",
    1: "Speed High",
    2: "Path High",
    3: "Flaps Late Setting",
}


class AviationFourClassDataset(Dataset):
    """
    PyTorch Dataset wrapper reading directly from the consolidated 31-channel Parquet 
    dataset generated in database_dir. Fully aligns with LoadEnvironment structure.
    """

    def __init__(
        self,
        manifest_dir: str | Path,
        database_dir: str | Path,
        stats_path: str | Path,
        split: str = "train",
        feature_set: str = "enabled_piml",
        logo_mask_col: str | None = None,
        window_size: int = 160,
        stride: int = 160,
        phase_filter: int | None = 6,
        parquet_filename: str = "four_class_derived_dataset.parquet",
        sigma_clip: float | None = 5.0,
    ):
        self.manifest_dir = Path(manifest_dir)
        self.database_dir = Path(database_dir)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.phase_filter = phase_filter
        self.sigma_clip = sigma_clip

        # 1. Load Feature Registry & Determine Active Channels via features.csv
        self.features_df = pd.read_csv(self.manifest_dir / "features.csv")

        active_mask = self.features_df[feature_set] == True
        if logo_mask_col and logo_mask_col in self.features_df.columns:
            active_mask = active_mask & (self.features_df[logo_mask_col] == True)

        self.active_features_df = self.features_df[active_mask].reset_index(drop=True)
        self.feature_names = self.active_features_df["feature_name"].tolist()
        self.num_channels = len(self.feature_names)

        # 2. Read Consolidated Parquet Dataset & Flight Manifest
        parquet_path = self.database_dir / parquet_filename
        if not parquet_path.exists():
            raise FileNotFoundError(f"Derived Parquet dataset not found at: {parquet_path}")

        logger.info(f"Loading master Parquet dataset from {parquet_path.name}...")
        master_table = pq.read_table(parquet_path)
        master_df = master_table.to_pandas()

        flights_df = pd.read_parquet(self.manifest_dir / "flights.parquet")

        # 3. Join with Flight Manifest to Filter Split ('train', 'val', or 'test')
        merged_meta = master_df.merge(flights_df, on="flight_record", how="inner")
        self.active_instances = merged_meta[merged_meta["split"] == split].reset_index(drop=True)

        # Pre-cache column reference series for efficient __getitem__ access
        self.channel_columns = [self.active_instances[col].tolist() for col in self.feature_names]
        self.labels = self.active_instances["label"].astype(int).tolist()
        self.flight_records = self.active_instances["flight_record"].astype(str).tolist()
        self.data_instances = self.active_instances["data_instance"].astype(int).tolist()

        # 4. Load Normalization Statistics
        self.means, self.stds = self._load_normalization_stats(Path(stats_path))

        logger.info(
            f"✅ AviationFourClassDataset Initialized [{split.upper()}] | "
            f"Active Features: {self.num_channels}/31 | "
            f"Instances: {len(self.active_instances):,} | "
            f"Phase ID: {self.phase_filter} (Approach)"
        )

    def _load_normalization_stats(self, stats_path: Path):
        """Loads mean and std vectors aligned with self.feature_names."""
        if not stats_path.exists():
            logger.warning(f"Stats file {stats_path} not found. Defaulting to Mean=0.0, Std=1.0")
            means = [0.0] * self.num_channels
            stds = [1.0] * self.num_channels
        else:
            with open(stats_path, "r") as f:
                stats_data = json.load(f)

            stats_dict = stats_data.get("sensor_stats", stats_data.get("metadata", {}))
            means, stds = [], []

            for feat in self.feature_names:
                feat_stat = stats_dict.get(feat, {})
                mean_val = feat_stat.get("mean", 0.0)
                std_val = feat_stat.get("std", 1.0)

                if std_val == 0.0 or np.isnan(std_val):
                    std_val = 1.0

                means.append(mean_val)
                stds.append(std_val)

        return (
            torch.tensor(means, dtype=torch.float32).view(1, 1, -1),
            torch.tensor(stds, dtype=torch.float32).view(1, 1, -1),
        )

    def __len__(self):
        return len(self.active_instances)

    def __getitem__(self, idx: int):
        flight_record = self.flight_records[idx]
        array_row_idx = self.data_instances[idx]
        class_label = self.labels[idx]

        # Extract 1D array sequences for each active feature and stack into [160, C]
        sequence_channels = [self.channel_columns[ch_idx][idx] for ch_idx in range(self.num_channels)]
        sequence_matrix = np.column_stack(sequence_channels).astype(np.float32)  # Shape: [160, C]

        # Convert to Tensor [1, T, C]
        tensor_window = torch.from_numpy(sequence_matrix).unsqueeze(0)

        # Apply Z-score normalization: (X - mu) / sigma
        normalized_window = (tensor_window - self.means) / self.stds
        normalized_window = normalized_window.squeeze(0)  # Shape: [T, C]

        # Metadata tuple matching production evaluator
        metadata = {
            "flight_record": flight_record,
            "phase_id": self.phase_filter,
            "data_instance": array_row_idx,
            "label": class_label,
            "class_name": CLASS_NAMES.get(class_label, "Unknown")
        }

        # -------------------------------------------------------------
        # Apply Sigma Clipping to remove extreme hardware/telemetry spikes
        # -------------------------------------------------------------
        if self.sigma_clip is not None:
            normalized_window = torch.clamp(
                normalized_window, 
                min=-self.sigma_clip, 
                max=self.sigma_clip
            )

        return normalized_window, class_label, metadata

    def get_targets(self, target_type: str = "binary") -> np.ndarray:
        """
        Extracts 1D numpy label arrays for evaluation metrics without needing 
        to iterate through a DataLoader.
        
        Args:
            target_type: "binary" (0: Nominal, 1: Anomaly) or "multiclass" (0-3).
            
        Returns:
            np.ndarray: 1D ground truth label array [N,]
        """
        y_multi = np.array(self.labels, dtype=int)

        if target_type == "binary":
            return (y_multi != 0).astype(int)
        elif target_type == "multiclass":
            return y_multi
        else:
            raise ValueError(f"Unknown target_type '{target_type}'. Expected 'binary' or 'multiclass'.")

    def get_nominal_subset(self) -> torch.utils.data.Subset:
        """
        Returns a PyTorch Subset containing strictly nominal instances (label == 0)
        for unsupervised model training (e.g., PatchTST, OmniAnomaly, TS2Vec).
        """
        nominal_indices = [idx for idx, label in enumerate(self.labels) if label == 0]
        return torch.utils.data.Subset(self, nominal_indices)
    
    def get_feature_names(self) -> list[str]:
        """Returns the active list of feature/sensor names aligned with the channel dimensions [C]."""
        return self.feature_names

    def get_metadata(self) -> pd.DataFrame:
        """
        Returns a DataFrame copy of instance-level metadata required for diagnostic reports
        and flight-level tracking in AnomalyFourClassEvaluator.
        """
        meta_df = self.active_instances.copy()
        
        # Ensure fallback column aliases expected by AnomalyFourClassEvaluator
        if "phase" not in meta_df.columns:
            meta_df["phase"] = "approach"
        if "window_start_idx" not in meta_df.columns and "start_sec" not in meta_df.columns:
            meta_df["window_start_idx"] = 0
            
        return meta_df

    def get_balanced_diagnostic_test_set(self) -> torch.utils.data.Subset:
        """
        Returns a 1:1 balanced PyTorch Subset (Nominal vs. Anomaly) for model 
        benchmarking, decoupling evaluation metrics from natural base-rate class imbalance.
        """
        labels = np.array(self.labels)  # Accesses self.labels directly from the dataset instance
        
        # 1. Identify indices for Nominal (0) vs. Anomalies (1, 2, 3)
        nominal_indices = np.where(labels == 0)[0]
        anomaly_indices = np.where(labels != 0)[0]
        
        num_anomalies = len(anomaly_indices)
        
        if num_anomalies == 0:
            logger.warning("No anomaly samples found in dataset split! Returning full dataset.")
            return torch.utils.data.Subset(self, range(len(self)))
        
        # 2. Downsample Nominal indices to match the anomaly count
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducible benchmarks
        balanced_nominal_indices = rng.choice(nominal_indices, size=num_anomalies, replace=False)
        
        # 3. Combine, shuffle, and return as a PyTorch Subset
        balanced_indices = np.concatenate([balanced_nominal_indices, anomaly_indices])
        rng.shuffle(balanced_indices)
        
        return torch.utils.data.Subset(self, balanced_indices)