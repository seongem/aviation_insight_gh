# workflows/build_four_class_pipeline.py

import sys
from pathlib import Path

# Resolve project root (/mnt/fast_storage/projects/aviation_insight)
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import logging
import random
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import GroupShuffleSplit

from tools.diagnostic_logger import DiagnosticFileHandler
from src.utils.load_environment import LoadEnvironment


logger = logging.getLogger("AviationInsight")
ISSUE_TOKEN = "GENERATE_FOUR_CLASS_DERIVED_STAGES"


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# Parameter Group Mapping
SENSOR_GROUPS = {
    1: "Dynamics",
    2: "Kinematics",
    3: "Configuration",
    4: "Propulsion",
    5: "Guidance",
    6: "System / Metadata",
    7: "Accelerations / Rates",
    8: "Physics_Residuals",
    0: "Unclassified",
}

# Full 31 Channel Specifications (20 Baseline + 6 Kinematics + 5 PIML Residuals)
CHANNEL_METADATA = [
    # Baseline Channels (0-19)
    (0, "Aileron Position LH", 1, "deg", "baseline_data", True, False, False),
    (1, "Aileron Position RH", 1, "deg", "baseline_data", True, False, False),
    (2, "Corrected Angle of Attack", 1, "deg", "baseline_data", True, False, False),
    (3, "Baro Correct Altitude LSP", 2, "ft", "baseline_data", True, False, False),
    (4, "Computed Airspeed LSP", 2, "knots", "baseline_data", True, False, False),
    (5, "Selected Course", 5, "deg", "baseline_data", True, False, False),
    (6, "Drift Angle", 2, "deg", "baseline_data", True, False, False),
    (7, "Elevator Position Left", 1, "deg", "baseline_data", True, False, False),
    (8, "T.E. Flap Position", 3, "deg", "baseline_data", True, False, False),
    (9, "Glideslope Deviation", 5, "dots", "baseline_data", True, False, False),
    (10, "Selected Heading", 5, "deg", "baseline_data", True, False, False),
    (11, "Localizer Deviation", 5, "dots", "baseline_data", True, False, False),
    (12, "Core Speed Avg", 4, "%", "baseline_data", True, False, False),
    (13, "Total Pressure LSP", 2, "mbar", "baseline_data", True, False, False),
    (14, "Pitch Angle LSP", 1, "deg", "baseline_data", True, False, False),
    (15, "Roll Angle LSP", 1, "deg", "baseline_data", True, False, False),
    (16, "Rudder Position", 1, "deg", "baseline_data", True, False, False),
    (17, "True Heading LSP", 2, "deg", "baseline_data", True, False, False),
    (18, "Vertical Acceleration", 7, "g", "baseline_data", True, False, False),
    (19, "Wind Speed", 2, "knots", "baseline_data", True, False, False),

    # Derived Kinematic Channels (20-25)
    (20, "p", 7, "rad/s", "acc_derived", False, True, False),
    (21, "q", 7, "rad/s", "acc_derived", False, True, False),
    (22, "r", 7, "rad/s", "acc_derived", False, True, False),
    (23, "p_dot", 7, "rad/s^2", "acc_derived", False, True, False),
    (24, "q_dot", 7, "rad/s^2", "acc_derived", False, True, False),
    (25, "r_dot", 7, "rad/s^2", "acc_derived", False, True, False),

    # Derived PIML Residual Channels (26-30)
    (26, "res_energy_rate", 8, "ft/s", "piml_derived", False, False, True),
    (27, "res_flight_path", 8, "ft/s", "piml_derived", False, False, True),
    (28, "res_side_force", 8, "g", "piml_derived", False, False, True),
    (29, "vspd_fps", 8, "ft/s", "piml_derived", False, False, True),
    (30, "vt_dot", 8, "ft/s^2", "piml_derived", False, False, True),
]


def execute_pipeline_build():
    set_seed(42)
    env = LoadEnvironment()
    
    manifest_dir = env.DATABASE_DIR / "manifests"
    metadata_dir = env.DATABASE_DIR / "metadata"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = env.DATABASE_DIR / "four_class_derived_dataset.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing master parquet dataset at: {parquet_path}")

    # Read Master Parquet Table
    logger.info(f"Loading master parquet dataset from {parquet_path}...")
    df = pq.read_table(parquet_path).to_pandas()
    n_instances = len(df)
    seq_len = 160

    # =========================================================================
    # STEP 1: GENERATE FEATURES.CSV (31-channel schema)
    # =========================================================================
    logger.info("Generating features.csv registry for 31 channels...")
    feature_rows = []
    for idx, name, group_id, units, source_stage, is_base, is_acc, is_piml in CHANNEL_METADATA:
        row = {
            "feature_idx": idx,
            "feature_name": name,
            "group_id": group_id,
            "group_name": SENSOR_GROUPS.get(group_id, "Unclassified"),
            "units": units,
            "source_stage": source_stage,
            "enabled_baseline": is_base,
            "enabled_acc": is_acc,
            "enabled_piml": is_piml,
            "enabled_all": True,
            "logo_drop_g1_dynamics": (group_id != 1),
            "logo_drop_g2_kinematics": (group_id != 2),
            "logo_drop_g3_configuration": (group_id != 3),
            "logo_drop_g4_propulsion": (group_id != 4),
            "logo_drop_g5_guidance": (group_id != 5),
            "logo_drop_g6_system___metadata": (group_id != 6),
            "logo_drop_g7_accelerations___rates": (group_id != 7),
            "logo_drop_g8_physics_residuals": (group_id != 8),
        }
        feature_rows.append(row)

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(manifest_dir / "features.csv", index=False)
    logger.info(f"✓ Saved features.csv ({len(features_df)} channels)")

    # =========================================================================
    # STEP 2: GROUPED SPLIT & FLIGHTS.PARQUET (80/10/10)
    # =========================================================================
    logger.info("Executing GroupShuffleSplit (80% Train / 10% Val / 10% Test)...")
    groups = df["flight_record"].values
    labels = df["label"].values if "label" in df.columns else np.zeros(n_instances)

    # 1. First split: Reserve 10% for Test, 90% for Train+Val
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df, labels, groups=groups))

    # 2. Second split: Reserve 1/9th (~11.11%) of remaining 90% chunk -> exactly 10% total for Val
    train_val_df = df.iloc[train_val_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=1/9, random_state=42)
    rel_train_idx, rel_val_idx = next(gss_val.split(train_val_df, train_val_df["label"].values, groups=train_val_df["flight_record"].values))

    # Map relative positions back to global DataFrame indices
    global_train_idx = train_val_idx[rel_train_idx]
    global_val_idx = train_val_idx[rel_val_idx]

    logger.info(f"✓ Split instance counts -> Train: {len(global_train_idx):,}, Val: {len(global_val_idx):,}, Test: {len(test_idx):,}")

    train_flights = set(df.iloc[global_train_idx]["flight_record"])
    val_flights = set(df.iloc[global_val_idx]["flight_record"])

    all_unique_flights = pd.DataFrame({"flight_record": df["flight_record"].unique()})

    def assign_split(fid):
        if fid in train_flights:
            return "train"
        elif fid in val_flights:
            return "val"
        return "test"

    all_unique_flights["split"] = all_unique_flights["flight_record"].apply(assign_split)
    all_unique_flights.to_parquet(manifest_dir / "flights.parquet", index=False)
    logger.info(f"✓ Saved flights.parquet ({len(all_unique_flights):,} unique flights)")

    # =========================================================================
    # STEP 3: AGGREGATED PHASES.PARQUET (1 entry per flight record)
    # =========================================================================
    logger.info("Aggregating approach phase metrics per flight_record...")
    phase_summary = (
        df.groupby("flight_record")
        .size()
        .reset_index(name="num_instances")
    )
    phase_summary["phase_id"] = 6
    phase_summary["frame_count"] = phase_summary["num_instances"] * seq_len
    phase_summary["start_frame"] = 0
    phase_summary["end_frame"] = phase_summary["frame_count"]

    phase_summary.to_parquet(manifest_dir / "phases.parquet", index=False)
    logger.info(f"✓ Saved phases.parquet ({len(phase_summary):,} flights aggregated)")

    # =========================================================================
    # STEP 4: GENERATE TRAIN STATISTICS JSON FROM MASTER PARQUET
    # =========================================================================
    logger.info("Computing train statistics across 31 channels from master parquet...")
    
    train_df = df.iloc[global_train_idx]
    feature_cols = [row["feature_name"] for _, row in features_df.iterrows()]

    # Stack 31 1D array columns into a 3D matrix -> [N_train, 160, 31]
    train_3d = np.stack([
        np.stack(train_df[col].values) for col in feature_cols
    ], axis=-1)
    
    logger.info(f"✓ Reconstructed Train 3D Matrix Shape: {train_3d.shape}")

    # Reshape to 2D for statistics computation -> [N_train * 160, 31]
    train_2d = train_3d.reshape(-1, len(feature_cols))
    logger.info(f"✓ Train 2D Matrix Shape for Stats: {train_2d.shape}")

    total_frames = train_2d.shape[0]
    means = np.mean(train_2d, axis=0)
    stds = np.std(train_2d, axis=0)
    stds_safe = np.where(stds == 0.0, 1.0, stds)
    mins = np.min(train_2d, axis=0)
    maxs = np.max(train_2d, axis=0)
    q25s = np.percentile(train_2d, 25, axis=0)
    q50s = np.percentile(train_2d, 50, axis=0)
    q75s = np.percentile(train_2d, 75, axis=0)

    sensor_stats = {}
    for idx, row in features_df.iterrows():
        feat_name = row["feature_name"]
        g_idx = int(row["feature_idx"])

        sensor_stats[feat_name] = {
            "feature_idx": g_idx,
            "group_id": int(row["group_id"]),
            "group_name": str(row["group_name"]),
            "source_stage": str(row["source_stage"]),
            "sample_count": int(total_frames),
            "mean": float(means[g_idx]),
            "std": float(stds_safe[g_idx]),
            "raw_std": float(stds[g_idx]),
            "min": float(mins[g_idx]),
            "q25": float(q25s[g_idx]),
            "q50": float(q50s[g_idx]),
            "q75": float(q75s[g_idx]),
            "max": float(maxs[g_idx]),
        }

    stats_payload = {
        "metadata": {
            "split": "train",
            "total_instances": len(global_train_idx),
            "total_frames": int(total_frames),
            "num_features": len(CHANNEL_METADATA),
        },
        "sensor_stats": sensor_stats,
    }

    stats_out_path = metadata_dir / "train_split_stats.json"
    with open(stats_out_path, "w") as f:
        json.dump(stats_payload, f, indent=2)

    logger.info(f"✓ Saved training statistics to {stats_out_path}")
    logger.info("🎉 Complete Four-Class Manifest & Stats Pipeline Built Successfully!")


if __name__ == "__main__":

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('▶ %(message)s'))
    logger.addHandler(console_handler)

    env = LoadEnvironment()
    if getattr(env, "ENVIRONMENT", "local_dev") == "local_dev":
        output_dir = Path("/mnt/fast_storage/projects/aviation_insight/logs/diagnostics")
        logger.info(f"Running in local environment. Logs will be saved to {output_dir}")
    else:
        output_dir = env.OUTPUT_DIR / "diagnostics"
        logger.info(f"Running in {env.ENVIRONMENT} environment. Logs will be saved to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = DiagnosticFileHandler(issue_token=ISSUE_TOKEN, output_dir=output_dir)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    execute_pipeline_build()