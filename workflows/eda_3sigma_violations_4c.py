# scripts/eda_3sigma_violations_4c.py
import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# Set project root
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.generate_four_class_derived_stages
from tools.diagnostic_logger import DiagnosticFileHandler
from src.utils.palette_manager import PaletteManager

logger = logging.getLogger("AviationInsight_3Sigma_EDA")
ISSUE_TOKEN = "GENERATE_EDA_3SIGMA_VIOLATIONS_4C"

ALL_31_FEATURE_NAMES = src.generate_four_class_derived_stages.ALL_31_FEATURE_NAMES

PARQUET_PATH = Path("/mnt/fast_storage/projects/aviation_insight/github_examples/dashlink-anomaly-data/dataset/four_class_derived_dataset.parquet")
STATS_PATH = Path("/mnt/fast_storage/projects/aviation_insight/github_examples/dashlink-anomaly-data/dataset/metadata/train_split_stats.json")
ETL_STAGE_DIRECTORY = Path("/mnt/fast_storage/projects/aviation_insight/datasets/nasa_tails/etl_staged")
OUTPUT_CSV_PATH = Path("/mnt/fast_storage/projects/aviation_insight/workflows/EDA/nominal_3sigma_violations.csv")

# Initialize publication-ready theme
theme = PaletteManager()
theme.apply_publication_style()


# -----------------------------------------------------------------------------
# CORRECTED FAA & DERIVED/RESIDUAL FEATURE DICTIONARIES
# -----------------------------------------------------------------------------
FAA_LIMITS = {
    "vspd_fps": (-16.67, 16.67),                       # > 1000 fpm descent threshold (ft/s)
    "Pitch Angle LSP": (-2.0, 10.0),                   # [-2, 10] deg
    "Roll Angle LSP": (-15.0, 15.0),                   # [-15, 15] deg
}

RESIDUAL_AND_DERIVED_FEATURES = [
    "alpha_est",
    "res_energy_rate",
    "res_side_force",
    "vt_dot",
    "res_flight_path_ae",
    "res_energy_rate_no_k3",
    "res_nz"
]

# =============================================================================
# ENHANCED SENSOR AUDIT & QUARANTINE SCRIPT WITH IQR OVERLAY PLOTTING
# =============================================================================

def enhanced_sensor_integrity_check(
    matrix_160xP: np.ndarray,
    row: pd.Series,
    feature_to_idx: dict
) -> tuple[bool, str, dict]:
    hard_limits = {
        "Total Pressure LSP": lambda sig: np.where(sig <= 0.0)[0],
        "alpha_est": lambda sig: np.where((sig < -10.0) | (sig > 30.0))[0],
    }

    audit_details = {}
    for feat_name, check_func in hard_limits.items():
        if feat_name in row and feat_name in feature_to_idx:
            signal = np.asarray(row[feat_name], dtype=np.float64)
            viol_indices = check_func(signal)
            if len(viol_indices) > 0:
                audit_details = {
                    "offending_feature": feat_name,
                    "violation_type": "Sensor_Dropout_Or_Invalid",
                    "min_value": float(np.min(signal)),
                    "max_value": float(np.max(signal)),
                    "violation_count": int(len(viol_indices)),
                    "violation_indices": list(viol_indices)
                }
                return True, f"Sensor_Dropout_Or_Invalid_{feat_name}", audit_details

    return False, "None", {}


def generate_energy_rate_audit_plots_from_raw(
    sampled_records: list, 
    norm_stats: dict,
    output_dir: Path = OUTPUT_CSV_PATH.parent / "energy_rate_audit_plots"
):
    """
    Generates multi-panel time-series plots with static IQR bounds overlayed 
    using PaletteManager styling for publication/Praxis defense.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Generating audit plots with IQR overlays for {len(sampled_records)} windows in '{output_dir}/'...")

    for i, item in enumerate(sampled_records):
        row = item.get("row", None)
        feature_to_idx = item.get("feature_to_idx", {})
        file_id = item.get("file_id", f"window_{i}")

        if row is None or not feature_to_idx:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True)
        time_steps = np.arange(160)

        def get_signal(feat_name):
            if feat_name in row:
                sig = np.asarray(row[feat_name], dtype=np.float64)
                if len(sig) == 160:
                    return sig
            return np.zeros(160)

        # Panel 1: Energy Rate Residual with static IQR Fences overlayed
        r_signal = get_signal("res_energy_rate")
        axes[0].plot(time_steps, r_signal, color=theme.OKABE_ITO[0], linewidth=1.5, label="res_energy_rate")
        
        if "res_energy_rate" in norm_stats:
            iqr_upper = norm_stats["res_energy_rate"]["iqr_upper"]
            iqr_lower = norm_stats["res_energy_rate"]["iqr_lower"]
            axes[0].axhline(iqr_upper, color=theme.CRIMSON, linestyle='--', linewidth=1.2, label=f"IQR Upper ({iqr_upper:.2f})")
            axes[0].axhline(iqr_lower, color=theme.CRIMSON, linestyle='--', linewidth=1.2, label=f"IQR Lower ({iqr_lower:.2f})")
            
        axes[0].axhline(0, color='black', linestyle=':', alpha=0.6)
        axes[0].set_ylabel("Energy Rate Res")
        axes[0].legend(loc="upper right", frameon=True)
        axes[0].grid(True, color=theme.GRID_COLOR)

        # Panel 2: Computed Airspeed LSP (Speed High Context)
        as_signal = get_signal("Computed Airspeed LSP")
        axes[1].plot(time_steps, as_signal, color=theme.OKABE_ITO[2], linewidth=1.5, label="Airspeed LSP")
        axes[1].set_ylabel("Airspeed (kts)")
        axes[1].legend(loc="upper right", frameon=True)
        axes[1].grid(True, color=theme.GRID_COLOR)

        # Panel 3: Vertical Speed (Path High Context)
        vs_signal = get_signal("vspd_fps")
        axes[2].plot(time_steps, vs_signal, color=theme.OKABE_ITO[1], linewidth=1.5, label="Vertical Speed")
        if "vspd_fps" in FAA_LIMITS:
            faa_low, faa_high = FAA_LIMITS["vspd_fps"]
            axes[2].axhline(faa_low, color=theme.CRIMSON, linestyle=':', alpha=0.8, label=f"FAA Limit ({faa_low} fps)")
        axes[2].set_ylabel("VSPD (fps)")
        axes[2].set_xlabel("Time Step (160 frames)")
        axes[2].legend(loc="upper right", frameon=True)
        axes[2].grid(True, color=theme.GRID_COLOR)

        fig.suptitle(f"Energy Rate Residual & IQR Boundary Audit | File ID: {file_id}", fontsize=10, weight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"audit_iqr_{file_id}_{i}.png")
        plt.savefig(plot_path, dpi=theme.DPI)
        plt.close(fig)

    print(f"[INFO] Successfully saved all IQR audit plots to {output_dir}/")


def generate_quarantine_and_audit_manifests(
    results_df: pd.DataFrame, 
    output_dir: str = "./quarantine_audit_outputs"
):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("QUARANTINE & AUDIT MANIFEST GENERATION")
    print("=" * 60)
    
    sensor_issues_df = results_df[results_df["assigned_category"] == "Sensor_Issue"].copy()
    quarantine_csv_path = os.path.join(output_dir, "quarantined_sensor_issues.csv")
    sensor_issues_df.to_csv(quarantine_csv_path, index=False)
    
    print(f"\n[QUARANTINE] Found {len(sensor_issues_df)} sensor-corrupted windows.")
    print(f"-> Exported quarantine manifest to: {quarantine_csv_path}")
    
    if not sensor_issues_df.empty:
        print("\n--- Corrupted Data Summary Table ---")
        summary_cols = [col for col in ["file_id", "window_start", "assigned_category", "trigger_reason", "offending_feature", "min_value", "max_value"] if col in sensor_issues_df.columns]
        if not summary_cols:
            summary_cols = sensor_issues_df.columns[:4]
        print(sensor_issues_df[summary_cols].head(15).to_string(index=False))
    
    energy_audit_df = results_df[results_df["assigned_category"] == "Category_D"].copy()
    if "trigger_reason" in energy_audit_df.columns:
        energy_audit_df = energy_audit_df[
            energy_audit_df["trigger_reason"].str.contains("res_energy_rate", na=False)
        ]
    
    sample_size = min(20, len(energy_audit_df))
    audit_sample = energy_audit_df.sample(n=sample_size, random_state=42) if sample_size > 0 else energy_audit_df
    
    audit_csv_path = os.path.join(output_dir, "energy_rate_audit_sample.csv")
    audit_sample.to_csv(audit_csv_path, index=False)
    
    print(f"\n[AUDIT] Extracted {len(audit_sample)} energy rate residual sample windows.")
    print(f"-> Exported audit sample manifest to: {audit_csv_path}")


# -----------------------------------------------------------------------------
# MODULAR CATEGORY FILTER FUNCTIONS & DIAGNOSTICS
# -----------------------------------------------------------------------------

def check_sensor_integrity_and_chatter(
    matrix_160xP: np.ndarray,
    row: pd.Series,
    feature_to_idx: dict
) -> tuple[bool, str, dict]:
    hard_limits = {
        "Total Pressure LSP": lambda sig: np.where(sig <= 0.0)[0],
        "alpha_est": lambda sig: np.where((sig < -10.0) | (sig > 30.0))[0],
    }

    audit_details = {}
    for feat_name, check_func in hard_limits.items():
        if feat_name in row and feat_name in feature_to_idx:
            signal = np.asarray(row[feat_name], dtype=np.float64)
            viol_indices = check_func(signal)
            if len(viol_indices) > 0:
                audit_details = {
                    "offending_feature": feat_name,
                    "violation_type": "Sensor_Dropout_Or_Invalid",
                    "min_value": float(np.min(signal)),
                    "max_value": float(np.max(signal)),
                    "violation_count": int(len(viol_indices)),
                    "violation_indices": list(viol_indices)
                }
                return True, f"Sensor_Dropout_Or_Invalid_{feat_name}", audit_details

    chatter_features = ["Flap Position"]
    for feat_name in chatter_features:
        if feat_name in row and feat_name in feature_to_idx:
            signal = np.asarray(row[feat_name], dtype=np.float64)
            diffs = np.diff(signal)
            sign_changes = np.sum(np.diff(np.signbit(diffs)))
            if sign_changes > 15:
                audit_details = {
                    "offending_feature": feat_name,
                    "violation_type": "Hardware_Chatter",
                    "min_value": float(np.min(signal)),
                    "max_value": float(np.max(signal)),
                    "violation_count": int(sign_changes),
                    "violation_indices": []
                }
                return True, f"Hardware_Chatter_{feat_name}", audit_details

    return False, "None", {}


def filter_category_a_boundary_artifacts(all_viol_indices: list, frame_count: int = 160, margin: int = 5) -> bool:
    if not all_viol_indices:
        return False
    boundary_frames = set(range(0, margin)).union(set(range(frame_count - margin, frame_count)))
    return all(idx in boundary_frames for idx in all_viol_indices)


def filter_category_b_atmospheric_gusts(all_viol_indices: list, violating_features: list, max_duration: int = 2) -> bool:
    if not all_viol_indices or len(violating_features) > 2:
        return False
    
    sorted_indices = sorted(list(set(all_viol_indices)))
    diffs = np.diff(sorted_indices)
    max_streak, curr_streak = 1, 1
    for d in diffs:
        if d == 1:
            curr_streak += 1
            max_streak = max(max_streak, curr_streak)
        else:
            curr_streak = 1
            
    return max_streak <= max_duration


def filter_category_c_flare_dynamics(all_viol_indices: list, flare_start_frame: int = 150) -> bool:
    if not all_viol_indices:
        return False
    return all(idx >= flare_start_frame for idx in all_viol_indices)


def check_category_d_trigger_details(
    matrix_160xP: np.ndarray,
    faa_indices: list,
    residual_derived_indices: list,
    feature_to_idx: dict,
    min_consecutive_frames: int = 12
) -> tuple[bool, str]:
    if faa_indices:
        c_faa_t = np.sum(matrix_160xP[:, faa_indices], axis=1)
        if np.any(c_faa_t >= 3):
            return True, "FAA_Cooccurrence_3plus"

    side_force_idx = feature_to_idx.get("res_side_force", -1)
    nz_idx = feature_to_idx.get("res_nz", -1)

    for r_idx in residual_derived_indices:
        r_frames = np.where(matrix_160xP[:, r_idx])[0]
        
        if r_idx == nz_idx:
            threshold = 35
        elif r_idx == side_force_idx:
            threshold = 25
        else:
            threshold = min_consecutive_frames

        if len(r_frames) >= threshold:
            diffs = np.diff(r_frames)
            curr_streak = 1
            for d in diffs:
                if d == 1:
                    curr_streak += 1
                    if curr_streak >= threshold:
                        feat_name = ALL_31_FEATURE_NAMES[r_idx]
                        return True, f"Sustained_Streak_{feat_name}"
                else:
                    curr_streak = 1

    return False, "None"


# -----------------------------------------------------------------------------
# HELPER DATA MAPPERS
# -----------------------------------------------------------------------------

def build_flight_to_mission_map(etl_staged_dir: Path) -> dict:
    flight_to_mission = {}
    if not etl_staged_dir.exists():
        return flight_to_mission

    for mission_dir in etl_staged_dir.iterdir():
        if mission_dir.is_dir():
            mission_id = mission_dir.name
            for flight_file in mission_dir.glob("*.parquet"):
                flight_id = flight_file.name.replace("_etl.parquet", "").replace(".parquet", "")
                flight_to_mission[flight_id] = mission_id
    return flight_to_mission


def parse_sensor_stats(stats_path: Path):
    with open(stats_path, "r") as f:
        raw_stats = json.load(f)

    if "sensor_stats" not in raw_stats:
        raise KeyError(f"Expected top-level 'sensor_stats' key in {stats_path}.")
    
    sensor_stats_dict = raw_stats["sensor_stats"]
    norm_stats = {}
    for idx, feat_name in enumerate(ALL_31_FEATURE_NAMES):
        entry = None

        if feat_name in sensor_stats_dict:
            entry = sensor_stats_dict[feat_name]
        else:
            for k, v in sensor_stats_dict.items():
                if isinstance(v, dict) and (v.get("feature_name") == feat_name or v.get("feature_idx") == idx):
                    entry = v
                    break

        if isinstance(entry, dict) and entry:
            q25 = entry.get("q25")
            q75 = entry.get("q75")
            
            if q25 is None or q75 is None:
                mean = float(entry.get("mean", 0.0))
                std = float(entry.get("std", 1.0))
                q25 = mean - 0.6745 * std
                q75 = mean + 0.6745 * std
            else:
                q25 = float(q25)
                q75 = float(q75)

            iqr = q75 - q25
            if iqr <= 1e-6:
                std = float(entry.get("std", 1.0))
                iqr = 2.0 * std if std > 1e-6 else 1.0
                q25, q75 = -iqr / 2.0, iqr / 2.0

            if feat_name == "res_side_force":
                multiplier = 2.5
            elif feat_name == "res_nz":
                multiplier = 2.5
            else:
                multiplier = 2.0

            norm_stats[feat_name] = {
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
                "iqr_lower": q25 - multiplier * iqr,
                "iqr_upper": q75 + multiplier * iqr,
            }
    return norm_stats


# -----------------------------------------------------------------------------
# MAIN ANALYSIS & CLASSIFICATION LOOP WITH DIAGNOSTICS
# -----------------------------------------------------------------------------

def analyze_3sigma_violations():
    norm_stats = parse_sensor_stats(STATS_PATH)
    flight_to_mission = build_flight_to_mission_map(ETL_STAGE_DIRECTORY)
    
    feature_to_idx = {feat: idx for idx, feat in enumerate(ALL_31_FEATURE_NAMES)}
    faa_indices = [feature_to_idx[f] for f in FAA_LIMITS if f in feature_to_idx]
    residual_derived_indices = [feature_to_idx[f] for f in RESIDUAL_AND_DERIVED_FEATURES if f in feature_to_idx]

    logger.info(f"Loading dataset from {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)

    nominal_df = df[df["label"] == 0].copy()
    logger.info(f"Loaded {len(df):,} total instances. Scanning {len(nominal_df):,} nominal instances.")

    results = []
    raw_records_cache = []
    
    category_counts = {
        "Category_A": 0, 
        "Category_B": 0, 
        "Category_C": 0, 
        "Category_D": 0, 
        "Sensor_Issue": 0,
        "Nominal_Clean": 0
    }
    category_d_trigger_counter = Counter()
    category_e_trigger_counter = Counter()

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    for idx_row, row in tqdm(nominal_df.iterrows(), total=len(nominal_df), desc="Scanning Windows"):
        flight_rec = str(row.get("flight_record", row.get("flight_id", "")))

        mission_id = row.get("mission_id", None)
        if pd.isna(mission_id) or str(mission_id).strip() in ["", "unknown_mission", "None"]:
            if flight_rec in flight_to_mission:
                mission_id = flight_to_mission[flight_rec]
            elif flight_rec.startswith("666"):
                mission_id = "Tail_666_8"
            else:
                mission_id = "unknown_mission"

        row_dict = {
            "filename": row.get("filename", f"{flight_rec}.parquet"),
            "mission_id": mission_id,
            "flight_id": row.get("flight_id", flight_rec),
            "flight_record": flight_rec
        }

        matrix_160xP = np.zeros((160, len(ALL_31_FEATURE_NAMES)), dtype=bool)
        violating_features = []
        all_window_viol_indices = []

        for col_idx, feature in enumerate(ALL_31_FEATURE_NAMES):
            if feature not in row:
                row_dict[feature] = "[]"
                continue

            signal = np.asarray(row[feature], dtype=np.float64)

            if feature in FAA_LIMITS:
                l_bound, u_bound = FAA_LIMITS[feature]
                viol_mask = (signal < l_bound) | (signal > u_bound)
            elif feature in norm_stats:
                l_bound = norm_stats[feature]["iqr_lower"]
                u_bound = norm_stats[feature]["iqr_upper"]
                viol_mask = (signal < l_bound) | (signal > u_bound)
            else:
                viol_mask = np.zeros(160, dtype=bool)

            viol_mask |= np.isnan(signal) | np.isinf(signal)
            matrix_160xP[:, col_idx] = viol_mask

            v_indices = np.where(viol_mask)[0].tolist()
            if len(v_indices) > 0:
                violating_features.append(feature)
                all_window_viol_indices.extend(v_indices)

            row_dict[feature] = str(v_indices)

        c_t = np.sum(matrix_160xP, axis=1)
        row_dict["max_concurrent_channels"] = int(np.max(c_t)) if len(c_t) > 0 else 0
        row_dict["total_violating_features"] = len(violating_features)

        is_cat_d, d_reason = check_category_d_trigger_details(
            matrix_160xP, faa_indices, residual_derived_indices, feature_to_idx, min_consecutive_frames=12
        )

        is_sensor_fault, sensor_reason, sensor_audit_details = enhanced_sensor_integrity_check(matrix_160xP, row, feature_to_idx)

        if len(violating_features) == 0:
            assigned_cat = "Nominal_Clean"
        elif is_sensor_fault:
            assigned_cat = "Sensor_Issue"
            category_e_trigger_counter[sensor_reason] += 1
            row_dict["trigger_reason"] = sensor_reason
            row_dict.update(sensor_audit_details)
        elif filter_category_a_boundary_artifacts(all_window_viol_indices):
            assigned_cat = "Category_A"
        elif filter_category_b_atmospheric_gusts(all_window_viol_indices, violating_features):
            assigned_cat = "Category_B"
        elif filter_category_c_flare_dynamics(all_window_viol_indices):
            assigned_cat = "Category_C"
        elif is_cat_d:
            assigned_cat = "Category_D"
            category_d_trigger_counter[d_reason] += 1
            row_dict["trigger_reason"] = d_reason
        else:
            assigned_cat = "Nominal_Clean"

        row_dict["assigned_category"] = assigned_cat
        category_counts[assigned_cat] += 1
        results.append(row_dict)

        raw_records_cache.append({
            "file_id": row_dict["flight_id"],
            "assigned_category": assigned_cat,
            "trigger_reason": row_dict.get("trigger_reason", "None"),
            "matrix_160xP": matrix_160xP,
            "row": row,
            "feature_to_idx": feature_to_idx
        })

    output_df = pd.DataFrame(results)

    logger.info("\n--- Category Breakdown ---")
    for cat, count in category_counts.items():
        pct = (count / len(nominal_df)) * 100
        logger.info(f"  • {cat}: {count:,} nominal windows ({pct:.2f}%)")

    logger.info("\n--- Category D Trigger Breakdown (Root Cause Diagnostics) ---")
    for reason, count in category_d_trigger_counter.most_common():
        pct = (count / category_counts["Category_D"]) * 100 if category_counts["Category_D"] > 0 else 0
        logger.info(f"  • {reason}: {count:,} windows ({pct:.1f}% of Category D)")

    output_df.to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info(f"Exported successfully to {OUTPUT_CSV_PATH.resolve()}")

    generate_quarantine_and_audit_manifests(output_df, output_dir=OUTPUT_CSV_PATH.parent / "quarantine_audit_outputs")

    energy_rate_cache = [r for r in raw_records_cache if r["assigned_category"] == "Category_D" and "res_energy_rate" in str(r.get("trigger_reason", ""))]
    sample_size = min(20, len(energy_rate_cache))
    sampled_audit_records = random.sample(energy_rate_cache, sample_size) if sample_size > 0 else energy_rate_cache

    generate_energy_rate_audit_plots_from_raw(
        sampled_records=sampled_audit_records, 
        norm_stats=norm_stats,
        output_dir=OUTPUT_CSV_PATH.parent / "energy_rate_audit_plots"
    )


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('▶ %(message)s'))
    logger.addHandler(console_handler)

    file_handler = DiagnosticFileHandler(
        issue_token=ISSUE_TOKEN, 
        output_dir=Path("/mnt/fast_storage/projects/aviation_insight/logs/diagnostics")
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    analyze_3sigma_violations()