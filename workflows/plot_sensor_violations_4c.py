# scripts/plot_sensor_violations_4c.py
import sys
import json
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set project root
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.palette_manager import PaletteManager

PARQUET_PATH = Path("/mnt/fast_storage/projects/aviation_insight/github_examples/dashlink-anomaly-data/dataset/four_class_derived_dataset.parquet")
QUARANTINE_CSV_PATH = Path("/mnt/fast_storage/projects/aviation_insight/workflows/EDA/quarantine_audit_outputs/quarantined_sensor_issues.csv")
OUTPUT_FIG_DIR = Path("/mnt/fast_storage/projects/aviation_insight/workflows/EDA/sensor_violation_plots")

def generate_sensor_violation_plots():
    PaletteManager.apply_publication_style()
    
    if not QUARANTINE_CSV_PATH.exists():
        print(f"⚠️ Quarantine CSV not found at {QUARANTINE_CSV_PATH}. Run EDA script first.")
        return

    quarantine_df = pd.read_csv(QUARANTINE_CSV_PATH)
    if quarantine_df.empty:
        print("⚠️ Quarantine CSV is empty. No sensor violations to plot.")
        return

    df = pd.read_parquet(PARQUET_PATH)
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Generating inspection plots for {len(quarantine_df)} quarantined sensor violation windows...")

    for idx, row_meta in quarantine_df.iterrows():
        flight_id = row_meta.get("flight_id", row_meta.get("flight_record", f"row_{idx}"))
        trigger_reason = row_meta.get("trigger_reason", "Unknown")
        offending_feat = row_meta.get("offending_feature", "Unknown")
        viol_indices_str = row_meta.get("violation_indices", "[]")

        try:
            viol_indices = ast.literal_eval(viol_indices_str)
        except Exception:
            viol_indices = []

        match_df = pd.DataFrame()
        for col in ["flight_id", "flight_record", "filename"]:
            if col in df.columns:
                match_df = df[df[col].astype(str) == str(flight_id)]
                if not match_df.empty:
                    break
        
        if match_df.empty:
            flight_rec = row_meta.get("flight_record", "")
            for col in ["flight_record", "flight_id", "filename"]:
                if col in df.columns and flight_rec:
                    match_df = df[df[col].astype(str) == str(flight_rec)]
                    if not match_df.empty:
                        break

        if match_df.empty:
            print(f"⚠️ Could not locate flight data for ID: {flight_id}")
            continue

        row_data = match_df.iloc[0]
        time_axis = np.arange(160)

        alt = np.asarray(row_data["Baro Correct Altitude LSP"]) if "Baro Correct Altitude LSP" in row_data else np.zeros(160)
        vspd = np.asarray(row_data["vspd_fps"]) if "vspd_fps" in row_data else np.zeros(160)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(PaletteManager.PAGE_WIDTH_INCHES, 5.0), sharex=True)

        c_purple = PaletteManager.OKABE_ITO[5]
        c_green = PaletteManager.OKABE_ITO[2]
        c_blue = PaletteManager.OKABE_ITO[0]
        c_vermilion = PaletteManager.CRIMSON
        c_yellow = PaletteManager.OKABE_ITO[6]

        # ---------------------------------------------------------------------
        # Panel 1: Trajectory Profile (Altitude & Vertical Speed)
        # ---------------------------------------------------------------------
        l1 = ax1.plot(time_axis, vspd, color=c_purple, linewidth=1.8, label=r"Vertical Speed $v_{spd}$ (ft/s)", zorder=3)
        ax1.set_ylabel(r"$v_{spd}$ (ft/s)", fontweight="bold", color="black")
        ax1.tick_params(axis='y', labelcolor="black")
        ax1.grid(True, alpha=0.3, zorder=0)

        ax1_alt = ax1.twinx()
        ax1_alt.set_zorder(2)
        ax1.set_zorder(3)
        ax1.patch.set_visible(False) # Allow transparency for twinx overlaying

        l2 = ax1_alt.plot(time_axis, alt, color=c_green, linestyle="--", linewidth=1.6, label="Baro Corrected Altitude (ft)", zorder=3)
        ax1_alt.set_ylabel("Altitude (ft)", fontweight="bold", color="black")
        ax1_alt.tick_params(axis='y', labelcolor="black")

        lines1 = l1 + l2
        leg1 = ax1.legend(lines1, [l.get_label() for l in lines1], loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
        leg1.set_zorder(10)  # Force legend above gridlines and traces
        
        ax1.set_title(f"Flight ID: {flight_id} | Trigger: {trigger_reason} | Offender: {offending_feat}", fontweight="bold", color="black")

        # ---------------------------------------------------------------------
        # Panel 2: Conditional Branching based on Offending Feature
        # ---------------------------------------------------------------------
        if "alpha_est" in str(offending_feat):
            alpha_est = np.asarray(row_data["alpha_est"]) if "alpha_est" in row_data else np.zeros(160)
            vert_accel = np.asarray(row_data["Vertical Acceleration"]) if "Vertical Acceleration" in row_data else np.zeros(160)

            a_line = ax2.plot(time_axis, alpha_est, color=c_blue, linewidth=1.8, label=r"Kinematic Estimate $\alpha_{est}$ (deg)", zorder=3)
            ax2.set_ylabel(r"$\alpha_{est}$ (deg)", fontweight="bold", color="black")
            ax2.tick_params(axis='y', labelcolor="black")

            ax2_accel = ax2.twinx()
            ax2_accel.set_zorder(4)  # Ensure twinx sits above ax2
            ax2.set_zorder(3)
            ax2.patch.set_visible(False)

            n_line = ax2_accel.plot(time_axis, vert_accel, color=c_vermilion, linestyle="--", linewidth=1.6, label="Vertical Accel $n_z$ (g)", zorder=3)
            ax2_accel.set_ylabel("Vertical Accel (g)", fontweight="bold", color="black")
            ax2_accel.tick_params(axis='y', labelcolor="black")

            lines_alpha = a_line + n_line
            
            # Attach legend to the top-most twin axis so it renders above its gridlines
            leg2 = ax2_accel.legend(lines_alpha, [l.get_label() for l in lines_alpha], loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
            leg2.set_zorder(100)  # Maximum canvas priority

        else:
            total_press = np.asarray(row_data["Total Pressure LSP"]) if "Total Pressure LSP" in row_data else np.zeros(160)
            
            ax2.plot(time_axis, total_press, color=c_vermilion, linewidth=1.8, label="Total Pressure LSP", zorder=3)
            
            if viol_indices:
                v_arr = np.array(viol_indices)
                ax2.scatter(v_arr, total_press[v_arr], color="black", s=30, zorder=5, label="Violation Frames")
                for vi in v_arr:
                    ax2.axvline(vi, color=c_yellow, alpha=0.4, linestyle="--", zorder=1)

            ax2.set_ylabel("Total Pressure LSP", fontweight="bold", color="black")
            ax2.tick_params(axis='y', labelcolor="black")
            leg2 = ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
            leg2.set_zorder(10)

        ax2.set_xlabel("Time Step in Approach Window (seconds)", fontweight="bold", color="black")
        ax2.grid(True, alpha=0.3, zorder=0)

        plt.tight_layout()
        out_path = OUTPUT_FIG_DIR / f"sensor_violation_{flight_id}_{idx}.png"
        plt.savefig(out_path, dpi=PaletteManager.DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"🎉 Successfully generated updated violation plots with z-indexed legends in: {OUTPUT_FIG_DIR.resolve()}")

if __name__ == "__main__":
    generate_sensor_violation_plots()