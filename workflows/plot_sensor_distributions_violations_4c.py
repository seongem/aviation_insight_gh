# scripts/plot_sensor_distributions_violations_4c.py
import sys
import ast
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.palette_manager import PaletteManager

PARQUET_FILE = Path("/mnt/fast_storage/projects/aviation_insight/github_examples/dashlink-anomaly-data/dataset/four_class_derived_dataset.parquet")
QUARANTINE_CSV_PATH = Path("/mnt/fast_storage/projects/aviation_insight/workflows/EDA/quarantine_audit_outputs/quarantined_sensor_issues.csv")
OUTPUT_FIG_PATH = Path("/mnt/fast_storage/projects/aviation_insight/workflows/EDA/sensor_distribution_violations_diagnostic.svg")
OUTPUT_TABLE_PATH = Path("/mnt/fast_storage/projects/aviation_insight/workflows/EDA/quarantine_audit_outputs/violation_summary_table.csv")

def plot_sensor_distribution_clean_markers():
    PaletteManager.apply_publication_style()
    
    cols = [
        "alpha_est",
        "Total Pressure LSP",
        "flight_record",
        "mission_id",
        "label"
    ]
    df = pd.read_parquet(PARQUET_FILE, columns=cols)
    nominal_df = df[df["label"] == 0]

    flight_ids = nominal_df["flight_record"].values
    missions = nominal_df["mission_id"].values
    time_steps = np.arange(160)

    alpha_mat = np.vstack(nominal_df["alpha_est"].values)
    press_mat = np.vstack(nominal_df["Total Pressure LSP"].values)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(PaletteManager.PAGE_WIDTH_INCHES, 6.5),
        sharex=True,
        dpi=PaletteManager.DPI
    )

    c_blue = PaletteManager.OKABE_ITO[0]
    c_green = PaletteManager.OKABE_ITO[2]
    c_vermilion = PaletteManager.CRIMSON

    # -------------------------------------------------------------------------
    # Panel 1: Alpha Estimate Distribution
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    q25 = np.percentile(alpha_mat, 25, axis=0)
    q50 = np.percentile(alpha_mat, 50, axis=0)
    q75 = np.percentile(alpha_mat, 75, axis=0)
    iqr = q75 - q25

    whisker_low = np.maximum(np.min(alpha_mat, axis=0), q25 - 1.5 * iqr)
    whisker_high = np.minimum(np.max(alpha_mat, axis=0), q75 + 1.5 * iqr)

    ax1.plot(time_steps, q50, color=c_blue, linewidth=1.5, label="Median ($Q_{50}$)", zorder=3)
    ax1.fill_between(time_steps, q25, q75, color=c_blue, alpha=0.35, label="IQR ($Q_{25}$–$Q_{75}$)", zorder=2)
    ax1.plot(time_steps, whisker_high, color=c_blue, linestyle="--", linewidth=0.8, label="Whiskers (1.5 IQR)", zorder=3)
    ax1.plot(time_steps, whisker_low, color=c_blue, linestyle="--", linewidth=0.8, zorder=3)

    # -------------------------------------------------------------------------
    # Panel 2: Total Pressure LSP Distribution
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    p_q25 = np.percentile(press_mat, 25, axis=0)
    p_q50 = np.percentile(press_mat, 50, axis=0)
    p_q75 = np.percentile(press_mat, 75, axis=0)
    p_iqr = p_q75 - p_q25

    p_whisker_low = np.maximum(np.min(press_mat, axis=0), p_q25 - 1.5 * p_iqr)
    p_whisker_high = np.minimum(np.max(press_mat, axis=0), p_q75 + 1.5 * p_iqr)

    ax2.plot(time_steps, p_q50, color=c_green, linewidth=1.5, label="Median ($Q_{50}$)", zorder=3)
    ax2.fill_between(time_steps, p_q25, p_q75, color=c_green, alpha=0.35, label="IQR ($Q_{25}$–$Q_{75}$)", zorder=2)
    ax2.plot(time_steps, p_whisker_high, color=c_green, linestyle="--", linewidth=0.8, label="Whiskers (1.5 IQR)", zorder=3)
    ax2.plot(time_steps, p_whisker_low, color=c_green, linestyle="--", linewidth=0.8, zorder=3)

    table_rows = []
    violation_counter = 1
    alpha_labeled_once = False
    press_labeled_once = False

    if QUARANTINE_CSV_PATH.exists():
        q_df = pd.read_csv(QUARANTINE_CSV_PATH)
        for _, row in q_df.iterrows():
            feat = str(row.get("offending_feature", ""))
            fid = str(row.get("flight_id", row.get("flight_record", "")))
            frec = str(row.get("flight_record", ""))
            viol_str = str(row.get("violation_indices", "[]"))
            
            try:
                cleaned_str = re.sub(r'np\.int64\((.*?)\)', r'\1', viol_str)
                indices = ast.literal_eval(cleaned_str)
                indices = [int(x) for x in indices]
            except Exception:
                indices = []

            if not indices:
                continue

            match_row = pd.DataFrame()
            for col in ["flight_record", "flight_id"]:
                if col in nominal_df.columns:
                    match_row = nominal_df[nominal_df[col].astype(str).str.strip() == fid.strip()]
                    if match_row.empty and frec:
                        match_row = nominal_df[nominal_df[col].astype(str).str.strip() == frec.strip()]
                    if not match_row.empty:
                        break

            if match_row.empty:
                continue

            r_data = match_row.iloc[0]
            mission = r_data.get("mission_id", "Unknown")

            if "alpha_est" in feat:
                sig = np.asarray(r_data["alpha_est"])
                for vi in indices:
                    if vi < len(sig):
                        val = sig[vi]
                        # Only label the first occurrence for the legend handle
                        label_str = "Violation Points (#8-#12)" if not alpha_end_labeled_check(violation_counter) else ""
                        if not alpha_labeled_once:
                            ax1.scatter([vi], [val], color=c_vermilion, s=25, zorder=5, label="Violation Points")
                            alpha_labeled_once = True
                        else:
                            ax1.scatter([vi], [val], color=c_vermilion, s=25, zorder=5)

                        table_rows.append({
                            "Index": violation_counter,
                            "Panel": "Alpha Estimate",
                            "Flight Record": frec if frec else fid,
                            "Mission ID": mission,
                            "Offending Feature": feat,
                            "Time Step (s)": vi,
                            "Recorded Value": round(val, 2)
                        })
                        violation_counter += 1

            elif "Total Pressure" in feat:
                sig = np.asarray(r_data["Total Pressure LSP"])
                for vi in indices:
                    if vi < len(sig):
                        val = sig[vi]
                        if not press_labeled_once:
                            ax2.scatter([vi], [val], color=c_vermilion, s=25, zorder=5, label="Violation Points")
                            press_labeled_once = True
                        else:
                            ax2.scatter([vi], [val], color=c_vermilion, s=25, zorder=5)

                        table_rows.append({
                            "Index": violation_counter,
                            "Panel": "Total Pressure",
                            "Flight Record": frec if frec else fid,
                            "Mission ID": mission,
                            "Offending Feature": feat,
                            "Time Step (s)": vi,
                            "Recorded Value": round(val, 2)
                        })
                        violation_counter += 1

    summary_table_df = pd.DataFrame(table_rows)
    if not summary_table_df.empty:
        OUTPUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary_table_df.to_csv(OUTPUT_TABLE_PATH, index=False)
        print(f"\n-> Saved violation summary table to: {OUTPUT_TABLE_PATH.resolve()}\n")

    # Panel 1 Formatting (Legend at lower left, single entry)
    ax1.set_ylabel(r"Kinematic Estimate $\alpha_{est}$" + "\n(deg)", fontsize=8, fontweight="bold", color="black", labelpad=10)
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.grid(True, linestyle=":", alpha=0.6, zorder=1)
    leg1 = ax1.legend(loc="lower left", framealpha=0.9, fontsize=7)
    leg1.set_zorder(10)
    ax1.set_title("Sensor Distribution Envelopes with Clean Violation Markers", fontweight="bold", color="black")

    # Panel 2 Formatting (Legend at lower left, single entry)
    ax2.set_ylabel("Total Pressure LSP", fontsize=8, fontweight="bold", color="black", labelpad=10)
    ax2.tick_params(axis='y', labelcolor="black")
    ax2.set_xlabel("Window Time Step (Seconds at 1 Hz)", fontweight="bold", color="black")
    ax2.set_xlim(0, 159)
    ax2.grid(True, linestyle=":", alpha=0.6, zorder=1)
    leg2 = ax2.legend(loc="lower left", framealpha=0.9, fontsize=7)
    leg2.set_zorder(10)

    plt.tight_layout()
    OUTPUT_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIG_PATH, format="svg", bbox_inches="tight")
    print(f"🎉 Clean SVG distribution figure saved for Quarto export: {OUTPUT_FIG_PATH.resolve()}")
    plt.close()

def alpha_end_labeled_check(counter):
    return counter > 8

if __name__ == "__main__":
    plot_sensor_distribution_clean_markers()