# scripts/plot_savgol_window_diagnostics_4c.py

from datetime import datetime
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.stats as stats
import seaborn as sns

# Ensure project root is in path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.load_environment import LoadEnvironment

# ==============================================================================
# OKABE-ITO COLOR PALETTE (Publication Standards)
# ==============================================================================
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

TARGET_FEATURES = ["vspd_fps", "p_dot", "q_dot", "r_dot"]


# ==============================================================================
# METADATA & FOOTER HELPERS
# ==============================================================================
def get_parquet_footer_text(parquet_path: Path) -> str:
    """Extracts embedded pipeline_config metadata from Parquet header across PyArrow schema and FileMetaData stores."""
    path_obj = Path(parquet_path)
    footer_text = f"Parquet: {path_obj.name}"

    try:
        parquet_file = pq.ParquetFile(path_obj)
        arrow_schema_meta = parquet_file.schema.to_arrow_schema().metadata or {}
        file_meta = (
            parquet_file.metadata.metadata
            if (parquet_file.metadata and parquet_file.metadata.metadata)
            else {}
        )
        combined_meta = {**file_meta, **arrow_schema_meta}

        if b"pipeline_config" in combined_meta:
            raw_json = combined_meta[b"pipeline_config"]
            config = json.loads(raw_json.decode("utf-8"))

            timestamp = config.get("generated_timestamp", "N/A")
            state_w = config.get("savgol_state_window", "N/A")
            body_w = config.get("savgol_body_rate_window", "N/A")

            footer_text += (
                f" | Generated: {timestamp} | SavGol [State N={state_w}, Body N={body_w}]"
            )
        else:
            footer_text += " | Config: N/A (No embedded pipeline_config)"

    except Exception as e:
        footer_text += f" | (Metadata error: {type(e).__name__} - {e})"

    return footer_text


def add_plot_footer(
    fig: plt.Figure,
    parquet_path: Path,
    source_file: str = "four_class_derived_dataset.parquet",
):
    """Appends a standardized metadata footer to the bottom of a Matplotlib figure."""
    meta_info = get_parquet_footer_text(parquet_path)
    footer_str = f"Source: {source_file}  •  {meta_info}"

    plt.figtext(
        0.01,
        0.005,
        footer_str,
        ha="left",
        va="bottom",
        fontsize=7,
        fontfamily="monospace",
        color="#444444",
        alpha=0.85,
    )


# ==============================================================================
# SWEEP EVALUATION & HEATMAPS
# ==============================================================================
def generate_sweep_report_and_heatmaps(
    database_dir: Path, output_dir: Path, timestamp: str
) -> pd.DataFrame:
    """Scans all sweep Parquet files, outputs summary metrics, and generates 5x5 Heatmaps."""
    parquet_files = sorted(
        list(database_dir.glob("four_class_derived_dataset_savgol_*.parquet"))
    )

    if not parquet_files:
        print(f"⚠️ No sweep Parquet files found in {database_dir}")
        return pd.DataFrame()

    print(f"🔍 Analyzing {len(parquet_files)} Parquet sweep files for metrics table...")

    records = []
    for filepath in parquet_files:
        try:
            parquet_file = pq.ParquetFile(filepath)
            arrow_schema_meta = (
                parquet_file.schema.to_arrow_schema().metadata or {}
            )
            file_meta = (
                parquet_file.metadata.metadata
                if (parquet_file.metadata and parquet_file.metadata.metadata)
                else {}
            )
            combined = {**file_meta, **arrow_schema_meta}

            if b"pipeline_config" not in combined:
                continue

            config = json.loads(combined[b"pipeline_config"].decode("utf-8"))
            n_state = config.get("savgol_state_window", None)
            n_body = config.get("savgol_body_rate_window", None)

            if n_state is None or n_body is None:
                continue

            df = pd.read_parquet(filepath, columns=TARGET_FEATURES)

            row_entry = {
                "file_name": filepath.name,
                "N_state": int(n_state),
                "N_body": int(n_body),
            }

            for col in TARGET_FEATURES:
                if isinstance(df[col].iloc[0], (np.ndarray, list)):
                    series = pd.Series(np.concatenate(df[col].values))
                else:
                    series = df[col]

                series = series.dropna()

                kurt_val = stats.kurtosis(series, fisher=False)
                max_abs = np.max(np.abs(series))

                row_entry[f"{col}_kurtosis"] = round(float(kurt_val), 2)
                row_entry[f"{col}_max_abs"] = round(float(max_abs), 4)

            records.append(row_entry)

        except Exception as e:
            print(f"⚠️ Error reading {filepath.name}: {e}")

    if not records:
        return pd.DataFrame()

    results_df = (
        pd.DataFrame(records)
        .sort_values(by=["N_state", "N_body"])
        .drop_duplicates(subset=["N_state", "N_body"], keep="last")  # <--- Deduplicate here
        .reset_index(drop=True)
    )

    # Save timestamped CSV and Markdown report
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = output_dir / f"savgol_sweep_metrics_{timestamp}.csv"
    md_out = output_dir / f"savgol_sweep_metrics_{timestamp}.md"

    results_df.to_csv(csv_out, index=False)
    with open(md_out, "w") as f:
        f.write(results_df.to_markdown(index=False))

    print(f"📊 Saved sweep metrics summary table: {csv_out}")

    # Render Heatmap plot
    if len(results_df) >= 4:
        heatmap_out = output_dir / f"savgol_kurtosis_heatmaps_{timestamp}.png"
        plot_kurtosis_heatmaps(results_df, output_path=heatmap_out)

    return results_df


def plot_kurtosis_heatmaps(results_df: pd.DataFrame, output_path: Path):
    """Renders 5x5 Heatmaps of Kurtosis across State and Body window sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    vspd_pivot = results_df.pivot(
        index="N_state", columns="N_body", values="vspd_fps_kurtosis"
    )
    qdot_pivot = results_df.pivot(
        index="N_state", columns="N_body", values="q_dot_kurtosis"
    )

    sns.heatmap(
        vspd_pivot,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Kurtosis (Target ≈ 3.0)"},
        ax=axes[0],
    )
    axes[0].set_title(
        rf"A. Vertical Speed ($\dot{{h}}$) Kurtosis Matrix",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_xlabel(r"Body Window Size ($N_{\text{body}}$)", fontsize=10)
    axes[0].set_ylabel(r"State Window Size ($N_{\text{state}}$)", fontsize=10)

    sns.heatmap(
        qdot_pivot,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        cbar_kws={"label": "Kurtosis"},
        ax=axes[1],
    )
    axes[1].set_title(
        rf"B. Pitch Acceleration ($\dot{{q}}$) Kurtosis Matrix",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_xlabel(r"Body Window Size ($N_{\text{body}}$)", fontsize=10)
    axes[1].set_ylabel(r"State Window Size ($N_{\text{state}}$)", fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.figtext(
        0.01,
        0.005,
        "Source: DASHlink Derived Parquet Sweep  •  Okabe-Ito Compliant  •  Target Kurtosis ≈ 3.0",
        ha="left",
        va="bottom",
        fontsize=7,
        fontfamily="monospace",
        color="#444444",
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"🖼️ Saved Kurtosis Heatmap plot to {output_path}")
    plt.close(fig)


# ==============================================================================
# INTEGRATED 4-PANEL DIAGNOSTIC TIME-SERIES PLOT
# ==============================================================================
def plot_savgol_diagnostics_4panel(
    parquet_path: Path, output_image_path: Path, flight_idx: int = 10
):
    """Plots pre-computed feature time-series directly from Parquet file using Okabe-Ito colors."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found at {parquet_path}")

    print(f"📖 Loading data from {parquet_path.name}...")

    df = pd.read_parquet(
        parquet_path,
        columns=["vspd_fps", "p_dot", "q_dot", "r_dot", "Pitch Angle LSP", "q"],
    )

    vspd = df["vspd_fps"].iloc[flight_idx]
    q_dot = df["q_dot"].iloc[flight_idx]
    p_dot = df["p_dot"].iloc[flight_idx]
    r_dot = df["r_dot"].iloc[flight_idx]
    q = df["q"].iloc[flight_idx]
    time = np.arange(len(vspd))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    # Panel A: Vertical Speed
    axes[0, 0].plot(
        time,
        vspd,
        color=OKABE_ITO["blue"],
        linewidth=1.5,
        label=r"Vertical Speed ($\dot{h}$)",
    )
    axes[0, 0].set_title(
        rf"A. State Derivative Differentiation: Vertical Speed $\dot{{h}}$ (Flight #{flight_idx})",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 0].set_ylabel("ft / s", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="upper right")

    # Panel B: Pitch Acceleration
    axes[0, 1].plot(
        time,
        q_dot,
        color=OKABE_ITO["vermillion"],
        linewidth=1.5,
        label=r"Pitch Accel ($\dot{q}$)",
    )
    axes[0, 1].set_title(
        rf"B. Body Rate Differentiation: Pitch Acceleration $\dot{{q}}$ (Flight #{flight_idx})",
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 1].set_ylabel(r"deg / $\text{s}^2$", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="upper right")

    # Panel C: Body Angular Rates (q)
    axes[1, 0].plot(
        time,
        q,
        color=OKABE_ITO["green"],
        linewidth=1.5,
        label=r"Pitch Rate ($q$)",
    )
    axes[1, 0].set_title(
        rf"C. Body Rate: Pitch Rate $q$ (Flight #{flight_idx})",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Time (seconds)", fontsize=10)
    axes[1, 0].set_ylabel("deg / s", fontsize=10)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="upper right")

    # Panel D: Roll and Yaw Acceleration
    axes[1, 1].plot(
        time,
        p_dot,
        color=OKABE_ITO["purple"],
        linewidth=1.2,
        alpha=0.85,
        label=r"Roll Accel ($\dot{p}$)",
    )
    axes[1, 1].plot(
        time,
        r_dot,
        color=OKABE_ITO["orange"],
        linewidth=1.2,
        alpha=0.85,
        label=r"Yaw Accel ($\dot{r}$)",
    )
    axes[1, 1].set_title(
        rf"D. Body Accelerations: $\dot{{p}}$ & $\dot{{r}}$ (Flight #{flight_idx})",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Time (seconds)", fontsize=10)
    axes[1, 1].set_ylabel(r"deg / $\text{s}^2$", fontsize=10)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    add_plot_footer(fig, parquet_path=parquet_path)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    print(f"✅ 4-Panel diagnostic plot saved to: {output_image_path}")
    plt.close(fig)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    env = LoadEnvironment()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define directories via LoadEnvironment
    database_dir = env.DATABASE_DIR
    reports_figures_dir = env.OUTPUT_DIR / "reports" / "figures"
    reports_figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run sweep evaluation and generate Heatmaps + CSV/MD reports
    generate_sweep_report_and_heatmaps(
        database_dir=database_dir,
        output_dir=reports_figures_dir,
        timestamp=timestamp,
    )

    # 2. Select target Parquet file for 4-panel diagnostic figure
    # Scans for latest generated file or defaults to primary symlink
    parquet_candidates = sorted(
        list(database_dir.glob("four_class_derived_dataset_savgol_*.parquet"))
    )

    # always use symlink
    target_parquet = database_dir / "four_class_derived_dataset.parquet"
    if not target_parquet.exists():
        raise FileNotFoundError(f"Active dataset symlink not found at {target_parquet}")

    out_plot = (
        reports_figures_dir
        / f"savgol_window_diagnostics_4panel_{timestamp}.png"
    )

    # 3. Generate 4-panel time-series diagnostic plot
    plot_savgol_diagnostics_4panel(
        parquet_path=target_parquet,
        output_image_path=out_plot,
        flight_idx=10,
    )