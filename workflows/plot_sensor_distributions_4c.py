import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.palette_manager import PaletteManager, theme

def plot_aoa_signals_boxplot(
    df: pd.DataFrame,
    flight_ids: np.ndarray,
    missions: np.ndarray,
    time_steps: np.ndarray,
    window_size: int = 160,
    aoa_col: str = "Corrected Angle of Attack",
    alpha_est_col: str = "alpha_est",
    save_path: str | Path | None = None,
):
    """Plots a 2-panel figure specifically for Angle of Attack signals (Measured vs Estimate)."""
    matrices = {
        "Measured AOA\n(deg)": np.vstack(df[aoa_col].values),
        r"Kinematic Estimate $\hat{\alpha}$" + "\n(deg)": np.vstack(df[alpha_est_col].values),
    }

    colors = [PaletteManager.OKABE_ITO[0], PaletteManager.OKABE_ITO[4]]  # Blue, Vermilion

    fig, axes = plt.subplots(
        2, 1,
        figsize=(PaletteManager.PAGE_WIDTH_INCHES, 6.0),
        sharex=True,
        dpi=PaletteManager.DPI
    )

    for ax, (label_text, mat), color in zip(axes, matrices.items(), colors):
        q25 = np.percentile(mat, 25, axis=0)
        q50 = np.percentile(mat, 50, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        iqr = q75 - q25

        whisker_low = np.maximum(np.min(mat, axis=0), q25 - 1.5 * iqr)
        whisker_high = np.minimum(np.max(mat, axis=0), q75 + 1.5 * iqr)

        # Plot Median line, IQR Box (25th-75th), and Whiskers
        ax.plot(time_steps, q50, color=color, linewidth=1.5, label="Median ($Q_{50}$)")
        ax.fill_between(time_steps, q25, q75, color=color, alpha=0.35, label="IQR ($Q_{25}$–$Q_{75}$)")
        ax.plot(time_steps, whisker_high, color=color, linestyle="--", linewidth=0.8, label="Whiskers (1.5 IQR)")
        ax.plot(time_steps, whisker_low, color=color, linestyle="--", linewidth=0.8)

        # Annotate global minimum outlier with Mission & Flight ID
        min_idx = np.unravel_index(np.argmin(mat), mat.shape)
        row_min, step_min = min_idx
        val_min = mat[row_min, step_min]
        fid_min = flight_ids[row_min]
        mission_min = missions[row_min]

        ax.scatter([step_min], [val_min], color="crimson", s=25, zorder=5)
        ax.annotate(
            f"Mission: {mission_min} | ID: {fid_min}\n({val_min:.1f}° at t={step_min}s)",
            xy=(step_min, val_min),
            xytext=(step_min + 3, val_min + (0.15 * abs(val_min) if val_min < 0 else -0.15 * abs(val_min))),
            fontsize=7,
            fontweight="bold",
            color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
        )

        ax.set_ylabel(label_text, fontsize=8, multialignment="center", labelpad=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=7)

    axes[-1].set_xlabel("Window Time Step (Seconds at 1 Hz)")
    axes[-1].set_xlim(0, window_size - 1)
    
    fig.suptitle("Angle of Attack Temporal Envelopes & Outliers", y=0.98, fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=PaletteManager.DPI, bbox_inches="tight")
    plt.show()


def plot_residuals_signals_boxplot(
    df: pd.DataFrame,
    flight_ids: np.ndarray,
    missions: np.ndarray,
    time_steps: np.ndarray,
    window_size: int = 160,
    res_fp_col: str = "res_flight_path",
    res_fp_ae_col: str = "res_flight_path_ae",
    save_path: str | Path | None = None,
):
    """Plots a 2-panel figure specifically for Flight Path Residual signals (Raw vs Est AOA)."""
    matrices = {
        r"Raw Flight Path Res ($res\_fp$)" + "\n(ft/s)": np.vstack(df[res_fp_col].values),
        r"Est Flight Path Res ($res\_fp\_ae$)" + "\n(ft/s)": np.vstack(df[res_fp_ae_col].values),
    }

    colors = [PaletteManager.OKABE_ITO[2], PaletteManager.OKABE_ITO[5]]  # Green, Purple

    fig, axes = plt.subplots(
        2, 1,
        figsize=(PaletteManager.PAGE_WIDTH_INCHES, 6.0),
        sharex=True,
        dpi=PaletteManager.DPI
    )

    for ax, (label_text, mat), color in zip(axes, matrices.items(), colors):
        q25 = np.percentile(mat, 25, axis=0)
        q50 = np.percentile(mat, 50, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        iqr = q75 - q25

        whisker_low = np.maximum(np.min(mat, axis=0), q25 - 1.5 * iqr)
        whisker_high = np.minimum(np.max(mat, axis=0), q75 + 1.5 * iqr)

        # Plot Median line, IQR Box (25th-75th), and Whiskers
        ax.plot(time_steps, q50, color=color, linewidth=1.5, label="Median ($Q_{50}$)")
        ax.fill_between(time_steps, q25, q75, color=color, alpha=0.35, label="IQR ($Q_{25}$–$Q_{75}$)")
        ax.plot(time_steps, whisker_high, color=color, linestyle="--", linewidth=0.8, label="Whiskers (1.5 IQR)")
        ax.plot(time_steps, whisker_low, color=color, linestyle="--", linewidth=0.8)

        # Annotate worst outlier (checking both max and min for extreme residual spikes)
        max_abs_idx = np.unravel_index(np.argmax(np.abs(mat)), mat.shape)
        row_out, step_out = max_abs_idx
        val_out = mat[row_out, step_out]
        fid_out = flight_ids[row_out]
        mission_out = missions[row_out]

        ax.scatter([step_out], [val_out], color="crimson", s=25, zorder=5)
        ax.annotate(
            f"Mission: {mission_out} | ID: {fid_out}\n({val_out:.1f} ft/s at t={step_out}s)",
            xy=(step_out, val_out),
            xytext=(step_out + 3, val_out - (0.2 * val_out if val_out > 0 else -0.2 * abs(val_out))),
            fontsize=7,
            fontweight="bold",
            color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
        )

        ax.set_ylabel(label_text, fontsize=8, multialignment="center", labelpad=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=7)

    axes[-1].set_xlabel("Window Time Step (Seconds at 1 Hz)")
    axes[-1].set_xlim(0, window_size - 1)
    
    fig.suptitle("Flight Path Residual Temporal Envelopes & Outliers", y=0.98, fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=PaletteManager.DPI, bbox_inches="tight")
    plt.show()


def run_separated_plots(
    parquet_file: str | Path,
    window_size: int = 160,
    flight_id_col: str = "flight_record",
    mission_col: str = "mission_id",
):
    parquet_path = Path(parquet_file)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

    PaletteManager.apply_publication_style()

    cols = [
        "Corrected Angle of Attack",
        "alpha_est",
        "res_flight_path",
        "res_flight_path_ae",
        flight_id_col,
        mission_col,
    ]
    df = pd.read_parquet(parquet_path, columns=cols)

    flight_ids = df[flight_id_col].values
    missions = df[mission_col].values
    time_steps = np.arange(window_size)

    # Generate Figure 1: Angle of Attack
    plot_aoa_signals_boxplot(
        df, flight_ids, missions, time_steps, window_size=window_size
    )

    # Generate Figure 2: Residuals
    plot_residuals_signals_boxplot(
        df, flight_ids, missions, time_steps, window_size=window_size
    )


if __name__ == "__main__":
    PARQUET_FILE = "/mnt/fast_storage/projects/aviation_insight/github_examples/dashlink-anomaly-data/dataset/four_class_derived_dataset.parquet"
    run_separated_plots(PARQUET_FILE)
