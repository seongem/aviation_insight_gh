import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.palette_manager import PaletteManager, theme

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_aoa_and_flight_path_residuals_3panel(
    parquet_file: str | Path,
    window_size: int = 160,
    aoa_col: str = "Corrected Angle of Attack",
    alpha_est_col: str = "alpha_est",
    res_fp_col: str = "res_flight_path",
    res_fp_ae_col: str = "res_flight_path_ae",
    save_path: str | Path | None = None,
):
    """Loads pre-computed AOA and flight path residual vectors from Parquet

    and visualizes them across three separate subplots with Q25–Q75 and Min–Max envelopes.
    """
    parquet_path = Path(parquet_file)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

    # 1. Apply centralized publication matplotlibrc styling
    PaletteManager.apply_publication_style()

    # 2. Read pre-computed channels directly from Parquet
    required_cols = [aoa_col, alpha_est_col, res_fp_col, res_fp_ae_col]
    df = pd.read_parquet(parquet_path, columns=required_cols)

    # 3. Extract and stack 1D array columns into 2D matrices (N_flights, Window_Size)
    aoa_matrix = np.vstack(df[aoa_col].values)
    alpha_est_matrix = np.vstack(df[alpha_est_col].values)
    res_fp_matrix = np.vstack(df[res_fp_col].values)
    res_fp_ae_matrix = np.vstack(df[res_fp_ae_col].values)

    # 4. Compute Mean, Q25, Q75, Min, and Max across flights (axis=0)
    time_steps = np.arange(window_size)

    # Panel 1: Angle of Attack stats
    aoa_mean = np.mean(aoa_matrix, axis=0)
    aoa_q25 = np.percentile(aoa_matrix, 25, axis=0)
    aoa_q75 = np.percentile(aoa_matrix, 75, axis=0)
    aoa_min = np.min(aoa_matrix, axis=0)
    aoa_max = np.max(aoa_matrix, axis=0)

    alpha_est_mean = np.mean(alpha_est_matrix, axis=0)
    alpha_est_q25 = np.percentile(alpha_est_matrix, 25, axis=0)
    alpha_est_q75 = np.percentile(alpha_est_matrix, 75, axis=0)
    alpha_est_min = np.min(alpha_est_matrix, axis=0)
    alpha_est_max = np.max(alpha_est_matrix, axis=0)

    # Panel 2: Raw Flight Path Residual stats
    res_fp_mean = np.mean(res_fp_matrix, axis=0)
    res_fp_q25 = np.percentile(res_fp_matrix, 25, axis=0)
    res_fp_q75 = np.percentile(res_fp_matrix, 75, axis=0)
    res_fp_min = np.min(res_fp_matrix, axis=0)
    res_fp_max = np.max(res_fp_matrix, axis=0)

    # Panel 3: Estimated AOA Flight Path Residual stats
    res_fp_ae_mean = np.mean(res_fp_ae_matrix, axis=0)
    res_fp_ae_q25 = np.percentile(res_fp_ae_matrix, 25, axis=0)
    res_fp_ae_q75 = np.percentile(res_fp_ae_matrix, 75, axis=0)
    res_fp_ae_min = np.min(res_fp_ae_matrix, axis=0)
    res_fp_ae_max = np.max(res_fp_ae_matrix, axis=0)

    # 5. Extract Okabe-Ito colors from PaletteManager
    c_blue = PaletteManager.OKABE_ITO[0]       # Measured AOA
    c_vermilion = PaletteManager.OKABE_ITO[4]  # Estimated AOA
    c_green = PaletteManager.OKABE_ITO[2]      # Raw Flight Path Residual
    c_purple = PaletteManager.OKABE_ITO[5]     # Estimated Flight Path Residual

    # 6. Plot 3 stacked panels
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, 
        figsize=(PaletteManager.PAGE_WIDTH_INCHES, 8.0), 
        sharex=True, 
        dpi=PaletteManager.DPI
    )

    # ---------------- PANEL 1: AOA Comparison ----------------
    # Measured AOA
    ax1.plot(time_steps, aoa_mean, color=c_blue, linewidth=1.5, label="Measured Corrected AOA")
    ax1.fill_between(time_steps, aoa_q25, aoa_q75, color=c_blue, alpha=0.25, label="Measured Q25–Q75")
    ax1.fill_between(time_steps, aoa_min, aoa_max, color=c_blue, alpha=0.08, label="Measured Min–Max")
    ax1.plot(time_steps, aoa_min, color=c_blue, linestyle=":", linewidth=0.75, alpha=0.5)
    ax1.plot(time_steps, aoa_max, color=c_blue, linestyle=":", linewidth=0.75, alpha=0.5)

    # Estimated AOA
    ax1.plot(time_steps, alpha_est_mean, color=c_vermilion, linestyle="--", linewidth=1.5, label=r"Kinematic Estimate ($\hat{\alpha}$)")
    ax1.fill_between(time_steps, alpha_est_q25, alpha_est_q75, color=c_vermilion, alpha=0.2, label="Estimate Q25–Q75")
    ax1.fill_between(time_steps, alpha_est_min, alpha_est_max, color=c_vermilion, alpha=0.06, label="Estimate Min–Max")
    ax1.plot(time_steps, alpha_est_min, color=c_vermilion, linestyle=":", linewidth=0.75, alpha=0.5)
    ax1.plot(time_steps, alpha_est_max, color=c_vermilion, linestyle=":", linewidth=0.75, alpha=0.5)

    ax1.set_title("Approach Windows: Angle of Attack and Kinematic Residual Analysis", pad=8)
    ax1.set_ylabel("AOA (deg)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9, ncol=2)

    # ---------------- PANEL 2: Raw Flight Path Residual ----------------
    ax2.plot(time_steps, res_fp_mean, color=c_green, linewidth=1.5, label=r"Raw Flight Path Residual ($res\_flight\_path$)")
    ax2.fill_between(time_steps, res_fp_q25, res_fp_q75, color=c_green, alpha=0.25, label="Q25–Q75")
    ax2.fill_between(time_steps, res_fp_min, res_fp_max, color=c_green, alpha=0.08, label="Min–Max")
    ax2.plot(time_steps, res_fp_min, color=c_green, linestyle=":", linewidth=0.75, alpha=0.5)
    ax2.plot(time_steps, res_fp_max, color=c_green, linestyle=":", linewidth=0.75, alpha=0.5)

    ax2.set_ylabel("Raw Residual (ft/s)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", framealpha=0.9, ncol=3)

    # ---------------- PANEL 3: Estimated AOA Flight Path Residual ----------------
    ax3.plot(time_steps, res_fp_ae_mean, color=c_purple, linewidth=1.5, label=r"Estimated AOA Residual ($res\_flight\_path\_ae$)")
    ax3.fill_between(time_steps, res_fp_ae_q25, res_fp_ae_q75, color=c_purple, alpha=0.25, label="Q25–Q75")
    ax3.fill_between(time_steps, res_fp_ae_min, res_fp_ae_max, color=c_purple, alpha=0.08, label="Min–Max")
    ax3.plot(time_steps, res_fp_ae_min, color=c_purple, linestyle=":", linewidth=0.75, alpha=0.5)
    ax3.plot(time_steps, res_fp_ae_max, color=c_purple, linestyle=":", linewidth=0.75, alpha=0.5)

    ax3.set_xlabel("Window Time Step (Seconds at 1 Hz)")
    ax3.set_ylabel("Estimated Residual (ft/s)")
    ax3.set_xlim(0, window_size - 1)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", framealpha=0.9, ncol=3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=PaletteManager.DPI, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    PARQUET_FILE = "/mnt/fast_storage/projects/aviation_insight/github_examples/dashlink-anomaly-data/dataset/four_class_derived_dataset.parquet"
    plot_aoa_and_flight_path_residuals_3panel(PARQUET_FILE)