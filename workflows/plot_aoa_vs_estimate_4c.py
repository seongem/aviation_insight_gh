from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_aoa_vs_gamma_estimate(
    parquet_file: str | Path,
    window_size: int = 160,
    aoa_col: str = "Corrected Angle of Attack",
    vspd_col: str = "vspd_fps",
    speed_col: str = "Computed Airspeed LSP",
    pitch_col: str = "Pitch Angle LSP",
):
    """Loads a single Parquet file where cells contain 1D array telemetry vectors,

    computes kinematic AOA, reverts measured AOA sign, and plots aggregated envelopes.
    """
    parquet_path = Path(parquet_file)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

    # 1. Read single Parquet file
    required_cols = [aoa_col, vspd_col, speed_col, pitch_col]
    df = pd.read_parquet(parquet_path, columns=required_cols)

    # 2. Extract and stack 1D array columns into 2D matrices (N_flights, Window_Size)
    # Shape of each matrix: (N, 160)
    aoa_matrix = np.vstack(df[aoa_col].values)
    vspd_matrix = np.vstack(df[vspd_col].values)
    speed_matrix = np.vstack(df[speed_col].values)
    pitch_matrix = np.vstack(df[pitch_col].values)

    # 3. Revert Corrected AOA sign back to original physical orientation (-1 * value)
    aoa_matrix_reverted = 1.0 * aoa_matrix

    # 4. Vectorized Kinematic Calculation across all flights & time steps
    # Convert knots -> ft/s (1 knot = 1.68781 ft/s)
    speed_fps = speed_matrix * 1.68781

    # Clamp sin value to [-1, 1] for numerical safety during arcsin
    sin_gamma = np.clip(vspd_matrix / np.maximum(speed_fps, 1.0), -1.0, 1.0)
    gamma_deg = np.arcsin(sin_gamma) * (180.0 / np.pi)

    # Kinematic AOA estimate: Pitch - Gamma
    gamma_aoa_matrix = pitch_matrix - gamma_deg

    # 5. Compute mean and Q25/Q75 across flights (axis=0) for each time step (0..159)
    time_steps = np.arange(window_size)

    aoa_mean = np.mean(aoa_matrix_reverted, axis=0)
    aoa_q25 = np.percentile(aoa_matrix_reverted, 25, axis=0)
    aoa_q75 = np.percentile(aoa_matrix_reverted, 75, axis=0)

    gamma_mean = np.mean(gamma_aoa_matrix, axis=0)
    gamma_q25 = np.percentile(gamma_aoa_matrix, 25, axis=0)
    gamma_q75 = np.percentile(gamma_aoa_matrix, 75, axis=0)

    # 6. Plotting Envelopes
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    # Reverted Measured AOA
    ax.plot(
        time_steps,
        aoa_mean,
        color="#1f77b4",
        linewidth=2,
        label="Measured AOA (Reverted Sign)",
    )
    ax.fill_between(
        time_steps,
        aoa_q25,
        aoa_q75,
        color="#1f77b4",
        alpha=0.25,
        label="Measured AOA (Q25–Q75)",
    )

    # Derived Kinematic AOA
    ax.plot(
        time_steps,
        gamma_mean,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=r"Kinematic Estimate ($\theta - \gamma$)",
    )
    ax.fill_between(
        time_steps,
        gamma_q25,
        gamma_q75,
        color="#d62728",
        alpha=0.2,
        label=r"Kinematic Estimate (Q25–Q75)",
    )

    # Formatting
    ax.set_title(
        r"160s Approach Windows: Corrected AOA (Original Sign) vs. Kinematic $\theta - \gamma$ Estimate",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("Window Time Step (Seconds at 1 Hz)", fontsize=11)
    ax.set_ylabel("Angle of Attack (deg)", fontsize=11)
    ax.set_xlim(0, window_size - 1)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    PARQUET_FILE = "/mnt/fast_storage/projects/aviation_insight/github_examples/dashlink-anomaly-data/dataset/four_class_derived_dataset.parquet"
    plot_aoa_vs_gamma_estimate(PARQUET_FILE)