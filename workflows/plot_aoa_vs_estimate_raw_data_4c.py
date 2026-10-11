import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.palette_manager import PaletteManager, theme


def diagnose_t100_spike(
    parquet_file: str | Path,
    target_step: int = 100,
    save_path: str | Path | None = None,
):
    PaletteManager.apply_publication_style()
    df = pd.read_parquet(
        parquet_file,
        columns=[
            "Pitch Angle LSP",
            "vspd_fps",
            "Computed Airspeed LSP",
            "alpha_est",
            "res_flight_path",
        ],
    )

    pitch = np.vstack(df["Pitch Angle LSP"].values)
    vspd = np.vstack(df["vspd_fps"].values)
    speed = np.vstack(df["Computed Airspeed LSP"].values) * 1.68781  # knots -> ft/s
    alpha_est = np.vstack(df["alpha_est"].values)
    res_fp = np.vstack(df["res_flight_path"].values)

    time_steps = np.arange(pitch.shape[1])

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(PaletteManager.PAGE_WIDTH_INCHES, 7.5), sharex=True, dpi=300
    )

    # Panel 1: Derived Quantities
    ax1.plot(
        time_steps,
        np.mean(alpha_est, axis=0),
        color=PaletteManager.OKABE_ITO[4],
        label=r"$\hat{\alpha}$ (deg)",
    )
    ax1.plot(
        time_steps,
        np.mean(res_fp, axis=0),
        color=PaletteManager.OKABE_ITO[2],
        label=r"$res\_flight\_path$ (ft/s)",
    )
    ax1.axvline(target_step, color="crimson", linestyle=":", alpha=0.8, label=f"t = {target_step}s")
    ax1.set_title("t = 100s Artifact Diagnostics")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Vertical Speed (vspd_fps)
    ax2.plot(
        time_steps,
        np.mean(vspd, axis=0),
        color=PaletteManager.OKABE_ITO[0],
        label=r"$V_z$ (vspd_fps)",
    )
    ax2.axvline(target_step, color="crimson", linestyle=":", alpha=0.8)
    ax2.set_ylabel("ft/s")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Pitch Angle & Airspeed
    ax3.plot(
        time_steps,
        np.mean(pitch, axis=0),
        color=PaletteManager.OKABE_ITO[1],
        label=r"Pitch $\theta$ (deg)",
    )
    ax3.plot(
        time_steps,
        np.mean(speed, axis=0) / 10.0,
        color=PaletteManager.OKABE_ITO[7],
        linestyle="--",
        label=r"Airspeed $V_t$ / 10 (ft/s)",
    )
    ax3.axvline(target_step, color="crimson", linestyle=":", alpha=0.8)
    ax3.set_xlabel("Window Time Step (Seconds at 1 Hz)")
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    PARQUET_FILE = "/mnt/fast_storage/projects/aviation_insight/github_examples/dashlink-anomaly-data/dataset/four_class_derived_dataset.parquet"
    diagnose_t100_spike(PARQUET_FILE)