from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define Okabe-Ito Color Palette for Quarto publishing standards
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


def plot_kurtosis_heatmaps(results_df: pd.DataFrame, output_path: Path):
    """Generates 5x5 Kurtosis Heatmaps for State and Body acceleration channels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Pivot data into 5x5 grids
    vspd_pivot = results_df.pivot(
        index="N_state", columns="N_body", values="vspd_fps_kurtosis"
    )
    qdot_pivot = results_df.pivot(
        index="N_state", columns="N_body", values="q_dot_kurtosis"
    )

    # Subplot A: Vertical Speed Kurtosis
    sns.heatmap(
        vspd_pivot,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Kurtosis (Target ≈ 3.0)"},
        ax=axes[0],
    )
    axes[0].set_title(
        r"A. State Derivative: Vertical Speed $\dot{h}$ Kurtosis",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_xlabel(r"Body Window Size ($N_{\text{body}}$)", fontsize=10)
    axes[0].set_ylabel(r"State Window Size ($N_{\text{state}}$)", fontsize=10)

    # Subplot B: Pitch Acceleration Kurtosis
    sns.heatmap(
        qdot_pivot,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        cbar_kws={"label": "Kurtosis"},
        ax=axes[1],
    )
    axes[1].set_title(
        r"B. Body Acceleration: Pitch Accel $\dot{q}$ Kurtosis",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_xlabel(r"Body Window Size ($N_{\text{body}}$)", fontsize=10)
    axes[1].set_ylabel(r"State Window Size ($N_{\text{state}}$)", fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # Append Okabe-Ito styling and footer
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