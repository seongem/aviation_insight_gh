import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set project root
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the centralized PaletteManager
from src.utils.palette_manager import PaletteManager

PARQUET_PATH = Path("/mnt/fast_storage/projects/aviation_insight/github_examples/dashlink-anomaly-data/dataset/four_class_derived_dataset.parquet")
OUTPUT_FIG_PATH = Path("/mnt/fast_storage/projects/aviation_insight/workflows/EDA/alpha_est_go_around_diagnostic.png")

logger = logging.getLogger("AviationInsight_Plotting")

def generate_aoa_saturation_8panel_plot():
    # Apply publication styling rules from PaletteManager
    PaletteManager.apply_publication_style()
    
    df = pd.read_parquet(PARQUET_PATH)
    nominal_df = df[df["label"] == 0]

    # Find target flight instance with low alpha_est artifact
    target_idx = None
    for idx, row in nominal_df.iterrows():
        alpha_est = np.asarray(row["alpha_est"])
        if np.min(alpha_est) < -40.0:
            target_idx = idx
            break

    if target_idx is None:
        print("⚠️ Could not locate alpha_est < -40 in nominal split, defaulting to first record.")
        target_idx = nominal_df.index[0]

    row = nominal_df.loc[target_idx]
    record_label = row.get("label", "Unknown")
    flight_rec = row.get("flight_record", row.get("flight_id", f"Index_{target_idx}"))
    print(f"🔍 Selected target_idx: {target_idx} | Label: {record_label} | Flight Record: {flight_rec}")

    time_axis = np.arange(160)
    
    # Trajectory & Kinematic Channels
    vt = np.asarray(row["Computed Airspeed LSP"])
    vspd = np.asarray(row["vspd_fps"]) if "vspd_fps" in row else np.zeros(160)
    alt = np.asarray(row["Baro Correct Altitude LSP"]) if "Baro Correct Altitude LSP" in row else np.zeros(160)
    raw_aoa = np.asarray(row["Corrected Angle of Attack"])
    alpha_est = np.asarray(row["alpha_est"])

    # Physical Verification Channels
    vert_accel = np.asarray(row["Vertical Acceleration"]) if "Vertical Acceleration" in row else np.zeros(160)
    total_press = np.asarray(row["Total Pressure LSP"]) if "Total Pressure LSP" in row else np.zeros(160)

    # Control Surfaces & Engine
    elevator = np.asarray(row["Elevator Position Left"]) if "Elevator Position Left" in row else np.zeros(160)
    flaps = np.asarray(row["T.E. Flap Position"]) if "T.E. Flap Position" in row else np.zeros(160)
    engine_rpm = np.asarray(row["Core Speed Avg"]) if "Core Speed Avg" in row else np.zeros(160)

    # Attitude, Guidance, and Wind Channels
    pitch = np.asarray(row["Pitch Angle LSP"]) if "Pitch Angle LSP" in row else np.zeros(160)
    roll = np.asarray(row["Roll Angle LSP"]) if "Roll Angle LSP" in row else np.zeros(160)
    true_hdg = np.asarray(row["True Heading LSP"]) if "True Heading LSP" in row else np.zeros(160)
    sel_hdg = np.asarray(row["Selected Heading"]) if "Selected Heading" in row else np.zeros(160)
    sel_crs = np.asarray(row["Selected Course"]) if "Selected Course" in row else np.zeros(160)
    wind_spd = np.asarray(row["Wind Speed"]) if "Wind Speed" in row else np.zeros(160)

    # Reconstruct flight path angle gamma
    vt_fps = vt * 1.68781
    vt_clamped = np.maximum(vt_fps, 30.0)
    sin_gamma = np.clip(vspd / vt_clamped, -1.0, 1.0)
    gamma_deg = np.rad2deg(np.arcsin(sin_gamma))

    # Construct 8-Panel Figure optimized for Quarto layout width (7.0")
    fig, axes = plt.subplots(8, 1, figsize=(PaletteManager.PAGE_WIDTH_INCHES, 18), sharex=True)
    (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = axes

    # Color Palette Mapping (Okabe-Ito + Crimson)
    c_blue = PaletteManager.OKABE_ITO[0]      # #0072B2
    c_orange = PaletteManager.OKABE_ITO[1]    # #E69F00
    c_green = PaletteManager.OKABE_ITO[2]     # #009E73
    c_skyblue = PaletteManager.OKABE_ITO[3]   # #56B4E9
    c_vermilion = PaletteManager.OKABE_ITO[4] # #D55E00
    c_purple = PaletteManager.OKABE_ITO[5]    # #CC79A7
    c_yellow = PaletteManager.OKABE_ITO[6]    # #F0E442
    c_black = PaletteManager.OKABE_ITO[7]     # #000000
    c_crimson = PaletteManager.CRIMSON        # #B22222

    # -------------------------------------------------------------------------
    # Panel 1: Trajectory Breakdown (Altitude vs. Vertical Speed)
    # -------------------------------------------------------------------------
    l1 = ax1.plot(time_axis, vspd, color=c_purple, linewidth=1.8, label=r"Vertical Speed $v_{spd}$ (ft/s)")
    ax1.set_ylabel(r"$v_{spd}$ (ft/s)", fontweight="bold", color=c_purple)
    ax1.tick_params(axis='y', labelcolor=c_purple)

    ax1_alt = ax1.twinx()
    l2 = ax1_alt.plot(time_axis, alt, color=c_green, linestyle="--", linewidth=1.6, label="Baro Corrected Altitude (ft)")
    ax1_alt.set_ylabel("Altitude (ft)", fontweight="bold", color=c_green)
    ax1_alt.tick_params(axis='y', labelcolor=c_green)

    lines1 = l1 + l2
    ax1.legend(lines1, [l.get_label() for l in lines1], loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax1.set_title(
        f"Flight Record: {flight_rec} | target_idx: {target_idx} | Label: {record_label} — Sensor Integrity & Anomaly Diagnostic", 
        fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 2: Physical Acceleration & Total Pressure
    # -------------------------------------------------------------------------
    a1 = ax2.plot(time_axis, vert_accel, color=c_blue, linewidth=1.8, label="Vertical Acceleration (g)")
    ax2.axhline(1.0, color=c_black, linestyle=":", alpha=0.6, label="1.0g Ref")
    ax2.set_ylabel("Accel (g)", fontweight="bold", color=c_blue)
    ax2.tick_params(axis='y', labelcolor=c_blue)

    ax2_press = ax2.twinx()
    p_line = ax2_press.plot(time_axis, total_press, color=c_vermilion, linestyle="-.", linewidth=1.6, label="Total Pressure LSP")
    ax2_press.set_ylabel("Total Press", fontweight="bold", color=c_vermilion)
    ax2_press.tick_params(axis='y', labelcolor=c_vermilion)

    lines_accel = a1 + p_line
    ax2.legend(lines_accel, [l.get_label() for l in lines_accel], loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 3: Aircraft Attitude (Pitch Angle & Roll Angle)
    # -------------------------------------------------------------------------
    pt = ax3.plot(time_axis, pitch, color=c_vermilion, linewidth=1.8, label="Pitch Angle LSP (deg)")
    ax3.set_ylabel("Pitch (deg)", fontweight="bold", color=c_vermilion)
    ax3.tick_params(axis='y', labelcolor=c_vermilion)

    ax3_roll = ax3.twinx()
    rl = ax3_roll.plot(time_axis, roll, color=c_skyblue, linestyle="--", linewidth=1.6, label="Roll Angle LSP (deg)")
    ax3_roll.set_ylabel("Roll (deg)", fontweight="bold", color=c_skyblue)
    ax3_roll.tick_params(axis='y', labelcolor=c_skyblue)

    lines_att = pt + rl
    ax3.legend(lines_att, [l.get_label() for l in lines_att], loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 4: Heading, Guidance & Winds
    # -------------------------------------------------------------------------
    h1 = ax4.plot(time_axis, true_hdg, color=c_black, linewidth=1.8, label="True Heading LSP (deg)")
    h2 = ax4.plot(time_axis, sel_hdg, color=c_blue, linestyle=":", linewidth=1.5, label="Selected Heading (deg)")
    h3 = ax4.plot(time_axis, sel_crs, color=c_green, linestyle="-.", linewidth=1.5, label="Selected Course (deg)")
    ax4.set_ylabel("Hdg / Crs (deg)", fontweight="bold")

    ax4_wind = ax4.twinx()
    w1 = ax4_wind.plot(time_axis, wind_spd, color=c_orange, linewidth=1.6, label="Wind Speed (kts)")
    ax4_wind.set_ylabel("Wind (kts)", fontweight="bold", color=c_orange)
    ax4_wind.tick_params(axis='y', labelcolor=c_orange)

    lines_hdg = h1 + h2 + h3 + w1
    ax4.legend(lines_hdg, [l.get_label() for l in lines_hdg], loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax4.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 5: Flight Controls (Elevator Deflection vs. Flap Position)
    # -------------------------------------------------------------------------
    e1 = ax5.plot(time_axis, elevator, color=c_blue, linewidth=1.8, label="Elevator Position Left (deg)")
    ax5.set_ylabel("Elevator (deg)", fontweight="bold", color=c_blue)
    ax5.tick_params(axis='y', labelcolor=c_blue)

    ax5_flap = ax5.twinx()
    f1 = ax5_flap.plot(time_axis, flaps, color=c_purple, linestyle="-.", linewidth=1.6, label="T.E. Flap Position (counts)")
    ax5_flap.set_ylabel("Flaps (counts)", fontweight="bold", color=c_purple)
    ax5_flap.tick_params(axis='y', labelcolor=c_purple)

    lines5 = e1 + f1
    ax5.legend(lines5, [l.get_label() for l in lines5], loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax5.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 6: Propulsion & Airspeed Response
    # -------------------------------------------------------------------------
    p1 = ax6.plot(time_axis, engine_rpm, color=c_crimson, linewidth=1.8, label="Core Speed Avg (% N1)")
    ax6.set_ylabel("Engine (% N1)", fontweight="bold", color=c_crimson)
    ax6.tick_params(axis='y', labelcolor=c_crimson)

    ax6_vt = ax6.twinx()
    p2 = ax6_vt.plot(time_axis, vt, color=c_black, linestyle=":", linewidth=1.6, label="Airspeed $V_T$ (kts)")
    ax6_vt.set_ylabel("Airspeed (kts)", fontweight="bold", color=c_black)
    ax6_vt.tick_params(axis='y', labelcolor=c_black)

    lines6 = p1 + p2
    ax6.legend(lines6, [l.get_label() for l in lines6], loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax6.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 7: Flight Path Angle Gamma Saturation
    # -------------------------------------------------------------------------
    ax7.plot(time_axis, gamma_deg, color=c_orange, linewidth=1.8, label=r"Flight Path Angle $\gamma = \arcsin(v_{spd} / V_T)$")
    ax7.axhline(90, color=c_crimson, linestyle="--", alpha=0.8, label=r"Saturation Boundary ($\gamma = +90^\circ$)")
    ax7.axhline(-90, color=c_crimson, linestyle="--", alpha=0.8, label=r"Saturation Boundary ($\gamma = -90^\circ$)")
    ax7.set_ylabel(r"$\gamma$ (deg)", fontweight="bold")
    ax7.grid(True, alpha=0.3)
    ax7.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 8: Derived vs. Raw Angle of Attack
    # -------------------------------------------------------------------------
    ax8.plot(time_axis, raw_aoa, color=c_crimson, linestyle="--", linewidth=1.6, label="Raw 'Corrected Angle of Attack'")
    ax8.plot(time_axis, alpha_est, color=c_blue, linewidth=1.8, label=r"Derived $\alpha_{est}$ ($\theta - \gamma$)")
    ax8.axhline(-58, color=c_crimson, linestyle=":", linewidth=1.5, label=r"Mathematical Saturation Artifact ($-58^\circ$)")
    ax8.set_ylabel("AoA (deg)", fontweight="bold")
    ax8.set_xlabel("Time Step in 160s Approach Window (seconds)", fontweight="bold")
    ax8.grid(True, alpha=0.3)
    ax8.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)

    # Highlight saturation zone using Okabe-Ito Yellow with gentle alpha overlay
    sat_indices = np.where(np.abs(gamma_deg) >= 80.0)[0]
    if len(sat_indices) > 0:
        start_f, end_f = max(0, sat_indices[0] - 2), min(159, sat_indices[-1] + 2)
        for ax in axes:
            ax.axvspan(start_f, end_f, color=c_yellow, alpha=0.35)

    plt.tight_layout()
    OUTPUT_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIG_PATH, dpi=PaletteManager.DPI, bbox_inches="tight")
    print(f"🎉 Publication-styled 8-panel diagnostic plot successfully saved to: {OUTPUT_FIG_PATH.resolve()}")

def generate_aoa_saturation_2panel_summary():
    """Generates a compact 2-panel figure featuring Panel 1 (Trajectory) 

    and Panel 8 (Derived AoA) for publication space constraints.
    """
    PaletteManager.apply_publication_style()
    
    df = pd.read_parquet(PARQUET_PATH)
    nominal_df = df[df["label"] == 0]

    target_idx = None
    for idx, row in nominal_df.iterrows():
        alpha_est = np.asarray(row["alpha_est"])
        if np.min(alpha_est) < -40.0:
            target_idx = idx
            break

    if target_idx is None:
        target_idx = nominal_df.index[0]

    row = nominal_df.loc[target_idx]
    record_label = row.get("label", "Unknown")
    flight_rec = row.get("flight_record", row.get("flight_id", f"Index_{target_idx}"))

    time_axis = np.arange(160)
    
    # Extract required series
    vt = np.asarray(row["Computed Airspeed LSP"])
    vspd = np.asarray(row["vspd_fps"]) if "vspd_fps" in row else np.zeros(160)
    alt = np.asarray(row["Baro Correct Altitude LSP"]) if "Baro Correct Altitude LSP" in row else np.zeros(160)
    raw_aoa = np.asarray(row["Corrected Angle of Attack"])
    alpha_est = np.asarray(row["alpha_est"])

    # Flight path angle gamma
    vt_fps = vt * 1.68781
    vt_clamped = np.maximum(vt_fps, 30.0)
    sin_gamma = np.clip(vspd / vt_clamped, -1.0, 1.0)
    gamma_deg = np.rad2deg(np.arcsin(sin_gamma))

    # Construct Compact 2-Panel Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(PaletteManager.PAGE_WIDTH_INCHES, 5.0), sharex=True)

    # Color definitions
    c_purple = PaletteManager.OKABE_ITO[5]
    c_green = PaletteManager.OKABE_ITO[2]
    c_blue = PaletteManager.OKABE_ITO[0]
    c_crimson = PaletteManager.CRIMSON
    c_yellow = PaletteManager.OKABE_ITO[6]

    # Panel 1: Trajectory Breakdown
    ax1.set_axisbelow(True)
    l1 = ax1.plot(time_axis, vspd, color=c_purple, linewidth=1.8, label=r"Vertical Speed $v_{spd}$ (ft/s)", zorder=3)
    ax1.set_ylabel(r"$v_{spd}$ (ft/s)", fontweight="bold")
    ax1.grid(True, alpha=0.3)  # ax1 controls the single grid

    ax1_alt = ax1.twinx()
    ax1_alt.grid(False)
    l2 = ax1_alt.plot(time_axis, alt, color=c_green, linestyle="--", linewidth=1.6, label="Baro Corrected Altitude (ft)", zorder=3)
    ax1_alt.set_ylabel("Altitude (ft)", fontweight="bold")

    lines1 = l1 + l2
    leg = ax1.legend(lines1, [l.get_label() for l in lines1], loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax1.set_title(f"Flight Record: {flight_rec} (target_idx: {target_idx}) | Label: {record_label} — Altitude Anomaly Impact on Derived AoA", 
                  fontweight="bold")
    leg.set_zorder(5)  # Guarantees legend sits on top of everything

    # Panel 2: Derived vs Raw AoA Impact
    ax2.plot(time_axis, raw_aoa, color=c_crimson, linestyle="--", linewidth=1.6, label="Raw Angle of Attack")
    ax2.plot(time_axis, alpha_est, color=c_blue, linewidth=1.8, label=r"Derived $\alpha_{est}$ ($\theta - \gamma$)")
    ax2.axhline(-58, color=c_crimson, linestyle=":", linewidth=1.5, label=r"Mathematical Saturation Floor ($-58^\circ$)")
    ax2.set_ylabel("AoA (deg)", fontweight="bold")
    ax2.set_xlabel("Time Step in Approach Window (seconds)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)

    # Highlight saturation window
    sat_indices = np.where(np.abs(gamma_deg) >= 80.0)[0]
    if len(sat_indices) > 0:
        start_f, end_f = max(0, sat_indices[0] - 2), min(159, sat_indices[-1] + 2)
        ax1.axvspan(start_f, end_f, color=c_yellow, alpha=0.35)
        ax2.axvspan(start_f, end_f, color=c_yellow, alpha=0.35)

    plt.tight_layout()
    
    summary_fig_path = OUTPUT_FIG_PATH.parent / "alpha_est_go_around_summary_2panel.svg"
    plt.savefig(summary_fig_path, format="svg", bbox_inches="tight")
    print(f"🎉 Compact 2-panel summary saved to: {summary_fig_path.resolve()}")

if __name__ == "__main__":
    generate_aoa_saturation_8panel_plot()
    generate_aoa_saturation_2panel_summary()