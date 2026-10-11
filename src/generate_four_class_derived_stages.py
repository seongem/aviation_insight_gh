# scripts/generate_four_class_derived_dataset.py

import sys
import json
from datetime import datetime
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from itertools import product
from scipy.signal import savgol_filter
from tqdm import tqdm

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.load_environment import LoadEnvironment
from tools.diagnostic_logger import DiagnosticFileHandler

logger = logging.getLogger("AviationInsight")
ISSUE_TOKEN = "GENERATE_FOUR_CLASS_DERIVED_STAGES"

# Physics Constants & Fleet-wide Fallback Coefficients
GRAVITY_FPS2 = 32.17405  # ft/s^2
DEFAULT_K1 = 4.768711e-02
DEFAULT_K2 = 2.311221e-07
DEFAULT_K3 = 3.961335e+03

PIML_FLEET_COEFS_FILE = Path("/mnt/fast_storage/projects/aviation_insight/datasets/nasa_tails/metadata/piml_fleet_coefficients.csv")
ETL_STAGE_DIRECTORY = Path("/mnt/fast_storage/projects/aviation_insight/datasets/nasa_tails/etl_staged")
QUARANTINE_CSV_PATH = Path("/mnt/fast_storage/projects/aviation_insight/workflows/EDA/quarantine_audit_outputs/quarantined_sensor_issues.csv")

# Complete ordered list of all 31 feature names (20 Baseline + 13 Derived)
ALL_31_FEATURE_NAMES = [
    "Aileron Position LH",
    "Aileron Position RH",
    "Corrected Angle of Attack",
    "Baro Correct Altitude LSP",
    "Computed Airspeed LSP",
    "Selected Course",
    "Drift Angle",
    "Elevator Position Left",
    "T.E. Flap Position",
    "Glideslope Deviation",
    "Selected Heading",
    "Localizer Deviation",
    "Core Speed Avg",
    "Total Pressure LSP",
    "Pitch Angle LSP",
    "Roll Angle LSP",
    "Rudder Position",
    "True Heading LSP",
    "Vertical Acceleration",
    "Wind Speed",
    "p",
    "q",
    "r",
    "p_dot",
    "q_dot",
    "r_dot",
    "res_energy_rate",
    "res_flight_path",
    "res_side_force",
    "vspd_fps",
    "vt_dot",
    "alpha_est",
    "res_flight_path_ae",
    "res_energy_rate_no_k3",
    "fractional_ttg",
    "absolute_ttg_sec",
    "res_nz"
]

import numpy as np

def compute_time_to_go_channels(n_samples: int, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes normalized fractional progression and countdown time-to-go channels.
    
    Args:
        n_samples (int): Total number of frames in the approach window (e.g., 160).
        dt (float): Sampling step in seconds (default 1.0 Hz).
        
    Returns:
        tuple[np.ndarray, np.ndarray]:
            - fractional_ttg: Normalized float array from 0.0 at window start to 1.0 at window end.
            - absolute_ttg_sec: Countdown float array from (n_samples-1)*dt down to 0.0 seconds.
    """
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=np.float64), np.zeros(n_samples, dtype=np.float64)

    # 1. Fractional Progression [0.0 -> 1.0]
    fractional_ttg = np.linspace(0.0, 1.0, num=n_samples, dtype=np.float64)

    # 2. Absolute Countdown in Seconds [159.0s -> 0.0s]
    # Represented as remaining seconds to window completion
    frame_indices = np.arange(n_samples, dtype=np.float64)
    absolute_ttg_sec = (n_samples - 1.0 - frame_indices) * dt

    return fractional_ttg, absolute_ttg_sec

def _savgol_padded(
    signal: np.ndarray,
    window_length: int,
    polyorder: int = 2,
    deriv: int = 0,
    delta: float = 1.0,
    pad_mode: str = "edge"
) -> np.ndarray:
    """
    Applies Savitzky-Golay filtering with endpoint padding to eliminate boundary 
    artefacts at array extremities.
    """
    pad_width = window_length
    padded = np.pad(signal, pad_width=pad_width, mode=pad_mode)
    filtered = savgol_filter(
        padded, 
        window_length=window_length, 
        polyorder=polyorder, 
        deriv=deriv, 
        delta=delta
    )
    return filtered[pad_width:-pad_width]

# Optional local conversion inside physics residual function if required:
def counts_to_flap_deg(counts):
    # Piecewise interpolation based on BAe 146 / Avro RJ detents
    return np.interp(
        counts, 
        [117.5, 1500.0, 2100.0, 2800.0, 3600.0], 
        [0.0,   18.0,   24.0,   33.0,   35.0]
    )

def build_flight_to_mission_map(etl_staged_dir: Path) -> dict:
    """Scans ETL staged directory to map flight_id -> mission_id."""
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


def load_fleet_coefficients(coefs_path: Path) -> dict:
    """Loads per-mission k1, k2, k3 coefficients into a dictionary."""
    if coefs_path.exists():
        fleet_df = pd.read_csv(coefs_path).set_index("mission_id")
        logger.info(f"📖 Loaded custom PIML coefficients for {len(fleet_df)} missions from {coefs_path}")

        disk_missions = set(d.name for d in ETL_STAGE_DIRECTORY.iterdir() if d.is_dir())
        csv_missions = set(fleet_df.index.astype(str))

        unmatched_missions = disk_missions - csv_missions
        if unmatched_missions:
            logger.warning(
                f"⚠️ {len(unmatched_missions)} mission(s) on disk lack PIML coefficients in CSV: {unmatched_missions}"
            )

        return fleet_df[['k1_thrust', 'k2_drag', 'k3_induced']].to_dict(orient="index")
    else:
        logger.warning(f"⚠️ Fleet coefficients file not found at {coefs_path}. Using default fallbacks.")
        return {}


def compute_derived_channels_1hz(
    instance_matrix: np.ndarray, 
    k1: float,
    k2: float,
    k3: float,
    dt: float = 1.0, 
    epsilon_fps: float = 50.0,
    savgol_state_window: int = 7,
    savgol_body_rate_window: int = 7,
    savgol_polyorder: int = 2,
    pad_mode: str = "interp"
) -> np.ndarray:
    """
    Computes body rates, body accelerations, and PIML residuals 
    returning a 2D float32 array of shape (160, 11) for the derived channels.
    Returns:
        np.ndarray: 2D float32 array of shape (N, 17) containing derived channels:
            0: p                       (roll rate, rad/s)
            1: q                       (pitch rate, rad/s)
            2: r                       (yaw rate, rad/s)
            3: p_dot                   (roll acceleration, rad/s^2)
            4: q_dot                   (pitch acceleration, rad/s^2)
            5: r_dot                   (yaw acceleration, rad/s^2)
            6: res_energy_rate         (full PIML energy rate residual with k3, ft/s)
            7: res_flight_path         (vertical velocity residual using measured AoA, ft/s)
            8: res_side_force          (lateral kinematic load factor residual)
            9: vspd_fps                (derived vertical speed, ft/s)
           10: vt_dot                  (airspeed acceleration, ft/s^2)
           11: alpha_est               (estimated angle of attack, deg)
           12: res_flight_path_ae      (vertical velocity residual using estimated gamma, ft/s)
           13: res_energy_rate_no_k3   (simplified thrust/parasite energy rate residual, ft/s)
           14: fractional_ttg          (normalized progression coordinate, 0.0 to 1.0)
           15: absolute_ttg_sec        (countdown time-to-go, seconds to window end)
           16: res_nz                  (kinematic vs measured normal acceleration residual, g)
    """
    aoa_deg = instance_matrix[:, 2]
    alt_ft = instance_matrix[:, 3]
    cas_knots = instance_matrix[:, 4]
    core_speed_pct = instance_matrix[:, 12] / 100.0  # Convert percentage to fraction
    pitch_deg = instance_matrix[:, 14]
    roll_deg = instance_matrix[:, 15]
    heading_deg = instance_matrix[:, 17]
    vert_accel_g = instance_matrix[:, 18]

    phi = np.deg2rad(roll_deg.astype(np.float64))
    theta = np.deg2rad(pitch_deg.astype(np.float64))
    psi = np.unwrap(np.deg2rad(heading_deg.astype(np.float64)))
    alpha = np.deg2rad(aoa_deg.astype(np.float64))
    vt_fps = cas_knots.astype(np.float64) * 1.68781

    polyorder = savgol_polyorder


    # =========================================================================
    # 1. EULER ANGLES & KINEMATIC EULER RATES (N_state)
        # 1. Kinematic Body Rates [p, q, r]
    # =========================================================================
    phi_smooth = savgol_filter(
        phi, window_length=savgol_state_window, polyorder=polyorder, mode='interp'
    )
    theta_smooth = savgol_filter(
        theta, window_length=savgol_state_window, polyorder=polyorder, mode='interp'
    )
    psi_smooth = savgol_filter(
        psi, window_length=savgol_state_window, polyorder=polyorder, mode='interp'
    )

    phi_dot = savgol_filter(
        phi_smooth, window_length=savgol_state_window, polyorder=polyorder, deriv=1, delta=dt, mode='nearest'
    )
    theta_dot = savgol_filter(
        theta_smooth, window_length=savgol_state_window, polyorder=polyorder, deriv=1, delta=dt, mode='nearest'
    )
    psi_dot = savgol_filter(
        psi_smooth, window_length=savgol_state_window, polyorder=polyorder, deriv=1, delta=dt, mode='nearest'
    )

    p = phi_dot - psi_dot * np.sin(theta)
    q = theta_dot * np.cos(phi) + psi_dot * np.cos(theta) * np.sin(phi)
    r = -theta_dot * np.sin(phi) + psi_dot * np.cos(theta) * np.cos(phi)

    
    # =========================================================================
    # 2. BODY ANGULAR RATES & ACCELERATIONS (N_body)
    # 2. Body Accelerations [p_dot, q_dot, r_dot]
    # =========================================================================
    # Body rates p derived from Euler transformation (p = phi_dot - psi_dot * sin(theta))
    p = phi_dot - psi_dot * np.sin(theta)

    p_dot = savgol_filter(
        p, window_length=savgol_body_rate_window, polyorder=polyorder, deriv=1, delta=dt, mode='interp'
    )
    q_dot = savgol_filter(
        q, window_length=savgol_body_rate_window, polyorder=polyorder, deriv=1, delta=dt, mode='interp'
    )
    r_dot = savgol_filter(
        r, window_length=savgol_body_rate_window, polyorder=polyorder, deriv=1, delta=dt, mode='interp'
    )
def compute_derived_channels_1hz(
    instance_matrix: np.ndarray, 
    k1: float,
    k2: float,
    k3: float,
    dt: float = 1.0, 
    epsilon_fps: float = 50.0,
    savgol_state_window: int = 7,
    savgol_body_rate_window: int = 7,
    savgol_polyorder: int = 2,
    pad_mode: str = "interp"
) -> np.ndarray:
    """
    Computes body rates, body accelerations, and PIML residuals 
    returning a 2D float32 array of shape (160, 17) for the derived channels.
    """
    aoa_deg = instance_matrix[:, 2]
    alt_ft = instance_matrix[:, 3]
    cas_knots = instance_matrix[:, 4]
    core_speed_pct = instance_matrix[:, 12] / 100.0  # Convert percentage to fraction
    pitch_deg = instance_matrix[:, 14]
    roll_deg = instance_matrix[:, 15]
    heading_deg = instance_matrix[:, 17]
    vert_accel_g = instance_matrix[:, 18]

    phi = np.deg2rad(roll_deg.astype(np.float64))
    theta = np.deg2rad(pitch_deg.astype(np.float64))
    psi = np.unwrap(np.deg2rad(heading_deg.astype(np.float64)))
    alpha = np.deg2rad(aoa_deg.astype(np.float64))
    vt_fps = cas_knots.astype(np.float64) * 1.68781

    polyorder = savgol_polyorder


    # =========================================================================
    # 1. EULER ANGLES & KINEMATIC EULER RATES (N_state)
        # 1. Kinematic Body Rates [p, q, r]
    # =========================================================================
    phi_smooth = savgol_filter(
        phi, window_length=savgol_state_window, polyorder=polyorder, mode='interp'
    )
    theta_smooth = savgol_filter(
        theta, window_length=savgol_state_window, polyorder=polyorder, mode='interp'
    )
    psi_smooth = savgol_filter(
        psi, window_length=savgol_state_window, polyorder=polyorder, mode='interp'
    )

    phi_dot = savgol_filter(
        phi_smooth, window_length=savgol_state_window, polyorder=polyorder, deriv=1, delta=dt, mode='nearest'
    )
    theta_dot = savgol_filter(
        theta_smooth, window_length=savgol_state_window, polyorder=polyorder, deriv=1, delta=dt, mode='nearest'
    )
    psi_dot = savgol_filter(
        psi_smooth, window_length=savgol_state_window, polyorder=polyorder, deriv=1, delta=dt, mode='nearest'
    )

    p = phi_dot - psi_dot * np.sin(theta)
    q = theta_dot * np.cos(phi) + psi_dot * np.cos(theta) * np.sin(phi)
    r = -theta_dot * np.sin(phi) + psi_dot * np.cos(theta) * np.cos(phi)

    
    # =========================================================================
    # 2. BODY ANGULAR RATES & ACCELERATIONS (N_body)
    # 2. Body Accelerations [p_dot, q_dot, r_dot]
    # =========================================================================
    # Body rates p derived from Euler transformation (p = phi_dot - psi_dot * sin(theta))
    p = phi_dot - psi_dot * np.sin(theta)

    p_dot = savgol_filter(
        p, window_length=savgol_body_rate_window, polyorder=polyorder, deriv=1, delta=dt, mode='interp'
    )
    q_dot = savgol_filter(
        q, window_length=savgol_body_rate_window, polyorder=polyorder, deriv=1, delta=dt, mode='interp'
    )
    r_dot = savgol_filter(
        r, window_length=savgol_body_rate_window, polyorder=polyorder, deriv=1, delta=dt, mode='interp'
    )



    # =========================================================================
    # 3. PIML KINEMATIC DERIVATIVES & RESIDUALS (N_state)
        # 3. PIML Derivatives & Residuals
    # =========================================================================
    alt_smooth = savgol_filter(
        alt_ft, window_length=savgol_state_window, polyorder=polyorder, mode='interp'
    )
    vspd_fps = savgol_filter(
        alt_smooth, window_length=savgol_state_window, polyorder=polyorder, deriv=1, delta=dt, mode='nearest'
    )

    vt_smooth = savgol_filter(
        vt_fps, window_length=savgol_state_window, polyorder=polyorder, mode='interp'
    )


    # =========================================================================
    # 3. PIML KINEMATIC DERIVATIVES & RESIDUALS (N_state)
        # 3. PIML Derivatives & Residuals
    # =========================================================================
    alt_smooth = savgol_filter(
        alt_ft, window_length=savgol_state_window, polyorder=polyorder, mode='interp'
    )
    vspd_fps = savgol_filter(
        alt_smooth, window_length=savgol_state_window, polyorder=polyorder, deriv=1, delta=dt, mode='nearest'
    )

    vt_smooth = savgol_filter(
        vt_fps, window_length=savgol_state_window, polyorder=polyorder, mode='interp'
    )
    vt_dot = savgol_filter(
        vt_smooth, window_length=savgol_state_window, polyorder=polyorder, deriv=1, delta=dt, mode='nearest'
    )

    inv_vt_reg = vt_smooth / ((vt_smooth ** 2) + (epsilon_fps ** 2))
    rpm_sq_sum = (core_speed_pct ** 2) * 4
    # 1. Individual Power Terms (All defined as POSITIVE magnitudes)
    thrust_proxy      = k1 * rpm_sq_sum * vt_smooth
    parasite_drag     = k2 * (vt_smooth ** 3)
    induced_drag      = k3 * inv_vt_reg

    # 2. Net Thrust-Drag Power Proxies
    net_power_proxy_full  = thrust_proxy - parasite_drag - induced_drag
    net_power_proxy_no_k3 = thrust_proxy - parasite_drag 

    # Estimate angle of attack from gamma
    # Ensure airspeed doesn't cause division spikes during low-speed frames
    vt_clamped = np.maximum(vt_smooth, epsilon_fps)
    sin_gamma = np.clip(vspd_fps / vt_clamped, -1.0, 1.0)
    gamma_rad = np.arcsin(sin_gamma)
    alpha_est = np.rad2deg(theta) - np.rad2deg(gamma_rad)  # Output in degrees

    # 3. Residuals (Kinematic Energy Rate minus Net Power Proxy)
    es_dot_kinematic = vspd_fps + ((vt_smooth * vt_dot) / GRAVITY_FPS2)
    res_energy_rate       = es_dot_kinematic - net_power_proxy_full
    res_energy_rate_no_k3 = es_dot_kinematic - net_power_proxy_no_k3

    res_flight_path = vspd_fps - (vt_fps * np.sin(theta - alpha))
    res_flight_path_ae = vspd_fps - (vt_fps * np.sin(gamma_rad))
    res_side_force = (vert_accel_g - 1.0) - ((vt_fps * r / GRAVITY_FPS2) - (np.sin(phi) * np.cos(theta)))

    # Kinematic Load Factor Residual (Nz)
    gamma_dot_rad = savgol_filter(
        gamma_rad, window_length=savgol_state_window, polyorder=polyorder, deriv=1, delta=dt, mode='nearest'
    )
    nz_kinematic = 1.0 + ((vt_clamped * gamma_dot_rad) / GRAVITY_FPS2)
    res_nz = vert_accel_g.astype(np.float64) - nz_kinematic

    # --- 4. Temporal Progression Channels ---
    n_frames = instance_matrix.shape[0]
    fractional_ttg, absolute_ttg_sec = compute_time_to_go_channels(n_frames, dt=1.0)

    # Return shape (160, 17)
    return np.column_stack([
        p, q, r, p_dot, q_dot, r_dot,
        res_energy_rate, res_flight_path, res_side_force,
        vspd_fps, vt_dot, alpha_est, res_flight_path_ae, res_energy_rate_no_k3,
        fractional_ttg, absolute_ttg_sec, res_nz
    ]).astype(np.float32)

def generate_consolidated_parquet(
    env: LoadEnvironment, 
    savgol_state_window: int = 7, 
    savgol_body_rate_window: int = 9, 
    savgol_polyorder: int = 2
):
    npz_path = env.DATABASE_DIR / "DASHlink_full_fourclass_raw_comp.npz"
    csv_path = env.DATABASE_DIR / "DASHlink_full_fourclass_raw_meta.csv"
    
    # 1. Generate timestamp and version-tagged output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_tag = f"savgol_stateN{savgol_state_window}_bodyN{savgol_body_rate_window}"
    
    output_parquet = env.DATABASE_DIR / f"four_class_derived_dataset_{version_tag}_{timestamp}.parquet"
    latest_parquet = env.DATABASE_DIR / "four_class_derived_dataset.parquet"

    flight_to_mission_map = build_flight_to_mission_map(ETL_STAGE_DIRECTORY)
    fleet_coefs_dict = load_fleet_coefficients(PIML_FLEET_COEFS_FILE)

    logger.info("Loading memory-mapped four_class_data.npz and metadata.csv...")
    data_mmap = np.load(npz_path, mmap_mode="r")["data"]  # Shape: [N, 160, 20]
    meta_df = pd.read_csv(csv_path)

    # --- NEW: Filter out quarantined sensor violation windows ---
    if QUARANTINE_CSV_PATH.exists():
        q_df = pd.read_csv(QUARANTINE_CSV_PATH)
        quarantined_ids = set()
        for col in ["flight_id", "flight_record", "filename"]:
            if col in q_df.columns:
                quarantined_ids.update(q_df[col].astype(str).str.strip().tolist())

        initial_len = len(meta_df)
        valid_indices = []
        for idx, row in meta_df.iterrows():
            f_rec = str(row.get("flight_record", ""))
            f_id = str(row.get("flight_id", "")) if "flight_id" in row else ""
            if f_rec not in quarantined_ids and f_id not in quarantined_ids:
                valid_indices.append(idx)

        # Filter metadata dataframe and slice numpy memory-map accordingly
        meta_df = meta_df.iloc[valid_indices].reset_index(drop=True)
        data_mmap = data_mmap[valid_indices]
        logger.info(f"🛡️ Quarantined Filter: Excluded {initial_len - len(meta_df)} corrupted flight records. Processing {len(meta_df):,} clean instances.")
    else:
        logger.warning(f"⚠️ Quarantine CSV not found at {QUARANTINE_CSV_PATH}. Proceeding without exclusion.")
    # -------------------------------------------------------------

    n_instances, seq_len, num_base_channels = data_mmap.shape
    logger.info(f"Processing {n_instances:,} instances across 31 channels (State N={savgol_state_window}, Body N={savgol_body_rate_window})...")

    # Allocate containers for 31 independent 1D channels
    channel_storage = {name: [] for name in ALL_31_FEATURE_NAMES}
    mission_ids = []

    for idx in tqdm(range(n_instances), desc="Computing 31 Channels"):
        row = meta_df.iloc[idx]
        flight_record = str(row["flight_record"])

        if "mission_id" in row and pd.notna(row["mission_id"]):
            mission_id = str(row["mission_id"])
        elif flight_record in flight_to_mission_map:
            mission_id = flight_to_mission_map[flight_record]
        else:
            mission_id = "unknown_mission"
        
        mission_ids.append(mission_id)

        if mission_id in fleet_coefs_dict:
            coefs = fleet_coefs_dict[mission_id]
            k1, k2, k3 = coefs['k1_thrust'], coefs['k2_drag'], coefs['k3_induced']
        else:
            k1, k2, k3 = DEFAULT_K1, DEFAULT_K2, DEFAULT_K3
            logger.warning(f"⚠️ No PIML coefficients for mission '{mission_id}'. Using default k1={k1}, k2={k2}, k3={k3}.")

        # 1. Pull memory-mapped slice and make a writable copy
        base_matrix = data_mmap[idx].copy()  # Shape: (160, 20)
        
        # Pass window parameters down to the derivation function
        derived_matrix = compute_derived_channels_1hz(
            base_matrix, 
            k1=k1, 
            k2=k2, 
            k3=k3, 
            dt=1.0,
            savgol_state_window=savgol_state_window,
            savgol_body_rate_window=savgol_body_rate_window,
            savgol_polyorder=savgol_polyorder
        )  # Shape: (160, 17)
        
        assert derived_matrix.shape == (160, 17), (
        f"Expected derived array shape (160, 17), got {derived_matrix.shape}"
)

        # Concatenate into full (160, 31) matrix
        full_matrix = np.hstack([base_matrix, derived_matrix]).astype(np.float32)

        # Append each channel as a 1D float32 array (length 160)
        for ch_idx, col_name in enumerate(ALL_31_FEATURE_NAMES):
            channel_storage[col_name].append(full_matrix[:, ch_idx])

    # Construct Master DataFrame
    master_df = meta_df.copy()
    master_df['mission_id'] = mission_ids

    # Add 31 individual 1D array columns
    for col_name in ALL_31_FEATURE_NAMES:
        master_df[col_name] = channel_storage[col_name]

    # 2. Convert to PyArrow Table
    table = pa.Table.from_pandas(master_df)

    # 3. Construct and inject configuration metadata payload
    config_metadata = {
        "generated_timestamp": timestamp,
        "savgol_state_window": savgol_state_window,
        "savgol_body_rate_window": savgol_body_rate_window,
        "savgol_polyorder": savgol_polyorder,
        "dt": 1.0,
        "num_instances": n_instances,
        "num_channels": len(ALL_31_FEATURE_NAMES),
        "kinematic_transformation": "option_a_two_stage_euler"
    }

    existing_metadata = table.schema.metadata or {}
    updated_metadata = {
        **existing_metadata,
        b"pipeline_config": json.dumps(config_metadata).encode("utf-8")
    }
    table = table.replace_schema_metadata(updated_metadata)

    # 4. Write version-tagged Parquet file
    logger.info(f"Writing versioned Parquet table to {output_parquet}...")
    pq.write_table(table, output_parquet, compression="snappy")

    # 5. Maintain 'latest' symlink for default downstream scripts
    if latest_parquet.is_symlink() or latest_parquet.exists():
        latest_parquet.unlink()
    latest_parquet.symlink_to(output_parquet.name)

    logger.info(f"🎉 Written dataset ({output_parquet.stat().st_size / 1e6:.2f} MB)")
    logger.info(f"🔗 Updated active link: {latest_parquet.name} -> {output_parquet.name}")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('▶ %(message)s'))
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(
        description="Generate consolidated Parquet dataset with Savitzky-Golay derived channels."
    )
    parser.add_argument(
        "--sweep-window-sizes",
        nargs="+",
        type=int,
        metavar="N",
        help="List of window sizes to sweep across both state and body rate channels (e.g., --sweep-window-sizes 3 5 7 9 11).",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=2,
        help="Polynomial order for Savitzky-Golay filter (default: 2).",
    )

    args = parser.parse_args()

    env = LoadEnvironment()
    if getattr(env, "ENVIRONMENT", "local_dev") == "local_dev":
        logger.info("Running in Local environment. Logs will be saved to /mnt/fast_storage/projects/aviation_insight/logs/diagnostics")
        output_dir = Path("/mnt/fast_storage/projects/aviation_insight/logs/diagnostics")
    else:
        logger.info(f"Running in {env.ENVIRONMENT} environment. Logs will be saved to {env.OUTPUT_DIR}/diagnostics")
        output_dir = env.OUTPUT_DIR / "diagnostics"

    file_handler = DiagnosticFileHandler(issue_token=ISSUE_TOKEN, output_dir=output_dir if 'output_dir' in locals() else Path("/mnt/fast_storage/projects/aviation_insight/logs/diagnostics"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    if args.sweep_window_sizes:
            # Validate that all window sizes are odd integers >= 3
            invalid_windows = [
                w for w in args.sweep_window_sizes if w % 2 == 0 or w < 3
            ]
            if invalid_windows:
                parser.error(
                    f"Savitzky-Golay window sizes must be odd integers >= 3. Invalid: {invalid_windows}"
                )

            window_sizes = sorted(list(set(args.sweep_window_sizes)))
            total_runs = len(window_sizes) ** 2

            logger.info(
                f"🚀 Starting parameter sweep across {len(window_sizes)} window sizes "
                f"({total_runs} combinations: {window_sizes})..."
            )

            # Run nested sweep over N_state x N_body combinations
            for idx, (n_state, n_body) in enumerate(
                product(window_sizes, window_sizes), 1
            ):
                logger.info(
                    f"\n--- [Run {idx}/{total_runs}] Executing pipeline for N_state={n_state}, N_body={n_body} ---"
                )
                generate_consolidated_parquet(
                    env=env,
                    savgol_state_window=n_state,
                    savgol_body_rate_window=n_body,
                    savgol_polyorder=args.polyorder,
                )

            logger.info(
                f"🎉 Completed sweep of all {total_runs} Parquet dataset variants!"
            )

    else:
        # Default single execution: State N=7, Body N=11
        logger.info(
            "Executing default single dataset generation (N_state=7, N_body=7)..."
        )
        generate_consolidated_parquet(
            env=env,
            savgol_state_window=7,
            savgol_body_rate_window=7,
            savgol_polyorder=args.polyorder,
        )