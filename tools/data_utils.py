import numpy as np
from typing import Dict, Tuple, Any, List

class DiagnosticDataPreparer:
    """Encapsulates telemetry window isolation and coordinate transformations."""
    
    @staticmethod
    def extract_synchronized_window(
        time_axis: np.ndarray,
        master_data: Dict[str, np.ndarray],
        stabilized_phases: List[int],
        dwell_status: List[bool],
        inertial_motion_status: List[bool],
        diagnostics: Dict[str, List[float]],
        start_sec: float,
        end_sec: float
    ) -> Dict[str, Any]:
        """
        Pinpoints exact array index thresholds and isolates an identically sized 
        slice window across all operational tracking matrices to preserve strict sample parity.
        """
        start_idx = np.searchsorted(time_axis, start_sec)
        end_idx = np.searchsorted(time_axis, end_sec)
        
        return {
            "t_win": time_axis[start_idx:end_idx],
            "d_win": {k: np.array(v)[start_idx:end_idx] for k, v in master_data.items()},
            "s_win": np.array(stabilized_phases)[start_idx:end_idx],
            "dwell_win": np.array(dwell_status)[start_idx:end_idx],
            "is_inertially_moving": np.array(inertial_motion_status)[start_idx:end_idx],
            "diag_win": {name: np.array(vals)[start_idx:end_idx] for name, vals in diagnostics.items()}
        }

    @staticmethod
    def project_to_flight_relative(
        window_package: Dict[str, Any],
        file_boundaries: List[Tuple[float, str]],
        t_zero: float,
        start_sec: float,
        end_sec: float
    ) -> Tuple[Dict[str, Any], List[Tuple[float, str]], float, float]:
        """
        Shifts absolute timeline arrays into a flight-deck relative coordinate framework 
        (where t=0.0s represents the true initial file sample start frame).
        """
        # Create deep/independent copies to protect core calculation state integrity
        proj_win = {
            "t_win": window_package["t_win"] - t_zero,
            "d_win": window_package["d_win"],  # Physical sensors are invariant to time offsets
            "s_win": window_package["s_win"],
            "dwell_win": window_package["dwell_win"],
            "is_inertially_moving": window_package["is_inertially_moving"],
            "diag_win": window_package["diag_win"]
        }
        
        # Shift engineering file separation ticks down by the identical frame offset
        proj_boundaries = [(b_time - t_zero, b_name) for b_time, b_name in file_boundaries]
        
        # Shift scalar bounding criteria variables
        proj_start_sec = start_sec - t_zero
        proj_end_sec   = end_sec - t_zero
        
        return proj_win, proj_boundaries, proj_start_sec, proj_end_sec