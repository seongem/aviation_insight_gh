import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import savgol_filter


class SignalInfoExtractor:
    """Extracts signal metadata and per-flight-phase statistics from the master JSON file."""

    DEFAULT_MASTER_PATH = Path(
        "/mnt/fast_storage/projects/aviation_insight/metadata/stats_v2/global_phase_stats_master_v2.json"
    )

    def __init__(self, master_json_path: Optional[Union[str, Path]] = None):
        self.file_path = Path(master_json_path or self.DEFAULT_MASTER_PATH)
        self._data: Dict[str, Any] = {}
        self._load_master_json()

    def _load_master_json(self) -> None:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Master JSON file not found at: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        if "metadata" not in self._data or "stats" not in self._data:
            raise KeyError("Master JSON missing required top-level key: 'metadata' or 'stats'.")

    def _get_single_metadata(self, signal_name: str) -> Dict[str, Any]:
        metadata = self._data.get("metadata", {})
        if signal_name not in metadata:
            raise KeyError(f"Signal '{signal_name}' not found in metadata.")
        return metadata[signal_name]

    def get_metadata(
        self, signal_names: Union[str, List[str]]
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        if isinstance(signal_names, str):
            return self._get_single_metadata(signal_names)
        return {sig: self._get_single_metadata(sig) for sig in signal_names}

    def _get_single_stats(
        self, signal_name: str, flight_phase: Optional[Union[int, str]] = None
    ) -> Dict[str, Any]:
        all_stats = self._data.get("stats", {})

        if flight_phase is not None:
            phase_str = str(flight_phase)
            if phase_str not in all_stats:
                raise KeyError(f"Flight phase '{flight_phase}' not found in stats.")

            phase_stats = all_stats[phase_str]
            if signal_name not in phase_stats:
                raise KeyError(
                    f"Signal '{signal_name}' not found in flight phase {flight_phase} stats."
                )
            return phase_stats[signal_name]

        signal_stats_by_phase = {}
        for phase_id, signals_dict in all_stats.items():
            if signal_name in signals_dict:
                signal_stats_by_phase[phase_id] = signals_dict[signal_name]

        if not signal_stats_by_phase:
            raise KeyError(
                f"Signal '{signal_name}' not found in any flight phase statistics."
            )

        return signal_stats_by_phase

    def get_stats(
        self,
        signal_names: Union[str, List[str]],
        flight_phase: Optional[Union[int, str]] = None,
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        if isinstance(signal_names, str):
            return self._get_single_stats(signal_names, flight_phase=flight_phase)
        return {
            sig: self._get_single_stats(sig, flight_phase=flight_phase)
            for sig in signal_names
        }

    def get_signal_info(
        self, signal_names: Union[str, List[str]]
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        if isinstance(signal_names, str):
            return {
                "signal_name": signal_names,
                "metadata": self._get_single_metadata(signal_names),
                "stats": self._get_single_stats(signal_names),
            }
        return {
            sig: {
                "signal_name": sig,
                "metadata": self._get_single_metadata(sig),
                "stats": self._get_single_stats(sig),
            }
            for sig in signal_names
        }

class SavgolFilterer:
    """Handles Savitzky-Golay window calculations, two-stage filtering, and derivative estimation.
    
    Default delta is set to 1/16.0 (0.0625s) matching the 16 Hz uniform time grid.
    """

    def __init__(self, polyorder: int = 2, delta: float = 1/16.0):
        self.polyorder = polyorder
        self.delta = delta

    @staticmethod
    def calculate_window_size(
        f_native: float,
        f_target: float = 16.0,
        cycles: float = 16.0,
        min_window: int = 7,
        n_samples: Optional[int] = None,
    ) -> int:
        """Calculates an odd window size spanning N native clock cycles to smooth ZOH artifacts.

        For 4 Hz native signal upsampled to 16 Hz grid:
            ratio = 16 / 4 = 4
            window_len = ceil(4 * 16.0) + 1 = 65 frames (~16.25s)
        """
        if f_native <= 0 or f_target <= 0:
            raise ValueError("Sampling frequencies must be positive numbers.")

        if f_native >= f_target:
            ratio = 1.0
        else:
            ratio = f_target / max(f_native, 0.25)

        window_len = math.ceil(ratio * cycles) + 1

        if window_len % 2 == 0:
            window_len += 1

        window_len = max(window_len, min_window)

        if n_samples is not None and n_samples > 0:
            if window_len >= n_samples:
                window_len = n_samples if n_samples % 2 != 0 else n_samples - 1

        return window_len

    def apply_two_stage(
        self,
        data: np.ndarray,
        window_size: int,
        polyorder: Optional[int] = None,
        delta: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies a two-stage Savitzky-Golay filter (smooth position -> derive velocity)."""
        p_order = polyorder if polyorder is not None else self.polyorder
        d_val = delta if delta is not None else self.delta

        n_samples = len(data)
        w_size = min(window_size, n_samples if n_samples % 2 != 0 else n_samples - 1)

        if w_size <= p_order:
            return data, np.gradient(data, d_val)

        # Stage 1: Position/Signal Smoothing
        smooth_signal = savgol_filter(
            data, window_length=w_size, polyorder=p_order, deriv=0
        )

        # Stage 2: Smooth Derivative Calculation
        deriv_signal = savgol_filter(
            smooth_signal,
            window_length=w_size,
            polyorder=p_order,
            deriv=1,
            delta=d_val,
        )

        return smooth_signal, deriv_signal

    def apply_rate_adaptive_smoothing(
        self,
        data: np.ndarray,
        f_native: float,
        f_target: float = 16.0,
        cycles: float = 16.0,
        polyorder: Optional[int] = None,
        delta: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Calculates adaptive window size and applies two-stage filtering in one call."""
        window_size = self.calculate_window_size(
            f_native=f_native,
            f_target=f_target,
            cycles=cycles,
            n_samples=len(data),
        )
        smooth, deriv = self.apply_two_stage(
            data, window_size=window_size, polyorder=polyorder, delta=delta
        )
        return smooth, deriv, window_size