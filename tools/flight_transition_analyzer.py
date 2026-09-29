from typing import Tuple

import numpy as np

from tools.flight_markers import FlightMarkerEngine

class FlightTransitionAnalyzer:
    """
    Analyzes sequential state changes in stabilized_phases to calculate metrics
    and inject FSM validation anomalies directly into the FlightMarkerEngine.
    """
    VALID_TRANSITION_MAP = {
        0: [0, 1, 2, 3, 4, 5, 6, 7], 
        1: [1, 2],       # Preflight -> Taxi
        2: [2, 3],       # Taxi -> Takeoff
        3: [3, 4],       # Takeoff -> Climb
        4: [4, 5, 6],    # Climb -> Cruise or Aborted Climb to Approach
        5: [5, 6],       # Monolithic Cruise -> Approach ONLY (4 is strictly blocked)
        6: [6, 4, 7],    # Approach -> Go-Around (Climb) or Rollout
        7: [7, 1, 2]     # Rollout -> Next Leg Preflight or Taxi off
    }

    def __init__(self, phase_map: dict):
        self.phase_map = phase_map

    def analyze_flight_profile(self, stabilized_phases: np.ndarray, marker_engine: FlightMarkerEngine) -> Tuple[int, int]:
        """
        Evaluates adjacent steps, registers statistics, and injects 
        INVALID_TRANSITION events directly into the active Marker Engine.
        """
        total_transitions = 0
        invalid_transitions = 0
        n_samples = len(stabilized_phases)

        for i in range(n_samples - 1):
            p_curr = int(stabilized_phases[i])
            p_next = int(stabilized_phases[i + 1])

            # 1. Detect adjacent time-step shift
            if p_curr != p_next:
                total_transitions += 1

                # 2. Check if the transition violates the custom FSM rules
                if p_next not in self.VALID_TRANSITION_MAP.get(p_curr, []):
                    invalid_transitions += 1

                    curr_name = self.phase_map.get(p_curr, "UNKNOWN")
                    next_name = self.phase_map.get(p_next, "UNKNOWN")
                    error_msg = f"INVALID_TRANSITION: {curr_name} -> {next_name}"

                    # 3. Dynamic injection into the existing FlightMarkerEngine architecture
                    # This mimics the internal dictionary structure expected by your log printer
                    marker_engine.event_stream.append({
                        'frame_index': i + 1,
                        'event_type': error_msg,
                        'metadata': {
                            'current_phase': p_curr,
                            'violated_phase': p_next
                        }
                    })

        return total_transitions, invalid_transitions