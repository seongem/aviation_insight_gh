# tools/flight_markers.py
import sys
import numpy as np
from scipy.signal import medfilt
from scipy.stats import norm
import logging

# Inherits configuration from the root or parent logger set in the runner script
logger = logging.getLogger("AviationInsight.FlightMarkers")

class FlightMarkerEngine:
    def __init__(self, analyzed_phases: np.ndarray, time_axis: np.ndarray, mission_id: str, file_boundaries: list, master_data: dict, n1_array: np.ndarray = None):
        """
        Flight Marker Engine that identifies distinct operational flight legs
        and dynamically aggregates per-flight FSM metrics using vectorized operations.
        """
        self.phases = np.array(analyzed_phases, dtype=int)
        self.time_axis = np.array(time_axis)
        self.mission_id = mission_id
        self.file_boundaries = file_boundaries
        self.n1 = n1_array
        self.n_samples = len(time_axis)
        self.event_stream = []
        self.flight_metrics = {}  # Stores tabulated per-flight diagnostics
        
        # -----------------------------------------------------------------
        # 🗺️ STEP 1: VECTORIZED SPATIAL DISCONTINUITY DETECTION (UPDATED)
        # -----------------------------------------------------------------
        self.leg_start_frames = []
        
        # 1. Pre-calculate indices for all 676 file boundaries in one shot
        boundary_times = [b[0] for b in self.file_boundaries]
        boundary_names = [b[1] for b in self.file_boundaries]
        
        # Vectorized binary search across the entire global time array at once
        boundary_indices = np.searchsorted(self.time_axis, boundary_times)
        
        # 🎯 FIX: Establish every individual file boundary as a definitive standalone leg
        for b_idx, b_name in zip(boundary_indices, boundary_names):
            self.leg_start_frames.append((int(b_idx), b_name))

        # 2. Detect coordinate jumps (Keep as a secondary backup fallback if needed)
        if len(self.file_boundaries) > 1 and 'lat' in master_data and 'lon' in master_data:
            lat_jumps = np.abs(np.diff(master_data['lat']))
            lon_jumps = np.abs(np.diff(master_data['lon']))
            jump_indices = np.where((lat_jumps > 0.10) | (lon_jumps > 0.10))[0] + 1

            if len(jump_indices) > 0:
                file_assignments = np.digitize(jump_indices, boundary_indices) - 1
                file_assignments = np.clip(file_assignments, 0, len(boundary_names) - 1)
                
                for idx, file_block_idx in zip(jump_indices, file_assignments):
                    self.leg_start_frames.append((int(idx), boundary_names[file_block_idx]))
                    
        # 3. Deduplicate and clean up anchors
        unique_anchors = {}
        for frame, f_name in self.leg_start_frames:
            # Keep the absolute earliest frame index mapping to this file identity
            if f_name not in unique_anchors or frame < unique_anchors[f_name]:
                unique_anchors[f_name] = frame
        
        # Re-build the sorted tracking frame parameters
        self.leg_start_frames = [(frame, f_name) for f_name, frame in unique_anchors.items()]
        # -----------------------------------------------------------------
        # 📊 STEP 2: VECTORIZED EVENT STREAM & METRIC GENERATION
        # -----------------------------------------------------------------
        self.generate_events()
        
    def _get_file_name_from_idx(self, idx: int) -> str:
        """Helper to map a raw array index back to its physical file boundary name."""
        current_file = self.file_boundaries[0][1]
        for b_time, b_name in self.file_boundaries:
            b_idx = int(np.searchsorted(self.time_axis, b_time))
            if idx >= b_idx:
                current_file = b_name
            else:
                break
        return current_file

    def get_context(self, idx: int) -> tuple:
        """On-demand lookup for diagnostic reports mapping index to active flight shard."""
        active_id = "ORPHANED_PREFLIGHT"
        for start_frame, f_name in self.leg_start_frames:
            if idx >= start_frame:
                active_id = f_name
            else:
                break
        return active_id, self.mission_id

    def generate_events(self) -> list:
        events = []
        ordered_legs = sorted(self.leg_start_frames, key=lambda x: x[0])
        
        # Mapping to match your reporting nomenclature
        phase_names = {
            1: "PREFLIGHT",
            2: "TAXI",
            3: "TAKEOFF",
            4: "CLIMB",
            5: "CRUISE",
            6: "APPROACH",
            7: "ROLLOUT"
        }

        for idx, (start_idx, flight_id) in enumerate(ordered_legs):
            end_idx = ordered_legs[idx + 1][0] - 1 if idx + 1 < len(ordered_legs) else self.n_samples - 1
            
            leg_phases = self.phases[start_idx:end_idx + 1]
            leg_times = self.time_axis[start_idx:end_idx + 1]
            flight_start_global_time = leg_times[0]
            
            # Keep original milestone baseline intact
            events.append(self._create_event_dict("M_START_OF_FLIGHT", flight_id, start_idx, flight_start_global_time))
            
            phase_diffs = np.diff(leg_phases)
            transition_relative_indices = np.where(phase_diffs != 0)[0] + 1
            
            inside_flight_leg = False
            last_takeoff_idx = None
            illegal_regressions = 0
            
            # --- PASS 1: FORWARD PASS (Seek Definitive END Markers Sequential) ---
            last_seen_exit_idx = {}
            has_rolled_out_forward = False
            expected_phase_forward = leg_phases[0] # Initialize to starting phase (e.g., PREFLIGHT)
            
            for rel_idx in transition_relative_indices:
                global_idx = start_idx + rel_idx
                prev_p = leg_phases[rel_idx - 1]
                curr_p = leg_phases[rel_idx]
                
                if prev_p == 7:
                    has_rolled_out_forward = True
                
                # Only log an exit milestone if it matches our forward progressing sequence
                if prev_p == expected_phase_forward:
                    p_label = phase_names.get(prev_p, "UNKNOWN")
                    if prev_p == 2 and has_rolled_out_forward:
                        p_label = "TAXI_IN"
                        
                    last_seen_exit_idx[p_label] = global_idx - 1
                    
                    # Keep existing historical milestone events unchanged
                    if prev_p == 1 and curr_p == 2:
                        inside_flight_leg = True
                        events.append(self._create_event_dict("M_TAXI_START", flight_id, global_idx, flight_start_global_time))
                    elif prev_p == 2 and curr_p == 3:
                        events.append(self._create_event_dict("M_TAKEOFF_ROLL", flight_id, global_idx, flight_start_global_time))
                        last_takeoff_idx = global_idx
                    elif prev_p == 4 and curr_p == 5:
                        events.append(self._create_event_dict("M_TOP_OF_CLIMB", flight_id, global_idx, flight_start_global_time))
                    elif prev_p == 5 and curr_p == 6:
                        events.append(self._create_event_dict("M_TOP_OF_DESCENT", flight_id, global_idx, flight_start_global_time))
                    elif prev_p == 6 and curr_p == 7:
                        events.append(self._create_event_dict("M_TOUCHDOWN", flight_id, global_idx, flight_start_global_time))
                    elif prev_p == 7 and curr_p == 2:
                        events.append(self._create_event_dict("M_TAXI_IN_START", flight_id, global_idx, flight_start_global_time))
                    elif prev_p == 2 and curr_p == 1 and inside_flight_leg:
                        events.append(self._create_event_dict("M_BLOCK_IN", flight_id, global_idx, flight_start_global_time))
                        events.append(self._create_event_dict("M_END_OF_FLIGHT", flight_id, global_idx, flight_start_global_time))
                        inside_flight_leg = False

                    # Advance the forward state target
                    expected_phase_forward = curr_p

            # Append the definitive End Markers discovered in Pass 1
            for p_label, exit_idx in last_seen_exit_idx.items():
                events.append(self._create_event_dict(f"M_{p_label}_END", flight_id, exit_idx, flight_start_global_time))
            
            # If a phase is active at the end of the file, its end index is the end of the leg
            final_phase_label = phase_names.get(leg_phases[-1], "UNKNOWN")
            if leg_phases[-1] == 2 and has_rolled_out_forward:
                final_phase_label = "TAXI_IN"
            events.append(self._create_event_dict(f"M_{final_phase_label}_END", flight_id, end_idx, flight_start_global_time))

            # --- PASS 2: BACKWARD PASS (Seek Definitive START Markers Sequential) ---
            first_seen_entry_idx = {}
            has_rolled_out_backward = False
            expected_phase_backward = leg_phases[-1] 
            
            for rel_idx in reversed(transition_relative_indices):
                global_idx = start_idx + rel_idx
                prev_p = leg_phases[rel_idx - 1] 
                curr_p = leg_phases[rel_idx]     
                
                if curr_p == 7:
                    has_rolled_out_backward = True
                
                if curr_p == expected_phase_backward:
                    p_label = phase_names.get(curr_p, "UNKNOWN")
                    if curr_p == 2 and not has_rolled_out_backward:
                        p_label = "TAXI_IN"
                    
                    first_seen_entry_idx[p_label] = global_idx
                    expected_phase_backward = prev_p

            # Write out the locked start markers
            for p_label, entry_idx in first_seen_entry_idx.items():
                events.append(self._create_event_dict(f"M_{p_label}_START", flight_id, entry_idx, flight_start_global_time))

            # --- Vectorized Engine Thrust Verification & Auto-Caps ---
            if last_takeoff_idx is not None and self.n1 is not None:
                n1_window = self.n1[last_takeoff_idx:end_idx + 1]
                power_crossings = np.where(n1_window >= 85.0)[0]
                if len(power_crossings) > 0:
                    power_global_idx = last_takeoff_idx + power_crossings[0]
                    events.append(self._create_event_dict("M_POWER_APP", flight_id, power_global_idx, flight_start_global_time))

            if inside_flight_leg or (idx == len(ordered_legs) - 1 and leg_phases[-1] != 1):
                events.append(self._create_event_dict("M_END_OF_FLIGHT", flight_id, end_idx, flight_start_global_time))

            # Record tabulated metrics
            duration_seconds = float(leg_times[-1] - leg_times[0])
            self.flight_metrics[flight_id] = {
                "duration_min": duration_seconds / 60.0,
                "transitions": len(transition_relative_indices),
                "violations": illegal_regressions,
                "samples": len(leg_phases)
            }

        # Sort and return cumulative stream safely
        events.sort(key=lambda x: x['global_mission_time'])
        self.event_stream = events
        return self.event_stream 

    def _create_event_dict(self, marker_type, flight_id, idx, flight_start_time):
        global_t = self.time_axis[idx]
        elapsed_f_t = global_t - flight_start_time
        return {
            "event_type": marker_type,
            "flight_id": flight_id,
            "mission_id": self.mission_id,
            "global_mission_time": float(global_t),
            "elapsed_flight_time": float(np.clip(elapsed_f_t, 0.0, None)),
            "phase_ttg": 0.0,
            "frame_index": int(idx)
        }
    
 