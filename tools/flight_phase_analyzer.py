import sys
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import inspect
from collections import Counter
from tqdm import tqdm
from scipy.stats import norm
import logging
import concurrent.futures

# Centralized imports
from tools.openap_wrapper import OpenAPWrapper, AircraftType
from tools.aircraft_api import AircraftAPI

# Import the core compute function from your existing file
from tools.flight_phase_state import compute_flight_phase_stateful, FlightPhaseState


logger = logging.getLogger("AviationInsight")

def _parallel_viterbi_worker(payload: dict) -> tuple:
    """
    Isolated worker execution context running on a distinct CPU core.
    Invokes the static method directly to avoid object instantiation overhead.
    """
    file_name = payload["file_name"]
    try:
        
        # 🎯 No instantiation! Call the static method directly on the class
        viterbi_results = FlightPhaseAnalyzer.decode_flight_phases_single_file(
            sub_phases=payload["sub_phases"],
            sub_time_axis=payload["sub_time"],
            mission_id=payload["mission_id"],
            flight_id=file_name,
            noise_std=payload["noise_std"]
        )
        return (
            payload["start_idx"], 
            payload["stop_idx"], 
            viterbi_results["inferred_values"].astype(int), 
            None
        )
    except Exception as e:
        return payload["start_idx"], payload["stop_idx"], payload["sub_phases"], e

class FlightPhaseAnalyzer:
    """A class to analyze flight phases using a heuristic state machine approach."""

    # Define as a class variable to avoid repeated signature inspection overhead
    _CACHED_VALID_ARGS = set(inspect.signature(compute_flight_phase_stateful).parameters.keys())

    def __init__(self, config: Optional[dict] = None):
        # 1. Pipeline-level settings (e.g., custom dwell times or margins)
        self.config = config or {}
        
        # 2. Extract the aircraft type from the config dictionary if provided, 
        # defaulting back to the standard fleet standard (e.g., RJ85)
        ac_type = self.config.get("aircraft_type", AircraftType.RJ85)
        
        # 3. Instantiate your brand-new, memory-persistent state tracking architecture
        # It automatically sets up state.payload, state.config, and state.context internally!
        self.state = FlightPhaseState(ac_type=ac_type)
        
        # Read the argument signature cache into the instance namespace cleanly
        self._valid_args = self._CACHED_VALID_ARGS

    def get_stabilized_phase(self, row_data: dict, timestamp: float) -> dict:
        """
        Processes a single window/step and returns the recommended phase.
        Safely handles signature defaults, transient variables, and filters SINDy keys.
        """
        filtered_data = {}
        
        # 1. Build up only the keys that exist in row_data AND are expected by the signature
        for k in self._CACHED_VALID_ARGS:
            if k in row_data:
                filtered_data[k] = row_data[k]

        # 2. Inject explicit transient variables if requested by the signature
        if "now" in self._CACHED_VALID_ARGS:
            filtered_data["now"] = timestamp
        if "state" in self._CACHED_VALID_ARGS:
            filtered_data["state"] = self.state
            
        # 3. Call the function safely. Any configuration parameters not added here 
        # (like 'min_dwell_seconds') will automatically fall back to their function signature defaults!
        return compute_flight_phase_stateful(**filtered_data)

    def clear_historical_confidence(self):
        """Resets the internal state machine memory cleanly between distinct flight legs."""
        ac_type = self.config.get("aircraft_type", AircraftType.RJ85)
        self.state = FlightPhaseState(ac_type=ac_type)

    def detect_flight_boundaries(self, wow_16hz: np.ndarray, alt_16hz: np.ndarray, gs_16hz: np.ndarray, ac_type: AircraftType = AircraftType.RJ85) -> list:
        """
        Pass 1: Vectorized flight boundary extraction using types and settings 
        sourced cleanly via the AircraftAPI. Now an instance method for uniform API design.
        """
        num_samples = len(wow_16hz)
        if num_samples == 0:
            return [(0, 0)]

        # Fetch configurations from our API
        thresholds = AircraftAPI.get_fleet_thresholds(ac_type)
        
        # Evaluate airborne array vectors cleanly via structural configurations
        airborne = AircraftAPI.is_airborne(wow_16hz, ac_type).astype(int)
        diff = np.diff(airborne)

        takeoff_indices = np.where(diff == 1)[0]
        landing_indices = np.where(diff == -1)[0]

        if airborne[0] == 1:
            takeoff_indices = np.insert(takeoff_indices, 0, 0)
        if airborne[-1] == 1:
            landing_indices = np.append(landing_indices, num_samples - 1)

        flight_windows = []
        
        for takeoff in takeoff_indices:
            valid_landings = landing_indices[landing_indices > takeoff]
            if len(valid_landings) == 0:
                continue
            landing = valid_landings[0]

            if (landing - takeoff) > thresholds["min_flight_duration_samples"]:
                start_padded = max(0, takeoff - thresholds["taxi_phase_padding_samples"])
                end_padded = min(num_samples, landing + thresholds["taxi_phase_padding_samples"])
                flight_windows.append((start_padded, end_padded))

        if not flight_windows:
            moving_indices = np.where(gs_16hz > thresholds["taxi_speed_threshold_kts"])[0]
            if len(moving_indices) > 0:
                flight_windows.append((
                    max(0, moving_indices[0] - thresholds["taxi_phase_padding_samples"]), 
                    min(num_samples, moving_indices[-1] + thresholds["taxi_phase_padding_samples"])
                ))
            else:
                flight_windows.append((0, num_samples))

        return flight_windows

    def analyze_isolated_flight_slice(self, flight_slice_dict: dict, start_sec=0, end_sec=None, ac_type: AircraftType = AircraftType.RJ85) -> dict:
        """
        Pass 2: Computes sliding windows across trajectories utilizing the multi-type OpenAP wrapper.
        """
        time_slice = flight_slice_dict['time_axis']
        alt_slice  = flight_slice_dict['alt']
        gs_slice   = flight_slice_dict['gs']
        altr_slice = flight_slice_dict.get('altr', np.zeros(len(time_slice)))
        wow_slice  = flight_slice_dict['wow']
        num_samples = len(time_slice)

        # 1. Downsample slice vector logs to 1Hz trends for wrapper integration
        time_1hz = time_slice[::16]
        alt_1hz  = alt_slice[::16]
        gs_1hz   = gs_slice[::16]
        altr_1hz = altr_slice[::16]

        # Invoke our new multi-type wrapper instead of openap directly
        fp_engine = OpenAPWrapper(ac_type)
        fp_engine.set_trajectory(time_1hz, alt_1hz, gs_1hz, altr_1hz)
        macro_labels_1hz = fp_engine.phaselabel()

        # 2. Stretch elements back to match high-frequency data matrices
        macro_labels_slice = np.repeat(macro_labels_1hz, 16)[:num_samples]
        if len(macro_labels_slice) < num_samples:
            macro_labels_slice = np.concatenate([
                macro_labels_slice, 
                np.full(num_samples - len(macro_labels_slice), 'GND')
            ])

        flight_slice_dict['openap_macro_guard'] = macro_labels_slice

        # 3. Stream through sliding window calculations
        # Clear historical state cleanly resets state, re-allocating a pure empty payload layer
        self.clear_historical_confidence()
        WINDOW_SIZE = 48  
        STEP_SIZE = 16    

        computed_phases = np.zeros(num_samples, dtype=int)
        confidences = np.zeros(num_samples, dtype=float)
        computed_phases[:WINDOW_SIZE] = 1
        confidences[:WINDOW_SIZE] = 1.0

        window_starts = list(range(0, num_samples - WINDOW_SIZE + 1, STEP_SIZE))

        def get_win_median(key, default=0.0):
            arr = flight_slice_dict.get(key, None)
            if arr is None or len(arr) == 0:
                return default
            val = np.median(arr[start:end])
            return default if val is None or np.isnan(val) else float(val)

        for start in tqdm(window_starts, desc="   Streaming Sliding Windows", leave=False):
            end = start + WINDOW_SIZE
            current_timestamp = time_slice[end - 1]

            if end_sec is not None and (current_timestamp < start_sec or current_timestamp > end_sec):
                continue

            win_wow = get_win_median('wow', 0.0)
            win_gs  = float(np.median(gs_slice[start:end]))
            win_alt = float(np.median(alt_slice[start:end]))
            win_altr = float(np.median(altr_slice[start:end]))
            current_macro_guard = macro_labels_slice[end - 1]

            res = compute_flight_phase_stateful(
                state=self.state,
                wow=int(win_wow),  
                gs=win_gs,
                cass=get_win_median('cass', 0.0),
                blac=get_win_median('blac', 0.0),
                blac_smooth=get_win_median('blac', 0.0),
                pla1=get_win_median('pla1', 0.0),
                pla2=get_win_median('pla2', 0.0),
                pla3=get_win_median('pla3', 0.0),
                pla4=get_win_median('pla4', 0.0),
                n1c=get_win_median('n1c', 0.0),       
                n1co=get_win_median('n1co', 0.0),     
                alt=win_alt,
                altr=win_altr,
                flap=get_win_median('flap', 0.0),
                now=current_timestamp,
                acms_phase=int(get_win_median('ph', 0.0)),
                openap_macro=current_macro_guard
            )

            computed_phases[start:end] = res["computed_flight_phase"]
            confidences[start:end] = res["computed_confidence"]

        flight_slice_dict['computed_phases'] = computed_phases
        flight_slice_dict['confidences'] = confidences

        return flight_slice_dict

    def process_mission_folder_hybrid(self, mission_workspace: dict) -> list:
        """
        Orchestration Entry Point. Splits master workspace vectors into true flights,
        processes each slice with internal state variables, and tracks histories cleanly.
        """
        # Call detect_flight_boundaries through self context
        boundaries = self.detect_flight_boundaries(
            wow_16hz=mission_workspace.get('WOW', np.ones(len(mission_workspace[list(mission_workspace.keys())[0]]))),
            alt_16hz=mission_workspace.get('ALT', np.zeros(len(mission_workspace[list(mission_workspace.keys())[0]]))),
            gs_16hz=mission_workspace.get('GS', np.zeros(len(mission_workspace[list(mission_workspace.keys())[0]])))
        )
        
        processed_flights_list = []

        # 2. Run Pass 2 Loop over each isolated flight index slice window
        for fl_idx, (start, end) in enumerate(boundaries):
            logger.info(f" Processing Isolated Trajectory Segment #{fl_idx+1} [Index Bounds: {start} to {end}]...")
            
            # Extract slices out of the master mission channels to create an isolated flight envelope
            flight_slice_dict = {}
            for channel, array in mission_workspace.items():
                if isinstance(array, np.ndarray):
                    flight_slice_dict[channel] = array[start:end]
                    
            # Run OpenAP-guided analysis on this clean flight tracking window
            processed_flight = self.analyze_isolated_flight_slice(flight_slice_dict)
            processed_flights_list.append(processed_flight)
            
        return processed_flights_list
    
    def analyze_flight_data_stream(self, master_data: dict, time_axis: np.ndarray, ac_type: AircraftType = AircraftType.RJ85, log_history: bool = False) -> dict:
        """
        Instance Method. Processes pre-stitched data vectors at 16Hz.
        Pre-vectorizes window medians to completely eliminate Python loop overhead.
        """
        from scipy.ndimage import median_filter

        if ac_type:
            self.config["aircraft_type"] = ac_type
        self.clear_historical_confidence()

        num_samples = len(time_axis)
        if num_samples == 0:
            return {"computed_phases": np.array([]), "confidences": np.array([]), "openap_macro_guard": np.array([])}

        # 1. Downsample to 1Hz for OpenAP
        time_1hz = time_axis[::16]
        alt_1hz  = master_data['alt'][::16]
        gs_1hz   = master_data['gs'][::16]
        altr_1hz = master_data.get('altr', np.zeros(num_samples))[::16]

        # 2. Run OpenAP Wrapper
        fp_engine = OpenAPWrapper(ac_type)
        fp_engine.set_trajectory(time_1hz, alt_1hz, gs_1hz, altr_1hz)
        macro_labels_1hz = fp_engine.phaselabel()

        # 3. Upsample back to 16Hz
        macro_labels_16hz = np.repeat(macro_labels_1hz, 16)[:num_samples]
        if len(macro_labels_16hz) < num_samples:
            padding = num_samples - len(macro_labels_16hz)
            macro_labels_16hz = np.concatenate([macro_labels_16hz, np.full(padding, 'GND')])
        
        WINDOW_SIZE = 48  
        STEP_SIZE = 16    
        HALF_WIN = WINDOW_SIZE // 2  # Used to align centered filter maps to trailing windows

        computed_phases = np.zeros(num_samples, dtype=int)
        confidences = np.zeros(num_samples, dtype=float)
        state_history = [""] * num_samples if log_history else []
        
        computed_phases[:WINDOW_SIZE] = 1
        confidences[:WINDOW_SIZE] = 1.0
        
        if log_history:
            initial_state_json = self.state.to_json()
            for i in range(WINDOW_SIZE):
                state_history[i] = initial_state_json

        window_starts = list(range(0, num_samples - WINDOW_SIZE + 1, STEP_SIZE))
        VALID_KEYS = self._valid_args

        # 🚀 4. PRE-COMPUTE ALL MEDIANS VECTORIALLY BEFORE THE LOOP
        logger.info("--> Vectorizing PASS 1 smoothing filters across master timeline...")
        
        target_median_keys = [
            'wow', 'gs', 'cass', 'blac', 'blac_smooth', 'pla1', 'pla2', 'pla3', 'pla4',
            'lgdn', 'lgup', 'ralt', 'alt', 'altr', 'ivv', 'abrk', 'splg', 'sply', 
            'flap', 'n1t', 'n1c', 'ph', 'apfd', 'alts', 'vmode', 'lmode'
        ]
        
        # Dictionary to store pre-smoothed arrays
        smoothed_data = {}
        for key in target_median_keys:
            # Map clean alternative lookup names if needed (e.g. blac_smooth pulls from blac)
            src_key = 'blac' if key == 'blac_smooth' else key
            arr = master_data.get(src_key, None)
            
            if arr is not None and len(arr) > 0:
                # Fast C-level rolling window median calculation
                smoothed_data[key] = median_filter(arr, size=WINDOW_SIZE, mode='nearest')
            else:
                smoothed_data[key] = None

        logger.info("\n--> Running Hybrid Phase Engine over stitched timeline...")
        for start in tqdm(window_starts, mininterval=10.0, desc="Processing Telemetry Windows", leave=True):
            end = start + WINDOW_SIZE
            current_timestamp = time_axis[end - 1]
            
            # The lookup index that lines up with the original trailing window slice
            lookup_idx = start + HALF_WIN

            # Pull modal guard label
            win_macro_slice = macro_labels_16hz[start:end]
            current_macro_guard = Counter(win_macro_slice).most_common(1)[0][0]

            # Helper to fetch precomputed scalar values instantly
            def fetch_smooth_val(key, default=0.0, is_int=False):
                s_arr = smoothed_data.get(key)
                if s_arr is None:
                    return default
                val = s_arr[lookup_idx]
                if np.isnan(val):
                    return default
                return int(np.floor(val)) if is_int else float(val)

            # 🤝 Build keyword mapping matching exact native signature strings
            function_kwargs = {
                "state": self.state,
                "now": current_timestamp,
                "wow": fetch_smooth_val('wow', 0, is_int=True),
                "gs": fetch_smooth_val('gs', 0.0),
                "cass": fetch_smooth_val('cass', 0.0),
                "blac": fetch_smooth_val('blac', 0.0),
                "blac_smooth": fetch_smooth_val('blac_smooth', 0.0),
                "pla1": fetch_smooth_val('pla1', 0.0),
                "pla2": fetch_smooth_val('pla2', 0.0),
                "pla3": fetch_smooth_val('pla3', 0.0),
                "pla4": fetch_smooth_val('pla4', 0.0),
                "lgdn": fetch_smooth_val('lgdn', 0, is_int=True),
                "lgup": fetch_smooth_val('lgup', 0, is_int=True),
                "ralt": fetch_smooth_val('ralt', 0.0),
                "alt": fetch_smooth_val('alt', 0.0),
                "altr": fetch_smooth_val('altr', 0.0),
                "ivv": fetch_smooth_val('ivv', 0.0),
                "abrk": fetch_smooth_val('abrk', 0.0),
                "splg": fetch_smooth_val('splg', 0, is_int=True),
                "sply": fetch_smooth_val('sply', 0, is_int=True),
                "flap": fetch_smooth_val('flap', 0.0),
                "n1t": fetch_smooth_val('n1t', 0.0),
                "n1c": fetch_smooth_val('n1c', 0.0),
                "n1co": fetch_smooth_val('n1co', 0, is_int=True),
                "acms_phase": fetch_smooth_val('ph', 0, is_int=True),
                "openap_macro_guard": str(current_macro_guard),
                "apfd": fetch_smooth_val('apfd', 0, is_int=True),
                "alts": fetch_smooth_val('alts', 0.0),
                "vmode": fetch_smooth_val('vmode', 0, is_int=True),
                "lmode": fetch_smooth_val('lmode', 0, is_int=True),
                "activate_openap_macro_guard": True
            }

            filtered_kwargs = {k: v for k, v in function_kwargs.items() if k in VALID_KEYS}

            # Execute safe stateful step
            res = compute_flight_phase_stateful(**filtered_kwargs)
            computed_phases[start:start + STEP_SIZE] = res["computed_flight_phase"]
            confidences[start:start + STEP_SIZE] = res["computed_confidence"]

            current_state_snapshot = self.state.to_json()
            if log_history:
                for i in range(start, start + STEP_SIZE):
                    state_history[i] = current_state_snapshot

        return {
            "computed_phases": computed_phases,
            "confidences": confidences,
            "openap_macro_guard": macro_labels_16hz,
            "fsm_state_history_timeline": state_history
        }
    
    @classmethod
    def compute_stateful_phases(
        cls, 
        master_data: Dict[str, np.ndarray], 
        time_axis: np.ndarray, 
        activate_openap_macro_guard: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        PURE COMPUTE: Executes the sequential, state-dependent tracking loop across 
        the full flight timeline. High performance calculation with terminal tracking.
        """
        analyzer = cls()  # Instantiate the class to access instance methods and variables
        
        # Determine total length of the telemetry timeline
        total_rows = len(master_data['alt'])
        
        # --- PRE-ALLOCATE CONTIGUOUS NUMPY ARRAYS ---
        stabilized_phases = np.empty(total_rows, dtype=np.int8)
        dwell_status = np.empty(total_rows, dtype=bool)
        inertial_motion_status = np.empty(total_rows, dtype=bool)
        
        diag_names = ["Preflight", "Taxi", "Takeoff", "Climb", "Cruise", "Approach", "Rollout"]
        valid_keys = [k for k in master_data if k not in ['ph', 'computed_phases', 'hybrid_confidences']]

        # pre-allocate float32 arrays for each phase confidence metric
        diagnostics = {name: np.empty(total_rows, dtype=np.float32) for name in diag_names}
        cached_diag_arrays = [(name, diagnostics[name]) for name in diag_names]
        
        
        # 🎯 FIX: Wrapped the loop in tqdm for instant visual tracking during math execution
        for i in tqdm(range(total_rows), mininterval=10.0, desc="Classifying Telemetry Phases", file=sys.__stdout__, leave=True):
            row_data = {k: master_data[k][i] for k in valid_keys}
            row_data['acms_phase'] = int(master_data['computed_phases'][i])
            row_data['activate_openap_macro_guard'] = activate_openap_macro_guard

            res = analyzer.get_stabilized_phase(row_data, time_axis[i])
            
            # --- ASSIGN DIRECTLY TO PRE-ALLOCATED STORAGE CORNERSTONES ---
            stabilized_phases[i] = res['recommended_phase']
            dwell_status[i] = res.get('dwell_time_ok', True)
            inertial_motion_status[i] = res.get('is_inertially_moving', False)
            
            p_scores = res['penalized_phase_confidences']
            # Direct array pointer assignment (zero string-lookup overhead)
            for name, array in cached_diag_arrays:
                array[i] = p_scores[name] if name in p_scores else 0.0

        return stabilized_phases, dwell_status, inertial_motion_status, diagnostics

    @staticmethod
    def decode_flight_phases_single_file(sub_phases: np.ndarray, sub_time_axis: np.ndarray, mission_id: str, flight_id: str, noise_std=0.5):
        """
        Decodes flight phases from a 1D telemetry stream using an 11-state HMM.
        Short-circuits early if the input signal is completely uniform, dynamically
        resolving the correct phase names.
        """
        n_samples = len(sub_phases)
        n_states = 11
        
        # Maps state index to raw floating-point value
        state_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.0, 1.0, 0.0])

        # Maps state index to descriptive reporting name
        phase_names = [
            "Unknown (Start)",   # State 0
            "Preflight (Start)", # State 1[cite: 1]
            "Taxi-Out",          # State 2[cite: 1]
            "Takeoff",           # State 3[cite: 1]
            "Climb",             # State 4[cite: 1]
            "Cruise",            # State 5[cite: 1]
            "Approach",          # State 6[cite: 1]
            "Rollout",           # State 7[cite: 1]
            "Taxi-In",           # State 8[cite: 1]
            "Preflight (End)",   # State 9[cite: 1]
            "Unknown (End)"      # State 10[cite: 1]
        ]

        # 🎯 DYNAMIC OPTIMIZATION SHORT-CIRCUIT: Handle uniform data safely
        raw_boundary_indices = np.where(sub_phases[:-1] != sub_phases[1:])[0]
        if len(raw_boundary_indices) == 0:
            # 1. Identify the uniform phase value in the data
            current_raw_val = float(sub_phases[0])
            
            # 2. Find the matching hidden state index. 
            # If it's 1.0 or 2.0, default to the initial phases (State 1 or 2) to be safe
            matching_states = np.where(state_values == current_raw_val)[0]
            detected_state = matching_states[0] if len(matching_states) > 0 else 0
            
            current_phase_name = phase_names[detected_state]
            
            # 3. Formulate structural edge boundaries for downstream pipeline predictability
            boundary_indices = np.array([0, n_samples - 1], dtype=int)
            boundary_times = np.array([sub_time_axis[0], sub_time_axis[-1]], dtype=float)
            
            decoded_events = [
                (0, current_phase_name, "File Termination", sub_time_axis[-1])
            ]
            
            return {
                "decoded_events": decoded_events,
                "inferred_values": sub_phases.astype(float),
                "boundary_indices": boundary_indices,
                "boundary_times": boundary_times,
                "mission_id": mission_id,
                "flight_id": flight_id,
            }

        # ----------------------------------------------------
        # 1. INITIALIZE THE TRANSITION MATRIX (A) BASED ON EXPECTED DURATIONS
        # ----------------------------------------------------
        # Define the expected number of samples spent in each phase
        # (Adjust these based on your telemetry sample rate and actual flight data)
        fleet_durations = [
            0.25,    # 0: Unknown_Start (highly agile baseline)
            0.5,     # 1: Preflight-1 (Tail_652_1) (0.5 secs)
            308.12,  # 2: Taxi-Out (Tail_652_1) (~5 mins)
            25.44,   # 3: Takeoff (Tail_652_1) (~25 secs)
            937.88,  # 4: Climb (Tail_652_1) (~15 mins)
            1184.63, # 5: Cruise (Tail_652_1) (~2 hours)
            666.19,  # 6: Approach (Tail_652_1) (~20 mins)
            31.69,   # 7: Rollout (Tail_652_1) (~1 min)
            641.19,  # 8: Taxi-In (Tail_652_1) (~10 mins)
            0.75,    # 9: Preflight_End
        ]

        frame_rate_hz = 16.0
        expected_samples = [d * frame_rate_hz for d in fleet_durations]
        
        # Build the transition matrix
        A = np.zeros((n_states, n_states))
        for i in range(n_states - 1):
            p_stay = 1.0 - (1.0 / expected_samples[i])
            p_move = 1.0 - p_stay
            
            A[i, i] = p_stay      # Probability to stay (p_i)
            A[i, i + 1] = p_move  # Probability to move forward (1 - p_i)
            
        A[-1, -1] = 1.0  # Final absorbing state
        
        # 1. Start with a matrix entirely filled with negative infinity (-inf)
        log_A = np.full((n_states, n_states), -np.inf)

        # 2. Only calculate the log where the original transition matrix is greater than 0
        mask = A > 0
        log_A[mask] = np.log(A[mask])

        # ----------------------------------------------------
        # PRECOMPUTE EMISSION LIKELIHOODS (WITH DUAL UNKNOWN VALUE LOGIC)
        # ------------------------------------- @staticmethod---------------
        log_emissions = np.zeros((n_samples, n_states))
        for s in range(n_states):
            if s == 0 or s == 10:
                # DUAL UNKNOWN HANDLING:
                # Calculate likelihood against 0.0 and against 15.0, then take the maximum
                prob_as_0 = norm.pdf(sub_phases, loc=0.0, scale=noise_std)
                prob_as_15 = norm.pdf(sub_phases, loc=15.0, scale=noise_std)
                # Take the max probability at each sample point to accept either value smoothly
                combined_prob = np.maximum(prob_as_0, prob_as_15)
                # Avoid log(0) warnings by clipping the minimum probability floor
                log_emissions[:, s] = np.log(np.clip(combined_prob, 1e-300, None))
            else:
                # Core standard states (1 through 7)
                log_emissions[:, s] = norm.logpdf(sub_phases, loc=state_values[s], scale=noise_std)


        # ----------------------------------------------------
        # 2. RUN VITERBI USING THE TRANSITION MATRIX WEIGHTS
        # ----------------------------------------------------
        # Initialize Viterbi score matrix and backtracking pointers
        V = np.full((n_samples, n_states), -np.inf)
        ptr = np.zeros((n_samples, n_states), dtype=int)
        
        # Plane must initialize its path evaluations in State 0
        V[0, :] = log_emissions[0, :]
        
        # 4. RUN VITERBI FORWARD LOOP
        for t in range(1, n_samples):
            # State 0 can only stay in State 0
            V[t, 0] = V[t-1, 0] + log_A[0, 0] + log_emissions[t, 0]
            ptr[t, 0] = 0
            
            # States 1 to 10 look back at staying vs. transitioning from the previous state
            for s in range(1, n_states):
                stay_score = V[t-1, s] + log_A[s, s]
                move_score = V[t-1, s-1] + log_A[s-1, s]
                
                if stay_score >= move_score:
                    V[t, s] = stay_score + log_emissions[t, s]
                    ptr[t, s] = s
                else:
                    V[t, s] = move_score + log_emissions[t, s]
                    ptr[t, s] = s - 1

        
        # 5. DYNAMIC BACKTRACKING INITIALIZATION
        path = np.zeros(n_samples, dtype=int)
        
        # Check the last 3 potential ending states depending on where telemetry cut out:
        # State 8 (Taxi-In), State 9 (Preflight-End), or State 10 (Unknown-End)
        # Add State 1 (Preflight-Start) and State 2 (Taxi-Out) as potential early cutouts for ground-only logs
        ending_options = [1, 2, 8, 9, 10]
        best_ending_idx = np.argmax([V[-1, 1], V[-1, 2], V[-1, 8], V[-1, 9], V[-1, 10]])
        path[-1] = ending_options[best_ending_idx]
        
        for t in range(n_samples - 2, -1, -1):
            path[t] = ptr[t + 1, path[t + 1]]
                    
        # Map the hidden states back to their raw expected values for clean plotting
        inferred_profile = state_values[path]
            
        # 🎯 STANDARD PATH UNIFICATION: Force absolute start and end anchors[cite: 1]
        interior_changes = np.where(path[:-1] != path[1:])[0]
        boundary_indices = np.concatenate(([0], interior_changes, [n_samples - 1]))
        boundary_times = sub_time_axis[boundary_indices]
        
        # Decoded events track the actual interior transitions
        decoded_events = []
        for idx in interior_changes:
            from_phase = phase_names[path[idx]]
            to_phase = phase_names[path[idx + 1]]
            time_at_boundary = sub_time_axis[idx]
            decoded_events.append((idx, from_phase, to_phase, time_at_boundary))
            
        return {
            "decoded_events": decoded_events,
            "inferred_values": inferred_profile,
            "boundary_indices": boundary_indices,
            "boundary_times": boundary_times,
            "mission_id": mission_id,
            "flight_id": flight_id
        }



    def apply_windowed_viterbi_decoding(
            self,
            stabilized_phases: np.ndarray,
            viterbi_payloads: List[dict]
        ) -> np.ndarray:
            """
            Pass 3: Pure execution entry point. Directly maps pre-constructed
            telemetry chunks into the parallel processing worker pool.
            """
            viterbi_phases = np.zeros_like(stabilized_phases)
            
            # 🚀 Directly hand off the payloads built during stitching to the CPU cores
            logger.info(f"Distributing Viterbi decoding for {len(viterbi_payloads)} pre-staged shards...")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(_parallel_viterbi_worker, viterbi_payloads)
                
            # 💾 Memory blit the results straight back into the main array block
            for start_idx, stop_idx, inferred_chunk, error in results:
                if error is not None:
                    logger.exception("🚨 Parallel Viterbi shard failure. Falling back to FSM baseline.")
                    if "pytest" in sys.modules:
                        raise error
                    viterbi_phases[start_idx:stop_idx] = stabilized_phases[start_idx:stop_idx]
                else:
                    viterbi_phases[start_idx:stop_idx] = inferred_chunk
                    
            return viterbi_phases
    