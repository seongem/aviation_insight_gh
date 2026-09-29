import inspect
import json
from multiprocessing import context
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import time
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging

from tools.aircraft_api import AircraftAPI
from tools.openap_wrapper import AircraftType

from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from tools.openap_wrapper import AircraftType
from tools.aircraft_api import AircraftAPI

# --------------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------------
logger = logging.getLogger("AviationInsight")

PHASE_MAP = {
    0: "Unknown",
    1: "Preflight",
    2: "Taxi",
    3: "Takeoff",
    4: "Climb",
    5: "Cruise",
    6: "Approach",
    7: "Rollout",
}


@dataclass
class FlightPhaseState:

    movement_history: List[float] = field(default_factory=list)
    apfd_history: List[int] = field(default_factory=list)

    def __init__(self, ac_type: AircraftType = AircraftType.RJ85):
        self.ac_type = ac_type
        self.current_phase: int = 0
        self.current_confidence: float = 0.0
        self.last_transition_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.dwell_start_time: Optional[float] = None
        self.phase_entry_count: Dict[int, int] = {k: 0 for k in PHASE_MAP.keys()}
        self.has_taken_off: bool = False
        self.max_altitude_seen: float = 0.0
        self.apfd_history = []
        self.movement_history = []
        self.last_confirmed_cruise_time: Optional[float] = None

        self.payload = TelemetryPayload()  # Initialize with default values
        self.config = PhaseConfig(ac_type=ac_type)
        self.derived_context = DerivedContext(cfg=self.config)  # Initialize with config

    def to_json(self) -> str:
        """Serializes the complete FSM state context to a JSON string for exact production debugging."""
        return json.dumps({
            "ac_type": self.ac_type.name if hasattr(self.ac_type, 'name') else str(self.ac_type),
            "current_phase": int(self.current_phase),
            "current_confidence": float(self.current_confidence),
            "last_transition_time": float(self.last_transition_time) if self.last_transition_time is not None else None,
            "last_update_time": float(self.last_update_time) if self.last_update_time is not None else None,
            "dwell_start_time": float(self.dwell_start_time) if self.dwell_start_time is not None else None,
            "phase_entry_count": {str(k): int(v) for k, v in self.phase_entry_count.items()},
            "has_taken_off": bool(self.has_taken_off),
            "max_altitude_seen": float(self.max_altitude_seen),  # 🎯 Fixes float32 leakage here
            "apfd_history": [int(x) for x in self.apfd_history],
            "movement_history": [float(x) for x in self.movement_history],  # 🎯 Fixes float32 array here
            "last_confirmed_cruise_time": float(self.last_confirmed_cruise_time) if self.last_confirmed_cruise_time is not None else None
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str, aircraft_enum_map=None):
        """Reconstructs a 1:1 identical copy of the FSM state for deterministic local testing."""
        data = json.loads(json_str)
        
        # Handle AircraftType enum resolution if applicable
        ac_type = data.get("ac_type", "RJ85")
        if aircraft_enum_map and ac_type in aircraft_enum_map:
            instance = cls(ac_type=aircraft_enum_map[ac_type])
        else:
            # Fallback/default instantiation
            instance = cls()
            
        instance.current_phase = data["current_phase"]
        instance.current_confidence = data["current_confidence"]
        instance.last_transition_time = data["last_transition_time"]
        instance.last_update_time = data["last_update_time"]
        instance.dwell_start_time = data["dwell_start_time"]
        
        # Restore dictionary structure with integer keys
        instance.phase_entry_count = {int(k): v for k, v in data["phase_entry_count"].items()}
        
        instance.has_taken_off = data["has_taken_off"]
        instance.max_altitude_seen = data["max_altitude_seen"]
        instance.apfd_history = data["apfd_history"]
        instance.movement_history = data["movement_history"]
        instance.last_confirmed_cruise_time = data["last_confirmed_cruise_time"]
        
        return instance

    def append_telemetry_frame(self, payload, current_movement_metric, time_window: float = 5.0):
        """
        Centralized frame ingest. Appends new data and manages the memory 
        rolling window for ALL historical tracking lists simultaneously.
        """
        # 1. Append the new incoming scalars
        self.movement_history.append(current_movement_metric)
        self.apfd_history.append(int(payload.apfd if payload.apfd is not None else 0))
        
        # 2. Enforce the historical window boundary ONCE for the frame
        window_size = int(time_window * 16)
        self.movement_history = self.movement_history[-window_size:]
        self.apfd_history = self.apfd_history[-window_size:] # Sliced together safely

    def check_recent_movement(self) -> bool:
        """Evaluates movement stability on the pre-rolled history block."""
        if len(self.movement_history) == 0:
            return False
        # 🎯 STABILITY FIX: Use a rolling average to ignore signal noise
        # 🎯 STABILITY FIX: Use native Python math to eliminate NumPy overhead
        return (sum(self.movement_history) / len(self.movement_history)) > 0.009
    
    def is_automation_actively_engaged(self, vmode_signal: int) -> bool:
        """Evaluates automation state intent on the pre-rolled history block."""
        if len(self.apfd_history) == 0:
            return False
        # Debounce: True if APFD was coupled/manual tracking and vertical intent is active
        return any(status in [1, 2] for status in self.apfd_history) and (vmode_signal != 0)
    
    def was_recently_cruising(self, current_time: float, window_seconds: float = 120.0) -> bool:
        """Evaluates if the state engine was established in Cruise within the temporal window."""
        # if current phase has changed to cruise, update timer
        if self.current_phase == 5:
            self.last_confirmed_cruise_time = current_time

        if self.last_confirmed_cruise_time is None:
            return False
        return (current_time - self.last_confirmed_cruise_time) <= window_seconds

@dataclass
class PhaseConfig:
    """
    Static operational thresholds customized per airframe class.
    Hydrated automatically via the centralized AircraftAPI gateway registry.
    """
    # Speed & Kinematic Thresholds
    taxi_threshold: float = 30.0          # Max speed allowed during taxi sequences
    rollout_threshold: float = 40.0         # Speed barrier distinguishing taxi from rollout speeds
    takeoff_threshold: float = 115.0       # Characteristic V1/VR speed barrier
    cruise_gs_min: float = 300.0          # Groundspeed floor required to sustain cruise scoring

    # Vertical Speed Thresholds (ft/min)
    roc_threshold: float = 500.0      # Rate of climb threshold indicating active climb
    rod_threshold: float = -500.0     # Rate of descent threshold indicating active descent
    level_flight_window: float = 150.0   # Strict kinetic window for level cruise

    # Engine Power Envelopes
    high_power_threshold: float = 70.0    # N1 percentage indicating takeoff/climb intent
    low_power_threshold: float = 35.0     # N1 percentage indicating idle/descent flight states

    # Barometric & Radar Altitude Caps
    cruise_alt_floor: float = 18000.0     # Barometric altitude floor separating cruise from climbs
    approach_ralt_max: float = 5000.0     # Radar altitude ceiling beneath which approach rules active
    landing_ralt_max: float = 50.0        # Radar altitude ceiling indicating imminent touchdown roll
    approach_sink_rate: float = -500.0    # Vertical velocity threshold (fpm) indicating active approach descent

    # ✈️ Operational Airspace Boundaries
    terminal_airspace_alt_ceiling: float = 10000.0  # Speed restriction lifting point FAR 91.117
    clean_climb_min_gs: float = 200.0               # Minimum speed proxy for clean configuration

    def __init__(self, ac_type: AircraftType = AircraftType.RJ85):
        """Looks up airframe boundaries dynamically from the master fleet database."""
        limits = AircraftAPI.get_fleet_thresholds(ac_type)
        
        self.taxi_threshold = float(limits.get("taxi_speed_threshold_kts", 30.0))
        self.rollout_threshold = float(limits.get("rollout_speed_threshold_kts", 40.0))
        self.takeoff_threshold = float(limits.get("takeoff_threshold_kts", 115.0))
        self.cruise_gs_min = float(limits.get("cruise_gs_min_kts", 300.0))
        self.high_power_threshold = float(limits.get("high_power_threshold_n1", 70.0))
        self.low_power_threshold = float(limits.get("low_power_threshold_n1", 35.0))
        self.cruise_alt_floor = float(limits.get("cruise_alt_floor_ft", 18000.0))
        self.approach_ralt_max = float(limits.get("approach_ralt_max_ft", 5000.0))
        self.landing_ralt_max = float(limits.get("landing_ralt_max_ft", 50.0))
        self.approach_sink_rate = float(limits.get("approach_sink_rate_fpm", -500.0))
        self.roc_threshold = float(limits.get("roc_threshold_fpm", 500.0))
        self.rod_threshold = float(limits.get("rod_threshold_fpm", -500.0))
        self.level_flight_window = float(limits.get("level_flight_window_fpm", 150.0)) 
        self.terminal_airspace_alt_ceiling = float(limits.get("terminal_airspace_alt_ceiling_ft", 10000.0))
        self.clean_climb_min_gs = float(limits.get("clean_climb_min_gs_kts", 200.0))


@dataclass
class TelemetryPayload:
    """
    Consolidates raw scalar inputs passed to the function boundary.
    All non-optional values are fully initialized with safe structural fallbacks.
    """
    # Primary Flight Dynamics Sensors
    wow: int = 1                          # Weight-on-Wheels Discrete (0=Ground, 1=Airborne)
    gs: float = 0.0                       # Ground Speed (knots)
    cass: float = 0.0                     # Selected Airspeed (knots) for pilot intent tracking
    blac: float = 0.0                     # Longitudinal Accelerometer Vector (raw)
    blac_smooth: float = 0.0              # Smoothed/Filtered Longitudinal Accelerometer
    
    # Throttle Quadrant Metrics
    pla1: float = 0.0; pla2: float = 0.0  # Power Lever Angles for Engines 1 through 4
    pla3: float = 0.0; pla4: float = 0.0
    
    # Airframe Configuration Mechanics
    lgdn: Optional[int] = None            # Landing Gear Down Discrete
    lgup: Optional[int] = None            # Landing Gear Up Discrete
    splg: int = 0                         # Speed Brake Ground Discrete
    sply: int = 0                         # Speed Brake Flight Discrete
    flap: Optional[float] = None          # Flap Handle Trajectory Angle (degrees)
    abrk: Optional[float] = None          # Auto-Brake Pressure Target Values
    
    # Altitude and Flight Path Vectors
    ralt: Optional[float] = None          # Radar Altimeter Reading (feet AGL)
    alt: Optional[float] = None           # Barometric Altitude Reading (feet MSL)
    altr: Optional[float] = None          # Altitude Rate of Change / Vertical Velocity Trend
    ivv: Optional[float] = None           # Instantaneous Vertical Velocity Vector (fpm)
    
    # Engine Core Performance Indicators
    n1t: Optional[float] = None           # N1 Target/Commanded Thrust Percentage
    n1c: Optional[float] = None           # N1 Current Actual Fan Rotor Speed Percentage
    n1co: Optional[int] = None            # N1 Overspeed Discrete Warning Check
    
    # Tracking Orchestration Contexts
    acms_phase: int = 0                   # Recorded phase code from flight data recorder
    now: Optional[float] = None           # Continuous relative elapsed timeline clock (seconds)
    openap_macro_guard: str = "UNKNOWN"         # Downsampled OpenAP macro tracking string label
    fadf: Optional[int] = None            # Flight Phase Change Flags
    fads: Optional[int] = None            # Flight Phase Stability Markers

    # Tracking Flight Director
    apfd: Optional[int] = None            # Auto Pilot Flight Director Status
    alts: Optional[int] = None            # Flight Director Seleced Altitude
    vmode: Optional[int] = None            # Vertical Mode Selection
    lmode: Optional[int] = None            # Lateral Mode Selection

    @property
    def ph(self) -> int:
        """Exposes a clean local alias matching the lower-cased MATLAB array dictionary key."""
        return self.acms_phase


@dataclass
class DerivedContext:
    """
    The Unified Semantic Layer for State Machine Rules.
    Encapsulates static threshold profiles alongside dynamic flying conditions.
    """
    def __init__(self, cfg: PhaseConfig):
        # ⚙️ Internalized Configuration Envelope
        self.config = cfg

    # 🏃 Dynamic Flying & Rolling Status Contexts
        self.on_ground: bool = True
        self.is_inertially_moving: bool = False
        self.is_at_altitude: bool = False
        self.is_within_approach_window: bool = False
        self.is_within_landing_window: bool = False
        self.low_power: bool = True
        self.high_power: bool = False
        self.is_taxi_moving: bool = False
        self.is_takeoff_rolling: bool = False
        self.is_landing_rolling: bool = False
        self._n1_effective: float = 0.0
        self.is_level_flight: bool = False
        self.is_structurally_climbing: bool = False
        self.is_structurally_descending: bool = False

    # 🚀 Automation Intent Metrics
        self.has_flight_director_telemetry: bool = False
        self.automation_engaged: bool = False
        self.altitude_error: float = 0.0
        self.automation_intent: int = 0  # 1 = Climb, -1 = Descend, 0 = Hold/Capture
    # =========================================================================
    # 🎯 INTERFACE SHORTHAND PROPERTIES
    # =========================================================================
    @property
    def n1_effective(self) -> float:
        """Exposes the API-computed effective engine core thrust performance."""
        return self._n1_effective   
    
    @property
    def taxi_threshold(self) -> float:
        return self.config.taxi_threshold

    @property
    def rollout_threshold(self) -> float:
        return self.config.rollout_threshold

    @property
    def takeoff_threshold(self) -> float:
        return self.config.takeoff_threshold

    @property
    def cruise_gs_min(self) -> float:
        return self.config.cruise_gs_min

    @property
    def high_power_threshold(self) -> float:
        return self.config.high_power_threshold

    @property
    def low_power_threshold(self) -> float:
        return self.config.low_power_threshold

    @property
    def cruise_alt_floor(self) -> float:
        return self.config.cruise_alt_floor

    @property
    def approach_ralt_max(self) -> float:
        return self.config.approach_ralt_max

    @property
    def landing_ralt_max(self) -> float:
        return self.config.landing_ralt_max

    @property
    def approach_sink_rate(self) -> float:
        """Shorthand shortcut resolving down to the internal config limit."""
        return self.config.approach_sink_rate

    def update_context(self, payload: TelemetryPayload, state: Any) -> None:
        # A. Hydrate boundaries tailored exactly to this target airframe profile
        cfg = self.config


        # 🎯 CALCULATE CORE THRUST VIA THE UPDATED MASTER API
        # This will now safely return just the base_n1 as requested
        self._n1_effective = AircraftAPI.calculate_effective_thrust(payload.n1c, payload.n1co, state.ac_type)

        # B. Map Weight-On-Wheels via AircraftAPI to capture custom polarities
        self.on_ground = not AircraftAPI.is_airborne(payload.wow, state.ac_type)
        
        # C. Verify altitude groupings safely (defending against missing/None telemetry values)
        current_alt = payload.alt if payload.alt is not None else 0.0
        current_ralt = payload.ralt if payload.ralt is not None else 99999.0  # Assumes high altitude if missing
        
        self.is_at_altitude = current_alt > (cfg.cruise_alt_floor - 2000.0)
        self.is_within_approach_window = current_ralt <= cfg.approach_ralt_max
        self.is_within_landing_window = current_ralt <= cfg.landing_ralt_max

        # D. Evaluate rate of climb and descent
        self.is_structurally_climbing = payload.altr > cfg.roc_threshold
        self.is_structurally_descending = payload.altr < cfg.rod_threshold
        self.is_level_flight = abs(payload.altr) <= cfg.level_flight_window
        
        # D. Track engine power gates cleanly
        current_n1c = payload.n1c if payload.n1c is not None else 0.0
        self.low_power = current_n1c < cfg.low_power_threshold
        self.high_power = current_n1c > cfg.high_power_threshold

        # E. Process inertial movement with parameter override safety protection
        current_blac = payload.blac_smooth if payload.blac_smooth != 0.0 else payload.blac
        self.is_inertially_moving = abs(current_blac) > 0.009

        # F. Synthesize semantic operational flight states
        self.is_taxi_moving = (self.on_ground and (payload.gs > 2.0 or self.is_inertially_moving) and payload.gs <= cfg.taxi_threshold)
        self.is_takeoff_rolling = (self.on_ground and payload.gs > cfg.taxi_threshold and self.high_power)
        self.is_landing_rolling = (self.on_ground and payload.gs > cfg.rollout_threshold and state.has_taken_off and state.current_phase == 7)

        # 🦾 AUTOMATION INTEGRATION ENGINE
        self.has_flight_director_telemetry = all(x is not None for x in [payload.apfd, payload.vmode, payload.alts])
        
        if self.has_flight_director_telemetry:
            # Check the state container's debounced history flag
            self.automation_engaged = state.is_automation_actively_engaged(payload.vmode)
            self.altitude_error = payload.alts - current_alt
            
            # Categorize Pilot Operational Intent with a 150ft deadband
            if self.altitude_error > 150.0:
                self.automation_intent = 1   # Intent to climb
            elif self.altitude_error < -150.0:
                self.automation_intent = -1  # Intent to descend
            else:
                self.automation_intent = 0   # Intent to hold/capture target altitude
        else:
            self.automation_engaged = False
            self.altitude_error = 0.0
            self.automation_intent = 0

def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute_flight_phase_stateful(
# =========================================================================
    # GROUP 1: THE CORE TRACKER (Required, No Defaults)
    # =========================================================================
    state: FlightPhaseState,

    # =========================================================================
    # GROUP 2: REQUIRED TELEMETRY METRICS (No Defaults)
    # Move and sort these however you like!
    # =========================================================================
    wow: int, 
    gs: float, 
    cass: float, 
    blac: float, 
    blac_smooth: float,
    pla1: float, 
    pla2: float, 
    pla3: float, 
    pla4: float,

    # =========================================================================
    # GROUP 3: OPTIONAL TELEMETRY METRICS (With Defaults)
    # Move and sort these however you like!
    # =========================================================================
    lgdn: Optional[int] = None, 
    lgup: Optional[int] = None,
    ralt: Optional[float] = None, 
    alt: Optional[float] = None, 
    altr: Optional[float] = None,
    ivv: Optional[float] = None, 
    abrk: Optional[float] = None, 
    flap: Optional[float] = None, 
    n1t: Optional[float] = None, 
    n1c: Optional[float] = None,
    n1co: Optional[int] = None, 
    fadf: Optional[int] = None, 
    fads: Optional[int] = None,
    apfd: Optional[int] = None,            # Auto Pilot Flight Director Status
    alts: Optional[int] = None,            # Flight Director Seleced Altitude
    vmode: Optional[int] = None,            # Vertical Mode Selection
    lmode: Optional[int] = None,            # Lateral Mode Selection
    
    # =========================================================================
    # GROUP 4: ENGINE CONFIGS & OVERRIDES (With Defaults)
    # Move and sort these however you like!
    # =========================================================================
    splg: int = 0, 
    sply: int = 0,
    acms_phase: int = 0,
    min_dwell_seconds: float = 10.0, 
    hysteresis_margin: float = 0.02,
    transition_penalty: float = 0.15, 
    switch_margin: float = 0.15,
    now: Optional[float] = None, 
    openap_macro_guard: str = "UNKNOWN",
    activate_openap_macro_guard: bool = False,
    
    # =========================================================================
    # THE CATCH-ALL WALL
    # =========================================================================
    **kwargs
) -> Dict[str, Any]:
    """
    Stateful flight phase classifier with:
      - persistence
      - hysteresis
      - minimum dwell time
      - transition penalties

    state: FlightPhaseState, maintained across sliding windows
    now: timestamp (seconds). If None, uses time.time()
    """
    # =========================================================================
    # 🏗️ INTEGRATED DATA LAYER ENTRY POINT
    # =========================================================================
    # 1. Extract shared payload object and update it with the latest telemetry values for this execution frame
    p = state.payload  # Reuse the payload object from state to minimize overhead
    context = state.derived_context  # Access the shared context object for this state instance

    # 2. Map continuous/discrete values straight from the explicit local variables
    p.wow = wow
    p.gs = gs
    p.cass = cass
    p.blac = blac
    p.blac_smooth = blac_smooth
    p.pla1 = pla1
    p.pla2 = pla2
    p.pla3 = pla3
    p.pla4 = pla4

# 3. Handle Optionals with type safety checks straight from local variables
    # (If a window filter returned None, preserve None; if it returned a number, cast it cleanly)
    # Guard discrete optionals against type leaks
    p.lgdn = int(lgdn) if lgdn is not None else None
    p.lgup = int(lgup) if lgup is not None else None
    p.ralt = float(ralt) if ralt is not None else None
    p.alt = float(alt) if alt is not None else None
    p.altr = float(altr) if altr is not None else None
    p.ivv = float(ivv) if ivv is not None else None
    p.abrk = float(abrk) if abrk is not None else None
    p.flap = float(flap) if flap is not None else None
    p.n1t = float(n1t) if n1t is not None else None
    p.n1c = float(n1c) if n1c is not None else None
    p.n1co = int(n1co) if n1co is not None else None
    p.fadf = int(fadf) if fadf is not None else None
    p.fads = int(fads) if fads is not None else None
    p.apfd = int(apfd) if apfd is not None else None
    p.alts = int(alts) if alts is not None else None
    p.vmode = int(vmode) if vmode is not None else None
    p.lmode = int(lmode) if lmode is not None else None

    p.splg = splg
    p.sply = sply
    p.acms_phase = acms_phase
    p.now = float(now) if now is not None else None
    p.openap_macro_guard = openap_macro_guard

    # 4. Update the shared context with the latest configuration and dynamic state
    context.update_context(payload=p, state=state)



    if now is None:
        now = time.time()

    def spoilers_deployed() -> bool:
        return (splg == 1) or (sply == 1)

    def abrk_deployed() -> bool:
        return abrk is not None and abrk > 0
    
    # --- Dynamic Apex Tracker ---
    current_alt = alt if alt is not None else 0.0
    if not context.on_ground and current_alt is not None:
        if current_alt > state.max_altitude_seen:
            state.max_altitude_seen = current_alt

    # --- Lifecycle Tracker ---
    # =========================================================================
    # 1. FLIGHT SEQUENCE LIFECYCLE (Supports Touch-and-Go & Multi-Hops)
    # =========================================================================
    
    # Condition A: We enter an airborne active flight regime -> Lock the flag True
    if not context.on_ground or state.current_phase in [3, 4, 5, 6]:
        state.has_taken_off = True

    # Condition B: TOUCH-AND-GO RESET
    # If we are on the ground but power spikes up significantly (pilot pushing throttles for another loop),
    # or our state machine climbs back out, reset 'has_taken_off' to False.
    # This prevents the breakout rule from forcing a "Taxi/Preflight" drop during the subsequent rollout.
    if context.on_ground and (context.high_power or state.current_phase in [3, 4]):
        state.has_taken_off = False

    # =========================================================================
    # 2. THE STATIONARY BREAKOUT GATING RULE
    # =========================================================================
    # Only allow the breakout rule to drop to ground tracking (Taxi/Preflight)
    # if the aircraft has flown AND has decelerated to a complete stop at the terminal gate.
    if state.has_taken_off and context.on_ground and gs < 2.0:
        if state.current_phase not in [1, 2]:
            state.current_phase = 2  # Drop back to Taxi-In safely
            state.last_transition_time = now
            state.dwell_start_time = now

    # Use CASS 'Intent' + Inertial 'Fact'
    pilot_intends_taxi = (cass is not None and 90 <= cass <= 110)

    phase_scores = {k: 0.0 for k in PHASE_MAP.keys()}

    # =========================================================================
    # INTERNAL TIMING HISTORIAN & HYSTERESIS MANAGEMENT
    # =========================================================================
    # Default to False to lock transitions until an active baseline is checked
    dwell_time_ok = False
    time_since_dwell = 0.0

    if state.dwell_start_time is not None:
        # Standard tracking: Measure elapsed time since the current phase began
        time_since_dwell = now - state.dwell_start_time
        dwell_time_ok = time_since_dwell >= min_dwell_seconds
    else:
        # Fallback tracking: If a breakout rule dropped dwell_start_time to None,
        # fallback to checking elapsed time since the last hard state transition.
        if state.last_transition_time is not None:
            time_since_dwell = now - state.last_transition_time
            dwell_time_ok = time_since_dwell >= min_dwell_seconds
        else:
            # Cold-start path: First frame of the entire flight log execution
            dwell_time_ok = True
            time_since_dwell = 0.0

    # --- Movement Detection (Using Smoothed BLAC) ---
    current_blac = blac_smooth if blac_smooth is not None else blac
    current_gs = gs if gs is not None else 0.0
    current_altr = altr if altr is not None else 0.0
  
      # 5. Append the current telemetry frame to the rolling window
    state.append_telemetry_frame(payload=p, current_movement_metric=current_blac, time_window=5.0)

    # 6. Evaluate historical presence dynamically from the state object
    was_recently_cruising = state.was_recently_cruising(now, window_seconds=120.0)

    # --- Phase 1: Pre-flight ---
    if context.on_ground:
        # If we have safely entered a later phase, do not allow Preflight to score
        if state.current_phase > 1:
            phase_scores[1] = 0.0
        elif current_gs <= 2.0 and not context.is_inertially_moving:
            phase_scores[1] = 1.0 
        else:
            # We are experiencing inertial or ground speed movement
            phase_scores[1] = 0.0
    else:
        phase_scores[1] = 0.0

    # --- Phase 2: Taxi (Combined Ground-Movement Envelope) ---
    is_moving_recently = state.check_recent_movement()

    if context.on_ground:
        if context.is_taxi_moving or is_moving_recently:
            if current_gs > 2.0:
                phase_scores[2] = 0.80
            else:
                # Aircraft is moving inertially but GS is 0.0 (Pushback/Early Tug Movement)
                # 🎯 FIX: Raised to 0.75 so that (0.75 - transition_penalty) firmly beats Preflight
                phase_scores[2] = 0.50
            
            is_takeoff_rolling = (current_gs > context.taxi_threshold and context.n1_effective > 50.0)
            
            # 🎯 FIX: Shield the landing logic so it cannot depress Taxi scores before departure
            is_landing_rolling = state.has_taken_off and (state.current_phase == 6 or (spoilers_deployed() or abrk_deployed()))
            
            if is_takeoff_rolling or is_landing_rolling:
                phase_scores[2] = 0.10  # Step aside for Phase 3 or Phase 7
        else:
            if state.current_phase == 1:
                # Firmly lock out taxi scoring if the aircraft is strictly static at the gate
                phase_scores[2] = 0.0
            else:
                # Maintain taxi profile if aircraft stops momentarily on a taxiway
                phase_scores[2] = 0.50
    else:
        phase_scores[2] = 0.0

    phase_scores[2] = _clip(phase_scores[2])

    # --- Phase 3: Takeoff ---
    if context.on_ground and gs > context.taxi_threshold:
        if context.n1_effective > 55.0:
            phase_scores[3] = 0.95
            
            # Reintroduced Flap Check for localized confidence boost
            if flap is not None and flap > 0:
                phase_scores[3] += 0.03
                
            if n1t is not None and context.n1_effective >= (0.9 * n1t):
                phase_scores[3] += 0.02
        else:
            phase_scores[3] = 0.0
        phase_scores[3] = _clip(phase_scores[3])

    # --- Phase 4: Climb (Inertial Decay Protection) ---
    if not context.on_ground:
        if context.is_structurally_climbing:
            phase_scores[4] = 0.90
        elif state.current_phase == 4 and current_alt > 10000.0 and current_gs > 200.0:
            # 🎯 FIX: Inertial Hysteresis Protection. If we are currently established in Climb,
            # flying at high altitudes and clean speeds, do not let a single noisy frame 
            # obliterate our score to 0.0. Provide a stable retention decay.
            phase_scores[4] = 0.55 if not context.is_structurally_descending else 0.0
        else:
            phase_scores[4] = 0.0
    else:
        phase_scores[4] = 0.0
        
    phase_scores[4] = _clip(phase_scores[4])

    # --- Phase 5: Cruise (The Top-of-Descent / Step-Climb Fix) ---
    if not context.on_ground and alt > context.cruise_alt_floor:
        # 1. TIGHTENED ENTRY GUARD: Must be genuinely flat (|altr| <= 250) 
        # and near our apex to initiate a true cruise state.
        is_leveling_at_apex = context.is_level_flight and (alt >= (state.max_altitude_seen - 500))
        
        has_passed_tod = (state.max_altitude_seen - alt) > 500.0

        if has_passed_tod:
            phase_scores[5] = 0.0  
        else:
            # 🎯 VMODE EN-ROUTE CAPTURE: If automation is tracking a stable hold profile,
            # enforce Cruise stability and override minor physical ballooning anomalies
            if context.automation_engaged and context.automation_intent == 0:
                phase_scores[5] = 0.98
                phase_scores[4] = 0.00  # Cut down climb chatter completely
            elif state.current_phase == 5 and context.is_structurally_climbing:
                phase_scores[5] = 0.30  
            elif is_leveling_at_apex or state.current_phase == 5:
                phase_scores[5] = 0.95
                if altr is not None and abs(altr) <= 100:
                    phase_scores[5] += 0.05
            else:
                phase_scores[5] = 0.30 
                
        phase_scores[5] = _clip(phase_scores[5])

    # --- Phase 6: Approach / Descent (Capturing Level-offs) ---
    if not context.on_ground:
        current_ralt = ralt if ralt is not None else alt
        has_passed_tod = (state.max_altitude_seen - alt) > 500.0
        
        # 🎯 FIX: Approach rules should only execute if we have definitively passed our TOD 
        # OR if we are low but have ALREADY established our cruise flight ceiling phase.
        is_valid_descent_regime = has_passed_tod or (current_ralt < context.approach_ralt_max and state.max_altitude_seen > context.cruise_alt_floor)

        if is_valid_descent_regime:
            if altr is not None and altr < context.approach_sink_rate:
                phase_scores[6] = 0.95  # Active descent segment
            elif abs(altr) <= 400 and has_passed_tod:
                phase_scores[6] = 0.90  # Captures level flight segments during descent
            else:
                phase_scores[6] = 0.50
        else:
            # Strictly zero out approach scoring during initial climb envelopes
            phase_scores[6] = 0.0
    else:
        phase_scores[6] = 0.0
        
    phase_scores[6] = _clip(phase_scores[6])

# --- Phase 7: Rollout (Strict Dynamic Exclusivity) ---
    if context.on_ground:
        if state.has_taken_off and gs > 10.0:
            # High score strictly during active touchdown deceleration
            if spoilers_deployed() or abrk_deployed() or gs > context.taxi_threshold:
                phase_scores[7] = 0.95
                if state.current_phase == 6: # Strong preference if transitioning directly from Approach
                    phase_scores[7] += 0.05
            else:
                # Drop to zero as soon as the aerodynamic braking stops and speed is low.
                # This forces an immediate, uninhibited handoff to Taxi (Phase 2).
                phase_scores[7] = 0.0 
            
            phase_scores[7] = _clip(phase_scores[7])
        else:
            phase_scores[7] = 0.0
    else:
        phase_scores[7] = 0.0

    # =========================================================================
    # HYBRID INTERACTION LAYER: Balanced Overrides & Neutral Hold States
    # =========================================================================
    if activate_openap_macro_guard:

        if not context.on_ground:
            # --- AIRBORNE STRATEGY ---

            # Extract decoupled intent features
            is_automation_holding = context.automation_engaged and (context.automation_intent == 0)
            is_automation_climbing = context.automation_engaged and (context.automation_intent == 1)
            is_automation_descending = context.automation_engaged and (context.automation_intent == -1)
            
            if openap_macro_guard == 'CL':
                # 🛠️ PROTECTION: If we are established in Cruise at high altitude, 
                # do not let a transient macro-climb flag crush our active phase.
                # 🛠️ FIX: Allow flight director checks to protect or transition into Cruise 
                # even if the current state is still tracking in Climb (Phase 4).
                is_established_or_holding = (state.current_phase == 5) or (context.is_at_altitude and is_automation_holding)
                
                if is_established_or_holding and not is_automation_climbing:
                    phase_scores[5] = max(phase_scores[5], 0.95)  # 🛠️ Firm up cruise dominance
                    phase_scores[4] = min(phase_scores[4], 0.05)
                    phase_scores[6] = 0.0  # 🛠️ Hard-kill approach bleed inside high cruise
                elif state.current_phase == 6:
                    # 🛡️ FIX FOR FLIGHT 2: If we are tracking an Approach/Descent profile,
                    # a positive jitter frame must NOT destroy our descent state momentum.
                    phase_scores[6] = max(phase_scores[6], 0.85)
                    phase_scores[4] = min(phase_scores[4], 0.10)
                    phase_scores[5] *= 0.01
                else:
                    # Lock Climb cleanly, but use soft suppression instead of hard zeroing
                    phase_scores[4] = max(phase_scores[4], 0.90)
                    # Actively crush Cruise/Descent scores to ensure no climb stutters
                    phase_scores[5] *= 0.01  
                    phase_scores[6] *= 0.01  
                
            elif openap_macro_guard == 'LVL':
                # 🎯 THE STABILIZING ANCHOR: 
                # OpenAP confirms level flight. If we are currently tracking an active
                # dynamic phase (Climb or Descent), protect that phase's continuity 
                # against physics-layer noise or regional altitude-floor crossovers.
                
                if state.current_phase == 4:  # Leveling off mid-Climb
                    # Boost current phase to ensure it stays highly competitive
                    phase_scores[4] = max(phase_scores[4], 0.85)
                    # Proportionally damp Cruise so a transient speed drop can't capture it
                    phase_scores[5] *= 0.10  
                    phase_scores[6] = 0.0  # 🛠️ Hard clamp premature descent noise
                    
                elif state.current_phase == 6:  # Intermediate level-off during Descent
                    # Protect the descent/approach tracking
                    phase_scores[6] = max(phase_scores[6], 0.85)
                    # Proportionally damp Cruise to avoid shifting back up mid-descent
                    phase_scores[5] *= 0.10
                    phase_scores[4] = 0.0  # 🛠️ ZERO out spurious Climb noise!

                elif state.current_phase == 5 or is_automation_holding:
                    # 🛠️ FIX FOR FLIGHT 2: If established in high-altitude cruise, level noise 
                    # should keep Cruise pristine and completely clear out approach bleed.
                    # Firmly anchor cruise if automation indicates an active hold pattern
                    phase_scores[5] = max(phase_scores[5], 0.95)
                    phase_scores[6] = 0.0
                    phase_scores[4] = 0.0
                else:
                    # If we are already in Cruise (5) or another state, let the 
                    # physical layer govern the profile naturally.
                    pass
                
            elif openap_macro_guard == 'CR':
                # --- GENUINE CRUISE PROFILE GATING ---
                is_automation_holding = context.automation_engaged and (context.automation_intent == 0)
                is_automation_climbing = context.automation_engaged and (context.automation_intent == 1)
                # 🛠️ FIX: Leverage the clean, top-level single source of truth 'was_recently_cruising'
                # to prevent a previous-frame state tracking lag from causing an FSM threshold lockout.
                # 🎯 THE ULTIMATE SHIELD: Combine physical telemetry validation AND historical memory
                is_established_level_or_recent = (
                    (state.current_phase == 5) or 
                    is_automation_holding or 
                    context.is_level_flight or 
                    was_recently_cruising
                )

                # 🛠️ THE REAL CORRECTION: Only let an automation climb intent kick us out of Cruise 
                # if the aircraft is NOT physically verified to be in level flight.
                is_actively_climbing_automation = is_automation_climbing and not context.is_level_flight

                # Genuine Cruise Altitude Stabilization
                if is_established_level_or_recent and not is_actively_climbing_automation:
                    # Let Cruise take over authoritatively
                    phase_scores[5] = max(phase_scores[5], 0.95)
                    phase_scores[4] *= 0.01  # Zero out climb noise
                    phase_scores[6] = 0.0  # 🛠️ Hard clear approach stutters
                elif state.current_phase == 4 and is_automation_climbing:
                    # Trapped intermediate step-climb hold segment: Protect Climb state tracking
                    phase_scores[5] *= 0.01
                    phase_scores[4] = max(phase_scores[4], 0.90)
                    phase_scores[6] *= 0.01 
                elif state.current_phase == 4 and not (context.is_level_flight or is_automation_holding):
                    # Trapped intermediate step-climb hold segment: Protect Climb state tracking
                    phase_scores[5] *= 0.01
                    phase_scores[4] = max(phase_scores[4], 0.90)
                    phase_scores[6] *= 0.01  
                else:
                    # Legitimate entrance gate to Cruise phase
                    phase_scores[5] = max(phase_scores[5], 0.90)
                    phase_scores[4] *= 0.01
                    phase_scores[6] = 0.0
                    
            elif openap_macro_guard == 'DE':
                # --- STEP-DOWN RESILIENCE ---

                # 3. Leverage Flight Director Modes (p.vmode/APFD) 
                # If the system is actively holding an altitude capture window (Vertical Navigation path/ALT Hold)
                # rather than capturing a terminal approach glide path indicator (like G/S or Land mode)
                is_fd_holding_flight_level = (p.vmode in [1, 2, 3]) and (p.apfd == 1) # Match your flight director mapping IDs

                # 4. Resolve robust step-down condition
                is_step_down = was_recently_cruising and (
                    acms_phase == 5 or 
                    is_automation_holding or 
                    is_automation_descending or 
                    is_fd_holding_flight_level
                )
    
                if is_step_down and (current_alt > context.cruise_alt_floor - 4000.0):                  # 🛡️ THE BULLETPROOF SHIELD: OpenAP claims 'DE' but the aircraft is physically 
                    # descending at a controlled rate. Maintain the integrity of the Cruise phase.
                    phase_scores[5] = max(phase_scores[5], 0.92)  # Secure Cruise confidence floor
                    phase_scores[6] = min(phase_scores[6], 0.40)  # Keep Approach tracking but blocked from winning
                    phase_scores[4] *= 0.01
                else:
                    # Genuinely out of the cruise window or showing a verified downward profile exit.
                    for phase_id in [1, 2, 3, 4, 5]:  
                        phase_scores[phase_id] *= 0.01
                    
                    # Lock in Descent minimum floor safely
                    phase_scores[6] = max(phase_scores[6], 0.90)

                    # 🛡️ Safe fallback block: always has scope availability now
                    if was_recently_cruising and acms_phase != 6:
                        phase_scores[5] = max(phase_scores[5], 0.85)
                
        # NOTE: Ground strategy else-block removed. The core physical layer 
        # handles ground phases flawlessly without needing OpenAP macro interference.

    # =========================================================================
    # 🎯 MONOTONIC FLIGHT-GATE CONSTRAINTS (ANTI-STUTTER & TRANSITIONAL GUARDS)
    # =========================================================================
    # Fix A: Resolve Ground-Stopping Stutters (7920s - 8500s)
    # If the aircraft has successfully flown and has now returned to the ground,
    # block it from ever reverting backward into a PREFLIGHT initialization state.
    if state.has_taken_off and context.on_ground:
        phase_scores[1] = 0.0  # Zero out Preflight completely post-flight
        if state.current_phase in [7, 2]:
            # If we are rolling out or taxiing in, ensure Taxi is heavily favored
            # over any lingering ground-state noise.
            phase_scores[2] = max(phase_scores[2], 0.95)

    # =========================================================================
    # 🎯 MONOTONIC FLIGHT-GATE CONSTRAINTS (THE COMBINED DEFENSE SHIELD)
    # =========================================================================
    # If the aircraft is not on the ground, apply additional constraints to prevent phase oscillations.
    if not context.on_ground:
        
        is_dual_descent_consensus = (openap_macro_guard == 'DE') and (acms_phase == 6)
        is_automation_holding_or_climbing = context.automation_engaged and (context.automation_intent in [0, 1])
        
        # Calculate structural descent state relative to seen apex memory
        has_passed_tod = False
        if state.max_altitude_seen is not None and current_alt is not None:
            has_passed_tod = (state.max_altitude_seen - current_alt) > 500.0

        # --- Rule A: Protect BOTH active Climb (4) and established Cruise (5) ---
        if state.current_phase in [4, 5]:
            
            # 1. Flight Director Mode Protection:
            # If the automation explicitly intends to hold or climb, lock out Approach entirely.
            if is_automation_holding_or_climbing and not is_dual_descent_consensus:
                phase_scores[6] = 0.0
                
            else:
                # 2. Kinetic / Sensor Loss Fallback Protection:
                # If automation data drops out or is disconnected, drop back to vertical rate thresholds.
                if altr is not None and altr > -1200.0 and not is_dual_descent_consensus: 
                    phase_scores[6] = min(phase_scores[6], 0.10)

        # --- Rule B: High Altitude Protection (The Low Altitude Shield) ---
        # 🛡️ Structural Defense: If both OpenAP and ACMS agree on terminal descent, 
        # bypass the altitude shield. Otherwise, strictly block premature Approach scoring.
        if is_dual_descent_consensus:
            pass
        else:
            # If the aircraft hasn't passed a real TOD or is safely above the terminal environment floor,
            # ensure Approach can NEVER spike high enough to win.
            # 🎯 Clean, scope-safe implementation using true state history
            if was_recently_cruising and not has_passed_tod:
                phase_scores[6] = min(phase_scores[6], 0.00)

        # --- Rule C: Climb Phase Suppression ---
        # Ensure that if OpenAP firmly mandates a climb sequence, Cruise does not experience
        # confidence leakage during minor intermediate leveling plateaus.
        if openap_macro_guard == 'CL' and context.automation_intent == 1:
            phase_scores[5] = min(phase_scores[5], 0.10)

    # FIX C: COLD REBOOT / NEXT FLIGHT RE-INITIALIZATION (Time = 8580s)
    # When the N1 Command (N1C) drops to 0.0, the aircraft is demanding zero thrust.
    # On the ground, this confirms an engine shutdown and a transition to the next flight sequence.
    effective_thrust_signal = context.n1_effective if context.n1_effective is not None else 99.0
    is_engine_shutdown = (effective_thrust_signal <= 0.1) and context.on_ground
    
    if is_engine_shutdown and state.has_taken_off:
        # Wipe the entire state machine history completely
        state.has_taken_off = False
        state.max_altitude_seen = 0.0
        state.current_phase = 1  # Force a clean drop back to PREFLIGHT
        state.current_confidence = 1.0
        
        # Zero out all scores and give Preflight total dominance for the next flight
        phase_scores = {k: 0.0 for k in PHASE_MAP.keys()}
        phase_scores[1] = 1.0
    


    # --- Apply transition penalty to non-current phases ---
    penalized_scores = phase_scores.copy()
    for p in penalized_scores:
        if p != state.current_phase and p != 0:
            penalized_scores[p] = _clip(penalized_scores[p] - transition_penalty)

    # --- Determine best candidate ---
    best_phase = max([1, 2, 3, 4, 5, 6, 7], key=lambda p: (penalized_scores[p], p))
    best_score = penalized_scores[best_phase]

    current_score = penalized_scores.get(state.current_phase, 0.0)

    # --- Hysteresis / switching logic ---
    allow_switch = False

    if state.current_phase == 0:
        # No current phase: allow switch if evidence is sufficient
        allow_switch = best_score >= 0.45
    else:
        # Need stronger evidence than current phase by some margin
        stronger_enough = (best_score - current_score) >= switch_margin

        # Extra hysteresis: if current phase is still reasonably supported, keep it
        current_supported = current_score >= (best_score - hysteresis_margin)

        allow_switch = stronger_enough and dwell_time_ok and not current_supported

    # --- Update phase state ---
    if allow_switch:
        if best_phase != state.current_phase:
            state.current_phase = best_phase
            state.current_confidence = best_score
            state.last_transition_time = now
            state.dwell_start_time = now
            state.phase_entry_count[best_phase] = state.phase_entry_count.get(best_phase, 0) + 1
    else:
        # Persist current phase, but refresh confidence from current evidence
        if state.current_phase == 0:
            # If no current phase, hold best candidate if it is good enough
            if best_score >= 0.45:
                state.current_phase = best_phase
                state.current_confidence = best_score
                state.last_transition_time = now
                state.dwell_start_time = now
                state.phase_entry_count[best_phase] = state.phase_entry_count.get(best_phase, 0) + 1
            else:
                state.current_phase = 0
                state.current_confidence = 1.0 - best_score
        else:
            # Keep current phase, update confidence based on current-phase score
            state.current_confidence = current_score

    state.last_update_time = now

    computed_phase = state.current_phase
    computed_confidence = state.current_confidence

    agree_with_acms = (computed_phase == acms_phase)

    if acms_phase == 0:
        recommended_phase = computed_phase
    elif agree_with_acms:
        recommended_phase = acms_phase
    else:
        recommended_phase = computed_phase


    return {
        "computed_flight_phase": computed_phase,
        "computed_phase_name": PHASE_MAP.get(computed_phase, "Unknown"),
        "computed_confidence": round(computed_confidence, 3),
        "phase_confidences": {PHASE_MAP[k]: round(v, 3) for k, v in phase_scores.items()},
        "penalized_phase_confidences": {PHASE_MAP[k]: round(v, 3) for k, v in penalized_scores.items()},
        "acms_flight_phase": acms_phase,
        "acms_phase_name": PHASE_MAP.get(acms_phase, "Unknown"),
        "agree_with_acms": agree_with_acms,
        "recommended_phase": recommended_phase,
        "recommended_phase_name": PHASE_MAP.get(recommended_phase, "Unknown"),
        "state": state,
        "dwell_time_ok": dwell_time_ok,  # Now guaranteed to exist
        "time_since_dwell": time_since_dwell,
        "is_inertially_moving": bool(context.is_inertially_moving),
    }

# Assumes FlightPhaseState and compute_flight_phase_stateful
# are already defined in your environment.

def _mode_or_last(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return np.nan
    mode_vals = series.mode()
    if not mode_vals.empty:
        return mode_vals.iloc[-1]
    return series.iloc[-1]

def _safe_last(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return np.nan
    return series.iloc[-1]

def _safe_mean(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return np.nan
    return float(series.mean())

def _safe_median(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return np.nan
    return float(series.median())

def summarize_window(window: pd.DataFrame) -> Dict[str, Any]:
    """
    Create robust summary values for a sliding window.
    """
    summary = {}

    # Discrete / stateful signals: last stable or mode
    for col in ["WOW", "LGDN", "LGUP", "SPLG", "SPLY"]:
        if col in window.columns:
            summary[col] = _mode_or_last(window[col])

    # Continuous signals: median is usually robust in windows
    for col in ["GS", "RALT", "ALT", "ALTR", "IVV", "PLA.1", "PLA.2", "PLA.3", "PLA.4", "N1T", "N1C", "FLAP", "BLAC", "blac_smooth"]:
        if col in window.columns:
            summary[col] = _safe_median(window[col])

    # ABRK: use last non-null or max if you prefer a "deployed at any point" view
    if "ABRK" in window.columns:
        summary["ABRK"] = _safe_last(window["ABRK"])

    # ACMS phase: use mode or last
    if "PH" in window.columns:
        summary["PH"] = _mode_or_last(window["PH"])

    return summary


def classify_flight_phases_sliding_window(
    df: pd.DataFrame,
    window_size: int = 30,
    step_size: int = 5,
    min_dwell_seconds: float = 20.0,
    hysteresis_margin: float = 0.12,
    transition_penalty: float = 0.08,
    switch_margin: float = 0.15,
    time_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Slide a window over df and classify each window.

    Parameters
    ----------
    df : pd.DataFrame
        Flight data.
    window_size : int
        Number of rows per window.
    step_size : int
        Step between windows.
    min_dwell_seconds : float
        Minimum time to stay in a phase before switching.
    hysteresis_margin : float
        Hysteresis threshold.
    transition_penalty : float
        Penalty applied to non-current phases.
    switch_margin : float
        Minimum score advantage required to switch phases.
    time_column : str or None
        Optional timestamp column. If provided, used for dwell timing.
        Otherwise uses row index as pseudo-time.

    Returns
    -------
    pd.DataFrame
        One row per window with phase results and summary stats.
    """
    if len(df) == 0:
        return pd.DataFrame()

    results = []
    state = FlightPhaseState()

    # Create a pseudo-time if no timestamp column exists
    if time_column is not None and time_column in df.columns:
        time_values = pd.to_datetime(df[time_column], errors="coerce")
    else:
        time_values = pd.Series(np.arange(len(df)), index=df.index)

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window = df.iloc[start:end]
        summary = summarize_window(window)

        # Pull values with defaults
        wow = int(summary.get("WOW", 0)) if not pd.isna(summary.get("WOW", np.nan)) else 0
        gs = float(summary.get("GS", 0.0)) if not pd.isna(summary.get("GS", np.nan)) else 0.0
        pla1 = float(summary.get("PLA.1", 0.0)) if not pd.isna(summary.get("PLA.1", np.nan)) else 0.0
        pla2 = float(summary.get("PLA.2", 0.0)) if not pd.isna(summary.get("PLA.2", np.nan)) else 0.0
        pla3 = float(summary.get("PLA.3", 0.0)) if not pd.isna(summary.get("PLA.3", np.nan)) else 0.0
        pla4 = float(summary.get("PLA.4", 0.0)) if not pd.isna(summary.get("PLA.4", np.nan)) else 0.0

        lgdn = summary.get("LGDN", None)
        lgdn = None if pd.isna(lgdn) else int(lgdn)

        lgup = summary.get("LGUP", None)
        lgup = None if pd.isna(lgup) else int(lgup)

        ralt = summary.get("RALT", None)
        ralt = None if pd.isna(ralt) else float(ralt)

        alt = summary.get("ALT", None)
        alt = None if pd.isna(alt) else float(alt)

        altr = summary.get("ALTR", None)
        altr = None if pd.isna(altr) else float(altr)

        ivv = summary.get("IVV", None)
        ivv = None if pd.isna(ivv) else float(ivv)

        abrk = summary.get("ABRK", None)
        abrk = None if pd.isna(abrk) else float(abrk)

        splg = summary.get("SPLG", 0)
        splg = 0 if pd.isna(splg) else int(splg)

        sply = summary.get("SPLY", 0)
        sply = 0 if pd.isna(sply) else int(sply)

        flap = summary.get("FLAP", None)
        flap = None if pd.isna(flap) else float(flap)

        n1t = summary.get("N1T", None)
        n1t = None if pd.isna(n1t) else float(n1t)

        n1c = summary.get("N1C", None)
        n1c = None if pd.isna(n1c) else float(n1c)

        n1co = summary.get("N1CO", None)
        n1co = None if pd.isna(n1co) else int(n1co)

        acms_phase = summary.get("PH", 0)
        acms_phase = 0 if pd.isna(acms_phase) else int(acms_phase)

        # 🎯 FIX 2: Extract the longitudinal accelerometers
        blac = float(summary.get("BLAC", 0.0)) if not pd.isna(summary.get("BLAC", np.nan)) else 0.0
        blac_smooth = float(summary.get("blac_smooth", 0.0)) if not pd.isna(summary.get("blac_smooth", np.nan)) else 0.0

        openap_macro_guard = summary.get("openap_macro_guard", "UNKNOWN")
        if pd.isna(openap_macro_guard):
            openap_macro_guard = "UNKNOWN"

        # Determine representative time for this window
        if time_column is not None and time_column in df.columns:
            t0 = time_values.iloc[start]
            t1 = time_values.iloc[end - 1]
            now = pd.Timestamp(t1).timestamp() if hasattr(t1, "timestamp") else float(end - 1)
        else:
            now = float(end - 1)

# 1. Capture the exact argument keys the state function expects
        valid_args = inspect.signature(compute_flight_phase_stateful).parameters.keys()

        # 2. Normalize the raw window dictionary to lowercase to match Python parameters
        # This safely maps 'CASS' -> 'cass', 'ALT' -> 'alt', etc., universally.
        normalized_window = {k.lower(): v for k, v in summary.items()}

        # 3. Inject our calculated loop state variables
        normalized_window["state"] = state
        normalized_window["wow"] = wow
        normalized_window["gs"] = gs
        normalized_window["now"] = now
        normalized_window["openap_macro_guard"] = str(openap_macro_guard)
        normalized_window["min_dwell_seconds"] = min_dwell_seconds
        normalized_window["hysteresis_margin"] = hysteresis_margin
        normalized_window["transition_penalty"] = transition_penalty
        normalized_window["switch_margin"] = switch_margin

        # 4. Filter out any extra metadata keys that don't belong in the function signature
        filtered_data = {k: v for k, v in normalized_window.items() if k in valid_args}

        # 5. Explode safely via pure keyword unpacking
        result = compute_flight_phase_stateful(**filtered_data)

        row = {
            "window_start": start,
            "window_end": end - 1,
            "computed_flight_phase": result["computed_flight_phase"],
            "computed_phase_name": result["computed_phase_name"],
            "computed_confidence": result["computed_confidence"],
            "acms_flight_phase": result["acms_flight_phase"],
            "acms_phase_name": result["acms_phase_name"],
            "agree_with_acms": result["agree_with_acms"],
            "recommended_phase": result["recommended_phase"],
            "recommended_phase_name": result["recommended_phase_name"],
        }

        # Add summary features to the output for debugging
        for k, v in summary.items():
            row[f"summary_{k}"] = v

        results.append(row)

    return pd.DataFrame(results)


def main():
    state = FlightPhaseState()

    result = compute_flight_phase_stateful(
        state=state,
        wow=1,
        gs=145.0,
        pla1=42, pla2=42, pla3=41, pla4=41,
        lgdn=1,
        lgup=1,
        ralt=8200,
        alt=21000,
        altr=1200,
        ivv=1100,
        abrk=0,
        splg=0,
        sply=0,
        flap=0,
        n1t=68,
        n1c=67,
        n1co=0,
        acms_phase=4,
        now=1710000000.0
    )

    print(result["computed_phase_name"], result["computed_confidence"])

# Recommmended starting values
# min_dwell_seconds = 20.0
# hysteresis_margin = 0.12
# transition_penalty = 0.08
# switch_margin = 0.15

    phase_df = classify_flight_phases_sliding_window(
        df=flight_data,
        window_size=30,
        step_size=5,
        min_dwell_seconds=20.0,
        hysteresis_margin=0.12,
        transition_penalty=0.08,
        switch_margin=0.15,
        time_column="timestamp"
    )

    print(phase_df.head())


if __name__ == "__main__":
    main()
