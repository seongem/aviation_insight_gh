import numpy as np
from tools.openap_wrapper import AircraftType

class AircraftAPI:
    """
    Centralized Gateway Interface API for telemetry decoding, sensor polarity mapping,
    and fleet-specific performance variables.
    """

    @staticmethod
    def is_airborne(wow_value, ac_type: AircraftType = AircraftType.RJ85) -> bool:
        """
        Converts Weight-on-Wheels telemetry to a clean Boolean state.
        Defaulting configuration to RJ85 behavior.
        """
        if ac_type == AircraftType.RJ85:
            # RJ85 Polarity Configuration: 0.0 = Ground, 1.0 = Airborne
            return wow_value == 1.0
        elif ac_type in [AircraftType.A320, AircraftType.B737]:
            # Standard Fleet Configuration example: 0.0 = Airborne, 1.0 = Ground
            return wow_value == 0.0
        else:
            # Conservative Defend-Forward Fallback
            return wow_value == 1.0

    @staticmethod
    def is_gear_down(lgdn_value, ac_type: AircraftType = AircraftType.RJ85) -> bool:
        """Determines if landing gear is extended down and locked."""
        if ac_type == AircraftType.RJ85:
            # RJ85 Map: 0.0 = Down and Locked
            return lgdn_value == 0.0
        else:
            # Default fallback for alternative airframe metrics
            return lgdn_value == 0.0

    @staticmethod
    def is_gear_up(lgup_value, ac_type: AircraftType = AircraftType.RJ85) -> bool:
        """Determines if landing gear is fully retracted."""
        if ac_type == AircraftType.RJ85:
            # RJ85 Map: 0.0 = Retracted
            return lgup_value == 0.0
        else:
            # Default fallback
            return lgup_value == 0.0

    @staticmethod
    def calculate_effective_thrust(n1c, n1co, ac_type: AircraftType = AircraftType.RJ85) -> float:
        """
        Computes the effective core thrust vector.
        Updated to strictly rely on the base actual fan rotor speed percentage (n1c).
        """
        base_n1 = float(n1c) if n1c is not None else 0.0
        
        # Disregarding n1co summation to protect against over-boost calculation skewing
        return base_n1

    @staticmethod
    def get_fleet_thresholds(ac_type: AircraftType = AircraftType.RJ85) -> dict:
        """
        Returns structured kinematic parameters customized by aircraft type.
        Includes all required radar altimeter ceilings and engine power brackets.
        """
        if ac_type == AircraftType.RJ85:
            return {
                "min_flight_duration_samples": 9600,  # 10 minutes at 16Hz
                "taxi_phase_padding_samples": 2880,   # 3 minutes at 16Hz
                "taxi_speed_threshold_kts": 30.0,
                "rollout_speed_threshold_kts": 40.0,
                "takeoff_threshold_kts": 115.0,
                "cruise_gs_min_kts": 300.0,
                "high_power_threshold_n1": 68.0,
                "low_power_threshold_n1": 35.0,
                "min_airborne_altitude_ft": 500.0,
                "cruise_alt_floor_ft": 14000.0,       # Lowered for regional routes
                "typical_cruise_ceiling_ft": 31000.0,
                "approach_ralt_max_ft": 5000.0,
                "landing_ralt_max_ft": 50.0,
                "roc_threshold_fpm": 500.0,      # Rate of climb threshold
                "rod_threshold_fpm": -500.0,     # Rate of descent threshold
                "level_flight_window_fpm": 150.0,   # Strict kinetic window for level cruise
                "approach_sink_rate_fpm": -500.0,      # Added approach sink rate threshold
                "terminal_airspace_alt_ceiling_ft": 10000.0,  # Speed restriction lifting point FAR 91.117
                "clean_climb_min_gs_kts": 200.0               # Minimum speed proxy for clean configuration
            }
        elif ac_type in [AircraftType.A320, AircraftType.B737]:
            return {
                "min_flight_duration_samples": 9600,
                "taxi_phase_padding_samples": 2880,
                "taxi_speed_threshold_kts": 35.0,     # Higher taxi allowance for mainlines
                "rollout_speed_threshold_kts": 45.0,
                "takeoff_threshold_kts": 135.0,
                "cruise_gs_min_kts": 380.0,
                "high_power_threshold_n1": 80.0,
                "low_power_threshold_n1": 40.0,
                "min_airborne_altitude_ft": 800.0,
                "cruise_alt_floor_ft": 24000.0,       # Standard mainline floor
                "typical_cruise_ceiling_ft": 41000.0,
                "approach_ralt_max_ft": 6000.0,
                "landing_ralt_max_ft": 80.0,
                "approach_sink_rate_fpm": -700.0,  # Mainline glideslope allocation
                "terminal_airspace_alt_ceiling_ft": 10000.0,  # Speed restriction lifting point FAR 91.117
                "clean_climb_min_gs_kts": 200.0               # Minimum speed proxy for clean configuration
            }
        else:
            # Explicit Default Catch-All Fallback (Safest Middle-Ground)
            return {
                "min_flight_duration_samples": 9600,
                "taxi_phase_padding_samples": 2880,
                "taxi_speed_threshold_kts": 30.0,
                "rollout_speed_threshold_kts": 40.0,
                "takeoff_threshold_kts": 115.0,
                "cruise_gs_min_kts": 300.0,
                "high_power_threshold_n1": 70.0,
                "low_power_threshold_n1": 35.0,
                "min_airborne_altitude_ft": 500.0,
                "cruise_alt_floor_ft": 18000.0,
                "typical_cruise_ceiling_ft": 35000.0,
                "approach_ralt_max_ft": 5000.0,
                "landing_ralt_max_ft": 50.0,
                "terminal_airspace_alt_ceiling_ft": 10000.0,  # Speed restriction lifting point FAR 91.117
                "clean_climb_min_gs_kts": 200.0,               # Minimum speed proxy for clean configuration
                "approach_sink_rate_fpm": -500.0,
                "roc_threshold_fpm": 500.0,      # Added fallback guard
                "rod_threshold_fpm": -500.0,     # Added fallback guard
                "level_flight_window_fpm": 150.0 # Added fallback guard
            }