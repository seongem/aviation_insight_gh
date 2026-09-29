from enum import Enum, auto
import numpy as np
from openap.phase import FlightPhase  # type: ignore

class AircraftType(Enum):
    A320 = auto()
    B737 = auto()
    RJ85 = auto()   # Custom added type
    UNKNOWN = auto()

class OpenAPWrapper:
    """
    Unified factory wrapper for OpenAP's FlightPhase analyzer.
    Provides native OpenAP calculations for supported fleet types, and falls back 
    to a robust rule-based kinematic profile for unsupported types like the RJ85.
    """
    def __init__(self, aircraft_type: AircraftType):
        self.aircraft_type = aircraft_type
        # Only initialize native OpenAP if supported
        if self.aircraft_type in [AircraftType.A320, AircraftType.B737]:
            self.fp = FlightPhase()
            self.is_native = True
        else:
            self.fp = None
            self.is_native = False

    def set_trajectory(self, time, alt, gs, altr):
        """Buffers trajectory data into the active tracking engine."""
        self.time = np.array(time)
        self.alt = np.array(alt)
        self.gs = np.array(gs)
        self.altr = np.array(altr)
        
        if self.is_native and self.fp is not None:
            self.fp.set_trajectory(time, alt, gs, altr)

    def phaselabel(self) -> np.ndarray:
        """
        Returns macro labels matching OpenAP string standards:
        ['GND', 'CL', 'CR', 'LVL', 'DE']
        """
        if self.is_native and self.fp is not None:
            return self.fp.phaselabel()
        
        # --- CUSTOM KINEMATIC BACKUP FOR RJ85 / UNSUPPORTED TYPES ---
        num_samples = len(self.time)
        labels = np.full(num_samples, 'GND', dtype='<U3')
        
        if num_samples == 0:
            return labels

        for i in range(num_samples):
            current_alt = self.alt[i]
            current_altr = self.altr[i]
            current_gs = self.gs[i]

            # Kinematic estimation thresholds based on RJ85 capabilities
            if current_alt < 500 and current_gs < 100:
                labels[i] = 'GND'
            elif current_altr > 300 and current_alt >= 500:
                labels[i] = 'CL'   # Climb
            elif current_altr < -300 and current_alt >= 500:
                labels[i] = 'DE'   # Descent
            elif abs(current_altr) <= 300 and current_alt > 20000:
                labels[i] = 'CR'   # Cruise (Typical RJ85 ceiling is ~29,000-31,000 ft)
            else:
                labels[i] = 'LVL'  # Level Flight / Intermediate Level-off
                
        return labels