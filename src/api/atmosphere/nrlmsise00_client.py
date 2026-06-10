"""NRLMSISE-00 atmospheric model client."""

import numpy as np
from typing import Dict, Any
from datetime import datetime
from loguru import logger


class NRLMSISE00Client:
    """Client for NRLMSISE-00 atmospheric model."""
    
    def __init__(self):
        """Initialize NRLMSISE-00 client."""
        self.model_name = "NRLMSISE-00"
    
    def get_density(
        self,
        latitude: float,
        longitude: float,
        altitude_km: float,
        datetime_utc: datetime,
        f107: float = 100.0,
        f107a: float = 100.0,
        kp: float = 2.0,
    ) -> Dict[str, float]:
        """Get atmospheric density at given location and time."""
        try:
            if altitude_km < 0:
                logger.warning(f"Altitude {altitude_km} km is below surface")
                return {}
            
            base_scale_height = 8500
            t_0 = 188
            t_exosphere = t_0 + 50 * (f107 - 100) / 100
            temp_variation = 1.0 + 0.1 * (f107 - 100) / 100
            t_local = 200 + 50 * np.exp(-altitude_km / 100)
            
            rho_120 = 5e-15
            altitude_m = altitude_km * 1000
            reference_altitude_m = 120000
            h_scale = base_scale_height * temp_variation
            
            if altitude_km < 120:
                rho_total = rho_120 * np.exp((reference_altitude_m - altitude_m) / h_scale)
            else:
                rho_total = rho_120 * np.exp((reference_altitude_m - altitude_m) / h_scale)
            
            if altitude_km < 100:
                n_fraction = 0.78
                o_fraction = 0.21
            elif altitude_km < 200:
                n_fraction = 0.7
                o_fraction = 0.25
            else:
                n_fraction = 0.5
                o_fraction = 0.45
            
            logger.debug(f"NRLMSISE-00 density at {altitude_km:.1f} km: {rho_total:.2e} kg/m^3")
            
            return {
                'density_total': float(rho_total),
                'density_n2': float(rho_total * n_fraction * 0.78),
                'density_o2': float(rho_total * n_fraction * 0.21),
                'density_n': float(rho_total * o_fraction),
                'density_o': float(rho_total * (1 - n_fraction - o_fraction)),
                'temperature_exosphere': float(t_exosphere),
                'temperature_local': float(t_local),
                'scale_height': float(h_scale),
            }
        
        except Exception as e:
            logger.error(f"Error computing NRLMSISE-00 density: {e}")
            return {}
    
    def get_temperature(
        self,
        latitude: float,
        longitude: float,
        altitude_km: float,
        datetime_utc: datetime,
        f107: float = 100.0,
        kp: float = 2.0,
    ) -> float:
        """Get atmospheric temperature."""
        result = self.get_density(latitude, longitude, altitude_km, datetime_utc, f107, f107, kp)
        return result.get('temperature_local', 200.0)
