"""JB2008 atmospheric model client."""

import numpy as np
from typing import Dict, Any
from datetime import datetime
from loguru import logger


class JB2008Client:
    """Client for JB2008 atmospheric model."""
    
    def __init__(self):
        """Initialize JB2008 client."""
        self.model_name = "JB2008"
    
    def get_density(
        self,
        latitude: float,
        longitude: float,
        altitude_km: float,
        datetime_utc: datetime,
        f107: float = 100.0,
        f107a: float = 100.0,
        kp: float = 2.0,
        lst: float = None,
    ) -> Dict[str, float]:
        """Get atmospheric density using JB2008 model."""
        try:
            if altitude_km < 70 or altitude_km > 2000:
                logger.warning(f"Altitude {altitude_km} km outside JB2008 valid range")
            
            if lst is None:
                hour = datetime_utc.hour
                minute = datetime_utc.minute
                lst = hour + minute / 60 + longitude / 15
                lst = lst % 24
            
            alt_ref = np.array([70, 100, 150, 200, 300, 400, 500, 1000])
            rho_ref = np.array([5e-12, 5e-15, 5e-18, 1e-19, 3e-20, 1e-20, 5e-21, 1e-23])
            
            if altitude_km >= alt_ref[-1]:
                rho_0 = rho_ref[-1]
            elif altitude_km <= alt_ref[0]:
                rho_0 = rho_ref[0]
            else:
                idx = np.searchsorted(alt_ref, altitude_km)
                t = (altitude_km - alt_ref[idx-1]) / (alt_ref[idx] - alt_ref[idx-1])
                rho_0 = rho_ref[idx-1] * (rho_ref[idx] / rho_ref[idx-1])**t
            
            f107_effect = 1.0 + 0.01 * (f107 - 100) / 100
            kp_effect = 1.0 + 0.05 * kp / 9.0
            
            lt_rad = np.radians(lst * 15)
            diurnal_effect = 1.0 + 0.2 * np.cos(lt_rad) if altitude_km < 300 else 1.0
            
            rho_total = rho_0 * f107_effect * kp_effect * diurnal_effect
            
            t_500 = 500 + 200 * (f107 - 100) / 100
            t_inf = 1000 + 300 * (f107 - 100) / 100
            
            if altitude_km < 125:
                t_local = 200 + (t_500 - 200) * (altitude_km / 125)
            else:
                t_local = t_500 + (t_inf - t_500) * (1 - np.exp(-(altitude_km - 125) / 150))
            
            h_scale = 8500 * (t_local / 288)
            
            logger.debug(f"JB2008 density at {altitude_km:.1f} km: {rho_total:.2e} kg/m^3")
            
            return {
                'density_total': float(rho_total),
                'temperature': float(t_local),
                'scale_height': float(h_scale),
                'exosphere_temperature': float(t_inf),
                'solar_flux_factor': float(f107_effect),
                'geomagnetic_factor': float(kp_effect),
                'diurnal_factor': float(diurnal_effect),
            }
        
        except Exception as e:
            logger.error(f"Error computing JB2008 density: {e}")
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
        return result.get('temperature', 200.0)
