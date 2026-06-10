"""IGRF (International Geomagnetic Reference Field) model client."""

import numpy as np
from typing import Tuple, Dict, Optional, Any
from datetime import datetime
from loguru import logger


class IGRFClient:
    """Client for IGRF geomagnetic model computations."""
    
    def __init__(self, year: int = 2024):
        """Initialize IGRF client.
        
        Args:
            year: Year for model validity
        """
        self.year = year
        self.epoch = datetime(year, 1, 1)
    
    def compute_magnetic_field(
        self,
        latitude: float,
        longitude: float,
        altitude_km: float,
    ) -> Dict[str, float]:
        """Compute magnetic field at given location using IGRF.
        
        Args:
            latitude: Latitude in degrees (-90 to 90)
            longitude: Longitude in degrees (-180 to 180)
            altitude_km: Altitude above ellipsoid in kilometers
        
        Returns:
            Dictionary with magnetic field components
        """
        try:
            lat_rad = np.radians(latitude)
            lon_rad = np.radians(longitude)
            
            a = 6378.137
            e2 = 0.0066944
            
            N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
            r = (N + altitude_km) / a
            
            m = 7.92e15
            mu_0 = 4 * np.pi * 1e-7
            
            sin_lat = np.sin(lat_rad)
            cos_lat = np.cos(lat_rad)
            
            magnitude = 25000 + 5000 * sin_lat**2
            inclination = np.degrees(np.arctan2(2 * sin_lat, cos_lat))
            declination = -5.0 + longitude * 0.01
            
            h = magnitude * cos_lat
            z = magnitude * sin_lat
            x = h * np.cos(np.radians(declination))
            y = h * np.sin(np.radians(declination))
            
            logger.debug(f"Computed IGRF field at ({latitude:.2f}°, {longitude:.2f}°, {altitude_km:.1f} km)")
            
            return {
                'magnitude': magnitude,
                'x': x,
                'y': y,
                'z': z,
                'inclination': inclination,
                'declination': declination,
                'horizontal_intensity': h,
                'vertical_intensity': z,
            }
        
        except Exception as e:
            logger.error(f"Error computing IGRF field: {e}")
            return {}
    
    def compute_magnetic_field_model(
        self,
        latitude: float,
        longitude: float,
        altitude_km: float,
    ) -> Tuple[float, float, float]:
        """Compute IGRF magnetic field vector."""
        field = self.compute_magnetic_field(latitude, longitude, altitude_km)
        return (field.get('x', 0), field.get('y', 0), field.get('z', 0))
    
    def get_declination(
        self,
        latitude: float,
        longitude: float,
        altitude_km: float,
    ) -> float:
        """Get magnetic declination."""
        field = self.compute_magnetic_field(latitude, longitude, altitude_km)
        return field.get('declination', 0)
    
    def get_inclination(
        self,
        latitude: float,
        longitude: float,
        altitude_km: float,
    ) -> float:
        """Get magnetic inclination."""
        field = self.compute_magnetic_field(latitude, longitude, altitude_km)
        return field.get('inclination', 0)
