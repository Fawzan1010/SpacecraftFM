"""Orbital mechanics feature engineering."""

import numpy as np
from typing import Dict, Any
from loguru import logger


class OrbitFeatureExtractor:
    """Extract features from orbital data."""
    
    # Gravitational constant
    GM = 3.986004418e14  # m^3/s^2
    EARTH_RADIUS = 6.371e6  # m
    
    def __init__(self):
        """Initialize orbit feature extractor."""
        pass
    
    def extract_orbit_features(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> Dict[str, float]:
        """Extract orbital features from position and velocity.
        
        Args:
            position: Position vector in meters [x, y, z]
            velocity: Velocity vector in m/s [vx, vy, vz]
        
        Returns:
            Orbital features
        """
        features = {}
        
        # Position and velocity magnitudes
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        
        features['position_magnitude_m'] = float(r)
        features['position_magnitude_km'] = float(r / 1000)
        features['velocity_magnitude_m_s'] = float(v)
        features['velocity_magnitude_km_s'] = float(v / 1000)
        
        # Altitude above Earth surface
        altitude_m = r - self.EARTH_RADIUS
        features['altitude_m'] = float(altitude_m)
        features['altitude_km'] = float(altitude_m / 1000)
        
        # Specific orbital energy
        specific_energy = 0.5 * v**2 - self.GM / r
        features['specific_orbital_energy'] = float(specific_energy)
        
        # Semi-major axis
        if specific_energy < 0:  # Elliptical orbit
            semi_major_axis = -self.GM / (2 * specific_energy)
            features['semi_major_axis_m'] = float(semi_major_axis)
            features['semi_major_axis_km'] = float(semi_major_axis / 1000)
        else:
            features['semi_major_axis_m'] = float(np.inf)
            features['semi_major_axis_km'] = float(np.inf)
        
        # Orbital period (if elliptical)
        if specific_energy < 0:
            period_s = 2 * np.pi * np.sqrt(semi_major_axis**3 / self.GM)
            features['orbital_period_s'] = float(period_s)
            features['orbital_period_min'] = float(period_s / 60)
            features['orbital_period_hours'] = float(period_s / 3600)
        
        # Angular momentum
        h = np.cross(position, velocity)
        h_mag = np.linalg.norm(h)
        features['angular_momentum_magnitude'] = float(h_mag)
        features['specific_angular_momentum'] = float(h_mag)
        
        # Eccentricity
        e_vec = np.cross(velocity, h) / self.GM - position / r
        eccentricity = np.linalg.norm(e_vec)
        features['eccentricity'] = float(eccentricity)
        
        # Orbital elements
        if eccentricity < 1.0:  # Elliptical
            # Periapsis and apoapsis
            periapsis = semi_major_axis * (1 - eccentricity)
            apoapsis = semi_major_axis * (1 + eccentricity)
            
            features['periapsis_m'] = float(periapsis)
            features['periapsis_km'] = float(periapsis / 1000)
            features['apoapsis_m'] = float(apoapsis)
            features['apoapsis_km'] = float(apoapsis / 1000)
            
            features['periapsis_altitude_km'] = float((periapsis - self.EARTH_RADIUS) / 1000)
            features['apoapsis_altitude_km'] = float((apoapsis - self.EARTH_RADIUS) / 1000)
        
        # Inclination
        inclination = np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0))
        features['inclination_rad'] = float(inclination)
        features['inclination_deg'] = float(np.degrees(inclination))
        
        # Flight path angle
        cos_gamma = np.dot(position, velocity) / (r * v)
        gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
        features['flight_path_angle_rad'] = float(gamma)
        features['flight_path_angle_deg'] = float(np.degrees(gamma))
        
        return features
    
    def extract_true_anomaly(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> float:
        """Extract true anomaly.
        
        Args:
            position: Position vector in meters
            velocity: Velocity vector in m/s
        
        Returns:
            True anomaly in radians
        """
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        
        # Specific orbital energy
        energy = 0.5 * v**2 - self.GM / r
        
        if energy < 0:
            semi_major_axis = -self.GM / (2 * energy)
            
            # Radial velocity
            vr = np.dot(position, velocity) / r
            
            # Eccentricity vector magnitude
            h = np.cross(position, velocity)
            h_mag = np.linalg.norm(h)
            e_vec = np.cross(velocity, h) / self.GM - position / r
            eccentricity = np.linalg.norm(e_vec)
            
            # True anomaly
            cos_nu = (h_mag**2 / (self.GM * r) - 1) / eccentricity
            cos_nu = np.clip(cos_nu, -1.0, 1.0)
            
            nu = np.arccos(cos_nu)
            if vr < 0:
                nu = 2 * np.pi - nu
            
            return float(nu)
        
        return 0.0
    
    def extract_orbital_regime(
        self,
        altitude_km: float,
        inclination_deg: float,
    ) -> str:
        """Classify orbital regime.
        
        Args:
            altitude_km: Altitude in kilometers
            inclination_deg: Inclination in degrees
        
        Returns:
            Orbital regime classification
        """
        if altitude_km < 2000:
            return "LEO"  # Low Earth Orbit
        elif altitude_km < 35786:
            if 51.6 <= inclination_deg <= 51.8:
                return "SSO"  # Sun-Synchronous Orbit
            else:
                return "MEO"  # Medium Earth Orbit
        elif 35700 <= altitude_km <= 35900:
            return "GEO"  # Geostationary Orbit
        else:
            return "HEO"  # High Earth Orbit
