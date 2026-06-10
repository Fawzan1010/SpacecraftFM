"""Environmental feature engineering."""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from loguru import logger


class EnvironmentFeatureExtractor:
    """Extract features from environmental data."""
    
    def __init__(self):
        """Initialize environment feature extractor."""
        pass
    
    def extract_space_weather_features(
        self,
        kp_index: float,
        dst_index: float,
        f107_flux: float,
        solar_wind_speed: float = None,
    ) -> Dict[str, Any]:
        """Extract space weather features.
        
        Args:
            kp_index: Kp geomagnetic index (0-9)
            dst_index: Dst index in nanoTesla
            f107_flux: F10.7 solar flux in SFU
            solar_wind_speed: Solar wind speed in km/s (optional)
        
        Returns:
            Space weather features
        """
        features = {}
        
        # Kp index and derived metrics
        features['kp_index'] = float(kp_index)
        features['kp_3hour'] = float(kp_index)
        
        # Geomagnetic activity level
        if kp_index < 3:
            geomag_level = "quiet"
        elif kp_index < 5:
            geomag_level = "unsettled"
        elif kp_index < 6:
            geomag_level = "active"
        elif kp_index < 7:
            geomag_level = "minor_storm"
        elif kp_index < 8:
            geomag_level = "moderate_storm"
        elif kp_index < 9:
            geomag_level = "severe_storm"
        else:
            geomag_level = "extreme_storm"
        
        features['geomagnetic_level'] = geomag_level
        features['is_geomagnetic_storm'] = 1.0 if kp_index >= 5 else 0.0
        features['is_major_storm'] = 1.0 if kp_index >= 7 else 0.0
        
        # Dst index
        features['dst_index'] = float(dst_index)
        features['dst_abs'] = float(np.abs(dst_index))
        
        # Storm intensity classification
        if dst_index > -20:
            storm_class = "quiet"
        elif dst_index > -50:
            storm_class = "weak"
        elif dst_index > -100:
            storm_class = "moderate"
        elif dst_index > -200:
            storm_class = "strong"
        else:
            storm_class = "severe"
        
        features['dst_storm_class'] = storm_class
        
        # Solar activity
        features['f107_flux'] = float(f107_flux)
        features['f107_normalized'] = float((f107_flux - 80) / 20)  # Normalize
        
        # Solar activity level
        if f107_flux < 100:
            solar_level = "low"
        elif f107_flux < 150:
            solar_level = "moderate"
        else:
            solar_level = "high"
        
        features['solar_activity_level'] = solar_level
        
        # Solar wind speed
        if solar_wind_speed is not None:
            features['solar_wind_speed_km_s'] = float(solar_wind_speed)
            
            # High speed stream detection
            if solar_wind_speed > 500:
                features['high_speed_stream'] = 1.0
            else:
                features['high_speed_stream'] = 0.0
        
        return features
    
    def extract_atmospheric_features(
        self,
        density: float,
        temperature: float,
        altitude_km: float,
    ) -> Dict[str, float]:
        """Extract atmospheric features.
        
        Args:
            density: Atmospheric density in kg/m^3
            temperature: Temperature in K
            altitude_km: Altitude in kilometers
        
        Returns:
            Atmospheric features
        """
        features = {}
        
        # Density features
        features['atmospheric_density'] = float(density)
        features['log_density'] = float(np.log10(max(density, 1e-30)))
        
        # Drag severity
        # High density = more drag
        if density > 1e-15:
            drag_severity = "extreme"
        elif density > 1e-17:
            drag_severity = "very_high"
        elif density > 1e-19:
            drag_severity = "high"
        elif density > 1e-21:
            drag_severity = "moderate"
        else:
            drag_severity = "low"
        
        features['drag_severity'] = drag_severity
        
        # Temperature features
        features['temperature_k'] = float(temperature)
        features['temperature_c'] = float(temperature - 273.15)
        
        # Atmospheric scale height estimate
        scale_height = 8500 * (temperature / 288)  # Simple model
        features['scale_height_m'] = float(scale_height)
        
        return features
    
    def extract_magnetic_environment(
        self,
        b_magnitude: float,
        inclination_deg: float,
        declination_deg: float,
    ) -> Dict[str, float]:
        """Extract magnetic environment features.
        
        Args:
            b_magnitude: Magnetic field magnitude in nanoTesla
            inclination_deg: Magnetic inclination in degrees
            declination_deg: Magnetic declination in degrees
        
        Returns:
            Magnetic environment features
        """
        features = {}
        
        features['magnetic_field_magnitude_nT'] = float(b_magnitude)
        features['magnetic_inclination_deg'] = float(inclination_deg)
        features['magnetic_declination_deg'] = float(declination_deg)
        
        # Magnetic field strength classification
        if b_magnitude < 25000:
            field_class = "weak"
        elif b_magnitude < 40000:
            field_class = "moderate"
        elif b_magnitude < 55000:
            field_class = "strong"
        else:
            field_class = "very_strong"
        
        features['magnetic_field_class'] = field_class
        
        return features
    
    def compute_environmental_risk_score(
        self,
        space_weather_features: Dict[str, Any],
        atmospheric_features: Dict[str, float],
        magnetic_features: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute overall environmental risk score.
        
        Args:
            space_weather_features: Space weather features
            atmospheric_features: Atmospheric features
            magnetic_features: Magnetic environment features
        
        Returns:
            Risk scores (0-1 scale)
        """
        risk_scores = {}
        
        # Geomagnetic storm risk (0-1)
        kp = space_weather_features.get('kp_index', 0)
        geomag_risk = min(kp / 9.0, 1.0)
        risk_scores['geomagnetic_risk'] = float(geomag_risk)
        
        # Atmospheric drag risk (0-1)
        density = atmospheric_features.get('atmospheric_density', 1e-25)
        drag_risk = 1.0 - np.exp(-density / 1e-15)
        risk_scores['atmospheric_drag_risk'] = float(drag_risk)
        
        # Solar activity risk (0-1)
        f107 = space_weather_features.get('f107_flux', 100)
        solar_risk = (f107 - 70) / 200  # Normalize
        solar_risk = np.clip(solar_risk, 0, 1)
        risk_scores['solar_activity_risk'] = float(solar_risk)
        
        # Overall risk (weighted combination)
        overall_risk = 0.4 * geomag_risk + 0.3 * drag_risk + 0.3 * solar_risk
        risk_scores['overall_environmental_risk'] = float(overall_risk)
        
        return risk_scores
