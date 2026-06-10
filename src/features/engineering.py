"""Feature engineering utilities."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from loguru import logger

from .quaternion import QuaternionFeatureExtractor
from .dynamics import DynamicsFeatureExtractor
from .orbit import OrbitFeatureExtractor
from .environment import EnvironmentFeatureExtractor


class FeatureEngineer:
    """Main feature engineering orchestrator."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.quaternion_extractor = QuaternionFeatureExtractor()
        self.dynamics_extractor = DynamicsFeatureExtractor()
        self.orbit_extractor = OrbitFeatureExtractor()
        self.environment_extractor = EnvironmentFeatureExtractor()
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        feature_groups: List[str] = None,
    ) -> pd.DataFrame:
        """Engineer features from raw data.
        
        Args:
            data: Input DataFrame with raw data
            feature_groups: List of feature groups to compute
        
        Returns:
            DataFrame with engineered features
        """
        if feature_groups is None:
            feature_groups = ['quaternion', 'dynamics', 'orbit', 'environment']
        
        logger.info(f"Engineering {len(feature_groups)} feature groups")
        
        features_list = []
        
        for idx, row in data.iterrows():
            row_features = {}
            
            # Quaternion features
            if 'quaternion' in feature_groups and all(f in row for f in ['q_w', 'q_x', 'q_y', 'q_z']):
                q = np.array([row['q_w'], row['q_x'], row['q_y'], row['q_z']])
                q_features = self.quaternion_extractor.extract_quaternion_features(q)
                row_features.update(q_features)
            
            # Dynamics features
            if 'dynamics' in feature_groups and all(f in row for f in ['omega_x', 'omega_y', 'omega_z']):
                omega = np.array([row['omega_x'], row['omega_y'], row['omega_z']])
                dyn_features = self.dynamics_extractor.extract_angular_velocity_features(omega)
                row_features.update(dyn_features)
            
            # Orbit features
            if 'orbit' in feature_groups and all(f in row for f in ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']):
                pos = np.array([row['pos_x'], row['pos_y'], row['pos_z']])
                vel = np.array([row['vel_x'], row['vel_y'], row['vel_z']])
                orbit_features = self.orbit_extractor.extract_orbit_features(pos, vel)
                row_features.update(orbit_features)
            
            # Environment features
            if 'environment' in feature_groups and all(f in row for f in ['kp_index', 'dst_index', 'f107_flux']):
                env_features = self.environment_extractor.extract_space_weather_features(
                    kp_index=row['kp_index'],
                    dst_index=row['dst_index'],
                    f107_flux=row['f107_flux'],
                )
                row_features.update(env_features)
            
            features_list.append(row_features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Engineered {len(features_df.columns)} features")
        
        return pd.concat([data, features_df], axis=1)
