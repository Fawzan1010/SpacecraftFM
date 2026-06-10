"""Quaternion feature engineering."""

import numpy as np
from typing import Tuple, Dict, Any
from loguru import logger

from ..utils.physics import (
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_normalize,
    quaternion_to_euler,
    euler_to_quaternion,
    quaternion_to_dcm,
    dcm_to_quaternion,
    quaternion_error,
    quaternion_kinematics,
)


class QuaternionFeatureExtractor:
    """Extract features from quaternion data."""
    
    def __init__(self):
        """Initialize quaternion feature extractor."""
        self.reference_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    
    def extract_quaternion_features(
        self,
        quaternion: np.ndarray,
        prev_quaternion: np.ndarray = None,
        dt: float = 1.0,
    ) -> Dict[str, float]:
        """Extract quaternion features.
        
        Args:
            quaternion: Current quaternion [w, x, y, z]
            prev_quaternion: Previous quaternion for rate computation
            dt: Time step in seconds
        
        Returns:
            Dictionary of quaternion features
        """
        features = {}
        
        # Normalize quaternion
        q_norm = quaternion_normalize(quaternion)
        features['quaternion_norm'] = float(np.linalg.norm(q_norm))
        
        # Individual components
        features['q_w'] = float(q_norm[0])
        features['q_x'] = float(q_norm[1])
        features['q_y'] = float(q_norm[2])
        features['q_z'] = float(q_norm[3])
        
        # Quaternion magnitude deviations
        features['q_norm_error'] = float(np.abs(features['quaternion_norm'] - 1.0))
        features['q_max_component'] = float(np.max(np.abs(q_norm)))
        features['q_min_component'] = float(np.min(np.abs(q_norm)))
        
        # Euler angles
        euler = quaternion_to_euler(q_norm)
        features['euler_roll_rad'] = float(euler[0])
        features['euler_pitch_rad'] = float(euler[1])
        features['euler_yaw_rad'] = float(euler[2])
        
        # Euler angles in degrees
        features['euler_roll_deg'] = float(np.degrees(euler[0]))
        features['euler_pitch_deg'] = float(np.degrees(euler[1]))
        features['euler_yaw_deg'] = float(np.degrees(euler[2]))
        
        # Quaternion rate (if previous quaternion provided)
        if prev_quaternion is not None:
            q_rate = (q_norm - quaternion_normalize(prev_quaternion)) / dt
            features['q_rate_w'] = float(q_rate[0])
            features['q_rate_x'] = float(q_rate[1])
            features['q_rate_y'] = float(q_rate[2])
            features['q_rate_z'] = float(q_rate[3])
            features['q_rate_magnitude'] = float(np.linalg.norm(q_rate))
        
        # Quaternion error relative to reference
        q_error = quaternion_error(self.reference_quaternion, q_norm)
        features['quaternion_geodesic_error_rad'] = float(q_error)
        features['quaternion_geodesic_error_deg'] = float(np.degrees(q_error))
        
        # Scalar-vector split
        features['q_scalar'] = float(q_norm[0])
        features['q_vector_magnitude'] = float(np.linalg.norm(q_norm[1:]))
        
        return features
    
    def extract_attitude_error(
        self,
        q_true: np.ndarray,
        q_estimated: np.ndarray,
    ) -> Dict[str, float]:
        """Extract attitude error features.
        
        Args:
            q_true: True quaternion
            q_estimated: Estimated quaternion
        
        Returns:
            Attitude error features
        """
        q_true = quaternion_normalize(q_true)
        q_estimated = quaternion_normalize(q_estimated)
        
        # Attitude error quaternion
        q_error_quat = quaternion_multiply(
            q_estimated,
            quaternion_conjugate(q_true)
        )
        
        # Extract error angle and axis
        error_angle = 2.0 * np.arccos(np.clip(q_error_quat[0], -1.0, 1.0))
        
        if np.sin(error_angle / 2.0) > 1e-6:
            error_axis = q_error_quat[1:] / np.sin(error_angle / 2.0)
        else:
            error_axis = np.array([0.0, 0.0, 1.0])
        
        return {
            'attitude_error_angle_rad': float(error_angle),
            'attitude_error_angle_deg': float(np.degrees(error_angle)),
            'attitude_error_axis_x': float(error_axis[0]),
            'attitude_error_axis_y': float(error_axis[1]),
            'attitude_error_axis_z': float(error_axis[2]),
        }
    
    def extract_dcm_features(
        self,
        quaternion: np.ndarray,
    ) -> Dict[str, float]:
        """Extract DCM-based features.
        
        Args:
            quaternion: Input quaternion
        
        Returns:
            DCM-based features
        """
        dcm = quaternion_to_dcm(quaternion_normalize(quaternion))
        
        features = {}
        
        # DCM properties
        features['dcm_determinant'] = float(np.linalg.det(dcm))
        features['dcm_trace'] = float(np.trace(dcm))
        
        # Orthogonality check: DCM @ DCM.T should be identity
        orthogonality = dcm @ dcm.T
        features['dcm_orthogonality_error'] = float(np.linalg.norm(orthogonality - np.eye(3)))
        
        # Eigenvalues (should be 1, e^(i*theta), e^(-i*theta))
        eigenvalues = np.linalg.eigvals(dcm)
        features['dcm_eigenvalue_magnitude'] = float(np.mean(np.abs(eigenvalues)))
        
        # DCM elements
        for i in range(3):
            for j in range(3):
                features[f'dcm_{i}_{j}'] = float(dcm[i, j])
        
        return features
