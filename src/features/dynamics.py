"""Attitude dynamics feature engineering."""

import numpy as np
from typing import Dict, Any, Optional
from loguru import logger


class DynamicsFeatureExtractor:
    """Extract features from attitude dynamics data."""
    
    def __init__(self):
        """Initialize dynamics feature extractor."""
        pass
    
    def extract_angular_velocity_features(
        self,
        angular_velocity: np.ndarray,
        prev_angular_velocity: np.ndarray = None,
        dt: float = 1.0,
    ) -> Dict[str, float]:
        """Extract angular velocity features.
        
        Args:
            angular_velocity: Angular velocity [wx, wy, wz] in rad/s
            prev_angular_velocity: Previous angular velocity
            dt: Time step in seconds
        
        Returns:
            Angular velocity features
        """
        features = {}
        
        # Individual components
        features['omega_x'] = float(angular_velocity[0])
        features['omega_y'] = float(angular_velocity[1])
        features['omega_z'] = float(angular_velocity[2])
        
        # Magnitude
        omega_mag = np.linalg.norm(angular_velocity)
        features['omega_magnitude'] = float(omega_mag)
        
        # Component magnitudes
        features['omega_x_abs'] = float(np.abs(angular_velocity[0]))
        features['omega_y_abs'] = float(np.abs(angular_velocity[1]))
        features['omega_z_abs'] = float(np.abs(angular_velocity[2]))
        
        # Statistics
        features['omega_mean'] = float(np.mean(np.abs(angular_velocity)))
        features['omega_std'] = float(np.std(angular_velocity))
        features['omega_max'] = float(np.max(np.abs(angular_velocity)))
        features['omega_min'] = float(np.min(np.abs(angular_velocity)))
        
        # Dominant axis
        dominant_axis = np.argmax(np.abs(angular_velocity))
        features['dominant_rotation_axis'] = float(dominant_axis)
        
        # Angular acceleration (if previous velocity available)
        if prev_angular_velocity is not None:
            alpha = (angular_velocity - prev_angular_velocity) / dt
            features['alpha_x'] = float(alpha[0])
            features['alpha_y'] = float(alpha[1])
            features['alpha_z'] = float(alpha[2])
            features['alpha_magnitude'] = float(np.linalg.norm(alpha))
        
        return features
    
    def extract_angular_acceleration_features(
        self,
        angular_acceleration: np.ndarray,
    ) -> Dict[str, float]:
        """Extract angular acceleration features.
        
        Args:
            angular_acceleration: Angular acceleration [ax, ay, az] in rad/s^2
        
        Returns:
            Angular acceleration features
        """
        features = {}
        
        # Individual components
        features['alpha_x'] = float(angular_acceleration[0])
        features['alpha_y'] = float(angular_acceleration[1])
        features['alpha_z'] = float(angular_acceleration[2])
        
        # Magnitude
        alpha_mag = np.linalg.norm(angular_acceleration)
        features['alpha_magnitude'] = float(alpha_mag)
        
        # Statistics
        features['alpha_mean'] = float(np.mean(np.abs(angular_acceleration)))
        features['alpha_std'] = float(np.std(angular_acceleration))
        features['alpha_max'] = float(np.max(np.abs(angular_acceleration)))
        
        # Direction
        if alpha_mag > 1e-6:
            direction = angular_acceleration / alpha_mag
            features['alpha_direction_x'] = float(direction[0])
            features['alpha_direction_y'] = float(direction[1])
            features['alpha_direction_z'] = float(direction[2])
        
        return features
    
    def estimate_torques(
        self,
        inertia_tensor: np.ndarray,
        angular_velocity: np.ndarray,
        angular_acceleration: np.ndarray,
    ) -> Dict[str, Any]:
        """Estimate torques using Euler's equation.
        
        Args:
            inertia_tensor: 3x3 inertia matrix
            angular_velocity: Angular velocity vector
            angular_acceleration: Angular acceleration vector
        
        Returns:
            Torque estimation results
        """
        # Euler's equation: tau = I * alpha + omega x (I * omega)
        
        # Gyroscopic torque term
        I_omega = inertia_tensor @ angular_velocity
        gyro_torque = np.cross(angular_velocity, I_omega)
        
        # Angular acceleration torque term
        inertial_torque = inertia_tensor @ angular_acceleration
        
        # Total torque
        total_torque = inertial_torque + gyro_torque
        
        return {
            'inertial_torque_x': float(inertial_torque[0]),
            'inertial_torque_y': float(inertial_torque[1]),
            'inertial_torque_z': float(inertial_torque[2]),
            'inertial_torque_magnitude': float(np.linalg.norm(inertial_torque)),
            'gyroscopic_torque_x': float(gyro_torque[0]),
            'gyroscopic_torque_y': float(gyro_torque[1]),
            'gyroscopic_torque_z': float(gyro_torque[2]),
            'gyroscopic_torque_magnitude': float(np.linalg.norm(gyro_torque)),
            'total_torque_x': float(total_torque[0]),
            'total_torque_y': float(total_torque[1]),
            'total_torque_z': float(total_torque[2]),
            'total_torque_magnitude': float(np.linalg.norm(total_torque)),
        }
    
    def extract_rotational_energy_features(
        self,
        inertia_tensor: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> Dict[str, float]:
        """Extract rotational energy features.
        
        Args:
            inertia_tensor: 3x3 inertia matrix
            angular_velocity: Angular velocity vector
        
        Returns:
            Energy-related features
        """
        # Rotational kinetic energy: E = 0.5 * omega^T * I * omega
        I_omega = inertia_tensor @ angular_velocity
        kinetic_energy = 0.5 * np.dot(angular_velocity, I_omega)
        
        # Angular momentum: L = I * omega
        angular_momentum = I_omega
        L_mag = np.linalg.norm(angular_momentum)
        
        return {
            'rotational_kinetic_energy': float(kinetic_energy),
            'angular_momentum_x': float(angular_momentum[0]),
            'angular_momentum_y': float(angular_momentum[1]),
            'angular_momentum_z': float(angular_momentum[2]),
            'angular_momentum_magnitude': float(L_mag),
        }
