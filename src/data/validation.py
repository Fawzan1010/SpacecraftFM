"""Data validation and quality checks."""

from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd


class DataValidator:
    """Validate data quality and completeness."""
    
    def __init__(self):
        """Initialize data validator."""
        self.validation_results = {}
    
    def validate_quaternion(
        self,
        quaternion: np.ndarray,
    ) -> Dict[str, Any]:
        """Validate quaternion data.
        
        Args:
            quaternion: Quaternion array [w, x, y, z]
        
        Returns:
            Validation result dictionary
        """
        results = {
            'is_valid': True,
            'issues': [],
            'norm': float(np.linalg.norm(quaternion)),
        }
        
        # Check norm
        norm = np.linalg.norm(quaternion)
        if np.abs(norm - 1.0) > 0.01:  # Allow 1% tolerance
            results['issues'].append(f"Quaternion norm {norm:.4f} deviates from 1.0")
            results['is_valid'] = False
        
        # Check for NaN or Inf
        if np.any(np.isnan(quaternion)) or np.any(np.isinf(quaternion)):
            results['issues'].append("Quaternion contains NaN or Inf values")
            results['is_valid'] = False
        
        return results
    
    def validate_angular_velocity(
        self,
        angular_velocity: np.ndarray,
        max_rate: float = 5.0,  # deg/s
    ) -> Dict[str, Any]:
        """Validate angular velocity data.
        
        Args:
            angular_velocity: Angular velocity vector [wx, wy, wz] in deg/s
            max_rate: Maximum acceptable rate in deg/s
        
        Returns:
            Validation result dictionary
        """
        results = {
            'is_valid': True,
            'issues': [],
            'magnitude': float(np.linalg.norm(angular_velocity)),
        }
        
        # Check magnitude
        magnitude = np.linalg.norm(angular_velocity)
        if magnitude > max_rate:
            results['issues'].append(f"Angular velocity magnitude {magnitude:.2f} deg/s exceeds limit {max_rate}")
            results['is_valid'] = False
        
        # Check for NaN or Inf
        if np.any(np.isnan(angular_velocity)) or np.any(np.isinf(angular_velocity)):
            results['issues'].append("Angular velocity contains NaN or Inf values")
            results['is_valid'] = False
        
        return results
    
    def validate_orbit_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        altitude_km: float,
    ) -> Dict[str, Any]:
        """Validate orbit state data.
        
        Args:
            position: Position vector in km
            velocity: Velocity vector in km/s
            altitude_km: Altitude in km
        
        Returns:
            Validation result dictionary
        """
        results = {
            'is_valid': True,
            'issues': [],
        }
        
        # Check altitude range
        if altitude_km < 0:
            results['issues'].append(f"Altitude {altitude_km} km is below surface")
            results['is_valid'] = False
        elif altitude_km > 2000:
            results['issues'].append(f"Altitude {altitude_km} km seems unusually high")
        
        # Check for NaN or Inf
        if np.any(np.isnan(position)) or np.any(np.isinf(position)):
            results['issues'].append("Position contains NaN or Inf values")
            results['is_valid'] = False
        
        if np.any(np.isnan(velocity)) or np.any(np.isinf(velocity)):
            results['issues'].append("Velocity contains NaN or Inf values")
            results['is_valid'] = False
        
        return results
    
    def validate_timeseries(
        self,
        data: pd.DataFrame,
        expected_columns: List[str],
        time_column: str = 'timestamp',
    ) -> Dict[str, Any]:
        """Validate time series data completeness.
        
        Args:
            data: DataFrame with time series data
            expected_columns: List of expected column names
            time_column: Name of timestamp column
        
        Returns:
            Validation result dictionary
        """
        results = {
            'is_valid': True,
            'issues': [],
            'total_records': len(data),
            'missing_records': 0,
            'nan_counts': {},
        }
        
        # Check for missing columns
        missing_cols = set(expected_columns) - set(data.columns)
        if missing_cols:
            results['issues'].append(f"Missing columns: {missing_cols}")
            results['is_valid'] = False
        
        # Check for NaN values
        for col in expected_columns:
            if col in data.columns:
                nan_count = data[col].isna().sum()
                results['nan_counts'][col] = int(nan_count)
                if nan_count > 0:
                    results['issues'].append(f"Column '{col}' has {nan_count} NaN values")
        
        # Check for duplicate timestamps
        if time_column in data.columns:
            duplicates = data[time_column].duplicated().sum()
            if duplicates > 0:
                results['issues'].append(f"Found {duplicates} duplicate timestamps")
        
        return results
    
    def validate_data_file(
        self,
        filepath: Path,
    ) -> Dict[str, Any]:
        """Validate a data file.
        
        Args:
            filepath: Path to data file
        
        Returns:
            Validation result dictionary
        """
        results = {
            'filepath': str(filepath),
            'exists': filepath.exists(),
            'size_bytes': filepath.stat().st_size if filepath.exists() else 0,
            'is_valid': True,
            'issues': [],
        }
        
        if not filepath.exists():
            results['issues'].append(f"File does not exist: {filepath}")
            results['is_valid'] = False
            return results
        
        # Check file size
        if filepath.stat().st_size == 0:
            results['issues'].append("File is empty")
            results['is_valid'] = False
        
        # Try to read file based on extension
        try:
            if filepath.suffix == '.csv':
                df = pd.read_csv(filepath)
                results['records'] = len(df)
                results['columns'] = list(df.columns)
            elif filepath.suffix in ['.nc', '.h5']:
                results['readable'] = True
        except Exception as e:
            results['issues'].append(f"Error reading file: {e}")
            results['is_valid'] = False
        
        return results
