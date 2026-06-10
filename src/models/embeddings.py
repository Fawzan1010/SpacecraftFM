"""Embedding layers for spacecraft sensor data."""

import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger


class SensorEmbedding:
    """Embedding layer for sensor data."""
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        sensor_type: str = 'quaternion',
    ):
        """Initialize sensor embedding.
        
        Args:
            input_dim: Input dimension
            embedding_dim: Embedding dimension
            sensor_type: Type of sensor ('quaternion', 'accelerometer', 'magnetometer', etc.)
        """
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.sensor_type = sensor_type
        
        # Embedding matrix
        self.embedding_matrix = np.random.randn(input_dim, embedding_dim) / np.sqrt(input_dim)
        
        # Sensor-specific bias
        self.bias = np.zeros(embedding_dim)
        
        # Learnable scaling
        self.scale = np.ones(embedding_dim)
        
        logger.info(f"Initialized SensorEmbedding for {sensor_type}")
    
    def embed(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Embed sensor data.
        
        Args:
            x: Input data (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        
        Returns:
            Embedded data (batch_size, seq_len, embedding_dim) or (batch_size, embedding_dim)
        """
        # Normalize input based on sensor type
        x_normalized = self._normalize_by_sensor(x)
        
        # Apply embedding
        if x_normalized.ndim == 2:
            # (batch_size, input_dim) -> (batch_size, embedding_dim)
            embedded = np.matmul(x_normalized, self.embedding_matrix)
        elif x_normalized.ndim == 3:
            # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, embedding_dim)
            batch_size, seq_len, _ = x_normalized.shape
            embedded = np.matmul(x_normalized, self.embedding_matrix)
        else:
            raise ValueError(f"Unexpected input dimension: {x_normalized.ndim}")
        
        # Add bias and scale
        embedded = embedded + self.bias
        embedded = embedded * self.scale
        
        return embedded
    
    def _normalize_by_sensor(self, x: np.ndarray) -> np.ndarray:
        """Normalize data based on sensor type.
        
        Args:
            x: Input data
        
        Returns:
            Normalized data
        """
        if self.sensor_type == 'quaternion':
            # Normalize to unit quaternion
            norm = np.linalg.norm(x, axis=-1, keepdims=True)
            return x / (norm + 1e-8)
        elif self.sensor_type in ['accelerometer', 'gyroscope', 'magnetometer']:
            # Standardize to zero mean, unit variance
            mean = np.mean(x, axis=-1, keepdims=True)
            std = np.std(x, axis=-1, keepdims=True)
            return (x - mean) / (std + 1e-8)
        else:
            # Default: no normalization
            return x


class TemporalEmbedding:
    """Temporal/positional embedding for sequences."""
    
    def __init__(
        self,
        max_seq_length: int,
        embedding_dim: int,
        frequency: float = 10000.0,
    ):
        """Initialize temporal embedding.
        
        Args:
            max_seq_length: Maximum sequence length
            embedding_dim: Embedding dimension
            frequency: Frequency for sinusoidal encoding
        """
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.frequency = frequency
        
        # Pre-compute positional embeddings
        self.pe = self._compute_positional_encoding()
        
        logger.info(f"Initialized TemporalEmbedding for sequences up to {max_seq_length}")
    
    def _compute_positional_encoding(self) -> np.ndarray:
        """Compute sinusoidal positional encodings.
        
        Returns:
            Positional encoding matrix (max_seq_length, embedding_dim)
        """
        pe = np.zeros((self.max_seq_length, self.embedding_dim))
        
        position = np.arange(0, self.max_seq_length).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.embedding_dim, 2) * 
                          -(np.log(self.frequency) / self.embedding_dim))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def embed(
        self,
        x: np.ndarray,
        start_idx: int = 0,
    ) -> np.ndarray:
        """Add temporal embedding to data.
        
        Args:
            x: Input data (batch_size, seq_len, embedding_dim)
            start_idx: Starting index in sequence
        
        Returns:
            Data with temporal embedding added
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract relevant positional encodings
        positions = np.arange(start_idx, start_idx + seq_len)
        pe = self.pe[positions]
        
        # Add to input
        return x + pe


class EnvironmentEmbedding:
    """Embedding for environmental context (space weather, atmospheric, etc.)."""
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
    ):
        """Initialize environment embedding.
        
        Args:
            input_dim: Number of environmental parameters
            embedding_dim: Embedding dimension
        """
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Embedding matrix
        self.embedding_matrix = np.random.randn(input_dim, embedding_dim) / np.sqrt(input_dim)
        
        # Parameter-specific scaling
        self.param_scale = np.ones(input_dim)
        
        logger.info("Initialized EnvironmentEmbedding")
    
    def embed(
        self,
        env_params: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Embed environmental parameters.
        
        Args:
            env_params: Dictionary of environmental parameters
        
        Returns:
            Embedded environment vector (embedding_dim,)
        """
        # Concatenate parameters in order
        param_vector = self._dict_to_vector(env_params)
        
        # Apply scaling
        param_vector = param_vector * self.param_scale
        
        # Apply embedding
        embedded = np.matmul(param_vector, self.embedding_matrix)
        
        return embedded
    
    def _dict_to_vector(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert parameter dictionary to vector.
        
        Args:
            params: Parameter dictionary
        
        Returns:
            Parameter vector
        """
        # Extract in consistent order
        keys = sorted(params.keys())
        values = [params[k].flatten() if isinstance(params[k], np.ndarray) else np.array([params[k]]) 
                  for k in keys]
        
        vector = np.concatenate(values)
        
        # Pad or truncate to input_dim
        if len(vector) < self.input_dim:
            vector = np.pad(vector, (0, self.input_dim - len(vector)))
        else:
            vector = vector[:self.input_dim]
        
        return vector
