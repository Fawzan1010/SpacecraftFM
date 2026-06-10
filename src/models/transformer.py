"""Transformer-based foundation model for spacecraft attitude estimation."""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from loguru import logger

from .attention import MultiHeadAttention, CrossAttention
from .embeddings import SensorEmbedding, TemporalEmbedding, EnvironmentEmbedding


class TransformerEncoderLayer:
    """Single transformer encoder layer."""
    
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_ff: int,
        dropout_rate: float = 0.1,
    ):
        """Initialize transformer encoder layer.
        
        Args:
            dim_model: Model dimension
            num_heads: Number of attention heads
            dim_ff: Dimension of feed-forward network
            dropout_rate: Dropout rate
        """
        self.dim_model = dim_model
        self.attention = MultiHeadAttention(dim_model, num_heads, dropout_rate)
        
        # Feed-forward network weights
        self.W1 = np.random.randn(dim_model, dim_ff) / np.sqrt(dim_model)
        self.W2 = np.random.randn(dim_ff, dim_model) / np.sqrt(dim_ff)
        self.b1 = np.zeros(dim_ff)
        self.b2 = np.zeros(dim_model)
        
        # Layer normalization parameters
        self.gamma1 = np.ones(dim_model)
        self.beta1 = np.zeros(dim_model)
        self.gamma2 = np.ones(dim_model)
        self.beta2 = np.zeros(dim_model)
        
        self.dropout_rate = dropout_rate
        logger.info("Initialized TransformerEncoderLayer")
    
    def forward(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass of encoder layer.
        
        Args:
            X: Input tensor (batch_size, seq_len, dim_model)
        
        Returns:
            Tuple of (output, attention_info)
        """
        # Multi-head self-attention
        attn_output, attn_info = self.attention.forward(X)
        
        # Apply dropout and residual connection
        attn_output = X + self._apply_dropout(attn_output)
        
        # Layer normalization
        attn_output = self._layer_norm(attn_output, self.gamma1, self.beta1)
        
        # Feed-forward network
        ff_output = np.matmul(attn_output, self.W1) + self.b1
        ff_output = self._relu(ff_output)
        ff_output = np.matmul(ff_output, self.W2) + self.b2
        
        # Apply dropout and residual connection
        ff_output = attn_output + self._apply_dropout(ff_output)
        
        # Layer normalization
        output = self._layer_norm(ff_output, self.gamma2, self.beta2)
        
        return output, attn_info
    
    def _layer_norm(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Layer normalization.
        
        Args:
            x: Input
            gamma: Scale parameter
            beta: Shift parameter
        
        Returns:
            Normalized output
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-6)
        return gamma * x_norm + beta
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _apply_dropout(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Apply dropout."""
        if self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape)
            return x * mask / (1 - self.dropout_rate)
        return x


class SpacecraftFoundationModel:
    """Foundation transformer model for spacecraft attitude estimation."""
    
    def __init__(
        self,
        dim_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_ff: int = 1024,
        max_seq_length: int = 512,
        dropout_rate: float = 0.1,
    ):
        """Initialize foundation model.
        
        Args:
            dim_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            dim_ff: Feed-forward dimension
            max_seq_length: Maximum sequence length
            dropout_rate: Dropout rate
        """
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        # Sensor embeddings
        self.quaternion_embedding = SensorEmbedding(4, dim_model, 'quaternion')
        self.gyro_embedding = SensorEmbedding(3, dim_model, 'gyroscope')
        self.accel_embedding = SensorEmbedding(3, dim_model, 'accelerometer')
        self.mag_embedding = SensorEmbedding(3, dim_model, 'magnetometer')
        
        # Temporal embedding
        self.temporal_embedding = TemporalEmbedding(max_seq_length, dim_model)
        
        # Environment embedding
        self.environment_embedding = EnvironmentEmbedding(10, dim_model)
        
        # Encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(dim_model, num_heads, dim_ff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Output projection heads
        self.quaternion_head = np.random.randn(dim_model, 4) / np.sqrt(dim_model)
        self.angular_velocity_head = np.random.randn(dim_model, 3) / np.sqrt(dim_model)
        self.uncertainty_head = np.random.randn(dim_model, 7) / np.sqrt(dim_model)
        
        logger.info(f"Initialized SpacecraftFoundationModel with {num_layers} layers")
    
    def forward(
        self,
        sensor_data: Dict[str, np.ndarray],
        environmental_context: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Forward pass of foundation model.
        
        Args:
            sensor_data: Dictionary with sensor readings
                - 'quaternion': (batch_size, seq_len, 4)
                - 'gyroscope': (batch_size, seq_len, 3)
                - 'accelerometer': (batch_size, seq_len, 3)
                - 'magnetometer': (batch_size, seq_len, 3)
            environmental_context: Environmental parameters
        
        Returns:
            Dictionary with predictions
        """
        batch_size = None
        seq_len = None
        embeddings = []
        
        # Embed sensor data
        if 'quaternion' in sensor_data:
            q_emb = self.quaternion_embedding.embed(sensor_data['quaternion'])
            embeddings.append(q_emb)
            batch_size, seq_len = q_emb.shape[0], q_emb.shape[1]
        
        if 'gyroscope' in sensor_data:
            gyro_emb = self.gyro_embedding.embed(sensor_data['gyroscope'])
            embeddings.append(gyro_emb)
        
        if 'accelerometer' in sensor_data:
            accel_emb = self.accel_embedding.embed(sensor_data['accelerometer'])
            embeddings.append(accel_emb)
        
        if 'magnetometer' in sensor_data:
            mag_emb = self.mag_embedding.embed(sensor_data['magnetometer'])
            embeddings.append(mag_emb)
        
        # Concatenate embeddings
        if embeddings:
            x = np.mean(np.stack(embeddings), axis=0)  # Average embeddings
        else:
            raise ValueError("No sensor data provided")
        
        # Add temporal embedding
        x = self.temporal_embedding.embed(x)
        
        # Add environmental context if provided
        if environmental_context is not None:
            env_emb = self.environment_embedding.embed(environmental_context)
            # Broadcast and add environment embedding
            x = x + env_emb[np.newaxis, np.newaxis, :]
        
        # Pass through encoder layers
        attention_history = []
        for layer in self.encoder_layers:
            x, attn_info = layer.forward(x)
            attention_history.append(attn_info)
        
        # Extract final hidden state (last token)
        final_hidden = x[:, -1, :]  # (batch_size, dim_model)
        
        # Output projections
        quaternion_pred = final_hidden @ self.quaternion_head
        angular_velocity_pred = final_hidden @ self.angular_velocity_head
        uncertainty_pred = np.exp(final_hidden @ self.uncertainty_head)  # Softplus
        
        # Normalize quaternion output
        quaternion_pred = quaternion_pred / (np.linalg.norm(quaternion_pred, axis=-1, keepdims=True) + 1e-8)
        
        return {
            'quaternion': quaternion_pred,
            'angular_velocity': angular_velocity_pred,
            'uncertainty': uncertainty_pred,
            'hidden_states': x,
            'attention_history': attention_history,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Model configuration and statistics
        """
        return {
            'dim_model': self.dim_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_length': self.max_seq_length,
        }
