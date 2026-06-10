"""Transformer attention mechanisms for spacecraft state estimation."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from loguru import logger


class MultiHeadAttention:
    """Multi-head attention mechanism."""
    
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dropout_rate: float = 0.1,
    ):
        """Initialize multi-head attention.
        
        Args:
            dim_model: Model dimension (embedding size)
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        assert dim_model % num_heads == 0, "dim_model must be divisible by num_heads"
        
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_model // num_heads
        self.dropout_rate = dropout_rate
        
        # Initialize weights
        self.W_q = np.random.randn(dim_model, dim_model) / np.sqrt(dim_model)
        self.W_k = np.random.randn(dim_model, dim_model) / np.sqrt(dim_model)
        self.W_v = np.random.randn(dim_model, dim_model) / np.sqrt(dim_model)
        self.W_o = np.random.randn(dim_model, dim_model) / np.sqrt(dim_model)
        
        logger.info(f"Initialized MultiHeadAttention with {num_heads} heads")
    
    def scaled_dot_product_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scaled dot-product attention.
        
        Args:
            Q: Query matrix (batch_size, seq_len, dim_model)
            K: Key matrix (batch_size, seq_len, dim_model)
            V: Value matrix (batch_size, seq_len, dim_model)
            mask: Attention mask (optional)
        
        Returns:
            Tuple of (output, attention_weights)
        """
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.dim_head)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        
        # Softmax to get attention weights
        attention_weights = self._softmax(scores)
        
        # Apply dropout
        if self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout_rate)
        
        # Apply attention to values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass of multi-head attention.
        
        Args:
            X: Input tensor (batch_size, seq_len, dim_model)
        
        Returns:
            Tuple of (output, attention_dict)
        """
        batch_size, seq_len, _ = X.shape
        
        # Linear projections
        Q = np.matmul(X, self.W_q)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.dim_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.dim_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.dim_head).transpose(0, 2, 1, 3)
        
        # Compute attention for each head
        attention_outputs = []
        attention_weights_list = []
        
        for head in range(self.num_heads):
            output, weights = self.scaled_dot_product_attention(
                Q[:, head],
                K[:, head],
                V[:, head],
            )
            attention_outputs.append(output)
            attention_weights_list.append(weights)
        
        # Concatenate heads
        output = np.concatenate(attention_outputs, axis=-1)
        
        # Final linear projection
        output = np.matmul(output, self.W_o)
        
        return output, {
            'attention_weights': np.stack(attention_weights_list),
            'query': Q,
            'key': K,
            'value': V,
        }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax.
        
        Args:
            x: Input array
        
        Returns:
            Softmax output
        """
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


class CrossAttention:
    """Cross-attention for fusing different modalities."""
    
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dropout_rate: float = 0.1,
    ):
        """Initialize cross-attention.
        
        Args:
            dim_model: Model dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        self.mha = MultiHeadAttention(dim_model, num_heads, dropout_rate)
        logger.info("Initialized CrossAttention")
    
    def forward(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cross-attention forward pass.
        
        Args:
            X: Query tensor (from main modality)
            Y: Key/Value tensor (from secondary modality)
        
        Returns:
            Tuple of (output, attention_weights)
        """
        # For cross-attention, queries come from X, keys and values from Y
        batch_size, seq_len_x, dim = X.shape
        _, seq_len_y, _ = Y.shape
        
        Q = np.matmul(X, self.mha.W_q)
        K = np.matmul(Y, self.mha.W_k)
        V = np.matmul(Y, self.mha.W_v)
        
        # Reshape for heads
        Q = Q.reshape(batch_size, seq_len_x, self.mha.num_heads, self.mha.dim_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len_y, self.mha.num_heads, self.mha.dim_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len_y, self.mha.num_heads, self.mha.dim_head).transpose(0, 2, 1, 3)
        
        # Compute cross-attention
        output, weights = self.mha.scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads
        output = np.concatenate([output], axis=-1).reshape(batch_size, seq_len_x, -1)
        
        # Final projection
        output = np.matmul(output, self.mha.W_o)
        
        return output, weights
