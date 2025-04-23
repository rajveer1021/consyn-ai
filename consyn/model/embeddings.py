# consyn/model/embeddings.py
"""
Embedding implementations for Consyn AI models.
This module provides token and positional embeddings for transformer models.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ConsynConfig


class ConsynEmbeddings(nn.Module):
    """
    Embeddings for Consyn models.
    
    Combines token embeddings and position embeddings.
    """
    
    def __init__(self, config: ConsynConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=config.pad_token_id
        )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        
        # Token type embeddings (if needed)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size,
                config.hidden_size
            )
        else:
            self.token_type_embeddings = None
            
        # Embedding dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Embedding normalization
        if config.use_rms_norm:
            self.layer_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for embeddings.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            position_ids: Position indices of shape [batch_size, seq_len]
            token_type_ids: Token type IDs of shape [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Combined embeddings of shape [batch_size, seq_len, hidden_size]
        """
        seq_length = input_ids.size(1)
        
        # Get word embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        # Add position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        
        # Add token type embeddings if applicable
        if token_type_ids is not None and self.token_type_embeddings is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
            
        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings as described in 'Attention Is All You Need'.
    
    An alternative to learned position embeddings where positions are encoded
    using sine and cosine functions of different frequencies.
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        
        # Create sinusoidal position embeddings
        position = torch.arange(max_position_embeddings).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )
        
        # Calculate sin and cos components
        pe = torch.zeros(max_position_embeddings, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get positional embeddings for given positions.
        
        Args:
            positions: Position indices of shape [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Positional embeddings of shape [batch_size, seq_len, dim]
        """
        # Truncate positions to max length
        positions = positions.clamp(max=self.max_position_embeddings - 1)
        
        # Look up embeddings for each position
        return self.pe[positions]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    A variant of layer normalization that doesn't require centering
    (subtracting the mean), making it more efficient.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the input tensor.
        
        Args:
            hidden_states: Input tensor of shape [..., hidden_size]
            
        Returns:
            torch.Tensor: Normalized tensor of the same shape
        """
        # Calculate RMS
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        
        # Apply scaling
        return self.weight * hidden_states