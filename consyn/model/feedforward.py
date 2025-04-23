# consyn/model/feedforward.py
"""
Feed-forward network implementations for Consyn AI models.
This module provides various MLP implementations for transformer architectures.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(activation: str) -> Callable:
    """
    Get the appropriate activation function by name.
    
    Args:
        activation: Name of the activation function
        
    Returns:
        callable: The activation function
        
    Raises:
        ValueError: If the activation function is not supported
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "gelu_new":
        # Implementation of GELU with a faster approximation
        return lambda x: 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    elif activation == "silu" or activation == "swish":
        return F.silu
    else:
        raise ValueError(f"Activation function '{activation}' not supported")


class ConsynMLP(nn.Module):
    """
    Standard MLP as used in transformer architectures.
    
    This is the feed-forward network applied after the attention layer,
    typically consisting of two linear transformations with an activation in between.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.act_fn = get_activation_fn(activation)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP.
        
        Args:
            hidden_states: Input tensor of shape [..., hidden_size]
            
        Returns:
            torch.Tensor: Output tensor of shape [..., hidden_size]
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class GatedConsynMLP(nn.Module):
    """
    Gated MLP variant for transformer architectures.
    
    This implements a gating mechanism in the feed-forward network,
    similar to the gated linear units used in some transformer variants.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.act_fn = get_activation_fn(activation)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the gated MLP.
        
        Args:
            hidden_states: Input tensor of shape [..., hidden_size]
            
        Returns:
            torch.Tensor: Output tensor of shape [..., hidden_size]
        """
        # Project to intermediate size through two separate projections
        gate = self.gate_proj(hidden_states)
        gate = self.act_fn(gate)
        
        up = self.up_proj(hidden_states)
        
        # Gate the intermediate representation
        intermediate = gate * up
        
        # Project back to hidden size
        output = self.down_proj(intermediate)
        output = self.dropout(output)
        
        return output


class GLUConsynMLP(nn.Module):
    """
    Gated Linear Unit MLP variant.
    
    This implements the Gated Linear Unit as described in
    "GLU Variants Improve Transformer" by Noam Shazeer.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        
        # Double the intermediate size for the GLU
        self.fc1 = nn.Linear(hidden_size, 2 * intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.act_fn = get_activation_fn(activation)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GLU MLP.
        
        Args:
            hidden_states: Input tensor of shape [..., hidden_size]
            
        Returns:
            torch.Tensor: Output tensor of shape [..., hidden_size]
        """
        # Split the output into two parts
        hidden_states = self.fc1(hidden_states)
        x, gate = hidden_states.chunk(2, dim=-1)
        
        # Apply activation to the gate
        gate = self.act_fn(gate)
        
        # Gate the x values
        x = x * gate
        
        # Project back to hidden size
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class GeGLUConsynMLP(nn.Module):
    """
    GeGLU (GELU Gated Linear Unit) MLP variant.
    
    This implements the GeGLU as described in
    "GLU Variants Improve Transformer" by Noam Shazeer.
    It specifically uses GELU as the activation function.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        
        # Double the intermediate size for the GeGLU
        self.fc1 = nn.Linear(hidden_size, 2 * intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GeGLU MLP.
        
        Args:
            hidden_states: Input tensor of shape [..., hidden_size]
            
        Returns:
            torch.Tensor: Output tensor of shape [..., hidden_size]
        """
        # Split the output into two parts
        hidden_states = self.fc1(hidden_states)
        x, gate = hidden_states.chunk(2, dim=-1)
        
        # Apply GELU specifically
        gate = F.gelu(gate)
        
        # Gate the x values
        x = x * gate
        
        # Project back to hidden size
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x