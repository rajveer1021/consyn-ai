# consyn/model/normalization.py
"""
Normalization layer implementations for Consyn AI models.
This module provides various normalization techniques used in transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Layer Normalization as described in "Layer Normalization" by Ba et al.
    
    This is a wrapper around nn.LayerNorm for consistency with other normalization layers.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to the input tensor.
        
        Args:
            x: Input tensor of shape [..., hidden_size]
            
        Returns:
            torch.Tensor: Normalized tensor of the same shape
        """
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, unbiased=False, keepdim=True)
        
        # Normalize
        x = (x - mean) / torch.sqrt(variance + self.variance_epsilon)
        
        # Apply scaling and shifting
        return x * self.weight + self.bias


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


class ScaleNorm(nn.Module):
    """
    Scale Normalization.
    
    A simpler variant of layer normalization that only requires
    a single learned scalar parameter.
    """
    
    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        
        self.scale = nn.Parameter(torch.ones(1))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply scale normalization to the input tensor.
        
        Args:
            x: Input tensor of shape [..., dim]
            
        Returns:
            torch.Tensor: Normalized tensor of the same shape
        """
        # Calculate the norm
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        
        # Normalize by the norm
        x = x / (norm + self.eps)
        
        # Apply the learned scale
        return self.scale * x


class GroupNorm(nn.Module):
    """
    Group Normalization for transformers.
    
    This applies group normalization to the hidden states,
    treating heads as groups.
    """
    
    def __init__(self, hidden_size: int, num_groups: int, eps: float = 1e-12):
        super().__init__()
        
        if hidden_size % num_groups != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_groups {num_groups}")
            
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.num_groups = num_groups
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply group normalization to the input tensor.
        
        Args:
            x: Input tensor of shape [..., hidden_size]
            
        Returns:
            torch.Tensor: Normalized tensor of the same shape
        """
        # Save original shape
        original_shape = x.shape
        
        # Reshape for group norm
        x = x.view(*x.shape[:-1], self.num_groups, -1)
        
        # Calculate mean and variance per group
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        x = x.view(*original_shape)
        
        # Apply scaling and shifting
        return x * self.weight + self.bias


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization.
    
    This is a layer normalization variant where the scaling and shifting
    parameters are predicted from an input rather than being fixed.
    Useful for conditional computations.
    """
    
    def __init__(self, hidden_size: int, condition_size: int, eps: float = 1e-12):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.eps = eps
        
        # Projection to predict scale and shift parameters
        self.projector = nn.Linear(condition_size, 2 * hidden_size)
        
        # Initialize to be close to standard layer norm
        self.projector.weight.data.zero_()
        self.projector.bias.data[: hidden_size] = 1.0  # Initialize scale params to 1
        self.projector.bias.data[hidden_size:] = 0.0  # Initialize shift params to 0
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization to the input tensor.
        
        Args:
            x: Input tensor of shape [..., hidden_size]
            condition: Conditioning tensor of shape [..., condition_size]
            
        Returns:
            torch.Tensor: Normalized tensor of the same shape as x
        """
        # Calculate mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Predict scale and shift parameters
        scale_shift = self.projector(condition)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Apply adaptive scaling and shifting
        return x_norm * scale + shift