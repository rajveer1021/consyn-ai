# consyn/model/attention.py
"""
Attention mechanisms for Consyn AI models.
This module provides various attention implementations including standard, rotary, and sparse attention.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention implementation.
    
    This is the classic attention mechanism used in transformer models,
    where each head attends to different parts of the input sequence.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Check that the dimensions are compatible
        if self.head_dim * num_heads != hidden_size:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
            
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout_prob)
        
        # Scaling factor for the dot product attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim).
        
        Args:
            tensor: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: Reshaped tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = tensor.size()
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        return tensor.transpose(1, 2)
        
    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Merge the head dimensions back into hidden_size.
        
        Args:
            tensor: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            torch.Tensor: Reshaped tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, _, seq_len, _ = tensor.size()
        
        # Transpose back to [batch_size, seq_len, num_heads, head_dim]
        tensor = tensor.transpose(1, 2)
        
        # Reshape to [batch_size, seq_len, hidden_size]
        return tensor.reshape(batch_size, seq_len, self.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple:
        """
        Forward pass for multi-head attention.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask to avoid attending to padding tokens
            position_ids: Position indices for positional embeddings (not used in standard attention)
            past_key_values: Cached key/value states for incremental decoding
            use_cache: Whether to use cached key/value states
            output_attentions: Whether to return attention weights
            
        Returns:
            tuple: (context_layer, (key, value), attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project inputs to queries, keys, and values
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Handle cached key/values for incremental decoding
        if past_key_values is not None:
            past_key, past_value = past_key_values
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)
            
        # Cache current key/values if needed
        if use_cache:
            current_key_value = (key, value)
        else:
            current_key_value = None
            
        # Split heads
        query = self._split_heads(query)  # [batch, num_heads, seq_len, head_dim]
        key = self._split_heads(key)      # [batch, num_heads, key_len, head_dim]
        value = self._split_heads(value)  # [batch, num_heads, key_len, head_dim]
        
        # Get sequence length of keys (which may differ from queries in incremental decoding)
        key_len = key.size(2)
        
        # Compute scaled dot-product attention
        # [batch, num_heads, seq_len, key_len]
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Normalize the attention scores
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context_layer = torch.matmul(attention_weights, value)  # [batch, num_heads, seq_len, head_dim]
        
        # Merge heads back
        context_layer = self._merge_heads(context_layer)  # [batch, seq_len, hidden_size]
        
        # Apply output projection
        output = self.out_proj(context_layer)
        
        outputs = (output, current_key_value)
        if output_attentions:
            outputs += (attention_weights,)
            
        return outputs


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
    This applies a rotation to query and key tensors based on their positions,
    which effectively encodes position information in the attention mechanism.
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        
        # Generate the rotation matrix frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def _get_rotary_embeddings(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal position embeddings for the given position IDs.
        
        Args:
            position_ids: Position indices of shape [batch_size, seq_len]
            
        Returns:
            tuple: (cos_pos, sin_pos) for the rotary embeddings
        """
        # Generate the sinusoidal positions
        # [seq_len, dim//2]
        sinusoidal_pos = torch.einsum("i,j->ij", position_ids.float(), self.inv_freq)
        
        # Apply sin and cos to alternate dimensions
        # [seq_len, dim]
        sin_pos = torch.cat([torch.sin(sinusoidal_pos), torch.sin(sinusoidal_pos)], dim=-1)
        cos_pos = torch.cat([torch.cos(sinusoidal_pos), torch.cos(sinusoidal_pos)], dim=-1)
        
        return cos_pos, sin_pos
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half of the dimensions of the input tensor.
        
        Args:
            x: Input tensor of shape [..., dim]
            
        Returns:
            torch.Tensor: Rotated tensor
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
        
    def _apply_rotary_embeddings(
        self, 
        tensor: torch.Tensor, 
        cos_pos: torch.Tensor, 
        sin_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rotary embeddings to the input tensor.
        
        Args:
            tensor: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            cos_pos: Cosine of the position embeddings
            sin_pos: Sine of the position embeddings
            
        Returns:
            torch.Tensor: Tensor with rotary embeddings applied
        """
        # Apply the rotation using the rotation matrix
        return tensor * cos_pos + self._rotate_half(tensor) * sin_pos
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.
        
        Args:
            query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor of shape [batch_size, num_heads, key_len, head_dim]
            position_ids: Position indices of shape [batch_size, seq_len]
            
        Returns:
            tuple: (rotated_query, rotated_key)
        """
        # Get the position embeddings
        cos_pos, sin_pos = self._get_rotary_embeddings(position_ids)
        
        # Reshape for broadcasting
        # [1, 1, seq_len, head_dim]
        cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)
        sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)
        
        # Apply the rotary embeddings to query and key
        query_rot = self._apply_rotary_embeddings(query, cos_pos, sin_pos)
        
        # For keys, we may need to handle different sequence lengths in incremental decoding
        key_position_ids = position_ids[:, :key.size(2)]
        key_cos_pos, key_sin_pos = self._get_rotary_embeddings(key_position_ids)
        key_cos_pos = key_cos_pos.unsqueeze(0).unsqueeze(0)
        key_sin_pos = key_sin_pos.unsqueeze(0).unsqueeze(0)
        key_rot = self._apply_rotary_embeddings(key, key_cos_pos, key_sin_pos)
        
        return query_rot, key_rot


class RotaryAttention(nn.Module):
    """
    Multi-head attention with rotary position embeddings.
    
    This attention mechanism incorporates position information through the
    rotary embeddings rather than adding positional encodings to the input.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotary_dim: int,
        dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_dim = min(rotary_dim, self.head_dim)  # Dimension to apply rotary embeddings
        
        # Check that the dimensions are compatible
        if self.head_dim * num_heads != hidden_size:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
            
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout_prob)
        
        # Scaling factor for the dot product attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Rotary position embeddings
        self.rotary_emb = RotaryPositionEmbedding(
            dim=self.rotary_dim,
            max_position_embeddings=max_position_embeddings,
        )
        
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim).
        
        Args:
            tensor: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: Reshaped tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = tensor.size()
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        return tensor.transpose(1, 2)
        
    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Merge the head dimensions back into hidden_size.
        
        Args:
            tensor: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            torch.Tensor: Reshaped tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, _, seq_len, _ = tensor.size()
        
        # Transpose back to [batch_size, seq_len, num_heads, head_dim]
        tensor = tensor.transpose(1, 2)
        
        # Reshape to [batch_size, seq_len, hidden_size]
        return tensor.reshape(batch_size, seq_len, self.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple:
        """
        Forward pass for rotary attention.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask to avoid attending to padding tokens
            position_ids: Position indices for rotary embeddings
            past_key_values: Cached key/value states for incremental decoding
            use_cache: Whether to use cached key/value states
            output_attentions: Whether to return attention weights
            
        Returns:
            tuple: (context_layer, (key, value), attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Default position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
        # Project inputs to queries, keys, and values
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Handle cached key/values for incremental decoding
        if past_key_values is not None:
            past_key, past_value = past_key_values
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)
            
        # Cache current key/values if needed
        if use_cache:
            current_key_value = (key, value)
        else:
            current_key_value = None
            
        # Split heads
        query = self._split_heads(query)  # [batch, num_heads, seq_len, head_dim]
        key = self._split_heads(key)      # [batch, num_heads, key_len, head_dim]
        value = self._split_heads(value)  # [batch, num_heads, key_len, head_dim]
        
        # Apply rotary position embeddings
        query, key = self.rotary_emb(query, key, position_ids)
        
        # Get sequence length of keys (which may differ from queries in incremental decoding)
        key_len = key.size(2)
        
        # Compute scaled dot-product attention
        # [batch, num_heads, seq_len, key_len]
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Normalize the attention scores
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context_layer = torch.matmul(attention_weights, value)  # [batch, num_heads, seq_len, head_dim]
        
        # Merge heads back
        context_layer = self._merge_heads(context_layer)  # [batch, seq_len, hidden_size]
        
        # Apply output projection
        output = self.out_proj(context_layer)
        
        outputs = (output, current_key_value)
        if output_attentions:
            outputs += (attention_weights,)
            
        return outputs


class SparseAttention(nn.Module):
    """
    Implementation of sparse attention which only attends to nearby positions.
    
    This attention mechanism reduces computation by having each position
    only attend to a local window of positions rather than the entire sequence.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_prob: float = 0.1,
        window_size: int = 256,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        
        # Check that the dimensions are compatible
        if self.head_dim * num_heads != hidden_size:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
            
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout_prob)
        
        # Scaling factor for the dot product attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim).
        
        Args:
            tensor: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: Reshaped tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = tensor.size()
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        return tensor.transpose(1, 2)
        
    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Merge the head dimensions back into hidden_size.
        
        Args:
            tensor: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            torch.Tensor: Reshaped tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, _, seq_len, _ = tensor.size()
        
        # Transpose back to [batch_size, seq_len, num_heads, head_dim]
        tensor = tensor.transpose(1, 2)
        
        # Reshape to [batch_size, seq_len, hidden_size]
        return tensor.reshape(batch_size, seq_len, self.hidden_size)
        
    def _create_local_attention_mask(
        self, 
        seq_len: int, 
        window_size: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Create a mask for local attention, where each position only attends to
        a window of nearby positions.
        
        Args:
            seq_len: Length of the sequence
            window_size: Size of the local attention window
            device: Device to create the mask on
            
        Returns:
            torch.Tensor: Local attention mask of shape [1, 1, seq_len, seq_len]
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Calculate position differences
        pos_diff = positions.unsqueeze(-1) - positions.unsqueeze(-2)
        
        # Create mask where True means positions to mask out (positions outside the window)
        mask = torch.abs(pos_diff) > window_size // 2
        
        # Convert to attention mask format (0 for positions to attend to, large negative values for masked positions)
        attention_mask = mask.float() * -10000.0
        
        # Add batch and head dimensions
        return attention_mask.unsqueeze(0).unsqueeze(0)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple:
        """
        Forward pass for sparse attention.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask to avoid attending to padding tokens
            position_ids: Position indices (not used in sparse attention)
            past_key_values: Cached key/value states for incremental decoding
            use_cache: Whether to use cached key/value states
            output_attentions: Whether to return attention weights
            
        Returns:
            tuple: (context_layer, (key, value), attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project inputs to queries, keys, and values
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Handle cached key/values for incremental decoding
        if past_key_values is not None:
            past_key, past_value = past_key_values
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)
            
        # Cache current key/values if needed
        if use_cache:
            current_key_value = (key, value)
        else:
            current_key_value = None
            
        # Split heads
        query = self._split_heads(query)  # [batch, num_heads, seq_len, head_dim]
        key = self._split_heads(key)      # [batch, num_heads, key_len, head_dim]
        value = self._split_heads(value)  # [batch, num_heads, key_len, head_dim]
        
        # Get sequence length of keys (which may differ from queries in incremental decoding)
        key_len = key.size(2)
        
        # Compute scaled dot-product attention
        # [batch, num_heads, seq_len, key_len]
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        
        # Create local attention mask
        local_mask = self._create_local_attention_mask(
            seq_len=max(seq_len, key_len),
            window_size=self.window_size,
            device=hidden_states.device,
        )
        
        # Apply both local mask and attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask + local_mask
        else:
            attention_scores = attention_scores + local_mask
            
        # Normalize the attention scores
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context_layer = torch.matmul(attention_weights, value)  # [batch, num_heads, seq_len, head_dim]
        
        # Merge heads back
        context_layer = self._merge_heads(context_layer)  # [batch, seq_len, hidden_size]
        
        # Apply output projection
        output = self.out_proj(context_layer)
        
        outputs = (output, current_key_value)
        if output_attentions:
            outputs += (attention_weights,)
            
        return outputs
