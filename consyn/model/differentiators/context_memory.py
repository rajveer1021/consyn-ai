# consyn/model/differentiators/context_memory.py
"""
Context-Aware Memory Module for Consyn AI models.

This module implements a long-term memory mechanism that allows the model
to maintain and access information beyond its immediate context window.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ConsynConfig


class MemoryCell(nn.Module):
    """
    A single memory cell that stores information and provides access mechanisms.
    """
    
    def __init__(self, hidden_size: int, memory_dim: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        
        # Projection to create memory keys
        self.key_proj = nn.Linear(hidden_size, memory_dim)
        
        # Projection to create memory values
        self.value_proj = nn.Linear(hidden_size, memory_dim)
        
        # Output projection when retrieving from memory
        self.output_proj = nn.Linear(memory_dim, hidden_size)
        
    def create_memory(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create memory keys and values from hidden states.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask to avoid attending to padding tokens
            
        Returns:
            tuple: (memory_keys, memory_values) of shape [batch_size, seq_len, memory_dim]
        """
        # Project to keys and values
        keys = self.key_proj(hidden_states)
        values = self.value_proj(hidden_states)
        
        # Apply mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match the dimensions
            mask = attention_mask.unsqueeze(-1).expand_as(keys)
            keys = keys * mask
            values = values * mask
            
        return keys, values
        
    def access_memory(
        self, 
        query: torch.Tensor, 
        memory_keys: torch.Tensor, 
        memory_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Access memory using attention mechanism.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len, hidden_size]
            memory_keys: Memory keys of shape [batch_size, mem_len, memory_dim]
            memory_values: Memory values of shape [batch_size, mem_len, memory_dim]
            
        Returns:
            torch.Tensor: Retrieved memory of shape [batch_size, seq_len, hidden_size]
        """
        # Project query to memory space
        query_proj = self.key_proj(query)
        
        # Compute attention scores
        attn_scores = torch.matmul(query_proj, memory_keys.transpose(-1, -2))
        
        # Scale attention scores
        attn_scores = attn_scores / math.sqrt(self.memory_dim)
        
        # Normalize attention scores
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to memory values
        context = torch.matmul(attn_probs, memory_values)
        
        # Project back to hidden space
        output = self.output_proj(context)
        
        return output


class ContextMemoryModule(nn.Module):
    """
    Context-Aware Memory Module for extending the model's context window.
    
    This module maintains a persistent memory of past context that the
    model can access and update during processing. It enables the model
    to maintain information over much longer contexts than would fit
    in the standard attention window.
    """
    
    def __init__(self, config: ConsynConfig):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.memory_dim = config.hidden_size // 2  # Compressed memory representation
        self.num_memory_cells = 16  # Number of separate memory cells
        
        # Memory cells - each cell specializes in different types of information
        self.memory_cells = nn.ModuleList([
            MemoryCell(self.hidden_size, self.memory_dim)
            for _ in range(self.num_memory_cells)
        ])
        
        # Cell router to determine which memory cell to use
        self.cell_router = nn.Linear(self.hidden_size, self.num_memory_cells)
        
        # Memory compression for summarizing long sequences
        self.compressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        
        # Gate to control memory influence
        self.memory_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def compress_memory(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compress long sequences into more compact memory representation.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask to avoid attending to padding tokens
            
        Returns:
            torch.Tensor: Compressed memory of shape [batch_size, compressed_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Apply compression directly (more sophisticated methods could be used)
        compressed_memory = self.compressor(hidden_states)
        
        # For long sequences, we can subsample to further compress
        if seq_len > 512:
            stride = seq_len // 512
            compressed_memory = compressed_memory[:, ::stride, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, ::stride]
                
        return compressed_memory, attention_mask
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process hidden states with context memory.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask to avoid attending to padding tokens
            memory_states: Previous memory states if available
            
        Returns:
            tuple: (updated_hidden_states, updated_memory_states)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Initialize memory if needed
        if memory_states is None:
            memory_states = {
                f"cell_{i}": {
                    "keys": torch.zeros(batch_size, 0, self.memory_dim, device=hidden_states.device),
                    "values": torch.zeros(batch_size, 0, self.memory_dim, device=hidden_states.device),
                }
                for i in range(self.num_memory_cells)
            }
            
        # Determine which memory cells to use for different parts of the sequence
        cell_weights = F.softmax(self.cell_router(hidden_states), dim=-1)  # [batch, seq_len, num_cells]
        
        # Initialize output tensor for gathered memory
        memory_output = torch.zeros_like(hidden_states)
        
        # Process with each memory cell
        for i, cell in enumerate(self.memory_cells):
            # Get the weight for this cell
            cell_weight = cell_weights[:, :, i].unsqueeze(-1)  # [batch, seq_len, 1]
            
            # Access memory from this cell
            cell_memory = cell.access_memory(
                hidden_states,
                memory_states[f"cell_{i}"]["keys"],
                memory_states[f"cell_{i}"]["values"]
            )
            
            # Weight the memory by cell relevance
            memory_output += cell_weight * cell_memory
            
        # Create new memories from current hidden states
        compressed_states, compressed_mask = self.compress_memory(
            hidden_states, attention_mask
        )
        
        # Update memory for each cell
        updated_memory_states = {}
        for i, cell in enumerate(self.memory_cells):
            # Create new memory
            new_keys, new_values = cell.create_memory(
                compressed_states, compressed_mask
            )
            
            # Concatenate with old memory, keeping a fixed maximum size
            old_keys = memory_states[f"cell_{i}"]["keys"]
            old_values = memory_states[f"cell_{i}"]["values"]
            
            # Limit memory size
            max_memory_len = 2048  # Maximum number of memory items to keep
            
            combined_keys = torch.cat([old_keys, new_keys], dim=1)
            combined_values = torch.cat([old_values, new_values], dim=1)
            
            if combined_keys.size(1) > max_memory_len:
                # Keep only the most recent memories
                combined_keys = combined_keys[:, -max_memory_len:, :]
                combined_values = combined_values[:, -max_memory_len:, :]
                
            updated_memory_states[f"cell_{i}"] = {
                "keys": combined_keys,
                "values": combined_values
            }
            
        # Gate the influence of memory on the output
        gate_input = torch.cat([hidden_states, memory_output], dim=-1)
        gate = self.memory_gate(gate_input)
        
        # Apply gated memory to hidden states
        updated_hidden_states = hidden_states + gate * memory_output
        
        return updated_hidden_states, updated_memory_states
