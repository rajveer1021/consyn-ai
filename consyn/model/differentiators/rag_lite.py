# consyn/model/differentiators/rag_lite.py
"""
Lightweight Retrieval-Augmented Generation (RAG-Lite) module for Consyn AI models.

This module enables the model to perform internal retrieval operations during
generation, without requiring an external retrieval system. It creates and
maintains an internal key-value memory that can be queried during inference.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ConsynConfig


class ChunkEncoder(nn.Module):
    """
    Encodes text chunks into dense vector representations for retrieval.
    """
    
    def __init__(self, hidden_size: int, embedding_size: int = 128):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        # Encoder projection
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, embedding_size),
            nn.LayerNorm(embedding_size),
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Encode hidden states into dense vectors.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: Encoded vectors of shape [batch_size, embedding_size]
        """
        # Mean pooling over sequence length
        pooled = hidden_states.mean(dim=1)
        
        # Project to embedding space
        embeddings = self.encoder(pooled)
        
        # Normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class ChunkIndexer(nn.Module):
    """
    Manages a collection of encoded text chunks for retrieval.
    """
    
    def __init__(self, embedding_size: int, max_chunks: int = 1024):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.max_chunks = max_chunks
        
        # Initialize empty index
        self.register_buffer(
            "index_embeddings", 
            torch.zeros(0, embedding_size)
        )
        
        # Track original positions for potential value retrieval
        self.register_buffer(
            "index_positions",
            torch.zeros(0, dtype=torch.long)
        )
        
    def add_to_index(
        self, 
        embeddings: torch.Tensor, 
        positions: torch.Tensor
    ) -> None:
        """
        Add new embeddings to the index.
        
        Args:
            embeddings: Tensor of shape [num_chunks, embedding_size]
            positions: Tensor of shape [num_chunks] with original positions
        """
        # Concatenate with existing index
        self.index_embeddings = torch.cat([self.index_embeddings, embeddings], dim=0)
        self.index_positions = torch.cat([self.index_positions, positions], dim=0)
        
        # Limit index size if needed
        if len(self.index_embeddings) > self.max_chunks:
            self.index_embeddings = self.index_embeddings[-self.max_chunks:]
            self.index_positions = self.index_positions[-self.max_chunks:]
            
    def search(
        self, 
        query_embeddings: torch.Tensor, 
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search the index for most similar chunks.
        
        Args:
            query_embeddings: Query tensor of shape [batch_size, embedding_size]
            top_k: Number of top results to return
            
        Returns:
            tuple: (scores, positions)
                scores: Similarity scores of shape [batch_size, top_k]
                positions: Original positions of shape [batch_size, top_k]
        """
        batch_size = query_embeddings.size(0)
        
        # Handle empty index case
        if len(self.index_embeddings) == 0:
            return (
                torch.zeros(batch_size, top_k, device=query_embeddings.device),
                torch.zeros(batch_size, top_k, dtype=torch.long, device=query_embeddings.device)
            )
            
        # Compute similarity scores
        # [batch_size, index_size]
        similarity = torch.matmul(query_embeddings, self.index_embeddings.T)
        
        # Get top-k scores and indices
        top_k = min(top_k, len(self.index_embeddings))
        scores, indices = torch.topk(similarity, k=top_k, dim=-1)
        
        # Get original positions
        positions = self.index_positions[indices]
        
        return scores, positions


class ValueStore(nn.Module):
    """
    Stores and retrieves value information associated with indexed chunks.
    """
    
    def __init__(self, hidden_size: int, max_chunks: int = 1024):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_chunks = max_chunks
        
        # Initialize empty value store
        self.register_buffer(
            "values",
            torch.zeros(0, hidden_size)
        )
        
    def add_values(self, values: torch.Tensor) -> None:
        """
        Add new values to the store.
        
        Args:
            values: Tensor of shape [num_chunks, hidden_size]
        """
        # Concatenate with existing values
        self.values = torch.cat([self.values, values], dim=0)
        
        # Limit store size if needed
        if len(self.values) > self.max_chunks:
            self.values = self.values[-self.max_chunks:]
            
    def retrieve(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Retrieve values at specified positions.
        
        Args:
            positions: Tensor of positions to retrieve
            
        Returns:
            torch.Tensor: Retrieved values
        """
        # Handle empty store case
        if len(self.values) == 0:
            return torch.zeros(
                positions.size(0), positions.size(1), self.hidden_size,
                device=positions.device
            )
            
        # Ensure positions are within bounds
        positions = positions.clamp(0, len(self.values) - 1)
        
        # Gather values using positions
        # For each batch item and top-k entry, get the corresponding value
        batch_size, top_k = positions.size()
        gathered_values = torch.zeros(
            batch_size, top_k, self.hidden_size,
            device=positions.device
        )
        
        for i in range(batch_size):
            gathered_values[i] = self.values[positions[i]]
            
        return gathered_values


class RAGLiteModule(nn.Module):
    """
    Lightweight Retrieval-Augmented Generation Module.
    
    This module enables the model to maintain a memory of previously processed
    text and retrieve relevant information during generation, similar to
    retrieval-augmented generation but without external knowledge sources.
    """
    
    def __init__(self, config: ConsynConfig):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.embedding_size = 128  # Size of retrieval embeddings
        self.chunk_size = 64       # Size of text chunks for indexing
        self.overlap = 16          # Overlap between chunks
        self.max_chunks = 1024     # Maximum number of chunks to store
        self.top_k = 3             # Number of chunks to retrieve
        
        # Components
        self.chunk_encoder = ChunkEncoder(
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size,
        )
        
        self.chunk_indexer = ChunkIndexer(
            embedding_size=self.embedding_size,
            max_chunks=self.max_chunks,
        )
        
        self.value_store = ValueStore(
            hidden_size=self.hidden_size,
            max_chunks=self.max_chunks,
        )
        
        # Query projection for retrieval
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Context integration
        self.retrieval_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 1),
            nn.Sigmoid(),
        )
        
        # Information fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
    def _chunk_sequence(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split sequence into overlapping chunks for indexing.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask to avoid attending to padding tokens
            
        Returns:
            tuple: (chunk_hiddens, chunk_masks, chunk_positions)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Only process if sequence is long enough
        if seq_len < self.chunk_size:
            return (
                hidden_states,
                attention_mask if attention_mask is not None else torch.ones(batch_size, seq_len, device=hidden_states.device),
                torch.arange(seq_len, device=hidden_states.device).expand(batch_size, -1)
            )
            
        # Create overlapping chunks
        chunk_hiddens = []
        chunk_masks = []
        chunk_positions = []
        
        for i in range(0, seq_len - self.chunk_size + 1, self.chunk_size - self.overlap):
            # Ensure we don't go past the end of the sequence
            end_idx = min(i + self.chunk_size, seq_len)
            
            # Extract chunk
            chunk = hidden_states[:, i:end_idx]
            chunk_hiddens.append(chunk)
            
            # Extract mask if available
            if attention_mask is not None:
                chunk_mask = attention_mask[:, i:end_idx]
                chunk_masks.append(chunk_mask)
                
            # Store original positions
            positions = torch.arange(i, end_idx, device=hidden_states.device)
            positions = positions.expand(batch_size, -1)
            chunk_positions.append(positions)
            
        # Concatenate chunks
        chunk_hiddens = torch.cat(chunk_hiddens, dim=1)
        
        if attention_mask is not None:
            chunk_masks = torch.cat(chunk_masks, dim=1)
        else:
            chunk_masks = torch.ones_like(chunk_hiddens[..., 0])
            
        chunk_positions = torch.cat(chunk_positions, dim=1)
        
        return chunk_hiddens, chunk_masks, chunk_positions
        
    def _encode_and_index(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> None:
        """
        Encode sequence chunks and add to the index.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask to avoid attending to padding tokens
        """
        # Only use the first batch item for indexing (assumption: batch_size=1 during inference)
        first_item = hidden_states[0:1]
        first_mask = attention_mask[0:1] if attention_mask is not None else None
        
        # Chunk the sequence
        chunk_hiddens, chunk_masks, chunk_positions = self._chunk_sequence(first_item, first_mask)
        
        # Encode each chunk
        batch_size, num_chunks, _ = chunk_hiddens.size()
        flattened_hiddens = chunk_hiddens.view(-1, self.hidden_size)
        
        # Process in smaller batches to avoid memory issues
        embeddings_list = []
        chunk_batch_size = 16
        
        for i in range(0, flattened_hiddens.size(0), chunk_batch_size):
            end_idx = min(i + chunk_batch_size, flattened_hiddens.size(0))
            batch_embeddings = self.chunk_encoder(flattened_hiddens[i:end_idx, None, :])
            embeddings_list.append(batch_embeddings)
            
        chunk_embeddings = torch.cat(embeddings_list, dim=0)
        
        # Store values (mean-pooled chunk representations)
        chunk_values = flattened_hiddens
        
        # Add to index and value store
        self.chunk_indexer.add_to_index(
            chunk_embeddings,
            chunk_positions.view(-1),
        )
        
        self.value_store.add_values(chunk_values)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ) -> torch.Tensor:
        """
        Process input with retrieval-augmented generation.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask to avoid attending to padding tokens
            is_training: Whether the model is in training mode
            
        Returns:
            torch.Tensor: Hidden states enhanced with retrieved context
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # During training, 50% of the time we update the index, 50% we retrieve
        if is_training:
            if torch.rand(1).item() < 0.5:
                self._encode_and_index(hidden_states, attention_mask)
                # Return input unchanged during training when updating index
                return hidden_states
        else:
            # During inference, always update the index
            self._encode_and_index(hidden_states, attention_mask)
            
        # Prepare query representation (use last token for retrieval in inference)
        query_hidden = hidden_states[:, -1:] if not is_training else hidden_states
        query_proj = self.query_proj(query_hidden)
        
        # Encode query for retrieval
        query_embeddings = self.chunk_encoder(query_proj)
        
        # Retrieve most relevant chunks
        scores, positions = self.chunk_indexer.search(
            query_embeddings, 
            top_k=self.top_k
        )
        
        # Get values for retrieved chunks
        retrieved_values = self.value_store.retrieve(positions)
        
        # Weight retrieved values by their scores
        scores = scores.unsqueeze(-1)  # [batch_size, top_k, 1]
        weighted_values = retrieved_values * scores
        
        # Combine retrieved values
        combined_retrieval = weighted_values.mean(dim=1)  # [batch_size, hidden_size]
        
        # For inference, replicate for each position in the sequence
        # For training, we already have batch_size x seq_len
        if not is_training:
            combined_retrieval = combined_retrieval.expand(batch_size, seq_len, -1)
        
        # Determine how much to use retrieved information (gating)
        gate_input = torch.cat([hidden_states, combined_retrieval], dim=-1)
        gate = self.retrieval_gate(gate_input)
        
        # Combine original and retrieved information
        fusion_input = torch.cat([hidden_states, gate * combined_retrieval], dim=-1)
        fused_output = self.fusion_layer(fusion_input)
        
        # Add residual connection
        enhanced_hidden_states = hidden_states + fused_output
        
        return enhanced_hidden_states