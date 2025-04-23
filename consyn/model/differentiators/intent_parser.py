# consyn/model/differentiators/intent_parser.py
"""
Intent Parsing Module for Consyn AI models.

This module provides capabilities for internal intent detection, query decomposition,
and state tracking to help the model understand user intentions more effectively.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ConsynConfig


class IntentClassifier(nn.Module):
    """
    Classifier for detecting the intent of user queries.
    
    This component analyzes input text to determine what the user is trying to accomplish,
    helping the model generate more relevant responses.
    """
    
    def __init__(self, hidden_size: int, num_intents: int = 8):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_intents = num_intents
        
        # Intent classification layers
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_intents),
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Classify input hidden states into intent probabilities.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: Intent logits of shape [batch_size, num_intents]
        """
        # Use the [CLS] token or the mean pooling of the sequence
        if hidden_states.size(1) > 1:
            # Mean pooling
            pooled_output = hidden_states.mean(dim=1)
        else:
            # Single token (e.g., [CLS])
            pooled_output = hidden_states[:, 0]
            
        # Classify intent
        intent_logits = self.intent_classifier(pooled_output)
        
        return intent_logits


class QueryDecomposer(nn.Module):
    """
    Decomposes complex queries into simpler sub-queries.
    
    This helps the model handle multi-part or complex requests by breaking
    them down into manageable pieces that can be addressed separately.
    """
    
    def __init__(self, hidden_size: int, max_sub_queries: int = 4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_sub_queries = max_sub_queries
        
        # Query decomposition mechanism
        self.decomposer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * max_sub_queries),
        )
        
        # Relevance detector to determine which sub-queries are meaningful
        self.relevance_detector = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose input query into sub-queries.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            tuple: (sub_queries, relevance_scores)
                sub_queries: Tensor of shape [batch_size, max_sub_queries, hidden_size]
                relevance_scores: Tensor of shape [batch_size, max_sub_queries]
        """
        # Use the [CLS] token or mean pooling for query representation
        if hidden_states.size(1) > 1:
            query_repr = hidden_states.mean(dim=1)
        else:
            query_repr = hidden_states[:, 0]
            
        # Generate decomposed sub-queries
        decomposed = self.decomposer(query_repr)
        
        # Reshape to get individual sub-queries
        batch_size = hidden_states.size(0)
        sub_queries = decomposed.view(batch_size, self.max_sub_queries, self.hidden_size)
        
        # Determine relevance of each sub-query
        relevance_scores = self.relevance_detector(sub_queries).squeeze(-1)
        relevance_scores = torch.sigmoid(relevance_scores)  # Convert to probability
        
        return sub_queries, relevance_scores


class StateTracker(nn.Module):
    """
    Tracks conversation state and query context over time.
    
    This component maintains a representation of the conversation state,
    allowing the model to maintain context across multiple turns.
    """
    
    def __init__(self, hidden_size: int, state_size: int = 256):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.state_size = state_size
        
        # State update mechanism (GRU-based)
        self.state_updater = nn.GRUCell(hidden_size, state_size)
        
        # State projection for compatibility with model
        self.state_projector = nn.Linear(state_size, hidden_size)
        
        # State initialization
        self.state_initializer = nn.Parameter(torch.zeros(1, state_size))
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update conversation state based on new input.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            prev_state: Previous state tensor of shape [batch_size, state_size]
            
        Returns:
            tuple: (state_context, new_state)
                state_context: Tensor of shape [batch_size, hidden_size]
                new_state: Updated state tensor of shape [batch_size, state_size]
        """
        batch_size = hidden_states.size(0)
        
        # Use mean pooling for query representation
        query_repr = hidden_states.mean(dim=1)
        
        # Initialize state if needed
        if prev_state is None:
            prev_state = self.state_initializer.expand(batch_size, -1)
            
        # Update state with new information
        new_state = self.state_updater(query_repr, prev_state)
        
        # Project state to model dimension for integration
        state_context = self.state_projector(new_state)
        
        return state_context, new_state


class IntentParsingModule(nn.Module):
    """
    Main intent parsing module that integrates classification, decomposition, and state tracking.
    
    This module helps the model understand user intentions, break down complex queries,
    and maintain context throughout a conversation.
    """
    
    def __init__(self, config: ConsynConfig):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Intent classification
        self.intent_classifier = IntentClassifier(
            hidden_size=self.hidden_size,
            num_intents=8,  # Configurable number of intent categories
        )
        
        # Query decomposition
        self.query_decomposer = QueryDecomposer(
            hidden_size=self.hidden_size,
            max_sub_queries=4,  # Configurable number of sub-queries
        )
        
        # State tracking
        self.state_tracker = StateTracker(
            hidden_size=self.hidden_size,
            state_size=256,  # Configurable state size
        )
        
        # Integration layer to combine all components
        self.intent_integration = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
        # Dynamic weighting of sub-queries
        self.subquery_weighting = nn.Linear(8, 4)  # Map intents to subquery weights
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prev_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process input with intent parsing capabilities.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask to avoid attending to padding tokens
            prev_state: Previous conversation state if available
            
        Returns:
            tuple: (updated_hidden_states, parsing_outputs)
                updated_hidden_states: Hidden states with intent context integrated
                parsing_outputs: Dictionary containing intent information
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Classify intent
        intent_logits = self.intent_classifier(hidden_states)
        intent_probs = F.softmax(intent_logits, dim=-1)
        
        # Decompose query
        sub_queries, relevance_scores = self.query_decomposer(hidden_states)
        
        # Update conversation state
        state_context, new_state = self.state_tracker(hidden_states, prev_state)
        
        # Weight sub-queries based on intent
        subquery_weights = F.softmax(self.subquery_weighting(intent_probs), dim=-1)
        
        # Apply weights to sub-queries
        weighted_subqueries = torch.einsum(
            "bn,bnd->bd", 
            subquery_weights * relevance_scores,  # Combine intent and relevance
            sub_queries
        )
        
        # Integrate state context and weighted sub-queries
        intent_context = torch.cat([state_context, weighted_subqueries], dim=-1)
        intent_features = self.intent_integration(intent_context)
        
        # Expand to match sequence length
        intent_features = intent_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Add intent features to original hidden states (residual connection)
        updated_hidden_states = hidden_states + intent_features
        
        # Collect outputs for potential use elsewhere
        parsing_outputs = {
            "intent_probs": intent_probs,
            "sub_queries": sub_queries,
            "relevance_scores": relevance_scores,
            "state": new_state,
        }
        
        return updated_hidden_states, parsing_outputs