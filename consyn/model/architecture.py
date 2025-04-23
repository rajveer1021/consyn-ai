# consyn/model/architecture.py
"""
Core architecture implementation for Consyn AI models.
This module defines the transformer-based architecture for the Consyn family of models.
"""

import os
import json
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ConsynConfig
from .attention import MultiHeadAttention, RotaryAttention, SparseAttention
from .embeddings import ConsynEmbeddings
from .feedforward import ConsynMLP
from .normalization import LayerNorm, RMSNorm


class ConsynLayer(nn.Module):
    """
    A single transformer layer for Consyn models.
    Contains self-attention and feed-forward components with residual connections.
    """
    
    def __init__(self, config: ConsynConfig):
        super().__init__()
        self.config = config
        
        # Select the appropriate attention mechanism based on config
        if config.attention_type == "rotary":
            self.attention = RotaryAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout_prob=config.attention_probs_dropout_prob,
                rotary_dim=config.rotary_dim,
            )
        elif config.attention_type == "sparse":
            self.attention = SparseAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout_prob=config.attention_probs_dropout_prob,
                window_size=config.sparse_attention_window or 256,
            )
        else:  # Default to standard attention
            self.attention = MultiHeadAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout_prob=config.attention_probs_dropout_prob,
            )
            
        # Choose between LayerNorm and RMSNorm based on config
        norm_class = RMSNorm if config.use_rms_norm else LayerNorm
        
        # Normalization layers
        self.ln_1 = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.mlp = ConsynMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=config.hidden_act,
            dropout_prob=config.hidden_dropout_prob,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
    ) -> Tuple:
        """
        Forward pass for a transformer layer.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask to avoid attending to padding tokens
            position_ids: Position indices for positional embeddings
            past_key_values: Cached key/value states for incremental decoding
            use_cache: Whether to use cached key/value states
            output_attentions: Whether to return attention weights
            
        Returns:
            tuple: (hidden_states, cache, attention_weights)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Self-attention block
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        # Unpack attention outputs
        attn_output = attention_outputs[0]
        cache = attention_outputs[1] if use_cache else None
        attn_weights = attention_outputs[2] if output_attentions else None
        
        # First residual connection
        hidden_states = residual + attn_output
        
        # Feed-forward block
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Second residual connection
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if use_cache:
            outputs += (cache,)
        if output_attentions:
            outputs += (attn_weights,)
            
        return outputs


class ConsynModel(nn.Module):
    """
    Base transformer model for Consyn AI.
    """
    
    def __init__(self, config: ConsynConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.embeddings = ConsynEmbeddings(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ConsynLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        norm_class = RMSNorm if config.use_rms_norm else LayerNorm
        self.ln_f = norm_class(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Enable gradient checkpointing if specified
        self.gradient_checkpointing = config.gradient_checkpointing
        
    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (LayerNorm, RMSNorm)):
            module.weight.data.fill_(1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
                
    def get_input_embeddings(self):
        """Get the model's input embeddings."""
        return self.embeddings.word_embeddings
        
    def set_input_embeddings(self, new_embeddings):
        """Set the model's input embeddings."""
        self.embeddings.word_embeddings = new_embeddings
        
    def _prepare_attention_mask(
        self, 
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        device: torch.device
    ) -> torch.Tensor:
        """Prepare the attention mask for the model."""
        if attention_mask is None:
            # Default to all 1s (attend to all positions)
            attention_mask = torch.ones(input_shape, device=device)
            
        # Make the mask broadcastable to [batch_size, num_heads, seq_len, seq_len]
        extended_mask = attention_mask[:, None, None, :]
        
        # Convert mask to the format expected by the attention layers
        # (1.0 for positions to attend to, 0.0 for masked positions)
        extended_mask = (1.0 - extended_mask) * -10000.0
        
        return extended_mask
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the entire model.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Mask to avoid attending to padding tokens
            position_ids: Position indices for positional embeddings
            inputs_embeds: Pre-computed input embeddings
            past_key_values: Cached key/value states for incremental decoding
            use_cache: Whether to use cached key/value states
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states from all layers
            
        Returns:
            dict: Model outputs including hidden states, cache, and optionally attentions
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Validate inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
            
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        # Handle past key/values for incremental decoding
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)
            
        # Prepare position IDs
        if position_ids is None:
            position_ids = torch.arange(
                past_length, seq_length + past_length, 
                dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(
                attention_mask, input_shape, device
            )
            
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids, position_ids)
            
        hidden_states = inputs_embeds
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        new_key_values = () if use_cache else None
        
        # Process through all transformer layers
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            # Apply gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,  # past_key_value
                    False,  # use_cache
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                
            hidden_states = layer_outputs[0]
            
            if use_cache:
                new_key_values += (layer_outputs[1],)
                
            if output_attentions:
                all_attentions += (layer_outputs[-1],)
                
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": new_key_values,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }
        
    def _gradient_checkpointing_func(self, *args, **kwargs):
        """Wrapper for gradient checkpointing."""
        return torch.utils.checkpoint.checkpoint(*args, **kwargs)


class ConsynLMHeadModel(nn.Module):
    """
    Consyn language model with a language modeling head on top.
    Extended with save and load functionality.
    """
    
    def __init__(self, config: ConsynConfig):
        super().__init__()
        self.config = config
        
        # Base transformer model
        self.transformer = ConsynModel(config)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between embedding and output layer if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.get_input_embeddings().weight
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def save_pretrained(self, save_directory: str):
        """
        Save the model to a directory.
        
        Args:
            save_directory (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Optional: Save tokenizer details or other metadata
        # You might want to pass and save tokenizer separately
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 20,
        min_length: int = 0,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum length of generated sequences
            min_length: Minimum length of generated sequences
            do_sample: Whether to use sampling; if False, greedy decoding is used
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            num_return_sequences: Number of sequences to return
            **kwargs: Additional arguments
            
        Returns:
            torch.Tensor: Generated token IDs
        """
        batch_size = input_ids.shape[0]
        
        # Start with the input IDs
        curr_ids = input_ids.clone()
        
        # Create multiple copies if returning multiple sequences
        if num_return_sequences > 1:
            curr_ids = curr_ids.repeat(num_return_sequences, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
        
        # How many tokens to generate
        new_tokens_count = max_length - curr_ids.shape[1]
        
        # Setup for tracking generated tokens
        generated = []
        past_key_values = None
        
        # Generate tokens auto-regressively
        for i in range(new_tokens_count):
            # Forward pass
            with torch.no_grad():
                if past_key_values is None:
                    outputs = self(curr_ids, attention_mask=attention_mask, use_cache=True)
                else:
                    # Use past key values for efficiency
                    last_token = curr_ids[:, -1].unsqueeze(-1)
                    outputs = self(last_token, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
                
                next_token_logits = outputs["logits"][:, -1, :] if isinstance(outputs, dict) else outputs[0][:, -1, :]
                past_key_values = outputs["past_key_values"] if isinstance(outputs, dict) else outputs[1]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for b in range(batch_size):
                        for token_idx in set(curr_ids[b].tolist()):
                            next_token_logits[b, token_idx] /= repetition_penalty
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Apply filtering
                    for b in range(batch_size):
                        indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                        next_token_logits[b, indices_to_remove] = -float('Inf')
                
                # Sample or greedy selection
                if do_sample:
                    # Apply softmax to convert logits to probabilities
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy selection
                    next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Append to the sequence
                curr_ids = torch.cat([curr_ids, next_tokens], dim=-1)
                
                # Update attention mask if needed
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1))
                    ], dim=-1)
                
                # Stop if any sequence has reached the EOS token
                if hasattr(self.config, "eos_token_id") and self.config.eos_token_id is not None:
                    if (next_tokens == self.config.eos_token_id).any():
                        break
        
        # Return the generated sequences
        return curr_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        """
        Load a pretrained model from a directory.
        
        Args:
            pretrained_model_name_or_path (str): Path to the pretrained model
        
        Returns:
            ConsynLMHeadModel: Loaded model instance
        """
        # Load configuration
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        
        # Load config from JSON
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Create config object
        config = ConsynConfig.from_dict(config_dict)
        
        # Instantiate model with loaded config
        model = cls(config)
        
        # Load model weights
        weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        
        # Load state dict, using map_location to handle device compatibility
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # Load the state dictionary
        model.load_state_dict(state_dict)
        
        return model
                
    def get_output_embeddings(self):
        """Get the model's output embeddings."""
        return self.lm_head
        
    def set_output_embeddings(self, new_embeddings):
        """Set the model's output embeddings."""
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the language model with head.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Mask to avoid attending to padding tokens
            position_ids: Position indices for positional embeddings
            inputs_embeds: Pre-computed input embeddings
            labels: Language modeling labels
            past_key_values: Cached key/value states for incremental decoding
            use_cache: Whether to use cached key/value states
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states from all layers
            
        Returns:
            dict: Model outputs including loss, logits, and additional tensors
        """
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_states = transformer_outputs["last_hidden_state"]
        
        # Apply language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift the target one position to the right
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss using cross-entropy
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": transformer_outputs.get("past_key_values"),
            "hidden_states": transformer_outputs.get("hidden_states"),
            "attentions": transformer_outputs.get("attentions"),
        }
        
    def prepare_inputs_for_generation(
        self, 
        input_ids: torch.Tensor, 
        past_key_values=None, 
        attention_mask=None,
        **kwargs
    ) -> Dict:
        """
        Prepare inputs for text generation.
        
        Args:
            input_ids: Token IDs
            past_key_values: Cached key/value states
            attention_mask: Attention mask
            
        Returns:
            dict: Inputs prepared for generation
        """
        # Only use the last token for generation
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
        # Expand attention mask if needed
        if attention_mask is not None and past_key_values is not None:
            # Add new position to the attention mask for each past token
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1
            )
            
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }
