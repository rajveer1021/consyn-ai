# consyn/model/config.py
"""
Configuration for Consyn AI models.
This module defines the configuration classes for different model sizes in the Consyn family.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class ConsynConfig:
    """Base configuration class for Consyn models."""
    
    # Model size and structure
    vocab_size: int = 50257  # Default GPT-2 vocabulary size
    max_position_embeddings: int = 2048
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072  # Size of the feed-forward layer
    hidden_act: str = "gelu"  # Activation function
    
    # Dropout rates
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Embeddings
    type_vocab_size: int = 1  # Used for token type embeddings if needed
    
    # Normalization
    layer_norm_eps: float = 1e-12
    use_rms_norm: bool = False  # Whether to use RMSNorm instead of LayerNorm
    
    # Attention
    attention_type: str = "standard"  # Options: standard, rotary, sparse
    rotary_dim: Optional[int] = None  # Dimension for rotary embeddings if used
    sparse_attention_window: Optional[int] = None  # Window size for sparse attention
    
    # Initialization
    initializer_range: float = 0.02
    
    # Advanced features
    tie_word_embeddings: bool = True
    use_cache: bool = True  # Whether to use KV cache for inference
    
    # Custom differentiators
    use_context_memory: bool = False
    use_intent_parsing: bool = False
    use_rag_lite: bool = False
    
    # Other
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Training specific (can be overridden by training config)
    gradient_checkpointing: bool = False
    
    def to_dict(self) -> Dict:
        """Convert the configuration to a dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ConsynConfig":
        """Create a configuration from a dictionary."""
        return cls(**config_dict)


@dataclass
class ConsynVerseConfig(ConsynConfig):
    """Configuration for Consyn Verse (Small) models."""
    
    # Overriding base parameters for small model (125M-350M params)
    hidden_size: int = 384
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    max_position_embeddings: int = 1024
    
    # Optimizations for smaller models
    hidden_dropout_prob: float = 0.05
    attention_probs_dropout_prob: float = 0.05


@dataclass
class ConsynStanzaConfig(ConsynConfig):
    """Configuration for Consyn Stanza (Medium) models."""
    
    # Overriding base parameters for medium model (1B-7B params)
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 2048
    
    # Advanced features more appropriate for medium-sized models
    attention_type: str = "rotary"
    rotary_dim: int = 32
    use_rms_norm: bool = True


@dataclass
class ConsynEpicConfig(ConsynConfig):
    """Configuration for Consyn Epic (Large) models."""
    
    # Overriding base parameters for large model (13B-65B+ params)
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 16384
    max_position_embeddings: int = 4096
    
    # Advanced features for large models
    attention_type: str = "rotary"
    rotary_dim: int = 64
    use_rms_norm: bool = True
    
    # Enable differentiators by default for the largest model
    use_context_memory: bool = True
    use_intent_parsing: bool = True
    use_rag_lite: bool = True
    
    # Optimizations for training efficiency on large models
    gradient_checkpointing: bool = True


def get_config_by_name(model_name: str) -> ConsynConfig:
    """Get the appropriate configuration by model name."""
    config_map = {
        "verse": ConsynVerseConfig,
        "stanza": ConsynStanzaConfig,
        "epic": ConsynEpicConfig,
    }
    
    # Default to base config if name not recognized
    config_class = config_map.get(model_name.lower(), ConsynConfig)
    return config_class()
