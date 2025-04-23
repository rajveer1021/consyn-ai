"""
Test suite for Consyn AI model components
"""

import pytest
import torch

from consyn.model.config import ConsynVerseConfig, ConsynStanzaConfig
from consyn.model.architecture import ConsynModel, ConsynLMHeadModel
from consyn.model.attention import MultiHeadAttention, RotaryAttention
from consyn.model.embeddings import ConsynEmbeddings
from consyn.model.feedforward import ConsynMLP


class TestModelArchitecture:
    def test_verse_model_initialization(self):
        """Test Verse model initialization"""
        config = ConsynVerseConfig()
        model = ConsynModel(config)
        assert model is not None
        assert model.config == config

    def test_lm_head_model_forward(self):
        """Test language model head forward pass"""
        config = ConsynVerseConfig()
        model = ConsynLMHeadModel(config)
        
        # Create dummy input
        input_ids = torch.randint(0, config.vocab_size, (2, 32))
        
        # Forward pass
        outputs = model(input_ids)
        assert 'loss' in outputs
        assert 'logits' in outputs


class TestAttentionMechanisms:
    def test_multi_head_attention(self):
        """Test multi-head attention initialization"""
        config = ConsynVerseConfig()
        attention = MultiHeadAttention(
            hidden_size=config.hidden_size, 
            num_heads=config.num_attention_heads
        )
        assert attention is not None

    def test_rotary_attention(self):
        """Test rotary attention initialization"""
        config = ConsynStanzaConfig()
        attention = RotaryAttention(
            hidden_size=config.hidden_size, 
            num_heads=config.num_attention_heads,
            rotary_dim=32
        )
        assert attention is not None


class TestEmbeddings:
    def test_consyn_embeddings(self):
        """Test Consyn embeddings initialization"""
        config = ConsynVerseConfig()
        embeddings = ConsynEmbeddings(config)
        assert embeddings is not None

        # Test embedding forward pass
        input_ids = torch.randint(0, config.vocab_size, (2, 32))
        position_ids = torch.arange(32).unsqueeze(0).expand(2, -1)
        
        embedded = embeddings(input_ids, position_ids)
        assert embedded.shape == (2, 32, config.hidden_size)


class TestFeedForward:
    def test_mlp_initialization(self):
        """Test MLP initialization"""
        config = ConsynVerseConfig()
        mlp = ConsynMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )
        assert mlp is not None

        # Test MLP forward pass
        input_tensor = torch.randn(2, 32, config.hidden_size)
        output = mlp(input_tensor)
        assert output.shape == input_tensor.shape