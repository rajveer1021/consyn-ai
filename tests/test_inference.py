"""
Test suite for Consyn AI inference components
"""

import pytest
import torch

from consyn.model.config import ConsynVerseConfig
from consyn.model.architecture import ConsynLMHeadModel
from consyn.tokenization.bpe import BPETokenizer
from consyn.inference.engine import ConsynInferenceEngine
from consyn.inference.sampling import (
    sample_next_token, 
    top_k_filtering, 
    top_p_filtering
)
from consyn.inference.quantization import quantize_model


class TestInferenceEngine:
    def test_inference_engine_initialization(self):
        """Test inference engine creation"""
        config = ConsynVerseConfig()
        model = ConsynLMHeadModel(config)
        tokenizer = BPETokenizer()
        
        engine = ConsynInferenceEngine(
            model=model,
            tokenizer=tokenizer
        )
        
        assert engine is not None

    def test_text_generation(self):
        """Test basic text generation"""
        config = ConsynVerseConfig()
        model = ConsynLMHeadModel(config)
        tokenizer = BPETokenizer()
        
        engine = ConsynInferenceEngine(
            model=model,
            tokenizer=tokenizer
        )
        
        prompt = "Once upon a time"
        generated_text = engine.generate(prompt, max_new_tokens=50)
        
        assert isinstance(generated_text, list)
        assert len(generated_text) > 0
        assert isinstance(generated_text[0], str)


class TestSamplingMethods:
    def test_sample_next_token(self):
        """Test token sampling method"""
        # Create dummy logits
        logits = torch.rand(1, 1000)
        
        # Sample token
        next_token, prob = sample_next_token(
            logits, 
            temperature=0.7, 
            top_k=50, 
            top_p=0.9
        )
        
        assert next_token is not None
        assert 0 <= prob <= 1

    def test_top_k_filtering(self):
        """Test top-k filtering"""
        logits = torch.rand(1, 1000)
        filtered_logits = top_k_filtering(logits, top_k=50)
        
        assert filtered_logits.shape == logits.shape

    def test_top_p_filtering(self):
        """Test top-p (nucleus) filtering"""
        logits = torch.rand(1, 1000)
        filtered_logits = top_p_filtering(logits, top_p=0.9)
        
        assert filtered_logits.shape == logits.shape


class TestQuantization:
    def test_model_quantization(self):
        """Test basic model quantization"""
        config = ConsynVerseConfig()
        model = ConsynLMHeadModel(config)
        
        # Quantize model
        quantized_model = quantize_model(
            model, 
            method='dynamic',
            bits=8
        )
        
        assert quantized_model is not None