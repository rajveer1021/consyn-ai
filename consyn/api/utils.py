# consyn/api/utils.py
"""
Utility functions for Consyn AI API.
This module provides helper functions for loading models and creating inference engines.
"""

import os
import logging
from typing import Dict, Optional, Tuple

import torch

from ..model.config import ConsynConfig, ConsynVerseConfig, ConsynStanzaConfig, ConsynEpicConfig
from ..model.architecture import ConsynLMHeadModel
from ..tokenization.bpe import BPETokenizer
from ..tokenization.sentencepiece_wrapper import SentencePieceTokenizer
from ..inference.engine import ConsynInferenceEngine

# Set up logging
logger = logging.getLogger(__name__)

# Initialize model and tokenizer caches
# Create these as module-level variables first
model_cache = {}
tokenizer_cache = {}


def get_model_path(model_name: str) -> str:
    """
    Get the path to the model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        str: Path to the model
    """
    # Check environment variable for model directory
    model_dir = os.environ.get("MODEL_DIR", "./models")
    
    # Build model path
    return os.path.join(model_dir, f"consyn_{model_name}")


def load_model(model_name: str) -> Tuple:
    """
    Load a model and tokenizer.
    
    Args:
        model_name: Name of the model
        
    Returns:
        tuple: (model, tokenizer)
        
    Raises:
        Exception: If the model cannot be loaded
    """
    # Check if model is already loaded
    global model_cache, tokenizer_cache
    
    if model_name in model_cache:
        return model_cache[model_name], tokenizer_cache[model_name]
        
    # Get model path
    model_path = get_model_path(model_name)
    
    # Check if model exists
    if not os.path.exists(model_path):
        # Model doesn't exist, so initialize a new one
        logger.info(f"Model {model_name} not found. Initializing new model.")
        
        # Create configuration based on model name
        if model_name == "verse":
            config = ConsynVerseConfig()
        elif model_name == "stanza":
            config = ConsynStanzaConfig()
        elif model_name == "epic":
            config = ConsynEpicConfig()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        # Initialize model with config
        model = ConsynLMHeadModel(config)
        
        # Initialize tokenizer
        tokenizer = BPETokenizer()
        
    else:
        try:
            # Try to load with Hugging Face transformers
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                model = AutoModelForCausalLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
            except (ImportError, ValueError):
                # Fallback to loading with PyTorch
                logger.info(f"Loading model {model_name} with PyTorch")
                
                # Load config
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    
                    config = ConsynConfig.from_dict(config_dict)
                else:
                    # Use default config based on model name
                    if model_name == "verse":
                        config = ConsynVerseConfig()
                    elif model_name == "stanza":
                        config = ConsynStanzaConfig()
                    elif model_name == "epic":
                        config = ConsynEpicConfig()
                    else:
                        raise ValueError(f"Unsupported model: {model_name}")
                        
                # Initialize model with config
                model = ConsynLMHeadModel(config)
                
                # Load model weights if available
                weights_path = os.path.join(model_path, "pytorch_model.bin")
                if os.path.exists(weights_path):
                    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
                    
                # Load tokenizer
                tokenizer_path = os.path.join(model_path, "tokenizer")
                if os.path.exists(tokenizer_path):
                    tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
                else:
                    # Try loading SentencePiece tokenizer
                    spiece_path = os.path.join(model_path, "spiece.model")
                    if os.path.exists(spiece_path):
                        tokenizer = SentencePieceTokenizer(model_file=spiece_path)
                    else:
                        # Fallback to default tokenizer
                        tokenizer = BPETokenizer()
                        
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
            
    # Cache model and tokenizer
    model_cache[model_name] = model
    tokenizer_cache[model_name] = tokenizer
    
    return model, tokenizer


def get_inference_engine(model_name: str) -> ConsynInferenceEngine:
    """
    Get an inference engine for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ConsynInferenceEngine: Inference engine
    """
    # Load model and tokenizer
    model, tokenizer = load_model(model_name)
    
    # Create inference engine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return ConsynInferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_kv_cache=True,
        use_flash_attention=True,
    )