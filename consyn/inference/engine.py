# consyn/inference/engine.py
"""
Core inference engine for Consyn AI models.
This module provides utilities for running inference and generating text with Consyn models.
"""

import os
import time
import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..model.config import ConsynConfig


class ConsynInferenceEngine:
    """
    Inference engine for Consyn AI models.
    
    This class handles model loading, parameter efficient inference,
    and provides utilities for text generation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[torch.device] = None,
        max_length: int = 2048,
        use_kv_cache: bool = True,
        use_flash_attention: bool = False,
        quantization: Optional[str] = None,  # "int8", "int4", etc.
        device_map: Optional[Dict[str, Union[int, str]]] = None,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model: Consyn model
            tokenizer: Tokenizer for the model
            device: Device to run inference on
            max_length: Maximum sequence length
            use_kv_cache: Whether to use key-value cache for efficient generation
            use_flash_attention: Whether to use flash attention if available
            quantization: Quantization method to use
            device_map: Device map for multi-GPU inference
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Configure inference parameters
        self.max_length = max_length
        self.use_kv_cache = use_kv_cache
        self.use_flash_attention = use_flash_attention
        
        # Apply quantization if specified
        if quantization is not None:
            self._apply_quantization(quantization)
            
        # Handle device mapping for multi-GPU inference
        if device_map is not None:
            self._apply_device_map(device_map)
        else:
            # Move model to device
            self.model = self.model.to(self.device)
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Optimize model for inference
        self._optimize_for_inference()
        
    def _apply_quantization(self, quantization: str):
        """
        Apply quantization to the model.
        
        Args:
            quantization: Quantization method to use
        """
        try:
            if quantization == "int8":
                # Try using bitsandbytes for int8 quantization
                try:
                    import bitsandbytes as bnb
                    
                    # Replace Linear layers with 8-bit equivalents
                    for name, module in self.model.named_modules():
                        if isinstance(module, nn.Linear):
                            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                            parent = self.model if parent_name == "" else get_module_by_name(self.model, parent_name)
                            attr_name = name.rsplit(".", 1)[1] if "." in name else name
                            
                            setattr(
                                parent, 
                                attr_name, 
                                bnb.nn.Linear8bitLt(
                                    module.in_features, 
                                    module.out_features, 
                                    bias=module.bias is not None
                                )
                            )
                            
                    logging.info("Applied 8-bit quantization using bitsandbytes")
                except ImportError:
                    logging.warning("bitsandbytes not available. Trying PyTorch native quantization")
                    
                    # Use PyTorch's dynamic quantization
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {nn.Linear}, dtype=torch.qint8
                    )
                    logging.info("Applied 8-bit dynamic quantization using PyTorch")
                    
            elif quantization == "int4":
                # Try using optimum for int4 quantization
                try:
                    from optimum.bettertransformer import BetterTransformer
                    self.model = BetterTransformer.transform(self.model)
                    logging.info("Applied optimizations using BetterTransformer")
                    
                    # Use GPTQ or AWQ if available
                    try:
                        from auto_gptq import AutoGPTQForCausalLM
                        # Would need to save quantized model first
                        logging.warning("GPTQ quantization requires pre-quantized model")
                    except ImportError:
                        logging.warning("auto_gptq not available for int4 quantization")
                        
                except ImportError:
                    logging.warning("optimum not available for model optimization")
                    
            else:
                logging.warning(f"Unsupported quantization method: {quantization}")
                
        except Exception as e:
            logging.error(f"Error applying quantization: {e}")
            logging.warning("Running model without quantization")
            
    def _apply_device_map(self, device_map: Dict[str, Union[int, str]]):
        """
        Apply device map for multi-GPU inference.
        
        Args:
            device_map: Device map for model layers
        """
        try:
            # Try using Hugging Face accelerate for device map
            try:
                from accelerate import dispatch_model
                
                self.model = dispatch_model(self.model, device_map=device_map)
                logging.info(f"Applied device map using accelerate: {device_map}")
            except ImportError:
                logging.warning("accelerate not available for device mapping")
                
                # Fallback to manual device mapping
                for name, device_id in device_map.items():
                    if name == "":
                        # Default device for remaining layers
                        default_device = torch.device(f"cuda:{device_id}" if isinstance(device_id, int) else device_id)
                    else:
                        # Move specific module to device
                        module = get_module_by_name(self.model, name)
                        if module is not None:
                            device_id_str = f"cuda:{device_id}" if isinstance(device_id, int) else device_id
                            module.to(torch.device(device_id_str))
                            
                # Move any remaining modules to default device
                self.model.to(default_device if 'default_device' in locals() else self.device)
                
                logging.info(f"Applied manual device mapping")
                
        except Exception as e:
            logging.error(f"Error applying device map: {e}")
            logging.warning("Moving model to single device")
            self.model = self.model.to(self.device)
            
    def _optimize_for_inference(self):
        """Apply inference optimizations to the model."""
        # Enable gradient checkpointing if available
        if hasattr(self.model, "gradient_checkpointing"):
            self.model.gradient_checkpointing = False
            
        # Enable key-value cache if requested
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = self.use_kv_cache
            
        # Enable flash attention if requested and available
        if self.use_flash_attention:
            try:
                from optimum.bettertransformer import BetterTransformer
                self.model = BetterTransformer.transform(self.model)
                logging.info("Enabled flash attention using BetterTransformer")
            except ImportError:
                logging.warning("optimum.bettertransformer not available for flash attention")
                
        # Disable dropout for inference
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
                
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        stopping_criteria: Optional[List[Callable]] = None,
        output_scores: bool = False,
        **kwargs
    ) -> Union[List[str], Tuple[List[str], List[List[float]]]]:
        """Generate text based on a prompt."""
        # Encode the prompt - with added safety checks
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        except (TypeError, AttributeError) as e:
            # Fallback to character encoding if tokenizer fails
            print(f"Warning: Tokenizer failed. Using character fallback. Error: {e}")
            char_ids = [ord(c) % 50000 for c in prompt]  # Use character codes as fallback
            # Add special tokens
            if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
                char_ids = [self.tokenizer.bos_token_id] + char_ids
            input_ids = torch.tensor([char_ids], dtype=torch.long).to(self.device)
        
        # Ensure input_ids is at least 1 token long
        if input_ids.size(1) == 0:
            input_ids = torch.tensor([[1]], dtype=torch.long).to(self.device)  # Use a default token ID
        
        # Set up generation parameters with safety checks
        try:
            gen_kwargs = {
                "max_length": input_ids.shape[1] + max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
            }
            
            # Convert kwargs - be careful with param names
            gen_kwargs.update(**kwargs)
            
            # Generate text with added error handling
            outputs = self.model.generate(input_ids, **gen_kwargs)
        except Exception as e:
            print(f"Error during generation: {e}")
            return [f"Error generating text: {str(e)}"]
        
        # Decode generated sequences
        generated_sequences = []
        
        for seq in outputs:
            # Remove input prompt tokens
            generated_tokens = seq[input_ids.shape[1]:]
            
            try:
                # Decode to text
                decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_sequences.append(decoded_text)
            except Exception as e:
                # Fallback if decoding fails
                print(f"Error decoding tokens: {e}")
                # Create a simple string representation of the tokens
                fallback_text = ' '.join([str(t.item()) for t in generated_tokens])
                generated_sequences.append(f"[Decoding Error] Token IDs: {fallback_text}")
        
        return generated_sequences
                
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        callback: Optional[Callable[[str, int, float], None]] = None,
        **kwargs
    ) -> str:
        """
        Generate text token by token with streaming.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_k: Top-k filtering parameter
            top_p: Top-p (nucleus) filtering parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            callback: Function to call with each new token
            **kwargs: Additional arguments for generation
            
        Returns:
            str: Generated text
        """
        from .sampling import sample_next_token
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        past_key_values = None
        
        # Initialize generation
        generated_tokens = []
        next_token_logits_history = []
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                # Update key-value cache
                past_key_values = outputs.get("past_key_values") if isinstance(outputs, dict) else outputs[1]
                
                # Get logits for next token
                next_token_logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]
                next_token_logits = next_token_logits[:, -1, :]
                next_token_logits_history.append(next_token_logits)
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(input_ids[0].tolist() + generated_tokens):
                        next_token_logits[:, token_id] /= repetition_penalty
                        
                # Sample next token
                next_token, next_token_prob = sample_next_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                )
                
                # Append to generated tokens
                generated_tokens.append(next_token.item())
                
                # Decode the new token
                new_token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=False)
                
                # Call callback if provided
                if callback:
                    callback(new_token_text, len(generated_tokens), next_token_prob)
                    
                # Update input_ids and attention_mask
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=1)
                
                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
        # Decode the full sequence
        full_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return full_text
        
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in batches.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of new tokens to generate per prompt
            temperature: Temperature for sampling
            top_k: Top-k filtering parameter
            top_p: Top-p (nucleus) filtering parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            batch_size: Batch size for generation
            **kwargs: Additional arguments for generation
            
        Returns:
            List[str]: Generated texts for each prompt
        """
        # Process prompts in batches
        all_results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Encode prompts
            encodings = self.tokenizer.batch_encode_plus(
                batch_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=self.max_length - max_new_tokens,
            )
            
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            
            # Set up generation parameters
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0,
                "eos_token_id": self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else None,
                "attention_mask": attention_mask,
                "use_cache": self.use_kv_cache,
            }
            
            # Update with any additional generation parameters
            gen_kwargs.update(kwargs)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(input_ids, **gen_kwargs)
                
                # Decode generated sequences
                for j, output in enumerate(outputs):
                    # Get number of input tokens for this prompt
                    input_length = len(self.tokenizer.encode(batch_prompts[j]))
                    
                    # Extract generated tokens
                    generated_tokens = output[input_length:]
                    
                    # Decode to text
                    decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    all_results.append(decoded_text)
                    
        return all_results
        
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer=None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> "ConsynInferenceEngine":
        """
        Create an inference engine from a pretrained model.
        
        Args:
            model_path: Path to the pretrained model
            tokenizer: Tokenizer for the model (if None, will be loaded from model_path)
            device: Device to run inference on
            **kwargs: Additional arguments for the inference engine
            
        Returns:
            ConsynInferenceEngine: Initialized inference engine
        """
        # Load the model
        try:
            # Try loading with Hugging Face transformers
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                # Load the model
                model = AutoModelForCausalLM.from_pretrained(model_path)
                
                # Load the tokenizer if not provided
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
                logging.info(f"Loaded model and tokenizer from {model_path} using transformers")
                
            except (ImportError, ValueError):
                # Fallback to loading with PyTorch
                logging.info(f"Loading model with PyTorch from {model_path}")
                
                # Get model configuration
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    
                    from ..model.config import ConsynConfig
                    config = ConsynConfig.from_dict(config_dict)
                    
                    # Initialize model with config
                    from ..model.architecture import ConsynLMHeadModel
                    model = ConsynLMHeadModel(config)
                else:
                    # Try to infer model class from directory name
                    model_name = os.path.basename(model_path.rstrip("/"))
                    if "verse" in model_name.lower():
                        from ..model.config import ConsynVerseConfig
                        config = ConsynVerseConfig()
                    elif "stanza" in model_name.lower():
                        from ..model.config import ConsynStanzaConfig
                        config = ConsynStanzaConfig()
                    elif "epic" in model_name.lower():
                        from ..model.config import ConsynEpicConfig
                        config = ConsynEpicConfig()
                    else:
                        from ..model.config import ConsynConfig
                        config = ConsynConfig()
                        
                    # Initialize model with config
                    from ..model.architecture import ConsynLMHeadModel
                    model = ConsynLMHeadModel(config)
                    
                # Load model weights
                model_weights_path = os.path.join(model_path, "pytorch_model.bin")
                if os.path.exists(model_weights_path):
                    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
                else:
                    raise FileNotFoundError(f"Model weights not found at {model_weights_path}")
                    
                # Load the tokenizer if not provided
                if tokenizer is None:
                    # Try loading BPE tokenizer
                    from ..tokenization.bpe import BPETokenizer
                    tokenizer_path = os.path.join(model_path, "tokenizer")
                    if os.path.exists(tokenizer_path):
                        tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
                    else:
                        # Try loading SentencePiece tokenizer
                        from ..tokenization.sentencepiece_wrapper import SentencePieceTokenizer
                        spiece_path = os.path.join(model_path, "spiece.model")
                        if os.path.exists(spiece_path):
                            tokenizer = SentencePieceTokenizer(model_file=spiece_path)
                        else:
                            raise FileNotFoundError(f"Tokenizer not found at {model_path}")
                
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
            
        # Create inference engine
        return cls(model, tokenizer, device=device, **kwargs)


def get_module_by_name(model, name):
    """
    Get a specific module in a model by its name.
    
    Args:
        model: PyTorch model
        name: Module name with dot notation (e.g., "encoder.layer.0")
        
    Returns:
        nn.Module: The requested module, or None if not found
    """
    names = name.split(".")
    module = model
    
    for name in names:
        if not hasattr(module, name):
            return None
        module = getattr(module, name)
        
    return module