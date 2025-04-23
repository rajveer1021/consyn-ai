# consyn/inference/onnx_export.py
"""
ONNX export utilities for Consyn AI models.
This module provides tools for exporting models to ONNX format for deployment.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import numpy as np


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int] = (1, 16),
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 14,
    optimize: bool = True,
) -> str:
    """
    Export a model to ONNX format.
    
    Args:
        model: Model to export
        output_path: Path to save the ONNX model
        input_shape: Shape of the input tensor (batch_size, sequence_length)
        dynamic_axes: Dynamic axes configuration for variable-length inputs
        opset_version: ONNX opset version
        optimize: Whether to optimize the model after export
        
    Returns:
        str: Path to the exported ONNX model
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.ones(input_shape, dtype=torch.long, device=next(model.parameters()).device)
    
    # Set default dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        }
        
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define forward wrapper if needed
    # This is to handle models with complex output structures
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids, attention_mask=None):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                return outputs[0]
            elif isinstance(outputs, dict) and "logits" in outputs:
                return outputs["logits"]
            elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
                return outputs["last_hidden_state"]
            else:
                return outputs
                
    # Wrap model if needed
    if hasattr(model, "generate"):
        wrapped_model = ModelWrapper(model)
    else:
        wrapped_model = model
        
    # Prepare inputs for export
    inputs = (dummy_input,)
    input_names = ["input_ids"]
    
    # Add attention mask if the model accepts it
    if "attention_mask" in dynamic_axes:
        attention_mask = torch.ones(input_shape, dtype=torch.long, device=next(model.parameters()).device)
        inputs = inputs + (attention_mask,)
        input_names.append("attention_mask")
        
    # Export to ONNX
    output_names = ["output"]
    
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            inputs,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False,
        )
        
    logging.info(f"Exported model to {output_path}")
    
    # Optimize model if requested
    if optimize:
        optimize_onnx_model(output_path)
        
    return output_path


def export_tokenizer_for_onnx(
    tokenizer,
    output_dir: str,
) -> str:
    """
    Export tokenizer configuration for use with ONNX models.
    
    Args:
        tokenizer: Tokenizer to export
        output_dir: Directory to save tokenizer configuration
        
    Returns:
        str: Path to the exported tokenizer configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get tokenizer configuration
    config = {
        "vocab_size": tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer.encoder),
        "bos_token_id": tokenizer.bos_token_id if hasattr(tokenizer, "bos_token_id") else None,
        "eos_token_id": tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
        "pad_token_id": tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else None,
        "unk_token_id": tokenizer.unk_token_id if hasattr(tokenizer, "unk_token_id") else None,
    }
    
    # Save tokenizer configuration
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        
    # Save tokenizer vocabulary
    if hasattr(tokenizer, "encoder"):
        # BPE-style tokenizer
        vocab_path = os.path.join(output_dir, "vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(tokenizer.encoder, f, indent=4)
            
    # Save tokenizer merges if available
    if hasattr(tokenizer, "bpe_ranks"):
        merges_path = os.path.join(output_dir, "merges.txt")
        with open(merges_path, "w") as f:
            for merge, _ in sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1]):
                f.write(" ".join(merge) + "\n")
                
    # Try to save the full tokenizer if it uses Hugging Face's PreTrainedTokenizer
    try:
        tokenizer.save_pretrained(output_dir)
        logging.info(f"Saved full tokenizer to {output_dir}")
    except (AttributeError, TypeError):
        logging.info(f"Saved basic tokenizer configuration to {output_dir}")
        
    return output_dir


def optimize_onnx_model(model_path: str) -> str:
    """
    Optimize an ONNX model for inference.
    
    Args:
        model_path: Path to the ONNX model
        
    Returns:
        str: Path to the optimized model
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer
        
        # Load the model
        onnx_model = onnx.load(model_path)
        
        # Create optimizer
        model_optimizer = optimizer.get_optimizer(onnx_model)
        
        # Apply optimizations
        optimized_model = model_optimizer.optimize()
        
        # Save optimized model
        optimized_path = model_path.replace(".onnx", "_optimized.onnx")
        onnx.save(optimized_model, optimized_path)
        
        logging.info(f"Optimized model saved to {optimized_path}")
        return optimized_path
        
    except ImportError:
        logging.warning("onnxruntime.transformers not available for optimization")
        return model_path


def export_for_tensorrt(
    model: nn.Module,
    output_dir: str,
    input_shape: Tuple[int, int] = (1, 16),
    fp16: bool = True,
    optimize: bool = True,
) -> str:
    """
    Export a model for TensorRT.
    
    Args:
        model: Model to export
        output_dir: Directory to save the exported model
        input_shape: Shape of the input tensor (batch_size, sequence_length)
        fp16: Whether to use FP16 precision
        optimize: Whether to optimize the ONNX model
        
    Returns:
        str: Path to the exported model
    """
    # Export to ONNX first
    onnx_path = os.path.join(output_dir, "model.onnx")
    export_to_onnx(model, onnx_path, input_shape, optimize=optimize)
    
    try:
        import tensorrt as trt
        
        # Create TensorRT builder
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        
        # Create network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Create parser
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logging.error(f"TensorRT parser error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")
                
        # Create config
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1 GB
        
        # Set precision
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logging.info("Using FP16 precision for TensorRT")
            
        # Build engine
        engine = builder.build_engine(network, config)
        
        # Save engine
        engine_path = os.path.join(output_dir, "model.engine")
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
            
        logging.info(f"Exported TensorRT engine to {engine_path}")
        return engine_path
        
    except ImportError:
        logging.warning("TensorRT not available for export")
        return onnx_path


def export_for_triton(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    model_name: str = "consyn",
    input_shape: Tuple[int, int] = (1, 16),
    optimize: bool = True,
) -> str:
    """
    Export a model for Triton Inference Server.
    
    Args:
        model: Model to export
        tokenizer: Tokenizer for the model
        output_dir: Directory to save the exported model
        model_name: Name of the model in Triton
        input_shape: Shape of the input tensor (batch_size, sequence_length)
        optimize: Whether to optimize the ONNX model
        
    Returns:
        str: Path to the exported model repository
    """
    # Create Triton model repository structure
    repo_dir = os.path.join(output_dir, "triton_models")
    model_dir = os.path.join(repo_dir, model_name)
    version_dir = os.path.join(model_dir, "1")
    
    os.makedirs(version_dir, exist_ok=True)
    
    # Export to ONNX
    onnx_path = os.path.join(version_dir, "model.onnx")
    export_to_onnx(model, onnx_path, input_shape, optimize=optimize)
    
    # Export tokenizer
    tokenizer_dir = os.path.join(repo_dir, f"{model_name}_tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    export_tokenizer_for_onnx(tokenizer, tokenizer_dir)
    
    # Create config.pbtxt for the model
    config = f"""
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 64
input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1]
  }}
]
output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, {model.config.vocab_size if hasattr(model, "config") else 50257}]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
    """
    
    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config)
        
    # Create config.pbtxt for the tokenizer
    tokenizer_config = f"""
name: "{model_name}_tokenizer"
backend: "python"
max_batch_size: 64
input [
  {{
    name: "text"
    data_type: TYPE_STRING
    dims: [1]
  }}
]
output [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
    """
    
    tokenizer_config_path = os.path.join(tokenizer_dir, "config.pbtxt")
    with open(tokenizer_config_path, "w") as f:
        f.write(tokenizer_config)
        
    # Create Python backend for tokenizer
    tokenizer_backend_dir = os.path.join(tokenizer_dir, "1")
    os.makedirs(tokenizer_backend_dir, exist_ok=True)
    
    tokenizer_model_code = f"""
import json
import os
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        
        # Load tokenizer config
        with open(os.path.join(self.model_dir, "tokenizer_config.json"), "r") as f:
            self.config = json.load(f)
            
        # Load vocabulary
        with open(os.path.join(self.model_dir, "vocab.json"), "r") as f:
            self.encoder = json.load(f)
            
        self.decoder = {{v: k for k, v in self.encoder.items()}}
        
        # Load merges if available
        self.bpe_ranks = {{}}
        merges_path = os.path.join(self.model_dir, "merges.txt")
        if os.path.exists(merges_path):
            with open(merges_path, "r") as f:
                merges = f.read().split("\\n")
                merges = [tuple(merge.split()) for merge in merges if merge]
                self.bpe_ranks = dict(zip(merges, range(len(merges))))
                
    def execute(self, requests):
        responses = []
        
        for request in requests:
            # Get input text
            text = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()[0][0]
            text = text.decode("utf-8")
            
            # Tokenize text
            tokens = self.tokenize(text)
            
            # Create output tensor
            output_tensor = pb_utils.Tensor("input_ids", np.array([tokens], dtype=np.int64))
            
            # Create response
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
            
        return responses
        
    def tokenize(self, text):
        # This is a simplified tokenization logic
        # In a real implementation, this would use the actual tokenizer logic
        
        # For demonstration purposes, we'll just return some dummy tokens
        # This should be replaced with actual tokenization logic
        return [self.encoder.get(t, self.encoder.get("<unk>", 0)) for t in text.split()]
    """
    
    tokenizer_model_path = os.path.join(tokenizer_backend_dir, "model.py")
    with open(tokenizer_model_path, "w") as f:
        f.write(tokenizer_model_code)
        
    logging.info(f"Exported model for Triton Inference Server to {repo_dir}")
    return repo_dir
