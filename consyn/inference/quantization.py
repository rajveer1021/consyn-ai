# consyn/inference/quantization.py
"""
Quantization utilities for Consyn AI models.
This module provides tools for post-training quantization to improve inference efficiency.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn as nn


def quantize_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    target_modules: List[nn.Module] = [nn.Linear],
) -> nn.Module:
    """
    Apply dynamic quantization to a model.
    
    Args:
        model: Model to quantize
        dtype: Quantization data type
        target_modules: List of module types to quantize
        
    Returns:
        nn.Module: Quantized model
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Create a module map for quantization
    module_map = {nn.Linear: torch.quantization.quantize_dynamic}
    
    # Filter target modules
    target_modules = [m for m in target_modules if m in module_map]
    
    # Define quantization configuration
    qconfig_mapping = torch.quantization.QConfig(
        activation=torch.quantization.PlaceholderObserver.with_args(dtype=dtype),
        weight=torch.quantization.PlaceholderObserver.with_args(dtype=dtype),
    )
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {m.__name__: qconfig_mapping for m in target_modules},
        dtype=dtype
    )
    
    logging.info(f"Applied dynamic quantization with dtype {dtype}")
    return quantized_model


def quantize_static(
    model: nn.Module,
    calibration_data_loader: torch.utils.data.DataLoader,
    dtype: torch.dtype = torch.qint8,
    target_modules: List[nn.Module] = [nn.Linear, nn.Conv2d],
) -> nn.Module:
    """
    Apply static quantization to a model using calibration data.
    
    Args:
        model: Model to quantize
        calibration_data_loader: DataLoader with calibration data
        dtype: Quantization data type
        target_modules: List of module types to quantize
        
    Returns:
        nn.Module: Quantized model
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Define quantization configuration
    qconfig = torch.quantization.get_default_qconfig("fbgemm")
    
    # Create quantization configuration mapping
    qconfig_mapping = torch.quantization.QConfigMapping()
    
    # Add target modules to mapping
    for module_type in target_modules:
        qconfig_mapping.register_module_instances(module_type, qconfig)
        
    # Prepare model for static quantization
    prepared_model = torch.quantization.prepare(model, qconfig_mapping)
    
    # Calibrate using data
    with torch.no_grad():
        for batch in calibration_data_loader:
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                prepared_model(**batch)
            else:
                batch = batch.to(model.device)
                prepared_model(batch)
                
    # Convert model to quantized version
    quantized_model = torch.quantization.convert(prepared_model)
    
    logging.info(f"Applied static quantization with dtype {dtype}")
    return quantized_model


def quantize_weights_only(
    model: nn.Module,
    bits: int = 8,
    groupsize: int = 128,
    sym: bool = True,
) -> nn.Module:
    """
    Apply weight-only quantization to a model.
    
    Args:
        model: Model to quantize
        bits: Number of bits for quantization
        groupsize: Size of quantization groups
        sym: Whether to use symmetric quantization
        
    Returns:
        nn.Module: Model with quantized weights
    """
    # Check if we can use optimum for quantization
    try:
        from optimum.gptq import GPTQQuantizer
        
        # Create quantizer
        quantizer = GPTQQuantizer(
            bits=bits,
            group_size=groupsize,
            sym=sym,
        )
        
        # Quantize model
        quantized_model = quantizer.quantize_model(model)
        
        logging.info(f"Applied weight-only quantization with {bits} bits")
        return quantized_model
        
    except ImportError:
        logging.warning("optimum.gptq not available, falling back to custom implementation")
        
        # Use simple custom quantization for weights
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Quantize weights
                weight = module.weight.data
                
                # Calculate scale and zero point
                if sym:
                    # Symmetric quantization
                    max_val = weight.abs().max()
                    scale = max_val / (2 ** (bits - 1) - 1)
                    zero_point = 0
                else:
                    # Asymmetric quantization
                    min_val = weight.min()
                    max_val = weight.max()
                    scale = (max_val - min_val) / (2 ** bits - 1)
                    zero_point = torch.round(-min_val / scale).to(torch.int)
                    
                # Apply quantization per group
                if groupsize > 0:
                    # Apply group-wise quantization
                    original_shape = weight.shape
                    weight = weight.view(-1, groupsize)
                    
                    # Calculate scales and zero points per group
                    if sym:
                        # Symmetric quantization
                        max_vals = weight.abs().max(dim=1, keepdim=True)[0]
                        scales = max_vals / (2 ** (bits - 1) - 1)
                        zero_points = torch.zeros_like(scales, dtype=torch.int)
                    else:
                        # Asymmetric quantization
                        min_vals = weight.min(dim=1, keepdim=True)[0]
                        max_vals = weight.max(dim=1, keepdim=True)[0]
                        scales = (max_vals - min_vals) / (2 ** bits - 1)
                        zero_points = torch.round(-min_vals / scales).to(torch.int)
                        
                    # Quantize
                    weight_q = torch.round(weight / scales + zero_points).to(torch.int)
                    
                    # Clamp values to quantization range
                    if sym:
                        weight_q = torch.clamp(weight_q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
                    else:
                        weight_q = torch.clamp(weight_q, 0, 2 ** bits - 1)
                        
                    # Dequantize for inference
                    weight_dq = (weight_q - zero_points) * scales
                    
                    # Restore original shape
                    weight_dq = weight_dq.view(original_shape)
                    
                    # Update module weight
                    module.weight.data = weight_dq
                else:
                    # Apply tensor-wise quantization
                    weight_q = torch.round(weight / scale + zero_point).to(torch.int)
                    
                    # Clamp values to quantization range
                    if sym:
                        weight_q = torch.clamp(weight_q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
                    else:
                        weight_q = torch.clamp(weight_q, 0, 2 ** bits - 1)
                        
                    # Dequantize for inference
                    weight_dq = (weight_q - zero_point) * scale
                    
                    # Update module weight
                    module.weight.data = weight_dq
                    
        logging.info(f"Applied custom weight-only quantization with {bits} bits")
        return model


def quantize_with_bitsandbytes(
    model: nn.Module,
    bits: int = 8,
) -> nn.Module:
    """
    Quantize model using bitsandbytes library.
    
    Args:
        model: Model to quantize
        bits: Number of bits for quantization (4 or 8)
        
    Returns:
        nn.Module: Quantized model
    """
    try:
        import bitsandbytes as bnb
        
        # Map bit width to corresponding quantization module
        if bits == 8:
            linear_class = bnb.nn.Linear8bitLt
        elif bits == 4:
            linear_class = bnb.nn.Linear4bit
        else:
            raise ValueError(f"Unsupported bits value: {bits}. Only 4 and 8 are supported.")
            
        # Replace linear layers with quantized versions
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get parent module
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                parent = model if parent_name == "" else get_module_by_name(model, parent_name)
                
                # Get attribute name
                attr_name = name.rsplit(".", 1)[1] if "." in name else name
                
                # Replace with quantized linear layer
                quantized_module = linear_class(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    threshold=6.0,  # Threshold for outlier detection
                    compute_dtype=torch.float16,  # Use half precision for compute
                )
                
                # Copy weights and bias
                quantized_module.weight.data = module.weight.data
                if module.bias is not None:
                    quantized_module.bias.data = module.bias.data
                    
                # Replace module
                setattr(parent, attr_name, quantized_module)
                
        logging.info(f"Applied {bits}-bit quantization using bitsandbytes")
        return model
        
    except ImportError:
        logging.warning("bitsandbytes not available for quantization")
        return model


def quantize_with_awq(
    model: nn.Module,
    bits: int = 4,
    groupsize: int = 128,
    calibration_data_loader: Optional[torch.utils.data.DataLoader] = None,
) -> nn.Module:
    """
    Quantize model using AWQ (Activation-aware Weight Quantization).
    
    Args:
        model: Model to quantize
        bits: Number of bits for quantization
        groupsize: Size of quantization groups
        calibration_data_loader: DataLoader with calibration data
        
    Returns:
        nn.Module: Quantized model
    """
    try:
        from awq import AutoAWQForCausalLM
        
        # Create AWQ model wrapper
        awq_model = AutoAWQForCausalLM.from_pretrained(model)
        
        # Prepare calibration data
        if calibration_data_loader is not None:
            calibration_data = []
            for batch in calibration_data_loader:
                if isinstance(batch, dict) and "input_ids" in batch:
                    calibration_data.append(batch["input_ids"])
                elif isinstance(batch, torch.Tensor):
                    calibration_data.append(batch)
                    
            calibration_data = torch.cat(calibration_data, dim=0)
        else:
            # Use random data if no calibration data provided
            calibration_data = torch.randint(
                0, model.config.vocab_size, (8, 512), device=model.device
            )
            
        # Quantize model
        quantized_model = awq_model.quantize(
            bits=bits,
            group_size=groupsize,
            zero_point=True,
            calibration_data=calibration_data,
        )
        
        logging.info(f"Applied {bits}-bit AWQ quantization")
        return quantized_model
        
    except ImportError:
        logging.warning("AWQ not available for quantization")
        return model


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


def quantize_model(
    model: nn.Module,
    method: str = "dynamic",
    bits: int = 8,
    groupsize: int = 128,
    calibration_data_loader: Optional[torch.utils.data.DataLoader] = None,
) -> nn.Module:
    """
    Quantize a model using the specified method.
    
    Args:
        model: Model to quantize
        method: Quantization method (dynamic, static, weight_only, bnb, or awq)
        bits: Number of bits for quantization
        groupsize: Size of quantization groups
        calibration_data_loader: DataLoader with calibration data
        
    Returns:
        nn.Module: Quantized model
        
    Raises:
        ValueError: If the quantization method is not supported
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Apply quantization based on method
    if method == "dynamic":
        return quantize_dynamic(model, torch.qint8)
    elif method == "static":
        if calibration_data_loader is None:
            raise ValueError("Calibration data loader is required for static quantization")
        return quantize_static(model, calibration_data_loader)
    elif method == "weight_only":
        return quantize_weights_only(model, bits, groupsize)
    elif method == "bnb":
        return quantize_with_bitsandbytes(model, bits)
    elif method == "awq":
        return quantize_with_awq(model, bits, groupsize, calibration_data_loader)
    else:
        raise ValueError(f"Unsupported quantization method: {method}")


def check_quantization_compatibility(model: nn.Module) -> Dict[str, bool]:
    """
    Check compatibility of the model with different quantization methods.
    
    Args:
        model: Model to check
        
    Returns:
        dict: Dictionary of compatibility results
    """
    compatibility = {}
    
    # Check for PyTorch dynamic quantization
    compatibility["dynamic"] = True
    
    # Check for PyTorch static quantization
    # Static quantization requires modules to have qconfig attributes
    compatibility["static"] = True
    
    # Check for weight-only quantization
    # Weight-only quantization should work for most models
    compatibility["weight_only"] = True
    
    # Check for bitsandbytes
    try:
        import bitsandbytes as bnb
        compatibility["bnb"] = True
    except ImportError:
        compatibility["bnb"] = False
        
    # Check for AWQ
    try:
        from awq import AutoAWQForCausalLM
        compatibility["awq"] = True
    except ImportError:
        compatibility["awq"] = False
        
    # Check if model has attention layers (for optimizations)
    has_attention = False
    for name, module in model.named_modules():
        if "attention" in name.lower() or "attn" in name.lower():
            has_attention = True
            break
            
    compatibility["has_attention"] = has_attention
    
    return compatibility
