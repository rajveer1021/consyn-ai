# consyn/scripts/convert_checkpoint.py
"""
Checkpoint conversion script for Consyn AI models.
This script provides utilities for converting models between formats and optimizing for inference.
"""

import os
import argparse
import logging
import json
import torch
from typing import Dict, Optional

from consyn.model import ConsynLMHeadModel, ConsynConfig
from consyn.tokenization import BPETokenizer, SentencePieceTokenizer
from consyn.inference import ConsynInferenceEngine
from consyn.inference.quantization import quantize_model
from consyn.inference.onnx_export import export_to_onnx, export_tokenizer_for_onnx, export_for_triton

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert Consyn AI model checkpoints")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save converted model",
    )
    
    # Conversion options
    parser.add_argument(
        "--format",
        type=str,
        default="pytorch",
        choices=["pytorch", "onnx", "tensorrt", "triton", "huggingface"],
        help="Output format for the model",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[None, "int8", "int4", "bnb", "awq"],
        help="Quantization method to apply",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Whether to optimize the model for inference",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run conversion on (cpu, cuda, cuda:0, etc.)",
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, device=None):
    """Load model and tokenizer from path."""
    logger.info(f"Loading model from {model_path}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try to load with Hugging Face transformers
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Move model to device
        model = model.to(device)
        
        return model, tokenizer
    except (ImportError, ValueError):
        # Load model with PyTorch
        from consyn.model.config import ConsynConfig
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            
            config = ConsynConfig.from_dict(config_dict)
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        # Create model with config
        model = ConsynLMHeadModel(config)
        
        # Load model weights
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
            
        # Move model to device
        model = model.to(device)
        
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
                raise FileNotFoundError(f"Tokenizer not found at {model_path}")
                
        return model, tokenizer


def optimize_model_for_inference(model):
    """Optimize model for inference."""
    logger.info("Optimizing model for inference")
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable dropout
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            
    # Enable key-value cache
    if hasattr(model, "config"):
        model.config.use_cache = True
        
    # Disable gradient checkpointing
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = False
        
    # Try to optimize with torch.jit if possible
    try:
        # Create dummy input
        batch_size = 1
        seq_len = 16
        dummy_input = torch.randint(
            0, model.config.vocab_size, (batch_size, seq_len),
            device=next(model.parameters()).device
        )
        
        # Trace model
        traced_model = torch.jit.trace(model, [dummy_input])
        logger.info("Successfully created traced model with torch.jit")
        
        return traced_model
    except Exception as e:
        logger.warning(f"Failed to create traced model: {e}")
        return model


def convert_to_huggingface(model, tokenizer, output_path):
    """Convert model to Hugging Face format."""
    logger.info("Converting model to Hugging Face format")
    
    try:
        from transformers import PreTrainedModel, PreTrainedTokenizer
        
        # Check if model and tokenizer are already Hugging Face-compatible
        is_hf_model = isinstance(model, PreTrainedModel)
        is_hf_tokenizer = isinstance(tokenizer, PreTrainedTokenizer)
        
        if is_hf_model and is_hf_tokenizer:
            # Save directly using the save_pretrained method
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            logger.info(f"Saved model and tokenizer to {output_path}")
            return
            
        # If not Hugging Face models, convert them
        logger.info("Converting native Consyn model to Hugging Face format")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save model config
        if hasattr(model, "config"):
            config_dict = model.config.to_dict()
            
            # Add Hugging Face-specific fields
            config_dict["model_type"] = "consyn"
            if hasattr(model.config, "hidden_size"):
                if model.config.hidden_size <= 512:
                    config_dict["architectures"] = ["ConsynForCausalLM"]
                elif model.config.hidden_size <= 1536:
                    config_dict["architectures"] = ["ConsynLMHeadModel"]
                else:
                    config_dict["architectures"] = ["ConsynLMHeadModel"]
                    
            # Save config
            with open(os.path.join(output_path, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
                
        # Save model weights
        torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        
        # Save tokenizer
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(os.path.join(output_path, "tokenizer"))
        else:
            # Manual tokenizer saving
            if hasattr(tokenizer, "encoder"):
                # Save vocab
                with open(os.path.join(output_path, "vocab.json"), "w") as f:
                    json.dump(tokenizer.encoder, f, indent=2)
                    
            if hasattr(tokenizer, "bpe_ranks"):
                # Save merges
                with open(os.path.join(output_path, "merges.txt"), "w") as f:
                    for merge, _ in sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1]):
                        f.write(" ".join(merge) + "\n")
                        
            # Save tokenizer config
            tokenizer_config = {
                "model_type": "consyn",
                "bos_token": tokenizer.bos_token if hasattr(tokenizer, "bos_token") else "<s>",
                "eos_token": tokenizer.eos_token if hasattr(tokenizer, "eos_token") else "</s>",
                "pad_token": tokenizer.pad_token if hasattr(tokenizer, "pad_token") else "<pad>",
                "unk_token": tokenizer.unk_token if hasattr(tokenizer, "unk_token") else "<unk>",
            }
            
            with open(os.path.join(output_path, "tokenizer_config.json"), "w") as f:
                json.dump(tokenizer_config, f, indent=2)
                
        logger.info(f"Saved model and tokenizer to {output_path}")
        
    except ImportError:
        logger.warning("transformers package not available. Saving in native format.")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save model config
        if hasattr(model, "config"):
            config_dict = model.config.to_dict()
            
            # Save config
            with open(os.path.join(output_path, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
                
        # Save model weights
        torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        
        # Save tokenizer
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_path)
        else:
            # Manual tokenizer saving
            if hasattr(tokenizer, "encoder"):
                # Save vocab
                with open(os.path.join(output_path, "vocab.json"), "w") as f:
                    json.dump(tokenizer.encoder, f, indent=2)
                    
            if hasattr(tokenizer, "bpe_ranks"):
                # Save merges
                with open(os.path.join(output_path, "merges.txt"), "w") as f:
                    for merge, _ in sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1]):
                        f.write(" ".join(merge) + "\n")
                        
        logger.info(f"Saved model and tokenizer to {output_path}")


def main():
    """Main conversion function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Apply optimization if requested
    if args.optimize:
        model = optimize_model_for_inference(model)
        
    # Apply quantization if requested
    if args.quantization is not None:
        logger.info(f"Applying {args.quantization} quantization")
        model = quantize_model(model, method=args.quantization)
        
    # Convert model to the requested format
    if args.format == "pytorch":
        # Save PyTorch model
        logger.info("Saving PyTorch model")
        
        # Save model config
        if hasattr(model, "config"):
            config_dict = model.config.to_dict()
            
            # Save config
            with open(os.path.join(args.output_path, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
                
        # Save model weights
        torch.save(model.state_dict(), os.path.join(args.output_path, "pytorch_model.bin"))
        
        # Save tokenizer
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(args.output_path)
        else:
            # Manual tokenizer saving
            if hasattr(tokenizer, "encoder"):
                # Save vocab
                with open(os.path.join(args.output_path, "vocab.json"), "w") as f:
                    json.dump(tokenizer.encoder, f, indent=2)
                    
            if hasattr(tokenizer, "bpe_ranks"):
                # Save merges
                with open(os.path.join(args.output_path, "merges.txt"), "w") as f:
                    for merge, _ in sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1]):
                        f.write(" ".join(merge) + "\n")
                        
        logger.info(f"Saved PyTorch model to {args.output_path}")
        
    elif args.format == "onnx":
        # Export to ONNX
        logger.info("Exporting to ONNX format")
        
        # Create inference engine
        engine = ConsynInferenceEngine(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        
        # Export model
        onnx_path = os.path.join(args.output_path, "model.onnx")
        export_to_onnx(
            model=model,
            output_path=onnx_path,
            optimize=args.optimize,
        )
        
        # Export tokenizer
        export_tokenizer_for_onnx(
            tokenizer=tokenizer,
            output_dir=args.output_path,
        )
        
        logger.info(f"Exported ONNX model to {args.output_path}")
        
    elif args.format == "tensorrt":
        # Export for TensorRT
        logger.info("Exporting for TensorRT")
        
        try:
            import tensorrt as trt
            
            # Create inference engine
            engine = ConsynInferenceEngine(
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            
            # First export to ONNX
            onnx_path = os.path.join(args.output_path, "model.onnx")
            export_to_onnx(
                model=model,
                output_path=onnx_path,
                optimize=args.optimize,
            )
            
            # Export tokenizer
            export_tokenizer_for_onnx(
                tokenizer=tokenizer,
                output_dir=args.output_path,
            )
            
            # Convert to TensorRT
            logger.info("Converting ONNX to TensorRT")
            
            # Create TensorRT builder
            logger_trt = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger_trt)
            
            # Create network
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Create parser
            parser = trt.OnnxParser(network, logger_trt)
            
            # Parse ONNX model
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"TensorRT parser error: {parser.get_error(error)}")
                    raise RuntimeError("Failed to parse ONNX model")
                    
            # Create config
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1 GB
            
            # Set precision
            if args.quantization == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("Using INT8 precision for TensorRT")
            elif args.device is not None and "cuda" in args.device and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Using FP16 precision for TensorRT")
                
            # Build engine
            engine_trt = builder.build_engine(network, config)
            
            # Save engine
            engine_path = os.path.join(args.output_path, "model.engine")
            with open(engine_path, "wb") as f:
                f.write(engine_trt.serialize())
                
            logger.info(f"Exported TensorRT engine to {engine_path}")
            
        except ImportError:
            logger.error("TensorRT not available. Please install TensorRT.")
            
    elif args.format == "triton":
        # Export for Triton Inference Server
        logger.info("Exporting for Triton Inference Server")
        
        # Create inference engine
        engine = ConsynInferenceEngine(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        
        # Export model
        export_for_triton(
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_path,
            model_name="consyn",
            optimize=args.optimize,
        )
        
        logger.info(f"Exported model for Triton Inference Server to {args.output_path}")
        
    elif args.format == "huggingface":
        # Convert to Hugging Face format
        convert_to_huggingface(model, tokenizer, args.output_path)
        
    else:
        logger.error(f"Unsupported format: {args.format}")


if __name__ == "__main__":
    main()