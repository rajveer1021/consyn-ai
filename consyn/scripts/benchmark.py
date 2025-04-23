# consyn/scripts/benchmark.py

"""
Benchmark script for Consyn AI models.
This script provides performance benchmarking for inference and training.
"""

import os
import argparse
import logging
import json
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from consyn.model import (
    ConsynConfig,
    ConsynVerseConfig,
    ConsynStanzaConfig,
    ConsynEpicConfig,
    ConsynLMHeadModel,
)
from consyn.tokenization import BPETokenizer
from consyn.inference import ConsynInferenceEngine
from consyn.inference.quantization import quantize_model

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark Consyn AI models")
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="verse",
        choices=["verse", "stanza", "epic"],
        help="Model to benchmark (verse, stanza, or epic)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model directory (overrides --model)",
    )
    
    # Benchmark type
    parser.add_argument(
        "--type",
        type=str,
        default="inference",
        choices=["inference", "training", "both"],
        help="Type of benchmark to run",
    )
    
    # Inference benchmark arguments
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list of batch sizes for inference",
    )
    parser.add_argument(
        "--sequence_lengths",
        type=str,
        default="128,256,512,1024",
        help="Comma-separated list of sequence lengths for inference",
    )
    parser.add_argument(
        "--generation_lengths",
        type=str,
        default="16,32,64,128",
        help="Comma-separated list of generation lengths for inference",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs for each benchmark configuration",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs before timing",
    )
    
    # Optimization arguments
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[None, "int8", "int4", "bnb", "weight_only"],
        help="Quantization method to apply",
    )
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use key-value cache for generation",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Whether to use flash attention if available",
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File to save benchmark results",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Whether to output results as JSON",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run benchmark on (cpu, cuda, cuda:0, etc.)",
    )
    
    return parser.parse_args()


def setup_model(args):
    """Set up the model for benchmarking."""
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    logger.info(f"Using device: {device}")
    
    # Determine model path
    if args.model_path is None:
        # Create model based on model name
        if args.model == "verse":
            config = ConsynVerseConfig()
        elif args.model == "stanza":
            config = ConsynStanzaConfig()
        elif args.model == "epic":
            config = ConsynEpicConfig()
        else:
            raise ValueError(f"Unsupported model: {args.model}")
            
        model = ConsynLMHeadModel(config)
        tokenizer = BPETokenizer()
        
    else:
        # Load model from path
        logger.info(f"Loading model from {args.model_path}")
        
        # Try to load with Hugging Face transformers
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(args.model_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            
        except (ImportError, ValueError):
            # Load with native Consyn API
            logger.info("Loading with native Consyn API")
            
            # Load config
            config_path = os.path.join(args.model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                
                config = ConsynConfig.from_dict(config_dict)
            else:
                # Use default config based on model name
                if args.model == "verse":
                    config = ConsynVerseConfig()
                elif args.model == "stanza":
                    config = ConsynStanzaConfig()
                elif args.model == "epic":
                    config = ConsynEpicConfig()
                else:
                    raise ValueError(f"Unsupported model: {args.model}")
                    
            # Create model with config
            model = ConsynLMHeadModel(config)
            
            # Load model weights if available
            weights_path = os.path.join(args.model_path, "pytorch_model.bin")
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            else:
                logger.warning(f"Model weights not found at {weights_path}")
                
            # Load tokenizer
            from consyn.tokenization.bpe import BPETokenizer
            from consyn.tokenization.sentencepiece_wrapper import SentencePieceTokenizer
            
            tokenizer_path = os.path.join(args.model_path, "tokenizer")
            if os.path.exists(tokenizer_path):
                tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
            else:
                # Try loading SentencePiece tokenizer
                spiece_path = os.path.join(args.model_path, "spiece.model")
                if os.path.exists(spiece_path):
                    tokenizer = SentencePieceTokenizer(model_file=spiece_path)
                else:
                    # Use default tokenizer
                    tokenizer = BPETokenizer()
                    
    # Apply quantization if requested
    if args.quantization is not None:
        logger.info(f"Applying {args.quantization} quantization")
        model = quantize_model(model, method=args.quantization)
        
    # Move model to device
    model = model.to(device)
    
    # Create inference engine
    engine = ConsynInferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_kv_cache=args.use_kv_cache,
        use_flash_attention=args.use_flash_attention,
    )
    
    return engine


def run_inference_benchmark(engine, args):
    """Run inference benchmark."""
    # Parse benchmark parameters
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    sequence_lengths = [int(sl) for sl in args.sequence_lengths.split(",")]
    generation_lengths = [int(gl) for gl in args.generation_lengths.split(",")]
    
    # Initialize results
    results = {
        "model": args.model,
        "device": str(engine.device),
        "quantization": args.quantization,
        "use_kv_cache": args.use_kv_cache,
        "use_flash_attention": args.use_flash_attention,
        "forward_pass": {},
        "text_generation": {},
    }
    
    # Run forward pass benchmark
    logger.info("Running forward pass benchmark")
    
    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            # Create benchmark key
            key = f"batch_{batch_size}_seq_{seq_len}"
            
            # Create random input
            input_ids = torch.randint(
                0, engine.model.config.vocab_size, (batch_size, seq_len),
                device=engine.device
            )
            
            # Warmup
            logger.info(f"Warming up for batch_size={batch_size}, seq_len={seq_len}")
            for _ in range(args.warmup):
                with torch.no_grad():
                    engine.model(input_ids)
                    
            # Benchmark runs
            logger.info(f"Benchmarking for batch_size={batch_size}, seq_len={seq_len}")
            forward_times = []
            
            for run in range(args.num_runs):
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Time forward pass
                start = time.time()
                
                with torch.no_grad():
                    engine.model(input_ids)
                    
                # Synchronize if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                end = time.time()
                forward_times.append(end - start)
                
            # Calculate statistics
            forward_times = np.array(forward_times)
            results["forward_pass"][key] = {
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "mean_time": float(forward_times.mean()),
                "std_time": float(forward_times.std()),
                "min_time": float(forward_times.min()),
                "max_time": float(forward_times.max()),
                "tokens_per_second": float(batch_size * seq_len / forward_times.mean()),
            }
            
    # Run text generation benchmark
    logger.info("Running text generation benchmark")
    
    # Create sample prompts
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In the beginning there was",
        "Scientists have discovered a new way to",
    ]
    
    for batch_size in batch_sizes[:2]:  # Limit batch sizes for generation
        for prompt_length in [8, 16]:  # Short prompts for benchmarking
            for gen_len in generation_lengths:
                # Skip if batch size > number of prompts
                if batch_size > len(prompts):
                    continue
                    
                # Create benchmark key
                key = f"batch_{batch_size}_prompt_{prompt_length}_gen_{gen_len}"
                
                # Select prompts
                batch_prompts = prompts[:batch_size]
                
                # Warmup
                logger.info(f"Warming up for batch_size={batch_size}, gen_len={gen_len}")
                for _ in range(args.warmup):
                    engine.batch_generate(
                        prompts=batch_prompts,
                        max_new_tokens=gen_len,
                        do_sample=False,  # Deterministic for benchmarking
                    )
                    
                # Benchmark runs
                logger.info(f"Benchmarking for batch_size={batch_size}, gen_len={gen_len}")
                generation_times = []
                
                for run in range(args.num_runs):
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    # Time generation
                    start = time.time()
                    
                    engine.batch_generate(
                        prompts=batch_prompts,
                        max_new_tokens=gen_len,
                        do_sample=False,  # Deterministic for benchmarking
                    )
                    
                    # Synchronize if using CUDA
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        
                    end = time.time()
                    generation_times.append(end - start)
                    
                # Calculate statistics
                generation_times = np.array(generation_times)
                results["text_generation"][key] = {
                    "batch_size": batch_size,
                    "prompt_length": prompt_length,
                    "generation_length": gen_len,
                    "mean_time": float(generation_times.mean()),
                    "std_time": float(generation_times.std()),
                    "min_time": float(generation_times.min()),
                    "max_time": float(generation_times.max()),
                    "tokens_per_second": float(batch_size * gen_len / generation_times.mean()),
                }
                
    return results


def run_training_benchmark(engine, args):
    """Run training benchmark."""
    # Parse benchmark parameters
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    sequence_lengths = [int(sl) for sl in args.sequence_lengths.split(",")]
    
    # Initialize results
    results = {
        "model": args.model,
        "device": str(engine.device),
        "training": {},
    }
    
    # Get model and setup optimizer
    model = engine.model
    model.train()  # Set to training mode
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Run training benchmark
    logger.info("Running training benchmark")
    
    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            # Create benchmark key
            key = f"batch_{batch_size}_seq_{seq_len}"
            
            # Create random input and labels
            input_ids = torch.randint(
                0, model.config.vocab_size, (batch_size, seq_len),
                device=engine.device
            )
            labels = torch.randint(
                0, model.config.vocab_size, (batch_size, seq_len),
                device=engine.device
            )
            
            # Warmup
            logger.info(f"Warming up for batch_size={batch_size}, seq_len={seq_len}")
            for _ in range(args.warmup):
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss.backward()
                optimizer.step()
                
            # Benchmark runs
            logger.info(f"Benchmarking for batch_size={batch_size}, seq_len={seq_len}")
            training_times = []
            
            for run in range(args.num_runs):
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Time forward and backward pass
                start = time.time()
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss.backward()
                optimizer.step()
                
                # Synchronize if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                end = time.time()
                training_times.append(end - start)
                
            # Calculate statistics
            training_times = np.array(training_times)
            results["training"][key] = {
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "mean_time": float(training_times.mean()),
                "std_time": float(training_times.std()),
                "min_time": float(training_times.min()),
                "max_time": float(training_times.max()),
                "tokens_per_second": float(batch_size * seq_len / training_times.mean()),
            }
            
    return results


def print_benchmark_results(results):
    """Print benchmark results in a nice format."""
    print("\n" + "="*80)
    print(f"Benchmark Results for {results['model']} on {results['device']}")
    print("="*80)
    
    if "forward_pass" in results:
        print("\nForward Pass Benchmark:")
        print("-"*80)
        print(f"{'Batch Size':^12} | {'Seq Length':^12} | {'Mean Time (s)':^14} | {'Tokens/Second':^16}")
        print("-"*80)
        
        for key, data in sorted(results["forward_pass"].items()):
            print(f"{data['batch_size']:^12} | {data['sequence_length']:^12} | {data['mean_time']:^14.4f} | {data['tokens_per_second']:^16,.0f}")
            
    if "text_generation" in results:
        print("\nText Generation Benchmark:")
        print("-"*100)
        print(f"{'Batch Size':^12} | {'Prompt Length':^14} | {'Gen Length':^12} | {'Mean Time (s)':^14} | {'Tokens/Second':^16}")
        print("-"*100)
        
        for key, data in sorted(results["text_generation"].items()):
            print(f"{data['batch_size']:^12} | {data['prompt_length']:^14} | {data['generation_length']:^12} | {data['mean_time']:^14.4f} | {data['tokens_per_second']:^16,.0f}")
            
    if "training" in results:
        print("\nTraining Benchmark:")
        print("-"*80)
        print(f"{'Batch Size':^12} | {'Seq Length':^12} | {'Mean Time (s)':^14} | {'Tokens/Second':^16}")
        print("-"*80)
        
        for key, data in sorted(results["training"].items()):
            print(f"{data['batch_size']:^12} | {data['sequence_length']:^12} | {data['mean_time']:^14.4f} | {data['tokens_per_second']:^16,.0f}")
            
    print("\n" + "="*80)


def main():
    """Main benchmark function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set up model
    engine = setup_model(args)
    
    # Run benchmarks
    results = {}
    
    if args.type in ["inference", "both"]:
        inference_results = run_inference_benchmark(engine, args)
        results.update(inference_results)
        
    if args.type in ["training", "both"]:
        training_results = run_training_benchmark(engine, args)
        results.update(training_results)
        
    # Print results
    if not args.json:
        print_benchmark_results(results)
        
    # Save results if output file is specified
    if args.output is not None:
        logger.info(f"Saving results to {args.output}")
        
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
            
    return results


if __name__ == "__main__":
    main()