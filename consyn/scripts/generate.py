# consyn/scripts/generate.py

"""
Text generation script for Consyn AI models.
This script provides a command-line interface for generating text with models.
"""

import os
import argparse
import logging
import json
import sys
import time
from typing import List, Optional

import torch

from consyn.inference import ConsynInferenceEngine
from consyn.model import ConsynLMHeadModel, ConsynVerseConfig

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate text with Consyn AI models")
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="verse",
        choices=["verse", "stanza", "epic"],
        help="Model to use (verse, stanza, or epic)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model directory (overrides --model)",
    )
    
    # Input arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="File containing prompts (one per line)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    
    # Generation parameters
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,  # Reduced default for faster debugging
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,  # Default to 1.0 for balanced sampling
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k filtering value",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,  # Default to 0.9 for stability
        help="Top-p filtering value",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,  # Default to 1.0 to avoid penalizing valid tokens
        help="Repetition penalty",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to return",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling (default: greedy decoding)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File to save generated text",
    )
    parser.add_argument(
        "--include_prompt",
        action="store_true",
        help="Whether to include the prompt in the output",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Whether to stream the output token by token",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run generation on (cpu, cuda, cuda:0, etc.)",
    )
    
    return parser.parse_args()

def setup_model(args):
    """Set up the model and inference engine."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else torch.device(args.device)
    logger.info(f"Using device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    model_path = args.model_path or os.path.join(os.environ.get("MODEL_DIR", "./models"), f"consyn_{args.model}")
    tokenizer_dir = os.path.join(model_path, "tokenizer")

    # Load tokenizer with AutoTokenizer
    from transformers import AutoTokenizer
    try:
        if not os.path.exists(tokenizer_dir):
            raise FileNotFoundError(f"Tokenizer directory {tokenizer_dir} does not exist")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        logger.info("Successfully loaded tokenizer from local directory")
    except Exception as e:
        logger.warning(f"Could not load tokenizer from {tokenizer_dir}: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        logger.info("Using default GPT2 tokenizer")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # Load model
    try:
        # Load ConsynLMHeadModel without passing config or pad_token_id
        model = ConsynLMHeadModel.from_pretrained(model_path)
        logger.info(f"Successfully loaded ConsynLMHeadModel from {model_path}")

        # Check for meta tensors
        if any(param.is_meta for param in model.parameters()):
            logger.warning("Model contains meta tensors. Moving to device with to_empty.")
            model = model.to_empty(device=device)
            weights_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=device))
                logger.info(f"Reloaded weights from {weights_path}")
            else:
                raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
        # Log model parameters for debugging
        logger.info("Model parameters (first 5):")
        for name, param in list(model.named_parameters())[:5]:
            logger.info(f"{name}: {param.shape}, is_meta={param.is_meta}, device={param.device}")
    
    except Exception as e:
        logger.warning(f"Could not load model from {model_path}: {e}")
        logger.info("Falling back to pretrained GPT-2 model")
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.pad_token_id)

    # Move model to device
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Create inference engine
    engine = ConsynInferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_kv_cache=True,
        max_length=1024,
    )
    return engine, model, tokenizer

def generate_text(engine, model, tokenizer, args, prompt: str):
    """Generate text from a prompt."""
    logger.info(f"Generating text with prompt: {prompt}")
    
    # Ensure compatibility between parameters
    if args.temperature > 1e-5 or args.top_p < 0.9999:
        args.do_sample = True
        logger.info(f"Automatically enabled sampling due to temperature/top_p settings")
    
    start_time = time.time()
    
    try:
        # Tokenize prompt for debugging
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(engine.device)
        decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        logger.info(f"Tokenized prompt: {input_ids.tolist()}")
        logger.info(f"Decoded prompt: {decoded_prompt}")

        # Generate text with appropriate error handling
        if args.stream:
            # Stream text generation
            print(f"\nPrompt: {prompt}\nGenerated: ", end="", flush=True)
            
            def token_callback(token: str, index: int, probability: float):
                print(token, end="", flush=True)
                
            generated_text = engine.generate_stream(
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                callback=token_callback,
            )
            
            print("\n")
            
        else:
            # Standard text generation with robust error handling
            try:
                # Ensure the prompt is a plain string
                if not isinstance(prompt, str):
                    prompt = str(prompt)
                
                # Try generation with requested parameters
                generated_texts = engine.generate(
                    prompt=prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=args.do_sample,
                    num_return_sequences=args.num_return_sequences,
                )
                
                # Check if generated_texts is empty or None
                if not generated_texts or (isinstance(generated_texts, list) and not generated_texts[0]):
                    logger.warning("Inference engine returned empty output. Falling back to greedy model.generate.")
                    # Fallback to simple greedy generation
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(engine.device)
                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids=input_ids,
                            max_length=input_ids.size(1) + args.max_tokens,
                            do_sample=False,  # Greedy decoding
                            num_return_sequences=1,
                        )
                        # Log raw output IDs for debugging
                        logger.info(f"Raw output IDs: {output_ids.tolist()}")
                    generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                
            except Exception as e:
                # Log the error
                logger.error(f"Error during generation with normal parameters: {e}")
                
                # Try with more conservative settings
                logger.info("Trying with more conservative settings...")
                
                try:
                    # Use more stable parameters
                    generated_texts = engine.generate(
                        prompt=prompt,
                        max_new_tokens=min(args.max_tokens, 20),  # Even shorter output
                        temperature=1.0,
                        top_k=0,
                        top_p=1.0,
                        repetition_penalty=1.0,
                        do_sample=False,  # Greedy decoding
                        num_return_sequences=1,
                    )
                    
                    # Check if still empty
                    if not generated_texts or (isinstance(generated_texts, list) and not generated_texts[0]):
                        logger.warning("Conservative inference engine returned empty output. Falling back to greedy model.generate.")
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(engine.device)
                        with torch.no_grad():
                            output_ids = model.generate(
                                input_ids=input_ids,
                                max_length=input_ids.size(1) + 20,
                                do_sample=False,  # Greedy decoding
                                num_return_sequences=1,
                            )
                            # Log raw output IDs for debugging
                            logger.info(f"Raw output IDs: {output_ids.tolist()}")
                        generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                
                except Exception as fallback_error:
                    # If even the conservative approach fails, log and return error
                    logger.error(f"Error during generation with conservative parameters: {fallback_error}")
                    return f"Error generating text: {str(e)}. Fallback error: {str(fallback_error)}"
                
            # Handle the case where a single text or a list is returned
            if isinstance(generated_texts, list):
                generated_text = "\n".join([text for text in generated_texts if text])
            else:
                generated_text = generated_texts if generated_texts else ""
                
            # Print results
            if generated_text:
                if args.include_prompt:
                    print(f"\nPrompt: {prompt}\nGenerated: {generated_text}\n")
                else:
                    print(f"\nGenerated: {generated_text}\n")
            else:
                logger.warning("No generated text produced after all attempts.")
                print("\nGenerated: [No output produced]\n")
                
        generation_time = time.time() - start_time
        tokens_per_second = args.max_tokens / max(generation_time, 0.001)
        
        logger.info(f"Generation took {generation_time:.2f} seconds ({tokens_per_second:.2f} tokens/s)")
        
        return generated_text if generated_text else "[No output produced]"
        
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}")
        return f"Error generating text: {str(e)}"

def interactive_mode(engine, model, tokenizer, args):
    """Run interactive generation mode."""
    print("\nConsyn AI Interactive Mode")
    print("Enter your prompt, type 'exit' to quit, or 'settings' to change generation parameters.\n")
    
    while True:
        try:
            # Get prompt from user
            prompt = input("Prompt: ")
            
            # Check for exit command
            if prompt.lower() in ["exit", "quit"]:
                break
                
            # Check for settings command
            if prompt.lower() == "settings":
                # Show current settings
                print("\nCurrent settings:")
                print(f"  Max tokens: {args.max_tokens}")
                print(f"  Temperature: {args.temperature}")
                print(f"  Top-k: {args.top_k}")
                print(f"  Top-p: {args.top_p}")
                print(f"  Repetition penalty: {args.repetition_penalty}")
                print(f"  Sampling: {args.do_sample}")
                print(f"  Stream output: {args.stream}")
                
                # Let user change settings
                print("\nEnter new settings (empty to keep current):")
                
                new_max_tokens = input(f"  Max tokens [{args.max_tokens}]: ")
                if new_max_tokens:
                    args.max_tokens = int(new_max_tokens)
                    
                new_temperature = input(f"  Temperature [{args.temperature}]: ")
                if new_temperature:
                    args.temperature = float(new_temperature)
                    
                new_top_k = input(f"  Top-k [{args.top_k}]: ")
                if new_top_k:
                    args.top_k = int(new_top_k)
                    
                new_top_p = input(f"  Top-p [{args.top_p}]: ")
                if new_top_p:
                    args.top_p = float(new_top_p)
                    
                new_rep_penalty = input(f"  Repetition penalty [{args.repetition_penalty}]: ")
                if new_rep_penalty:
                    args.repetition_penalty = float(new_rep_penalty)
                    
                new_do_sample = input(f"  Sampling (True/False) [{args.do_sample}]: ")
                if new_do_sample:
                    args.do_sample = new_do_sample.lower() == "true"
                    
                new_stream = input(f"  Stream output (True/False) [{args.stream}]: ")
                if new_stream:
                    args.stream = new_stream.lower() == "true"
                    
                print("\nSettings updated.\n")
                continue
                
            # Generate text
            generate_text(engine, model, tokenizer, args, prompt)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def process_file(engine, model, tokenizer, args, file_path: str):
    """Process prompts from a file."""
    logger.info(f"Processing prompts from {file_path}")
    
    # Read prompts from file
    with open(file_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
        
    logger.info(f"Found {len(prompts)} prompts")
    
    # Generate text for each prompt
    results = []
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i+1}/{len(prompts)}")
        
        generated_text = generate_text(engine, model, tokenizer, args, prompt)
        
        # Add to results
        if args.include_prompt:
            results.append(f"Prompt: {prompt}\nGenerated: {generated_text}\n")
        else:
            results.append(generated_text)
            
    # Save results if output file is specified
    if args.output is not None:
        logger.info(f"Saving results to {args.output}")
        
        with open(args.output, "w") as f:
            f.write("\n---\n".join(results))
            
    return results

def main():
    """Main generation function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Check that at least one input method is specified
    if not args.prompt and not args.file and not args.interactive:
        print("Error: Please specify either --prompt, --file, or --interactive")
        sys.exit(1)
        
    # Set up model
    engine, model, tokenizer = setup_model(args)
    
    # Process input
    if args.interactive:
        interactive_mode(engine, model, tokenizer, args)
    elif args.file:
        results = process_file(engine, model, tokenizer, args, args.file)
    elif args.prompt:
        # Process single prompt
        generated_text = generate_text(engine, model, tokenizer, args, args.prompt)
        
        # Save result if output file is specified
        if args.output is not None:
            logger.info(f"Saving result to {args.output}")
            
            with open(args.output, "w") as f:
                if args.include_prompt:
                    f.write(f"Prompt: {args.prompt}\nGenerated: {generated_text}")
                else:
                    f.write(generated_text)

if __name__ == "__main__":
    main()