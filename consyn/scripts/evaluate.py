# consyn/scripts/evaluate.py
"""
Evaluation script for Consyn AI models.
This script provides a command-line interface for evaluating models.
"""

import os
import argparse
import logging
import json
import torch
from typing import Dict, List, Optional

from consyn.model import ConsynLMHeadModel
from consyn.tokenization import BPETokenizer, SentencePieceTokenizer
from consyn.inference import ConsynInferenceEngine

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a Consyn AI model")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Path to evaluation data file (jsonl)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum generation length",
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
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
        default=0.9,
        help="Top-p filtering value",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling (default: greedy decoding)",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Whether to use flash attention if available",
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from path."""
    logger.info(f"Loading model from {model_path}")
    
    # Try to load with Hugging Face transformers
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
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


def load_evaluation_data(eval_file: str) -> List[Dict]:
    """Load evaluation data from file."""
    logger.info(f"Loading evaluation data from {eval_file}")
    
    data = []
    with open(eval_file, "r") as f:
        for line in f:
            # Parse JSON data
            item = json.loads(line.strip())
            
            # Ensure required fields are present
            if "input" not in item:
                raise ValueError(f"Evaluation data must contain 'input' field: {item}")
                
            # Add reference field if not present
            if "reference" not in item:
                item["reference"] = None
                
            data.append(item)
                
    logger.info(f"Loaded {len(data)} evaluation examples")
    return data


def main():
    """Main evaluation function."""
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
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Create inference engine
    engine = ConsynInferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_flash_attention=args.use_flash_attention,
    )
    
    # Load evaluation data
    eval_data = load_evaluation_data(args.eval_file)
    
    # Extract prompts
    prompts = [item["input"] for item in eval_data]
    
    # Generate responses
    logger.info("Generating responses")
    generated_texts = engine.batch_generate(
        prompts=prompts,
        max_new_tokens=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        batch_size=args.batch_size,
    )
    
    # Add generated responses to evaluation data
    for i, item in enumerate(eval_data):
        item["generated"] = generated_texts[i]
        
    # Calculate metrics if references are available
    if all(item["reference"] is not None for item in eval_data):
        logger.info("Calculating evaluation metrics")
        
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from rouge import Rouge
            
            # Initialize metrics
            total_bleu = 0.0
            rouge = Rouge()
            rouge_scores = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            
            # Calculate metrics for each example
            for item in eval_data:
                # Calculate BLEU score
                reference = item["reference"].split()
                hypothesis = item["generated"].split()
                item["bleu"] = sentence_bleu([reference], hypothesis)
                total_bleu += item["bleu"]
                
                # Calculate ROUGE scores
                try:
                    scores = rouge.get_scores(item["generated"], item["reference"])[0]
                    item["rouge"] = scores
                    
                    # Accumulate ROUGE scores
                    for key in rouge_scores:
                        rouge_scores[key]["f"] += scores[key]["f"]
                except:
                    # Handle cases where ROUGE calculation fails
                    item["rouge"] = None
                    
            # Calculate average metrics
            avg_bleu = total_bleu / len(eval_data)
            avg_rouge = {
                key: {"f": value["f"] / len(eval_data)}
                for key, value in rouge_scores.items()
            }
            
            # Add average metrics to results
            eval_results = {
                "examples": eval_data,
                "metrics": {
                    "bleu": avg_bleu,
                    "rouge": avg_rouge,
                }
            }
            
            logger.info(f"BLEU score: {avg_bleu:.4f}")
            logger.info(f"ROUGE-1 F1: {avg_rouge['rouge-1']['f']:.4f}")
            logger.info(f"ROUGE-2 F1: {avg_rouge['rouge-2']['f']:.4f}")
            logger.info(f"ROUGE-L F1: {avg_rouge['rouge-l']['f']:.4f}")
            
        except ImportError:
            logger.warning("NLTK or Rouge not installed. Skipping metric calculation.")
            
            # Just include examples without metrics
            eval_results = {
                "examples": eval_data,
            }
    else:
        # No references available, just include examples
        eval_results = {
            "examples": eval_data,
        }
        
    # Save results if output file is specified
    if args.output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        
        with open(args.output_file, "w") as f:
            json.dump(eval_results, f, indent=2)
            
        logger.info(f"Saved evaluation results to {args.output_file}")
    
    # Return results
    return eval_results


if __name__ == "__main__":
    main()