# consyn/scripts/train.py
"""
Training script for Consyn AI models.
This script provides a command-line interface for training models.
"""

import os
import json
import argparse
import logging
import torch
from typing import Optional

from consyn.model import (
    ConsynConfig,
    ConsynVerseConfig,
    ConsynStanzaConfig,
    ConsynEpicConfig,
    ConsynLMHeadModel,
)
from consyn.tokenization import BPETokenizer, SentencePieceTokenizer
from consyn.training import (
    get_dataset,
    create_dataloader,
    ConsynTrainer,
    get_optimizer,
    get_scheduler,
    setup_logging,
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Consyn AI model")
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="verse",
        choices=["verse", "stanza", "epic"],
        help="Model size to train (verse, stanza, or epic)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training from",
    )
    
    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data directory or file",
    )
    parser.add_argument(
        "--validation",
        type=str,
        default=None,
        help="Path to validation data directory or file",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="text",
        choices=["text", "jsonl", "sharded", "huggingface"],
        help="Type of dataset to use",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Whether to use streaming dataset for large data",
    )
    
    # Training arguments
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Ratio of steps for warmup",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Type of learning rate scheduler",
    )
    
    # Hardware and optimization arguments
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to use gradient checkpointing",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Whether to use distributed training",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    
    # Logging arguments
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Number of steps between logging",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Number of steps between evaluations",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Number of steps between saving checkpoints",
    )
    parser.add_argument(
        "--logging_backend",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "both"],
        help="Logging backend to use",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="consyn-ai",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    
    return parser.parse_args()


def load_or_create_model(args):
    """Load or create a model."""
    # Create model config based on model type
    if args.model == "verse":
        config = ConsynVerseConfig()
    elif args.model == "stanza":
        config = ConsynStanzaConfig()
    elif args.model == "epic":
        config = ConsynEpicConfig()
    else:
        raise ValueError(f"Unsupported model: {args.model}")
        
    # Enable gradient checkpointing if specified
    if args.gradient_checkpointing:
        config.gradient_checkpointing = True
        
    # Create new model if no checkpoint is provided
    if args.checkpoint is None:
        logger.info(f"Creating new {args.model} model")
        model = ConsynLMHeadModel(config)
    else:
        # Load model from checkpoint
        logger.info(f"Loading model from {args.checkpoint}")
        
        # Try to load with Hugging Face transformers
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
        except (ImportError, ValueError):
            # Load config from checkpoint
            config_path = os.path.join(args.checkpoint, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                
                config = ConsynConfig.from_dict(config_dict)
                
            # Create model with config
            model = ConsynLMHeadModel(config)
            
            # Load model weights
            weights_path = os.path.join(args.checkpoint, "pytorch_model.bin")
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            else:
                raise FileNotFoundError(f"Model weights not found at {weights_path}")
                
    return model


def load_or_create_tokenizer(args):
    """Load or create a tokenizer compatible with transformers."""
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    tokenizer_path = os.path.join(args.output, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)

    # 1. Check if checkpoint already has tokenizer
    if args.checkpoint is not None:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(args.checkpoint)
        except Exception as e:
            logger.warning(f"Could not load tokenizer from checkpoint: {e}")

    # 2. Check if tokenizer already exists in output path
    if os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
        try:
            logger.info("Loading existing tokenizer from output path.")
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            logger.warning(f"Could not load existing tokenizer: {e}")

    # 3. Auto-train tokenizer using provided training file(s)
    logger.warning("Tokenizer not found. Auto-training BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=10000, show_progress=True)  # Adjust vocab_size as needed

    if os.path.isdir(args.data):
        files = [os.path.join(args.data, f) for f in os.listdir(args.data) if f.endswith(".txt")]
    else:
        files = [args.data]

    tokenizer.train(files, trainer)

    # Convert to PreTrainedTokenizerFast
    transformers_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
    )

    # Save tokenizer in transformers-compatible format
    transformers_tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"✅ Tokenizer trained and saved to {tokenizer_path}")
    return transformers_tokenizer


def main():
    """Main training function."""
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    logging_dir = os.path.join(args.output, "logs")
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(logging_dir, "train.log"))],
    )
    logger.info(f"Training arguments: {args}")

    # Load or create tokenizer first
    tokenizer = load_or_create_tokenizer(args)
    vocab_size = len(tokenizer)

    # Create model config with matching vocab_size
    if args.model == "verse":
        config = ConsynVerseConfig(vocab_size=vocab_size)
    elif args.model == "stanza":
        config = ConsynStanzaConfig(vocab_size=vocab_size)
    elif args.model == "epic":
        config = ConsynEpicConfig(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.gradient_checkpointing:
        config.gradient_checkpointing = True

    if args.checkpoint is None:
        logger.info(f"Creating new {args.model} model with vocab_size={vocab_size}")
        model = ConsynLMHeadModel(config)
    else:
        logger.info(f"Loading model from {args.checkpoint}")
        model = ConsynLMHeadModel.from_pretrained(args.checkpoint)
        if model.config.vocab_size != vocab_size:
            logger.warning(f"Resizing model token embeddings to match tokenizer vocab_size {vocab_size}")
            model.resize_token_embeddings(vocab_size)

    # Set up distributed training if specified
    if args.distributed:
        from consyn.training import setup_distributed
        
        local_rank = args.local_rank
        if local_rank == -1:
            if "LOCAL_RANK" in os.environ:
                local_rank = int(os.environ["LOCAL_RANK"])
            else:
                local_rank = 0
                
        setup_distributed(local_rank=local_rank)
        
    # Load training dataset
    train_dataset = get_dataset(
        data_path=args.data,
        tokenizer=tokenizer,
        dataset_type=args.dataset_type,
        streaming=args.streaming,
    )
    
    # Create training dataloader
    train_dataloader = create_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    # Load validation dataset if specified
    eval_dataloader = None
    if args.validation is not None:
        eval_dataset = get_dataset(
            data_path=args.validation,
            tokenizer=tokenizer,
            dataset_type=args.dataset_type,
            streaming=False,  # Don't stream validation data
        )
        
        eval_dataloader = create_dataloader(
            dataset=eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
        
    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Create trainer
    trainer = ConsynTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        output_dir=args.output,
        num_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        use_gradient_checkpointing=args.gradient_checkpointing,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_dir=logging_dir,
        logging_backend=args.logging_backend,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    
    # Train the model
    trainer.train()
            
    # Save the final model
    logger.info(f"Saving model to {args.output}")
    model.save_pretrained(args.output)

    # Save tokenizer in tokenizer subfolder
    tokenizer_dir = os.path.join(args.output, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    logger.info(f"Tokenizer saved to {tokenizer_dir}")

    # Clean up distributed training
    if args.distributed:
        from consyn.training import cleanup_distributed
        cleanup_distributed()


if __name__ == "__main__":
    main()