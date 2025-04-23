# consyn/training/trainer.py
"""
Main training loop for Consyn AI models.
This module handles the training, evaluation, and checkpoint saving for models.
"""

import os
import logging
import time
import json
import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from .optimizer import get_optimizer
from .scheduler import get_scheduler

logger = logging.getLogger(__name__)

class ConsynTrainer:
    """
    Trainer class for Consyn AI models.
    
    Handles all aspects of model training including training loop execution,
    gradient accumulation, mixed precision, distributed training,
    checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        output_dir: str = "./output",
        num_epochs: int = 3,
        max_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        eval_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        logging_steps: int = 100,
        mixed_precision: str = "no",  # "no", "fp16", or "bf16"
        use_gradient_checkpointing: bool = False,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "cosine",
        warmup_ratio: float = 0.1,
        logging_dir: Optional[str] = None,
        load_best_model_at_end: bool = False,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
        do_eval: bool = True,
        device: Optional[torch.device] = None,
        device_map: Optional[Dict[str, Union[int, str]]] = None,
        distributed_training: bool = False,
        local_rank: int = -1,
        world_size: int = 1,
        resume_from_checkpoint: Optional[str] = None,
        logging_backend: str = "tensorboard",  # "tensorboard", "wandb", or "both"
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation data
            optimizer: Optimizer for training (if None, one will be created)
            scheduler: Learning rate scheduler (if None, one will be created)
            output_dir: Directory to save model checkpoints
            num_epochs: Number of training epochs
            max_steps: Maximum number of training steps (overrides num_epochs if set)
            gradient_accumulation_steps: Number of steps to accumulate gradients
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between saving checkpoints
            logging_steps: Number of steps between logging
            mixed_precision: Whether to use mixed precision training
            use_gradient_checkpointing: Whether to use gradient checkpointing
            max_grad_norm: Maximum gradient norm for gradient clipping
            seed: Random seed for reproducibility
            lr: Learning rate (used if optimizer is None)
            weight_decay: Weight decay factor (used if optimizer is None)
            scheduler_type: Type of learning rate scheduler
            warmup_ratio: Ratio of total training steps to use for warmup
            logging_dir: Directory for logs (defaults to output_dir/logs)
            load_best_model_at_end: Whether to load the best model at the end of training
            metric_for_best_model: Metric to use for determining the best model
            greater_is_better: Whether higher values of metric_for_best_model are better
            push_to_hub: Whether to push the model to the Hub
            hub_model_id: Model ID for the Hub
            hub_token: Token for the Hub
            do_eval: Whether to perform evaluation
            device: Device to train on
            device_map: Device map for model parallelism
            distributed_training: Whether to use distributed training
            local_rank: Local rank for distributed training
            world_size: World size for distributed training
            resume_from_checkpoint: Path to checkpoint to resume from
            logging_backend: Logging backend to use
            wandb_project: Weights & Biases project name
            wandb_run_name: Weights & Biases run name
            wandb_config: Weights & Biases configuration
        """
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Set up distributed training
        self.distributed_training = distributed_training
        self.local_rank = local_rank
        self.world_size = world_size
        
        if self.distributed_training:
            if self.local_rank != -1:
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend="nccl")
                self.device = torch.device("cuda", self.local_rank)
                torch.cuda.set_device(self.local_rank)
                
        # Set up model
        self.model = model
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing"):
            self.model.gradient_checkpointing = True
            
        # Move model to device
        if device_map is not None:
            # Use device map for model parallelism
            if hasattr(model, "parallelize"):
                model.parallelize(device_map)
            else:
                # Try to use HF's .to_device for models that support it
                try:
                    from accelerate import dispatch_model
                    self.model = dispatch_model(self.model, device_map=device_map)
                except ImportError:
                    # Fall back to moving the whole model to a single device
                    self.model = self.model.to(self.device)
        else:
            # Move model to device
            self.model = self.model.to(self.device)
            
        # Set up DataLoaders
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = get_optimizer(
                model=self.model,
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optimizer
            
        # Compute total number of training steps
        if max_steps is not None:
            self.max_steps = max_steps
            self.num_epochs = math.ceil(max_steps / (len(train_dataloader) // gradient_accumulation_steps))
        else:
            self.num_epochs = num_epochs
            self.max_steps = num_epochs * (len(train_dataloader) // gradient_accumulation_steps)
            
        # Set up scheduler
        if scheduler is None:
            num_warmup_steps = int(self.max_steps * warmup_ratio)
            self.scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.max_steps,
            )
        else:
            self.scheduler = scheduler
            
        # Set up mixed precision training
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision == "fp16" else None
        
        # Set up gradient accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Set up evaluation and saving
        self.do_eval = do_eval
        self.eval_steps = eval_steps or (len(train_dataloader) // gradient_accumulation_steps)
        self.save_steps = save_steps or self.eval_steps
        self.logging_steps = logging_steps
        
        # Set up directories
        self.output_dir = output_dir
        self.logging_dir = logging_dir or os.path.join(output_dir, "logs")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        # Set up best model tracking
        self.load_best_model_at_end = load_best_model_at_end
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.best_metric = float("inf") if not greater_is_better else float("-inf")
        self.best_model_checkpoint = None
        
        # Hub integration
        self.push_to_hub = push_to_hub
        self.hub_model_id = hub_model_id
        self.hub_token = hub_token
        
        # Set up loggers
        self.logging_backend = logging_backend
        self._setup_loggers(wandb_project, wandb_run_name, wandb_config)
        
        # Set up state
        self.global_step = 0
        self.epoch = 0
        self.step_loss = 0.0
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)
            
    def _setup_loggers(self, wandb_project, wandb_run_name, wandb_config):
        """Set up logging backends."""
        self.tb_writer = None
        self.wandb_run = None
        
        # Set up TensorBoard logging
        if (self.logging_backend == "tensorboard" or self.logging_backend == "both") and HAS_TENSORBOARD:
            self.tb_writer = SummaryWriter(log_dir=self.logging_dir)
            
        # Set up Weights & Biases logging
        if (self.logging_backend == "wandb" or self.logging_backend == "both") and HAS_WANDB:
            wandb_config = wandb_config or {}
            
            # Add training parameters to config
            wandb_config.update({
                "batch_size": self.train_dataloader.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_steps": self.max_steps,
                "mixed_precision": self.mixed_precision,
                "optimizer": self.optimizer.__class__.__name__,
                "scheduler": self.scheduler.__class__.__name__ if self.scheduler else None,
            })
            
            # Initialize W&B
            self.wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=wandb_config,
                dir=self.logging_dir,
                resume="allow",
            )
            
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to all configured backends."""
        # Add prefix to metrics if provided
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
            
        log_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # Log to TensorBoard
        if self.tb_writer is not None:
            for k, v in log_metrics.items():
                self.tb_writer.add_scalar(k, v, self.global_step)
                
        # Log to W&B
        if self.wandb_run is not None:
            self.wandb_run.log(log_metrics, step=self.global_step)
            
        # Log to console
        if self.local_rank <= 0:  # Only log from first process in distributed training
            prefix_str = f"{prefix[:-1]} " if prefix else ""
            log_str = f"{prefix_str}metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            logging.info(log_str)
            
    def _save_checkpoint(self, output_dir: str, metrics: Optional[Dict[str, float]] = None):
        """Save model checkpoint and training state."""
        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Only save from main process in distributed training
        if self.local_rank <= 0:
            # Save model weights
            torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            
            # Save optimizer and scheduler states
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            if self.scheduler is not None:
                torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                
            # Save training state
            training_state = {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "random_state": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                training_state["cuda_random_state"] = torch.cuda.get_rng_state_all()
                
            torch.save(training_state, os.path.join(output_dir, "training_state.pt"))
            
            # Save metrics if provided
            if metrics is not None:
                with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
                    
            logging.info(f"Saved checkpoint to {output_dir}")
            
    def _load_from_checkpoint(self, checkpoint_dir: str):
        """Load model and training state from checkpoint."""
        # Load model weights
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logging.info(f"Loaded model weights from {model_path}")
            
        # Load optimizer state
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            logging.info(f"Loaded optimizer state from {optimizer_path}")
            
        # Load scheduler state
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_path) and self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(scheduler_path))
            logging.info(f"Loaded scheduler state from {scheduler_path}")
            
        # Load training state
        training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)
            self.global_step = training_state["global_step"]
            self.epoch = training_state["epoch"]
            
            # Restore random state
            torch.set_rng_state(training_state["random_state"])
            if torch.cuda.is_available() and "cuda_random_state" in training_state:
                torch.cuda.set_rng_state_all(training_state["cuda_random_state"])
                
            logging.info(f"Loaded training state from {training_state_path}")
            
        # Load metrics to restore best model tracking
        metrics_path = os.path.join(checkpoint_dir, "metrics.json")
        if os.path.exists(metrics_path) and self.load_best_model_at_end:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                
            if self.metric_for_best_model in metrics:
                metric_value = metrics[self.metric_for_best_model]
                
                # Check if this is the best model so far
                is_better = (
                    (self.greater_is_better and metric_value > self.best_metric) or
                    (not self.greater_is_better and metric_value < self.best_metric)
                )
                
                if is_better:
                    self.best_metric = metric_value
                    self.best_model_checkpoint = checkpoint_dir
                    
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate the model on the provided dataloader.
        
        Args:
            dataloader: DataLoader for evaluation (defaults to self.eval_dataloader)
            
        Returns:
            dict: Evaluation metrics
        """
        # Use provided dataloader or default to self.eval_dataloader
        if dataloader is None:
            if self.eval_dataloader is None:
                raise ValueError("No evaluation dataloader provided")
            dataloader = self.eval_dataloader
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Keep track of evaluation statistics
        eval_loss = 0.0
        num_eval_steps = 0
        
        # Evaluate model on all batches
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", disable=self.local_rank > 0):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                with autocast(enabled=self.mixed_precision != "no", dtype=torch.float16 if self.mixed_precision == "fp16" else torch.bfloat16):
                    outputs = self.model(**batch)
                    
                # Get loss
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                # Accumulate loss
                eval_loss += loss.item()
                num_eval_steps += 1
                
        if num_eval_steps == 0:
            logger.warning("Evaluation had zero steps, skipping metric calculation.")
            return {"loss": float("inf"), "perplexity": float("inf")}
            
        # Compute average loss
        eval_loss = eval_loss / num_eval_steps

        # Compute perplexity
        perplexity = math.exp(eval_loss)
        
        # Prepare metrics
        metrics = {
            "loss": eval_loss,
            "perplexity": perplexity,
        }
        
        # Log metrics
        self._log_metrics(metrics, prefix="eval")
        
        # Set model back to training mode
        self.model.train()
        
        return metrics
        
    def train(self):
        """
        Train the model for the specified number of epochs/steps.
        
        Returns:
            dict: Training metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Keep track of training statistics
        tr_loss = 0.0
        logging_loss = 0.0
        epoch_loss = 0.0
        
        # Initialize progress bar
        progress_bar = tqdm(
            total=self.max_steps,
            disable=self.local_rank > 0,
            desc=f"Training (step {self.global_step})",
            position=0,
        )
        
        # Update progress bar with current step
        progress_bar.update(self.global_step)
        
        # Main training loop
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            epoch_loss = 0.0
            steps_in_epoch = 0  # Track actual steps executed in this epoch
            
            # Reset loss accumulation after each epoch
            logging_loss = tr_loss
            
            # Iterate over batches
            for step, batch in enumerate(self.train_dataloader):
                # Track actual steps
                steps_in_epoch += 1
                
                # Skip steps that have already been processed when resuming
                if self.global_step // self.gradient_accumulation_steps >= len(self.train_dataloader) * epoch + step:
                    continue
                    
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision if enabled
                with autocast(enabled=self.mixed_precision != "no", dtype=torch.float16 if self.mixed_precision == "fp16" else torch.bfloat16):
                    outputs = self.model(**batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                    
                # Backward pass with mixed precision if enabled
                if self.mixed_precision == "fp16":
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                # Update loss statistics
                self.step_loss += loss.item() * self.gradient_accumulation_steps
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                
                # Only update parameters after accumulating enough gradients
                if (step + 1) % self.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        if self.mixed_precision == "fp16":
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            
                    # Update parameters
                    if self.mixed_precision == "fp16":
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    # Update learning rate
                    if self.scheduler is not None:
                        self.scheduler.step()
                        
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    self.global_step += 1
                    
                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_description(f"Training (loss: {self.step_loss:.4f}, step: {self.global_step})")
                    
                    # Accumulate loss for logging
                    tr_loss += self.step_loss
                    self.step_loss = 0.0
                    
                    # Log metrics at logging_steps
                    if self.global_step % self.logging_steps == 0:
                        # Compute average loss since last logging
                        avg_loss = (tr_loss - logging_loss) / self.logging_steps
                        
                        # Get current learning rate
                        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]["lr"]
                        
                        # Prepare and log metrics
                        metrics = {
                            "loss": avg_loss,
                            "learning_rate": current_lr,
                            "epoch": self.epoch + (step + 1) / len(self.train_dataloader),
                        }
                        
                        self._log_metrics(metrics, prefix="train")
                        
                        # Reset logging loss
                        logging_loss = tr_loss
                        
                    # Evaluate at eval_steps
                    if self.do_eval and self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        
                        # Check if this is the best model so far
                        if self.load_best_model_at_end and self.metric_for_best_model in eval_metrics:
                            metric_value = eval_metrics[self.metric_for_best_model]
                            
                            is_better = (
                                (self.greater_is_better and metric_value > self.best_metric) or
                                (not self.greater_is_better and metric_value < self.best_metric)
                            )
                            
                            if is_better:
                                self.best_metric = metric_value
                                self.best_model_checkpoint = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
                                
                    # Save checkpoint at save_steps
                    if self.global_step % self.save_steps == 0:
                        output_dir = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
                        
                        # Save checkpoint
                        self._save_checkpoint(output_dir)
                        
                    # Check if we've reached the maximum number of steps
                    if self.global_step >= self.max_steps:
                        break
                        
            # End of epoch
            
            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            
            # Safely calculate average loss to avoid division by zero
            if steps_in_epoch > 0:
                avg_epoch_loss = epoch_loss / steps_in_epoch
            else:
                avg_epoch_loss = 0.0
                logger.warning(f"Epoch {epoch} had zero steps, cannot calculate average loss")
            
            epoch_metrics = {
                "epoch_loss": avg_epoch_loss,
                "epoch_time": epoch_time,
                "steps_in_epoch": steps_in_epoch,
            }
            
            self._log_metrics(epoch_metrics, prefix=f"epoch_{epoch}")
            
            # Check if we've reached the maximum number of steps
            if self.global_step >= self.max_steps:
                break
                
        # End of training
        progress_bar.close()
        
        # Save the final model
        self._save_checkpoint(os.path.join(self.output_dir, "final"))
        
        # Evaluate the final model
        if self.do_eval and self.eval_dataloader is not None:
            final_metrics = self.evaluate()
        else:
            final_metrics = {}
            
        # Load the best model if requested
        if self.load_best_model_at_end and self.best_model_checkpoint is not None:
            logger.info(f"Loading best model from {self.best_model_checkpoint}")
            self._load_from_checkpoint(self.best_model_checkpoint)
            
            # Save the best model to the final output directory
            self._save_checkpoint(os.path.join(self.output_dir, "best"))
            
        # Push to hub if requested
        if self.push_to_hub:
            self._push_to_hub()
            
        # Clean up
        if self.tb_writer:
            self.tb_writer.close()
            
        # Return metrics
        return final_metrics

    def _push_to_hub(self):
        """Push the model to the Hub."""
        try:
            from huggingface_hub import HfApi
            
            # Set up Hub API
            api = HfApi()
            model_id = self.hub_model_id or os.path.basename(self.output_dir)
            
            # Create model repository if it doesn't exist
            try:
                repo_url = api.create_repo(model_id, token=self.hub_token, exist_ok=True)
            except Exception as e:
                logging.error(f"Error creating repository: {e}")
                return
                
            # Upload model files
            for root, _, files in os.walk(os.path.join(self.output_dir, "best" if self.load_best_model_at_end else "final")):
                for file in files:
                    local_path = os.path.join(root, file)
                    # Get relative path within output directory
                    relative_path = os.path.relpath(local_path, self.output_dir)
                    
                    # Upload the file
                    try:
                        api.upload_file(
                            path_or_fileobj=local_path,
                            path_in_repo=relative_path,
                            repo_id=model_id,
                            token=self.hub_token,
                        )
                    except Exception as e:
                        logging.error(f"Error uploading {local_path}: {e}")
                        
            logging.info(f"Model pushed to {repo_url}")
            
        except ImportError:
            logging.error("huggingface_hub package is required to push to the Hub")
