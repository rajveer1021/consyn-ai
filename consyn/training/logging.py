# consyn/training/logging.py
"""
Logging utilities for Consyn AI models.
This module provides utilities for logging training progress and metrics.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Optional, Union, Any

# Set up basic logging configuration
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Dictionary to store metric history
_metric_history = {}


class TensorBoardLogger:
    """Logger for TensorBoard."""
    
    def __init__(self, log_dir: str, **kwargs):
        """
        Initialize the TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            **kwargs: Additional arguments for SummaryWriter
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir, **kwargs)
            self.initialized = True
        except ImportError:
            logging.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.initialized = False
            
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value to TensorBoard.
        
        Args:
            tag: Name of the scalar
            value: Value to log
            step: Training step
        """
        if self.initialized:
            self.writer.add_scalar(tag, value, step)
            
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """
        Log multiple scalars to TensorBoard.
        
        Args:
            main_tag: Main tag for the scalars
            tag_scalar_dict: Dictionary of tags and values
            step: Training step
        """
        if self.initialized:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
            
    def log_histogram(self, tag: str, values, step: int):
        """
        Log a histogram to TensorBoard.
        
        Args:
            tag: Name of the histogram
            values: Values to log
            step: Training step
        """
        if self.initialized:
            self.writer.add_histogram(tag, values, step)
            
    def log_text(self, tag: str, text_string: str, step: int):
        """
        Log text to TensorBoard.
        
        Args:
            tag: Name of the text
            text_string: Text to log
            step: Training step
        """
        if self.initialized:
            self.writer.add_text(tag, text_string, step)
            
    def log_figure(self, tag: str, figure, step: int):
        """
        Log a figure to TensorBoard.
        
        Args:
            tag: Name of the figure
            figure: Figure to log
            step: Training step
        """
        if self.initialized:
            self.writer.add_figure(tag, figure, step)
            
    def close(self):
        """Close the TensorBoard writer."""
        if self.initialized:
            self.writer.close()


class WeightsAndBiasesLogger:
    """Logger for Weights & Biases."""
    
    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize the Weights & Biases logger.
        
        Args:
            project: W&B project name
            name: W&B run name
            config: W&B configuration
            **kwargs: Additional arguments for wandb.init
        """
        try:
            import wandb
            self.wandb = wandb
            
            # Initialize W&B
            if wandb.run is None:
                wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    **kwargs
                )
                
            self.initialized = True
        except ImportError:
            logging.warning("Weights & Biases not available. Install with: pip install wandb")
            self.initialized = False
            
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """
        Log data to W&B.
        
        Args:
            data: Dictionary of data to log
            step: Training step
        """
        if self.initialized:
            self.wandb.log(data, step=step)
            
    def log_artifact(self, artifact_name: str, artifact_type: str, artifact_path: str):
        """
        Log an artifact to W&B.
        
        Args:
            artifact_name: Name of the artifact
            artifact_type: Type of the artifact
            artifact_path: Path to the artifact
        """
        if self.initialized:
            artifact = self.wandb.Artifact(name=artifact_name, type=artifact_type)
            artifact.add_file(artifact_path)
            self.wandb.log_artifact(artifact)
            
    def save_model(self, model_path: str, name: Optional[str] = None):
        """
        Save a model to W&B.
        
        Args:
            model_path: Path to the model
            name: Name of the model
        """
        if self.initialized:
            self.wandb.save(model_path, base_path=os.path.dirname(model_path))
            
            if name is not None:
                artifact = self.wandb.Artifact(name=name, type="model")
                artifact.add_file(model_path)
                self.wandb.log_artifact(artifact)
                
    def finish(self):
        """Finish the W&B run."""
        if self.initialized and self.wandb.run is not None:
            self.wandb.finish()


class FileLogger:
    """Logger for writing metrics to files."""
    
    def __init__(self, log_dir: str):
        """
        Initialize the file logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create main log file
        self.main_log_file = os.path.join(log_dir, "training_log.jsonl")
        
        # Create metric history file
        self.metric_history_file = os.path.join(log_dir, "metric_history.json")
        
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """
        Log data to file.
        
        Args:
            data: Dictionary of data to log
            step: Training step
        """
        # Add timestamp and step
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step": step,
            **data
        }
        
        # Append to main log file
        with open(self.main_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        # Update metric history
        for key, value in data.items():
            if key not in _metric_history:
                _metric_history[key] = []
                
            _metric_history[key].append((step, value))
            
        # Save metric history
        with open(self.metric_history_file, "w") as f:
            json.dump(_metric_history, f, indent=2)
            
    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration to file.
        
        Args:
            config: Configuration dictionary
        """
        config_file = os.path.join(self.log_dir, "config.json")
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)


class MultiLogger:
    """Combines multiple loggers."""
    
    def __init__(self, loggers: List):
        """
        Initialize the multi-logger.
        
        Args:
            loggers: List of loggers
        """
        self.loggers = loggers
        
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value to all loggers.
        
        Args:
            tag: Name of the scalar
            value: Value to log
            step: Training step
        """
        for logger in self.loggers:
            if hasattr(logger, "log_scalar"):
                logger.log_scalar(tag, value, step)
            elif hasattr(logger, "log"):
                logger.log({tag: value}, step=step)
                
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """
        Log data to all loggers.
        
        Args:
            data: Dictionary of data to log
            step: Training step
        """
        for logger in self.loggers:
            if hasattr(logger, "log"):
                logger.log(data, step=step)
                
    def close(self):
        """Close all loggers."""
        for logger in self.loggers:
            if hasattr(logger, "close"):
                logger.close()
            elif hasattr(logger, "finish"):
                logger.finish()


def setup_logging(
    output_dir: str,
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    use_file_logging: bool = True,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
) -> MultiLogger:
    """
    Set up logging for training.
    
    Args:
        output_dir: Directory for logs
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use Weights & Biases
        use_file_logging: Whether to use file logging
        wandb_project: W&B project name
        wandb_name: W&B run name
        wandb_config: W&B configuration
        
    Returns:
        MultiLogger: Combined logger
    """
    loggers = []
    
    # Set up TensorBoard logger
    if use_tensorboard:
        tb_log_dir = os.path.join(output_dir, "tensorboard")
        tb_logger = TensorBoardLogger(log_dir=tb_log_dir)
        
        if tb_logger.initialized:
            loggers.append(tb_logger)
            
    # Set up W&B logger
    if use_wandb:
        wandb_logger = WeightsAndBiasesLogger(
            project=wandb_project,
            name=wandb_name,
            config=wandb_config,
            dir=os.path.join(output_dir, "wandb"),
        )
        
        if wandb_logger.initialized:
            loggers.append(wandb_logger)
            
    # Set up file logger
    if use_file_logging:
        file_log_dir = os.path.join(output_dir, "logs")
        file_logger = FileLogger(log_dir=file_log_dir)
        loggers.append(file_logger)
        
        # Log the config if provided
        if wandb_config is not None:
            file_logger.log_config(wandb_config)
            
    # Return combined logger
    return MultiLogger(loggers)


def log_metrics(metrics: Dict[str, float], step: int, logger: Optional[Any] = None):
    """
    Log metrics using the provided logger or to console.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Training step
        logger: Logger to use (if None, log to console)
    """
    # Format metrics for console output
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    message = f"Step {step}: {metrics_str}"
    
    # Log to console
    logging.info(message)
    
    # Log to logger if provided
    if logger is not None:
        if hasattr(logger, "log"):
            logger.log(metrics, step=step)
        elif hasattr(logger, "log_scalar"):
            for key, value in metrics.items():
                logger.log_scalar(key, value, step)


def get_text_table(table_data: List[List[str]], header: Optional[List[str]] = None) -> str:
    """
    Create a formatted text table.
    
    Args:
        table_data: List of rows, where each row is a list of cells
        header: Optional header row
        
    Returns:
        str: Formatted text table
    """
    # Combine header and data
    if header is not None:
        table_data = [header] + table_data
        
    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in table_data) for i in range(len(table_data[0]))]
    
    # Create the header separator
    if header is not None:
        header_sep = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"
    else:
        header_sep = ""
        
    # Create the row format
    row_format = "| " + " | ".join("{:" + str(width) + "}" for width in col_widths) + " |"
    
    # Create the table
    table = []
    for i, row in enumerate(table_data):
        table.append(row_format.format(*[str(cell) for cell in row]))
        if i == 0 and header is not None:
            table.append(header_sep)
            
    # Add borders
    border = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"
    table = [border] + table + [border]
    
    return "\n".join(table)


def format_time(seconds: float) -> str:
    """
    Format time in a human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    # Handle edge cases
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
        
    # Convert to minutes and seconds
    minutes, seconds = divmod(seconds, 60)
    
    # Convert to hours, minutes, and seconds
    hours, minutes = divmod(minutes, 60)
    
    # Format based on magnitude
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.1f}s"


def log_training_start(
    model_name: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    max_steps: int,
    gradient_accumulation_steps: int,
    output_dir: str,
    logger: Optional[Any] = None,
):
    """
    Log the start of training with configuration details.
    
    Args:
        model_name: Name of the model
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Number of epochs
        max_steps: Maximum number of steps
        gradient_accumulation_steps: Number of gradient accumulation steps
        output_dir: Output directory
        logger: Logger to use
    """
    # Create configuration table
    config = [
        ["Model", model_name],
        ["Batch Size", batch_size],
        ["Gradient Accumulation Steps", gradient_accumulation_steps],
        ["Effective Batch Size", batch_size * gradient_accumulation_steps],
        ["Learning Rate", learning_rate],
        ["Number of Epochs", num_epochs],
        ["Maximum Steps", max_steps],
        ["Output Directory", output_dir],
    ]
    
    # Format as table
    table = get_text_table(config, header=["Parameter", "Value"])
    
    # Log to console
    logging.info("Starting training with the following configuration:")
    for line in table.split("\n"):
        logging.info(line)
        
    # Log to logger if provided
    if logger is not None:
        if hasattr(logger, "log_text"):
            logger.log_text("training_config", table, step=0)
        elif hasattr(logger, "log"):
            logger.log({"training_config": table}, step=0)


def log_training_complete(
    total_time: float,
    final_loss: float,
    best_loss: float,
    best_step: int,
    logger: Optional[Any] = None,
):
    """
    Log the completion of training with summary metrics.
    
    Args:
        total_time: Total training time in seconds
        final_loss: Final training loss
        best_loss: Best validation loss
        best_step: Step with the best validation loss
        logger: Logger to use
    """
    # Format time
    formatted_time = format_time(total_time)
    
    # Create summary table
    summary = [
        ["Total Training Time", formatted_time],
        ["Final Loss", f"{final_loss:.4f}"],
        ["Best Loss", f"{best_loss:.4f}"],
        ["Best Step", best_step],
    ]
    
    # Format as table
    table = get_text_table(summary, header=["Metric", "Value"])
    
    # Log to console
    logging.info("Training complete:")
    for line in table.split("\n"):
        logging.info(line)
        
    # Log to logger if provided
    if logger is not None:
        metrics = {
            "training_time": total_time,
            "final_loss": final_loss,
            "best_loss": best_loss,
            "best_step": best_step,
        }
        
        if hasattr(logger, "log"):
            logger.log(metrics)
            
            if hasattr(logger, "log_text"):
                logger.log_text("training_summary", table, step=best_step)
                
        elif hasattr(logger, "log_scalar"):
            for key, value in metrics.items():
                logger.log_scalar(key, value, step=best_step)
