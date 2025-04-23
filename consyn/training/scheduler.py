# consyn/training/scheduler.py
"""
Learning rate schedulers for Consyn AI models.
This module provides learning rate scheduling utilities for training.
"""

import math
from typing import Callable, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def get_constant_schedule(optimizer: Optimizer) -> LambdaLR:
    """
    Create a scheduler with a constant learning rate.
    
    Args:
        optimizer: Optimizer to schedule
        
    Returns:
        LambdaLR: Scheduler with constant learning rate
    """
    return LambdaLR(optimizer, lambda _: 1.0)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int
) -> LambdaLR:
    """
    Create a scheduler with a constant learning rate after warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        
    Returns:
        LambdaLR: Scheduler with warmup and constant learning rate
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0
        
    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    min_ratio: float = 0.0
) -> LambdaLR:
    """
    Create a scheduler with a learning rate that decreases linearly after warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_ratio: Minimum learning rate ratio at the end of training
        
    Returns:
        LambdaLR: Scheduler with warmup and linear decay
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_ratio, 1.0 - progress * (1.0 - min_ratio))
        
    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    min_ratio: float = 0.0,
    num_cycles: float = 0.5
) -> LambdaLR:
    """
    Create a scheduler with a learning rate that decreases with a cosine curve after warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_ratio: Minimum learning rate ratio at the end of training
        num_cycles: Number of cosine cycles
        
    Returns:
        LambdaLR: Scheduler with warmup and cosine decay
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
            
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        # Cosine decay with a multiplier for number of cycles
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
        # Scale between 1.0 and min_ratio
        return min_ratio + (1.0 - min_ratio) * cosine_decay
        
    return LambdaLR(optimizer, lr_lambda)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    power: float = 1.0,
) -> LambdaLR:
    """
    Create a scheduler with a learning rate that decreases polynomially after warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_ratio: Minimum learning rate ratio at the end of training
        power: Power of the polynomial decay
        
    Returns:
        LambdaLR: Scheduler with warmup and polynomial decay
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
            
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        # Polynomial decay
        poly_decay = (1.0 - progress) ** power
        
        # Scale between 1.0 and min_ratio
        return min_ratio + (1.0 - min_ratio) * poly_decay
        
    return LambdaLR(optimizer, lr_lambda)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    min_ratio: float = 0.0,
) -> LambdaLR:
    """
    Create a scheduler with a learning rate that decreases with cosine curve
    and hard restarts after warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of hard restart cycles
        min_ratio: Minimum learning rate ratio at the end of each cycle
        
    Returns:
        LambdaLR: Scheduler with warmup, cosine decay, and hard restarts
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
            
        # Progress after warmup
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        # Calculate which cycle we're in
        if progress >= 1.0:
            return min_ratio
            
        # Calculate progress within the current cycle
        cycle_progress = progress * num_cycles
        cycle = int(cycle_progress)
        cycle_progress = cycle_progress - cycle  # Progress within the cycle (0 to 1)
        
        # Cosine decay within the cycle
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        
        # Scale between 1.0 and min_ratio
        return min_ratio + (1.0 - min_ratio) * cosine_decay
        
    return LambdaLR(optimizer, lr_lambda)


def get_one_cycle_schedule(
    optimizer: Optimizer,
    num_training_steps: int,
    max_lr: Optional[Union[float, List[float]]] = None,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> LRScheduler:
    """
    Create a scheduler with a one-cycle policy.
    
    The 1cycle policy anneals the learning rate from an initial learning rate to some
    maximum learning rate and then back to the initial learning rate. It also employs
    a cyclical momentum (beta1 in Adam) schedule.
    
    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Total number of training steps
        max_lr: Maximum learning rate (defaults to base_lr * div_factor)
        pct_start: Percentage of cycle spent increasing the learning rate
        div_factor: Initial learning rate is max_lr / div_factor
        final_div_factor: Final learning rate is max_lr / final_div_factor
        
    Returns:
        OneCycleLR: One-cycle learning rate scheduler
    """
    # Import here to avoid circular imports
    from torch.optim.lr_scheduler import OneCycleLR
    
    # Get max_lr from optimizer's param_groups if not specified
    if max_lr is None:
        max_lr = []
        for group in optimizer.param_groups:
            max_lr.append(group['lr'] * div_factor)
    elif not isinstance(max_lr, list):
        max_lr = [max_lr] * len(optimizer.param_groups)
        
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=num_training_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        three_phase=False,
    )


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
) -> LRScheduler:
    """
    Get the appropriate learning rate scheduler.
    
    Args:
        name: Name of the scheduler ('linear', 'cosine', 'cosine_with_restarts',
              'polynomial', 'constant', 'constant_with_warmup', 'one_cycle')
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        **kwargs: Additional arguments for specific schedulers
        
    Returns:
        LRScheduler: Learning rate scheduler
        
    Raises:
        ValueError: If the scheduler name is unknown or required arguments are missing
    """
    # For schedulers that require num_training_steps
    if name in ["linear", "cosine", "cosine_with_restarts", "polynomial", "one_cycle"]:
        if num_training_steps is None:
            raise ValueError(f"num_training_steps must be specified for {name} scheduler")
            
    # For schedulers that require num_warmup_steps
    if name in ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"]:
        if num_warmup_steps is None:
            raise ValueError(f"num_warmup_steps must be specified for {name} scheduler")
            
    name = name.lower()
    
    # Return the appropriate scheduler
    if name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps,
            min_ratio=kwargs.get("min_ratio", 0.0)
        )
    elif name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps,
            min_ratio=kwargs.get("min_ratio", 0.0),
            num_cycles=kwargs.get("num_cycles", 0.5),
        )
    elif name == "cosine_with_restarts":
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps,
            num_cycles=kwargs.get("num_cycles", 1),
            min_ratio=kwargs.get("min_ratio", 0.0),
        )
    elif name == "polynomial":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps,
            min_ratio=kwargs.get("min_ratio", 0.0),
            power=kwargs.get("power", 1.0),
        )
    elif name == "constant":
        return get_constant_schedule(optimizer)
    elif name == "constant_with_warmup":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    elif name == "one_cycle":
        return get_one_cycle_schedule(
            optimizer, num_training_steps,
            max_lr=kwargs.get("max_lr"),
            pct_start=kwargs.get("pct_start", 0.3),
            div_factor=kwargs.get("div_factor", 25.0),
            final_div_factor=kwargs.get("final_div_factor", 1e4),
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")
