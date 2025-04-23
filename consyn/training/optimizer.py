# consyn/training/optimizer.py
"""
Optimizer configurations for Consyn AI models.
This module provides utilities for creating and configuring optimizers.
"""

from typing import Dict, List, Optional, Union

import torch
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(
    model: torch.nn.Module,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    optimizer_type: str = "adamw",
    use_fused: bool = False,
) -> Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay factor
        beta1: Beta1 factor for Adam optimizer
        beta2: Beta2 factor for Adam optimizer
        eps: Epsilon factor for Adam optimizer
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        use_fused: Whether to use fused implementation (requires CUDA)
        
    Returns:
        Optimizer: Configured optimizer
    """
    # Prepare optimizer parameters, with weight decay only on non-bias and non-LayerNorm weights
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Create the optimizer
    if optimizer_type.lower() == "adamw":
        if use_fused and torch.cuda.is_available():
            try:
                from apex.optimizers import FusedAdam
                optimizer = FusedAdam(
                    optimizer_grouped_params,
                    lr=lr,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=weight_decay,
                    adam_w_mode=True,
                )
            except ImportError:
                optimizer = AdamW(
                    optimizer_grouped_params,
                    lr=lr,
                    betas=(beta1, beta2),
                    eps=eps,
                )
        else:
            optimizer = AdamW(
                optimizer_grouped_params,
                lr=lr,
                betas=(beta1, beta2),
                eps=eps,
            )
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_params,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_params,
            lr=lr,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
    return optimizer


def get_grouped_params(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    lr_multipliers: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Group parameters for optimization with different learning rates or weight decay.
    
    Args:
        model: Model to optimize
        weight_decay: Weight decay factor
        lr_multipliers: Dictionary mapping parameter name patterns to learning rate multipliers
        
    Returns:
        list: Grouped parameters for optimizer
    """
    # Default: no learning rate multipliers
    if lr_multipliers is None:
        lr_multipliers = {}
        
    # Parameters with no weight decay
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"]
    
    # Group parameters by learning rate multiplier and weight decay
    param_groups = []
    
    # First, collect parameters with default learning rate
    default_with_decay = {
        "params": [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay) and not any(pattern in n for pattern in lr_multipliers)
        ],
        "weight_decay": weight_decay,
    }
    
    default_without_decay = {
        "params": [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay) and not any(pattern in n for pattern in lr_multipliers)
        ],
        "weight_decay": 0.0,
    }
    
    param_groups.append(default_with_decay)
    param_groups.append(default_without_decay)
    
    # Then, create groups for each learning rate multiplier
    for pattern, multiplier in lr_multipliers.items():
        with_decay = {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and pattern in n
            ],
            "weight_decay": weight_decay,
            "lr_multiplier": multiplier,
        }
        
        without_decay = {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and pattern in n
            ],
            "weight_decay": 0.0,
            "lr_multiplier": multiplier,
        }
        
        param_groups.append(with_decay)
        param_groups.append(without_decay)
        
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    
    return param_groups


class MultiplicativeLR(LambdaLR):
    """
    Learning rate scheduler that applies different multipliers to parameter groups.
    
    This scheduler applies a learning rate schedule to each parameter group individually,
    taking into account custom multipliers defined for each group.
    """
    
    def __init__(self, optimizer: Optimizer, lr_lambda, last_epoch: int = -1):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            lr_lambda: Function or list of functions to calculate learning rate multiplier
            last_epoch: The index of the last epoch
        """
        super().__init__(optimizer, lr_lambda, last_epoch)
        
    def get_lr(self):
        """
        Calculate the learning rate for each parameter group.
        
        Returns:
            list: Learning rates for each parameter group
        """
        if not self._get_lr_called_within_step:
            self._get_lr_called_within_step = True
            
        return [
            base_lr * lmbda(self.last_epoch) * group.get("lr_multiplier", 1.0)
            for lmbda, base_lr, group in zip(self.lr_lambdas, self.base_lrs, self.optimizer.param_groups)
        ]