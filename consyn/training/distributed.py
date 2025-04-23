# consyn/training/distributed.py
"""
Distributed training utilities for Consyn AI models.
This module provides utilities for distributed training across multiple GPUs.
"""

import os
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Union


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Set up distributed training environment.
    
    Args:
        backend: Distributed backend (usually 'nccl' for GPU, 'gloo' for CPU)
        init_method: URL to coordinate processes (defaults to env vars)
        rank: Global rank of this process (defaults to environment variable)
        world_size: Total number of processes (defaults to environment variable)
        local_rank: Local rank of this process (defaults to environment variable)
        
    Returns:
        tuple: (local_rank, global_rank, world_size)
    """
    # Get rank from environment if not specified
    if rank is None:
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
        elif "SLURM_PROCID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
        else:
            rank = 0
            
    # Get world size from environment if not specified
    if world_size is None:
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        elif "SLURM_NTASKS" in os.environ:
            world_size = int(os.environ["SLURM_NTASKS"])
        else:
            world_size = 1
            
    # Get local rank from environment if not specified
    if local_rank is None:
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            local_rank = rank % torch.cuda.device_count()
            
    # Set the device
    torch.cuda.set_device(local_rank)
    
    # Initialize the process group
    init_method = init_method or "env://"
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    
    return local_rank, rank, world_size


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    """Get the total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank():
    """Get the global rank of this process."""
    return dist.get_rank() if dist.is_initialized() else 0


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    """
    Perform all-reduce operation on a tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation (default: sum)
        
    Returns:
        torch.Tensor: Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
        
    # Clone tensor to avoid modifying the original
    result = tensor.clone()
    dist.all_reduce(result, op=op)
    return result


def all_gather(tensor):
    """
    Gather tensors from all processes and concatenate them.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        torch.Tensor: Gathered tensor
    """
    if not dist.is_initialized():
        return tensor.unsqueeze(0)
        
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # List to store gathered tensors
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Gather tensors from all processes
    dist.all_gather(gathered_tensors, tensor)
    
    return torch.stack(gathered_tensors)


def reduce_dict(data_dict, average=True):
    """
    Reduce a dictionary of tensors across all processes.
    
    Args:
        data_dict: Dictionary of tensors to reduce
        average: Whether to average the values (otherwise sum)
        
    Returns:
        dict: Reduced dictionary
    """
    if not dist.is_initialized():
        return data_dict
        
    world_size = dist.get_world_size()
    
    # If empty dictionary, return as is
    if len(data_dict) == 0:
        return data_dict
        
    # Extract keys and values
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    
    # Ensure all values are tensors on CUDA
    values = [v.to(torch.cuda.current_device()) if isinstance(v, torch.Tensor) else torch.tensor(v).to(torch.cuda.current_device()) for v in values]
    
    # Flatten and concatenate all tensors
    values = torch.stack([v.flatten()[0] for v in values])
    
    # All-reduce
    dist.all_reduce(values)
    
    if average:
        values /= world_size
        
    # Reconstruct dictionary
    reduced_dict = {k: v for k, v in zip(keys, values)}
    
    return reduced_dict


def DistributedDataParallel(
    model,
    device_ids=None,
    output_device=None,
    dim=0,
    broadcast_buffers=True,
    find_unused_parameters=False,
):
    """
    Wrap model in DistributedDataParallel.
    
    This is a helper function to handle imports and common options.
    
    Args:
        model: Model to wrap
        device_ids: List of device IDs
        output_device: Device to output to
        dim: Dimension to split inputs
        broadcast_buffers: Whether to broadcast buffers
        find_unused_parameters: Whether to find unused parameters
        
    Returns:
        DistributedDataParallel: Wrapped model
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    # If device_ids is None, use current device
    if device_ids is None:
        device_ids = [torch.cuda.current_device()]
        
    # If output_device is None, use first device
    if output_device is None:
        output_device = device_ids[0]
        
    # Wrap model in DDP
    return DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        dim=dim,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
    )


def prepare_model_for_distributed(model, local_rank):
    """
    Prepare model for distributed training.
    
    Args:
        model: Model to prepare
        local_rank: Local rank of this process
        
    Returns:
        torch.nn.Module: Prepared model
    """
    # Move model to device
    model = model.to(f"cuda:{local_rank}")
    
    # Wrap model in DistributedDataParallel
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    
    return model


def prepare_dataloader_for_distributed(
    dataloader,
    num_replicas=None,
    rank=None,
    seed=42,
):
    """
    Prepare dataloader for distributed training.
    
    Args:
        dataloader: Dataloader to prepare
        num_replicas: Number of replicas (defaults to world size)
        rank: Rank of this process (defaults to global rank)
        seed: Random seed for sampler
        
    Returns:
        torch.utils.data.DataLoader: Prepared dataloader
    """
    from torch.utils.data import DistributedSampler, DataLoader
    
    # If not distributed, return as is
    if not dist.is_initialized():
        return dataloader
        
    # Get rank and world size if not specified
    if num_replicas is None:
        num_replicas = dist.get_world_size()
    if rank is None:
        rank = dist.get_rank()
        
    # Create distributed sampler
    sampler = DistributedSampler(
        dataloader.dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
        seed=seed,
    )
    
    # Create new dataloader with distributed sampler
    new_dataloader = DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
    )
    
    return new_dataloader
