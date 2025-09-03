#!/usr/bin/env python3
"""
Distributed training utilities for multi-GPU training.
"""

import functools
import builtins
import torch
import torch.distributed as dist
from typing import Tuple

def setup_distributed(backend: str = "nccl") -> Tuple[int, int]:
    """
    Initialize distributed training.
    
    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
        
    Returns:
        rank, world_size
    """
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

def is_main(rank: int) -> bool:
    """Check if current process is the main process."""
    return rank == 0

def suppress_non_main_print(rank: int):
    """Suppress print statements from non-main processes."""
    if not is_main(rank):
        builtins.print = functools.partial(lambda *a, **k: None)

@torch.no_grad()
def concat_all_gather(t: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all ranks and concatenate on dim=0.
    This version is used only on detached inputs so it won't carry a graph.
    
    Args:
        t: Tensor to gather [B, D]
        
    Returns:
        Concatenated tensor from all ranks [B*world, D]
    """
    world = dist.get_world_size()
    tensors_gather = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(tensors_gather, t.contiguous())
    return torch.cat(tensors_gather, dim=0)

def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_effective_lr(base_lr: float, world_size: int) -> float:
    """
    Get effective learning rate for distributed training.
    
    Args:
        base_lr: Base learning rate
        world_size: Number of GPUs
        
    Returns:
        Effective learning rate
    """
    return base_lr * world_size

def get_batch_size_per_gpu(total_batch_size: int, world_size: int) -> int:
    """
    Get batch size per GPU for distributed training.
    
    Args:
        total_batch_size: Total batch size across all GPUs
        world_size: Number of GPUs
        
    Returns:
        Batch size per GPU
    """
    return total_batch_size // world_size 