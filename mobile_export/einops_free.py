# Copyright (c) Meta Platforms, Inc. and affiliates.
# einops-free tensor operations for TFLite compatibility.
#
# This module provides native PyTorch alternatives to common einops operations.
# These are needed because TFLite doesn't understand einops.rearrange.

import torch
from torch import Tensor


def rearrange_bhwc_to_bchw(x: Tensor) -> Tensor:
    """
    Convert tensor from (B, H, W, C) to (B, C, H, W).
    
    Equivalent to: einops.rearrange(x, 'b h w c -> b c h w')
    
    Args:
        x: Tensor of shape (B, H, W, C)
        
    Returns:
        Tensor of shape (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)


def rearrange_bchw_to_bhwc(x: Tensor) -> Tensor:
    """
    Convert tensor from (B, C, H, W) to (B, H, W, C).
    
    Equivalent to: einops.rearrange(x, 'b c h w -> b h w c')
    
    Args:
        x: Tensor of shape (B, C, H, W)
        
    Returns:
        Tensor of shape (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def rearrange_bsn_to_bns(x: Tensor) -> Tensor:
    """
    Convert tensor from (B, S, N) to (B, N, S).
    
    Equivalent to: einops.rearrange(x, 'b s n -> b n s')
    
    Args:
        x: Tensor of shape (B, S, N)
        
    Returns:
        Tensor of shape (B, N, S)
    """
    return x.permute(0, 2, 1)


def merge_batch_frames(x: Tensor) -> Tensor:
    """
    Merge batch and frame dimensions: (B, S, C, H, W) -> (B*S, C, H, W).
    
    Equivalent to: einops.rearrange(x, 'b s c h w -> (b s) c h w')
    
    Args:
        x: Tensor of shape (B, S, C, H, W)
        
    Returns:
        Tensor of shape (B*S, C, H, W)
    """
    B, S, C, H, W = x.shape
    return x.reshape(B * S, C, H, W)


def split_batch_frames(x: Tensor, batch_size: int, num_frames: int) -> Tensor:
    """
    Split merged batch-frame dimension: (B*S, C, H, W) -> (B, S, C, H, W).
    
    Equivalent to: einops.rearrange(x, '(b s) c h w -> b s c h w', b=B, s=S)
    
    Args:
        x: Tensor of shape (B*S, C, H, W)
        batch_size: Original batch size B
        num_frames: Number of frames S
        
    Returns:
        Tensor of shape (B, S, C, H, W)
    """
    BS, C, H, W = x.shape
    assert BS == batch_size * num_frames, f"Expected {batch_size * num_frames} but got {BS}"
    return x.reshape(batch_size, num_frames, C, H, W)


def flatten_spatial(x: Tensor) -> Tensor:
    """
    Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W).
    
    Equivalent to: einops.rearrange(x, 'b c h w -> b c (h w)')
    
    Args:
        x: Tensor of shape (B, C, H, W)
        
    Returns:
        Tensor of shape (B, C, H*W)
    """
    B, C, H, W = x.shape
    return x.reshape(B, C, H * W)


def unflatten_spatial(x: Tensor, height: int, width: int) -> Tensor:
    """
    Unflatten to spatial dimensions: (B, C, H*W) -> (B, C, H, W).
    
    Equivalent to: einops.rearrange(x, 'b c (h w) -> b c h w', h=H, w=W)
    
    Args:
        x: Tensor of shape (B, C, H*W)
        height: Target height H
        width: Target width W
        
    Returns:
        Tensor of shape (B, C, H, W)
    """
    B, C, HW = x.shape
    assert HW == height * width, f"Expected {height * width} but got {HW}"
    return x.reshape(B, C, height, width)


def repeat_batch(x: Tensor, repeats: int) -> Tensor:
    """
    Repeat tensor along batch dimension.
    
    Equivalent to: einops.repeat(x, 'b ... -> (b r) ...', r=repeats)
    
    Args:
        x: Tensor of shape (B, ...)
        repeats: Number of times to repeat
        
    Returns:
        Tensor of shape (B*repeats, ...)
    """
    return x.repeat_interleave(repeats, dim=0)


def tokens_to_spatial(x: Tensor, height: int, width: int) -> Tensor:
    """
    Convert token sequence to spatial: (B, N, C) -> (B, H, W, C) where N = H*W.
    
    Equivalent to: einops.rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
    
    Args:
        x: Tensor of shape (B, N, C) where N = H*W
        height: Target height H
        width: Target width W
        
    Returns:
        Tensor of shape (B, H, W, C)
    """
    B, N, C = x.shape
    assert N == height * width, f"Expected {height * width} tokens but got {N}"
    return x.reshape(B, height, width, C)


def spatial_to_tokens(x: Tensor) -> Tensor:
    """
    Convert spatial to token sequence: (B, H, W, C) -> (B, H*W, C).
    
    Equivalent to: einops.rearrange(x, 'b h w c -> b (h w) c')
    
    Args:
        x: Tensor of shape (B, H, W, C)
        
    Returns:
        Tensor of shape (B, H*W, C)
    """
    B, H, W, C = x.shape
    return x.reshape(B, H * W, C)
