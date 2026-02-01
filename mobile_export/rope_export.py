# Copyright (c) Meta Platforms, Inc. and affiliates.
# Static RoPE implementation for TFLite export.
#
# This module provides a static (non-caching) version of RoPE that is
# compatible with torch.export and TFLite conversion.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class StaticRoPE2D(nn.Module):
    """
    Static 2D Rotary Position Embedding for TFLite export.
    
    Unlike the original RoPE implementation, this version:
    - Pre-computes all frequency components at initialization
    - Does not use dynamic caching
    - Has fixed sequence length
    
    Args:
        dim: Feature dimension per head (must be even)
        max_seq_len: Maximum sequence length (num_patches)
        frequency: Base frequency for position embeddings
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 1370,  # 37*37 patches for 518px image
        frequency: float = 100.0,
    ):
        super().__init__()
        
        assert dim % 2 == 0, "Feature dimension must be even for RoPE"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.frequency = frequency
        
        # Pre-compute frequency components
        half_dim = dim // 4  # Split between x and y dimensions
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (frequency ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute position indices for maximum size
        # Will be sliced at runtime based on actual size
        positions = torch.arange(0, int(max_seq_len ** 0.5) + 1, dtype=torch.float32)
        self.register_buffer("positions", positions, persistent=False)
    
    def _compute_rope_components(
        self,
        seq_len: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute cosine and sine components for RoPE.
        
        Args:
            seq_len: Total sequence length (height * width)
            height: Grid height in patches
            width: Grid width in patches
            device: Target device
            dtype: Target dtype
            
        Returns:
            Tuple of (cos, sin) tensors, each of shape (1, 1, seq_len, dim)
        """
        # Get position indices
        y_pos = self.positions[:height].to(dtype)
        x_pos = self.positions[:width].to(dtype)
        
        # Compute position embeddings for y and x
        y_emb = y_pos.unsqueeze(1) * self.inv_freq.unsqueeze(0).to(dtype)  # (H, half_dim//2)
        x_emb = x_pos.unsqueeze(1) * self.inv_freq.unsqueeze(0).to(dtype)  # (W, half_dim//2)
        
        # Create full position embedding grid
        # y varies along rows, x varies along columns
        y_emb_full = y_emb.unsqueeze(1).expand(-1, width, -1)  # (H, W, half_dim//2)
        x_emb_full = x_emb.unsqueeze(0).expand(height, -1, -1)  # (H, W, half_dim//2)
        
        # Concatenate y and x embeddings
        emb = torch.cat([y_emb_full, x_emb_full], dim=-1)  # (H, W, half_dim)
        emb = emb.reshape(seq_len, -1)  # (H*W, half_dim)
        
        # Double for cos and sin pairing
        emb = torch.cat([emb, emb], dim=-1)  # (H*W, dim)
        
        # Compute cos and sin
        cos = torch.cos(emb).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        sin = torch.sin(emb).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        
        return cos.to(device), sin.to(device)
    
    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor of shape (B, num_heads, N, head_dim)
            pos: Position tensor of shape (B, N, 2) containing (y, x) coordinates
            
        Returns:
            Tensor with RoPE applied, same shape as input
        """
        B, num_heads, N, head_dim = x.shape
        
        # Infer grid size from positions
        # Note: For export, we assume standard grid without special tokens
        max_y = pos[:, :, 0].max().int().item() + 1
        max_x = pos[:, :, 1].max().int().item() + 1
        
        # Get precomputed cos/sin or compute on the fly
        cos, sin = self._compute_rope_components(
            N, max_y, max_x, x.device, x.dtype
        )
        
        # Apply rotary embedding
        return self._apply_rope(x, cos, sin)
    
    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Apply rotary embedding using the rotation formula.
        
        Args:
            x: Input tensor of shape (B, num_heads, N, dim)
            cos: Cosine tensor of shape (1, 1, N, dim)
            sin: Sine tensor of shape (1, 1, N, dim)
            
        Returns:
            Rotated tensor of same shape
        """
        # Split into two halves for rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Get corresponding cos/sin
        cos1, cos2 = cos[..., ::2], cos[..., 1::2]
        sin1, sin2 = sin[..., ::2], sin[..., 1::2]
        
        # Apply rotation: [cos, -sin; sin, cos] @ [x1; x2]
        out1 = x1 * cos1 - x2 * sin1
        out2 = x1 * sin2 + x2 * cos2
        
        # Interleave results
        out = torch.stack([out1, out2], dim=-1).flatten(-2)
        
        return out
    
    @classmethod
    def from_original(cls, original_rope: nn.Module, max_seq_len: int = 1370) -> "StaticRoPE2D":
        """
        Create StaticRoPE2D from original RotaryPositionEmbedding2D.
        
        Args:
            original_rope: Original RoPE module
            max_seq_len: Maximum sequence length to support
            
        Returns:
            StaticRoPE2D instance
        """
        # Get dim from the original module's frequency cache or infer from usage
        # This is a simplified version - may need adjustment based on actual usage
        return cls(
            dim=64,  # Default head_dim for VGGT
            max_seq_len=max_seq_len,
            frequency=original_rope.base_frequency,
        )


def convert_rope_modules(model: nn.Module, max_seq_len: int = 1370) -> nn.Module:
    """
    Convert all RoPE modules in a model to static versions.
    
    Args:
        model: Model containing RoPE modules
        max_seq_len: Maximum sequence length for static RoPE
        
    Returns:
        Model with converted RoPE modules
    """
    from vggt.layers.rope import RotaryPositionEmbedding2D
    
    for name, module in model.named_modules():
        if isinstance(module, RotaryPositionEmbedding2D):
            # Get parent module
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            # Replace with static version
            static_rope = StaticRoPE2D.from_original(module, max_seq_len)
            setattr(parent, attr_name, static_rope)
    
    return model
