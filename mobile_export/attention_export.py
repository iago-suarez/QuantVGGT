# Copyright (c) Meta Platforms, Inc. and affiliates.
# Exportable attention module without F.scaled_dot_product_attention.
#
# This module provides a TFLite-compatible attention implementation that
# decomposes the fused SDPA into explicit Q@K -> softmax -> @V operations.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class ExportableAttention(nn.Module):
    """
    TFLite-exportable attention module.
    
    Replaces F.scaled_dot_product_attention with manual implementation:
    - Q @ K.T -> scale -> softmax -> @ V
    
    This is compatible with torch.export and TFLite conversion.
    
    Args:
        dim: Input embedding dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to include bias in QKV projection
        proj_bias: Whether to include bias in output projection
        attn_drop: Attention dropout rate (disabled during export)
        proj_drop: Projection dropout rate (disabled during export)
        qk_norm: Whether to apply LayerNorm to Q and K
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        rope: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        
    def forward(self, x: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with explicit attention computation.
        
        Args:
            x: Input tensor of shape (B, N, C)
            pos: Optional position tensor for RoPE
            
        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        
        # QKV projection: (B, N, 3*C) -> (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply RoPE if provided
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        
        # Manual scaled dot-product attention (TFLite compatible)
        # Step 1: Q @ K^T with scaling
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, N, N)
        
        # Step 2: Softmax
        attn = torch.softmax(attn, dim=-1)
        
        # Step 3: Dropout (will be identity during export/inference)
        attn = self.attn_drop(attn)
        
        # Step 4: Attention @ V
        x = torch.matmul(attn, v)  # (B, num_heads, N, head_dim)
        
        # Reshape back: (B, num_heads, N, head_dim) -> (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    @classmethod
    def from_original(cls, original_attention: nn.Module) -> "ExportableAttention":
        """
        Convert an original Attention module to ExportableAttention.
        
        Copies all weights and configuration from the original module.
        
        Args:
            original_attention: Original Attention module from vggt.layers.attention
            
        Returns:
            ExportableAttention with copied weights
        """
        # Create new module with same config
        exportable = cls(
            dim=original_attention.qkv.in_features,
            num_heads=original_attention.num_heads,
            qkv_bias=original_attention.qkv.bias is not None,
            proj_bias=original_attention.proj.bias is not None,
            attn_drop=original_attention.attn_drop.p,
            proj_drop=original_attention.proj_drop.p,
            qk_norm=not isinstance(original_attention.q_norm, nn.Identity),
            rope=original_attention.rope,
        )
        
        # Copy weights
        exportable.qkv.load_state_dict(original_attention.qkv.state_dict())
        exportable.proj.load_state_dict(original_attention.proj.state_dict())
        
        # Copy norm weights if they exist
        if not isinstance(original_attention.q_norm, nn.Identity):
            exportable.q_norm.load_state_dict(original_attention.q_norm.state_dict())
            exportable.k_norm.load_state_dict(original_attention.k_norm.state_dict())
        
        return exportable


def convert_attention_modules(model: nn.Module) -> nn.Module:
    """
    Recursively replace all Attention modules with ExportableAttention.
    
    This function walks through the model and replaces any module that has
    'fused_attn' attribute (indicating it's an original Attention module)
    with an ExportableAttention version.
    
    Args:
        model: PyTorch model containing Attention modules
        
    Returns:
        Model with all Attention modules replaced
    """
    from vggt.layers.attention import Attention
    
    for name, module in model.named_children():
        if isinstance(module, Attention):
            # Replace with exportable version
            exportable = ExportableAttention.from_original(module)
            setattr(model, name, exportable)
        else:
            # Recurse into children
            convert_attention_modules(module)
    
    return model
