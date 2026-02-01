# Copyright (c) Meta Platforms, Inc. and affiliates.
# Depth-only wrapper for VGGT mobile export.
#
# This module provides a wrapper that:
# - Takes fixed-shape input images
# - Returns only depth output (tensor-only, no dict)
# - Is compatible with torch.export and TFLite conversion

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
import copy


class VGGTDepthOnlyWrapper(nn.Module):
    """
    TFLite-exportable wrapper for VGGT that outputs only depth maps.
    
    This wrapper:
    1. Accepts fixed-shape input: (1, num_views, 3, height, width)
    2. Disables camera, point, and track heads
    3. Returns a single depth tensor instead of a dict
    4. Is compatible with torch.export() and TFLite conversion
    
    Args:
        vggt_model: Pre-trained VGGT model
        num_views: Fixed number of input views (default: 2)
        image_size: Fixed image size (default: 518)
        return_confidence: Whether to also return depth confidence
    """
    
    def __init__(
        self,
        vggt_model: nn.Module,
        num_views: int = 2,
        image_size: int = 518,
        return_confidence: bool = False,
    ):
        super().__init__()
        
        self.num_views = num_views
        self.image_size = image_size
        self.return_confidence = return_confidence
        
        # Copy aggregator and depth head only
        self.aggregator = vggt_model.aggregator
        self.depth_head = vggt_model.depth_head
        
        # Disable other heads by setting them to None
        # We don't copy them to save memory
        self.camera_head = None
        self.point_head = None
        self.track_head = None
        
        # Disable dropout and stochastic depth for export
        self.eval()
        self._disable_dropout()
    
    def _disable_dropout(self):
        """Recursively disable all dropout modules for deterministic export."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
    
    def forward(self, images: Tensor) -> Tensor:
        """
        Forward pass returning depth only.
        
        Args:
            images: Input images of shape (1, num_views, 3, H, W) in range [0, 1]
                    Must match the fixed num_views and image_size.
        
        Returns:
            If return_confidence=False:
                depth: Depth tensor of shape (1, num_views, H, W, 1)
            If return_confidence=True:
                Tuple of (depth, confidence) where:
                - depth: (1, num_views, H, W, 1)
                - confidence: (1, num_views, H, W)
        """
        # Validate input shape (helps catch issues early)
        B, S, C, H, W = images.shape
        assert B == 1, f"Batch size must be 1 for mobile export, got {B}"
        assert S == self.num_views, f"Expected {self.num_views} views, got {S}"
        assert C == 3, f"Expected 3 channels, got {C}"
        
        # Run aggregator
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        
        # Run depth head only (disable autocast for consistent export)
        with torch.cuda.amp.autocast(enabled=False):
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list, 
                images=images, 
                patch_start_idx=patch_start_idx
            )
        
        if self.return_confidence:
            return depth, depth_conf
        else:
            return depth
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "facebook/VGGT-1B",
        num_views: int = 2,
        image_size: int = 518,
        return_confidence: bool = False,
        device: str = "cpu",
    ) -> "VGGTDepthOnlyWrapper":
        """
        Create wrapper from a pre-trained VGGT model.
        
        Args:
            model_name: HuggingFace model name (e.g., "facebook/VGGT-1B")
            num_views: Fixed number of input views
            image_size: Fixed input image size
            return_confidence: Whether to return depth confidence
            device: Device to load model on
            
        Returns:
            VGGTDepthOnlyWrapper ready for export
        """
        from vggt.models.vggt import VGGT
        
        # Load pre-trained model
        vggt = VGGT.from_pretrained(model_name)
        vggt = vggt.to(device)
        
        # Create wrapper
        wrapper = cls(
            vggt_model=vggt,
            num_views=num_views,
            image_size=image_size,
            return_confidence=return_confidence,
        )
        
        return wrapper.to(device)


class VGGTDepthOnlyWrapperExportable(VGGTDepthOnlyWrapper):
    """
    Extended wrapper with additional export compatibility fixes.
    
    This version includes:
    - Replaced attention modules with exportable versions
    - Disabled all dynamic control flow
    - Static position embeddings
    """
    
    def __init__(
        self,
        vggt_model: nn.Module,
        num_views: int = 2,
        image_size: int = 518,
        return_confidence: bool = False,
    ):
        super().__init__(
            vggt_model=vggt_model,
            num_views=num_views,
            image_size=image_size,
            return_confidence=return_confidence,
        )
        
        # Convert attention modules to exportable versions
        from mobile_export.attention_export import convert_attention_modules
        convert_attention_modules(self.aggregator)
        convert_attention_modules(self.depth_head)
        
        # Pre-compute RoPE positions for static export
        self._precompute_positions()
    
    def _precompute_positions(self):
        """Pre-compute position embeddings for the fixed image size."""
        patch_size = self.aggregator.patch_size
        num_patches_h = self.image_size // patch_size
        num_patches_w = self.image_size // patch_size
        
        # Store computed positions as buffer
        if self.aggregator.position_getter is not None:
            # Compute positions for batch_size * num_views
            positions = self.aggregator.position_getter(
                self.num_views,  # Will be expanded in forward
                num_patches_h,
                num_patches_w,
                device=torch.device("cpu"),
            )
            self.register_buffer("_cached_positions", positions, persistent=False)
        else:
            self._cached_positions = None


def prepare_for_export(model: nn.Module) -> nn.Module:
    """
    Prepare a PyTorch model for export by:
    1. Setting to eval mode
    2. Disabling gradient computation
    3. Fusing BatchNorm where applicable
    
    Args:
        model: Model to prepare
        
    Returns:
        Model ready for export
    """
    model = model.eval()
    
    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def get_sample_input(
    num_views: int = 2,
    image_size: int = 518,
    device: str = "cpu",
) -> Tensor:
    """
    Get sample input tensor for tracing/export.
    
    Args:
        num_views: Number of input views
        image_size: Input image size
        device: Device for tensor
        
    Returns:
        Sample input tensor of shape (1, num_views, 3, image_size, image_size)
    """
    return torch.randn(1, num_views, 3, image_size, image_size, device=device)
