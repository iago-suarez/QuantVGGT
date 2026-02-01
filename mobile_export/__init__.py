# Copyright (c) Meta Platforms, Inc. and affiliates.
# Mobile export utilities for VGGT model.
#
# This module provides tools to convert VGGT to TFLite-compatible format.

from .wrapper import (
    VGGTDepthOnlyWrapper,
    VGGTDepthOnlyWrapperExportable,
    prepare_for_export,
    get_sample_input,
)
from .attention_export import ExportableAttention, convert_attention_modules
from .einops_free import (
    rearrange_bhwc_to_bchw,
    rearrange_bchw_to_bhwc,
    merge_batch_frames,
    split_batch_frames,
    tokens_to_spatial,
    spatial_to_tokens,
)
from .rope_export import StaticRoPE2D, convert_rope_modules
from .op_audit import OperatorAuditor, audit_model

__all__ = [
    # Wrappers
    "VGGTDepthOnlyWrapper",
    "VGGTDepthOnlyWrapperExportable",
    "prepare_for_export",
    "get_sample_input",
    # Attention
    "ExportableAttention",
    "convert_attention_modules",
    # Tensor operations
    "rearrange_bhwc_to_bchw",
    "rearrange_bchw_to_bhwc",
    "merge_batch_frames",
    "split_batch_frames",
    "tokens_to_spatial",
    "spatial_to_tokens",
    # RoPE
    "StaticRoPE2D",
    "convert_rope_modules",
    # Auditing
    "OperatorAuditor",
    "audit_model",
]
