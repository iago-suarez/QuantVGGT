#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Export script for VGGT mobile deployment.
#
# Usage:
#   python -m mobile_export.export_tflite --help
#   python -m mobile_export.export_tflite --output vggt_depth.tflite

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_to_tflite(
    model_name: str = "facebook/VGGT-1B",
    output_path: str = "vggt_depth.tflite",
    num_views: int = 2,
    image_size: int = 518,
    quantize: bool = False,
    validate: bool = True,
) -> Path:
    """
    Export VGGT depth model to TFLite format.
    
    This function:
    1. Loads the VGGT model and wraps it for depth-only output
    2. Converts attention to exportable format
    3. Exports via torch.export
    4. Converts to TFLite using AI Edge Torch
    
    Args:
        model_name: HuggingFace model name or local path
        output_path: Output TFLite file path
        num_views: Number of input views (fixed for export)
        image_size: Input image size (fixed for export)
        quantize: Whether to apply INT8 post-training quantization
        validate: Whether to validate the exported model
        
    Returns:
        Path to the exported TFLite file
    """
    from mobile_export.wrapper import (
        VGGTDepthOnlyWrapperExportable,
        prepare_for_export,
        get_sample_input,
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and wrap model
    logger.info(f"Loading model: {model_name}")
    wrapper = VGGTDepthOnlyWrapperExportable.from_pretrained(
        model_name=model_name,
        num_views=num_views,
        image_size=image_size,
        return_confidence=False,
        device="cpu",
    )
    wrapper = prepare_for_export(wrapper)
    logger.info("Model loaded and wrapped for depth-only output")
    
    # Step 2: Create sample input
    sample_input = get_sample_input(num_views=num_views, image_size=image_size, device="cpu")
    logger.info(f"Sample input shape: {sample_input.shape}")
    
    # Step 3: Test PyTorch inference first
    logger.info("Testing PyTorch inference...")
    with torch.no_grad():
        pytorch_output = wrapper(sample_input)
    logger.info(f"PyTorch output shape: {pytorch_output.shape}")
    
    # Step 4: Export via torch.export
    logger.info("Exporting via torch.export...")
    try:
        exported_program = torch.export.export(wrapper, (sample_input,))
        logger.info("torch.export successful")
    except Exception as e:
        logger.error(f"torch.export failed: {e}")
        logger.error("Attempting to identify unsupported ops...")
        _diagnose_export_failure(wrapper, sample_input)
        raise
    
    # Step 5: Convert to TFLite via AI Edge Torch
    logger.info("Converting to TFLite via AI Edge Torch...")
    try:
        import ai_edge_torch
        
        # Convert to TFLite
        edge_model = ai_edge_torch.convert(exported_program, (sample_input,))
        
        # Apply quantization if requested
        if quantize:
            logger.info("Applying INT8 quantization...")
            # Note: Requires calibration data for best results
            edge_model = ai_edge_torch.quantize(
                edge_model,
                ai_edge_torch.quantization.Quantization.INT8,
            )
        
        # Save TFLite model
        edge_model.export(str(output_path))
        logger.info(f"TFLite model saved to: {output_path}")
        
    except ImportError:
        logger.warning("ai_edge_torch not installed. Attempting ONNX → TFLite path...")
        _export_via_onnx(wrapper, sample_input, output_path, quantize)
    
    # Step 6: Validate exported model
    if validate:
        _validate_tflite(output_path, sample_input, pytorch_output)
    
    return output_path


def _export_via_onnx(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    output_path: Path,
    quantize: bool = False,
) -> None:
    """
    Fallback export path: PyTorch → ONNX → TFLite.
    
    This is less optimal than AI Edge Torch but works when that's not available.
    """
    import tempfile
    
    onnx_path = output_path.with_suffix(".onnx")
    
    logger.info(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        sample_input,
        str(onnx_path),
        input_names=["images"],
        output_names=["depth"],
        dynamic_axes=None,  # Fixed shapes for mobile
        opset_version=17,
    )
    logger.info("ONNX export successful")
    
    # Convert ONNX to TFLite
    logger.info("Converting ONNX to TFLite...")
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        
        # Load ONNX model
        onnx_model = onnx.load(str(onnx_path))
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_path = output_path.with_suffix("")
        tf_rep.export_graph(str(tf_path))
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        
        tflite_model = converter.convert()
        
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite model saved to: {output_path}")
        
    except ImportError as e:
        logger.error(f"ONNX → TFLite conversion requires: onnx, onnx-tf, tensorflow")
        logger.error(f"Install with: pip install onnx onnx-tf tensorflow")
        raise


def _diagnose_export_failure(model: torch.nn.Module, sample_input: torch.Tensor) -> None:
    """
    Diagnose why torch.export failed by checking for unsupported operations.
    """
    logger.info("Diagnosing export failure...")
    
    # Try to trace the model
    try:
        traced = torch.jit.trace(model, sample_input)
        logger.info("torch.jit.trace succeeded - issue may be with dynamic shapes")
    except Exception as e:
        logger.error(f"torch.jit.trace also failed: {e}")
    
    # List potentially problematic ops
    problematic_ops = [
        "F.scaled_dot_product_attention",
        "einops.rearrange",
        "torch.compile",
        "xformers",
    ]
    
    logger.info("Check for these potentially problematic operations:")
    for op in problematic_ops:
        logger.info(f"  - {op}")


def _validate_tflite(
    tflite_path: Path,
    sample_input: torch.Tensor,
    pytorch_output: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Validate TFLite model output against PyTorch reference.
    """
    logger.info("Validating TFLite model...")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"TFLite input shape: {input_details[0]['shape']}")
        logger.info(f"TFLite output shape: {output_details[0]['shape']}")
        
        # Run inference
        input_data = sample_input.numpy()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Compare outputs
        pytorch_np = pytorch_output.numpy()
        
        if np.allclose(tflite_output, pytorch_np, rtol=rtol, atol=atol):
            logger.info("✓ TFLite output matches PyTorch output")
            return True
        else:
            max_diff = np.max(np.abs(tflite_output - pytorch_np))
            mean_diff = np.mean(np.abs(tflite_output - pytorch_np))
            logger.warning(f"⚠ Output mismatch - max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")
            return False
            
    except ImportError:
        logger.warning("TensorFlow not installed - skipping TFLite validation")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Export VGGT depth model to TFLite for Android deployment"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/VGGT-1B",
        help="HuggingFace model name or local checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vggt_depth.tflite",
        help="Output TFLite file path",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=2,
        help="Number of input views (fixed for export)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=518,
        help="Input image size (fixed for export)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 post-training quantization",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of exported model",
    )
    
    args = parser.parse_args()
    
    try:
        output_path = export_to_tflite(
            model_name=args.model,
            output_path=args.output,
            num_views=args.num_views,
            image_size=args.image_size,
            quantize=args.quantize,
            validate=not args.no_validate,
        )
        
        logger.info(f"Export complete: {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
