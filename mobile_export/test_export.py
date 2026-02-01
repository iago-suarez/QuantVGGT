#!/usr/bin/env python3
"""
Test VGGT export to TFLite using LiteRT Torch.

This script tests the export pipeline step by step:
1. Load VGGT model
2. Wrap with VGGTDepthOnlyWrapper
3. Test PyTorch inference
4. Export to TFLite (FP32)
5. Validate TFLite output
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_pytorch_inference(wrapper: nn.Module, sample_input: torch.Tensor) -> torch.Tensor:
    """Test PyTorch inference and return output."""
    print("\n=== Step 1: PyTorch Inference ===")
    
    wrapper.eval()
    with torch.no_grad():
        start = time.time()
        output = wrapper(sample_input)
        elapsed = time.time() - start
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Inference time: {elapsed*1000:.1f}ms")
    
    return output


def test_torch_export(wrapper: nn.Module, sample_input: torch.Tensor):
    """Test torch.export.export() - the first step of TFLite conversion."""
    print("\n=== Step 2: torch.export.export() ===")
    
    try:
        exported = torch.export.export(
            wrapper,
            (sample_input,),
            strict=False,  # Allow some non-strict behaviors
        )
        print("✓ torch.export.export() succeeded!")
        print(f"Exported graph inputs: {[str(spec) for spec in exported.graph_signature.input_specs][:3]}...")
        return exported
    except Exception as e:
        print(f"✗ torch.export.export() failed: {e}")
        
        # Try to get more details
        print("\nAttempting to identify problematic operations...")
        try:
            # Try with dynamo tracing for better error messages
            from torch._dynamo import explain
            explanation = explain(wrapper, sample_input)
            print(f"Graph breaks: {explanation.break_reasons}")
        except Exception as e2:
            print(f"Could not get detailed explanation: {e2}")
        
        return None


def test_litert_torch_export(wrapper: nn.Module, sample_input: torch.Tensor, output_path: str):
    """Test LiteRT Torch conversion to TFLite."""
    print("\n=== Step 3: LiteRT Torch Conversion ===")
    
    try:
        import litert_torch
        
        # Convert to TFLite
        print("Converting to TFLite...")
        start = time.time()
        
        tflite_model = litert_torch.convert(
            wrapper,
            (sample_input,),
        )
        
        elapsed = time.time() - start
        print(f"Conversion time: {elapsed:.1f}s")
        
        # Save the model
        tflite_model.export(output_path)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ TFLite model saved: {output_path} ({file_size:.1f} MB)")
        
        return tflite_model
        
    except Exception as e:
        print(f"✗ LiteRT Torch conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_tflite_inference(tflite_path: str, sample_input: np.ndarray) -> np.ndarray:
    """Test TFLite inference and return output."""
    print("\n=== Step 4: TFLite Inference ===")
    
    try:
        import tensorflow as tf
        
        # Load the model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input details: {input_details[0]['shape']}, {input_details[0]['dtype']}")
        print(f"Output details: {output_details[0]['shape']}, {output_details[0]['dtype']}")
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], sample_input)
        
        start = time.time()
        interpreter.invoke()
        elapsed = time.time() - start
        
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"TFLite output shape: {output.shape}")
        print(f"TFLite output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"TFLite inference time: {elapsed*1000:.1f}ms")
        
        return output
        
    except Exception as e:
        print(f"✗ TFLite inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_outputs(pytorch_output: torch.Tensor, tflite_output: np.ndarray):
    """Compare PyTorch and TFLite outputs."""
    print("\n=== Step 5: Output Comparison ===")
    
    # Convert PyTorch output to numpy
    pytorch_np = pytorch_output.detach().cpu().numpy()
    
    # Check shapes
    if pytorch_np.shape != tflite_output.shape:
        print(f"⚠ Shape mismatch: PyTorch {pytorch_np.shape} vs TFLite {tflite_output.shape}")
        return
    
    # Compute metrics
    abs_diff = np.abs(pytorch_np - tflite_output)
    rel_diff = abs_diff / (np.abs(pytorch_np) + 1e-6)
    
    print(f"Absolute error: max={abs_diff.max():.6f}, mean={abs_diff.mean():.6f}")
    print(f"Relative error: max={rel_diff.max():.4f}, mean={rel_diff.mean():.4f}")
    
    # Correlation
    corr = np.corrcoef(pytorch_np.flatten(), tflite_output.flatten())[0, 1]
    print(f"Correlation: {corr:.6f}")
    
    if corr > 0.99:
        print("✓ Outputs match well (correlation > 0.99)")
    elif corr > 0.95:
        print("⚠ Outputs reasonably close (correlation > 0.95)")
    else:
        print("✗ Outputs diverge significantly (correlation < 0.95)")


def create_simple_test_wrapper():
    """Create a minimal test model for debugging export issues."""
    
    class SimpleDepthModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 1, 3, padding=1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # x: (B, S, C, H, W) -> process each view
            B, S, C, H, W = x.shape
            x = x.view(B * S, C, H, W)
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            x = x.view(B, S, H, W, 1)
            return x
    
    return SimpleDepthModel()


def main():
    parser = argparse.ArgumentParser(description="Test VGGT TFLite export")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/VGGT-1B",
        help="Model name or path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vggt_depth_fp32.tflite",
        help="Output TFLite file path",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=2,
        help="Number of input views",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=518,
        help="Input image size",
    )
    parser.add_argument(
        "--simple-test",
        action="store_true",
        help="Test with a simple model first",
    )
    parser.add_argument(
        "--skip-tflite",
        action="store_true",
        help="Skip TFLite conversion, only test torch.export",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VGGT TFLite Export Test")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Create sample input
    sample_input = torch.randn(
        1, args.num_views, 3, args.image_size, args.image_size,
        dtype=torch.float32,
        device=args.device,
    )
    print(f"\nSample input shape: {sample_input.shape}")
    
    if args.simple_test:
        print("\n>>> Testing with SIMPLE model first <<<")
        wrapper = create_simple_test_wrapper().to(args.device)
    else:
        print(f"\n>>> Loading VGGT model: {args.model} <<<")
        from mobile_export.wrapper import VGGTDepthOnlyWrapper
        wrapper = VGGTDepthOnlyWrapper.from_pretrained(
            args.model,
            num_views=args.num_views,
            image_size=args.image_size,
            device=args.device,
        )
    
    # Step 1: PyTorch inference
    pytorch_output = test_pytorch_inference(wrapper, sample_input)
    
    # Step 2: torch.export test
    exported = test_torch_export(wrapper, sample_input)
    
    if args.skip_tflite:
        print("\n>>> Skipping TFLite conversion (--skip-tflite) <<<")
        return
    
    # Step 3: LiteRT Torch export
    tflite_model = test_litert_torch_export(wrapper, sample_input, args.output)
    
    if tflite_model is None:
        print("\n✗ TFLite export failed, cannot continue")
        return
    
    # Step 4: TFLite inference
    sample_np = sample_input.detach().cpu().numpy()
    tflite_output = test_tflite_inference(args.output, sample_np)
    
    if tflite_output is None:
        print("\n✗ TFLite inference failed")
        return
    
    # Step 5: Compare outputs
    compare_outputs(pytorch_output, tflite_output)
    
    print("\n" + "=" * 60)
    print("Export test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
