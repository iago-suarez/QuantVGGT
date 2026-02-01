#!/usr/bin/env python3
"""
QuantVGGT vs VGGT Comparison Test

This script compares the accuracy and performance of:
1. Original VGGT model (FP32/FP16)
2. QuantVGGT W4A4 quantized model

It downloads sample multi-view images and evaluates:
- Depth prediction quality
- Camera pose estimation accuracy
- Inference speed
- Memory usage
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Sample multi-view image URLs (from public datasets)
SAMPLE_IMAGE_URLS = {
    "co3d_hydrant": [
        "https://raw.githubusercontent.com/facebookresearch/co3d/main/co3d/examples/apple/110_13051_23361/images/frame000001.jpg",
        "https://raw.githubusercontent.com/facebookresearch/co3d/main/co3d/examples/apple/110_13051_23361/images/frame000010.jpg",
    ],
    # Fallback to any available multi-view images
    "mvimgnet_sample": [
        "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=518",
        "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=518",  # Same image, different crop
    ],
}


def download_sample_images(output_dir: Path, num_views: int = 2) -> List[str]:
    """
    Download sample images for testing.
    Returns list of local file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    
    # Try to download real multi-view images from the web
    logger.info("Downloading sample images...")
    
    # URLs for sample images (from free image sources)
    sample_urls = [
        # Different angles of similar scenes
        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=518&h=518&fit=crop",
        "https://images.unsplash.com/photo-1558618047-f4b511ef7a7e?w=518&h=518&fit=crop",
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=518&h=518&fit=crop",
        "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=518&h=518&fit=crop",
    ]
    
    for i in range(num_views):
        img_path = output_dir / f"test_view_{i:03d}.jpg"
        
        if not img_path.exists():
            # Try to download from URL first
            url_idx = i % len(sample_urls)
            try:
                logger.info(f"Downloading image {i+1}/{num_views} from web...")
                response = requests.get(sample_urls[url_idx], timeout=10)
                if response.status_code == 200:
                    # Save and verify it's a valid image
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    # Verify and resize
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    img = img.resize((518, 518), Image.Resampling.LANCZOS)
                    img.save(img_path, "JPEG", quality=95)
                    logger.info(f"  Downloaded and resized to 518x518")
                else:
                    raise Exception(f"HTTP {response.status_code}")
            except Exception as e:
                logger.info(f"  Download failed ({e}), creating synthetic image")
                img = create_synthetic_multiview_image(i, num_views)
                img.save(img_path, "JPEG", quality=95)
        
        image_paths.append(str(img_path))
    
    logger.info(f"Prepared {len(image_paths)} test images")
    return image_paths


def create_synthetic_multiview_image(view_idx: int, total_views: int, size: int = 518) -> Image.Image:
    """
    Create a synthetic multi-view test image with geometric patterns.
    Different views have slightly different perspectives of the same "scene".
    """
    img = Image.new('RGB', (size, size), color=(100, 100, 100))
    pixels = img.load()
    
    # Create a 3D-like pattern that varies with view angle
    angle_offset = (view_idx / total_views) * 0.3  # Simulate camera rotation
    
    for y in range(size):
        for x in range(size):
            # Normalized coordinates
            nx = (x - size/2) / size
            ny = (y - size/2) / size
            
            # Apply perspective shift based on view
            nx_shifted = nx + angle_offset * (1 - abs(ny))
            
            # Create depth-like pattern (sphere)
            r = np.sqrt(nx_shifted**2 + ny**2)
            
            if r < 0.4:
                # Sphere surface
                depth = np.sqrt(0.16 - r**2)
                intensity = int(150 + 100 * depth)
                pixels[x, y] = (intensity, intensity - 20, intensity - 40)
            elif r < 0.42:
                # Edge
                pixels[x, y] = (50, 50, 50)
            else:
                # Background with grid
                grid = ((x // 40) + (y // 40)) % 2
                base = 80 + grid * 40
                pixels[x, y] = (base, base, base)
    
    return img


def load_model(model_type: str, device: str, quantized_path: Optional[str] = None) -> nn.Module:
    """
    Load either the original VGGT or QuantVGGT model.
    
    Args:
        model_type: 'fp32', 'fp16', or 'quantized'
        device: 'cuda' or 'cpu'
        quantized_path: Path to quantized model parameters
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading {model_type} model...")
    
    # Load base model
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    
    if model_type == 'quantized' and quantized_path:
        logger.info("Applying W4A4 quantization...")
        # Import quantization utilities
        from evaluation.quarot.utils import set_ignore_quantize, load_qs_parameters, quantize_linear, after_resume_qs
        from evaluation.quarot.args_utils import get_config
        
        # Set up quantization config
        args = get_config()
        args.w_bits = 4
        args.a_bits = 4
        args.lwc = True
        args.lac = True
        args.not_smooth = True  # Skip smoothing since we're loading pre-trained params
        args.not_rot = False
        
        # Move to device first
        model = model.to(device)
        
        # Apply quantization structure
        set_ignore_quantize(model, ignore_quantize=True)
        quantize_linear(model, device=device, args=args)
        
        # Load pre-trained quantization parameters
        # Check both potential paths
        potential_paths = [
            quantized_path,
            os.path.join(quantized_path, 'a44_quant_model_tracker_fixed_e20.pt_sym'),
        ]
        
        loaded = False
        for path in potential_paths:
            frame_path = os.path.join(path, 'qs_frame_parameters_total.pth')
            global_path = os.path.join(path, 'qs_global_parameters_total.pth')
            if os.path.exists(frame_path) and os.path.exists(global_path):
                logger.info(f"Loading quantization parameters from {path}")
                load_qs_parameters(args, model, path=path)
                after_resume_qs(model)
                loaded = True
                break
        
        if not loaded:
            logger.warning(f"Quantization parameters not found in any of: {potential_paths}")
    else:
        model = model.to(device)
    
    if model_type == 'fp16':
        model = model.half()
    
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: nn.Module, 
    images: torch.Tensor, 
    device: str,
    num_warmup: int = 2,
    num_runs: int = 5,
) -> Dict:
    """
    Run inference and collect metrics.
    
    Returns:
        Dictionary with predictions and timing info
    """
    images = images.to(device)
    
    # Warmup
    for _ in range(num_warmup):
        _ = model(images)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        predictions = model(images)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    # Memory usage
    if device == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_mb = 0
    
    return {
        'predictions': predictions,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'memory_mb': memory_mb,
    }


def compare_predictions(pred1: Dict, pred2: Dict, name1: str, name2: str) -> Dict:
    """
    Compare predictions from two models.
    
    Returns:
        Dictionary of comparison metrics
    """
    metrics = {}
    
    # Compare depth maps
    if 'depth' in pred1 and 'depth' in pred2:
        depth1 = pred1['depth'].float()
        depth2 = pred2['depth'].float()
        
        # Relative error
        rel_error = torch.abs(depth1 - depth2) / (torch.abs(depth1) + 1e-6)
        metrics['depth_rel_error_mean'] = rel_error.mean().item()
        metrics['depth_rel_error_max'] = rel_error.max().item()
        
        # Absolute error
        abs_error = torch.abs(depth1 - depth2)
        metrics['depth_abs_error_mean'] = abs_error.mean().item()
        
        # Correlation
        d1_flat = depth1.flatten()
        d2_flat = depth2.flatten()
        correlation = torch.corrcoef(torch.stack([d1_flat, d2_flat]))[0, 1]
        metrics['depth_correlation'] = correlation.item()
    
    # Compare pose encoding
    if 'pose_enc' in pred1 and 'pose_enc' in pred2:
        pose1 = pred1['pose_enc'].float()
        pose2 = pred2['pose_enc'].float()
        
        pose_error = torch.abs(pose1 - pose2)
        metrics['pose_error_mean'] = pose_error.mean().item()
        metrics['pose_error_max'] = pose_error.max().item()
    
    return metrics


def visualize_comparison(
    results: Dict,
    images: torch.Tensor,
    output_dir: Path,
):
    """
    Create visualization of model comparison results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Input image
    img_np = images[0, 0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Input Image (View 1)')
    axes[0, 0].axis('off')
    
    # FP32 depth
    if 'fp32' in results and 'predictions' in results['fp32']:
        depth_fp32 = results['fp32']['predictions']['depth'][0, 0, :, :, 0].cpu().numpy()
        im1 = axes[0, 1].imshow(depth_fp32, cmap='viridis')
        axes[0, 1].set_title('FP32 Depth')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Quantized depth (if available)
    if 'quantized' in results and 'predictions' in results['quantized']:
        depth_quant = results['quantized']['predictions']['depth'][0, 0, :, :, 0].cpu().numpy()
        im2 = axes[0, 2].imshow(depth_quant, cmap='viridis')
        axes[0, 2].set_title('Quantized (W4A4) Depth')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        # Difference map
        diff = np.abs(depth_fp32 - depth_quant)
        im3 = axes[1, 0].imshow(diff, cmap='hot')
        axes[1, 0].set_title('Absolute Difference')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # Performance comparison bar chart
    model_names = []
    times = []
    for name, data in results.items():
        if 'mean_time' in data:
            model_names.append(name)
            times.append(data['mean_time'] * 1000)  # Convert to ms
    
    if model_names:
        axes[1, 1].bar(model_names, times)
        axes[1, 1].set_ylabel('Inference Time (ms)')
        axes[1, 1].set_title('Performance Comparison')
    
    # Memory comparison
    memories = []
    for name, data in results.items():
        if 'memory_mb' in data:
            memories.append(data['memory_mb'])
    
    if memories and model_names:
        axes[1, 2].bar(model_names, memories)
        axes[1, 2].set_ylabel('Memory (MB)')
        axes[1, 2].set_title('Memory Usage')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_results.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved visualization to {output_dir / 'comparison_results.png'}")


def print_results(results: Dict, comparison: Dict):
    """Print formatted results summary."""
    print("\n" + "=" * 60)
    print("VGGT vs QuantVGGT Comparison Results")
    print("=" * 60)
    
    # Performance metrics
    print("\n📊 Performance Metrics:")
    print("-" * 40)
    for name, data in results.items():
        if 'mean_time' in data:
            print(f"  {name}:")
            print(f"    Inference time: {data['mean_time']*1000:.2f} ± {data['std_time']*1000:.2f} ms")
            print(f"    Memory usage: {data['memory_mb']:.1f} MB")
    
    # Note about quantization
    if 'quantized' in results:
        print("\n⚠️  Note: Quantized model uses SIMULATED W4A4 quantization.")
        print("    This is for ACCURACY testing, not performance testing.")
        print("    Real speedup requires TFLite/ExecuTorch with INT4 support.")
    
    # Accuracy comparison
    if comparison:
        print("\n📐 Accuracy Comparison (FP32 vs Quantized):")
        print("-" * 40)
        for key, value in comparison.items():
            print(f"  {key}: {value:.6f}")
    
    # Summary
    if 'fp32' in results and 'quantized' in results and comparison:
        print("\n🎯 Accuracy Summary:")
        print("-" * 40)
        if 'depth_correlation' in comparison:
            corr = comparison['depth_correlation']
            print(f"  Depth correlation: {corr:.4f} ({corr*100:.2f}%)")
        if 'depth_rel_error_mean' in comparison:
            err = comparison['depth_rel_error_mean']
            print(f"  Mean relative error: {err:.4f} ({err*100:.2f}%)")
        if 'pose_error_mean' in comparison:
            print(f"  Pose error: {comparison['pose_error_mean']:.6f}")
        
        # Quality assessment
        print("\n📋 Quality Assessment:")
        print("-" * 40)
        if comparison.get('depth_correlation', 0) > 0.99:
            print("  ✅ Excellent: Depth correlation > 99%")
        elif comparison.get('depth_correlation', 0) > 0.95:
            print("  ✓ Good: Depth correlation > 95%")
        else:
            print("  ⚠️ Warning: Depth correlation < 95%")
        
        if comparison.get('depth_rel_error_mean', 1) < 0.05:
            print("  ✅ Excellent: Mean relative error < 5%")
        elif comparison.get('depth_rel_error_mean', 1) < 0.10:
            print("  ✓ Good: Mean relative error < 10%")
        else:
            print("  ⚠️ Warning: Mean relative error > 10%")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare VGGT vs QuantVGGT')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num-views', type=int, default=2, help='Number of views')
    parser.add_argument('--image-dir', type=str, default='test_images', help='Directory for test images')
    parser.add_argument('--output-dir', type=str, default='comparison_results', help='Output directory')
    parser.add_argument('--quantized-path', type=str, default='evaluation/outputs/w4a4',
                        help='Path to quantized model parameters')
    parser.add_argument('--skip-quantized', action='store_true', help='Skip quantized model (FP32 only)')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of inference runs for timing')
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    image_dir = project_root / args.image_dir
    output_dir = project_root / args.output_dir
    quantized_path = project_root / args.quantized_path
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    logger.info(f"Using device: {args.device}")
    
    # Prepare test images
    image_paths = download_sample_images(image_dir, args.num_views)
    
    # Load and preprocess images
    logger.info("Loading and preprocessing images...")
    images = load_and_preprocess_images(image_paths)
    images = images.unsqueeze(0)  # Add batch dimension
    logger.info(f"Input shape: {images.shape}")
    
    results = {}
    
    # Test FP32 model
    try:
        logger.info("\n" + "="*40)
        logger.info("Testing FP32 model...")
        model_fp32 = load_model('fp32', args.device)
        results['fp32'] = run_inference(model_fp32, images, args.device, num_runs=args.num_runs)
        del model_fp32
        torch.cuda.empty_cache() if args.device == 'cuda' else None
        logger.info("FP32 testing complete")
    except Exception as e:
        logger.error(f"FP32 model failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Quantized model
    if not args.skip_quantized:
        try:
            logger.info("\n" + "="*40)
            logger.info("Testing Quantized (W4A4) model...")
            model_quant = load_model('quantized', args.device, str(quantized_path))
            results['quantized'] = run_inference(model_quant, images, args.device, num_runs=args.num_runs)
            del model_quant
            torch.cuda.empty_cache() if args.device == 'cuda' else None
            logger.info("Quantized testing complete")
        except Exception as e:
            logger.error(f"Quantized model failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare results
    comparison = {}
    if 'fp32' in results and 'quantized' in results:
        comparison = compare_predictions(
            results['fp32']['predictions'],
            results['quantized']['predictions'],
            'FP32', 'Quantized'
        )
    
    # Visualize
    try:
        visualize_comparison(results, images, output_dir)
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
    
    # Print results
    print_results(results, comparison)
    
    # Save results to JSON
    json_results = {
        name: {k: v for k, v in data.items() if k != 'predictions'}
        for name, data in results.items()
    }
    json_results['comparison'] = comparison
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
