"""
BlendedMVS Calibration Data Loader for VGGT INT8 Quantization.

Loads image pairs from BlendedMVS dataset for use as calibration data
in TFLite post-training quantization.
"""

import os
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# VGGT expected input size
VGGT_INPUT_SIZE = 518

# Default calibration scenes from BlendedMVS
DEFAULT_CALIBRATION_SCENES = [
    "5a48c4e9c7dab83a7d7b5cc7",
    "5a3ca9cb270f0e3f14d0eddb",
    "5a3cb4e4270f0e3f14d12f43",
    "5a3f4aba5889373fbbc5d3b5",
    "5a4a38dad38c8a075495b5d2",
    "5a7d3db14989e929563eb153",
    "5a8aa0fab18050187cbe060e",
]

# ImageNet normalization (used by DINOv2/VGGT)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_scene_images(scene_path: Path) -> List[Path]:
    """Get sorted list of images from a BlendedMVS scene."""
    images_dir = scene_path / "blended_images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    
    return images


def load_and_preprocess_image(
    image_path: Path,
    target_size: int = VGGT_INPUT_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load and preprocess a single image for VGGT.
    
    Args:
        image_path: Path to the image file
        target_size: Target size (square) for resizing
        normalize: Whether to apply ImageNet normalization
    
    Returns:
        Preprocessed image as numpy array with shape (3, H, W)
    """
    img = Image.open(image_path).convert("RGB")
    
    # Resize to target size (square crop from center)
    w, h = img.size
    min_dim = min(w, h)
    
    # Center crop to square
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    
    # Resize to target size
    img = img.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to numpy (H, W, C) -> normalize -> (C, H, W)
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    if normalize:
        img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    
    # HWC -> CHW
    img_np = img_np.transpose(2, 0, 1)
    
    return img_np


class BlendedMVSCalibrationDataset(Dataset):
    """
    PyTorch Dataset for loading BlendedMVS image pairs for VGGT calibration.
    
    Creates pairs of consecutive images from each scene, which provides
    natural multi-view pairs for VGGT input.
    """
    
    def __init__(
        self,
        dataset_root: str,
        scenes: Optional[List[str]] = None,
        num_views: int = 2,
        target_size: int = VGGT_INPUT_SIZE,
        max_pairs_per_scene: Optional[int] = None,
        stride: int = 1,
        normalize: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            dataset_root: Path to BlendedMVS dataset_low_res directory
            scenes: List of scene IDs to use (default: DEFAULT_CALIBRATION_SCENES)
            num_views: Number of views per sample (default: 2)
            target_size: Target image size (default: 518 for VGGT)
            max_pairs_per_scene: Maximum pairs to sample per scene (None = all)
            stride: Frame stride for creating pairs (1 = consecutive)
            normalize: Whether to apply ImageNet normalization
            seed: Random seed for reproducibility
        """
        self.dataset_root = Path(dataset_root)
        self.scenes = scenes or DEFAULT_CALIBRATION_SCENES
        self.num_views = num_views
        self.target_size = target_size
        self.normalize = normalize
        self.stride = stride
        
        random.seed(seed)
        
        # Build list of all valid image pairs
        self.pairs: List[Tuple[List[Path], str]] = []
        
        for scene_id in self.scenes:
            scene_path = self.dataset_root / scene_id
            if not scene_path.exists():
                print(f"Warning: Scene {scene_id} not found, skipping")
                continue
            
            images = get_scene_images(scene_path)
            
            # Create pairs with the specified stride
            scene_pairs = []
            for i in range(len(images) - (num_views - 1) * stride):
                pair = [images[i + j * stride] for j in range(num_views)]
                scene_pairs.append((pair, scene_id))
            
            # Optionally limit pairs per scene
            if max_pairs_per_scene and len(scene_pairs) > max_pairs_per_scene:
                scene_pairs = random.sample(scene_pairs, max_pairs_per_scene)
            
            self.pairs.extend(scene_pairs)
        
        if not self.pairs:
            raise ValueError(f"No valid pairs found in {dataset_root}")
        
        print(f"Loaded {len(self.pairs)} image pairs from {len(self.scenes)} scenes")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (num_views, 3, H, W)
        """
        image_paths, scene_id = self.pairs[idx]
        
        images = []
        for path in image_paths:
            img = load_and_preprocess_image(
                path,
                target_size=self.target_size,
                normalize=self.normalize,
            )
            images.append(img)
        
        # Stack to (num_views, C, H, W)
        images = np.stack(images, axis=0)
        
        return torch.from_numpy(images)


def load_calibration_pairs(
    dataset_root: str,
    scenes: Optional[List[str]] = None,
    num_samples: int = 200,
    num_views: int = 2,
    batch_size: int = 1,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for calibration image pairs.
    
    Args:
        dataset_root: Path to BlendedMVS dataset_low_res directory
        scenes: List of scene IDs to use
        num_samples: Target number of samples to use
        num_views: Number of views per sample
        batch_size: Batch size for the DataLoader
        **kwargs: Additional arguments passed to BlendedMVSCalibrationDataset
    
    Returns:
        DataLoader yielding batches of shape (batch_size, num_views, 3, H, W)
    """
    dataset = BlendedMVSCalibrationDataset(
        dataset_root=dataset_root,
        scenes=scenes,
        num_views=num_views,
        **kwargs,
    )
    
    # If we have more samples than needed, create a subset
    if len(dataset) > num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Keep simple for calibration
    )


def representative_dataset_generator(
    dataset_root: str,
    scenes: Optional[List[str]] = None,
    num_samples: int = 100,
    num_views: int = 2,
) -> Iterator[List[np.ndarray]]:
    """
    Generator function for TFLite representative dataset.
    
    Yields samples in the format expected by TFLite converter:
    a list containing a single numpy array of shape (1, num_views, 3, H, W).
    
    Args:
        dataset_root: Path to BlendedMVS dataset_low_res directory
        scenes: List of scene IDs to use
        num_samples: Number of samples to generate
        num_views: Number of views per sample
    
    Yields:
        List containing single numpy array of shape (1, num_views, 3, 518, 518)
    """
    loader = load_calibration_pairs(
        dataset_root=dataset_root,
        scenes=scenes,
        num_samples=num_samples,
        num_views=num_views,
        batch_size=1,
    )
    
    for batch in loader:
        # batch shape: (1, num_views, 3, H, W)
        yield [batch.numpy().astype(np.float32)]


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test BlendedMVS calibration loader")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets/dataset_low_res",
        help="Path to BlendedMVS dataset_low_res directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to load",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=2,
        help="Number of views per sample",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize sample pairs",
    )
    
    args = parser.parse_args()
    
    print(f"Loading calibration data from: {args.dataset_root}")
    print(f"Scenes: {DEFAULT_CALIBRATION_SCENES}")
    print()
    
    # Test the DataLoader
    loader = load_calibration_pairs(
        dataset_root=args.dataset_root,
        num_samples=args.num_samples,
        num_views=args.num_views,
    )
    
    print(f"Created loader with {len(loader.dataset)} samples")
    
    # Load a few samples and check shapes
    for i, batch in enumerate(loader):
        print(f"Batch {i}: shape={batch.shape}, dtype={batch.dtype}, "
              f"min={batch.min():.3f}, max={batch.max():.3f}")
        if i >= 2:
            break
    
    # Test the representative dataset generator
    print("\nTesting representative_dataset_generator:")
    gen = representative_dataset_generator(
        dataset_root=args.dataset_root,
        num_samples=5,
        num_views=args.num_views,
    )
    
    for i, sample in enumerate(gen):
        arr = sample[0]
        print(f"Sample {i}: shape={arr.shape}, dtype={arr.dtype}")
    
    # Visualize if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            
            # Reload for visualization (without normalization for better viewing)
            dataset = BlendedMVSCalibrationDataset(
                dataset_root=args.dataset_root,
                num_views=args.num_views,
                normalize=False,  # Don't normalize for visualization
                max_pairs_per_scene=5,
            )
            
            fig, axes = plt.subplots(2, args.num_views, figsize=(4 * args.num_views, 8))
            
            for row in range(2):
                sample = dataset[row]  # (num_views, 3, H, W)
                for col in range(args.num_views):
                    img = sample[col].numpy().transpose(1, 2, 0)  # CHW -> HWC
                    img = np.clip(img, 0, 1)
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(f"Sample {row}, View {col}")
                    axes[row, col].axis("off")
            
            plt.tight_layout()
            plt.savefig("calibration_samples.png", dpi=150)
            print("\nSaved visualization to calibration_samples.png")
            plt.show()
        except ImportError:
            print("matplotlib not installed, skipping visualization")
