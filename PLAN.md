# VGGT → Mobile (Android + ARCore)


## 1. Download a dataset that we can use to do a meaningfull quatization
We will use BlendedMVS Dataset. 

```
cd /home/iago/workspace/QuantVGGT/datasets
rm -f BlendedMVS.zip && pip install gdown -q && gdown 1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb -O BlendedMVS_lowres.zip
ls -lh *.zip && echo "--- Unzipping ---" && unzip -q BlendedMVS_lowres.zip && ls -la
cd dataset_low_res && ls | head -20 && echo "--- Total scenes: $(ls -d */ | wc -l) ---" && echo "--- Checking first scene structure ---" && ls -la "$(ls -d */ | head -1)"
cd 57f8d9bbe73f6760f10e916a && echo "=== Images ===" && ls blended_images/ | head -10 && echo "... $(ls blended_images/*.jpg 2>/dev/null | wc -l) total images" && echo "=== Cameras ===" && ls cams/ | head -5 && echo "=== Depths ===" && ls rendered_depth_maps/ | head -5 && echo "=== Sample image info ===" && file blended_images/00000000.jpg && identify blended_images/00000000.jpg 2>/dev/null || echo "(ImageMagick not installed)"
```

Everything should be ready after this.

## 2. Quatize the model

We will use the following scenes for Quatization:
- 5a48c4e9c7dab83a7d7b5cc7
- 5a3ca9cb270f0e3f14d0eddb
- 5a3cb4e4270f0e3f14d12f43
- 5a3f4aba5889373fbbc5d3b5
- 5a4a38dad38c8a075495b5d2
- 5a7d3db14989e929563eb153
- 5a8aa0fab18050187cbe060e
- 5a48d4b2c7dab83a7d7b9851

### 2.1 VGGT / QuantVGGT Model Format

* **VGGT is not released as a mobile-ready model**
* Official checkpoints:

  * `facebook/VGGT-1B`
  * `facebook/VGGT_tracker_fixed`
* **Checkpoint size:** ~5 GB (`model.pt` / `model.safetensors`)
* **Framework:** PyTorch only
* **No official ONNX / TFLite / ExecuTorch exports**

### 2.2 QuantVGGT Is *Not* a Drop-In Mobile Model

QuantVGGT:

* Is **post-training quantization (W4A4)** applied *inside PyTorch*
* Uses:

  * Hadamard rotations
  * Channel smoothing
  * Custom quantized linear layers
* Ships as:

  * `.pt_sym` files (~3.8 GB)
  * PyTorch runtime logic that **does not translate automatically** to TFLite

⚠️ **Conclusion:**
QuantVGGT **cannot be directly exported** to TFLite or run on Android without major re-engineering.

### 2.3 Verified Environment Setup

```bash
# Create conda environment
conda create -n quantvggt python=3.11 -y
conda activate quantvggt

# Install dependencies
pip install -r requirements.txt

# Run comparison test
python evaluation/compare_models.py --num-runs 3
```

**Test outputs saved to:**
- `comparison_results/comparison_results.png` - Visual comparison
- `comparison_results/results.json` - Numeric metrics

### 2.4 INT8 Quantization Pipeline

#### Calibration Scenes (499 total images)

| Scene ID | Images |
|----------|--------|
| 5a48c4e9c7dab83a7d7b5cc7 | 25 |
| 5a3ca9cb270f0e3f14d0eddb | 64 |
| 5a3cb4e4270f0e3f14d12f43 | 68 |
| 5a3f4aba5889373fbbc5d3b5 | 29 |
| 5a4a38dad38c8a075495b5d2 | 174 |
| 5a7d3db14989e929563eb153 | 29 |
| 5a8aa0fab18050187cbe060e | 110 |

#### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Create calibration data loader                     │
│  - Load images from 7 scenes                                │
│  - Resize to 518×518, normalize                             │
│  - Create pairs of views (VGGT needs 2+ views)              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Wrap VGGT for export                               │
│  - Use VGGTDepthOnlyWrapper (already created)               │
│  - Fixed input shape: (1, 2, 3, 518, 518)                   │
│  - Output: depth map only                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Export to TFLite FP32 first                        │
│  - torch.export.export() → ExportedProgram                  │
│  - ai_edge_torch.convert() → TFLite                         │
│  - Validate FP32 works before quantizing                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: INT8 quantization with calibration                 │
│  - Create representative_dataset() generator                │
│  - TFLite converter runs ~100-200 samples                   │
│  - Computes min/max ranges for each tensor                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Validate quantized model                           │
│  - Compare INT8 vs FP32 outputs                             │
│  - Check depth correlation, relative error                  │
│  - Visual inspection of depth maps                          │
└─────────────────────────────────────────────────────────────┘
```

#### Key Code Components

**1. Calibration Dataset Loader** (`calibration/blendedmvs_loader.py`)
```python
def load_blendedmvs_pairs(scenes, dataset_root, num_samples=200):
    """Load image pairs from BlendedMVS for calibration"""
    # For each scene, pick consecutive frames as view pairs
    # Return generator yielding (1, 2, 3, 518, 518) tensors
```

**2. Representative Dataset Generator** (for TFLite)
```python
def representative_dataset():
    for images in calibration_loader:
        yield [images.numpy()]  # TFLite expects list of numpy arrays
```

**3. Quantized Export**
```python
converter = tf.lite.TFLiteConverter.from_saved_model(...)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32  # Keep depth as float
```

#### Configuration Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Calibration samples | 100-200 | More = slower, marginally better |
| View pairing | Consecutive frames | Natural multi-view setup |
| Input quantization | INT8 | Images naturally 0-255 |
| Output quantization | FP32 | Depth needs precision |
| Fallback ops | Allow FP32 | Handle unsupported ops gracefully |

#### Prerequisites (Must Complete First)

1. ✅ Calibration dataset downloaded (BlendedMVS)
2. ⬜ FP32 TFLite export working (Step 3)
3. ⬜ AI Edge Torch installed
4. ⬜ Memory management for large model

---

## 3. Hard Constraints Identified

### 3.1 Tooling Constraints

* **TFLite**

  * Fully supports FP32 / FP16 / INT8
  * **INT4 (W4) is NOT supported** in standard pipelines
* **LiteRT Torch**

  * Promising PyTorch → TFLite path
  * Still requires:

    * Fixed input shapes
    * Tensor-only inputs/outputs
    * No dynamic control flow
* **ExecuTorch**

  * Better long-term fit for Qualcomm
  * Still experimental for large ViTs

### 3.2 Practical Constraints

* VGGT checkpoints are **too large to casually download**
* Conversion requires:

  * Careful wrapping
  * Output simplification
  * Shape fixing
* End-to-end quantization must be **postponed**

---

## 4. Core Strategy (Risk-Minimized)

### 🔑 Guiding Principle

> **Get *something* running on Android first. Optimize later.**

---

## 5. Phased Execution Plan

---

## Phase 0 — Define the MVP Output (Very Important)

**Do NOT export the full VGGT output dict.**

### MVP Output

* Single **depth map** tensor
* Shape: `(1, H, W)` or `(1, 1, H, W)`
* FP32 or FP16

Why:

* Depth is sufficient for:

  * AR visualization
  * Point cloud generation
  * Mesh reconstruction
* Simplifies export massively

---

## Phase 0.5 — Operator Audit

Before wrapping, create an **op compatibility report**:
1. Run `torch.export.export()` on vanilla VGGT
2. Collect all unsupported ops
3. Prioritize: `scaled_dot_product_attention`, RoPE, einops

---

## Phase 1 — PyTorch Wrapper (Desktop)

### Objective

Make VGGT **exportable**.

### Tasks

1. Create a **wrapper module**:

   * Inputs:

     * Fixed number of views `V` (start with `V=2`)
     * Fixed resolution (e.g. `224×224`)
     * Tensor only
   * Output:

     * Depth map tensor only
2. Remove / bypass:

   * Camera heads
   * Tracking heads
   * Dict outputs

### Result

A clean PyTorch module with:

```python
forward(images: Tensor) -> depth: Tensor
```

---

## Phase 2 — Desktop Inference Sanity Check

Before conversion:

1. Run original VGGT
2. Run wrapped VGGT
3. Compare:

   * Depth statistics (min / max / mean)
   * Visual similarity
   * Relative error (not exact match required)

✅ If similar → proceed
❌ If broken → fix wrapper first

---

## Phase 3 — Export to TFLite (FP32 / FP16)

### Preferred Path

**LiteRT Torch**

Steps:

1. Use `torch.export()` compatible wrapper
2. Convert:

   ```python
   edge_model = litert_torch.convert(model, sample_inputs)
   edge_model.export("vggt_depth.tflite")
   ```
3. Start with:

   * FP32
   * CPU inference

---

## Phase 3 Clarification

Recommended export path:
1. `torch.export.export()` → ExportedProgram  
2. **AI Edge Torch** (`ai_edge_torch.convert()`) → TFLite  
   - Better than ONNX intermediate for PyTorch models

---

## Phase 4 — Desktop TFLite Validation

### Objective

Ensure TFLite ≈ PyTorch.

Tasks:

1. Run TFLite inference on same input
2. Compare:

   * Output shape
   * Value distribution
   * Visual depth maps

Expected:

* Small numerical differences
* No catastrophic artifacts

---

## Phase 5 — Android MVP Integration

### Objective

**First mobile success**

Steps:

1. Load `vggt_depth.tflite` in Android
2. Run on:

   * CPU first
3. Feed:

   * Camera frames from ARCore
4. Post-process:

   * Depth → point cloud using ARCore intrinsics
   * No learning here yet

🎯 At this point:

* You have a working **VGGT-based mobile depth pipeline**

---

## Phase 6 — Performance Optimization (Only After MVP)

### Step 6.1 — INT8 Post-Training Quantization

* Use TFLite PTQ
* Representative dataset:

  * Capture frames from phone camera
* Result:

  * Smaller model
  * Faster inference
  * Still portable

### Step 6.2 — Delegate Acceleration

Try in order:

1. NNAPI
2. GPU delegate
3. Qualcomm QNN (if available)

---

## Phase 7 — Camera-Aware Model (Optional, Later)

Two options:

### Option A (Recommended)

**Keep camera math outside the network**

* Network predicts depth in camera frame
* ARCore handles:

  * Unprojection
  * Fusion
  * World alignment

### Option B (Research)

* Feed intrinsics / poses as tokens
* Requires:

  * Model modification
  * Fine-tuning
  * More risk

---

## Phase 8 — True QuantVGGT-Level Optimization (Long-Term)

Only if needed:

* ExecuTorch + Qualcomm backend
* Or custom QNN pipeline
* Possibly re-implement QuantVGGT PTQ logic for mobile

⚠️ **Not MVP-compatible**

---

## 6. What Should Be Delegated Next

### ✅ Completed Implementation

The following modules have been implemented in `mobile_export/`:

| Module | File | Purpose |
|--------|------|---------|
| **VGGTDepthOnlyWrapper** | `wrapper.py` | Fixed-shape depth-only wrapper |
| **ExportableAttention** | `attention_export.py` | Manual SDPA replacement |
| **StaticRoPE2D** | `rope_export.py` | Static position embeddings |
| **OperatorAuditor** | `op_audit.py` | Export compatibility checker |
| **einops-free ops** | `einops_free.py` | Native PyTorch replacements |
| **Export script** | `export_tflite.py` | Full export pipeline |

### Usage

```bash
# Audit model for export issues
python -c "
from mobile_export import audit_model, get_sample_input
from vggt.models.vggt import VGGT
model = VGGT.from_pretrained('facebook/VGGT-1B')
audit_model(model, get_sample_input())
"

# Export to TFLite
python -m mobile_export.export_tflite --output vggt_depth.tflite
```

---

## 7. Critical Implementation Notes

### 7.1 Attention Module Replacement

The original `F.scaled_dot_product_attention` in `vggt/layers/attention.py` is a fused CUDA kernel that cannot be exported. The `ExportableAttention` class provides an equivalent implementation:

```python
# Original (not exportable)
x = F.scaled_dot_product_attention(q, k, v)

# Replacement (exportable)
q = q * self.scale
attn = torch.matmul(q, k.transpose(-2, -1))
attn = torch.softmax(attn, dim=-1)
x = torch.matmul(attn, v)
```

### 7.2 RoPE Static Computation

The `PositionGetter` class uses dynamic caching which doesn't export cleanly. `StaticRoPE2D` pre-computes positions for the fixed input size (518×518 → 37×37 patches).

### 7.3 Memory Considerations

| Component | Estimated Size (FP16) |
|-----------|----------------------|
| Aggregator (DINOv2-L) | ~1.2 GB |
| Depth Head | ~50 MB |
| **Total MVP** | ~1.3 GB |

⚠️ **Warning**: This exceeds typical Android heap limits. Consider:
- Weight streaming
- Model splitting (aggregator on GPU, heads on CPU)
- Target devices: Snapdragon 8 Gen 2+ with 12GB+ RAM

### 7.4 Calibration for INT8

For Phase 6 INT8 quantization, create a calibration dataset:

```python
def representative_dataset():
    for _ in range(100):
        yield [np.random.randn(1, 2, 3, 518, 518).astype(np.float32)]
```

---

## 8. Summary (Executive)

* VGGT **can** run on mobile, but not directly
* QuantVGGT is **not** mobile-ready
* Fastest path:

  * Depth-only
  * FP32 → INT8
  * ARCore handles geometry
* **MVP first, research later**

---

## 9. Alternative: ExecuTorch Path

If TFLite proves too limiting, consider ExecuTorch:

| Aspect | TFLite | ExecuTorch |
|--------|--------|------------|
| PyTorch compatibility | Medium | High |
| INT4 support | ❌ | ✓ (Qualcomm QNN) |
| Maturity | High | Medium |
| Tooling | Better | Improving |

ExecuTorch may eventually support the real QuantVGGT W4A4 scheme.
