# VGGT → Mobile (Android + ARCore)

## Incremental Execution Plan & Findings

---

## 1. Goal (Restated Precisely)

Deploy a **VGGT-based 3D reconstruction model** on an **Android device (Qualcomm SoC)** that:

* Runs **on-device**
* Integrates with **ARCore**
* Uses **known camera intrinsics & poses** (provided by ARCore)
* Focuses the network on **geometry reconstruction**
* Is **incrementally developed**, with a fast MVP

---

## 2. Key Findings (Important Reality Checks)

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

### Immediate Tasks for Another AI / Engineer

1. Implement **VGGTDepthOnlyWrapper**
2. Validate PyTorch output consistency
3. Make wrapper compatible with `torch.export`
4. Attempt LiteRT export
5. Document any unsupported ops

---

## 7. Summary (Executive)

* VGGT **can** run on mobile, but not directly
* QuantVGGT is **not** mobile-ready
* Fastest path:

  * Depth-only
  * FP32 → INT8
  * ARCore handles geometry
* **MVP first, research later**

