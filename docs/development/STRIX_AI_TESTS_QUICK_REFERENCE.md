# Strix AI Tests - Quick Reference

**Full Documentation**: [STRIX_AI_TESTS_DETAILED_GUIDE.md](./STRIX_AI_TESTS_DETAILED_GUIDE.md)

## Test Categories Overview

| Category | What It Tests | Key Models | Success Criteria | Time |
|----------|--------------|------------|------------------|------|
| **VLM** | Vision-Language understanding | CLIP (151M params) | Similarity scores correct, no OOM | ~5 min |
| **VLA** | Vision-Language-Action | Action classifiers | Action predictions valid | ~4 min |
| **ViT** | Vision Transformers | ViT-Base (86M params) | >30 FPS throughput | ~6 min |
| **CV** | Object Detection | YOLOv8n (3.2M params) | >15 FPS real-time | ~5 min |
| **Optimization** | Model compression | FP16, INT8, ONNX | 50-75% size reduction | ~8 min |
| **Profiling** | Performance analysis | All models + PyTorch | Metrics captured | ~10 min |
| **Quick** | Smoke tests | Simple ops | GPU functional | <2 min |

## What Each Test Evaluates

### VLM Tests (Vision-Language Models)
**File**: `tests/strix_ai/vlm/test_clip.py`

```
Internal Process:
1. Load CLIP model (151M params, ~600MB)
2. Process image: resize → normalize → [batch, 3, 224, 224]
3. Tokenize text: encode → [batch, max_length]
4. Vision encoder: ViT-B/32 → 768-dim embedding
5. Text encoder: Transformer → 512-dim embedding
6. Compute similarity: cosine(vision_emb, text_emb)
7. Get probabilities: softmax(similarities)

Evaluates:
✓ Model loading on Strix GPU
✓ Image-text matching accuracy
✓ Inference latency (<100ms target)
✓ Memory efficiency (<1GB)
✓ Cross-modal understanding
```

### VLA Tests (Vision-Language-Action)
**File**: `tests/strix_ai/vla/test_action_prediction.py`

```
Internal Process:
1. Load action recognition model
2. Extract visual features from frames
3. Classify action from feature vectors
4. Output action probabilities

Evaluates:
✓ Action classification accuracy
✓ Visual feature extraction
✓ Temporal consistency (video)
✓ Multi-modal reasoning
```

### ViT Tests (Vision Transformers)
**File**: `tests/strix_ai/vit/test_vit_base.py`

```
Internal Process:
1. Load ViT-Base/16 (86M params)
2. Split image into 16x16 patches → 196 patches
3. Patch embedding → [batch, 197, 768]
4. 12 Transformer layers:
   - Multi-head self-attention
   - Feed-forward network
   - Layer normalization
5. Extract CLS token → classify

Evaluates:
✓ Patch embedding correctness
✓ Attention mechanism efficiency
✓ Throughput (target: >30 FPS)
✓ Batch processing (1, 2, 4, 8, 16)
✓ Memory scaling
```

### CV Tests (Computer Vision)
**File**: `tests/strix_ai/cv/test_yolo.py`

```
Internal Process:
1. Load YOLOv8n (3.2M params)
2. Preprocess: letterbox → 640x640
3. Backbone: Extract multi-scale features
   - P3: 80x80 (small objects)
   - P4: 40x40 (medium objects)
   - P5: 20x20 (large objects)
4. Neck: Feature fusion (PAN)
5. Head: Predict [bbox, objectness, class]
6. NMS: Filter overlapping boxes

Evaluates:
✓ Detection accuracy (mAP)
✓ Real-time performance (>15 FPS)
✓ Bounding box precision
✓ Class classification accuracy
✓ NMS efficiency
```

### Optimization Tests
**File**: `tests/strix_ai/optimization/test_quantization.py`

```
FP16 Test:
- Convert FP32 → FP16 (32bit → 16bit)
- Memory: 50% reduction
- Speed: 1.5-2x faster (tensor cores)
- Accuracy: <1% degradation
Evaluates: Memory savings, speedup, accuracy impact

INT8 Test:
- Quantize FP32 → INT8 (32bit → 8bit)
- Memory: 75% reduction
- Speed: 2-4x faster
- Accuracy: <3% degradation
Evaluates: Model size, inference speed, accuracy loss

ONNX Test:
- Export PyTorch → ONNX format
- Validate operator compatibility
- Test cross-platform deployment
Evaluates: Export success, runtime compatibility
```

### Profiling Tests
**File**: `tests/strix_ai/profiling/test_*.py`

```
PyTorch Profiler:
with torch.profiler.profile() as prof:
    output = model(input)
    
Captures:
- GPU kernel execution times
- Memory transfers (H2D, D2H)
- Operation breakdown (matmul, conv, etc.)
- GPU utilization percentage

ROCProfiler:
rocprof --stats python script.py

Captures:
- HIP kernel traces
- Hardware counter statistics
- API call timing
- Device memory access patterns

Evaluates:
✓ Bottleneck identification
✓ GPU utilization
✓ Memory bandwidth
✓ Kernel efficiency
```

### Quick Smoke Tests
**File**: `tests/strix_ai/test_simple.py`

```
1. GPU Detection:
   - Check torch.cuda.is_available()
   - Validate gfx1150/gfx1151 device
   
2. Basic Ops:
   - Tensor allocation on GPU
   - Matrix multiplication
   - Element-wise operations
   
3. Simple Model:
   - 2-layer neural network
   - Forward pass on GPU
   - Output shape validation

Target: <2 minutes total
```

## Performance Targets

| Metric | Target | Model |
|--------|--------|-------|
| **ViT Throughput** | >30 FPS | ViT-Base |
| **YOLO FPS** | >15 FPS | YOLOv8n |
| **CLIP Latency** | <100ms | CLIP-ViT-B/32 |
| **Memory Usage** | <4GB peak | All models |
| **FP16 Speedup** | 1.5-2x | Any model |
| **INT8 Speedup** | 2-4x | Any model |

## Memory Footprints

| Model | FP32 | FP16 | INT8 |
|-------|------|------|------|
| **CLIP** | ~600MB | ~300MB | ~150MB |
| **ViT-Base** | ~350MB | ~175MB | ~90MB |
| **YOLOv8n** | ~12MB | ~6MB | ~3MB |

## Test Workflow Steps

```
1. ✅ Checkout Repository
2. ✅ Display System Info
   └─ Python version, GPU info, ROCm version
3. ✅ Check ROCm/GPU (Linux)
   └─ rocminfo, hipconfig, device access
4. ✅ Install AI/ML Dependencies
   └─ transformers, ultralytics, opencv, pytest
5. ✅ Verify Dependencies
   └─ torch.cuda.is_available() check
6. ✅ Run Test Category
   ├─ VLM Tests
   ├─ VLA Tests
   ├─ ViT Tests
   ├─ CV Tests
   ├─ Optimization Tests
   └─ Profiling Tests
7. ✅ Display Test Results XML
   └─ JUnit XML with metrics
8. ✅ Archive Test Results
   └─ Upload artifacts (30 days)
9. ✅ Test Summary
```

## Common Evaluation Patterns

### All Tests Follow This Pattern:

```python
1. Setup:
   - Load model to GPU
   - Prepare test data
   - Warmup iterations (3-5 runs)

2. Execute:
   - Run inference/operation
   - Measure performance metrics
   - Capture outputs

3. Validate:
   - Check output correctness
   - Verify performance targets
   - Validate memory usage

4. Cleanup:
   - Clear GPU cache
   - Synchronize device
   - Free resources
```

## Environment Variables

| Variable | Purpose | Values |
|----------|---------|--------|
| `AMDGPU_FAMILIES` | Target GPU | gfx1150, gfx1151 |
| `TEST_TYPE` | Test depth | smoke, quick, full |
| `TEST_CATEGORY` | Which tests | all, vlm, vit, cv, etc. |
| `PLATFORM` | OS platform | linux, windows |

## Success/Failure Indicators

### ✅ Success
- All tests pass
- Performance meets targets
- No GPU errors
- XML results generated

### ⚠️ Warning
- Tests skipped (missing deps)
- Performance slightly low
- Non-critical warnings

### ❌ Failure
- GPU not detected
- Test crashes
- Out of memory
- Timeout (>120 min)

## Running Tests Locally

```bash
# All tests
pytest tests/strix_ai/ -v -s

# Specific category
pytest tests/strix_ai/vlm/ -v -s
pytest tests/strix_ai/vit/ -v -s
pytest tests/strix_ai/profiling/ -v -s

# Quick smoke tests only
pytest tests/strix_ai/ -v -s -m quick

# With GPU variant
AMDGPU_FAMILIES=gfx1151 pytest tests/strix_ai/ -v -s
```

## Key Metrics in Test Results

### Captured Metrics:
- **Inference Time**: Per-sample latency (ms)
- **Throughput**: Samples/second (FPS)
- **GPU Time**: CUDA kernel execution (ms)
- **CPU Time**: Host overhead (ms)
- **Memory Usage**: Peak GPU allocation (MB)
- **Accuracy**: Model output correctness (%)
- **Batch Performance**: Scaling with batch size

### Example Output:
```
=== ViT Inference Test ===
✓ Model: google/vit-base-patch16-224
✓ Input: [1, 3, 224, 224]
✓ Output: [1, 1000]
✓ Inference time: 32.45 ms
✓ Throughput: 30.8 FPS
✓ GPU memory: 342 MB
✓ Predicted class: 281 (tabby cat)
```

## Troubleshooting Quick Guide

| Issue | Solution |
|-------|----------|
| GPU not detected | Check rocminfo, device permissions |
| Out of memory | Use FP16, reduce batch size |
| Slow inference | Check GPU utilization, enable optimizations |
| Model download fails | Check HF_HOME, internet connectivity |
| Test skipped | Install missing dependencies |
| Profiler error | Check rocprof installation |

---

**For complete details, see**: [STRIX_AI_TESTS_DETAILED_GUIDE.md](./STRIX_AI_TESTS_DETAILED_GUIDE.md)

