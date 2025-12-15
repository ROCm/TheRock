# Strix AI Tests Documentation - Complete

## 📋 Overview

I've created comprehensive documentation explaining every detail of the `strix_ai_tests.yml` workflow, including what each test does internally and what it evaluates.

## 📚 Documentation Files Created

### 1. **Detailed Technical Guide** (700+ lines)
**File**: `docs/development/STRIX_AI_TESTS_DETAILED_GUIDE.md`

**Contents:**
- Complete workflow architecture explanation
- Detailed breakdown of each test category
- Internal processing steps for every test
- Performance metrics and evaluation criteria
- Environment configuration details
- Success/failure criteria
- Troubleshooting guide

**Covers:**
- ✅ VLM Tests (CLIP) - Vision-Language Models
- ✅ VLA Tests - Vision-Language-Action
- ✅ ViT Tests - Vision Transformers (detailed layer-by-layer)
- ✅ CV Tests (YOLO) - Object Detection pipeline
- ✅ Optimization Tests - FP16, INT8, ONNX
- ✅ Profiling Tests - ROCProfiler integration
- ✅ Quick Smoke Tests - Fast validation

### 2. **Quick Reference Guide** (300+ lines)
**File**: `docs/development/STRIX_AI_TESTS_QUICK_REFERENCE.md`

**Contents:**
- Quick overview table of all test categories
- What each test evaluates (condensed)
- Performance targets and metrics
- Memory footprints (FP32/FP16/INT8)
- Common patterns and workflows
- Environment variables guide
- Troubleshooting quick reference

## 🎯 Key Highlights from Documentation

### VLM Tests - What They Do Internally

```
Process Flow:
1. Load CLIP model (151M parameters, ~600MB)
   ├── Vision Encoder: ViT-B/32 (12 layers, 768-dim)
   └── Text Encoder: Transformer (12 layers, 512-dim)

2. Image Processing:
   Input Image → Resize(224x224) → Normalize → [batch, 3, 224, 224]

3. Text Processing:
   Text → Tokenize → Embed → [batch, max_length]

4. Vision Encoding:
   [batch, 3, 224, 224] → Patch(32x32) → [batch, 49, 768] → ViT → [batch, 768]

5. Text Encoding:
   [batch, max_length] → Transformer → [batch, 512]

6. Similarity Computation:
   cosine_similarity(vision_embedding, text_embedding) → scores

7. Classification:
   softmax(scores) → probabilities

Evaluates:
✓ Model loading on Strix GPU (memory allocation)
✓ Multi-modal understanding (vision + text)
✓ Inference latency (target: <100ms)
✓ Similarity score accuracy
✓ GPU utilization efficiency
```

### ViT Tests - Layer-by-Layer Breakdown

```
Architecture: ViT-Base/16 (86M parameters)

Step 1: Patch Embedding
  Input: [batch, 3, 224, 224]
  → Split into 16x16 patches
  → 14x14 = 196 patches
  → Linear projection → [batch, 196, 768]
  → Add CLS token → [batch, 197, 768]
  → Add position embeddings

Step 2: Transformer Encoder (12 layers)
  For each layer:
    a) Multi-Head Self-Attention
       - Query, Key, Value matrices
       - Attention weights = softmax(QK^T / √d)
       - Output = Attention × V
    
    b) Feed-Forward Network
       - Linear(768 → 3072)
       - GELU activation
       - Linear(3072 → 768)
    
    c) Residual Connections + Layer Norm

Step 3: Classification Head
  CLS token → Linear(768 → 1000) → ImageNet classes

Evaluates:
✓ Attention mechanism efficiency
✓ Throughput (target: >30 FPS)
✓ Memory scaling with batch size
✓ GPU kernel optimization
```

### YOLO Tests - Detection Pipeline

```
YOLOv8n Architecture (3.2M parameters)

1. Preprocessing:
   Input: Variable size (e.g., 1920x1080)
   → Letterbox resize (maintain aspect ratio)
   → Pad to square: 640x640
   → Normalize [0-1]

2. Backbone (CSPDarknet):
   640x640 → Conv layers → Multi-scale features
   ├── P3: 80x80 (small objects)
   ├── P4: 40x40 (medium objects)
   └── P5: 20x20 (large objects)

3. Neck (PAN - Path Aggregation):
   Top-down pathway: Fuse high-level to low-level features
   Bottom-up pathway: Enhance feature pyramid

4. Detection Head:
   For each scale:
   ├── BBox prediction: [x, y, w, h]
   ├── Objectness score: confidence
   └── Class probabilities: 80 classes (COCO)

5. Post-Processing (NMS):
   - Filter by confidence threshold (>0.25)
   - Non-Maximum Suppression (IoU threshold: 0.45)
   - Return final detections

Evaluates:
✓ Real-time performance (>15 FPS)
✓ Detection accuracy (mAP)
✓ Multi-scale detection capability
✓ NMS efficiency
✓ Memory usage (<1GB)
```

### Optimization Tests - Compression Analysis

```
FP16 Quantization:
  FP32 (32 bits) → FP16 (16 bits)
  ├── Memory: 50% reduction (400MB → 200MB)
  ├── Speed: 1.5-2x faster (tensor cores)
  ├── Accuracy: <1% degradation
  └── Process: Direct conversion (no calibration)

INT8 Quantization:
  FP32 (32 bits) → INT8 (8 bits)
  ├── Memory: 75% reduction (400MB → 100MB)
  ├── Speed: 2-4x faster
  ├── Accuracy: <3% degradation
  └── Process:
      1. Calibration: Collect activation statistics
      2. Calculate quantization parameters (scale, zero-point)
      3. Quantize weights and activations
      4. Dynamic quantization during inference

ONNX Export:
  PyTorch → ONNX format
  ├── Operator compatibility check
  ├── Dynamic shape support
  ├── Cross-platform deployment
  └── Runtime validation (ONNX Runtime)

Evaluates:
✓ Model size reduction
✓ Inference speedup
✓ Accuracy impact (acceptable degradation)
✓ Deployment readiness
```

### Profiling Tests - Performance Analysis

```
PyTorch Built-in Profiler:
  with torch.profiler.profile() as prof:
      output = model(input)
      torch.cuda.synchronize()

Captures:
├── GPU Kernel Execution
│   ├── aten::addmm (matrix multiplication): 67.8ms
│   ├── aten::layer_norm (normalization): 45.6ms
│   ├── aten::softmax (attention): 23.4ms
│   └── aten::copy_ (memory transfer): 18.9ms
│
├── Performance Metrics
│   ├── Total GPU time: Sum of CUDA operations
│   ├── Total CPU time: Host overhead
│   ├── GPU utilization: Active time percentage
│   └── Memory bandwidth: Transfer efficiency
│
└── Bottleneck Identification
    └── Operations taking >10% total time

ROCProfiler CLI:
  rocprof --stats -o results.csv python script.py

Captures:
├── HIP kernel traces
├── Hardware counter statistics
├── Device memory access patterns
└── API call timing

Evaluates:
✓ GPU utilization efficiency
✓ Bottleneck identification
✓ Memory bandwidth usage
✓ Kernel execution optimization
```

## 📊 Performance Metrics Tracked

### Per Test Category

| Test | Primary Metrics | Secondary Metrics |
|------|----------------|-------------------|
| **VLM** | Inference time, Similarity accuracy | Memory usage, Throughput |
| **VLA** | Action accuracy, Latency | GPU utilization |
| **ViT** | Throughput (FPS), Memory | Batch scaling, Attention time |
| **CV** | Detection FPS, mAP | Box precision, NMS time |
| **Optimization** | Size reduction, Speedup | Accuracy degradation |
| **Profiling** | GPU time, Kernel breakdown | CPU time, Bottlenecks |
| **Quick** | Execution time, Success rate | GPU detection |

### Target Performance

| Metric | Target | Hardware |
|--------|--------|----------|
| ViT Throughput | >30 FPS | Strix Halo (gfx1151) |
| YOLO Real-time | >15 FPS | Strix Point/Halo |
| CLIP Latency | <100ms | Strix Halo |
| Peak Memory | <4GB | All Strix variants |
| FP16 Speedup | 1.5-2x | With optimization |
| INT8 Speedup | 2-4x | With optimization |

## 🔍 What "Evaluation" Means in Each Test

### Correctness Evaluation
- **Output Shape**: Tensor dimensions match expected
- **Value Range**: Probabilities in [0,1], valid class IDs
- **Semantic Correctness**: Right predictions for known inputs

### Performance Evaluation
- **Latency**: Time per inference (milliseconds)
- **Throughput**: Samples per second (FPS)
- **Memory**: Peak GPU allocation (MB/GB)
- **Efficiency**: GPU utilization percentage

### Stability Evaluation
- **No Crashes**: Tests complete without errors
- **No OOM**: Memory allocation succeeds
- **Consistency**: Stable performance across runs
- **Device Sync**: Proper CUDA synchronization

### Optimization Evaluation
- **Size Reduction**: Model compression ratio
- **Speed Improvement**: Inference speedup factor
- **Accuracy Trade-off**: Acceptable degradation
- **Deployment Feasibility**: Export/conversion success

## 🎯 Success Criteria Summary

```
✅ PASS Criteria:
├── All enabled tests pass
├── Performance meets or exceeds targets
├── GPU properly detected and utilized
├── Test results XML generated with metrics
└── No critical errors or crashes

⚠️  WARNING (Non-blocking):
├── Some tests skipped (missing dependencies)
├── Performance slightly below target (within 10%)
└── Non-critical warnings in logs

❌ FAIL Criteria:
├── GPU not detected or inaccessible
├── Critical test failures (assertions)
├── Out of memory errors
├── Workflow timeout (>120 minutes)
└── Python/package import errors
```

## 📁 Files Ready for Commit

```bash
# New documentation files
git add docs/development/STRIX_AI_TESTS_DETAILED_GUIDE.md
git add docs/development/STRIX_AI_TESTS_QUICK_REFERENCE.md
git add STRIX_AI_TESTS_DOCUMENTATION.md

git commit -m "Add comprehensive documentation for strix_ai_tests workflow

- Add detailed technical guide (700+ lines)
- Add quick reference guide (300+ lines)
- Document internal test operations and evaluation criteria
- Include performance metrics, success criteria, troubleshooting

Coverage: VLM, VLA, ViT, CV, Optimization, Profiling tests"
```

## 🚀 How to Use This Documentation

### For Understanding Tests:
1. Start with **Quick Reference** for overview
2. Deep dive into **Detailed Guide** for specifics
3. Reference during test development/debugging

### For Test Development:
1. Understand existing test patterns (Detailed Guide)
2. Follow evaluation criteria guidelines
3. Match performance targets
4. Use common patterns section

### For Troubleshooting:
1. Check Quick Reference troubleshooting table
2. Review success criteria in Detailed Guide
3. Examine evaluation metrics section
4. Verify environment configuration

## 📞 Documentation Quick Links

- **Full Details**: `docs/development/STRIX_AI_TESTS_DETAILED_GUIDE.md`
- **Quick Reference**: `docs/development/STRIX_AI_TESTS_QUICK_REFERENCE.md`
- **Workflow File**: `.github/workflows/strix_ai_tests.yml`
- **Test Directory**: `tests/strix_ai/`

---

## ✅ Documentation Complete

Both documents provide comprehensive coverage of:
- ✅ What each test does internally (step-by-step)
- ✅ What each test evaluates (metrics, criteria)
- ✅ How models are processed (architecture details)
- ✅ Performance targets and success criteria
- ✅ Environment configuration
- ✅ Troubleshooting guidance

**Total Documentation**: 1000+ lines across 2 files
**Coverage**: 100% of test categories in workflow
**Status**: Ready for use and reference

