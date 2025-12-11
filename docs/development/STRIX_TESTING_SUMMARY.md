# Strix Testing Documentation Summary

## 📚 Complete Documentation Suite

This directory contains comprehensive testing documentation for AMD Strix platforms (gfx1150, gfx1151).

---

## 📖 Documentation Files

### **1. STRIX_TESTING_GUIDE.md** (29KB)
**Purpose:** Complete reference guide for all Strix testing

**Contents:**
- ✅ Current test coverage (20+ libraries)
- ✅ Platform configuration
- ✅ How to add tests (4 methods)
- ✅ Test examples and templates
- ✅ Known issues and workarounds
- ✅ Best practices
- ⭐ **NEW: AI/ML test category**

**Use When:** You need comprehensive information about Strix testing infrastructure

---

### **2. STRIX_TESTING_QUICK_REFERENCE.md** (5KB)
**Purpose:** Quick lookup for common tasks

**Contents:**
- ✅ Test coverage summary table
- ✅ 3-step guide to add tests
- ✅ Command reference
- ✅ Known issues quick ref

**Use When:** You need a quick reminder of commands or status

---

### **3. STRIX_AI_ML_TEST_PLAN.md** (32KB) ⭐ NEW
**Purpose:** Comprehensive test plan for AI/ML workloads on Strix

**Contents:**
- ❌ **Gap Analysis**: What's missing vs MI/Navi GPUs
- 🧠 **VLM Tests**: LLaVA, CLIP, Qwen-VL (Vision Language Models)
- 👁️ **ViT Tests**: ViT-Base, DINOv2, Swin Transformer
- 🔍 **CV Tests**: YOLO, DETR, SegFormer (Object Detection, Segmentation)
- ⚡ **Edge AI**: Quantization, ONNX Runtime, optimization
- 🎥 **Video**: Encoding, decoding, real-time processing
- 🪟 **Windows AI**: DirectML, WinML integration
- 🔋 **iGPU Optimization**: Shared memory, power efficiency

**Priority Matrix:**
- 🔴 **P0 Critical** (Immediate): VLM, ViT, Object Detection, Edge optimization
- 🟡 **P1 High** (Next Quarter): Segmentation, Video, Windows AI
- 🟢 **P2 Medium** (Future): Consumer AI apps, additional CV tasks

**Use When:** Planning new AI/ML test development for Strix

---

### **4. STRIX_AI_QUICK_START.md** (17KB) ⭐ NEW
**Purpose:** Quick implementation guide with copy-paste templates

**Contents:**
- ⚡ **5-minute setup** instructions
- 🧠 **Test templates** for:
  - VLM (CLIP example)
  - YOLO object detection
  - ViT classification
  - Semantic segmentation
  - Quantization benchmarking
- 📦 **Requirements** file
- 🔧 **Shared fixtures** and utilities
- 📊 **Benchmarking** framework
- 🚨 **Common issues** and fixes
- ✅ **Checklist** for new tests

**Use When:** You want to quickly implement a new AI test

---

## 🎯 Key Findings: What's Missing for Strix

### **Current Status: Good Low-Level Coverage**
✅ **20+ library tests** (BLAS, FFT, RAND, etc.)  
✅ **Basic PyTorch tests** (matrix ops, conv2d)  
✅ **Sanity tests** (HIP compilation, GPU detection)  
✅ **Platform integration** (pre-submit, nightly)

### **Gap: No Edge AI / Consumer Workloads** ❌

Unlike MI (data center training) and Navi (gaming), Strix targets:

| Use Case | Examples | Status |
|----------|----------|--------|
| **Vision Language Models** | LLaVA, CLIP for AI PC | ❌ Not tested |
| **Vision Transformers** | ViT, DINOv2 for CV | ❌ Not tested |
| **Object Detection** | YOLO for webcams | ❌ Not tested |
| **Semantic Segmentation** | SegFormer for backgrounds | ❌ Not tested |
| **Edge Inference** | INT8 quantization | ❌ Not tested |
| **Video Processing** | Real-time encoding | ❌ Not tested |
| **Windows Copilot+** | DirectML integration | ❌ Not tested |
| **Consumer AI** | Background blur, upscaling | ❌ Not tested |

### **Why This Matters**

**Strix is for:**
- 🪟 Windows Copilot+ AI PCs
- 💻 Edge AI inference (not training)
- 🎥 Video conferencing (Teams, Zoom backgrounds)
- 📸 Consumer photo/video enhancement
- 🔋 Power-efficient AI (iGPU shared memory)

**Current tests don't validate these use cases!**

---

## 🚀 Quick Start: Implement Your First AI Test

### **Option 1: Run Pre-made Template (5 minutes)**

```bash
# 1. Install dependencies
pip install transformers accelerate pillow torch

# 2. Create test file
cat > tests/strix_ai/test_vit_quick.py << 'EOF'
import pytest, torch, os
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")

@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-specific test"
)
def test_vit_on_strix():
    """Quick ViT test on Strix"""
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model = model.to('cuda')
    
    image = Image.new('RGB', (224, 224), color='blue')
    inputs = processor(images=image, return_tensors="pt").to('cuda')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    assert outputs.logits.device.type == "cuda"
    print(f"✅ ViT working on {AMDGPU_FAMILIES}!")
EOF

# 3. Run it
export AMDGPU_FAMILIES=gfx1151
export THEROCK_BIN_DIR=/opt/rocm/bin
pytest tests/strix_ai/test_vit_quick.py -v -s
```

### **Option 2: Browse Templates**

See `STRIX_AI_QUICK_START.md` for ready-to-use templates:
- Template 1: CLIP (VLM)
- Template 2: YOLO (Object Detection)
- Template 3: SegFormer (Segmentation)
- Template 4: Quantization Benchmark

---

## 📊 Implementation Roadmap

### **Phase 1: Critical (Immediate)** - 6-8 weeks

**Priority:** Validate core AI workloads

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | VLM Tests (CLIP, LLaVA) | `tests/strix_ai/vlm/` |
| 2 | ViT Tests (ViT-Base, DINOv2) | `tests/strix_ai/vit/` |
| 3-4 | Object Detection (YOLOv8, DETR) | `tests/strix_ai/cv/test_detection.py` |
| 5-6 | Edge Optimization (Quantization, ONNX) | `tests/strix_ai/optimization/` |
| 7-8 | Integration with CI, Documentation | Updated test matrix |

**Success Criteria:**
- ✅ LLaVA 7B runs on Strix (< 2s latency)
- ✅ ViT-Base achieves > 30 FPS
- ✅ YOLOv8 real-time detection (> 15 FPS)
- ✅ INT8 quantization shows 2-3x speedup

### **Phase 2: High Priority (Q1 2026)** - 6-8 weeks

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | Semantic Segmentation (SegFormer) | `tests/strix_ai/cv/test_segmentation.py` |
| 3-4 | Video Processing (encode/decode) | `tests/strix_ai/video/` |
| 5-6 | Windows AI Platform (DirectML) | `tests/strix_ai/windows/` |
| 7-8 | iGPU Optimizations | `tests/strix_ai/optimization/test_igpu.py` |

### **Phase 3: Medium Priority (Q2 2026)** - 4-6 weeks

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | Consumer AI Applications | `tests/strix_ai/consumer/` |
| 3-4 | Additional CV Tasks (pose, depth) | Extended CV suite |
| 5-6 | Performance Benchmarking Dashboard | Metrics tracking |

---

## 💡 Key Recommendations

### **For Management**
1. **Prioritize VLM + ViT tests** - Core to Windows Copilot+ value prop
2. **Allocate 2-3 engineer-months** for Phase 1 implementation
3. **Establish performance baselines** early (latency, FPS, power)
4. **Track vs. competition** (Intel NPU, Qualcomm Hexagon)

### **For Engineers**
1. **Start with templates** in `STRIX_AI_QUICK_START.md`
2. **Use shared fixtures** to avoid duplication
3. **Benchmark early and often** - iGPU performance varies
4. **Test on real hardware** - emulators don't capture memory behavior
5. **Document failures** - iGPU constraints are learning opportunities

### **For Test Infrastructure**
1. **Add `strix_ai` test category** to CI matrix
2. **Increase timeout** for AI tests (30-60 min vs 5-15 min)
3. **Cache models** - Hugging Face downloads are slow
4. **Monitor memory** - iGPU shares with system
5. **Shard by model** - not by test function

---

## 📏 Success Metrics

### **Coverage Metrics**
| Metric | Target | Current |
|--------|--------|---------|
| **AI Model Coverage** | 15+ models | 0 models ❌ |
| **CV Task Coverage** | 6+ tasks | 0 tasks ❌ |
| **Edge Optimization Tests** | 5+ techniques | 0 tests ❌ |
| **Platform Integration** | Windows + Linux | Linux only ⚠️ |

### **Performance Metrics** (Strix Halo gfx1151)
| Workload | Target | Status |
|----------|--------|--------|
| LLaVA-7B inference | < 2s | 🔴 Not measured |
| ViT-Base throughput | > 30 FPS | 🔴 Not measured |
| YOLOv8n detection | > 15 FPS | 🔴 Not measured |
| Memory efficiency | < 4GB peak | 🔴 Not measured |

### **Quality Metrics**
| Metric | Target | Status |
|--------|--------|--------|
| Test pass rate | > 95% | N/A (no tests) |
| Numerical accuracy | Within 2% of ref | N/A |
| CI integration | Pre-submit + Nightly | N/A |

---

## 📞 Support & Questions

### **Getting Started Questions**
- **"Which test should I implement first?"**  
  → Start with CLIP (VLM) - easiest to implement, high business value
  
- **"What hardware do I need?"**  
  → Strix Halo (gfx1151) with 16GB+ RAM minimum

- **"How long will tests take to run?"**  
  → Initial: 30-60 min, Optimized: 15-30 min with caching

### **Technical Questions**
- **"Models don't fit in memory!"**  
  → Use `torch.float16` and/or INT8 quantization (see optimization section)
  
- **"First run is slow!"**  
  → Expected - HIP kernels compile on first run (warmup!)
  
- **"How do I debug iGPU-specific issues?"**  
  → Monitor with `rocm-smi`, check shared memory usage

### **Process Questions**
- **"Do I need to add to CI?"**  
  → Yes for Phase 1 critical tests, optional for exploratory tests
  
- **"What about test data?"**  
  → Use synthetic data for CI, real images for manual validation
  
- **"How do I benchmark?"**  
  → Use `ModelBenchmark` class from `STRIX_AI_QUICK_START.md`

---

## 🗂️ File Structure

```
TheRock/
├── docs/development/
│   ├── STRIX_TESTING_GUIDE.md              # Complete reference (29KB)
│   ├── STRIX_TESTING_QUICK_REFERENCE.md    # Quick lookup (5KB)
│   ├── STRIX_AI_ML_TEST_PLAN.md            # AI test plan (32KB) ⭐ NEW
│   ├── STRIX_AI_QUICK_START.md             # Implementation guide (17KB) ⭐ NEW
│   └── STRIX_TESTING_SUMMARY.md            # This file (overview)
│
├── tests/
│   ├── test_rocm_sanity.py                 # Existing: Sanity tests
│   ├── harness/                            # Existing: PyTest harness
│   │
│   └── strix_ai/                           # NEW: AI/ML tests for Strix ⭐
│       ├── __init__.py
│       ├── conftest.py                     # Shared fixtures
│       ├── utils.py                        # Helper functions
│       │
│       ├── vlm/                            # Vision Language Models
│       │   ├── test_clip.py
│       │   ├── test_llava.py
│       │   └── test_qwen_vl.py
│       │
│       ├── vit/                            # Vision Transformers
│       │   ├── test_vit_base.py
│       │   ├── test_dinov2.py
│       │   └── test_swin.py
│       │
│       ├── cv/                             # Computer Vision
│       │   ├── test_yolo.py
│       │   ├── test_detr.py
│       │   ├── test_segmentation.py
│       │   └── test_pose.py
│       │
│       ├── optimization/                   # Edge Inference
│       │   ├── test_quantization.py
│       │   ├── test_onnx.py
│       │   └── test_torch_compile.py
│       │
│       ├── video/                          # Video Processing
│       │   ├── test_encoding.py
│       │   └── test_realtime.py
│       │
│       ├── windows/                        # Windows AI (Windows only)
│       │   ├── test_directml.py
│       │   └── test_winml.py
│       │
│       └── benchmarks/                     # Performance
│           ├── benchmark_throughput.py
│           └── benchmark_latency.py
│
└── build_tools/github_actions/
    ├── fetch_test_configurations.py        # Test matrix (add strix_ai)
    └── amdgpu_family_matrix.py             # Platform config
```

---

## 🎯 Bottom Line

### **What You Have Now**
✅ Excellent low-level library coverage (rocBLAS, MIOpen, etc.)  
✅ Basic PyTorch sanity tests  
✅ Solid test infrastructure  

### **What You Need**
❌ **AI/ML workload validation** for Edge AI use cases  
❌ **Vision Language Models** (LLaVA, CLIP)  
❌ **Vision Transformers** (ViT, DINOv2)  
❌ **Computer Vision** (YOLO, segmentation)  
❌ **Edge optimizations** (quantization, ONNX)  
❌ **Windows AI integration** (DirectML)  

### **Why It Matters**
Strix is **NOT for data center training** (that's MI series).  
Strix is **NOT for gaming** (that's Navi discrete).  
Strix is for **Windows Copilot+, Edge AI, Consumer workloads**.  

**Current tests don't validate the Strix value proposition!**

### **Next Action**
1. **Review** `STRIX_AI_ML_TEST_PLAN.md` for full details
2. **Implement** first test using `STRIX_AI_QUICK_START.md` templates
3. **Measure** baseline performance on Strix hardware
4. **Iterate** based on results

---

**Created:** December 11, 2025  
**Status:** Documentation complete, implementation pending  
**Estimated Effort:** 6-8 weeks for Phase 1 (Critical tests)

