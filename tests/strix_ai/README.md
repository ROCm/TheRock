# Strix AI/ML Test Suite

This directory contains AI/ML tests for AMD Strix platforms (gfx1150, gfx1151).

## 🎯 Purpose

Validate AI/ML workloads specific to Strix Edge AI use cases that are **NOT covered** by existing MI (datacenter) or Navi (gaming) test suites.

## 📁 Directory Structure

```
strix_ai/
├── vlm/              # Vision Language Models (CLIP, LLaVA)
├── vla/              # Vision Language Action (OWL-ViT, Action Recognition)
├── vit/              # Vision Transformers (ViT, DINOv2, Swin)
├── cv/               # Computer Vision (YOLO, DETR, Segmentation)
├── optimization/     # Edge Inference (Quantization, ONNX)
├── profiling/        # ROCProfiler integration tests ⭐ NEW
├── video/            # Video Processing (Encode/Decode)
├── windows/          # Windows AI Platform (DirectML, WinML)
└── benchmarks/       # Performance Benchmarking
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install transformers accelerate ultralytics onnxruntime-rocm opencv-python pillow torch torchvision
```

### Set Environment Variables

```bash
# Linux
export THEROCK_BIN_DIR=/opt/rocm/bin
export AMDGPU_FAMILIES=gfx1151

# Windows
$env:THEROCK_BIN_DIR="C:\rocm\bin"
$env:AMDGPU_FAMILIES="gfx1151"
```

### Run Tests

```bash
# Run all Strix AI tests
pytest tests/strix_ai/ -v

# Run specific category
pytest tests/strix_ai/vlm/ -v        # VLM tests only
pytest tests/strix_ai/vit/ -v        # ViT tests only
pytest tests/strix_ai/cv/ -v         # CV tests only
pytest tests/strix_ai/profiling/ -v  # Profiling tests only

# Run by priority
pytest tests/strix_ai/ -m "p0" -v    # Critical tests only
pytest tests/strix_ai/ -m "p1" -v    # High priority tests

# Run quick smoke tests
pytest tests/strix_ai/ -m "quick" -v

# Skip slow tests
pytest tests/strix_ai/ -m "not slow" -v
```

## 📊 Test Categories

| Category | Priority | Models | Status |
|----------|----------|--------|--------|
| **VLM** | 🔴 P0 | CLIP, LLaVA | ✅ Implemented |
| **VLA** | 🔴 P0 | OWL-ViT, Action Recognition | ✅ Implemented |
| **ViT** | 🔴 P0 | ViT-Base, DINOv2 | ✅ Implemented |
| **CV** | 🔴 P0 | YOLOv8, DETR | ✅ Implemented |
| **Optimization** | 🔴 P0 | Quantization | ✅ Implemented |
| **Profiling** | 🟡 P1 | ROCProfiler + PyTorch/AI | ✅ Implemented ⭐ NEW |
| **Video** | 🟡 P1 | Encode/Decode | ⏳ TODO |
| **Windows** | 🟡 P1 | DirectML | ⏳ TODO |

## 🔧 Test Markers

Tests use pytest markers for categorization:

- `@pytest.mark.strix` - Strix-specific tests
- `@pytest.mark.vlm` - Vision Language Model tests
- `@pytest.mark.vit` - Vision Transformer tests
- `@pytest.mark.cv` - Computer Vision tests
- `@pytest.mark.profiling` - ROCProfiler integration tests ⭐ NEW
- `@pytest.mark.slow` - Tests taking > 30 seconds
- `@pytest.mark.quick` - Quick smoke tests
- `@pytest.mark.p0` - Priority 0 (Critical)
- `@pytest.mark.p1` - Priority 1 (High)
- `@pytest.mark.p2` - Priority 2 (Medium)
- `@pytest.mark.windows` - Windows-only tests

## 📚 Documentation

For detailed information, see:

- [`STRIX_AI_ML_TEST_PLAN.md`](../../docs/development/STRIX_AI_ML_TEST_PLAN.md) - Comprehensive test plan
- [`STRIX_AI_QUICK_START.md`](../../docs/development/STRIX_AI_QUICK_START.md) - Implementation guide
- [`STRIX_TESTING_GUIDE.md`](../../docs/development/STRIX_TESTING_GUIDE.md) - General Strix testing

## ⚠️ Known Issues

### Memory Constraints
- **Windows Strix Halo (gfx1151)**: Limited memory - use smaller models or quantization
- **Solution**: Tests automatically use `torch.float16` or INT8 quantization

### First Run Performance
- **Issue**: First run is slow due to kernel compilation
- **Solution**: Tests include warmup iterations

### Model Downloads
- **Issue**: Hugging Face models can be large (5-10GB)
- **Solution**: Set `HF_HOME` to cache directory with sufficient space

## 🐛 Troubleshooting

### CUDA Not Available
```bash
# Check ROCm installation
rocm-smi
rocminfo

# Verify THEROCK_BIN_DIR
echo $THEROCK_BIN_DIR
```

### Out of Memory (OOM)
```python
# Use smaller models or quantization
model = AutoModel.from_pretrained("model-name", torch_dtype=torch.float16)
```

### Skipped Tests
```bash
# Check AMDGPU_FAMILIES environment variable
echo $AMDGPU_FAMILIES

# Should be "gfx1150" or "gfx1151"
```

## 📈 Performance Targets

| Workload | Target | Measurement |
|----------|--------|-------------|
| LLaVA-7B inference | < 2s | Time to first token |
| ViT-Base throughput | > 30 FPS | Images per second |
| YOLOv8n detection | > 15 FPS | Real-time video |
| Memory usage | < 4GB peak | Peak allocation |

## 🤝 Contributing

When adding new tests:

1. Follow existing test structure
2. Use shared fixtures from `conftest.py`
3. Add appropriate pytest markers
4. Document expected performance
5. Handle cleanup (GPU memory)
6. Add to appropriate category directory

## 📞 Support

For questions or issues:
- Check documentation in `docs/development/`
- Review existing test examples
- File issues on GitHub

