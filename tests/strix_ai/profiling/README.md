# ROCProfiler Tests for Strix AI/ML Workloads

This directory contains ROCProfiler integration tests for Strix AI/ML workloads running on AMD Strix GPUs (gfx1150, gfx1151).

## 📋 Overview

These tests validate and profile AI/ML workloads using ROCProfiler tools on Strix integrated GPUs. They help measure performance, identify bottlenecks, and ensure ROCm profiling tools work correctly with PyTorch and AI models.

## 🎯 Test Categories

### 1. PyTorch Profiling (`test_pytorch_profiling.py`)
Tests basic ROCProfiler integration with PyTorch operations:
- ✅ **GPU availability check** - Verify Strix GPU is detected
- ✅ **ROCProfiler installation** - Check rocprof and rocprofv3 tools
- ✅ **Simple inference profiling** - Profile basic neural network inference
- ✅ **Training step profiling** - Profile forward/backward pass with gradients
- ✅ **External profiling** - Test rocprof command-line tool
- ✅ **Quick smoke test** - Fast validation of profiling capability

### 2. AI Workload Profiling (`test_ai_workload_profiling.py`)
Profiles real-world AI models on Strix:
- 🖼️ **CLIP profiling** - Vision-Language Model (openai/clip-vit-base-patch32)
- 🎨 **ViT profiling** - Vision Transformer (google/vit-base-patch16-224)
- 🎯 **YOLO profiling** - Object detection (YOLOv8)
- 📊 **Batch size analysis** - Profile different batch sizes
- ⚡ **Quick smoke test** - Fast Conv2d operation profiling

## 🚀 Running Tests

### Run All Profiling Tests
```bash
# From TheRock root directory
python3 -m pytest tests/strix_ai/profiling/ -v -s
```

### Run Specific Test Categories
```bash
# PyTorch profiling only
python3 -m pytest tests/strix_ai/profiling/test_pytorch_profiling.py -v -s

# AI workload profiling only
python3 -m pytest tests/strix_ai/profiling/test_ai_workload_profiling.py -v -s

# Quick smoke tests only
python3 -m pytest tests/strix_ai/profiling/ -v -s -m quick

# VLM profiling only
python3 -m pytest tests/strix_ai/profiling/ -v -s -m vlm

# ViT profiling only
python3 -m pytest tests/strix_ai/profiling/ -v -s -m vit
```

### Run with Specific GPU
```bash
# Strix Halo (gfx1151)
AMDGPU_FAMILIES=gfx1151 python3 -m pytest tests/strix_ai/profiling/ -v -s

# Strix Point (gfx1150)
AMDGPU_FAMILIES=gfx1150 python3 -m pytest tests/strix_ai/profiling/ -v -s
```

### Generate JUnit XML Results
```bash
python3 -m pytest tests/strix_ai/profiling/ -v -s \
  --junit-xml=profiling-results.xml
```

## 🔧 Prerequisites

### Container Environment (Recommended)
Using the `rocm/pytorch:latest` container (as configured in CI):
```bash
docker run -it --rm \
  --ipc=host \
  --group-add video \
  --device /dev/kfd \
  --device /dev/dri \
  -v $(pwd):/workspace \
  rocm/pytorch:latest \
  bash

# Inside container
cd /workspace
pip install pytest pytest-check transformers ultralytics
python3 -m pytest tests/strix_ai/profiling/ -v -s
```

### Native Installation
Requirements:
- ✅ **ROCm 6.x** or later
- ✅ **PyTorch with ROCm** support
- ✅ **rocprofiler-sdk** (provides rocprofv3) OR **roctracer** (provides rocprof)
- ✅ **Python 3.8+**
- ✅ **Strix GPU** (gfx1150 or gfx1151)

Install dependencies:
```bash
pip install pytest pytest-check torch transformers ultralytics pillow
```

## 📊 Profiling Tools

### PyTorch Built-in Profiler
All tests use PyTorch's built-in profiler which integrates with ROCm:
```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    # Your code here
    model(input)
    torch.cuda.synchronize()

# View results
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### External ROCProfiler Tools

#### rocprof (roctracer)
Legacy profiling tool:
```bash
rocprof --stats -o results.csv python my_script.py
```

#### rocprofv3 (rocprofiler-sdk)
New SDK-based profiler:
```bash
rocprofv3 --hip-trace python my_script.py
```

## 🎨 Test Markers

Tests are organized with pytest markers:
- `@pytest.mark.strix` - Strix platform tests
- `@pytest.mark.profiling` - Profiling tests
- `@pytest.mark.vlm` - Vision Language Model tests
- `@pytest.mark.vit` - Vision Transformer tests
- `@pytest.mark.cv` - Computer Vision tests
- `@pytest.mark.quick` - Quick smoke tests
- `@pytest.mark.slow` - Long-running tests (>30s)
- `@pytest.mark.p0` - Priority 0 (Critical)
- `@pytest.mark.p1` - Priority 1 (High)
- `@pytest.mark.p2` - Priority 2 (Medium)

Run by marker:
```bash
pytest tests/strix_ai/profiling/ -m "profiling and quick"
pytest tests/strix_ai/profiling/ -m "vlm or vit"
pytest tests/strix_ai/profiling/ -m "p0 or p1"
```

## 📈 Interpreting Results

### Profiling Output Example
```
=== Top 15 GPU Operations ===
-------------------------------------------------------  ------------  ------------  
Name                                                     CPU time      CUDA time     
-------------------------------------------------------  ------------  ------------  
aten::linear                                             1.234 ms      45.678 ms     
aten::matmul                                             0.567 ms      23.456 ms     
aten::addmm                                              0.345 ms      12.345 ms     
...

✓ Total GPU time: 123.45 ms
✓ Total CPU time: 23.45 ms
```

### Key Metrics
- **CUDA time** - GPU execution time (most important for GPU workloads)
- **CPU time** - Host CPU time (data prep, host-side operations)
- **Operations** - Individual kernel launches and operations
- **Shapes** - Tensor dimensions (helps identify memory usage)

### Performance Tips
1. **GPU Time >> CPU Time** - Good GPU utilization
2. **Many small operations** - Consider operator fusion
3. **Large data transfers** - Consider pinned memory or data caching
4. **Long CPU times** - Bottleneck in data preprocessing

## 🔍 Troubleshooting

### GPU Not Detected
```bash
# Check GPU visibility
rocminfo | grep gfx115

# Check PyTorch GPU detection
python3 -c "import torch; print(torch.cuda.is_available())"
```

### ROCProfiler Not Found
```bash
# Check for rocprof (roctracer)
which rocprof
rocprof --version

# Check for rocprofv3 (rocprofiler-sdk)
which rocprofv3
rocprofv3 --version

# In container, install if needed
apt-get update && apt-get install -y rocprofiler-dev
```

### Test Skipped
Tests may skip if:
- Not running on Strix GPU (gfx1150/gfx1151)
- PyTorch/GPU not available
- Required libraries not installed (transformers, ultralytics)
- ROCProfiler tools not found

Check skip reasons:
```bash
pytest tests/strix_ai/profiling/ -v -ra
```

### Memory Issues
```bash
# Clear GPU cache between tests
python3 -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU memory
rocm-smi
watch -n 1 rocm-smi
```

## 🔗 Integration with CI/CD

### GitHub Actions Workflow
Tests automatically run via `.github/workflows/strix_ai_tests.yml`:

```bash
# Manual trigger with profiling category
gh workflow run strix_ai_tests.yml \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=profiling \
  -f test_type=full
```

### Quick Run
```bash
# Quick smoke tests (for PR validation)
gh workflow run strix_ai_tests.yml \
  -f test_category=profiling \
  -f test_type=quick
```

### Full Validation
```bash
# All profiling tests
gh workflow run strix_ai_tests.yml \
  -f test_category=profiling \
  -f test_type=full
```

## 📚 Related Documentation

- [Strix AI Testing Guide](../docs/development/STRIX_TESTING_GUIDE.md)
- [Strix Client Architecture](../docs/development/STRIX_CLIENT_ARCHITECTURE.md)
- [ROCProfiler Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
- [PyTorch Profiler Guide](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

## 🤝 Contributing

When adding new profiling tests:
1. Follow the existing test structure
2. Use appropriate pytest markers
3. Include warmup iterations before profiling
4. Always call `torch.cuda.synchronize()` before/after timing
5. Print clear profiling results with units
6. Add test to appropriate class (PyTorch vs AI Workload)
7. Document expected behavior and metrics

Example:
```python
@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.p1
def test_my_model_profile(self, strix_device, cleanup_gpu):
    """Profile my custom model"""
    model = MyModel().to(strix_device)
    model.eval()
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(input)
    torch.cuda.synchronize()
    
    # Profile
    with torch.profiler.profile(...) as prof:
        with torch.no_grad():
            output = model(input)
            torch.cuda.synchronize()
    
    # Print results
    print(prof.key_averages().table(...))
```

## ✅ Test Status

| Test Category | Status | Priority | Notes |
|--------------|--------|----------|-------|
| GPU Detection | ✅ Done | P0 | Critical |
| ROCProf Installation | ✅ Done | P0 | Critical |
| Simple PyTorch | ✅ Done | P1 | Basic profiling |
| Training Step | ✅ Done | P1 | Backward pass |
| CLIP Profiling | ✅ Done | P1 | VLM workload |
| ViT Profiling | ✅ Done | P1 | Transformer |
| YOLO Profiling | ✅ Done | P1 | Object detection |
| Batch Analysis | ✅ Done | P2 | Performance |
| External rocprof | ✅ Done | P2 | CLI tool |

## 📞 Support

For issues or questions:
- Check [TROUBLESHOOTING.md](../TROUBLESHOOTING.md)
- Review [Strix Testing Guide](../docs/development/STRIX_TESTING_GUIDE.md)
- Open an issue on [ROCm/TheRock](https://github.com/ROCm/TheRock/issues)

