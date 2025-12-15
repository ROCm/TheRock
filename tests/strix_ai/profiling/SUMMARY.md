# ROCProfiler Integration for Strix AI - Summary

## 🎯 What Was Added

This implementation adds comprehensive ROCProfiler integration tests for Strix AI/ML workloads running on AMD Strix GPUs (gfx1150, gfx1151).

## 📦 New Files Created

### Test Files
1. **`__init__.py`** - Package initialization
2. **`test_pytorch_profiling.py`** - PyTorch profiling tests
   - GPU availability checks
   - ROCProfiler tool detection (rocprof, rocprofv3)
   - Simple neural network inference profiling
   - Training step profiling (forward + backward)
   - External rocprof command-line profiling
   - Quick smoke tests

3. **`test_ai_workload_profiling.py`** - AI/ML model profiling
   - CLIP (Vision-Language Model) profiling
   - ViT (Vision Transformer) profiling
   - YOLO (Object Detection) profiling
   - Batch size performance analysis
   - Quick smoke tests

### Documentation
4. **`README.md`** - Comprehensive profiling test documentation
   - Test overview and categories
   - Running instructions
   - Prerequisites and setup
   - Profiling tools guide
   - Troubleshooting
   - CI/CD integration

5. **`SUMMARY.md`** - This file

## 🔧 Modified Files

### Configuration Updates
1. **`tests/strix_ai/conftest.py`**
   - Added `rocprof_available` fixture
   - Added `rocprofiler_sdk_available` fixture
   - Added `profiler_context` fixture for temporary profiling directories
   - Added `@pytest.mark.profiling` marker

2. **`.github/workflows/strix_ai_tests.yml`**
   - Added "profiling" to test category options
   - Added new test step "Run Strix AI Tests - ROCProfiler Integration"
   - Checks for ROCProfiler tools availability
   - Generates profiling-results.xml

3. **`tests/strix_ai/README.md`**
   - Added profiling to directory structure
   - Added profiling test examples
   - Added profiling category to test matrix
   - Added profiling marker documentation

## ✅ Test Coverage

### PyTorch Profiling Tests
- ✅ GPU detection and validation
- ✅ ROCProfiler installation checks (rocprof + rocprofv3)
- ✅ Simple inference profiling (3-layer MLP)
- ✅ Training step profiling (forward/backward/optimizer)
- ✅ External rocprof CLI tool profiling
- ✅ Quick smoke test (matrix multiplication)

### AI Workload Profiling Tests
- ✅ CLIP inference profiling (openai/clip-vit-base-patch32)
- ✅ ViT inference profiling (google/vit-base-patch16-224)
- ✅ ViT batch size analysis (1, 2, 4, 8)
- ✅ YOLO inference profiling (YOLOv8n)
- ✅ Quick smoke test (Conv2d operation)

## 🚀 How to Run

### Local Testing
```bash
# From TheRock root
cd tests/strix_ai/profiling/

# Run all profiling tests
python3 -m pytest . -v -s

# Quick smoke tests only
python3 -m pytest . -v -s -m quick

# Specific model profiling
python3 -m pytest . -v -s -m vlm  # CLIP
python3 -m pytest . -v -s -m vit  # ViT
python3 -m pytest . -v -s -m cv   # YOLO
```

### CI/CD (GitHub Actions)
```bash
# Trigger profiling tests via workflow dispatch
gh workflow run strix_ai_tests.yml \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=profiling \
  -f test_type=full
```

### Container Environment
```bash
# Using rocm/pytorch container (as in CI)
docker run -it --rm \
  --ipc=host \
  --group-add video \
  --device /dev/kfd \
  --device /dev/dri \
  rocm/pytorch:latest \
  bash

# Inside container
cd /path/to/TheRock
pip install pytest pytest-check transformers ultralytics
AMDGPU_FAMILIES=gfx1151 python3 -m pytest tests/strix_ai/profiling/ -v -s
```

## 📊 Profiling Tools Used

### 1. PyTorch Built-in Profiler
Primary profiling method - integrates with ROCm:
```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    model(input)
    torch.cuda.synchronize()
```

### 2. External ROCProfiler Tools
- **rocprof** (roctracer) - Legacy profiling tool
- **rocprofv3** (rocprofiler-sdk) - New SDK-based profiler

Tests check for availability and use appropriately.

## 🎨 Test Organization

### Markers
All tests use pytest markers:
- `@pytest.mark.strix` - Strix platform
- `@pytest.mark.profiling` - Profiling tests
- `@pytest.mark.vlm` / `@pytest.mark.vit` / `@pytest.mark.cv` - Model category
- `@pytest.mark.quick` - Fast smoke tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.p0` / `@pytest.mark.p1` - Priority levels

### Class Organization
- `TestPyTorchProfiling` - Basic PyTorch operations
- `TestVLMProfiling` - Vision-Language Models
- `TestViTProfiling` - Vision Transformers
- `TestYOLOProfiling` - Object Detection

## 📈 Key Features

### 1. Comprehensive Coverage
- Basic operations to complex AI models
- Multiple profiling methods (built-in + external)
- Performance metrics capture

### 2. Strix-Specific
- GPU detection validates gfx1150/gfx1151
- Tests skip gracefully if not on Strix
- Optimized for integrated GPU memory constraints

### 3. CI/CD Ready
- JUnit XML output for GitHub Actions
- Automatic ROCProfiler tool detection
- Graceful handling of missing dependencies

### 4. Developer-Friendly
- Clear profiling output with timing
- Top operations table (sorted by GPU time)
- Warmup iterations for accurate timing
- GPU memory cleanup fixtures

## 🔍 Example Output

```
=== Profiling ViT Inference on Strix ===
Loading model: google/vit-base-patch16-224
Warming up...
Profiling ViT inference...
✓ Predicted class: 281

=== Top 15 GPU Operations ===
-------------------------------------------------------  ------------  ------------  
Name                                                     CPU time      CUDA time     
-------------------------------------------------------  ------------  ------------  
aten::addmm                                              2.345 ms      67.890 ms     
aten::native_layer_norm                                  1.234 ms      45.678 ms     
aten::copy_                                              0.567 ms      23.456 ms     
...

✓ Total GPU time: 234.56 ms
✓ Total CPU time: 45.67 ms
```

## 🎯 Integration Points

### With Existing Strix AI Tests
- Uses same fixtures from `conftest.py` (strix_device, test_image_224, etc.)
- Follows same marker conventions
- Integrated into same CI workflow
- Complementary to existing AI tests

### With ROCm Stack
- Tests both roctracer (rocprof) and rocprofiler-sdk (rocprofv3)
- Validates profiler tools work on Strix iGPUs
- Provides performance baseline data

## 📋 Next Steps

### Recommended Enhancements
1. Add memory profiling (peak usage, transfers)
2. Add power/efficiency metrics (if available)
3. Add multi-batch throughput tests
4. Add kernel-level analysis tests
5. Add comparison with baseline performance

### Future Work
- Windows profiling support (DirectML integration)
- Video encode/decode profiling
- Mixed precision (FP16/INT8) profiling analysis
- Cross-platform comparison (Linux vs Windows)

## 🤝 Contributing

When adding new profiling tests:
1. Follow existing test structure in `test_*.py` files
2. Use appropriate markers (`@pytest.mark.*`)
3. Include warmup iterations (3-5 runs)
4. Call `torch.cuda.synchronize()` before/after timing
5. Print clear results with units (ms, GB/s, etc.)
6. Clean up GPU memory (`cleanup_gpu` fixture)
7. Document expected behavior in docstring

## 📞 Support

- See [README.md](./README.md) for detailed documentation
- See [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) for common issues
- See [Strix Testing Guide](../../docs/development/STRIX_TESTING_GUIDE.md)

## 🎉 Summary

This implementation successfully integrates ROCProfiler testing into the Strix AI test suite, providing:
- ✅ Comprehensive profiling coverage
- ✅ Multiple profiling methods
- ✅ CI/CD integration
- ✅ Clear documentation
- ✅ Developer-friendly output
- ✅ Strix-optimized tests

The tests are ready to run and provide valuable performance insights for AI/ML workloads on Strix GPUs!

