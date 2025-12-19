# Strix AI/ML Comprehensive Testing Workflow Guide

## Overview

The `strix_ai_comprehensive_tests.yml` workflow provides complete testing coverage for all Strix AI/ML models across multiple market segments.

## Workflow Features

### ✅ **13 Parallel Test Jobs**
1. **VLM Tests** - Vision-Language Models (Qwen-VL, CLIP/BLIP)
2. **VLA Tests** - Vision-Language-Action (OpenVLA, Pi0)
3. **Omni Tests** - Multimodal models (Qwen Omni)
4. **Instruct Tests** - Instruction following (Qwen3-Instruct)
5. **Segmentation Tests** - SAM2
6. **ASR Tests** - Speech recognition (zipformer, crossformer)
7. **Diffusion Tests** - Generative AI (Flux, SD3)
8. **LLM Tests** - Language models (Llama, Deepseek)
9. **Optimization Tests** - Quantization (AWQ, GPTQ)
10. **Profiling Tests** - ROCProfiler v3
11. **Benchmark Tests** - Market segment scorecards
12. **Quick Tests** - Smoke tests
13. **Summary Report** - Aggregates results

### 🎯 **Key Capabilities**
- ✅ Parallel execution for faster testing
- ✅ Conditional job execution based on test category
- ✅ Market segment filtering (automotive, industrial, robotics, healthcare)
- ✅ Test type selection (functional, performance, full)
- ✅ **Uses rocm/pytorch:latest Docker container with pre-installed ROCm**
- ✅ GPU variant selection (gfx1150, gfx1151)
- ✅ Automatic artifact upload (30-day retention)
- ✅ Summary report generation

---

## Workflow Inputs

### **Platform** (Required)
- `linux` (default)
- `windows`

### **Strix Variant** (Required)
- `gfx1150` - Strix Point
- `gfx1151` - Strix Halo (default)

### **Test Category** (Required)
Select which tests to run:
- `all` - Run all test categories
- `vlm` - Vision-Language Models only
- `vla` - Vision-Language-Action only
- `omni` - Multimodal Omni models only
- `instruct` - Instruction following only
- `segmentation` - SAM2 segmentation only
- `asr` - Speech recognition only
- `diffusion` - Generative AI only
- `llm` - Language models only
- `optimization` - Quantization tests only
- `profiling` - ROCProfiler tests only
- `benchmarks` - Market segment benchmarks only
- `quick` - Quick smoke tests (default)

### **Market Segment** (Optional, for benchmarks)
- `all` (default)
- `automotive`
- `industrial`
- `robotics`
- `healthcare`

### **Test Type** (Required)
- `functional` - Correctness validation only (default)
- `performance` - Performance metrics only
- `full` - Functional + Performance

### **Runner Label** (Optional)
- Leave empty for default (`linux-strix-halo-gpu-rocm`)
- Custom runner label for testing

---

## ROCm Environment Setup

### **Docker Container Approach**

This workflow uses the **rocm/pytorch:latest** Docker container which includes:
- ✅ **Pre-installed ROCm** - No manual installation needed
- ✅ **PyTorch with ROCm support** - Ready to use
- ✅ **GPU device access** - /dev/kfd and /dev/dri configured
- ✅ **All ROCm tools** - rocminfo, hipconfig, rocprofv3 available

**Container Configuration**:
```yaml
container:
  image: rocm/pytorch:latest
  options: '--ipc host --group-add video --device /dev/kfd --device /dev/dri --group-add 110 --user 0:0'
```

**Benefits**:
- No dependency on `install_rocm_from_artifacts.py`
- No boto3 or AWS credentials needed
- Consistent ROCm environment across runs
- Faster job startup (no ROCm download/install)

**Note**: This eliminates the `ModuleNotFoundError: No module named 'boto3'` error from manual ROCm installation.

---

## Usage Examples

### **1. Quick Smoke Test (Default)**
```yaml
# Runs automatically on push/PR to strix_poc branch
# Or manually trigger with defaults:
Inputs:
  platform: linux
  strix_variant: gfx1151
  test_category: quick
  test_type: functional
```

**Result**: Runs quick smoke tests (~10-15 minutes)

---

### **2. VLM Functional Tests**
```yaml
Inputs:
  platform: linux
  strix_variant: gfx1151
  test_category: vlm
  test_type: functional
```

**Result**: Runs all VLM functional tests (Qwen2.5-VL, Qwen3-VL, CLIP)

---

### **3. VLM Performance Tests**
```yaml
Inputs:
  platform: linux
  strix_variant: gfx1151
  test_category: vlm
  test_type: performance
```

**Result**: Runs VLM latency, throughput, memory tests

---

### **4. VLM Full Tests (Functional + Performance)**
```yaml
Inputs:
  platform: linux
  strix_variant: gfx1151
  test_category: vlm
  test_type: full
```

**Result**: Runs both functional and performance VLM tests

---

### **5. Automotive Market Segment Benchmark**
```yaml
Inputs:
  platform: linux
  strix_variant: gfx1151
  test_category: benchmarks
  market_segment: automotive
```

**Result**: Runs automotive scorecard (Qwen-VL, CLIP, ASR latency tests)

---

### **6. All Tests (Complete Suite)**
```yaml
Inputs:
  platform: linux
  strix_variant: gfx1151
  test_category: all
  test_type: full
  market_segment: all
```

**Result**: Runs all 12 test jobs in parallel (~2 hours)

---

### **7. ROCProfiler Profiling Tests**
```yaml
Inputs:
  platform: linux
  strix_variant: gfx1151
  test_category: profiling
```

**Result**: Runs ROCProfiler v3 profiling tests, generates .pftrace files

---

### **8. Strix Point (gfx1150) Testing**
```yaml
Inputs:
  platform: linux
  strix_variant: gfx1150
  test_category: quick
```

**Result**: Runs smoke tests on Strix Point hardware

---

### **9. Industrial Market Tests**
```yaml
Inputs:
  platform: linux
  strix_variant: gfx1151
  test_category: benchmarks
  market_segment: industrial
```

**Result**: Runs industrial scorecard (SAM2, Qwen3-Instruct, Qwen-VL)

---

## Manual Workflow Trigger

### Via GitHub UI:
1. Go to: `Actions` → `Strix AI/ML Comprehensive Testing`
2. Click `Run workflow`
3. Select branch: `users/rponnuru/strix_poc`
4. Fill in inputs
5. Click `Run workflow`

### Via GitHub CLI:
```bash
gh workflow run strix_ai_comprehensive_tests.yml \
  --ref users/rponnuru/strix_poc \
  -f platform=linux \
  -f strix_variant=gfx1151 \
  -f test_category=vlm \
  -f test_type=functional
```

---

## Test Job Execution Matrix

| Category | Jobs Run | Approx. Time | Artifacts Generated |
|----------|----------|--------------|---------------------|
| **quick** | 1 job | 10-15 min | test-results-quick-smoke.xml |
| **vlm** | 1 job | 30-45 min | test-results-vlm-*.xml |
| **vla** | 1 job | 20-30 min | test-results-vla.xml |
| **instruct** | 1 job | 20-30 min | test-results-instruct.xml |
| **segmentation** | 1 job | 15-25 min | test-results-segmentation.xml |
| **asr** | 1 job | 20-30 min | test-results-asr.xml |
| **llm** | 1 job | 25-35 min | test-results-llm.xml |
| **optimization** | 1 job | 20-30 min | test-results-optimization.xml |
| **profiling** | 1 job | 40-60 min | profiling-results.xml + .pftrace files |
| **benchmarks** | 1-4 jobs | 60-90 min | benchmark-results-*.xml |
| **all** | 12 jobs | 90-120 min | All artifact files |

---

## Artifacts

### **Artifact Retention**: 30 days

### **Artifact Types**:

1. **Functional Test Results**
   - `strix-vlm-test-results-gfx1151/`
     - `test-results-vlm-functional.xml`
   - `strix-instruct-test-results/`
     - `test-results-instruct.xml`
   - etc.

2. **Performance Test Results**
   - `strix-vlm-test-results-gfx1151/`
     - `test-results-vlm-performance.xml`

3. **Profiling Results**
   - `strix-profiling-test-results/`
     - `test-results-profiling.xml`
     - `v3_traces/*.pftrace`
     - Kernel statistics

4. **Benchmark Results**
   - `strix-benchmark-results-automotive/`
     - `benchmark-results-automotive.xml`
   - `strix-benchmark-results-industrial/`
     - `benchmark-results-industrial.xml`
   - etc.

### **Downloading Artifacts**:
1. Go to workflow run
2. Scroll to "Artifacts" section at bottom
3. Click artifact name to download ZIP

---

## Test Categories Explained

### **VLM (Vision-Language Models)**
**Models**: Qwen2.5-VL 3B/7B, Qwen3-VL 4B, CLIP 0.5B (all with AWQ)  
**Tests**: Model loading, inference, latency, memory  
**Market**: Automotive, Industrial, Robotics

### **VLA (Vision-Language-Action)**
**Models**: OpenVLA 7B (GPTQ/AWQ), Pi0 0.5B  
**Tests**: Action prediction, policy inference  
**Market**: Robotics

### **Omni (Multimodal)**
**Models**: Qwen2.5-Omni 7B, Qwen3-Omni 7B (AWQ)  
**Tests**: Multimodal fusion, audio-visual-text processing  
**Market**: Automotive

### **Instruct (Instruction Following)**
**Models**: Qwen3-Instruct 4B (AWQ)  
**Tests**: Natural language instruction execution  
**Market**: Industrial, Robotics, Healthcare

### **Segmentation**
**Models**: SAM2 0.2B  
**Tests**: Image segmentation, mask generation, FPS  
**Market**: Industrial, Healthcare

### **ASR (Speech Recognition)**
**Models**: zipformer, crossformer (<0.3B)  
**Tests**: Speech-to-text, WER, Real-Time Factor  
**Market**: Automotive, Robotics, Industrial

### **Diffusion (Generative AI)**
**Models**: Flux-1-schnell, Stable Diffusion 3, SD 3.5 XL Turbo  
**Tests**: Image generation, quality validation  
**Market**: Industrial

### **LLM (Language Models)**
**Models**: Llama-3.2-1B/3B, Deepseek-R1-1.5B  
**Tests**: Text generation, reasoning, tokens/sec  
**Market**: General AI

### **Optimization (Quantization)**
**Tests**: AWQ/GPTQ validation, speedup vs FP16, memory reduction  
**Market**: All (Edge AI optimization)

### **Profiling**
**Tool**: ROCProfiler v3  
**Tests**: Kernel timing, memory bandwidth, bottleneck analysis  
**Output**: Perfetto traces (.pftrace)

### **Benchmarks**
**Types**: Automotive, Industrial, Robotics, Healthcare scorecards  
**Tests**: Market-specific performance targets validation

---

## Job Dependencies

```
All test jobs run in parallel
         ↓
    test_summary
    (aggregates all results)
```

**No inter-job dependencies** - All test jobs can run simultaneously for maximum speed.

---

## Automatic Triggers

### **Push Trigger**:
```yaml
on:
  push:
    branches:
      - 'users/rponnuru/strix_poc'
    paths:
      - 'tests/strix_ai/**'
      - '.github/workflows/strix_ai*.yml'
```

**Result**: Runs quick smoke tests automatically

### **Pull Request Trigger**:
```yaml
on:
  pull_request:
    branches:
      - 'main'
      - 'develop'
    paths:
      - 'tests/strix_ai/**'
      - '.github/workflows/strix_ai*.yml'
```

**Result**: Runs quick smoke tests on PRs

---

## Environment Variables

Each job sets:
```bash
# Test Configuration
AMDGPU_FAMILIES=gfx1151      # GPU variant
TEST_CATEGORY=vlm            # Test category
TEST_TYPE=functional         # Test type
MARKET_SEGMENT=automotive    # Market segment (benchmarks)
PYTHONUNBUFFERED=1           # Python output

# ROCm GPU Detection (Critical for Strix)
HSA_OVERRIDE_GFX_VERSION=11.0.1  # 11.0.0 for gfx1150, 11.0.1 for gfx1151
ROCR_VISIBLE_DEVICES=0           # GPU device visibility
GPU_DEVICE_ORDINAL=0             # GPU device ordinal
```

**ROCm Environment Variables Explained**:
- **`HSA_OVERRIDE_GFX_VERSION`**: Critical for Strix GPU recognition. Tells ROCm runtime which GPU architecture to use.
  - `11.0.0` for Strix Point (gfx1150)
  - `11.0.1` for Strix Halo (gfx1151)
- **`ROCR_VISIBLE_DEVICES`**: Controls which ROCm devices are visible to the application
- **`GPU_DEVICE_ORDINAL`**: Specifies the GPU device ordinal for HIP/ROCm

**Note**: ROCm paths (PATH, LD_LIBRARY_PATH) are pre-configured in the rocm/pytorch container.

---

## Troubleshooting

### **Job Skipped**
- **Cause**: Test category doesn't match
- **Solution**: Check `if` condition in job definition

### **GPU Not Detected**
- **Cause**: Tests skip if not on Strix GPU
- **Solution**: Verify runner has gfx1150 or gfx1151

### **Model Not Found**
- **Cause**: Model not available on Hugging Face or requires setup
- **Solution**: Check test for `pytest.skip()` - some models are placeholders

### **rocprofv3 Not Found**
- **Cause**: ROCProfiler SDK not installed
- **Solution**: Tests skip automatically if rocprofv3 not available

### **Out of Memory**
- **Cause**: Model too large for Strix memory
- **Solution**: Use quantized models (AWQ/GPTQ) or reduce batch size

---

## Performance Expectations

### **Quick Smoke Tests**: < 15 minutes
- Fast validation of basic functionality
- All P0 models with minimal iterations

### **Functional Tests**: 20-45 minutes per category
- Complete correctness validation
- Multiple models per category

### **Performance Tests**: 30-60 minutes per category
- Latency measurements (10-100 iterations)
- Memory profiling
- Throughput benchmarks

### **Profiling Tests**: 40-90 minutes
- ROCProfiler v3 trace generation
- Kernel analysis
- Bottleneck identification

### **Full Suite**: 90-120 minutes
- All 12 test jobs in parallel
- Complete validation and benchmarking

---

## Best Practices

1. **Start with Quick Tests**: Use `test_category: quick` for initial validation
2. **Iterate on Specific Categories**: Focus on one category at a time during development
3. **Use Functional First**: Run functional tests before performance tests
4. **Benchmark Last**: Run benchmarks after individual tests pass
5. **Profile Selectively**: Only profile models with performance concerns
6. **Check Artifacts**: Always download and review XML results
7. **Monitor Time**: Set appropriate timeouts for long-running tests

---

## Related Files

- **Workflow**: `.github/workflows/strix_ai_comprehensive_tests.yml`
- **Tests**: `tests/strix_ai/*/`
- **Config**: `tests/strix_ai/conftest.py`
- **Summary**: `tests/strix_ai/TEST_SUMMARY.md`

---

## Contact & Support

For issues or questions:
- Check workflow logs for error details
- Review test file documentation
- Verify GPU access and ROCm installation
- Check model availability on Hugging Face

