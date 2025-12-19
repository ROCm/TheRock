# Strix AI/ML Test Suite - Complete Test Files

## Overview

This test suite provides comprehensive testing for AMD Strix AI/ML models and workloads across multiple market segments.

## Test Structure

```
tests/strix_ai/
├── vlm/                          # Vision-Language Models
│   ├── test_qwen25_vl.py        # Qwen2.5-VL-Instruct 3B/7B AWQ
│   ├── test_qwen3_vl.py         # Qwen3-VL-Instruct 4B AWQ
│   └── test_clip_blip.py        # CLIP/BLIP 0.5B AWQ
├── vla/                          # Vision-Language-Action
│   ├── test_openvla.py          # OpenVLA 7B GPTQ/AWQ
│   └── test_pi0.py              # Pi0 0.5B
├── omni/                         # Multimodal
│   └── test_qwen_omni.py        # Qwen2.5/3-Omni 7B AWQ
├── instruct/                     # Instruction Following
│   └── test_qwen3_instruct.py   # Qwen3-Instruct 4B AWQ
├── segmentation/                 # Segmentation
│   └── test_sam2.py             # SAM2 0.2B
├── asr/                          # Speech Recognition
│   ├── test_zipformer.py        # zipformer <0.3B
│   └── test_crossformer.py      # crossformer <0.3B
├── diffusion/                    # Generative AI
│   └── test_flux_sd.py          # Flux, SD3, SD3.5 XL Turbo
├── llm/                          # Language Models
│   ├── test_llama32.py          # Llama-3.2-1B/3B
│   └── test_deepseek_r1.py      # Deepseek-R1-Distill-Qwen-1.5B
├── optimization/                 # Quantization
│   ├── test_awq_quantization.py # AWQ quantization tests
│   └── test_gptq_quantization.py# GPTQ quantization tests
├── profiling/                    # Deep Profiling
│   └── test_strix_models_profiling.py  # ROCProfiler v3
├── benchmarks/                   # Market Segment Benchmarks
│   ├── automotive_bench.py      # Automotive scorecard
│   ├── industrial_bench.py      # Industrial scorecard
│   ├── robotics_bench.py        # Robotics scorecard
│   └── healthcare_bench.py      # Healthcare scorecard
└── conftest.py                   # Pytest configuration
```

## Test Categories

### By Model Category

| Category | Models | Test Count | Priority |
|----------|--------|-----------|----------|
| **VLM** | Qwen2.5-VL (3B, 7B), Qwen3-VL (4B), CLIP/BLIP (0.5B) | 15+ | P0 |
| **VLA** | OpenVLA (7B), Pi0 (0.5B) | 8+ | P1 |
| **Omni** | Qwen2.5-Omni (7B), Qwen3-Omni (7B) | 6+ | P2 |
| **Instruct** | Qwen3-Instruct (4B) | 5+ | P0 |
| **Segmentation** | SAM2 (0.2B) | 6+ | P0 |
| **ASR** | zipformer, crossformer (<0.3B) | 10+ | P1 |
| **Diffusion** | Flux-1-schnell, SD3, SD3.5 XL Turbo | 6+ | P2 |
| **LLM** | Llama-3.2-1B/3B, Deepseek-R1-1.5B | 10+ | P2 |

### By Market Segment

| Segment | Primary Models | Test Focus |
|---------|---------------|------------|
| **Automotive** | Qwen2.5/3-VL, Qwen Omni, CLIP, ASR | Latency < 100ms, Real-time |
| **Industrial** | Qwen3-Instruct, Qwen-VL, SAM2, CLIP | Throughput, Batch processing |
| **Robotics** | OpenVLA, Pi0, Qwen-VL, ASR | Action prediction, Closed-loop |
| **Healthcare** | Qwen3-Instruct, SAM2 | Accuracy > Speed, Segmentation |

### By Test Type

| Type | Purpose | Marker |
|------|---------|--------|
| **Functional** | Correctness validation | `@pytest.mark.functional` |
| **Performance** | Latency, throughput, memory | `@pytest.mark.performance` |
| **Profiling** | ROCProfiler v3 traces | `@pytest.mark.profiling` |
| **Benchmark** | Market segment scorecards | `@pytest.mark.benchmark` |
| **Quick** | Smoke tests (< 10s) | `@pytest.mark.quick` |

## Running Tests

### Basic Usage

```bash
# Run all Strix AI tests
pytest tests/strix_ai/ -v

# Run specific category
pytest tests/strix_ai/vlm/ -v                 # VLM tests
pytest tests/strix_ai/segmentation/ -v        # Segmentation tests

# Run by market segment
pytest tests/strix_ai/ -m "automotive" -v
pytest tests/strix_ai/ -m "industrial" -v
pytest tests/strix_ai/ -m "robotics" -v

# Run by test type
pytest tests/strix_ai/ -m "functional" -v     # Functional only
pytest tests/strix_ai/ -m "performance" -v    # Performance only
pytest tests/strix_ai/ -m "quick" -v          # Quick smoke tests

# Run by priority
pytest tests/strix_ai/ -m "p0" -v             # Critical tests only
pytest tests/strix_ai/ -m "p1" -v             # High priority

# Run AWQ quantized models
pytest tests/strix_ai/ -m "awq" -v

# Run specific model
pytest tests/strix_ai/vlm/test_qwen25_vl.py -v
pytest tests/strix_ai/vlm/test_clip_blip.py::test_clip_latency -v
```

### Advanced Usage

```bash
# Combine markers (AND)
pytest tests/strix_ai/ -m "automotive and performance" -v

# Combine markers (OR)
pytest tests/strix_ai/ -m "automotive or industrial" -v

# Exclude markers
pytest tests/strix_ai/ -m "not slow" -v

# Run with profiling
pytest tests/strix_ai/profiling/ -v -s

# Generate XML reports
pytest tests/strix_ai/vlm/ -v --junit-xml=test-results-vlm.xml

# Run benchmarks
pytest tests/strix_ai/benchmarks/automotive_bench.py -v
pytest tests/strix_ai/benchmarks/industrial_bench.py -v
```

## Test Execution Matrix

### Automotive Market

```bash
# P0 Critical Tests
pytest tests/strix_ai/ -m "automotive and p0" -v

# Specific models
pytest tests/strix_ai/vlm/test_qwen25_vl.py::test_qwen25_vl_3b_awq_latency -v
pytest tests/strix_ai/vlm/test_qwen3_vl.py::test_qwen3_vl_4b_awq_latency -v
pytest tests/strix_ai/vlm/test_clip_blip.py::test_clip_latency -v

# Full automotive suite
pytest tests/strix_ai/benchmarks/automotive_bench.py -v
```

### Industrial Market

```bash
# Industrial tests
pytest tests/strix_ai/ -m "industrial and p0" -v

# Specific models
pytest tests/strix_ai/segmentation/test_sam2.py -v
pytest tests/strix_ai/instruct/test_qwen3_instruct.py -v

# Full industrial suite
pytest tests/strix_ai/benchmarks/industrial_bench.py -v
```

### Robotics Market

```bash
# Robotics tests
pytest tests/strix_ai/ -m "robotics and p1" -v

# Specific models
pytest tests/strix_ai/vla/test_openvla.py -v
pytest tests/strix_ai/vla/test_pi0.py -v

# Full robotics suite
pytest tests/strix_ai/benchmarks/robotics_bench.py -v
```

## Environment Setup

### Required Environment Variables

```bash
# Required
export AMDGPU_FAMILIES=gfx1151
export ROCM_HOME=/opt/rocm
export THEROCK_BIN_DIR=/opt/rocm/bin
export PATH=$THEROCK_BIN_DIR:$PATH

# Optional
export HF_HOME=/path/to/huggingface/cache
export PYTHONUNBUFFERED=1
```

### Required Dependencies

```bash
# Core dependencies
pip install pytest pytest-check
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# AI/ML libraries
pip install transformers accelerate
pip install ultralytics opencv-python pillow
pip install timm einops scipy matplotlib

# Optional (for specific models)
pip install auto-gptq  # For GPTQ models
pip install icefall    # For ASR models
pip install segment-anything  # For SAM2
```

## Test Coverage

### Functional Tests (Correctness)
- ✅ Model loading validation
- ✅ Inference execution
- ✅ Output correctness (shape, no NaN/Inf)
- ✅ Prediction accuracy
- ✅ AWQ/GPTQ quantization validation

### Performance Tests (Metrics)
- ✅ Mean latency (ms)
- ✅ P50/P95/P99 latency percentiles
- ✅ Throughput (FPS, tokens/sec)
- ✅ Peak memory usage (GB)
- ✅ First-run overhead
- ✅ Batch size comparison

### Profiling Tests (Optimization)
- ✅ ROCProfiler v3 integration
- ✅ Kernel timing traces
- ✅ Memory bandwidth analysis
- ✅ HIP API overhead
- ✅ Bottleneck identification
- ✅ Perfetto trace generation

### Benchmark Tests (Market Segments)
- ✅ Automotive scorecard
- ✅ Industrial scorecard
- ✅ Robotics scorecard
- ✅ Healthcare scorecard

## Model Status

| Model | Size | Quantization | Status | Priority |
|-------|------|-------------|--------|----------|
| Qwen2.5-VL-Instruct | 3B | AWQ | ✅ Implemented | P0 |
| Qwen2.5-VL-Instruct | 7B | AWQ | ✅ Implemented | P1 |
| Qwen3-VL-Instruct | 4B | AWQ | ✅ Implemented | P0 |
| Qwen3-Instruct | 4B | AWQ | ✅ Implemented | P0 |
| CLIP | 0.5B | AWQ | ✅ Implemented | P0 |
| SAM2 | 0.2B | - | ⏳ Placeholder | P0 |
| OpenVLA | 7B | GPTQ/AWQ | ⏳ Placeholder | P1 |
| Pi0 | 0.5B | - | ⏳ Placeholder | P1 |
| Qwen2.5-Omni | 7B | AWQ | ⏳ Placeholder | P2 |
| Qwen3-Omni | 7B | AWQ | ⏳ Placeholder | P2 |
| zipformer | <0.3B | - | ⏳ Placeholder | P1 |
| crossformer | <0.3B | - | ⏳ Placeholder | P1 |
| Llama-3.2-1B | 1B | - | ✅ Implemented | P2 |
| Llama-3.2-3B | 3B | - | ✅ Implemented | P2 |
| Deepseek-R1 | 1.5B | - | ✅ Implemented | P2 |
| Flux-1-schnell | - | - | ⏳ Placeholder | P2 |
| Stable Diffusion 3 | - | - | ⏳ Placeholder | P2 |

## Performance Targets

### Automotive
- Qwen2.5-VL 3B AWQ: < 100ms latency
- Qwen3-VL 4B AWQ: < 120ms latency
- CLIP 0.5B AWQ: < 40ms latency
- ASR models: RTF < 0.1 (real-time)

### Industrial
- SAM2 0.2B: > 20 FPS
- Qwen3-Instruct 4B AWQ: < 80ms latency
- Batch processing: Optimized for throughput

### Robotics
- OpenVLA 7B: < 500ms action prediction
- Pi0 0.5B: < 30ms policy inference
- Closed-loop: < 100ms total latency

### Healthcare
- SAM2: Accuracy > Speed
- Qwen3-Instruct: High reliability
- Latency: < 100ms acceptable

## Notes

- **P0 tests** are fully implemented and runnable
- **P1/P2 tests** with placeholders need model setup
- All tests include proper GPU cleanup
- Tests skip automatically if not on Strix GPU (gfx1150/1151)
- Quantized models (AWQ/GPTQ) are prioritized for edge deployment

## Contributing

When adding new tests:
1. Follow existing test structure
2. Add appropriate pytest markers
3. Include warmup iterations for performance tests
4. Add GPU memory cleanup
5. Document expected targets
6. Update this README

## Support

For issues or questions:
- Check test output and error messages
- Verify environment variables are set
- Ensure GPU is accessible (`torch.cuda.is_available()`)
- Review model availability on Hugging Face

