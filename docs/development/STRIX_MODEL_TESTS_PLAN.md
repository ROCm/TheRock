# Strix Edge AI Model Tests Plan

## Overview

Test plan for representative AI models targeting Automotive, Industrial, Robotics, and Healthcare market segments on Strix platforms (gfx1151/gfx1150).

---

## Representative Models Summary

| Model | Market Segment | Size | Quantization | Strix Compatibility |
|-------|---------------|------|--------------|---------------------|
| Qwen3-Instruct | Industrial, Robotics, Healthcare | 4B | AWQ | ✅ Both platforms |
| Qwen2.5-VL-Instruct | Automotive, Industrial, Robotics | 3B, 7B | AWQ | ✅ 3B both, 7B Linux |
| Qwen3-VL-Instruct | Automotive, Industrial, Robotics | 4B | AWQ | ✅ Both platforms |
| Qwen2.5-Omni | Automotive | 7B | AWQ | ⚠️ Linux only |
| Qwen3-Omni | Automotive | 7B | AWQ | ⚠️ Linux only |
| SAM2 | Industrial, Healthcare | 0.2B | - | ✅ Both platforms |
| CLIP/BLIP | Automotive, Industrial | 0.5B | AWQ | ✅ Both platforms |
| OpenVLA | Robotics | 7B | GPTQ/AWQ | ⚠️ Linux only |
| CogACT | Robotics | 7B | - | ⚠️ Linux only |
| Pi0 | Robotics | 0.5B | - | ✅ Both platforms |
| zipformer | Automotive, Robotics, Industrial | <0.3B | - | ✅ Both platforms |
| crossformer | Automotive, Robotics, Industrial | <0.3B | - | ✅ Both platforms |

---

## Test Categories

| Category | Purpose | Models Tested | Market Segments |
|----------|---------|---------------|-----------------|
| **VLM** (Vision Language Models) | Text-image understanding, VQA | Qwen2.5-VL (3B/7B), Qwen3-VL (4B), CLIP (0.5B), BLIP (0.5B) | Automotive, Industrial, Robotics |
| **VLA** (Vision Language Action) | Robot action prediction | OpenVLA (7B), CogACT (7B), Pi0 (0.5B) | Robotics |
| **Multimodal** | Vision + Audio + Text | Qwen2.5-Omni (7B), Qwen3-Omni (7B) | Automotive |
| **Segmentation** | Image segmentation | SAM2 (0.2B) | Industrial, Healthcare |
| **LLM** (Language Models) | Text generation | Qwen3-Instruct (4B) | Industrial, Robotics, Healthcare |
| **ASR** (Speech Recognition) | Speech-to-text | zipformer (<0.3B), crossformer (<0.3B) | Automotive, Robotics, Industrial |
| **Quantization** | AWQ/GPTQ validation | All AWQ/GPTQ models | All (memory efficiency) |
| **Strix-Specific** | iGPU optimization | All models | Edge AI deployment |

---

## Test Implementation Plan

### Test Directory Structure

```
tests/strix_ai/edge_models/
├── vlm/
│   ├── test_qwen_vl_3b.py
│   ├── test_qwen_vl_4b.py
│   ├── test_qwen_vl_7b.py
│   └── test_clip_blip.py
├── vla/
│   ├── test_openvla.py
│   ├── test_cogact.py
│   └── test_pi0.py
├── multimodal/
│   ├── test_qwen_omni_2_5.py
│   └── test_qwen_omni_3.py
├── segmentation/
│   └── test_sam2.py
├── llm/
│   └── test_qwen_instruct.py
├── asr/
│   ├── test_zipformer.py
│   ├── test_crossformer.py
│   └── test_asr_comparison.py
├── quantization/
│   ├── test_awq_quantization.py
│   ├── test_gptq_quantization.py
│   └── test_quantization_accuracy.py
└── strix_specific/
    ├── test_strix_memory_constraints.py
    ├── test_strix_unified_memory.py
    ├── test_strix_batch_sizes.py
    └── test_strix_concurrent.py
```

---

## Phase 1: High Priority (Weeks 1-2)

### Tests to Implement

1. **test_qwen_vl_3b.py** - VLM 3B model
2. **test_clip_blip.py** - Expand existing CLIP test
3. **test_sam2.py** - Segmentation for industrial/healthcare
4. **test_zipformer.py** - ASR for automotive

**Why First:**
- Small models (<5B) - fit on both platforms
- Critical for multiple market segments
- Low memory requirements
- Real-time performance targets achievable

---

## Phase 2: Medium Priority (Weeks 3-4)

### Tests to Implement

5. **test_qwen_instruct.py** - Language understanding
6. **test_pi0.py** - Small robotics policy
7. **test_crossformer.py** - Second ASR model
8. **test_awq_quantization.py** - Quantization validation

**Why Next:**
- Foundation for larger models
- Validates quantization strategy
- Covers remaining small models

---

## Phase 3: Advanced (Month 2)

### Tests to Implement

9. **test_qwen_vl_7b.py** - Large VLM (Linux only)
10. **test_qwen_omni.py** - Multimodal models
11. **test_openvla.py** - VLA for robotics
12. **test_strix_specific.py** - Platform optimizations

**Why Last:**
- 7B models - memory intensive
- Linux-only due to memory constraints
- Requires optimization knowledge from Phases 1-2

---

## Test Template Example

### File: `tests/strix_ai/edge_models/vlm/test_qwen_vl_3b.py`

```python
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import time

class TestQwenVL3B:
    
    @pytest.fixture(scope="class")
    def qwen_vl_model(self, strix_device):
        """Load Qwen2.5-VL-Instruct-3B-AWQ"""
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-VL-Instruct-3B-AWQ",
            device_map="cuda",
            torch_dtype=torch.float16
        )
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-Instruct-3B-AWQ"
        )
        return model, processor
    
    def test_inference_latency(self, qwen_vl_model, test_image_224):
        """Test VLM inference latency on Strix"""
        model, processor = qwen_vl_model
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": test_image_224},
                {"type": "text", "text": "Describe this image."}
            ]
        }]
        
        inputs = processor(messages, return_tensors="pt").to("cuda")
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        assert latency < 150, f"Latency {latency:.2f}ms exceeds 150ms target"
        assert len(response) > 0, "Empty response"
        
        print(f"✅ Qwen-VL-3B inference: {latency:.2f}ms")
    
    def test_automotive_scene(self, qwen_vl_model):
        """Test automotive scene understanding"""
        # Use case: Road scene analysis
        # Validate: Vehicle detection, traffic sign recognition
        pass
    
    def test_industrial_inspection(self, qwen_vl_model):
        """Test industrial quality inspection"""
        # Use case: Defect detection
        # Validate: Anomaly identification
        pass
```

---

## Test Registration

### File: `build_tools/github_actions/fetch_test_configurations.py`

```python
test_matrix = {
    # VLM Tests
    "qwen_vl": {
        "job_name": "qwen_vl",
        "fetch_artifact_args": "--tests",
        "timeout_minutes": 30,
        "test_script": "pytest tests/strix_ai/edge_models/vlm/ -v",
        "platform": ["linux", "windows"],
        "total_shards": 2,
    },
    
    # VLA Tests (Robotics)
    "vla_robotics": {
        "job_name": "vla_robotics",
        "fetch_artifact_args": "--tests",
        "timeout_minutes": 40,
        "test_script": "pytest tests/strix_ai/edge_models/vla/ -v",
        "platform": ["linux"],
        "total_shards": 1,
        "exclude_family": {
            "windows": ["gfx1151"]  # 7B models
        }
    },
    
    # Multimodal Tests
    "multimodal_omni": {
        "job_name": "multimodal_omni",
        "fetch_artifact_args": "--tests",
        "timeout_minutes": 35,
        "test_script": "pytest tests/strix_ai/edge_models/multimodal/ -v",
        "platform": ["linux"],
        "total_shards": 1,
    },
    
    # Segmentation Tests
    "sam2_segmentation": {
        "job_name": "sam2_segmentation",
        "fetch_artifact_args": "--tests",
        "timeout_minutes": 20,
        "test_script": "pytest tests/strix_ai/edge_models/segmentation/ -v",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    
    # LLM Tests
    "llm_qwen_instruct": {
        "job_name": "llm_qwen_instruct",
        "fetch_artifact_args": "--tests",
        "timeout_minutes": 25,
        "test_script": "pytest tests/strix_ai/edge_models/llm/ -v",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    
    # ASR Tests
    "asr_models": {
        "job_name": "asr_models",
        "fetch_artifact_args": "--tests",
        "timeout_minutes": 15,
        "test_script": "pytest tests/strix_ai/edge_models/asr/ -v",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
    
    # Quantization Tests
    "quantization_validation": {
        "job_name": "quantization_validation",
        "fetch_artifact_args": "--tests",
        "timeout_minutes": 45,
        "test_script": "pytest tests/strix_ai/edge_models/quantization/ -v",
        "platform": ["linux", "windows"],
        "total_shards": 2,
    },
    
    # Strix-Specific Tests
    "strix_specific": {
        "job_name": "strix_specific",
        "fetch_artifact_args": "--tests",
        "timeout_minutes": 30,
        "test_script": "pytest tests/strix_ai/edge_models/strix_specific/ -v",
        "platform": ["linux", "windows"],
        "total_shards": 1,
    },
}
```

---

## Metrics to Collect

### Performance Metrics
- **Latency:** P50, P95, P99, Max (ms)
- **Throughput:** FPS, Inferences/sec, Tokens/sec
- **Bandwidth:** Memory bandwidth (GB/s)
- **Real-time Factor:** For ASR (<0.3 = faster than real-time)

### Resource Metrics
- **GPU Utilization:** % active
- **Memory Usage:** MB (critical for 7B models)
- **Power:** Watts (Strix power budget)
- **Temperature:** °C

### Quality Metrics
- **Accuracy:** Model output correctness
- **Quantization Loss:** AWQ/GPTQ vs FP16
- **Task-Specific:** IoU (segmentation), WER (ASR), mAP (detection)

---

## Validation Criteria

### Small Models (<1B)
- Latency: < 50ms
- Throughput: > 30 FPS
- Memory: < 2GB
- Both platforms: ✅

### Medium Models (3-5B)
- Latency: < 200ms
- Throughput: > 10 FPS
- Memory: < 8GB
- Both platforms: ✅

### Large Models (7B)
- Latency: < 300ms
- Throughput: > 5 FPS
- Memory: < 14GB
- Linux only: ⚠️

### Quantization
- Accuracy loss: < 2% vs FP16
- Memory reduction: > 70%
- Latency improvement: > 30%

---

## Summary

**Total Tests:** 21 test suites  
**Total Models:** 11 models (12 variants)  
**Market Segments:** 4 (Automotive, Industrial, Robotics, Healthcare)  
**Categories:** 8 test categories  
**Implementation:** 3 phases over 2 months  

**Priority:**
1. Small models (<1B) - Both platforms
2. Medium models (3-5B) - Both platforms
3. Large models (7B) - Linux only

**Key Focus:**
- Edge AI workloads
- Quantization efficiency (AWQ/GPTQ)
- Strix memory optimization
- Real-time performance targets
- Multi-market segment coverage

