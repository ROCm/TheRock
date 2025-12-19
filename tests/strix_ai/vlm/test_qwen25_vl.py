"""
Qwen2.5-VL-Instruct Vision-Language Model Tests for Strix

Tests for Qwen2.5-VL-Instruct 3B and 7B models with AWQ quantization.
Market segments: Automotive, Industrial, Robotics
"""

import pytest
import torch
import time
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


@pytest.fixture(scope="module")
def check_strix_gpu():
    """Verify Strix GPU is available"""
    import os
    amdgpu_family = os.getenv("AMDGPU_FAMILIES", "")
    if "gfx115" not in amdgpu_family:
        pytest.skip(f"Strix GPU not detected. AMDGPU_FAMILIES={amdgpu_family}")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    
    return torch.cuda.get_device_name(0)


@pytest.fixture(scope="module")
def sample_image():
    """Create a sample test image"""
    # Create a 224x224 RGB image with simple pattern
    img = Image.new('RGB', (224, 224), color='white')
    return img


# ============================================================================
# Qwen2.5-VL-Instruct 3B Tests
# ============================================================================

@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.industrial
@pytest.mark.robotics
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_qwen25_vl_3b_awq_load(check_strix_gpu):
    """Test: Load Qwen2.5-VL-Instruct 3B with AWQ quantization"""
    model_id = "Qwen/Qwen2.5-VL-Instruct-3B-AWQ"
    
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        assert model is not None, "Model failed to load"
        assert processor is not None, "Processor failed to load"
        
        # Verify model is on GPU
        assert next(model.parameters()).is_cuda, "Model not on GPU"
        
    except Exception as e:
        pytest.fail(f"Failed to load Qwen2.5-VL-Instruct 3B AWQ: {e}")


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_qwen25_vl_3b_awq_inference(check_strix_gpu, sample_image):
    """Test: Run inference with Qwen2.5-VL-Instruct 3B AWQ"""
    model_id = "Qwen/Qwen2.5-VL-Instruct-3B-AWQ"
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample_image},
                {"type": "text", "text": "Describe this image."}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[sample_image], return_tensors="pt").to("cuda")
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=20)
    
    # Test inference
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
    
    # Validate output
    assert len(generated_text) > 0, "No output generated"
    assert isinstance(generated_text[0], str), "Output is not a string"
    assert len(generated_text[0]) > 0, "Empty output generated"
    
    # Check for NaN in output token IDs
    assert not torch.isnan(outputs.float()).any(), "NaN detected in output"
    assert not torch.isinf(outputs.float()).any(), "Inf detected in output"
    
    # Cleanup
    del model, processor, inputs, outputs
    torch.cuda.empty_cache()


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.performance
@pytest.mark.awq
def test_qwen25_vl_3b_awq_latency(check_strix_gpu, sample_image):
    """Test: Measure latency for Qwen2.5-VL-Instruct 3B AWQ"""
    model_id = "Qwen/Qwen2.5-VL-Instruct-3B-AWQ"
    target_latency_ms = 100  # Target < 100ms for automotive
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample_image},
                {"type": "text", "text": "What is in this image?"}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[sample_image], return_tensors="pt").to("cuda")
    
    # Warmup (5 iterations)
    for _ in range(5):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20)
    
    # Measure latency (20 iterations)
    latencies = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20)
        
        torch.cuda.synchronize()
        end = time.time()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    # Calculate statistics
    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    print(f"\n=== Qwen2.5-VL-Instruct 3B AWQ Latency ===")
    print(f"Mean: {mean_latency:.2f} ms")
    print(f"P50:  {p50_latency:.2f} ms")
    print(f"P95:  {p95_latency:.2f} ms")
    print(f"P99:  {p99_latency:.2f} ms")
    print(f"Target: < {target_latency_ms} ms")
    
    # Check against target
    assert mean_latency < target_latency_ms * 1.5, \
        f"Mean latency {mean_latency:.2f}ms exceeds 1.5x target ({target_latency_ms}ms)"
    
    # Cleanup
    del model, processor, inputs
    torch.cuda.empty_cache()


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.performance
@pytest.mark.awq
def test_qwen25_vl_3b_awq_memory(check_strix_gpu, sample_image):
    """Test: Measure memory usage for Qwen2.5-VL-Instruct 3B AWQ"""
    model_id = "Qwen/Qwen2.5-VL-Instruct-3B-AWQ"
    target_memory_gb = 4.0  # Target < 4GB for Strix
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample_image},
                {"type": "text", "text": "Describe this image."}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[sample_image], return_tensors="pt").to("cuda")
    
    # Run inference
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)
    
    # Measure memory
    peak_memory = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB
    allocated_memory = torch.cuda.memory_allocated() / 1e9
    
    print(f"\n=== Qwen2.5-VL-Instruct 3B AWQ Memory ===")
    print(f"Peak Memory: {peak_memory:.2f} GB")
    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Target: < {target_memory_gb} GB")
    
    assert peak_memory < target_memory_gb, \
        f"Peak memory {peak_memory:.2f}GB exceeds target {target_memory_gb}GB"
    
    # Cleanup
    del model, processor, inputs
    torch.cuda.empty_cache()


# ============================================================================
# Qwen2.5-VL-Instruct 7B Tests
# ============================================================================

@pytest.mark.vlm
@pytest.mark.industrial
@pytest.mark.robotics
@pytest.mark.p1
@pytest.mark.functional
@pytest.mark.awq
def test_qwen25_vl_7b_awq_load(check_strix_gpu):
    """Test: Load Qwen2.5-VL-Instruct 7B with AWQ quantization"""
    model_id = "Qwen/Qwen2.5-VL-Instruct-7B-AWQ"
    
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        assert model is not None, "Model failed to load"
        assert processor is not None, "Processor failed to load"
        assert next(model.parameters()).is_cuda, "Model not on GPU"
        
    except Exception as e:
        pytest.fail(f"Failed to load Qwen2.5-VL-Instruct 7B AWQ: {e}")


@pytest.mark.vlm
@pytest.mark.industrial
@pytest.mark.p1
@pytest.mark.performance
@pytest.mark.awq
def test_qwen25_vl_7b_awq_latency(check_strix_gpu, sample_image):
    """Test: Measure latency for Qwen2.5-VL-Instruct 7B AWQ"""
    model_id = "Qwen/Qwen2.5-VL-Instruct-7B-AWQ"
    target_latency_ms = 250  # Target < 250ms for industrial
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample_image},
                {"type": "text", "text": "What is this?"}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[sample_image], return_tensors="pt").to("cuda")
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20)
    
    # Measure
    latencies = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20)
        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)
    
    mean_latency = np.mean(latencies)
    print(f"\n=== Qwen2.5-VL-Instruct 7B AWQ Latency ===")
    print(f"Mean: {mean_latency:.2f} ms (Target: < {target_latency_ms} ms)")
    
    assert mean_latency < target_latency_ms * 1.5, \
        f"Mean latency {mean_latency:.2f}ms exceeds 1.5x target"
    
    # Cleanup
    del model, processor, inputs
    torch.cuda.empty_cache()


# ============================================================================
# Quick Smoke Tests
# ============================================================================

@pytest.mark.vlm
@pytest.mark.quick
@pytest.mark.p0
@pytest.mark.awq
def test_qwen25_vl_3b_quick_smoke(check_strix_gpu, sample_image):
    """Quick smoke test for Qwen2.5-VL-Instruct 3B AWQ"""
    model_id = "Qwen/Qwen2.5-VL-Instruct-3B-AWQ"
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample_image},
                {"type": "text", "text": "Hi"}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[sample_image], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
    
    assert len(generated_text) > 0, "No output in smoke test"
    
    # Cleanup
    del model, processor, inputs, outputs
    torch.cuda.empty_cache()

