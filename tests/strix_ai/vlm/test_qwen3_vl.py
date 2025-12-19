"""
Qwen3-VL-Instruct Vision-Language Model Tests for Strix

Tests for Qwen3-VL-Instruct 4B model with AWQ quantization.
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
    img = Image.new('RGB', (224, 224), color='blue')
    return img


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.industrial
@pytest.mark.robotics
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_qwen3_vl_4b_awq_load(check_strix_gpu):
    """Test: Load Qwen3-VL-Instruct 4B with AWQ quantization"""
    model_id = "Qwen/Qwen3-VL-Instruct-4B-AWQ"
    
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
        pytest.fail(f"Failed to load Qwen3-VL-Instruct 4B AWQ: {e}")


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_qwen3_vl_4b_awq_inference(check_strix_gpu, sample_image):
    """Test: Run inference with Qwen3-VL-Instruct 4B AWQ"""
    model_id = "Qwen/Qwen3-VL-Instruct-4B-AWQ"
    
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
                {"type": "text", "text": "What color is this image?"}
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
    
    # Validate
    assert len(generated_text) > 0, "No output generated"
    assert isinstance(generated_text[0], str), "Output is not a string"
    assert not torch.isnan(outputs.float()).any(), "NaN detected"
    assert not torch.isinf(outputs.float()).any(), "Inf detected"
    
    # Cleanup
    del model, processor, inputs, outputs
    torch.cuda.empty_cache()


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.performance
@pytest.mark.awq
def test_qwen3_vl_4b_awq_latency(check_strix_gpu, sample_image):
    """Test: Measure latency for Qwen3-VL-Instruct 4B AWQ"""
    model_id = "Qwen/Qwen3-VL-Instruct-4B-AWQ"
    target_latency_ms = 120  # Target < 120ms for automotive
    
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
                {"type": "text", "text": "Analyze this."}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[sample_image], return_tensors="pt").to("cuda")
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20)
    
    # Measure
    latencies = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20)
        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)
    
    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"\n=== Qwen3-VL-Instruct 4B AWQ Latency ===")
    print(f"Mean: {mean_latency:.2f} ms")
    print(f"P95:  {p95_latency:.2f} ms")
    print(f"Target: < {target_latency_ms} ms")
    
    assert mean_latency < target_latency_ms * 1.5, \
        f"Mean latency {mean_latency:.2f}ms exceeds target"
    
    # Cleanup
    del model, processor, inputs
    torch.cuda.empty_cache()


@pytest.mark.vlm
@pytest.mark.quick
@pytest.mark.p0
@pytest.mark.awq
def test_qwen3_vl_4b_quick_smoke(check_strix_gpu, sample_image):
    """Quick smoke test for Qwen3-VL-Instruct 4B AWQ"""
    model_id = "Qwen/Qwen3-VL-Instruct-4B-AWQ"
    
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
                {"type": "text", "text": "Hello"}
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

