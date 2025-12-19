"""
Qwen3-Instruct Model Tests for Strix

Tests for Qwen3-Instruct 4B model with AWQ quantization.
Market segments: Industrial, Robotics, Healthcare
"""

import pytest
import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


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


@pytest.mark.instruct
@pytest.mark.industrial
@pytest.mark.robotics
@pytest.mark.healthcare
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_qwen3_instruct_4b_awq_load(check_strix_gpu):
    """Test: Load Qwen3-Instruct 4B with AWQ quantization"""
    model_id = "Qwen/Qwen3-Instruct-4B-AWQ"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        assert model is not None, "Model failed to load"
        assert tokenizer is not None, "Tokenizer failed to load"
        assert next(model.parameters()).is_cuda, "Model not on GPU"
        
    except Exception as e:
        pytest.fail(f"Failed to load Qwen3-Instruct 4B AWQ: {e}")


@pytest.mark.instruct
@pytest.mark.industrial
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_qwen3_instruct_4b_awq_inference(check_strix_gpu):
    """Test: Run inference with Qwen3-Instruct 4B AWQ"""
    model_id = "Qwen/Qwen3-Instruct-4B-AWQ"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare input
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    
    # Test inference
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Validate
    assert len(generated_text) > 0, "No output generated"
    assert isinstance(generated_text[0], str), "Output is not a string"
    assert not torch.isnan(outputs.float()).any(), "NaN detected"
    assert not torch.isinf(outputs.float()).any(), "Inf detected"
    
    print(f"\nGenerated: {generated_text[0]}")
    
    # Cleanup
    del model, tokenizer, inputs, outputs
    torch.cuda.empty_cache()


@pytest.mark.instruct
@pytest.mark.industrial
@pytest.mark.p0
@pytest.mark.performance
@pytest.mark.awq
def test_qwen3_instruct_4b_awq_latency(check_strix_gpu):
    """Test: Measure latency for Qwen3-Instruct 4B AWQ"""
    model_id = "Qwen/Qwen3-Instruct-4B-AWQ"
    target_latency_ms = 80  # Target < 80ms
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    messages = [
        {"role": "user", "content": "Hi"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)
    
    # Measure
    latencies = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)
        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)
    
    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"\n=== Qwen3-Instruct 4B AWQ Latency ===")
    print(f"Mean: {mean_latency:.2f} ms")
    print(f"P95:  {p95_latency:.2f} ms")
    print(f"Target: < {target_latency_ms} ms")
    
    assert mean_latency < target_latency_ms * 1.5, \
        f"Mean latency {mean_latency:.2f}ms exceeds target"
    
    # Cleanup
    del model, tokenizer, inputs
    torch.cuda.empty_cache()


@pytest.mark.instruct
@pytest.mark.quick
@pytest.mark.p0
@pytest.mark.awq
def test_qwen3_instruct_4b_quick_smoke(check_strix_gpu):
    """Quick smoke test for Qwen3-Instruct 4B AWQ"""
    model_id = "Qwen/Qwen3-Instruct-4B-AWQ"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    messages = [{"role": "user", "content": "Hi"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    assert len(generated_text) > 0, "No output in smoke test"
    
    # Cleanup
    del model, tokenizer, inputs, outputs
    torch.cuda.empty_cache()

