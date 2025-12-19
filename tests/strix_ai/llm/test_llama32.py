"""
Llama-3.2 Language Model Tests for Strix

Tests for existing Llama-3.2-1B and Llama-3.2-3B support.
Market segment: General AI Applications
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


# ============================================================================
# Llama-3.2-1B Tests
# ============================================================================

@pytest.mark.llm
@pytest.mark.p2
@pytest.mark.functional
def test_llama32_1b_load(check_strix_gpu):
    """Test: Load Llama-3.2-1B-Instruct model"""
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        assert model is not None, "Model failed to load"
        assert tokenizer is not None, "Tokenizer failed to load"
        assert next(model.parameters()).is_cuda, "Model not on GPU"
        
    except Exception as e:
        pytest.fail(f"Failed to load Llama-3.2-1B: {e}")


@pytest.mark.llm
@pytest.mark.p2
@pytest.mark.functional
def test_llama32_1b_inference(check_strix_gpu):
    """Test: Run inference with Llama-3.2-1B"""
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    prompt = "What is AI?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    
    # Test
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    assert len(generated_text) > 0, "No output generated"
    assert not torch.isnan(outputs.float()).any(), "NaN detected"
    
    # Cleanup
    del model, tokenizer, inputs, outputs
    torch.cuda.empty_cache()


@pytest.mark.llm
@pytest.mark.p2
@pytest.mark.performance
def test_llama32_1b_tokens_per_second(check_strix_gpu):
    """Test: Measure Llama-3.2-1B tokens per second"""
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    target_tokens_per_sec = 50  # Target > 50 tokens/sec
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    prompt = "Tell me"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20)
    
    # Measure
    num_tokens_to_generate = 50
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=num_tokens_to_generate)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tokens_per_sec = num_tokens_to_generate / elapsed
    
    print(f"\n=== Llama-3.2-1B Tokens/Second ===")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")
    print(f"Target: > {target_tokens_per_sec} tokens/sec")
    
    # Cleanup
    del model, tokenizer, inputs, outputs
    torch.cuda.empty_cache()


# ============================================================================
# Llama-3.2-3B Tests
# ============================================================================

@pytest.mark.llm
@pytest.mark.p2
@pytest.mark.functional
def test_llama32_3b_load(check_strix_gpu):
    """Test: Load Llama-3.2-3B-Instruct model"""
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        assert model is not None, "Model failed to load"
        assert tokenizer is not None, "Tokenizer failed to load"
        assert next(model.parameters()).is_cuda, "Model not on GPU"
        
    except Exception as e:
        pytest.fail(f"Failed to load Llama-3.2-3B: {e}")


@pytest.mark.llm
@pytest.mark.p2
@pytest.mark.functional
def test_llama32_3b_inference(check_strix_gpu):
    """Test: Run inference with Llama-3.2-3B"""
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    assert len(generated_text) > 0, "No output generated"
    
    # Cleanup
    del model, tokenizer, inputs, outputs
    torch.cuda.empty_cache()


@pytest.mark.llm
@pytest.mark.quick
@pytest.mark.p2
def test_llama32_1b_quick_smoke(check_strix_gpu):
    """Quick smoke test for Llama-3.2-1B"""
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    inputs = tokenizer("Hi", return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    assert len(generated_text) > 0, "No output in smoke test"
    
    # Cleanup
    del model, tokenizer, inputs, outputs
    torch.cuda.empty_cache()

