"""
Deepseek-R1 Language Model Tests for Strix

Tests for existing Deepseek-R1-Distill-Qwen-1.5B support.
Market segment: General AI Applications
"""

import pytest
import torch
import time
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


@pytest.mark.llm
@pytest.mark.p2
@pytest.mark.functional
def test_deepseek_r1_load(check_strix_gpu):
    """Test: Load Deepseek-R1-Distill-Qwen-1.5B model"""
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
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
        pytest.fail(f"Failed to load Deepseek-R1: {e}")


@pytest.mark.llm
@pytest.mark.p2
@pytest.mark.functional
def test_deepseek_r1_inference(check_strix_gpu):
    """Test: Run inference with Deepseek-R1"""
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    prompt = "Solve: 2 + 2 ="
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    
    # Test
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    assert len(generated_text) > 0, "No output generated"
    assert not torch.isnan(outputs.float()).any(), "NaN detected"
    
    print(f"\nDeepseek-R1 output: {generated_text[0]}")
    
    # Cleanup
    del model, tokenizer, inputs, outputs
    torch.cuda.empty_cache()


@pytest.mark.llm
@pytest.mark.p2
@pytest.mark.functional
def test_deepseek_r1_reasoning(check_strix_gpu):
    """Test: Deepseek-R1 reasoning capability"""
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Test reasoning with a simple math problem
    prompt = "If John has 3 apples and gives 1 to Mary, how many does he have left?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    assert len(generated_text) > 0, "No output generated"
    
    print(f"\nDeepseek-R1 reasoning output: {generated_text[0]}")
    
    # Cleanup
    del model, tokenizer, inputs, outputs
    torch.cuda.empty_cache()


@pytest.mark.llm
@pytest.mark.p2
@pytest.mark.performance
def test_deepseek_r1_tokens_per_second(check_strix_gpu):
    """Test: Measure Deepseek-R1 tokens per second"""
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    target_tokens_per_sec = 60  # Target > 60 tokens/sec for 1.5B model
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    prompt = "Calculate"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=20)
    
    # Measure
    num_tokens = 50
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=num_tokens)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tokens_per_sec = num_tokens / elapsed
    
    print(f"\n=== Deepseek-R1 Tokens/Second ===")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")
    print(f"Target: > {target_tokens_per_sec} tokens/sec")
    
    # Cleanup
    del model, tokenizer, inputs
    torch.cuda.empty_cache()

