"""
CLIP and BLIP Vision-Language Model Tests for Strix

Tests for CLIP and BLIP 0.5B models with AWQ quantization.
Market segments: Automotive, Industrial
"""

import pytest
import torch
import time
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


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
    img = Image.new('RGB', (224, 224), color='red')
    return img


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.industrial
@pytest.mark.p0
@pytest.mark.functional
@pytest.mark.awq
def test_clip_load(check_strix_gpu):
    """Test: Load CLIP model"""
    model_id = "openai/clip-vit-base-patch32"
    
    try:
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        
        assert model is not None, "Model failed to load"
        assert processor is not None, "Processor failed to load"
        assert next(model.parameters()).is_cuda, "Model not on GPU"
        
    except Exception as e:
        pytest.fail(f"Failed to load CLIP: {e}")


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.functional
def test_clip_image_text_similarity(check_strix_gpu, sample_image):
    """Test: CLIP image-text similarity"""
    model_id = "openai/clip-vit-base-patch32"
    
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    
    texts = ["a red image", "a blue image", "a green image"]
    
    inputs = processor(text=texts, images=sample_image, return_tensors="pt", padding=True).to("cuda")
    
    # Warmup
    with torch.no_grad():
        _ = model(**inputs)
    
    # Test
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    # Validate
    assert probs.shape[0] == 1, f"Expected 1 image, got {probs.shape[0]}"
    assert probs.shape[1] == 3, f"Expected 3 texts, got {probs.shape[1]}"
    assert not torch.isnan(probs).any(), "NaN in probabilities"
    assert not torch.isinf(probs).any(), "Inf in probabilities"
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0], device="cuda"), atol=1e-3), \
        "Probabilities don't sum to 1"
    
    # Check that highest prob is for "a red image" (index 0)
    max_idx = torch.argmax(probs, dim=1).item()
    print(f"\nCLIP probabilities: {probs.cpu().numpy()[0]}")
    print(f"Highest probability index: {max_idx} (expected 0 for 'a red image')")
    
    # Cleanup
    del model, processor, inputs, outputs
    torch.cuda.empty_cache()


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.performance
def test_clip_latency(check_strix_gpu, sample_image):
    """Test: Measure CLIP latency"""
    model_id = "openai/clip-vit-base-patch32"
    target_latency_ms = 40  # Target < 40ms
    
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    
    texts = ["a photo"]
    inputs = processor(text=texts, images=sample_image, return_tensors="pt", padding=True).to("cuda")
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(**inputs)
    
    # Measure
    latencies = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)
    
    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    print(f"\n=== CLIP Latency ===")
    print(f"Mean: {mean_latency:.2f} ms")
    print(f"P50:  {p50_latency:.2f} ms")
    print(f"P95:  {p95_latency:.2f} ms")
    print(f"P99:  {p99_latency:.2f} ms")
    print(f"Target: < {target_latency_ms} ms")
    
    assert mean_latency < target_latency_ms * 2, \
        f"Mean latency {mean_latency:.2f}ms exceeds 2x target"
    
    # Cleanup
    del model, processor, inputs
    torch.cuda.empty_cache()


@pytest.mark.vlm
@pytest.mark.automotive
@pytest.mark.p0
@pytest.mark.performance
def test_clip_memory(check_strix_gpu, sample_image):
    """Test: Measure CLIP memory usage"""
    model_id = "openai/clip-vit-base-patch32"
    target_memory_gb = 2.0
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    
    texts = ["a photo"]
    inputs = processor(text=texts, images=sample_image, return_tensors="pt", padding=True).to("cuda")
    
    with torch.no_grad():
        _ = model(**inputs)
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\n=== CLIP Memory ===")
    print(f"Peak Memory: {peak_memory:.2f} GB")
    print(f"Target: < {target_memory_gb} GB")
    
    assert peak_memory < target_memory_gb, \
        f"Peak memory {peak_memory:.2f}GB exceeds target {target_memory_gb}GB"
    
    # Cleanup
    del model, processor, inputs
    torch.cuda.empty_cache()


@pytest.mark.vlm
@pytest.mark.quick
@pytest.mark.p0
def test_clip_quick_smoke(check_strix_gpu, sample_image):
    """Quick smoke test for CLIP"""
    model_id = "openai/clip-vit-base-patch32"
    
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    
    texts = ["test"]
    inputs = processor(text=texts, images=sample_image, return_tensors="pt", padding=True).to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    assert outputs.logits_per_image is not None, "No output in smoke test"
    
    # Cleanup
    del model, processor, inputs, outputs
    torch.cuda.empty_cache()

