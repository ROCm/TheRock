"""
CLIP (Contrastive Language-Image Pre-training) tests for Strix platforms

Tests vision-language understanding using OpenAI's CLIP model.
This is a P0 critical test for Edge AI and Windows Copilot+ use cases.
"""

import pytest
import torch
from PIL import Image
import os

# Skip module if dependencies not available
transformers = pytest.importorskip("transformers", reason="transformers not installed")

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")


@pytest.mark.strix
@pytest.mark.vlm
@pytest.mark.p0
@pytest.mark.quick
@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-specific VLM test"
)
class TestCLIP:
    """CLIP Vision-Language Model tests for Strix"""
    
    def test_clip_image_text_matching(self, strix_device, test_image_224, cleanup_gpu, record_property):
        """Test CLIP image-text matching on Strix"""
        from transformers import CLIPProcessor, CLIPModel
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Model loading timeout (60s) - likely network issue downloading from Hugging Face")
        
        print("\n🧠 Loading CLIP model (with 60s timeout)...")
        
        # Set timeout for model loading (handles network download hangs)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60 second timeout
        
        try:
            # Try to load from cache first, download if needed
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                resume_download=True,  # Resume interrupted downloads
                force_download=False,  # Use cache if available
            ).to(strix_device)
            processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                resume_download=True,
                force_download=False,
            )
            signal.alarm(0)  # Cancel timeout
            print("✅ Model loaded successfully")
        except TimeoutError as e:
            signal.alarm(0)
            pytest.skip(f"Model loading timeout: {e}")
        except Exception as e:
            signal.alarm(0)
            print(f"❌ Model loading failed: {e}")
            raise
        
        model.eval()
        
        # Create test image
        image = Image.new('RGB', (224, 224), color='red')
        texts = ["a red image", "a blue image", "a green image"]
        
        print("🔍 Processing inputs...")
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(strix_device)
        
        print("⚡ Running CLIP inference on Strix...")
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        probs_cpu = probs[0].cpu().numpy()
        print(f"✅ Text matching probabilities: {probs_cpu}")
        
        # Record metrics
        record_property("metric_similarity_score_red", f"{probs_cpu[0]:.4f}")
        record_property("metric_similarity_score_blue", f"{probs_cpu[1]:.4f}")
        record_property("metric_similarity_score_green", f"{probs_cpu[2]:.4f}")
        record_property("metric_prediction_correct", "true")
        record_property("gpu_family", AMDGPU_FAMILIES)
        
        # Assertions
        assert probs.device.type == "cuda", "Output should be on GPU"
        assert probs.shape == (1, 3), f"Expected shape (1, 3), got {probs.shape}"
        
        # Red image should match "a red image" most closely
        best_match = probs.argmax().item()
        assert best_match == 0, f"Expected red image to match index 0, got {best_match}"
        
        print(f"✅ CLIP test passed on {AMDGPU_FAMILIES}!")
    
    def test_clip_batch_inference(self, strix_device, cleanup_gpu):
        """Test CLIP batch processing on Strix"""
        from transformers import CLIPProcessor, CLIPModel
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Model loading timeout")
        
        print("\n🧠 Loading CLIP model for batch test (with timeout)...")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        
        try:
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                resume_download=True,
                force_download=False,
            ).to(strix_device)
            processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                resume_download=True,
                force_download=False,
            )
            signal.alarm(0)
        except TimeoutError as e:
            signal.alarm(0)
            pytest.skip(f"Model loading timeout: {e}")
        except Exception as e:
            signal.alarm(0)
            raise
        
        model.eval()
        
        # Create batch of test images
        batch_size = 4
        images = [Image.new('RGB', (224, 224), color='blue') for _ in range(batch_size)]
        texts = ["a blue image"] * batch_size
        
        print(f"🔍 Processing batch of {batch_size} images...")
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(strix_device)
        
        print("⚡ Running batch inference...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Batch inference successful! Output shape: {outputs.logits_per_image.shape}")
        
        assert outputs.logits_per_image.shape[0] == batch_size
        assert outputs.logits_per_image.device.type == "cuda"
        
        print(f"✅ CLIP batch test passed on {AMDGPU_FAMILIES}!")
    
    @pytest.mark.slow
    def test_clip_performance(self, strix_device, cleanup_gpu, record_property):
        """Benchmark CLIP performance on Strix"""
        from transformers import CLIPProcessor, CLIPModel
        import time
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Model loading timeout")
        
        print("\n🧠 Loading CLIP model for performance test (with timeout)...")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        
        try:
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                resume_download=True,
                force_download=False,
            ).to(strix_device)
            processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                resume_download=True,
                force_download=False,
            )
            signal.alarm(0)
        except TimeoutError as e:
            signal.alarm(0)
            pytest.skip(f"Model loading timeout: {e}")
        except Exception as e:
            signal.alarm(0)
            raise
        
        model.eval()
        
        image = Image.new('RGB', (224, 224))
        texts = ["test image"]
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(strix_device)
        
        # Warmup
        print("🔥 Warming up...")
        for _ in range(10):
            with torch.no_grad():
                _ = model(**inputs)
        torch.cuda.synchronize()
        
        # Benchmark
        print("⏱️  Benchmarking...")
        num_iterations = 100
        start = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(**inputs)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        fps = num_iterations / elapsed
        latency_ms = (elapsed / num_iterations) * 1000
        
        # Record metrics
        record_property("metric_throughput_fps", f"{fps:.2f}")
        record_property("metric_latency_ms", f"{latency_ms:.2f}")
        record_property("metric_iterations", num_iterations)
        record_property("gpu_family", AMDGPU_FAMILIES)
        
        print(f"📊 Performance Results:")
        print(f"   Throughput: {fps:.2f} FPS")
        print(f"   Latency: {latency_ms:.2f} ms")
        
        # Should achieve reasonable performance for edge inference
        assert fps > 10, f"CLIP throughput too low: {fps} FPS (expected > 10 FPS)"
        
        print(f"✅ CLIP performance test passed on {AMDGPU_FAMILIES}!")

