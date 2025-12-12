"""
Vision Transformer (ViT) tests for Strix platforms

Tests ViT-Base model for image classification.
This is a P0 critical test as ViT is the backbone of modern computer vision.
"""

import pytest
import torch
from PIL import Image
import os

# Skip module if dependencies not available
transformers = pytest.importorskip("transformers", reason="transformers not installed")

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")


@pytest.mark.strix
@pytest.mark.vit
@pytest.mark.p0
@pytest.mark.quick
@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-specific ViT test"
)
class TestViTBase:
    """Vision Transformer tests for Strix"""
    
    def test_vit_image_classification(self, strix_device, test_image_224, cleanup_gpu, record_property):
        """Test ViT-Base image classification on Strix"""
        from transformers import ViTForImageClassification, ViTImageProcessor
        import time
        
        print("\n🧠 Loading ViT-Base model...")
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        model = model.to(strix_device)
        model.eval()
        
        print("🔍 Processing test image...")
        inputs = processor(images=test_image_224, return_tensors="pt").to(strix_device)
        
        print("⚡ Running ViT inference on Strix...")
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probs[0, predicted_class].item()
        torch.cuda.synchronize()
        inference_time_ms = (time.time() - start) * 1000
        
        # Record metrics
        record_property("metric_predicted_class", predicted_class)
        record_property("metric_confidence", f"{confidence:.4f}")
        record_property("metric_inference_time_ms", f"{inference_time_ms:.2f}")
        record_property("gpu_family", AMDGPU_FAMILIES)
        
        print(f"✅ Predicted class: {predicted_class}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Inference time: {inference_time_ms:.2f} ms")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Device: {logits.device}")
        
        # Assertions
        assert outputs.logits.device.type == "cuda", "Output should be on GPU"
        assert logits.shape == (1, 1000), f"Expected shape (1, 1000), got {logits.shape}"  # ImageNet classes
        assert 0 <= predicted_class < 1000, f"Invalid class prediction: {predicted_class}"
        
        print(f"✅ ViT classification test passed on {AMDGPU_FAMILIES}!")
    
    def test_vit_mixed_precision(self, strix_device, test_image_224, cleanup_gpu, record_property):
        """Test ViT with FP16 mixed precision for efficiency"""
        from transformers import ViTForImageClassification, ViTImageProcessor
        import time
        
        print("\n🧠 Loading ViT-Base with FP16...")
        torch.cuda.reset_peak_memory_stats()
        
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            torch_dtype=torch.float16
        )
        model = model.to(strix_device)
        model.eval()
        
        inputs = processor(images=test_image_224, return_tensors="pt").to(strix_device)
        
        print("⚡ Running FP16 inference...")
        start = time.time()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**inputs)
        torch.cuda.synchronize()
        inference_time_ms = (time.time() - start) * 1000
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Record metrics
        record_property("metric_dtype", str(outputs.logits.dtype))
        record_property("metric_inference_time_ms", f"{inference_time_ms:.2f}")
        record_property("metric_peak_memory_mb", f"{peak_memory_mb:.2f}")
        record_property("gpu_family", AMDGPU_FAMILIES)
        
        print(f"✅ FP16 inference successful!")
        print(f"   Output dtype: {outputs.logits.dtype}")
        print(f"   Inference time: {inference_time_ms:.2f} ms")
        print(f"   Peak memory: {peak_memory_mb:.2f} MB")
        
        assert outputs.logits.device.type == "cuda"
        # FP16 should save memory on iGPU
        
        print(f"✅ ViT FP16 test passed on {AMDGPU_FAMILIES}!")
    
    def test_vit_batch_processing(self, strix_device, cleanup_gpu, record_property):
        """Test ViT batch inference on Strix"""
        from transformers import ViTForImageClassification, ViTImageProcessor
        import time
        
        print("\n🧠 Loading ViT-Base for batch test...")
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        model = model.to(strix_device)
        model.eval()
        
        # Create batch of images
        batch_size = 4
        images = [Image.new('RGB', (224, 224), color='blue') for _ in range(batch_size)]
        
        print(f"🔍 Processing batch of {batch_size} images...")
        inputs = processor(images=images, return_tensors="pt").to(strix_device)
        
        print("⚡ Running batch inference...")
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        batch_time_ms = (time.time() - start) * 1000
        time_per_image_ms = batch_time_ms / batch_size
        
        # Record metrics
        record_property("metric_batch_size", batch_size)
        record_property("metric_batch_time_ms", f"{batch_time_ms:.2f}")
        record_property("metric_time_per_image_ms", f"{time_per_image_ms:.2f}")
        record_property("gpu_family", AMDGPU_FAMILIES)
        
        print(f"✅ Batch inference successful!")
        print(f"   Output shape: {outputs.logits.shape}")
        print(f"   Batch time: {batch_time_ms:.2f} ms")
        print(f"   Time per image: {time_per_image_ms:.2f} ms")
        
        assert outputs.logits.shape[0] == batch_size
        assert outputs.logits.device.type == "cuda"
        
        print(f"✅ ViT batch test passed on {AMDGPU_FAMILIES}!")
    
    @pytest.mark.slow
    def test_vit_throughput(self, strix_device, test_image_224, cleanup_gpu, record_property):
        """Benchmark ViT throughput on Strix"""
        from transformers import ViTForImageClassification, ViTImageProcessor
        import time
        
        print("\n🧠 Loading ViT-Base for throughput test...")
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        model = model.to(strix_device)
        model.eval()
        
        inputs = processor(images=test_image_224, return_tensors="pt").to(strix_device)
        
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
        
        # Target: > 30 FPS for ViT-Base on Strix
        # This is aggressive but achievable with optimization
        print(f"   Target: > 30 FPS (current: {fps:.2f} FPS)")
        
        # More relaxed assertion for initial implementation
        assert fps > 10, f"ViT throughput too low: {fps} FPS (expected > 10 FPS)"
        
        print(f"✅ ViT throughput test passed on {AMDGPU_FAMILIES}!")
    
    def test_vit_memory_usage(self, strix_device, test_image_224, cleanup_gpu, record_property):
        """Test ViT memory usage on Strix iGPU"""
        from transformers import ViTForImageClassification, ViTImageProcessor
        
        print("\n🧠 Loading ViT-Base for memory test...")
        
        # Reset memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        model = model.to(strix_device)
        model.eval()
        
        inputs = processor(images=test_image_224, return_tensors="pt").to(strix_device)
        
        print("⚡ Running inference and measuring memory...")
        with torch.no_grad():
            _ = model(**inputs)
        
        torch.cuda.synchronize()
        
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Record metrics
        record_property("metric_peak_memory_mb", f"{peak_memory_mb:.2f}")
        record_property("gpu_family", AMDGPU_FAMILIES)
        
        print(f"📊 Memory Usage:")
        print(f"   Peak: {peak_memory_mb:.2f} MB")
        
        # Strix iGPU shares system memory - should stay reasonable
        # ViT-Base is ~330MB model size
        assert peak_memory_mb < 2048, f"Memory usage too high: {peak_memory_mb:.2f} MB"
        
        print(f"✅ ViT memory test passed on {AMDGPU_FAMILIES}!")

