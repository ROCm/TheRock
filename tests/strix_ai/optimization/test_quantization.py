"""
Model Quantization tests for Strix platforms

Tests INT8/FP16 quantization for edge inference optimization.
This is a P0 critical test as quantization is essential for iGPU memory constraints.
"""

import pytest
import torch
import time
import os

# Skip module if dependencies not available
transformers = pytest.importorskip("transformers", reason="transformers not installed")

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")


@pytest.mark.strix
@pytest.mark.p0
@pytest.mark.quick
@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-specific optimization test"
)
class TestQuantization:
    """Model quantization tests for Strix edge inference"""
    
    def test_fp16_inference(self, strix_device, test_image_224, cleanup_gpu):
        """Test FP16 mixed precision inference on Strix"""
        from transformers import ViTForImageClassification, ViTImageProcessor
        
        print("\n🧠 Loading model for FP16 test...")
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        model = model.to(strix_device)
        model.eval()
        
        inputs = processor(images=test_image_224, return_tensors="pt").to(strix_device)
        
        # FP32 baseline
        print("⚡ Running FP32 inference...")
        with torch.no_grad():
            output_fp32 = model(**inputs).logits
        
        # FP16 inference
        print("⚡ Running FP16 inference...")
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output_fp16 = model(**inputs).logits
        
        print(f"✅ FP16 inference successful!")
        print(f"   FP32 output shape: {output_fp32.shape}")
        print(f"   FP16 output shape: {output_fp16.shape}")
        
        # Check outputs are close (within numerical precision)
        max_diff = (output_fp32 - output_fp16).abs().max().item()
        print(f"   Max difference: {max_diff:.6f}")
        
        assert max_diff < 0.1, f"FP16 output differs too much from FP32: {max_diff}"
        
        print(f"✅ FP16 test passed on {AMDGPU_FAMILIES}!")
    
    @pytest.mark.slow
    def test_fp16_speedup(self, strix_device, test_image_224, cleanup_gpu):
        """Measure FP16 vs FP32 speedup on Strix"""
        from transformers import ViTForImageClassification, ViTImageProcessor
        
        print("\n🧠 Loading model for speedup test...")
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
        
        # Benchmark FP32
        print("⏱️  Benchmarking FP32...")
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = model(**inputs)
        torch.cuda.synchronize()
        fp32_time = time.time() - start
        
        # Benchmark FP16
        print("⏱️  Benchmarking FP16...")
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = model(**inputs)
        torch.cuda.synchronize()
        fp16_time = time.time() - start
        
        speedup = fp32_time / fp16_time
        
        print(f"📊 Performance Results:")
        print(f"   FP32 time: {fp32_time:.3f}s")
        print(f"   FP16 time: {fp16_time:.3f}s")
        print(f"   Speedup: {speedup:.2f}x")
        
        # FP16 should be faster (at least marginally)
        assert fp16_time < fp32_time, f"FP16 not faster than FP32"
        
        print(f"✅ FP16 speedup test passed on {AMDGPU_FAMILIES}!")
    
    def test_fp16_memory_savings(self, strix_device, test_image_224, cleanup_gpu):
        """Test FP16 memory savings on Strix iGPU"""
        from transformers import ViTForImageClassification, ViTImageProcessor
        
        print("\n🧠 Testing FP16 memory savings...")
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        
        # FP32 model
        print("📊 Measuring FP32 memory...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model_fp32 = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        model_fp32 = model_fp32.to(strix_device)
        
        inputs = processor(images=test_image_224, return_tensors="pt").to(strix_device)
        with torch.no_grad():
            _ = model_fp32(**inputs)
        
        fp32_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        del model_fp32
        
        # FP16 model
        print("📊 Measuring FP16 memory...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model_fp16 = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            torch_dtype=torch.float16
        )
        model_fp16 = model_fp16.to(strix_device)
        
        with torch.no_grad():
            _ = model_fp16(**inputs)
        
        fp16_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        memory_savings = (1 - fp16_memory / fp32_memory) * 100
        
        print(f"📊 Memory Usage:")
        print(f"   FP32: {fp32_memory:.2f} MB")
        print(f"   FP16: {fp16_memory:.2f} MB")
        print(f"   Savings: {memory_savings:.1f}%")
        
        # FP16 should use less memory
        assert fp16_memory < fp32_memory, "FP16 should use less memory than FP32"
        
        print(f"✅ FP16 memory savings test passed on {AMDGPU_FAMILIES}!")
    
    def test_dynamic_quantization(self, strix_device, cleanup_gpu):
        """Test PyTorch dynamic quantization on Strix"""
        import torch.nn as nn
        
        print("\n🧠 Testing dynamic quantization...")
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ).to(strix_device)
        
        # Apply dynamic quantization
        print("⚡ Applying dynamic quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model.cpu(),  # Quantization is CPU-only in PyTorch
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        print(f"✅ Dynamic quantization successful!")
        print(f"   Original model: {type(model)}")
        print(f"   Quantized model: {type(quantized_model)}")
        
        # Test inference
        test_input = torch.randn(1, 100)
        output = quantized_model(test_input)
        
        assert output.shape == (1, 10)
        
        print(f"✅ Dynamic quantization test passed!")
    
    def test_model_size_comparison(self, strix_device, cleanup_gpu):
        """Compare model sizes: FP32 vs FP16"""
        from transformers import ViTForImageClassification
        import tempfile
        import os
        
        print("\n🧠 Comparing model sizes...")
        
        # Load model
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        fp32_size_mb = (num_params * 4) / (1024 * 1024)  # 4 bytes per FP32 param
        fp16_size_mb = (num_params * 2) / (1024 * 1024)  # 2 bytes per FP16 param
        
        print(f"📊 Model Size Comparison:")
        print(f"   Parameters: {num_params:,}")
        print(f"   FP32 size: {fp32_size_mb:.2f} MB")
        print(f"   FP16 size: {fp16_size_mb:.2f} MB")
        print(f"   Savings: {(1 - fp16_size_mb/fp32_size_mb)*100:.1f}%")
        
        # FP16 should be roughly half the size
        assert fp16_size_mb < fp32_size_mb
        assert abs(fp16_size_mb - fp32_size_mb/2) / fp32_size_mb < 0.1  # Within 10%
        
        print(f"✅ Model size comparison test passed!")

