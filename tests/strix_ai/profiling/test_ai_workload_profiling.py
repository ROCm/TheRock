"""
ROCProfiler tests for AI/ML workloads on Strix
Profile real-world AI models: CLIP, ViT, YOLO on Strix GPUs
"""

import pytest
import os
import subprocess
import tempfile
from pathlib import Path

# Optional imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from transformers import CLIPModel, CLIPProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from transformers import ViTForImageClassification, ViTImageProcessor
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def check_rocprof_available():
    """Check if rocprof is available"""
    try:
        result = subprocess.run(
            ["rocprof", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.vlm
@pytest.mark.p1
class TestVLMProfiling:
    """Profile Vision Language Models on Strix"""
    
    def test_clip_inference_profile(self, strix_device, test_image_224, cleanup_gpu):
        """Profile CLIP model inference"""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            pytest.skip("PyTorch or Transformers not available")
        
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        print("\n=== Profiling CLIP Inference on Strix ===")
        
        # Load CLIP model
        model_name = "openai/clip-vit-base-patch32"
        print(f"Loading model: {model_name}")
        
        model = CLIPModel.from_pretrained(model_name).to(strix_device)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()
        
        # Prepare inputs
        text = ["a photo of a cat", "a photo of a dog"]
        inputs = processor(
            text=text,
            images=test_image_224,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(strix_device) for k, v in inputs.items()}
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(3):
                _ = model(**inputs)
        torch.cuda.synchronize()
        
        # Profile inference
        print("Profiling CLIP inference...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                torch.cuda.synchronize()
        
        print(f"✓ Output shape: {logits_per_image.shape}")
        
        # Print profiling results
        print("\n=== Top 15 GPU Operations ===")
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=15
        ))
        
        # Calculate metrics
        key_averages = prof.key_averages()
        total_cuda_time = sum([item.cuda_time_total for item in key_averages])
        print(f"\n✓ Total GPU time: {total_cuda_time / 1000:.2f} ms")
        
        # Record metrics as test properties
        if hasattr(pytest, 'current_test_info'):
            pytest.current_test_info['model'] = model_name
            pytest.current_test_info['gpu_time_ms'] = total_cuda_time / 1000
            pytest.current_test_info['num_operations'] = len(key_averages)


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.vit
@pytest.mark.p1
class TestViTProfiling:
    """Profile Vision Transformer models on Strix"""
    
    def test_vit_inference_profile(self, strix_device, test_image_224, cleanup_gpu):
        """Profile ViT model inference"""
        if not TORCH_AVAILABLE or not VIT_AVAILABLE:
            pytest.skip("PyTorch or ViT not available")
        
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        print("\n=== Profiling ViT Inference on Strix ===")
        
        # Load ViT model
        model_name = "google/vit-base-patch16-224"
        print(f"Loading model: {model_name}")
        
        model = ViTForImageClassification.from_pretrained(model_name).to(strix_device)
        processor = ViTImageProcessor.from_pretrained(model_name)
        model.eval()
        
        # Prepare inputs
        inputs = processor(images=test_image_224, return_tensors="pt")
        inputs = {k: v.to(strix_device) for k, v in inputs.items()}
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(3):
                _ = model(**inputs)
        torch.cuda.synchronize()
        
        # Profile inference
        print("Profiling ViT inference...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                torch.cuda.synchronize()
        
        predicted_class = logits.argmax(-1).item()
        print(f"✓ Predicted class: {predicted_class}")
        
        # Print profiling results
        print("\n=== Top 15 GPU Operations ===")
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=15
        ))
        
        # Calculate metrics
        key_averages = prof.key_averages()
        total_cuda_time = sum([item.cuda_time_total for item in key_averages])
        total_cpu_time = sum([item.cpu_time_total for item in key_averages])
        
        print(f"\n✓ Total GPU time: {total_cuda_time / 1000:.2f} ms")
        print(f"✓ Total CPU time: {total_cpu_time / 1000:.2f} ms")
        
        # Record metrics
        if hasattr(pytest, 'current_test_info'):
            pytest.current_test_info['model'] = model_name
            pytest.current_test_info['gpu_time_ms'] = total_cuda_time / 1000
            pytest.current_test_info['cpu_time_ms'] = total_cpu_time / 1000
    
    @pytest.mark.slow
    def test_vit_batch_inference_profile(self, strix_device, cleanup_gpu):
        """Profile ViT with different batch sizes"""
        if not TORCH_AVAILABLE or not VIT_AVAILABLE or not PIL_AVAILABLE:
            pytest.skip("Required libraries not available")
        
        print("\n=== Profiling ViT with Different Batch Sizes ===")
        
        model_name = "google/vit-base-patch16-224"
        model = ViTForImageClassification.from_pretrained(model_name).to(strix_device)
        processor = ViTImageProcessor.from_pretrained(model_name)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n--- Batch size: {batch_size} ---")
            
            # Create batch of images
            images = [Image.new('RGB', (224, 224), color='blue') for _ in range(batch_size)]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(strix_device) for k, v in inputs.items()}
            
            # Warmup
            with torch.no_grad():
                for _ in range(2):
                    _ = model(**inputs)
            torch.cuda.synchronize()
            
            # Profile
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                with torch.no_grad():
                    _ = model(**inputs)
                    torch.cuda.synchronize()
            
            key_averages = prof.key_averages()
            total_cuda_time = sum([item.cuda_time_total for item in key_averages])
            time_per_image = total_cuda_time / batch_size / 1000
            
            results[batch_size] = {
                'total_time_ms': total_cuda_time / 1000,
                'time_per_image_ms': time_per_image
            }
            
            print(f"  Total time: {total_cuda_time / 1000:.2f} ms")
            print(f"  Time per image: {time_per_image:.2f} ms")
        
        # Print summary
        print("\n=== Batch Processing Summary ===")
        for bs, metrics in results.items():
            print(f"Batch {bs}: {metrics['total_time_ms']:.2f} ms total, "
                  f"{metrics['time_per_image_ms']:.2f} ms per image")


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.cv
@pytest.mark.p1
class TestYOLOProfiling:
    """Profile YOLO object detection on Strix"""
    
    def test_yolo_inference_profile(self, strix_device, test_image_512, cleanup_gpu):
        """Profile YOLO model inference"""
        if not TORCH_AVAILABLE or not ULTRALYTICS_AVAILABLE:
            pytest.skip("PyTorch or Ultralytics not available")
        
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        print("\n=== Profiling YOLO Inference on Strix ===")
        
        # Load YOLO model
        model_name = "yolov8n.pt"  # Nano model for faster testing
        print(f"Loading model: {model_name}")
        
        try:
            model = YOLO(model_name)
        except Exception as e:
            pytest.skip(f"Could not load YOLO model: {e}")
        
        # Warmup
        print("Warming up...")
        for _ in range(3):
            _ = model.predict(test_image_512, verbose=False, device=0)
        torch.cuda.synchronize()
        
        # Profile inference
        print("Profiling YOLO inference...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            results = model.predict(test_image_512, verbose=False, device=0)
            torch.cuda.synchronize()
        
        print(f"✓ Detection completed")
        if results and len(results) > 0:
            print(f"  Detected {len(results[0].boxes)} objects")
        
        # Print profiling results
        print("\n=== Top 15 GPU Operations ===")
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=15
        ))
        
        # Calculate metrics
        key_averages = prof.key_averages()
        total_cuda_time = sum([item.cuda_time_total for item in key_averages])
        print(f"\n✓ Total GPU time: {total_cuda_time / 1000:.2f} ms")
        
        # Record metrics
        if hasattr(pytest, 'current_test_info'):
            pytest.current_test_info['model'] = model_name
            pytest.current_test_info['gpu_time_ms'] = total_cuda_time / 1000
            if results and len(results) > 0:
                pytest.current_test_info['num_detections'] = len(results[0].boxes)


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.quick
@pytest.mark.p0
def test_quick_ai_profiling_smoke(strix_device):
    """Quick smoke test for AI model profiling"""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    
    print("\n=== Quick AI Profiling Smoke Test ===")
    
    # Simple Conv2d operation (common in AI models)
    conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(strix_device)
    conv.eval()
    
    input_tensor = torch.randn(1, 3, 224, 224, device=strix_device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = conv(input_tensor)
    torch.cuda.synchronize()
    
    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with torch.no_grad():
            output = conv(input_tensor)
            torch.cuda.synchronize()
    
    assert output.shape == (1, 64, 224, 224), "Unexpected output shape"
    
    events = prof.key_averages()
    assert len(events) > 0, "No profiling events captured"
    
    total_cuda_time = sum([item.cuda_time_total for item in events])
    print(f"✓ Conv2d operation profiled: {total_cuda_time / 1000:.2f} ms")
    print(f"✓ Captured {len(events)} profiling events")

