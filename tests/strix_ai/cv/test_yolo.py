"""
YOLO (You Only Look Once) Object Detection tests for Strix platforms

Tests YOLOv8 for real-time object detection.
This is a P0 critical test for webcam processing and video conferencing use cases.
"""

import pytest
import torch
import numpy as np
import os

# Skip module if dependencies not available
ultralytics = pytest.importorskip("ultralytics", reason="ultralytics not installed")

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")


@pytest.mark.strix
@pytest.mark.cv
@pytest.mark.p0
@pytest.mark.quick
@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-specific CV test"
)
class TestYOLO:
    """YOLOv8 Object Detection tests for Strix"""
    
    def test_yolo_basic_detection(self, strix_device, cleanup_gpu):
        """Test basic YOLOv8 object detection on Strix"""
        from ultralytics import YOLO
        
        print("\n🧠 Loading YOLOv8-nano model...")
        # Use nano model for edge devices
        model = YOLO('yolov8n.pt')
        model.to('cuda')
        
        # Create test image
        print("🔍 Creating test image...")
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        print("⚡ Running YOLO detection on Strix...")
        results = model(image, device='cuda', verbose=False)
        
        print(f"✅ YOLO detection complete!")
        print(f"   Results: {len(results)} image(s) processed")
        
        # Assertions
        assert len(results) > 0, "Should have at least one result"
        assert results[0].boxes.data.device.type == "cuda", "Boxes should be on GPU"
        
        print(f"✅ YOLO basic test passed on {AMDGPU_FAMILIES}!")
    
    def test_yolo_confidence_threshold(self, strix_device, cleanup_gpu):
        """Test YOLO with different confidence thresholds"""
        from ultralytics import YOLO
        
        print("\n🧠 Loading YOLOv8-nano for confidence test...")
        model = YOLO('yolov8n.pt')
        model.to('cuda')
        
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test different confidence thresholds
        thresholds = [0.25, 0.5, 0.75]
        for conf in thresholds:
            print(f"🔍 Testing with confidence threshold: {conf}")
            results = model(image, device='cuda', conf=conf, verbose=False)
            assert len(results) > 0
        
        print(f"✅ YOLO confidence test passed on {AMDGPU_FAMILIES}!")
    
    @pytest.mark.slow
    def test_yolo_realtime_fps(self, strix_device, cleanup_gpu):
        """Test YOLO real-time detection FPS on Strix"""
        from ultralytics import YOLO
        import time
        
        print("\n🧠 Loading YOLOv8-nano for FPS test...")
        model = YOLO('yolov8n.pt')
        model.to('cuda')
        
        # Simulate webcam frames (640x480 like typical webcam)
        print("📹 Simulating webcam frames at 640x480...")
        
        # Warmup
        print("🔥 Warming up...")
        for _ in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            _ = model(frame, device='cuda', verbose=False)
        torch.cuda.synchronize()
        
        # Benchmark
        print("⏱️  Benchmarking real-time performance...")
        num_frames = 90  # Simulate 3 seconds at 30 FPS
        frame_times = []
        
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            start = time.time()
            _ = model(frame, device='cuda', verbose=False)
            torch.cuda.synchronize()
            frame_times.append(time.time() - start)
        
        avg_fps = 1.0 / np.mean(frame_times)
        min_fps = 1.0 / np.max(frame_times)
        max_fps = 1.0 / np.min(frame_times)
        
        print(f"📊 Performance Results:")
        print(f"   Average FPS: {avg_fps:.2f}")
        print(f"   Min FPS: {min_fps:.2f}")
        print(f"   Max FPS: {max_fps:.2f}")
        print(f"   Avg latency: {np.mean(frame_times)*1000:.2f} ms")
        
        # Target: > 15 FPS for real-time webcam processing
        # This is critical for Teams/Zoom background processing
        print(f"   Target: > 15 FPS (current: {avg_fps:.2f} FPS)")
        
        assert avg_fps >= 15, f"YOLO FPS too low for real-time: {avg_fps:.2f} FPS (expected >= 15 FPS)"
        
        print(f"✅ YOLO real-time test passed on {AMDGPU_FAMILIES}!")
    
    def test_yolo_batch_detection(self, strix_device, cleanup_gpu):
        """Test YOLO batch processing on Strix"""
        from ultralytics import YOLO
        
        print("\n🧠 Loading YOLOv8-nano for batch test...")
        model = YOLO('yolov8n.pt')
        model.to('cuda')
        
        # Create batch of images
        batch_size = 4
        print(f"🔍 Processing batch of {batch_size} images...")
        images = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(batch_size)]
        
        print("⚡ Running batch detection...")
        results = model(images, device='cuda', verbose=False)
        
        print(f"✅ Batch detection successful!")
        print(f"   Processed: {len(results)} images")
        
        assert len(results) == batch_size
        
        print(f"✅ YOLO batch test passed on {AMDGPU_FAMILIES}!")
    
    def test_yolo_memory_efficiency(self, strix_device, cleanup_gpu):
        """Test YOLO memory usage on Strix iGPU"""
        from ultralytics import YOLO
        
        print("\n🧠 Loading YOLOv8-nano for memory test...")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model = YOLO('yolov8n.pt')
        model.to('cuda')
        
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        print("⚡ Running detection and measuring memory...")
        _ = model(image, device='cuda', verbose=False)
        torch.cuda.synchronize()
        
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"📊 Memory Usage:")
        print(f"   Peak: {peak_memory_mb:.2f} MB")
        
        # YOLOv8n is very small (~6MB), should be very memory efficient
        assert peak_memory_mb < 1024, f"Memory usage too high: {peak_memory_mb:.2f} MB"
        
        print(f"✅ YOLO memory test passed on {AMDGPU_FAMILIES}!")
    
    def test_yolo_model_variants(self, strix_device, cleanup_gpu):
        """Test different YOLO model sizes on Strix"""
        from ultralytics import YOLO
        
        # Test nano and small models (appropriate for edge)
        models = ['yolov8n.pt']  # Nano for quick test
        
        for model_name in models:
            print(f"\n🧠 Testing {model_name}...")
            model = YOLO(model_name)
            model.to('cuda')
            
            image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(image, device='cuda', verbose=False)
            
            assert len(results) > 0
            print(f"✅ {model_name} works on {AMDGPU_FAMILIES}")
        
        print(f"✅ YOLO variants test passed!")

