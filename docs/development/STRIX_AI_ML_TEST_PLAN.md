# Strix AI/ML Test Plan: Edge AI & Vision Workloads

## 🎯 Executive Summary

This document outlines **NEW test cases** required for AMD Strix platforms (gfx1150, gfx1151) that are **NOT covered** by existing MI (data center) or Navi (discrete GPU) test suites. Strix targets **Edge AI, Computer Vision, and Consumer AI** use cases unique to integrated GPUs.

---

## 📊 Current Test Gap Analysis

### ✅ **What's Already Tested** (MI/Navi Coverage)
- Low-level libraries (rocBLAS, hipBLAS, MIOpen)
- Basic PyTorch operations (matrix multiply, conv2d)
- HPC workloads (FFT, sparse, linear algebra)
- Training-focused tests

### ❌ **What's Missing for Strix** (Edge AI/Consumer Focus)
- ❌ **Vision Language Models (VLM/VLA)**
- ❌ **Vision Transformers (ViT)**
- ❌ **Computer Vision (Object Detection, Segmentation)**
- ❌ **Edge AI Inference** (Quantized models, ONNX)
- ❌ **Video Processing** (Encode/Decode acceleration)
- ❌ **Windows AI Platform** (DirectML, WinML)
- ❌ **Consumer AI Apps** (Background blur, upscaling, enhancement)
- ❌ **iGPU-Specific Optimizations** (Shared memory, power efficiency)

---

## 🧠 Test Category 1: Vision Language Models (VLM/VLA)

### **Priority: CRITICAL**
**Why Strix Needs This:** Strix powers Windows Copilot+, AI PCs requiring multimodal understanding

### **Models to Test**

| Model | Size | Use Case | Priority |
|-------|------|----------|----------|
| **LLaVA-v1.6** | 7B, 13B | Visual Q&A | 🔴 Critical |
| **CLIP (OpenAI)** | ViT-B/32, ViT-L/14 | Image-text alignment | 🔴 Critical |
| **Qwen-VL** | 7B | Multimodal chat | 🟡 High |
| **MiniGPT-4** | 7B | Vision understanding | 🟡 High |
| **BLIP-2** | Multiple | Image captioning | 🟢 Medium |
| **InternVL** | Multiple | Multimodal large model | 🟢 Medium |

### **Test Cases**

#### **1.1 VLM Inference Test**

```python
# File: tests/strix_ai/test_vlm_inference.py

import pytest
import os
import torch
from PIL import Image

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")

@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-specific VLM tests"
)
class TestVLMInference:
    """Vision Language Model inference tests for Strix"""
    
    def test_llava_image_understanding(self):
        """Test LLaVA-1.6-7B on Strix iGPU"""
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
        # Load lightweight model
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Test image
        image = Image.new('RGB', (224, 224), color='red')
        prompt = "What is in this image?"
        
        inputs = processor(prompt, image, return_tensors="pt").to("cuda")
        
        # Run inference
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50)
        
        result = processor.decode(output[0], skip_special_tokens=True)
        
        assert len(result) > 0
        assert inputs['pixel_values'].device.type == "cuda"
    
    def test_clip_image_text_matching(self):
        """Test CLIP vision-text alignment on Strix"""
        from transformers import CLIPProcessor, CLIPModel
        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Test image and texts
        image = Image.new('RGB', (224, 224), color='blue')
        texts = ["a photo of a cat", "a photo of a dog", "a blue image"]
        
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Blue image should match "a blue image" most closely
        assert probs.device.type == "cuda"
        assert probs.shape[1] == 3  # 3 text options
    
    def test_vlm_batch_processing(self):
        """Test VLM batch inference for throughput"""
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            # Test that batch processing works without OOM
            images = [Image.new('RGB', (224, 224)) for _ in range(batch_size)]
            # ... process batch ...
    
    def test_vlm_memory_efficiency(self):
        """Test VLM memory usage on iGPU shared memory"""
        # Monitor memory usage during inference
        # Strix has shared system memory - test efficiency
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Run inference
        # ...
        
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Should stay under reasonable limits for iGPU
        assert peak_memory_mb < 4096, f"Memory usage too high: {peak_memory_mb}MB"
```

#### **1.2 VLM Quantization Test**

```python
def test_vlm_int8_quantization(self):
    """Test INT8 quantized VLM inference for edge deployment"""
    # Critical for Strix: quantization enables larger models on iGPU
    
    from transformers import AutoModelForCausalLM
    from auto_gptq import AutoGPTQForCausalLM
    
    # Load quantized model
    model = AutoGPTQForCausalLM.from_quantized(
        "TheBloke/LLaVA-7B-GPTQ",
        device="cuda:0"
    )
    
    # Test inference
    # Should be 2-4x faster and use less memory
```

---

## 👁️ Test Category 2: Vision Transformers (ViT)

### **Priority: CRITICAL**
**Why Strix Needs This:** ViT is the backbone of modern CV, used in Windows AI features

### **Models to Test**

| Model | Parameters | Use Case | Priority |
|-------|------------|----------|----------|
| **ViT-Base** | 86M | General vision | 🔴 Critical |
| **ViT-Large** | 304M | High accuracy | 🟡 High |
| **DINOv2** | Multiple | Self-supervised | 🔴 Critical |
| **Swin Transformer** | Multiple | Hierarchical vision | 🟡 High |
| **BEiT** | Multiple | Masked image modeling | 🟢 Medium |
| **DeiT** | Multiple | Distilled ViT | 🟢 Medium |

### **Test Cases**

#### **2.1 ViT Inference Test**

```python
# File: tests/strix_ai/test_vision_transformers.py

import pytest
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-specific ViT tests"
)
class TestVisionTransformers:
    """Vision Transformer tests for Strix platforms"""
    
    def test_vit_image_classification(self):
        """Test ViT-Base image classification on Strix"""
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        model = model.to('cuda')
        
        # Test image
        image = Image.new('RGB', (224, 224), color='green')
        inputs = processor(images=image, return_tensors="pt").to('cuda')
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
        
        assert outputs.logits.device.type == "cuda"
        assert 0 <= predicted_class < 1000  # ImageNet classes
    
    def test_dinov2_feature_extraction(self):
        """Test DINOv2 self-supervised features on Strix"""
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model = model.to('cuda')
        model.eval()
        
        # Test image
        img = torch.randn(1, 3, 224, 224).to('cuda')
        
        with torch.no_grad():
            features = model(img)
        
        assert features.device.type == "cuda"
        assert features.shape[0] == 1  # Batch size
        assert features.shape[1] == 384  # Feature dim for ViT-S/14
    
    def test_swin_transformer_hierarchical(self):
        """Test Swin Transformer hierarchical features"""
        from transformers import AutoImageProcessor, SwinForImageClassification
        
        processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        model = model.to('cuda')
        
        image = Image.new('RGB', (224, 224))
        inputs = processor(images=image, return_tensors="pt").to('cuda')
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert outputs.logits.device.type == "cuda"
    
    def test_vit_attention_patterns(self):
        """Test ViT attention mechanism on Strix"""
        # Verify multi-head attention works correctly on iGPU
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            output_attentions=True
        ).to('cuda')
        
        image = Image.new('RGB', (224, 224))
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        inputs = processor(images=image, return_tensors="pt").to('cuda')
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Check attention tensors
        assert len(outputs.attentions) == 12  # 12 layers
        assert outputs.attentions[0].device.type == "cuda"
    
    def test_vit_throughput_benchmark(self):
        """Benchmark ViT throughput on Strix"""
        import time
        
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to('cuda')
        model.eval()
        
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        image = Image.new('RGB', (224, 224))
        inputs = processor(images=image, return_tensors="pt").to('cuda')
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(**inputs)
        
        torch.cuda.synchronize()
        
        # Benchmark
        num_iterations = 100
        start = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(**inputs)
        
        torch.cuda.synchronize()
        end = time.time()
        
        fps = num_iterations / (end - start)
        print(f"ViT throughput: {fps:.2f} FPS")
        
        # Strix should achieve reasonable FPS for edge deployment
        assert fps > 10, f"Throughput too low: {fps} FPS"
```

---

## 🔍 Test Category 3: Computer Vision Workloads

### **Priority: CRITICAL**
**Why Strix Needs This:** Core use case for consumer AI (webcam processing, video calls, etc.)

### **CV Tasks to Test**

| Task | Models | Use Case | Priority |
|------|--------|----------|----------|
| **Object Detection** | YOLO, DETR, Faster R-CNN | Real-time detection | 🔴 Critical |
| **Semantic Segmentation** | SegFormer, Mask2Former | Scene understanding | 🔴 Critical |
| **Instance Segmentation** | Mask R-CNN, YOLACT | Object isolation | 🟡 High |
| **Pose Estimation** | OpenPose, MediaPipe | Human pose | 🟡 High |
| **Face Detection** | RetinaFace, MTCNN | Face recognition | 🔴 Critical |
| **Depth Estimation** | MiDaS, DPT | 3D understanding | 🟢 Medium |

### **Test Cases**

#### **3.1 Object Detection Test**

```python
# File: tests/strix_ai/test_computer_vision.py

import pytest
import torch
from PIL import Image
import cv2
import numpy as np

@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-specific CV tests"
)
class TestComputerVision:
    """Computer Vision workload tests for Strix"""
    
    def test_yolov8_object_detection(self):
        """Test YOLOv8 real-time object detection on Strix"""
        from ultralytics import YOLO
        
        # Load YOLOv8 model
        model = YOLO('yolov8n.pt')  # Nano model for edge devices
        model.to('cuda')
        
        # Create test image
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(image, device='cuda')
        
        # Verify results
        assert len(results) > 0
        assert results[0].boxes.data.device.type == "cuda"
    
    def test_detr_object_detection(self):
        """Test DETR (Detection Transformer) on Strix"""
        from transformers import DetrImageProcessor, DetrForObjectDetection
        
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        model = model.to('cuda')
        
        image = Image.new('RGB', (800, 800))
        inputs = processor(images=image, return_tensors="pt").to('cuda')
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process outputs
        target_sizes = torch.tensor([[800, 800]]).to('cuda')
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]
        
        assert outputs.logits.device.type == "cuda"
    
    def test_segformer_semantic_segmentation(self):
        """Test SegFormer for semantic segmentation"""
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        
        processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        model = model.to('cuda')
        
        image = Image.new('RGB', (512, 512))
        inputs = processor(images=image, return_tensors="pt").to('cuda')
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Upsample to original size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode='bilinear',
            align_corners=False
        )
        
        assert upsampled_logits.device.type == "cuda"
        assert upsampled_logits.shape[2:] == (512, 512)
    
    def test_mask_rcnn_instance_segmentation(self):
        """Test Mask R-CNN for instance segmentation"""
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision import transforms
        
        model = maskrcnn_resnet50_fpn(pretrained=True)
        model = model.to('cuda')
        model.eval()
        
        # Create test image
        image = torch.rand(3, 800, 800).to('cuda')
        
        with torch.no_grad():
            predictions = model([image])
        
        assert len(predictions) == 1
        assert predictions[0]['boxes'].device.type == "cuda"
        assert 'masks' in predictions[0]
    
    def test_mediapipe_pose_estimation(self):
        """Test MediaPipe Pose estimation on Strix"""
        # MediaPipe uses TensorFlow Lite
        # Test ROCm backend for TFLite
        import mediapipe as mp
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        
        # Test image
        image = np.zeros((640, 480, 3), dtype=np.uint8)
        results = pose.process(image)
        
        # Verify pose landmarks detected (if any)
        # Even with empty image, should not crash
        assert results is not None
    
    def test_face_detection_retinaface(self):
        """Test RetinaFace for face detection"""
        # Critical for Windows Hello, video conferencing
        # ... implementation ...
    
    def test_realtime_webcam_processing(self):
        """Test real-time video stream processing on Strix"""
        # Simulate webcam frames at 30 FPS
        model = YOLO('yolov8n.pt').to('cuda')
        
        num_frames = 90  # 3 seconds at 30 FPS
        frame_times = []
        
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            start = time.time()
            results = model(frame, device='cuda')
            torch.cuda.synchronize()
            frame_times.append(time.time() - start)
        
        avg_fps = 1.0 / np.mean(frame_times)
        print(f"Average FPS: {avg_fps:.2f}")
        
        # Should achieve at least 15 FPS for real-time
        assert avg_fps >= 15, f"FPS too low for real-time: {avg_fps}"
```

#### **3.2 CV Pipeline Tests**

```python
def test_cv_pipeline_background_blur(self):
    """Test background blur pipeline (Teams/Zoom use case)"""
    # 1. Semantic segmentation to extract person
    # 2. Blur background
    # 3. Composite
    
    # This is a key Strix use case!

def test_cv_pipeline_background_replacement(self):
    """Test virtual background replacement"""
    # Video conferencing feature

def test_cv_pipeline_face_beautification(self):
    """Test face enhancement pipeline"""
    # Consumer webcam applications
```

---

## ⚡ Test Category 4: Edge AI Inference Optimization

### **Priority: CRITICAL**
**Why Strix Needs This:** iGPU requires optimization for power/performance

### **Test Cases**

#### **4.1 Model Quantization Tests**

```python
# File: tests/strix_ai/test_edge_inference.py

class TestEdgeInference:
    """Edge AI inference optimization tests"""
    
    def test_int8_quantization_performance(self):
        """Test INT8 quantized model performance"""
        # FP16 vs INT8 comparison
        # INT8 should be 2-3x faster on Strix
    
    def test_dynamic_quantization(self):
        """Test PyTorch dynamic quantization on Strix"""
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    def test_onnx_runtime_rocm(self):
        """Test ONNX Runtime with ROCm backend"""
        import onnxruntime as ort
        
        # Create ONNX Runtime session with ROCm
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession("model.onnx", providers=providers)
        
        # Run inference
        # Should utilize Strix GPU
    
    def test_torch_compile_optimization(self):
        """Test torch.compile() optimization on Strix"""
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        model = model.to('cuda')
        
        # Compile model
        compiled_model = torch.compile(model, mode="reduce-overhead")
        
        # Compare performance
        # Compiled should be faster
    
    def test_mixed_precision_inference(self):
        """Test FP16 mixed precision for faster inference"""
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Run inference
            # Should be 2x faster than FP32
            pass
    
    def test_batch_size_optimization(self):
        """Find optimal batch size for Strix iGPU"""
        batch_sizes = [1, 2, 4, 8, 16]
        throughputs = []
        
        for batch_size in batch_sizes:
            # Measure throughput
            # Find sweet spot for iGPU memory
            pass
```

---

## 🎥 Test Category 5: Video Processing

### **Priority: HIGH**
**Why Strix Needs This:** Integrated media engines, video conferencing

### **Test Cases**

```python
# File: tests/strix_ai/test_video_processing.py

class TestVideoProcessing:
    """Video processing tests for Strix"""
    
    def test_video_encoding_h264(self):
        """Test H.264 video encoding on Strix VCN"""
        # Strix has hardware video encode/decode (VCN 4.0)
        import av
        
        # Encode video using hardware acceleration
        # Should use VCN, not GPU compute
    
    def test_video_decoding_h265(self):
        """Test H.265/HEVC decoding on Strix"""
        # Hardware decode test
    
    def test_realtime_video_inference(self):
        """Test AI inference on video stream"""
        # Read video, run object detection frame-by-frame
        # Should maintain 30 FPS
    
    def test_video_super_resolution(self):
        """Test video upscaling with AI"""
        # ESRGAN, RealESRGAN for video enhancement
    
    def test_video_stabilization(self):
        """Test AI-powered video stabilization"""
        # Consumer camera applications
```

---

## 🪟 Test Category 6: Windows AI Platform Integration

### **Priority: HIGH**
**Why Strix Needs This:** Windows Copilot+, DirectML integration

### **Test Cases**

```python
# File: tests/strix_ai/test_windows_ai_platform.py

import pytest
import platform

@pytest.mark.skipif(
    platform.system() != "Windows",
    reason="Windows-specific tests"
)
class TestWindowsAIPlatform:
    """Windows AI Platform tests for Strix"""
    
    def test_directml_backend(self):
        """Test DirectML as PyTorch backend on Strix"""
        # Windows ML uses DirectML
        import torch_directml
        
        device = torch_directml.device()
        model = model.to(device)
        
        # Run inference on DirectML
    
    def test_windows_ml_onnx(self):
        """Test Windows ML (WinML) with ONNX models"""
        # Windows AI APIs
        import winrt.windows.ai.machinelearning as winml
        
        # Load and run model through WinML
    
    def test_copilot_studio_integration(self):
        """Test integration with Windows Copilot Studio"""
        # Copilot+ PC features
    
    def test_windows_studio_effects(self):
        """Test Windows Studio Effects (background blur, etc.)"""
        # Windows 11 22H2+ features
```

---

## 🎨 Test Category 7: Consumer AI Applications

### **Priority: MEDIUM**
**Why Strix Needs This:** End-user facing features

### **Test Cases**

```python
# File: tests/strix_ai/test_consumer_ai.py

class TestConsumerAI:
    """Consumer AI application tests"""
    
    def test_background_blur_quality(self):
        """Test background blur quality metrics"""
        # PSNR, SSIM for blur quality
    
    def test_image_super_resolution(self):
        """Test image upscaling (RealESRGAN, ESRGAN)"""
        from realesrgan import RealESRGANer
        
        upsampler = RealESRGANer(device='cuda')
        # Test 2x, 4x upscaling
    
    def test_photo_enhancement(self):
        """Test automatic photo enhancement"""
        # Brightness, contrast, color correction
    
    def test_noise_reduction(self):
        """Test AI-powered noise reduction"""
        # Low-light photo enhancement
    
    def test_style_transfer(self):
        """Test neural style transfer"""
        # Artistic filters
    
    def test_generative_fill(self):
        """Test AI image inpainting"""
        # Remove objects, fill backgrounds
```

---

## 🔋 Test Category 8: iGPU-Specific Optimizations

### **Priority: HIGH**
**Why Strix Needs This:** Unique to integrated GPUs

### **Test Cases**

```python
# File: tests/strix_ai/test_igpu_optimizations.py

class TestIGPUOptimizations:
    """Tests specific to iGPU characteristics"""
    
    def test_shared_memory_efficiency(self):
        """Test efficient use of shared system memory"""
        # Strix shares memory with CPU
        # Test zero-copy transfers
        
        import torch
        
        # Allocate on CPU
        cpu_tensor = torch.randn(1000, 1000)
        
        # Pin memory for fast transfer
        cpu_tensor = cpu_tensor.pin_memory()
        gpu_tensor = cpu_tensor.to('cuda', non_blocking=True)
        
        # Measure transfer time (should be minimal)
    
    def test_power_efficiency_inference(self):
        """Test power consumption during inference"""
        # iGPU should use less power than dGPU
        
        # Monitor power draw (if available)
        # Run inference workload
        # Verify power stays in reasonable range
    
    def test_thermal_throttling_behavior(self):
        """Test behavior under thermal constraints"""
        # iGPU may throttle in laptops
        
        # Run sustained workload
        # Monitor performance degradation
    
    def test_cpu_gpu_collaboration(self):
        """Test CPU+GPU collaborative inference"""
        # Some layers on CPU, some on GPU
        # Optimal for iGPU systems
    
    def test_unified_memory_access(self):
        """Test unified memory access patterns"""
        # AMD Infinity Cache for iGPU
        
        # Test memory access patterns
        # Verify cache coherency
```

---

## 📋 Implementation Priority Matrix

### **Phase 1: Critical (Immediate)**
| Priority | Test Category | Estimated Effort | Impact |
|----------|---------------|------------------|--------|
| 🔴 P0 | **VLM Inference** (LLaVA, CLIP) | 2-3 weeks | Very High |
| 🔴 P0 | **ViT Models** (ViT-Base, DINOv2) | 2 weeks | Very High |
| 🔴 P0 | **Object Detection** (YOLOv8, DETR) | 2 weeks | Very High |
| 🔴 P0 | **Edge Optimization** (Quantization, ONNX) | 1-2 weeks | Very High |

### **Phase 2: High (Next Quarter)**
| Priority | Test Category | Estimated Effort | Impact |
|----------|---------------|------------------|--------|
| 🟡 P1 | **Semantic Segmentation** (SegFormer) | 1-2 weeks | High |
| 🟡 P1 | **Video Processing** (Encode/Decode) | 2 weeks | High |
| 🟡 P1 | **Windows AI Platform** (DirectML) | 2-3 weeks | High |
| 🟡 P1 | **iGPU Optimizations** | 1 week | High |

### **Phase 3: Medium (Future)**
| Priority | Test Category | Estimated Effort | Impact |
|----------|---------------|------------------|--------|
| 🟢 P2 | **Consumer AI Apps** | 1-2 weeks | Medium |
| 🟢 P2 | **Additional CV Tasks** | 1-2 weeks | Medium |
| 🟢 P2 | **Pose Estimation** | 1 week | Medium |

---

## 🚀 Quick Start: Implementing First Test

### **Example: Implementing LLaVA VLM Test**

**Step 1:** Create test directory structure
```bash
mkdir -p tests/strix_ai
touch tests/strix_ai/__init__.py
touch tests/strix_ai/test_vlm_inference.py
```

**Step 2:** Install dependencies
```bash
pip install transformers accelerate pillow torch torchvision
```

**Step 3:** Implement test (see code above in Section 1.1)

**Step 4:** Run test
```bash
export THEROCK_BIN_DIR=/opt/rocm/bin
export AMDGPU_FAMILIES=gfx1151
pytest tests/strix_ai/test_vlm_inference.py::TestVLMInference::test_llava_image_understanding -v
```

**Step 5:** Add to CI (optional)
```python
# Edit fetch_test_configurations.py
"strix_vlm": {
    "job_name": "strix_vlm",
    "fetch_artifact_args": "--pytorch --tests",
    "timeout_minutes": 30,
    "test_script": "pytest tests/strix_ai/test_vlm_inference.py -v",
    "platform": ["linux", "windows"],
    "total_shards": 1,
}
```

---

## 📊 Success Metrics

### **Performance Targets for Strix**

| Metric | Target | Measurement |
|--------|--------|-------------|
| **VLM Inference Latency** | < 2s for 7B model | Time to first token |
| **ViT Throughput** | > 30 FPS (ViT-Base) | Images per second |
| **YOLO Detection** | > 15 FPS (YOLOv8n) | Real-time video |
| **Memory Usage** | < 4GB peak | PyTorch memory stats |
| **Power Efficiency** | < 25W sustained | System power monitoring |
| **Model Load Time** | < 10s | Cold start |

### **Quality Metrics**

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Accuracy** | Within 2% of reference | Compare to CPU/CUDA |
| **Numerical Stability** | No NaN/Inf | Output validation |
| **Reproducibility** | Deterministic results | Multiple runs |

---

## 🔗 Model Sources & Datasets

### **Hugging Face Models**
```python
RECOMMENDED_MODELS = {
    "vlm": [
        "llava-hf/llava-1.5-7b-hf",
        "openai/clip-vit-base-patch32",
    ],
    "vit": [
        "google/vit-base-patch16-224",
        "facebook/dinov2-base",
    ],
    "detection": [
        "facebook/detr-resnet-50",
        # YOLOv8 from Ultralytics
    ],
    "segmentation": [
        "nvidia/segformer-b0-finetuned-ade-512-512",
    ]
}
```

### **Test Datasets**
- **COCO** - Object detection
- **ImageNet** - Classification
- **ADE20K** - Segmentation
- **CelebA** - Face detection
- **Custom Strix Benchmark** - Edge AI scenarios

---

## 🛠️ Test Infrastructure Needs

### **Dependencies to Add**
```bash
# requirements_strix_ai.txt
transformers>=4.35.0
accelerate>=0.24.0
bitsandbytes>=0.41.0  # For quantization
ultralytics>=8.0.0    # YOLOv8
onnxruntime-rocm>=1.16.0
opencv-python>=4.8.0
pillow>=10.0.0
timm>=0.9.0           # PyTorch Image Models
einops>=0.7.0         # Tensor operations
```

### **Hardware Requirements**
- **Minimum**: Strix Halo (gfx1151) with 16GB RAM
- **Recommended**: Strix Halo with 32GB RAM
- **Storage**: 50GB for models

### **CI/CD Integration**
- Add `strix_ai` test suite to nightly builds
- Run on Strix hardware runners
- Timeout: 60-90 minutes per suite
- Sharding: Split by model category

---

## 📚 References

### **AMD Documentation**
- [ROCm AI Documentation](https://rocm.docs.amd.com)
- [Strix Architecture Guide](https://www.amd.com/en/products/specifications/processors.html)

### **Model Documentation**
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Ultralytics YOLO](https://docs.ultralytics.com)
- [ONNX Runtime](https://onnxruntime.ai/docs/)

### **Windows AI**
- [Windows ML](https://docs.microsoft.com/en-us/windows/ai/windows-ml/)
- [DirectML](https://docs.microsoft.com/en-us/windows/ai/directml/dml)

---

## ✅ Action Items

### **Immediate (This Quarter)**
- [ ] Set up `tests/strix_ai/` directory structure
- [ ] Implement VLM test suite (LLaVA, CLIP)
- [ ] Implement ViT test suite (ViT-Base, DINOv2)
- [ ] Implement Object Detection suite (YOLOv8, DETR)
- [ ] Add quantization/optimization tests
- [ ] Document baseline performance metrics

### **Short Term (Next Quarter)**
- [ ] Add semantic segmentation tests
- [ ] Add video processing tests
- [ ] Add Windows AI platform tests
- [ ] Add iGPU optimization tests
- [ ] Create Strix AI benchmark dashboard

### **Long Term (6+ Months)**
- [ ] Comprehensive CV task coverage
- [ ] Consumer AI application tests
- [ ] Performance regression tracking
- [ ] Power efficiency benchmarking

---

## 💡 Summary

**Key Gaps for Strix:**
1. ❌ **No VLM/Vision Language tests** - Critical for AI PC use cases
2. ❌ **No Vision Transformer tests** - Backbone of modern CV
3. ❌ **No practical CV workloads** - Object detection, segmentation
4. ❌ **No edge inference optimization** - Quantization, ONNX
5. ❌ **No iGPU-specific tests** - Shared memory, power efficiency
6. ❌ **No Windows AI integration** - DirectML, WinML

**Recommended First Steps:**
1. Start with **VLM inference tests** (LLaVA, CLIP) - highest business impact
2. Add **ViT tests** (ViT-Base, DINOv2) - foundational
3. Implement **YOLO object detection** - real-time use case
4. Add **quantization tests** - critical for edge deployment

**These tests differentiate Strix from MI (datacenter training) and Navi (gaming) GPUs!**

