# Strix AI Testing Quick Start

## 🚀 Quick Implementation Guide for VLM/ViT/CV Tests

This guide helps you **quickly implement** the AI/ML tests for Strix platforms identified in the [STRIX_AI_ML_TEST_PLAN.md](./STRIX_AI_ML_TEST_PLAN.md).

---

## ⚡ 5-Minute Setup

### **Step 1: Install Dependencies**

```bash
# Navigate to TheRock directory
cd TheRock

# Create AI test environment
python -m pip install --upgrade pip

# Install AI/ML packages
pip install transformers>=4.35.0 \
            accelerate>=0.24.0 \
            ultralytics>=8.0.0 \
            onnxruntime-rocm>=1.16.0 \
            opencv-python>=4.8.0 \
            pillow>=10.0.0 \
            timm>=0.9.0
```

### **Step 2: Create Test Structure**

```bash
# Create AI test directory
mkdir -p tests/strix_ai
touch tests/strix_ai/__init__.py
```

### **Step 3: Run Your First AI Test**

Create `tests/strix_ai/test_quick_vit.py`:

```python
import pytest
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")

@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-specific test"
)
def test_vit_on_strix():
    """Quick ViT test on Strix"""
    print("Loading ViT model...")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model = model.to('cuda')
    
    print("Creating test image...")
    image = Image.new('RGB', (224, 224), color='blue')
    inputs = processor(images=image, return_tensors="pt").to('cuda')
    
    print("Running inference on Strix...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"✅ ViT inference successful on {AMDGPU_FAMILIES}!")
    print(f"Output shape: {outputs.logits.shape}")
    print(f"Running on device: {outputs.logits.device}")
    
    assert outputs.logits.device.type == "cuda"
    assert outputs.logits.shape[0] == 1
```

**Run it:**
```bash
export THEROCK_BIN_DIR=/opt/rocm/bin
export AMDGPU_FAMILIES=gfx1151
pytest tests/strix_ai/test_quick_vit.py -v -s
```

---

## 🧠 Test Templates

### **Template 1: VLM Test (CLIP)**

```python
# tests/strix_ai/test_clip_quick.py
import pytest, torch, os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")

@pytest.mark.skipif(AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"], reason="Strix only")
def test_clip_image_text():
    """CLIP vision-language test"""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    image = Image.new('RGB', (224, 224), color='red')
    texts = ["a red image", "a blue image", "a green image"]
    
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
    
    print(f"Text probabilities: {probs}")
    assert probs.argmax() == 0  # "a red image" should have highest probability
    assert probs.device.type == "cuda"
```

### **Template 2: YOLO Object Detection**

```python
# tests/strix_ai/test_yolo_quick.py
import pytest, os
import numpy as np

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")

@pytest.mark.skipif(AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"], reason="Strix only")
def test_yolo_detection():
    """YOLOv8 object detection test"""
    from ultralytics import YOLO
    
    model = YOLO('yolov8n.pt')  # Nano model
    model.to('cuda')
    
    # Create test image
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Run inference
    results = model(image, device='cuda')
    
    print(f"✅ YOLO inference on {AMDGPU_FAMILIES}")
    assert len(results) > 0
```

### **Template 3: Semantic Segmentation**

```python
# tests/strix_ai/test_segmentation_quick.py
import pytest, torch, os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")

@pytest.mark.skipif(AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"], reason="Strix only")
def test_segformer():
    """SegFormer semantic segmentation test"""
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = model.to('cuda')
    
    image = Image.new('RGB', (512, 512))
    inputs = processor(images=image, return_tensors="pt").to('cuda')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"✅ Segmentation on {AMDGPU_FAMILIES}")
    print(f"Output shape: {outputs.logits.shape}")
    assert outputs.logits.device.type == "cuda"
```

### **Template 4: Quantization Test**

```python
# tests/strix_ai/test_quantization_quick.py
import pytest, torch, os, time

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")

@pytest.mark.skipif(AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"], reason="Strix only")
def test_int8_vs_fp32():
    """Compare INT8 vs FP32 performance"""
    from transformers import ViTForImageClassification
    
    model_fp32 = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to('cuda')
    model_fp32.eval()
    
    # Test input
    test_input = torch.randn(1, 3, 224, 224).to('cuda')
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model_fp32(test_input)
    
    # Benchmark FP32
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model_fp32(test_input)
    torch.cuda.synchronize()
    fp32_time = time.time() - start
    
    # FP16 mixed precision
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = model_fp32(test_input)
        torch.cuda.synchronize()
        fp16_time = time.time() - start
    
    print(f"FP32 time: {fp32_time:.3f}s")
    print(f"FP16 time: {fp16_time:.3f}s")
    print(f"Speedup: {fp32_time/fp16_time:.2f}x")
    
    assert fp16_time < fp32_time, "FP16 should be faster"
```

---

## 📋 Priority Implementation Order

### **Week 1: Core VLM/ViT**
```bash
tests/strix_ai/
├── __init__.py
├── test_clip_inference.py      # ✅ PRIORITY 1
├── test_vit_classification.py  # ✅ PRIORITY 2
└── test_dinov2_features.py     # ✅ PRIORITY 3
```

**Run tests:**
```bash
pytest tests/strix_ai/test_clip_inference.py -v
pytest tests/strix_ai/test_vit_classification.py -v
```

### **Week 2: Object Detection**
```bash
tests/strix_ai/
├── test_yolo_detection.py      # ✅ PRIORITY 4
├── test_detr_detection.py      # ✅ PRIORITY 5
└── test_realtime_fps.py        # ✅ PRIORITY 6
```

### **Week 3: Optimization**
```bash
tests/strix_ai/
├── test_quantization.py        # ✅ PRIORITY 7
├── test_onnx_runtime.py        # ✅ PRIORITY 8
└── test_mixed_precision.py     # ✅ PRIORITY 9
```

---

## 🎯 Full Test Suite Structure

```
tests/strix_ai/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── utils.py                    # Helper functions
│
├── vlm/                        # Vision Language Models
│   ├── test_clip.py
│   ├── test_llava.py
│   └── test_qwen_vl.py
│
├── vit/                        # Vision Transformers
│   ├── test_vit_base.py
│   ├── test_dinov2.py
│   └── test_swin.py
│
├── cv/                         # Computer Vision
│   ├── test_yolo.py
│   ├── test_detr.py
│   ├── test_segmentation.py
│   └── test_pose.py
│
├── optimization/               # Edge Inference
│   ├── test_quantization.py
│   ├── test_onnx.py
│   └── test_torch_compile.py
│
├── video/                      # Video Processing
│   ├── test_encoding.py
│   └── test_realtime.py
│
├── windows/                    # Windows AI (Windows only)
│   ├── test_directml.py
│   └── test_winml.py
│
└── benchmarks/                 # Performance
    ├── benchmark_throughput.py
    └── benchmark_latency.py
```

---

## 🔧 Shared Test Fixtures

Create `tests/strix_ai/conftest.py`:

```python
import pytest
import torch
import os

@pytest.fixture(scope="session")
def strix_device():
    """Get Strix GPU device"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device_name = torch.cuda.get_device_name(0)
    if "gfx115" not in device_name.lower():
        pytest.skip(f"Not a Strix device: {device_name}")
    
    return torch.device("cuda")

@pytest.fixture(scope="session")
def test_image():
    """Create standard test image"""
    from PIL import Image
    return Image.new('RGB', (224, 224), color='blue')

@pytest.fixture(scope="function")
def cleanup_gpu():
    """Cleanup GPU memory after each test"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

@pytest.fixture(scope="session")
def amdgpu_family():
    """Get AMDGPU family"""
    return os.getenv("AMDGPU_FAMILIES", "")

@pytest.fixture(scope="session")
def is_strix(amdgpu_family):
    """Check if running on Strix"""
    return amdgpu_family in ["gfx1150", "gfx1151"]
```

**Use fixtures:**
```python
def test_with_fixtures(strix_device, test_image, cleanup_gpu):
    """Test using shared fixtures"""
    model = model.to(strix_device)
    # ... use test_image ...
    # cleanup_gpu runs automatically after test
```

---

## 📊 Performance Benchmarking

Create `tests/strix_ai/benchmarks/benchmark_template.py`:

```python
import pytest
import torch
import time
import numpy as np

class ModelBenchmark:
    """Base class for model benchmarking"""
    
    def __init__(self, model, input_shape):
        self.model = model.to('cuda').eval()
        self.input_shape = input_shape
    
    def warmup(self, iterations=10):
        """Warmup runs"""
        test_input = torch.randn(self.input_shape).to('cuda')
        for _ in range(iterations):
            with torch.no_grad():
                _ = self.model(test_input)
        torch.cuda.synchronize()
    
    def benchmark_latency(self, iterations=100):
        """Measure latency"""
        self.warmup()
        
        test_input = torch.randn(self.input_shape).to('cuda')
        latencies = []
        
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                _ = self.model(test_input)
            
            torch.cuda.synchronize()
            latencies.append(time.time() - start)
        
        return {
            'mean_ms': np.mean(latencies) * 1000,
            'std_ms': np.std(latencies) * 1000,
            'p50_ms': np.percentile(latencies, 50) * 1000,
            'p95_ms': np.percentile(latencies, 95) * 1000,
            'p99_ms': np.percentile(latencies, 99) * 1000,
        }
    
    def benchmark_throughput(self, duration_seconds=10):
        """Measure throughput"""
        self.warmup()
        
        test_input = torch.randn(self.input_shape).to('cuda')
        
        torch.cuda.synchronize()
        start = time.time()
        iterations = 0
        
        while time.time() - start < duration_seconds:
            with torch.no_grad():
                _ = self.model(test_input)
            iterations += 1
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        return {
            'throughput_fps': iterations / elapsed,
            'iterations': iterations,
            'duration': elapsed
        }

# Usage example
def test_vit_benchmark():
    from transformers import ViTForImageClassification
    
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    bench = ModelBenchmark(model, (1, 3, 224, 224))
    
    latency = bench.benchmark_latency()
    print(f"Latency: {latency['mean_ms']:.2f}ms (±{latency['std_ms']:.2f}ms)")
    print(f"P95: {latency['p95_ms']:.2f}ms, P99: {latency['p99_ms']:.2f}ms")
    
    throughput = bench.benchmark_throughput()
    print(f"Throughput: {throughput['throughput_fps']:.2f} FPS")
```

---

## 🎬 Running Tests

### **Single Test**
```bash
pytest tests/strix_ai/test_clip_quick.py::test_clip_image_text -v -s
```

### **Full Suite**
```bash
pytest tests/strix_ai/ -v
```

### **By Category**
```bash
pytest tests/strix_ai/vlm/ -v          # VLM tests only
pytest tests/strix_ai/vit/ -v          # ViT tests only
pytest tests/strix_ai/cv/ -v           # CV tests only
```

### **With Performance Output**
```bash
pytest tests/strix_ai/ -v -s --durations=10
```

### **Skip Slow Tests**
```bash
pytest tests/strix_ai/ -v -m "not slow"
```

---

## ⚙️ Configuration

Create `pytest.ini` in project root:

```ini
[pytest]
markers =
    strix: Tests specific to Strix platforms
    vlm: Vision Language Model tests
    vit: Vision Transformer tests
    cv: Computer Vision tests
    slow: Tests that take > 30 seconds
    quick: Quick smoke tests
    windows: Windows-specific tests
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

**Use markers:**
```python
@pytest.mark.strix
@pytest.mark.vlm
@pytest.mark.slow
def test_llava_large_model():
    """Test marked as strix, vlm, and slow"""
    pass
```

**Run specific markers:**
```bash
pytest -m "strix and vlm" -v          # Strix VLM tests only
pytest -m "strix and not slow" -v     # Strix tests, skip slow ones
```

---

## 📦 Requirements File

Create `requirements_strix_ai.txt`:

```txt
# Core ML frameworks
torch>=2.1.0
torchvision>=0.16.0

# Transformers & Models
transformers>=4.35.0
accelerate>=0.24.0
timm>=0.9.0

# Object Detection
ultralytics>=8.0.0

# Quantization
bitsandbytes>=0.41.0
optimum>=1.14.0

# ONNX Runtime
onnxruntime-rocm>=1.16.0

# Computer Vision
opencv-python>=4.8.0
pillow>=10.0.0

# Utilities
numpy>=1.24.0
einops>=0.7.0
pytest>=7.4.0
pytest-timeout>=2.2.0
```

**Install:**
```bash
pip install -r requirements_strix_ai.txt
```

---

## 🚨 Common Issues & Fixes

### **Issue 1: Model Download Hangs**
```bash
# Set Hugging Face cache
export HF_HOME=/path/to/large/disk/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
```

### **Issue 2: Out of Memory**
```python
# Use smaller batch size or model variant
model = AutoModel.from_pretrained("model-name", torch_dtype=torch.float16)

# Or use quantization
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModel.from_pretrained("model-name", quantization_config=quantization_config)
```

### **Issue 3: Slow First Run**
```python
# Models compile kernels on first run - warmup!
def warmup_model(model, input_shape, iterations=10):
    dummy_input = torch.randn(input_shape).to('cuda')
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()
```

---

## ✅ Checklist for New Tests

When adding a new AI test:

- [ ] Test runs on both gfx1150 and gfx1151
- [ ] Test respects `AMDGPU_FAMILIES` environment variable
- [ ] Test uses `@pytest.mark.skipif` for platform filtering
- [ ] Test includes proper cleanup (GPU memory)
- [ ] Test has timeout (for CI)
- [ ] Test documents expected performance
- [ ] Test has failure message explaining issue
- [ ] Test is added to appropriate category directory
- [ ] Test uses shared fixtures where possible

---

## 🎯 Next Steps

1. **Start with Template 1** (CLIP) - easiest to implement
2. **Verify on hardware** - run on actual Strix device
3. **Measure baseline** - record performance metrics
4. **Expand coverage** - add more models
5. **Optimize** - tune for iGPU characteristics
6. **Document** - share results with team

---

## 📚 Resources

- **Full Test Plan**: [STRIX_AI_ML_TEST_PLAN.md](./STRIX_AI_ML_TEST_PLAN.md)
- **Strix Testing Guide**: [STRIX_TESTING_GUIDE.md](./STRIX_TESTING_GUIDE.md)
- **Hugging Face Models**: https://huggingface.co/models
- **Ultralytics YOLO**: https://docs.ultralytics.com
- **PyTorch Vision**: https://pytorch.org/vision/stable/index.html

---

**Ready to start? Copy a template, run it, and iterate!** 🚀

