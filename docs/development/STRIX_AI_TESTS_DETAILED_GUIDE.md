# Strix AI Tests - Detailed Technical Guide

## Overview

This document provides an in-depth explanation of the `strix_ai_tests.yml` GitHub Actions workflow, detailing what each test category does internally, what it evaluates, and how it validates AI/ML workloads on AMD Strix GPUs (gfx1150, gfx1151).

**Workflow File**: `.github/workflows/strix_ai_tests.yml`  
**Test Location**: `tests/strix_ai/`  
**Target Hardware**: AMD Strix Point (gfx1150), Strix Halo (gfx1151)  
**Container**: `rocm/pytorch:latest`

---

## Table of Contents

1. [Workflow Architecture](#workflow-architecture)
2. [Test Categories](#test-categories)
3. [VLM Tests (Vision Language Models)](#vlm-tests)
4. [VLA Tests (Vision Language Action)](#vla-tests)
5. [ViT Tests (Vision Transformers)](#vit-tests)
6. [CV Tests (Computer Vision)](#cv-tests)
7. [Optimization Tests](#optimization-tests)
8. [Profiling Tests](#profiling-tests)
9. [Quick Smoke Tests](#quick-smoke-tests)
10. [Environment & Configuration](#environment--configuration)
11. [Success Criteria](#success-criteria)

---

## Workflow Architecture

### Trigger Mechanisms

```yaml
on:
  push:
    branches:
      - 'users/*/strix_*'      # Any user's strix branch
      - 'main'
      - 'develop'
    paths:
      - 'tests/strix_ai/**'
      - '.github/workflows/strix_ai*.yml'
      
  pull_request:
    branches:
      - 'main'
      - 'develop'
      
  workflow_dispatch:
    # Manual trigger with parameters
```

**What it does:**
- **Automatic**: Triggers on push/PR to Strix-related branches when test files change
- **Path-filtered**: Only runs when `tests/strix_ai/` or workflow files are modified
- **Manual**: Can be triggered with custom parameters (platform, GPU variant, test category)

### Workflow Parameters

| Parameter | Options | Default | Purpose |
|-----------|---------|---------|---------|
| `platform` | linux, windows | linux | Target OS platform |
| `strix_variant` | gfx1150, gfx1151 | gfx1151 | Strix GPU architecture |
| `test_category` | all, vlm, vla, vit, cv, optimization, profiling, quick | quick | Which tests to run |
| `test_type` | smoke, quick, full | quick | Test execution depth |

### Container Infrastructure

**Linux Container:**
```yaml
container:
  image: rocm/pytorch:latest
  options: '--ipc host --group-add video --device /dev/kfd --device /dev/dri --group-add 110 --user 0:0'
```

**What's included:**
- ✅ ROCm 6.x runtime
- ✅ PyTorch 2.x with ROCm backend
- ✅ torchvision
- ✅ CUDA API compatibility layer (HIP)
- ✅ Python 3.10+
- ✅ NumPy, Pillow, OpenCV basics

**What gets installed:**
- transformers (Hugging Face)
- accelerate (model loading optimization)
- ultralytics (YOLO)
- opencv-python (computer vision)
- timm (PyTorch Image Models)
- einops (tensor operations)
- pytest framework

---

## Test Categories

### Workflow Steps Overview

```
1. Checkout Repository
2. Display System Info (Python, GPU, ROCm)
3. Check ROCm/GPU
4. Install AI/ML Dependencies
5. Verify Dependencies
6. Run Test Category:
   ├── VLM Tests
   ├── VLA Tests
   ├── ViT Tests
   ├── CV Tests
   ├── Optimization Tests
   └── Profiling Tests
7. Display Test Results XML
8. Archive Test Results
9. Test Summary
```

---

## VLM Tests (Vision Language Models)

**Test Files**: `tests/strix_ai/vlm/test_clip.py`  
**Workflow Step**: "Run Strix AI Tests - VLM"  
**Markers**: `@pytest.mark.vlm`, `@pytest.mark.strix`

### What VLM Tests Do Internally

#### 1. **CLIP Model Loading & Inference**

```python
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

**Model Architecture:**
- **Vision Encoder**: ViT-B/32 (12 layers, 768 hidden dim, 12 attention heads)
- **Text Encoder**: Transformer (12 layers, 512 hidden dim)
- **Image Size**: 224x224 pixels
- **Patch Size**: 32x32 (7x7 patches)
- **Parameters**: ~151M total

**What it evaluates:**

1. **Model Loading**
   - Can Strix GPU load 151M parameter model?
   - Memory allocation success (~600MB)
   - Model transfer to GPU device

2. **Image Processing**
   - Image preprocessing (resize, normalize)
   - Tensor conversion and GPU transfer
   - Input shape validation: `[batch, 3, 224, 224]`

3. **Text Processing**
   - Tokenization of text inputs
   - Token embeddings
   - Attention mask generation
   - Input shape: `[batch, max_length]`

4. **Inference Execution**
   ```python
   outputs = model(pixel_values=images, input_ids=text_tokens)
   logits_per_image = outputs.logits_per_image  # [batch_images, batch_texts]
   probs = logits_per_image.softmax(dim=1)      # Similarity scores
   ```

5. **Zero-Shot Classification**
   - Image-text similarity computation
   - Cosine similarity between embeddings
   - Probability distribution over text labels
   - Classification accuracy

**Performance Metrics:**
- ✅ Inference time per image
- ✅ GPU memory usage
- ✅ Throughput (images/second)
- ✅ Similarity score accuracy
- ✅ Model outputs correctness

**Success Criteria:**
- Model loads without OOM errors
- Inference completes without crashes
- Output shapes match expected dimensions
- Similarity scores are reasonable (0-1 range)
- GPU utilization is detected

#### 2. **Image-Text Matching Test**

**What it evaluates:**
- Multi-modal understanding (vision + language)
- Semantic similarity computation
- Cross-modal attention mechanisms

**Example Validation:**
```
Input: Image of a cat + ["a cat", "a dog", "a car"]
Expected: Highest score for "a cat"
Validates: Model correctly matches image to text
```

---

## VLA Tests (Vision Language Action)

**Test Files**: `tests/strix_ai/vla/test_action_prediction.py`  
**Workflow Step**: "Run Strix AI Tests - VLA"  
**Markers**: `@pytest.mark.vla`, `@pytest.mark.strix`

### What VLA Tests Do Internally

#### 1. **Action Recognition from Visual Input**

**Model Architecture** (varies by test):
- Vision encoder (ResNet, ViT, or Swin)
- Action classifier head
- Temporal modeling (for video)

**What it evaluates:**

1. **Visual Feature Extraction**
   - Extract features from video frames/images
   - Spatial feature maps
   - Temporal aggregation (if applicable)

2. **Action Classification**
   ```python
   features = vision_encoder(frames)           # Extract visual features
   action_logits = classifier(features)        # Predict action class
   action_probs = action_logits.softmax(dim=1) # Get probabilities
   predicted_action = action_probs.argmax()    # Select most likely action
   ```

3. **Action Vocabulary**
   - Predefined set of actions (e.g., "walk", "run", "sit", "jump")
   - Multi-class classification
   - Top-K accuracy evaluation

**Performance Metrics:**
- ✅ Action recognition accuracy
- ✅ Inference latency per frame
- ✅ Temporal consistency (for video)
- ✅ GPU memory efficiency

**Success Criteria:**
- Model loads and runs on Strix GPU
- Action predictions are sensible
- Output probabilities sum to 1.0
- No GPU memory overflows

#### 2. **Vision-Language Action Grounding**

**What it evaluates:**
- Understanding natural language action commands
- Visual grounding (localizing actions in images)
- Multi-modal reasoning

**Example:**
```
Input: Image + "person jumping"
Task: Verify model can ground the action in visual space
Validates: Cross-modal action understanding
```

---

## ViT Tests (Vision Transformers)

**Test Files**: `tests/strix_ai/vit/test_vit_base.py`  
**Workflow Step**: "Run Strix AI Tests - ViT"  
**Markers**: `@pytest.mark.vit`, `@pytest.mark.strix`

### What ViT Tests Do Internally

#### 1. **ViT-Base Model Architecture**

```python
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
```

**Architecture Details:**
- **Model**: ViT-Base/16
- **Layers**: 12 transformer encoder layers
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Patch Size**: 16x16 (14x14 patches for 224x224 image)
- **Parameters**: ~86M
- **Output**: 1000 ImageNet classes

**What it evaluates:**

1. **Image Patching & Embedding**
   ```
   Input: [batch, 3, 224, 224]
   ↓ Patch Extraction (16x16 patches)
   ↓ [batch, 196, 768] (14x14 = 196 patches)
   ↓ Add CLS token + Position Embeddings
   ↓ [batch, 197, 768] (196 patches + 1 CLS token)
   ```

2. **Transformer Encoder Processing**
   ```python
   for layer in transformer_layers:
       # Multi-head self-attention
       attention_output = multi_head_attention(
           query=hidden_states,
           key=hidden_states,
           value=hidden_states
       )
       
       # Feed-forward network
       hidden_states = ffn(attention_output + hidden_states)
   ```

   **Each layer evaluates:**
   - ✅ Self-attention computation (Q, K, V matrices)
   - ✅ Attention weight calculation
   - ✅ Multi-head attention aggregation
   - ✅ Feed-forward network (2-layer MLP)
   - ✅ Layer normalization
   - ✅ Residual connections

3. **Classification Head**
   ```python
   cls_token = hidden_states[:, 0, :]  # Extract CLS token
   logits = classification_head(cls_token)  # [batch, 1000]
   predicted_class = logits.argmax(dim=-1)
   ```

**Performance Metrics:**
- ✅ Inference time per image
- ✅ Throughput (images/second)
- ✅ GPU memory usage (~350MB for ViT-Base)
- ✅ Attention computation efficiency
- ✅ Batch processing capability

**Success Criteria:**
- Model loads without memory errors
- Forward pass completes successfully
- Output shape: `[batch, 1000]`
- Predicted class is valid (0-999)
- Attention mechanisms work correctly

#### 2. **Batch Inference Test**

**What it evaluates:**
```python
batch_sizes = [1, 2, 4, 8, 16]
for bs in batch_sizes:
    images = [Image.new('RGB', (224, 224)) for _ in range(bs)]
    outputs = model(images)
    # Measure throughput and memory
```

- **Memory Scaling**: How memory usage scales with batch size
- **Throughput**: Images processed per second
- **Efficiency**: Optimal batch size for Strix GPU
- **Stability**: No OOM errors at reasonable batch sizes

#### 3. **Throughput Benchmark**

**What it evaluates:**
```python
# Warmup
for _ in range(10):
    _ = model(test_image)

# Benchmark
start = time.time()
for _ in range(100):
    _ = model(test_image)
torch.cuda.synchronize()
end = time.time()

throughput = 100 / (end - start)  # images/second
```

- **Target**: >30 FPS for ViT-Base on Strix
- **Latency**: Per-image inference time
- **Consistency**: Variance in inference times
- **GPU Utilization**: Effective use of compute units

---

## CV Tests (Computer Vision)

**Test Files**: `tests/strix_ai/cv/test_yolo.py`  
**Workflow Step**: "Run Strix AI Tests - CV"  
**Markers**: `@pytest.mark.cv`, `@pytest.mark.strix`

### What CV Tests Do Internally

#### 1. **YOLO Object Detection**

```python
model = YOLO("yolov8n.pt")  # YOLOv8 Nano
results = model.predict(image, device=0)
```

**YOLOv8n Architecture:**
- **Type**: Single-stage object detector
- **Backbone**: CSPDarknet with C2f modules
- **Neck**: PAN (Path Aggregation Network)
- **Head**: Decoupled detection head
- **Parameters**: ~3.2M (Nano variant)
- **Input Size**: 640x640 (default)

**What it evaluates:**

1. **Image Preprocessing**
   ```
   Input: Variable size image (e.g., 1920x1080)
   ↓ Letterbox resize (maintain aspect ratio)
   ↓ Pad to square (640x640)
   ↓ Normalize [0-1]
   ↓ Tensor conversion [1, 3, 640, 640]
   ```

2. **Feature Extraction (Backbone)**
   ```python
   # Multi-scale feature extraction
   P3 = backbone_layer3(input)   # 80x80 feature map
   P4 = backbone_layer4(P3)      # 40x40 feature map
   P5 = backbone_layer5(P4)      # 20x20 feature map
   ```

   **Evaluates:**
   - ✅ Convolutional operations on GPU
   - ✅ Feature pyramid generation
   - ✅ Multi-scale representation

3. **Neck (Feature Fusion)**
   ```python
   # Top-down pathway
   N5 = P5
   N4 = upsample(N5) + P4
   N3 = upsample(N4) + P3
   
   # Bottom-up pathway
   N3_out = N3
   N4_out = downsample(N3_out) + N4
   N5_out = downsample(N4_out) + N5
   ```

4. **Detection Head**
   ```python
   for feature_map in [N3_out, N4_out, N5_out]:
       # Predict bounding boxes
       bbox_pred = bbox_head(feature_map)    # [x, y, w, h]
       
       # Predict objectness
       obj_pred = obj_head(feature_map)      # confidence
       
       # Predict class probabilities
       cls_pred = cls_head(feature_map)      # 80 COCO classes
   ```

5. **Post-Processing**
   ```python
   # Non-Maximum Suppression (NMS)
   boxes = decode_boxes(bbox_pred)
   scores = obj_pred * cls_pred
   
   # Filter by confidence threshold
   mask = scores > conf_threshold  # e.g., 0.25
   
   # Apply NMS
   keep_indices = nms(boxes[mask], scores[mask], iou_threshold=0.45)
   final_detections = boxes[keep_indices]
   ```

**Performance Metrics:**
- ✅ Detection accuracy (mAP - mean Average Precision)
- ✅ Inference speed (FPS)
- ✅ Number of detections per image
- ✅ Box coordinate accuracy
- ✅ Class prediction accuracy
- ✅ NMS efficiency

**Success Criteria:**
- Model loads and initializes
- Inference runs without errors
- Detections returned in correct format:
  ```python
  Detection {
      box: [x1, y1, x2, y2],
      confidence: float (0-1),
      class_id: int (0-79),
      class_name: str
  }
  ```
- Target: >15 FPS for real-time performance
- Memory usage: <1GB for YOLOv8n

#### 2. **Real-Time Video Detection Test**

**What it evaluates:**
```python
# Simulate video stream
for frame in video_frames:
    start = time.time()
    results = model.predict(frame, device=0)
    fps = 1.0 / (time.time() - start)
    
    # Validate real-time capability
    assert fps >= 15.0, "Not meeting real-time requirements"
```

- **Real-time performance**: Consistent >15 FPS
- **Frame-to-frame consistency**: Stable detections
- **Memory stability**: No memory leaks over time
- **GPU thermal**: Sustained workload handling

---

## Optimization Tests

**Test Files**: `tests/strix_ai/optimization/test_quantization.py`  
**Workflow Step**: "Run Strix AI Tests - Optimization"  
**Markers**: `@pytest.mark.optimization`, `@pytest.mark.strix`

### What Optimization Tests Do Internally

#### 1. **FP16 (Half Precision) Conversion**

```python
model_fp32 = Model.from_pretrained("model-name")
model_fp16 = model_fp32.half()  # Convert to float16
```

**What it evaluates:**

1. **Memory Reduction**
   ```
   FP32: 32 bits per parameter
   FP16: 16 bits per parameter
   Reduction: 50% memory savings
   
   Example:
   Model with 100M parameters
   FP32: 400MB
   FP16: 200MB
   ```

2. **Computation Speed**
   - Tensor cores utilization (if available)
   - Matrix multiplication speedup
   - Memory bandwidth efficiency

3. **Accuracy Impact**
   ```python
   output_fp32 = model_fp32(input)
   output_fp16 = model_fp16(input)
   
   # Measure accuracy degradation
   mae = torch.abs(output_fp32 - output_fp16).mean()
   relative_error = mae / torch.abs(output_fp32).mean()
   
   # Acceptable: relative_error < 1%
   ```

**Performance Metrics:**
- ✅ Memory savings percentage
- ✅ Inference speedup (FP16 vs FP32)
- ✅ Accuracy degradation (mean absolute error)
- ✅ Throughput improvement

**Success Criteria:**
- FP16 conversion succeeds
- Memory usage reduced by ~50%
- Accuracy loss < 1%
- Inference speed improves or maintains

#### 2. **INT8 Quantization**

```python
import torch.quantization as quant

# Post-training static quantization
model_int8 = quant.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

**What it evaluates:**

1. **Quantization Process**
   ```
   FP32 → INT8 conversion:
   1. Calibration: Collect activation statistics
   2. Scale calculation: Determine quantization parameters
   3. Weight quantization: Convert weights to INT8
   4. Activation quantization: Dynamic quantization during inference
   ```

2. **Model Size Reduction**
   ```
   FP32: 32 bits per parameter
   INT8: 8 bits per parameter
   Reduction: 75% size savings
   
   Example:
   Model with 100M parameters
   FP32: 400MB
   INT8: 100MB
   ```

3. **Inference Performance**
   ```python
   # Benchmark quantized vs original
   time_fp32 = benchmark(model_fp32, input)
   time_int8 = benchmark(model_int8, input)
   speedup = time_fp32 / time_int8
   
   # Target: 2-4x speedup on edge devices
   ```

4. **Accuracy Validation**
   ```python
   # Compare outputs
   output_fp32 = model_fp32(test_dataset)
   output_int8 = model_int8(test_dataset)
   
   # Measure accuracy drop
   accuracy_fp32 = compute_accuracy(output_fp32, labels)
   accuracy_int8 = compute_accuracy(output_int8, labels)
   
   # Acceptable: <3% accuracy drop
   ```

**Performance Metrics:**
- ✅ Model size reduction: ~75%
- ✅ Inference speedup: 2-4x
- ✅ Accuracy impact: <3% degradation
- ✅ Memory footprint: ~25% of original

**Success Criteria:**
- Quantization completes without errors
- Model size significantly reduced
- Inference runs on INT8 operations
- Accuracy degradation acceptable
- Performance improves for edge deployment

#### 3. **ONNX Export Test**

```python
import torch.onnx

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

**What it evaluates:**
- ✅ Model convertibility to ONNX format
- ✅ Operator compatibility
- ✅ Dynamic shape support
- ✅ ONNX Runtime inference
- ✅ Cross-platform deployment readiness

---

## Profiling Tests

**Test Files**: `tests/strix_ai/profiling/test_pytorch_profiling.py`, `test_ai_workload_profiling.py`  
**Workflow Step**: "Run Strix AI Tests - ROCProfiler Integration"  
**Markers**: `@pytest.mark.profiling`, `@pytest.mark.strix`

### What Profiling Tests Do Internally

#### 1. **PyTorch Built-in Profiler**

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    output = model(input)
    torch.cuda.synchronize()

# Analyze results
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**What it evaluates:**

1. **GPU Kernel Execution**
   - Kernel launch times
   - Kernel execution durations
   - GPU occupancy
   - Memory transfers (Host ↔ Device)

2. **Operation Breakdown**
   ```
   Top Operations by GPU Time:
   ├── aten::addmm          (Matrix multiplication)  67.8ms
   ├── aten::layer_norm     (Normalization)          45.6ms
   ├── aten::softmax        (Attention softmax)      23.4ms
   ├── aten::copy_          (Memory transfer)        18.9ms
   └── aten::mul            (Element-wise ops)       12.3ms
   ```

3. **Performance Metrics**
   - **Total GPU Time**: Sum of all CUDA operations
   - **Total CPU Time**: Host-side overhead
   - **GPU Utilization**: Percentage of time GPU is active
   - **Memory Bandwidth**: Data transfer efficiency

4. **Bottleneck Identification**
   ```python
   # Identify slow operations
   for event in prof.key_averages():
       if event.cuda_time_total > threshold:
           print(f"Bottleneck: {event.key}")
           print(f"  GPU Time: {event.cuda_time_total / 1000:.2f}ms")
           print(f"  Calls: {event.count}")
           print(f"  Avg: {event.cuda_time_total / event.count / 1000:.2f}ms")
   ```

**Success Criteria:**
- Profiler captures operations successfully
- GPU time is measurable
- Operation breakdown makes sense
- No profiling overhead crashes

#### 2. **ROCProfiler CLI Integration**

```bash
rocprof --stats -o results.csv python model_inference.py
```

**What it evaluates:**

1. **HIP Kernel Tracing**
   - Kernel dispatch information
   - Kernel arguments
   - Kernel execution times
   - Device memory accesses

2. **Hardware Counters** (if available)
   - Compute unit utilization
   - Memory controller activity
   - Cache hit rates
   - Instruction throughput

3. **API Call Tracing**
   - HIP API calls
   - Memory allocation/deallocation
   - Kernel launches
   - Stream synchronization

**Success Criteria:**
- rocprof runs without errors
- CSV output generated
- Kernel information captured
- Statistics are reasonable

#### 3. **AI Model Profiling**

**CLIP Profiling:**
```python
# Profile vision encoder
with profiler:
    vision_features = model.vision_model(pixel_values)

# Profile text encoder
with profiler:
    text_features = model.text_model(input_ids)

# Profile similarity computation
with profiler:
    logits = text_features @ vision_features.T
```

**ViT Profiling:**
```python
# Profile patch embedding
with profiler:
    embeddings = model.embeddings(pixel_values)

# Profile each transformer layer
for i, layer in enumerate(model.encoder.layer):
    with profiler:
        hidden_states = layer(hidden_states)
```

**YOLO Profiling:**
```python
# Profile backbone
with profiler:
    features = model.backbone(images)

# Profile neck
with profiler:
    fused_features = model.neck(features)

# Profile detection head
with profiler:
    predictions = model.head(fused_features)
```

**Performance Metrics:**
- ✅ Per-layer execution time
- ✅ Memory usage per operation
- ✅ GPU utilization per stage
- ✅ Bottleneck identification

---

## Quick Smoke Tests

**Workflow Step**: "Run Strix AI Tests - Quick Smoke Tests"  
**Markers**: `@pytest.mark.quick`, `@pytest.mark.p0`

### What Quick Tests Do Internally

**Purpose**: Fast validation (< 2 minutes total)

#### 1. **GPU Detection Test**

```python
def test_gpu_available():
    assert torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0)
    assert "gfx115" in device_name or "strix" in device_name.lower()
```

**Evaluates:**
- ✅ ROCm/HIP runtime functional
- ✅ GPU device enumeration
- ✅ Correct Strix GPU detected

#### 2. **Basic Tensor Operations**

```python
def test_basic_tensor_ops():
    x = torch.randn(100, 100, device='cuda')
    y = torch.randn(100, 100, device='cuda')
    
    # Matrix multiplication
    z = torch.matmul(x, y)
    
    # Element-wise operations
    w = x + y
    v = torch.relu(z)
    
    torch.cuda.synchronize()
```

**Evaluates:**
- ✅ GPU memory allocation
- ✅ Basic GPU operations (matmul, add, relu)
- ✅ Device synchronization

#### 3. **Simple Model Inference**

```python
def test_simple_model():
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to('cuda')
    
    input = torch.randn(32, 100, device='cuda')
    output = model(input)
    
    assert output.shape == (32, 10)
```

**Evaluates:**
- ✅ Model transfer to GPU
- ✅ Forward pass execution
- ✅ Output correctness

---

## Environment & Configuration

### Environment Variables

```yaml
env:
  AMDGPU_FAMILIES: ${{ inputs.strix_variant || 'gfx1151' }}
  TEST_TYPE: ${{ inputs.test_type || 'quick' }}
  TEST_CATEGORY: ${{ inputs.test_category || 'all' }}
  PLATFORM: ${{ inputs.platform || 'linux' }}
  PYTHONUNBUFFERED: 1
```

**What each variable controls:**

1. **AMDGPU_FAMILIES**
   - Used by tests to validate correct GPU
   - Skips tests if not on target hardware
   - Values: `gfx1150` (Strix Point), `gfx1151` (Strix Halo)

2. **TEST_TYPE**
   - `smoke`: Minimal tests (< 1 min)
   - `quick`: Reduced test set (< 10 min)
   - `full`: Complete test suite (> 30 min)

3. **TEST_CATEGORY**
   - Selects which test group to run
   - `all`: Runs all test categories
   - Specific: Runs only selected category

4. **PYTHONUNBUFFERED**
   - Forces Python to output immediately
   - Ensures logs appear in real-time in GitHub Actions

### GPU Access Configuration

```yaml
container:
  options: '--ipc host --group-add video --device /dev/kfd --device /dev/dri --group-add 110 --user 0:0'
```

**What each option does:**

- `--ipc host`: Share host IPC namespace (for GPU memory)
- `--group-add video`: Add video group for GPU access
- `--device /dev/kfd`: AMD KFD (Kernel Fusion Driver) device
- `--device /dev/dri`: Direct Rendering Infrastructure devices
- `--group-add 110`: Render group for GPU access
- `--user 0:0`: Run as root (necessary for GPU access)

---

## Success Criteria

### Per-Test Success Criteria

| Test Category | Success Criteria |
|---------------|------------------|
| **VLM** | • Model loads without OOM<br>• Inference completes<br>• Similarity scores in valid range<br>• Output shapes correct |
| **VLA** | • Action predictions reasonable<br>• Multi-modal processing works<br>• No GPU errors |
| **ViT** | • >30 FPS throughput<br>• Memory usage <500MB<br>• Batch processing stable<br>• Attention mechanisms functional |
| **CV** | • >15 FPS real-time detection<br>• Detections in valid format<br>• NMS operates correctly<br>• Memory <1GB |
| **Optimization** | • FP16: 50% memory savings, <1% accuracy loss<br>• INT8: 75% size reduction, <3% accuracy loss<br>• ONNX export successful |
| **Profiling** | • Profiler captures operations<br>• GPU times measurable<br>• Bottlenecks identifiable<br>• No profiling crashes |
| **Quick** | • All complete in <2 minutes<br>• GPU detected correctly<br>• Basic ops functional |

### Overall Workflow Success

✅ **Pass Criteria:**
- All enabled tests pass
- No unhandled exceptions
- GPU properly utilized
- Test results XML generated

⚠️ **Warning (Non-Blocking):**
- Some tests skipped (dependencies missing)
- Performance slightly below target
- Non-critical warnings in logs

❌ **Fail Criteria:**
- GPU not detected
- Critical test failures
- Out of memory errors
- Workflow timeout (>120 min)

---

## Test Result Artifacts

### JUnit XML Output

Each test category generates XML:
```xml
<testsuite name="pytest" tests="10" failures="0" errors="0" time="45.123">
  <testcase classname="test_clip" name="test_clip_inference" time="4.567">
    <properties>
      <property name="model" value="openai/clip-vit-base-patch32"/>
      <property name="gpu_time_ms" value="234.56"/>
      <property name="throughput_fps" value="42.3"/>
    </properties>
  </testcase>
</testsuite>
```

### Metrics Captured

- **Timing**: Per-test execution time
- **Performance**: FPS, throughput, latency
- **Memory**: Peak GPU memory usage
- **Accuracy**: Model output correctness
- **Hardware**: GPU utilization, kernel times

---

## Troubleshooting Test Failures

### Common Issues

1. **GPU Not Available**
   - Check: ROCm installed, GPU visible (`rocm-smi`)
   - Check: Device access permissions
   - Check: Container GPU passthrough

2. **Out of Memory**
   - Use smaller models or batch sizes
   - Enable FP16/INT8 quantization
   - Reduce input resolution

3. **Model Download Failures**
   - Check: Internet connectivity
   - Check: Hugging Face Hub access
   - Set: `HF_HOME` for cache location

4. **Profiling Errors**
   - Check: ROCProfiler tools installed
   - Check: Tool versions compatible
   - Fallback: PyTorch built-in profiler

---

## Summary

This workflow provides comprehensive validation of AI/ML workloads on Strix GPUs by:

1. **Testing Multiple Model Types**: VLM, VLA, ViT, YOLO
2. **Evaluating Performance**: Throughput, latency, memory
3. **Validating Optimization**: FP16, INT8, ONNX
4. **Profiling Execution**: Bottleneck identification
5. **Ensuring Correctness**: Output validation, accuracy checks

All tests run in a reproducible container environment with proper GPU access and comprehensive result reporting.

