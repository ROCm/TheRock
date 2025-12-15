# Strix ROCProfiler v3 (rocprofv3) Guide

## ⚡ **rocprofv3 for Strix GPU Profiling**

This guide explains how to use `rocprofv3` for profiling AI/ML workloads on AMD Strix GPUs (gfx1150, gfx1151).

### **Command Format**

```bash
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./v3_traces -- python3 app.py
```

**⚠ Note:** `--rccl-trace` is **NOT used for Strix** because:
- Strix is a **single iGPU** (no multi-GPU communication)
- **RCCL is excluded** from Strix builds ([Issue #150](https://github.com/ROCm/TheRock/issues/150))
- RCCL (ROCm Communication Collectives) is for multi-GPU scenarios only

## 🚀 **Quick Start**

### **1. Verify Installation**

```bash
# Check rocprofv3 is available
rocprofv3 --version

# Should output:
# ROCProfiler: v3.x.x
```

### **2. Run Strix Profiling Tests**

```bash
# All rocprofv3 tests
cd tests/strix_ai/profiling/
python3 -m pytest test_strix_rocprofv3.py -v -s

# Quick smoke test
python3 -m pytest test_strix_rocprofv3.py::TestStrixRocprofv3::test_rocprofv3_simple_kernel -v -s

# PyTorch inference profiling
python3 -m pytest test_strix_rocprofv3.py::TestStrixRocprofv3::test_rocprofv3_pytorch_inference -v -s

# CLIP model profiling
python3 -m pytest test_strix_rocprofv3.py::TestStrixRocprofv3::test_rocprofv3_clip_inference -v -s
```

### **3. Profile Your Own Scripts**

```bash
# Template for Strix
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d OUTPUT_DIR -- python3 YOUR_SCRIPT.py

# Example: Profile PyTorch training
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./strix_train_traces -- python3 train.py

# Example: Profile inference
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./strix_infer_traces -- python3 inference.py
```

## 📊 **rocprofv3 Flags Reference**

### **Strix-Recommended Flags**

| Flag | Description | Purpose |
|------|-------------|---------|
| `--hip-trace` | Trace HIP API calls | Captures hipMalloc, hipMemcpy, hipLaunchKernel, etc. |
| `--kernel-trace` | Trace GPU kernel launches | Records kernel execution times and parameters |
| `--memory-copy-trace` | Trace memory operations | Tracks host↔device and device↔device transfers |
| `--output-format pftrace` | Output format | Generates performance traces |
| `-d <directory>` | Output directory | Where to save trace files |
| `--` | Separator | Required before the python command |

**⚠ Flag NOT Used for Strix:**

| Flag | Why NOT Used |
|------|--------------|
| `~~--rccl-trace~~` | ❌ RCCL excluded from Strix builds ([Issue #150](https://github.com/ROCm/TheRock/issues/150))<br>❌ Strix is single iGPU (no multi-GPU comms)<br>❌ RCCL is for multi-GPU scenarios only |

### **Additional Useful Flags**

```bash
# Add HSA-level tracing (lower-level than HIP)
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --hsa-trace \                    # HSA API tracing
          --output-format pftrace \
          -d ./traces -- python3 app.py

# Filter specific HIP APIs
rocprofv3 --hip-trace \
          --hip-api-filter "hipMemcpy,hipLaunchKernel" \
          --output-format pftrace \
          -d ./traces -- python3 app.py
```

## 🔍 **Analyzing Trace Output**

### **Output Files**

After running `rocprofv3`, check the output directory:

```bash
ls -lh ./v3_traces/

# Typical output files:
# - *.pftrace           - Perfetto trace format
# - *.json              - JSON trace data
# - *_stats.txt         - Statistics summary
# - *_hip_api.txt       - HIP API call log
# - *_kernel_trace.txt  - Kernel execution log
```

### **View Traces**

**Option 1: Perfetto UI (Recommended)**
```bash
# Open in browser
https://ui.perfetto.dev

# Upload the .pftrace file
# Provides interactive timeline visualization
```

**Option 2: rocprof Tools**
```bash
# If JSON output available
python3 -m json.tool trace.json > trace_formatted.json

# View statistics
cat *_stats.txt
```

### **Key Metrics to Look For**

1. **Kernel Execution Times**
   - GPU kernel duration
   - Kernel launch overhead
   - Occupancy metrics

2. **Memory Transfer Times**
   - Host to Device (H2D) bandwidth
   - Device to Host (D2H) bandwidth
   - Device to Device (D2D) transfers

3. **API Call Overhead**
   - HIP API latency
   - Synchronization points
   - Memory allocation time

4. **RCCL Operations** (multi-GPU)
   - AllReduce, Broadcast times
   - Communication bandwidth

## 📝 **Example Workflows**

### **Profile PyTorch Model**

```python
# my_model.py
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
).to('cuda')

model.eval()

# Warmup
for _ in range(10):
    x = torch.randn(32, 1024, device='cuda')
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()

# Profiled section
for _ in range(100):
    x = torch.randn(32, 1024, device='cuda')
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()

print(f"Inference complete: {y.shape}")
```

```bash
# Profile it
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./my_model_traces -- python3 my_model.py
```

### **Profile CLIP Model**

```bash
# Run the built-in CLIP profiling test
python3 -m pytest test_strix_rocprofv3.py::TestStrixRocprofv3::test_rocprofv3_clip_inference -v -s

# Or profile your own CLIP script
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format pftrace -d ./clip_traces -- python3 my_clip_inference.py
```

## 🎯 **Best Practices for Strix Profiling**

### **1. Warmup Before Profiling**

```python
# Always warmup to avoid capturing initialization overhead
for _ in range(10):
    output = model(input)
    torch.cuda.synchronize()

# Now profile
for _ in range(100):
    output = model(input)
    torch.cuda.synchronize()
```

### **2. Use Synchronization**

```python
# Ensure GPU operations complete
torch.cuda.synchronize()

# Before exiting profiled section
```

### **3. Profile Representative Workloads**

```python
# Use realistic batch sizes
batch_size = 32  # Not 1

# Use realistic input sizes
input_shape = (32, 3, 224, 224)  # NCHW format
```

### **4. Profile Multiple Iterations**

```python
# Profile multiple runs for statistical reliability
num_iterations = 100  # Not just 1-2

for i in range(num_iterations):
    output = model(input)
    torch.cuda.synchronize()
```

## 🐛 **Troubleshooting**

### **rocprofv3 not found**

```bash
# Check ROCm installation
ls /opt/rocm/bin/rocprofv3

# Add to PATH if needed
export PATH=/opt/rocm/bin:$PATH

# Verify
rocprofv3 --version
```

### **No trace files generated**

```bash
# Check output directory exists
mkdir -p ./v3_traces

# Check permissions
chmod 755 ./v3_traces

# Run with verbose output
rocprofv3 --hip-trace --verbose -d ./v3_traces -- python3 app.py
```

### **Traces are empty or incomplete**

```python
# Ensure GPU synchronization in your script
torch.cuda.synchronize()  # Add this!

# Ensure you're actually using the GPU
assert torch.cuda.is_available()
model = model.to('cuda')
input = input.to('cuda')
```

## 📦 **Prerequisites**

- ✅ ROCm 6.2+ (includes rocprofiler-sdk)
- ✅ rocprofv3 installed (`rocprofv3 --version`)
- ✅ AMD Strix GPU (gfx1150 or gfx1151)
- ✅ PyTorch with ROCm backend
- ✅ Python 3.9+

## 🔗 **Resources**

- [ROCProfiler Documentation](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [Perfetto UI](https://ui.perfetto.dev)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Strix AI Tests](../README.md)

## 📧 **Support**

For Strix-specific profiling questions, see:
- [TheRock Issues](https://github.com/ROCm/TheRock/issues)
- [Strix Testing Guide](../../../docs/development/STRIX_TESTING_GUIDE.md)

