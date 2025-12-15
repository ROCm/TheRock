"""
ROCProfiler v3 tests for Strix AI/ML workloads
Uses rocprofv3 with Strix-specific flags for comprehensive GPU profiling
Command format: rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --output-format pftrace -d ./v3_traces -- python3 app.py

Note: --rccl-trace is NOT used for Strix (single iGPU, RCCL excluded from Strix builds)
"""

import pytest
import os
import subprocess
import sys
import tempfile
import json
from pathlib import Path

# Optional imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from transformers import CLIPModel, CLIPProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def check_rocprofv3_available():
    """Check if rocprofv3 (ROCProfiler v3) is available for Strix profiling"""
    try:
        result = subprocess.run(
            ["rocprofv3", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_rocprofv3(script_path, output_dir, timeout=120):
    """
    Run rocprofv3 with Strix-specific profiling flags
    
    Command format:
    rocprofv3 --hip-trace --kernel-trace --memory-copy-trace 
              --output-format pftrace -d ./v3_traces -- python3 app.py
    
    Note: --rccl-trace is NOT used for Strix:
      - Strix is a single iGPU (no multi-GPU communication)
      - RCCL is excluded from Strix builds (Issue #150)
    """
    cmd = [
        "rocprofv3",
        "--hip-trace",           # Trace HIP API calls
        "--kernel-trace",        # Trace kernel launches
        "--memory-copy-trace",   # Trace memory copy operations
        # NOTE: --rccl-trace omitted - RCCL excluded from Strix builds
        "--output-format", "pftrace",  # Output format for performance traces
        "-d", str(output_dir),   # Output directory
        "--",                    # Separator
        sys.executable,          # python3
        str(script_path)         # app.py
    ]
    
    print(f"Running rocprofv3 command:")
    print(f"{' '.join(cmd)}\n")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    
    return result


class SimpleNet(nn.Module):
    """Simple neural network for profiling tests"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.p0
class TestStrixRocprofv3:
    """Test rocprofv3 profiling on Strix GPUs"""
    
    def test_rocprofv3_installation(self):
        """Verify rocprofv3 is installed and available"""
        has_rocprofv3 = check_rocprofv3_available()
        
        assert has_rocprofv3, \
            "rocprofv3 not found. Install rocprofiler-sdk for Strix profiling"
        
        print(f"\n✓ rocprofv3 available")
        
        # Print version
        result = subprocess.run(
            ["rocprofv3", "--version"],
            capture_output=True,
            text=True
        )
        print(f"\nrocprofv3 version:\n{result.stdout}")
        
        print("\n✓ Strix profiling flags:")
        print("  --hip-trace          : Trace HIP API calls")
        print("  --kernel-trace       : Trace kernel launches")
        print("  --memory-copy-trace  : Trace memory copy operations")
        print("  --output-format pftrace : Performance trace format")
        print("\n⚠ Note: --rccl-trace NOT used (RCCL excluded from Strix builds)")
    
    def test_rocprofv3_pytorch_inference(self, strix_device, cleanup_gpu):
        """Profile PyTorch inference using rocprofv3 with Strix flags"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        if not check_rocprofv3_available():
            pytest.skip("rocprofv3 not available")
        
        print("\n" + "="*70)
        print("ROCProfiler v3: PyTorch Inference on Strix")
        print("="*70)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "pytorch_inference.py"
            output_dir = Path(tmpdir) / "v3_traces"
            output_dir.mkdir()
            
            # Create profiling script
            script_content = """
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device('cuda')
model = SimpleNet().to(device)
model.eval()

print("=== Starting profiled PyTorch inference ===")

# Warmup
for _ in range(5):
    x = torch.randn(32, 1024, device=device)
    with torch.no_grad():
        y = model(x)
torch.cuda.synchronize()

# Profiled inference runs
for i in range(20):
    x = torch.randn(32, 1024, device=device)
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()

print(f"✓ Completed 20 inference iterations")
print(f"✓ Output shape: {y.shape}")
print("=== Profiling completed ===")
"""
            script_path.write_text(script_content)
            
            # Run rocprofv3 with Strix flags
            result = run_rocprofv3(script_path, output_dir, timeout=120)
            
            # Check execution
            assert result.returncode == 0, \
                f"rocprofv3 failed: {result.stderr}"
            
            print("✓ rocprofv3 execution successful")
            print(f"\nScript output:\n{result.stdout}")
            
            # Check for output files
            trace_files = list(output_dir.glob("*"))
            assert len(trace_files) > 0, \
                f"No trace files generated in {output_dir}"
            
            print(f"\n✓ Generated {len(trace_files)} profiling trace file(s):")
            for f in trace_files[:5]:  # Show first 5 files
                print(f"  - {f.name} ({f.stat().st_size} bytes)")
            
            if len(trace_files) > 5:
                print(f"  ... and {len(trace_files) - 5} more files")
            
            print(f"\n✓ Profiling traces saved to: {output_dir}")
    
    @pytest.mark.vlm
    def test_rocprofv3_clip_inference(self, strix_device, cleanup_gpu):
        """Profile CLIP model inference using rocprofv3"""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            pytest.skip("PyTorch or Transformers not available")
        
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        if not check_rocprofv3_available():
            pytest.skip("rocprofv3 not available")
        
        print("\n" + "="*70)
        print("ROCProfiler v3: CLIP Model Profiling on Strix")
        print("="*70)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "clip_inference.py"
            output_dir = Path(tmpdir) / "v3_traces_clip"
            output_dir.mkdir()
            
            # Create CLIP profiling script
            script_content = """
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

print("=== Loading CLIP model ===")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to('cuda')
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# Create test image
image = Image.new('RGB', (224, 224), color='red')
texts = ["a red image", "a blue image"]

print("=== Starting profiled CLIP inference ===")

# Warmup
for _ in range(3):
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    torch.cuda.synchronize()

# Profiled inference runs
for i in range(10):
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
    torch.cuda.synchronize()

print(f"✓ Completed 10 CLIP inference iterations")
print(f"✓ Logits shape: {logits.shape}")
print("=== Profiling completed ===")
"""
            script_path.write_text(script_content)
            
            # Run rocprofv3
            result = run_rocprofv3(script_path, output_dir, timeout=180)
            
            # Check execution
            assert result.returncode == 0, \
                f"rocprofv3 failed: {result.stderr}"
            
            print("✓ rocprofv3 CLIP profiling successful")
            print(f"\nScript output:\n{result.stdout}")
            
            # Check for output files
            trace_files = list(output_dir.glob("*"))
            assert len(trace_files) > 0, \
                f"No trace files generated in {output_dir}"
            
            print(f"\n✓ Generated {len(trace_files)} CLIP profiling trace file(s):")
            for f in trace_files[:5]:
                print(f"  - {f.name} ({f.stat().st_size} bytes)")
            
            print(f"\n✓ CLIP profiling traces saved to: {output_dir}")
    
    @pytest.mark.quick
    def test_rocprofv3_simple_kernel(self, strix_device, cleanup_gpu):
        """Quick test of rocprofv3 with simple matrix multiplication"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        if not check_rocprofv3_available():
            pytest.skip("rocprofv3 not available")
        
        print("\n=== ROCProfiler v3: Quick Kernel Test ===")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "matmul.py"
            output_dir = Path(tmpdir) / "v3_traces_quick"
            output_dir.mkdir()
            
            # Simple matmul script
            script_content = """
import torch

print("=== Profiling matrix multiplication ===")
device = torch.device('cuda')

# Create matrices
A = torch.randn(512, 512, device=device)
B = torch.randn(512, 512, device=device)

# Warmup
for _ in range(5):
    C = torch.matmul(A, B)
torch.cuda.synchronize()

# Profiled operations
for i in range(10):
    C = torch.matmul(A, B)
    torch.cuda.synchronize()

print(f"✓ Matrix multiplication completed: {C.shape}")
"""
            script_path.write_text(script_content)
            
            # Run rocprofv3
            result = run_rocprofv3(script_path, output_dir, timeout=60)
            
            assert result.returncode == 0, \
                f"rocprofv3 failed: {result.stderr}"
            
            print("✓ Quick rocprofv3 test passed")
            
            # Verify trace files
            trace_files = list(output_dir.glob("*"))
            assert len(trace_files) > 0, "No trace files generated"
            
            print(f"✓ Generated {len(trace_files)} trace file(s) in {output_dir}")


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.p1
@pytest.mark.slow
class TestStrixRocprofv3Advanced:
    """Advanced rocprofv3 profiling tests for Strix"""
    
    def test_rocprofv3_memory_profiling(self, strix_device, cleanup_gpu):
        """Profile memory operations with rocprofv3 --memory-copy-trace"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        if not check_rocprofv3_available():
            pytest.skip("rocprofv3 not available")
        
        print("\n" + "="*70)
        print("ROCProfiler v3: Memory Operations Profiling")
        print("="*70)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "memory_ops.py"
            output_dir = Path(tmpdir) / "v3_traces_memory"
            output_dir.mkdir()
            
            # Memory operations script
            script_content = """
import torch

print("=== Profiling memory operations ===")
device = torch.device('cuda')

# Test various memory operations
for i in range(10):
    # Allocate
    A = torch.randn(1024, 1024, device=device)
    
    # Copy device to device
    B = A.clone()
    
    # Copy to CPU
    C = B.cpu()
    
    # Copy back to GPU
    D = C.to(device)
    
    torch.cuda.synchronize()

print("✓ Memory operations profiling completed")
"""
            script_path.write_text(script_content)
            
            # Run rocprofv3
            result = run_rocprofv3(script_path, output_dir, timeout=90)
            
            assert result.returncode == 0, \
                f"rocprofv3 failed: {result.stderr}"
            
            print("✓ Memory profiling successful")
            print(f"\nScript output:\n{result.stdout}")
            
            # Check for trace files with memory info
            trace_files = list(output_dir.glob("*"))
            assert len(trace_files) > 0, "No trace files generated"
            
            print(f"\n✓ Generated {len(trace_files)} memory trace file(s)")
            print("✓ Use --memory-copy-trace flag to analyze memory transfer patterns")

