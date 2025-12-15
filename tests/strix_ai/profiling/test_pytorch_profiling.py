"""
ROCProfiler tests for PyTorch models on Strix
Tests basic profiling capabilities with PyTorch workloads
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


def check_rocprof_available():
    """Check if rocprof is available in the system"""
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


def check_rocprofiler_sdk():
    """Check if rocprofiler-sdk is available"""
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
@pytest.mark.p1
class TestPyTorchProfiling:
    """Test ROCProfiler integration with PyTorch on Strix"""
    
    def test_gpu_available(self, strix_device):
        """Verify GPU is available for profiling tests"""
        assert TORCH_AVAILABLE, "PyTorch not available"
        assert torch.cuda.is_available(), "CUDA/ROCm not available"
        
        device_name = torch.cuda.get_device_name(0)
        print(f"\n✓ GPU detected: {device_name}")
        
        # Record test properties
        if hasattr(pytest, 'current_test_info'):
            pytest.current_test_info['device'] = device_name
    
    def test_rocprof_installation(self):
        """Check if ROCProfiler tools are installed"""
        has_rocprof = check_rocprof_available()
        has_rocprofv3 = check_rocprofiler_sdk()
        
        print(f"\n✓ rocprof available: {has_rocprof}")
        print(f"✓ rocprofv3 available: {has_rocprofv3}")
        
        # At least one profiler should be available
        assert has_rocprof or has_rocprofv3, \
            "No ROCProfiler tools found. Install rocprofiler-sdk or roctracer"
        
        # Record which profilers are available
        if has_rocprof:
            result = subprocess.run(
                ["rocprof", "--version"],
                capture_output=True,
                text=True
            )
            print(f"\nrocprof version:\n{result.stdout}")
        
        if has_rocprofv3:
            result = subprocess.run(
                ["rocprofv3", "--version"],
                capture_output=True,
                text=True
            )
            print(f"\nrocprofv3 version:\n{result.stdout}")
    
    def test_pytorch_simple_inference_profile(self, strix_device, cleanup_gpu):
        """Profile a simple PyTorch inference operation"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        if not check_rocprof_available() and not check_rocprofiler_sdk():
            pytest.skip("No ROCProfiler tools available")
        
        # Create simple model
        model = SimpleNet().to(strix_device)
        model.eval()
        
        # Create input
        batch_size = 32
        input_tensor = torch.randn(batch_size, 1024, device=strix_device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)
        torch.cuda.synchronize()
        
        # Profiled inference
        print("\n=== Running profiled inference ===")
        with torch.no_grad():
            # Use PyTorch's profiler with ROCm backend
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                output = model(input_tensor)
                torch.cuda.synchronize()
        
        # Check output
        assert output.shape == (batch_size, 10), "Unexpected output shape"
        
        # Print profiling results
        print("\n=== Top 10 Operations by GPU Time ===")
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=10
        ))
        
        # Record metrics
        key_averages = prof.key_averages()
        total_cuda_time = sum([item.cuda_time_total for item in key_averages])
        print(f"\n✓ Total GPU time: {total_cuda_time / 1000:.2f} ms")
    
    def test_pytorch_training_step_profile(self, strix_device, cleanup_gpu):
        """Profile a single PyTorch training step"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        if not check_rocprof_available() and not check_rocprofiler_sdk():
            pytest.skip("No ROCProfiler tools available")
        
        # Create model and optimizer
        model = SimpleNet().to(strix_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create dummy data
        batch_size = 32
        input_tensor = torch.randn(batch_size, 1024, device=strix_device)
        target = torch.randint(0, 10, (batch_size,), device=strix_device)
        
        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        
        # Profiled training step
        print("\n=== Running profiled training step ===")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
        
        print(f"✓ Training step completed. Loss: {loss.item():.4f}")
        
        # Print profiling results
        print("\n=== Top 10 Operations by GPU Time ===")
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=10
        ))
        
        # Analyze forward/backward breakdown
        key_averages = prof.key_averages()
        total_cuda_time = sum([item.cuda_time_total for item in key_averages])
        print(f"\n✓ Total GPU time: {total_cuda_time / 1000:.2f} ms")
    
    @pytest.mark.slow
    def test_rocprof_external_profile(self, strix_device, cleanup_gpu):
        """Test external rocprof profiling of a PyTorch script"""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        if not check_rocprof_available():
            pytest.skip("rocprof not available")
        
        # Create a temporary Python script to profile
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "profile_target.py"
            output_dir = Path(tmpdir) / "profile_output"
            output_dir.mkdir()
            
            # Write test script
            script_content = f"""
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

device = torch.device('cuda')
model = Net().to(device)
model.eval()

# Warmup
for _ in range(5):
    x = torch.randn(32, 1024, device=device)
    with torch.no_grad():
        y = model(x)
torch.cuda.synchronize()

# Profiled section
for _ in range(10):
    x = torch.randn(32, 1024, device=device)
    with torch.no_grad():
        y = model(x)
torch.cuda.synchronize()

print("Completed")
"""
            script_path.write_text(script_content)
            
            # Run with rocprof
            print("\n=== Running rocprof on external script ===")
            cmd = [
                "rocprof",
                "--stats",
                "-o", str(output_dir / "results.csv"),
                sys.executable,
                str(script_path)
            ]
            
            print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(f"\nReturn code: {result.returncode}")
            print(f"Stdout:\n{result.stdout}")
            if result.stderr:
                print(f"Stderr:\n{result.stderr}")
            
            # Check for output files
            output_files = list(output_dir.glob("*"))
            print(f"\n✓ Output files created: {[f.name for f in output_files]}")
            
            assert len(output_files) > 0, "No profiling output files created"
            
            # Try to read results if CSV exists
            csv_files = list(output_dir.glob("*.csv"))
            if csv_files:
                print(f"\n=== Profiling results ===")
                print(csv_files[0].read_text()[:1000])  # First 1000 chars


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.quick
@pytest.mark.p0
def test_quick_profiling_smoke():
    """Quick smoke test for profiling capability"""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    # Quick tensor operation with profiling
    device = torch.device("cuda")
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        x = torch.randn(100, 100, device=device)
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
    
    # Verify profiler captured something
    events = prof.key_averages()
    assert len(events) > 0, "No profiling events captured"
    print(f"\n✓ Captured {len(events)} profiling events")

