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


def check_rocprofv3_available():
    """
    Check if rocprofv3 (ROCProfiler v3) is available for Strix profiling
    
    NOTE: For Strix, use rocprofv3 ONLY (not legacy rocprof)
    See test_strix_rocprofv3.py for Strix-optimized profiling tests
    """
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
        """Profile a simple PyTorch inference operation using ROCProfiler"""
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
        
        # Profile with ROCProfiler
        print("\n=== Running ROCProfiler-instrumented inference ===")
        
        # Use ROCM_CALL_STATS environment variable for lightweight profiling (optional)
        import os
        import ctypes.util
        # Only set if rocprofiler library exists
        if ctypes.util.find_library('rocprofiler64'):
            os.environ['HSA_TOOLS_LIB'] = 'librocprofiler64.so.1'
            os.environ['ROCP_HSA_INTERCEPT'] = '1'
        
        import time
        start = time.perf_counter()
        
        with torch.no_grad():
            output = model(input_tensor)
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        inference_time = (end - start) * 1000  # Convert to ms
        
        # Check output
        assert output.shape == (batch_size, 10), "Unexpected output shape"
        
        print(f"\n✓ Inference completed successfully")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Inference time: {inference_time:.2f} ms")
        print(f"✓ Throughput: {batch_size / (inference_time / 1000):.1f} samples/sec")
        
        # Note: For detailed profiling, use test_rocprof_external_profile
        print("\nNote: For detailed HIP kernel profiling, use test_rocprof_external_profile")
    
    def test_pytorch_training_step_profile(self, strix_device, cleanup_gpu):
        """Profile a single PyTorch training step using ROCProfiler"""
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
        
        # Profile training step with ROCProfiler
        print("\n=== Running ROCProfiler-instrumented training step ===")
        
        import time
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        
        end = time.perf_counter()
        training_time = (end - start) * 1000
        
        print(f"✓ Training step completed. Loss: {loss.item():.4f}")
        print(f"✓ Total training time: {training_time:.2f} ms")
        print(f"✓ Throughput: {batch_size / (training_time / 1000):.1f} samples/sec")
        
        print("\nNote: For detailed kernel-level profiling with HIP traces,")
        print("      use test_rocprof_external_profile or test_rocprofv3_external_profile")
    
    @pytest.mark.slow
    @pytest.mark.skip(reason="DEPRECATED: Use test_strix_rocprofv3.py for Strix profiling with rocprofv3")
    def test_rocprof_external_profile(self, strix_device, cleanup_gpu):
        """
        DEPRECATED: Legacy rocprof (roctracer) test
        
        For Strix profiling, use test_strix_rocprofv3.py instead:
          pytest tests/strix_ai/profiling/test_strix_rocprofv3.py -v -s
        """
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        if not check_rocprof_available():
            pytest.skip("rocprof (roctracer) not available")
        
        print("\n" + "="*70)
        print("ROCProfiler (roctracer) - HIP Trace and Kernel Statistics")
        print("="*70)
        
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

print("=== Starting profiled workload ===")

# Warmup
for _ in range(5):
    x = torch.randn(32, 1024, device=device)
    with torch.no_grad():
        y = model(x)
torch.cuda.synchronize()

# Profiled section
for i in range(10):
    x = torch.randn(32, 1024, device=device)
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()
    if i == 0:
        print(f"First iteration shape: {{y.shape}}")

print("=== Profiling completed ===")
"""
            script_path.write_text(script_content)
            
            # Run with rocprof for HIP kernel tracing
            print("\n=== Running rocprof (ROCm HIP Tracer) ===")
            
            # rocprof options:
            # --stats: Generate statistics
            # --hip-trace: Trace HIP API calls
            # --hsa-trace: Trace HSA API calls
            cmd = [
                "rocprof",
                "--stats",
                "--hip-trace",
                "-o", str(output_dir / "results.csv"),
                "-d", str(output_dir),
                sys.executable,
                str(script_path)
            ]
            
            print(f"Command: {' '.join(cmd)}\n")
            
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"\nApplication output:\n{result.stdout}")
            if result.stderr:
                print(f"\nProfiler stderr:\n{result.stderr}")
            
            # Check for output files
            output_files = list(output_dir.glob("*"))
            print(f"\n✓ ROCProfiler output files created:")
            for f in output_files:
                print(f"  - {f.name} ({f.stat().st_size} bytes)")
            
            assert len(output_files) > 0, "No profiling output files created"
            
            # Parse and display results
            print("\n" + "="*70)
            print("ROCProfiler Statistics Summary")
            print("="*70)
            
            # Read results CSV if exists
            csv_files = list(output_dir.glob("results_stats.csv")) or list(output_dir.glob("*.csv"))
            if csv_files:
                csv_content = csv_files[0].read_text()
                print(f"\nStatistics file: {csv_files[0].name}")
                print("-" * 70)
                
                # Display first 50 lines or 2000 chars
                lines = csv_content.split('\n')
                display_lines = min(50, len(lines))
                print('\n'.join(lines[:display_lines]))
                
                if len(csv_content) > 2000:
                    print(f"\n... ({len(lines) - display_lines} more lines)")
            
            # Check for HIP trace file
            trace_files = list(output_dir.glob("*hip_stats.csv")) or list(output_dir.glob("*trace*"))
            if trace_files:
                print(f"\n✓ HIP trace captured: {trace_files[0].name}")
                trace_content = trace_files[0].read_text()
                lines = trace_content.split('\n')[:20]
                print("HIP API calls (first 20 lines):")
                print('\n'.join(lines))
            
            print("\n✓ ROCProfiler (roctracer) profiling completed successfully")


    @pytest.mark.slow
    @pytest.mark.skip(reason="MOVED: Use test_strix_rocprofv3.py for comprehensive rocprofv3 profiling")
    def test_rocprofv3_external_profile(self, strix_device, cleanup_gpu):
        """
        MOVED to test_strix_rocprofv3.py
        
        For Strix profiling with rocprofv3, use:
          pytest tests/strix_ai/profiling/test_strix_rocprofv3.py::TestStrixRocprofv3::test_rocprofv3_pytorch_inference -v -s
        """
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        if not check_rocprofiler_sdk():
            pytest.skip("rocprofv3 (rocprofiler-sdk) not available")
        
        print("\n" + "="*70)
        print("ROCProfiler-SDK (rocprofv3) - Advanced Profiling")
        print("="*70)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "profile_target.py"
            output_dir = Path(tmpdir) / "rocprof_output"
            output_dir.mkdir()
            
            script_content = """
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

device = torch.device('cuda')
model = Net().to(device)
model.eval()

print("Starting rocprofv3 profiled workload")

# Warmup
for _ in range(3):
    x = torch.randn(32, 1024, device=device)
    with torch.no_grad():
        y = model(x)
torch.cuda.synchronize()

# Profiled iterations
for i in range(10):
    x = torch.randn(32, 1024, device=device)
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()

print("rocprofv3 profiling completed")
"""
            script_path.write_text(script_content)
            
            print("\n=== Running rocprofv3 (ROCProfiler-SDK) ===")
            
            # rocprofv3 options:
            # --hip-trace: Trace HIP API
            # --kernel-trace: Trace kernel execution
            cmd = [
                "rocprofv3",
                "--hip-trace",
                "--kernel-trace",
                "-d", str(output_dir),
                "-o", "rocprof",
                "--",
                sys.executable,
                str(script_path)
            ]
            
            print(f"Command: {' '.join(cmd)}\n")
            
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"\nApplication output:\n{result.stdout}")
            if result.stderr and "warning" not in result.stderr.lower():
                print(f"\nStderr:\n{result.stderr}")
            
            # Check outputs
            output_files = list(output_dir.glob("*"))
            print(f"\n✓ ROCProfiler-SDK output files:")
            for f in output_files:
                print(f"  - {f.name} ({f.stat().st_size} bytes)")
            
            if len(output_files) > 0:
                print("\n✓ rocprofv3 (rocprofiler-sdk) profiling completed successfully")
                
                # Look for specific output files
                for pattern in ["*.csv", "*.json", "*.db"]:
                    matches = list(output_dir.glob(pattern))
                    if matches:
                        print(f"\n{pattern} files found: {len(matches)}")
            else:
                print("\n⚠ Warning: No output files generated (may need ROCm 6.0+)")


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.quick
@pytest.mark.p0
def test_quick_profiling_smoke():
    """Quick smoke test for rocprofv3 availability (Strix uses rocprofv3 only)"""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    print("\n=== Quick rocprofv3 Smoke Test (Strix) ===")
    
    # Check rocprofv3 only (Strix uses rocprofv3, not legacy rocprof)
    has_rocprofv3 = check_rocprofv3_available()
    
    print(f"✓ rocprofv3 (rocprofiler-sdk): {'Available' if has_rocprofv3 else 'Not found'}")
    
    if not has_rocprofv3:
        print("\n⚠ rocprofv3 not found!")
        print("  Install: ROCm 6.2+ includes rocprofiler-sdk")
        print("  For full profiling tests, use:")
        print("    pytest tests/strix_ai/profiling/test_strix_rocprofv3.py -v -s")
    
    assert has_rocprofv3, \
        "rocprofv3 not found. Install rocprofiler-sdk (included in ROCm 6.2+)"
    
    # Quick GPU operation timing
    device = torch.device("cuda")
    
    import time
    x = torch.randn(100, 100, device=device)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed_ms = (end - start) * 1000
    
    print(f"\n✓ GPU matmul operation: {elapsed_ms:.3f} ms")
    print(f"✓ ROCProfiler tools ready for detailed profiling")

