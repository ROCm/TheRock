#!/usr/bin/env python3
"""Quick ROCm GPU Benchmark - No Flask, No SQLAlchemy, Just Pure PyTorch"""

import torch
import time
import sys


def check_gpu():
    """Verify ROCm GPU is available"""
    if not torch.cuda.is_available():
        print("❌ CUDA/ROCm not available")
        sys.exit(1)

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ ROCm Version: {torch.version.hip}")
    print(f"✅ PyTorch: {torch.__version__}")
    print()


def benchmark_matmul(size=4096, iterations=100):
    """Matrix multiplication benchmark"""
    print(f"🔧 Matrix Multiply ({size}x{size})...")

    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Calculate TFLOPS: 2*N^3 operations for NxN matmul
    flops = 2 * size**3 * iterations
    tflops = (flops / elapsed) / 1e12

    print(f"   Time: {elapsed:.2f}s | {tflops:.2f} TFLOPS")
    return tflops


def benchmark_conv2d(batch=64, iterations=50):
    """Conv2D benchmark (common in image processing)"""
    print(f"🔧 Conv2D (batch={batch})...")

    # Typical ResNet-like layer
    conv = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1).cuda()
    x = torch.randn(batch, 64, 56, 56, device="cuda")

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = conv(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            y = conv(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = (batch * iterations) / elapsed
    print(f"   Time: {elapsed:.2f}s | {throughput:.0f} images/sec")
    return throughput


def benchmark_memory(size_gb=2):
    """Memory bandwidth test"""
    print(f"🔧 Memory Bandwidth ({size_gb}GB)...")

    elements = (size_gb * 1024**3) // 4  # float32 = 4 bytes
    a = torch.randn(elements, device="cuda", dtype=torch.float32)
    b = torch.randn(elements, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(5):
        _ = a + b
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(20):
        c = a + b
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Read 2 arrays + write 1 array = 3x data movement
    bytes_transferred = 3 * elements * 4 * 20
    bandwidth_gbs = (bytes_transferred / elapsed) / 1e9

    print(f"   Time: {elapsed:.2f}s | {bandwidth_gbs:.0f} GB/s")
    return bandwidth_gbs


def main():
    print("=" * 60)
    print("ROCm GPU Quick Benchmark")
    print("=" * 60)
    print()

    check_gpu()

    print("Running benchmarks...")
    print()

    results = {}
    results["matmul_tflops"] = benchmark_matmul(4096, 100)
    torch.cuda.empty_cache()

    results["conv2d_imgs_sec"] = benchmark_conv2d(64, 50)
    torch.cuda.empty_cache()

    results["memory_gbs"] = benchmark_memory(1)  # Use 1GB to be safe
    torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Matrix Multiply: {results['matmul_tflops']:.2f} TFLOPS")
    print(f"Conv2D:          {results['conv2d_imgs_sec']:.0f} images/sec")
    print(f"Memory:          {results['memory_gbs']:.0f} GB/s")
    print()

    # Quick comparison to your RX 6700 XT specs
    print("Hardware: AMD RX 6700 XT (gfx1031)")
    print(f"Theoretical Peak: ~13.2 TFLOPS (FP32)")
    print(f"Achieved: {(results['matmul_tflops']/13.2)*100:.1f}% of peak")


if __name__ == "__main__":
    main()
