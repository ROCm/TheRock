# Client Example: Split-K GEMM

## Theory

This client example demonstrates **Split-K GEMM**, a technique for parallelizing matrix multiplication along the K dimension. Split-K is used to improve parallelism and memory bandwidth utilization for large GEMM operations, especially when K is large.

**Mathematical Formulation:**
- Standard GEMM: $C = A \times B$
- Split-K: Partition the K dimension into $K_s$ splits, compute partial results, then reduce:
  $$
  C = \sum_{s=1}^{K_s} (A_{[:, K_s]} \times B_{[K_s, :]})
  $$

**Algorithmic Background:**
- Each split computes a partial GEMM over a chunk of K.
- Partial results are reduced (summed) to produce the final output.
- Useful for large K, limited workspace, or maximizing GPU occupancy.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/20_splitk_gemm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (FP16 compute, FP8 output)
./splitK_gemm_fp16_f8
```

## Source Code Structure

### Directory Layout
```
client_example/20_splitk_gemm/
├── splitK_gemm_fp16_f8.cpp         # Main client example: Split-K GEMM (FP16 compute, FP8 output)
├── CMakeLists.txt                  # Build configuration for the example
```

### Key Functions

- **main()** (in `splitK_gemm_fp16_f8.cpp`):  
  Sets up input matrices, configures Split-K parameters, launches the Split-K GEMM kernel, and verifies the result.
- **Split-K kernel invocation**:  
  Uses the Composable Kernel device API to launch the Split-K GEMM operation.

---

## Additional Details

- Supports FP16 compute with FP8 output for memory efficiency.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [35_splitK_gemm](../../example/35_splitK_gemm/README.md): Split-K GEMM in the main example directory

---
[Back to Client Examples](../README.md)
