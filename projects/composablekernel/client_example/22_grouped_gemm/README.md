# Client Example: Grouped GEMM (Multiple Data Types)

## Theory

This client example demonstrates **grouped GEMM** for multiple data types (FP16, BF16, FP8, INT8). Grouped GEMM performs multiple independent GEMM operations (with potentially different shapes) in a single kernel launch, which is useful for transformer models, mixture-of-experts, and variable-length sequence processing.

**Mathematical Formulation:**
For $G$ groups, each with its own $A_g$, $B_g$:
- GEMM: $Y_g = A_g \times B_g$

**Algorithmic Background:**
- Each group can have different matrix sizes and strides.
- The kernel launches a grid covering all groups, with each block assigned to a group.
- Supports multiple data types for flexibility and performance tuning.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/22_grouped_gemm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (FP16)
./grouped_gemm_fixed_nk_fp16

# Example run (BF16)
./grouped_gemm_fixed_nk_bf16

# Example run (FP8)
./grouped_gemm_fixed_nk_fp8

# Example run (INT8)
./grouped_gemm_fixed_nk_i8
```

## Source Code Structure

### Directory Layout
```
client_example/22_grouped_gemm/
├── grouped_gemm_fixed_nk_fp16.cpp         # Grouped GEMM (FP16)
├── grouped_gemm_fixed_nk_bf16.cpp         # Grouped GEMM (BF16)
├── grouped_gemm_fixed_nk_fp8.cpp          # Grouped GEMM (FP8)
├── grouped_gemm_fixed_nk_i8.cpp           # Grouped GEMM (INT8)
├── CMakeLists.txt                         # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input matrices for each group, configures GEMM parameters, launches the grouped kernel, and verifies the result.
- **Grouped GEMM kernel invocation**:  
  Uses the Composable Kernel device API to launch grouped GEMM for different data types.

---

## Additional Details

- Supports multiple groups with different matrix shapes and data types.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [15_grouped_gemm](../../example/15_grouped_gemm/README.md): Grouped GEMM in the main example directory
- [17_grouped_gemm_fastgelu](../17_grouped_gemm_fastgelu/README.md): Grouped GEMM with FastGELU activation

---
[Back to Client Examples](../README.md)
