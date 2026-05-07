# Client Example: Grouped GEMM with Bias

## Theory

This client example demonstrates **grouped GEMM fused with bias addition**. Grouped GEMM performs multiple independent GEMM operations (with potentially different shapes) in a single kernel launch, and bias addition is a standard pattern in neural network layers.

**Mathematical Formulation:**
For $G$ groups, each with its own $A_g$, $B_g$, $b_g$:
- GEMM: $Y_g = A_g \times B_g$
- Bias: $E_g = Y_g + b_g$

**Algorithmic Background:**
- Each group can have different matrix sizes and strides.
- The kernel launches a grid covering all groups, with each block assigned to a group.
- Bias is added in the epilogue for each group.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/21_grouped_gemm_bias
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (grouped GEMM with bias, FP16)
./grouped_gemm_fixed_nk_bias_fp16
```

## Source Code Structure

### Directory Layout
```
client_example/21_grouped_gemm_bias/
├── grouped_gemm_fixed_nk_bias_fp16.cpp         # Main client example: grouped GEMM + bias (FP16)
├── CMakeLists.txt                              # Build configuration for the example
```

### Key Functions

- **main()** (in `grouped_gemm_fixed_nk_bias_fp16.cpp`):  
  Sets up input matrices for each group, configures GEMM and bias parameters, launches the grouped kernel, and verifies the result.
- **Grouped GEMM kernel invocation**:  
  Uses the Composable Kernel device API to launch grouped GEMM with bias addition.

---

## Additional Details

- Supports multiple groups with different matrix shapes.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [15_grouped_gemm](../../example/15_grouped_gemm/README.md): Grouped GEMM in the main example directory
- [11_convnd_fwd_bias](../../example/11_convnd_fwd_bias/README.md): Convolution with bias fusion

---
[Back to Client Examples](../README.md)
