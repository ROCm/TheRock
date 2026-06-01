# Client Example: GEMM with bf16A/int8B and Fused Epilogues

## Theory

This client example demonstrates **GEMM with mixed-precision input types (bf16 for A, int8 for B)** and various fused epilogue operations (bias, GELU, FastGELU, multiply). Mixed-precision GEMM is used for efficient inference and training in deep learning, especially for transformer and MLP layers.

**Mathematical Formulation:**
- GEMM: $Y = A \times B$
  - $A$: bf16 (brain floating point)
  - $B$: int8 (8-bit integer)
- Fused epilogues:
  - Bias: $Z = Y + \text{bias}$
  - GELU: $E = \text{GELU}(Z)$
  - FastGELU: $E = \text{FastGELU}(Z)$
  - Multiply: $E = Z \odot D_1$

**Algorithmic Background:**
- Mixed-precision computation reduces memory and compute requirements.
- Fused epilogues improve efficiency by combining bias, activation, and scaling in a single kernel.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

```bash
cd composable_kernel/build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D DTYPES="bf16;int8" ..
make -j
make install
```

### Build and run
```bash
cd composable_kernel/client_example/30_gemm_bf16Aint8B
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (basic GEMM)
./gemm_xdl_bf16_i8

# Example run (GEMM + bias)
./gemm_bias_xdl_bf16_i8

# Example run (GEMM + bias + GELU)
./gemm_xdl_gelu_bf16_i8

# Example run (GEMM + bias + FastGELU)
./gemm_bias_fastgelu_xdl_bf16_i8

# Example run (GEMM + multiply)
./gemm_xdl_multiply_bf16_i8
```

## Source Code Structure

### Directory Layout
```
client_example/30_gemm_bf16Aint8B/
├── gemm_xdl_bf16_i8.cpp                # GEMM (bf16A, int8B)
├── gemm_bias_xdl_bf16_i8.cpp           # GEMM + bias
├── gemm_xdl_gelu_bf16_i8.cpp           # GEMM + bias + GELU
├── gemm_bias_fastgelu_xdl_bf16_i8.cpp  # GEMM + bias + FastGELU
├── gemm_xdl_multiply_bf16_i8.cpp       # GEMM + multiply
├── CMakeLists.txt                      # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input matrices, configures GEMM and epilogue parameters, launches the kernel, and verifies the result.
- **Fused kernel invocation**:  
  Uses the Composable Kernel device API to launch GEMM with various fused epilogues.

---

## Additional Details

- Supports bf16 and int8 input types for efficient mixed-precision computation.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [14_gemm_quantization](../../example/14_gemm_quantization/README.md): GEMM quantization in the main example directory
- [46_gemm_add_multiply](../../example/46_gemm_add_multiply/README.md): GEMM with add and multiply in the main example directory

---
[Back to Client Examples](../README.md)
