# Client Example: Grouped GEMM with bf16A/int8B and Fused Epilogues

## Theory

This client example demonstrates **grouped GEMM with mixed-precision input types (bf16 for A, int8 for B)** and various fused epilogue operations (bias, FastGELU, multiply). Grouped GEMM performs multiple independent GEMM operations (with potentially different shapes) in a single kernel launch, and mixed-precision is used for efficient inference and training.

**Mathematical Formulation:**
For $G$ groups, each with its own $A_g$, $B_g$:
- GEMM: $Y_g = A_g \times B_g$
  - $A_g$: bf16 (brain floating point)
  - $B_g$: int8 (8-bit integer)
- Fused epilogues:
  - Bias: $Z_g = Y_g + \text{bias}_g$
  - FastGELU: $E_g = \text{FastGELU}(Z_g)$
  - Multiply: $E_g = Z_g \odot D_{1,g}$

**Algorithmic Background:**
- Each group can have different matrix sizes and strides.
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
cd composable_kernel/client_example/31_grouped_gemm_bf16Aint8B
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (basic grouped GEMM)
./grouped_gemm_xdl_bf16_i8

# Example run (grouped GEMM + bias + FastGELU)
./grouped_gemm_bias_fastgelu_xdl_bf16_i8

# Example run (grouped GEMM + FastGELU)
./grouped_gemm_fastgelu_xdl_bf16_i8

# Example run (grouped GEMM + multiply)
./grouped_gemm_multiply_xdl_bf16_i8

# Example run (grouped GEMM + multiply + bias + FastGELU)
./grouped_gemm_multiply_bias_fastgelu_xdl_bf16_i8
```

## Source Code Structure

### Directory Layout
```
client_example/31_grouped_gemm_bf16Aint8B/
├── grouped_gemm_xdl_bf16_i8.cpp                # Grouped GEMM (bf16A, int8B)
├── grouped_gemm_bias_fastgelu_xdl_bf16_i8.cpp  # Grouped GEMM + bias + FastGELU
├── grouped_gemm_fastgelu_xdl_bf16_i8.cpp       # Grouped GEMM + FastGELU
├── grouped_gemm_multiply_xdl_bf16_i8.cpp       # Grouped GEMM + multiply
├── grouped_gemm_multiply_bias_fastgelu_xdl_bf16_i8.cpp # Grouped GEMM + multiply + bias + FastGELU
├── CMakeLists.txt                              # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input matrices for each group, configures GEMM and epilogue parameters, launches the grouped kernel, and verifies the result.
- **Grouped GEMM kernel invocation**:  
  Uses the Composable Kernel device API to launch grouped GEMM with various fused epilogues.

---

## Additional Details

- Supports multiple groups with different matrix shapes and bf16/int8 input types.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [30_gemm_bf16Aint8B](../30_gemm_bf16Aint8B/README.md): GEMM with bf16A/int8B and fused epilogues
- [15_grouped_gemm](../../example/15_grouped_gemm/README.md): Grouped GEMM in the main example directory

---
[Back to Client Examples](../README.md)
