# Client Example: GEMM with Add, Add, and FastGELU Fusion

## Theory

This client example demonstrates **GEMM fused with two addition operations and FastGELU activation**. This pattern is common in transformer feed-forward networks and other neural architectures where a linear transformation is followed by bias addition, residual addition, and a non-linear activation.

**Mathematical Formulation:**
$$
E = \text{FastGELU}((A \times B) + D_0 + D_1)
$$
- $A$: [M, K] input matrix
- $B$: [K, N] weight matrix
- $D_0$: [N] bias vector (broadcasted)
- $D_1$: [M, N] residual tensor
- $E$: [M, N] output

FastGELU is an efficient approximation of GELU:
$$
\text{FastGELU}(x) = x \cdot \sigma(1.702 \cdot x)
$$
where $\sigma$ is the sigmoid function.

**Algorithmic Background:**
- The GEMM result is kept in registers, bias and residual are added, and FastGELU is applied before writing to global memory.
- No intermediate results are written to global memory.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/02_gemm_add_add_fastgelu
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./gemm_add_add_fastgelu
```

## Source Code Structure

### Directory Layout
```
client_example/02_gemm_add_add_fastgelu/
├── gemm_add_add_fastgelu.cpp         # Main client example: GEMM+Add+Add+FastGELU
├── gemm_add_add_fastgelu_generic.cpp # Generic variant
├── gemm_add_fastgelu.cpp             # GEMM+Add+FastGELU
├── gemm_add_fastgelu_generic.cpp     # Generic variant
├── gemm_fastgelu.cpp                 # GEMM+FastGELU only
├── gemm_fastgelu_generic.cpp         # Generic variant
├── CMakeLists.txt                    # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input matrices, configures GEMM and epilogue parameters, launches the fused kernel, and verifies the result.
- **Fused kernel invocation**:  
  Uses the Composable Kernel device API to launch the GEMM with fused addition and FastGELU.

This client example provides several variants to demonstrate different levels of fusion and genericity for transformer-style MLP layers.
