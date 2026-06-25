# Client Example: GEMM with LayerNorm Fusion

## Theory

This client example demonstrates **GEMM fused with layer normalization** and additional elementwise operations. This pattern is common in transformer feed-forward networks and other architectures where a linear transformation is followed by normalization and activation.

**Mathematical Formulation:**
- GEMM: $Y = A \times B$
- Additions: $Z = Y + D_0 + D_1$ (bias, residual, etc.)
- Activation: $A = \text{ReLU}(Z)$ (or other activation)
- LayerNorm: $\text{LayerNorm}(A) = \gamma \cdot \frac{A - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$

$\mu$, $\sigma^2$ are mean and variance over the normalization axis; $\gamma$, $\beta$ are learnable scale and shift.

**Algorithmic Background:**
- The GEMM result is kept in registers, elementwise ops and layer normalization are fused in the epilogue.
- LayerNorm is typically applied over the last dimension (features).
- This fusion reduces memory traffic and is common in transformer MLP blocks.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/03_gemm_layernorm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (naive)
./gemm_add_add_layernorm_naive

# Example run (with ReLU and Welford)
./gemm_add_relu_add_layernorm_welford
```

## Source Code Structure

### Directory Layout
```
client_example/03_gemm_layernorm/
├── gemm_add_add_layernorm_naive.cpp         # GEMM + Add + Add + LayerNorm (naive)
├── gemm_add_relu_add_layernorm_welford.cpp  # GEMM + Add + ReLU + Add + LayerNorm (Welford)
├── CMakeLists.txt                           # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input matrices, configures GEMM and epilogue parameters, launches the fused kernel, and verifies the result.
- **LayerNorm implementation**:  
  Demonstrates both naive and numerically stable (Welford) algorithms for mean/variance.

This client example provides variants to demonstrate different levels of fusion and normalization for transformer-style MLP layers.
