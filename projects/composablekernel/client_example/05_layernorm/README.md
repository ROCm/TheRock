# Client Example: Layer Normalization (Forward and Backward)

## Theory

This client example demonstrates **layer normalization** in both forward and backward modes, for 2D and 4D tensors. Layer normalization is used in transformers and other neural networks to normalize activations across the feature dimension, improving training stability.

**Mathematical Formulation:**
Given input $X$:
- Mean: $\mu = \frac{1}{N} \sum_{i=1}^N X_i$
- Variance: $\sigma^2 = \frac{1}{N} \sum_{i=1}^N (X_i - \mu)^2$
- Normalized: $\hat{X}_i = \frac{X_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
- Output: $Y_i = \gamma \hat{X}_i + \beta$

$\gamma$, $\beta$ are learnable scale and shift parameters.

**Algorithmic Background:**
- Forward pass computes mean, variance, normalization, and affine transformation.
- Backward pass computes gradients with respect to input, gamma, and beta.
- Supports both 2D (batch, feature) and 4D (batch, channel, height, width) tensors.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/05_layernorm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (2D forward)
./layernorm2d_fwd

# Example run (4D forward)
./layernorm4d_fwd

# Example run (2D backward, data)
./layernorm2d_bwd_data

# Example run (2D backward, gamma/beta)
./layernorm2d_bwd_gamma_beta
```

## Source Code Structure

### Directory Layout
```
client_example/05_layernorm/
├── layernorm2d_fwd.cpp         # 2D layernorm forward
├── layernorm4d_fwd.cpp         # 4D layernorm forward
├── layernorm2d_bwd_data.cpp    # 2D layernorm backward (data)
├── layernorm2d_bwd_gamma_beta.cpp # 2D layernorm backward (gamma/beta)
├── CMakeLists.txt              # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input tensors, configures normalization parameters, launches the forward or backward kernel, and verifies the result.
- **LayerNorm implementation**:  
  Demonstrates both forward and backward passes for different tensor shapes.

This client example provides a comprehensive demonstration of layer normalization for both inference and training in deep learning models.
