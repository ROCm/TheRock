# 2D Layer Normalization Forward

## Theory

This example demonstrates **2D layer normalization forward pass**. Layer normalization is used in transformers and other neural networks to normalize activations across the feature dimension, improving training stability.

**Mathematical Formulation:**
Given input $X[N, C, H, W]$:
- Mean: $\mu = \frac{1}{CHW} \sum_{c,h,w} X_{n,c,h,w}$
- Variance: $\sigma^2 = \frac{1}{CHW} \sum_{c,h,w} (X_{n,c,h,w} - \mu)^2$
- Normalized: $\hat{X}_{n,c,h,w} = \frac{X_{n,c,h,w} - \mu}{\sqrt{\sigma^2 + \epsilon}}$
- Output: $Y_{n,c,h,w} = \gamma \hat{X}_{n,c,h,w} + \beta$

$\gamma$, $\beta$ are learnable scale and shift parameters.

**Algorithmic Background:**
- Computes mean and variance per sample (across all features).
- Applies normalization and affine transformation.
- Used in transformer blocks and normalization layers.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/27_layernorm2d_fwd
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./layernorm2d_fwd_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/27_layernorm2d_fwd/
├── layernorm2d_fwd_xdl.cpp         # Main example: sets up, runs, and verifies 2D layernorm
include/ck/tensor_operation/gpu/device/
│   └── device_layernorm_fwd.hpp       # Device-level layernorm API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_layernorm_fwd_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
    └── gridwise_layernorm_fwd.hpp     # Grid-level kernel
```

### Key Classes and Functions

- **DeviceLayernormFwd** (in `device_layernorm_fwd.hpp`):  
  Device API for layer normalization.
- **gridwise_layernorm_fwd** (in `gridwise_layernorm_fwd.hpp`):  
  Implements the tiled/blocking layernorm kernel.

This example demonstrates how Composable Kernel implements efficient layer normalization for transformer and deep learning models.
