# Client Example: Batch Normalization (Forward, Backward, Inference)

## Theory

This client example demonstrates **batch normalization** in forward, backward, and inference modes for NHWC tensors. Batch normalization is used in deep neural networks to normalize activations across the batch and spatial dimensions, improving training stability and convergence.

**Mathematical Formulation:**
Given input $X[N, H, W, C]$:
- Mean: $\mu_c = \frac{1}{NHW} \sum_{n,h,w} X_{n,h,w,c}$
- Variance: $\sigma^2_c = \frac{1}{NHW} \sum_{n,h,w} (X_{n,h,w,c} - \mu_c)^2$
- Normalized: $\hat{X}_{n,h,w,c} = \frac{X_{n,h,w,c} - \mu_c}{\sqrt{\sigma^2_c + \epsilon}}$
- Output: $Y_{n,h,w,c} = \gamma_c \hat{X}_{n,h,w,c} + \beta_c$

$\gamma_c$, $\beta_c$ are learnable scale and shift parameters per channel.

**Algorithmic Background:**
- Forward pass computes mean, variance, normalization, and affine transformation.
- Backward pass computes gradients with respect to input, gamma, and beta.
- Inference uses running mean and variance for normalization.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/13_batchnorm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (forward)
./batchnorm_fwd_nhwc

# Example run (backward)
./batchnorm_bwd_nhwc

# Example run (inference)
./batchnorm_infer_nhwc
```

## Source Code Structure

### Directory Layout
```
client_example/13_batchnorm/
├── batchnorm_fwd_nhwc.cpp         # Batchnorm forward (NHWC)
├── batchnorm_bwd_nhwc.cpp         # Batchnorm backward (NHWC)
├── batchnorm_infer_nhwc.cpp       # Batchnorm inference (NHWC)
├── CMakeLists.txt                 # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input tensors, configures batchnorm parameters, launches the forward, backward, or inference kernel, and verifies the result.
- **BatchNorm kernel invocation**:  
  Uses the Composable Kernel device API to launch batch normalization for different modes.

---

## Additional Details

- Supports NHWC layout for image and vision models.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [34_batchnorm](../../example/34_batchnorm/README.md): Batch normalization in the main example directory

---
[Back to Client Examples](../README.md)
