# 2D Layer Normalization Backward

This example demonstrates the **backward pass of 2D Layer Normalization**. This operation computes the gradients of the loss with respect to the input, gamma, and beta parameters of a layer normalization layer, which is essential for training neural networks that use layer normalization, particularly Transformers.

## Mathematical Formulation

The backward pass of layer normalization involves computing gradients for three components: input `X`, scale parameter `gamma`, and shift parameter `beta`.

Given:
- Input tensor `X` with shape `[M, N]`
- Scale parameter `gamma` with shape `[N]`
- Shift parameter `beta` with shape `[N]`
- Output gradients `dL/dY` with shape `[M, N]`

From the forward pass, we have:
- Mean: $\mu_i = \frac{1}{N} \sum_{j=0}^{N-1} X_{ij}$ for each row `i`
- Variance: $\sigma_i^2 = \frac{1}{N} \sum_{j=0}^{N-1} (X_{ij} - \mu_i)^2$
- Normalized: $\hat{X}_{ij} = \frac{X_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$
- Output: $Y_{ij} = \gamma_j \cdot \hat{X}_{ij} + \beta_j$

### Gradient Computations

**Gradient w.r.t. beta**:
$\frac{\partial L}{\partial \beta_j} = \sum_{i=0}^{M-1} \frac{\partial L}{\partial Y_{ij}}$

**Gradient w.r.t. gamma**:
$\frac{\partial L}{\partial \gamma_j} = \sum_{i=0}^{M-1} \frac{\partial L}{\partial Y_{ij}} \cdot \hat{X}_{ij}$

**Gradient w.r.t. input** (most complex):
$\frac{\partial L}{\partial X_{ij}} = \frac{\gamma_j}{\sqrt{\sigma_i^2 + \epsilon}} \left[ \frac{\partial L}{\partial Y_{ij}} - \frac{1}{N}\left(\frac{\partial L}{\partial \beta_j} + \hat{X}_{ij} \frac{\partial L}{\partial \gamma_j}\right) \right]$

Where the gradient w.r.t. input involves the normalized input values and requires careful handling of the mean and variance computations.

## Algorithmic Strategy: Multi-Pass Gradient Computation

The backward pass requires multiple reduction operations and careful coordination between gradient computations.

1.  **Pass 1: Compute Gamma and Beta Gradients**
    -   **Grid Scheduling**: Parallelize over features (`N` dimension).
    -   **Reduction per Feature**: For each feature `j`, reduce across the batch dimension (`M`) to compute:
        -   `grad_beta[j] = sum(grad_output[:, j])`
        -   `grad_gamma[j] = sum(grad_output[:, j] * x_normalized[:, j])`

2.  **Pass 2: Compute Input Gradients**
    -   **Grid Scheduling**: Parallelize over rows (`M` dimension).
    -   **Per-Row Computation**: For each row `i`:
        -   Read the previously computed `grad_beta` and `grad_gamma`
        -   Compute intermediate values needed for the input gradient formula
        -   Apply the complex gradient formula for each element in the row

3.  **Memory Management**: 
    -   Store intermediate statistics (mean, variance, normalized values) from forward pass or recompute them
    -   Use shared memory for efficient intra-block reductions
    -   Optimize memory access patterns for coalescing

## Source Code Organization

-   [`layernorm2d_bwd_xdl.cpp`](./layernorm2d_bwd_xdl.cpp): The main example file. It sets up the forward pass results, output gradients, and instantiates the `DeviceLayernormBwd` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_layernorm_bwd.hpp`](../../include/ck/tensor_operation/gpu/device/device_layernorm_bwd.hpp): The high-level device interface for layer normalization backward operations.
-   The underlying implementation coordinates multiple reduction kernels and gradient computation stages to efficiently compute all required gradients.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/53_layernorm2d_bwd
mkdir build && cd build

cmake \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_PREFIX_PATH="/opt/rocm;${CK_INSTALL_PATH}" \
  ..

make -j
```

### Run the Example
```bash
# Run the example with default settings
./layernorm2d_bwd_xdl

# Run with verification, data initialization, and timing
./layernorm2d_bwd_xdl 1 2 1
```

## Computational Complexity

The backward pass of layer normalization has similar computational complexity to the forward pass but requires additional memory for storing gradients:

-   **Time Complexity**: O(M × N) for each gradient computation
-   **Memory Complexity**: O(M × N) for input gradients plus O(N) for parameter gradients
-   **Numerical Stability**: Requires careful handling of the variance computation and division operations

## Role in Transformer Training

Layer normalization backward is crucial for training Transformer models:

-   **Gradient Flow**: Provides stable gradient propagation through normalization layers
-   **Parameter Updates**: Enables learning of the scale (`gamma`) and shift (`beta`) parameters
-   **Training Stability**: The normalization helps maintain stable gradients throughout the network
-   **Convergence**: Proper implementation is essential for achieving good convergence rates in Transformer training

The efficient implementation of this operation is critical for the overall training performance of large language models and other Transformer-based architectures.
