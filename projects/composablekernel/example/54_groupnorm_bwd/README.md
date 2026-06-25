# Group Normalization Backward

This example demonstrates the **backward pass of Group Normalization**. This operation computes the gradients of the loss with respect to the input, gamma, and beta parameters of a group normalization layer, which is essential for training neural networks that use group normalization, particularly in computer vision applications where batch size independence is important.

## Mathematical Formulation

The backward pass of group normalization involves computing gradients for three components: input `X`, scale parameter `gamma`, and shift parameter `beta`.

Given:
- Input tensor `X` with shape `[N, C, H, W]`
- Number of groups `G` (where `C` must be divisible by `G`)
- Scale parameter `gamma` with shape `[C]`
- Shift parameter `beta` with shape `[C]`
- Output gradients `dL/dY` with shape `[N, C, H, W]`

From the forward pass, for each batch item `n` and group `g`:
- Channels in group: $S_g = \{c : c \text{ belongs to group } g\}$ where $|S_g| = C/G$
- Mean: $\mu_{ng} = \frac{1}{(C/G) \cdot H \cdot W} \sum_{c \in S_g} \sum_{h,w} X_{nchw}$
- Variance: $\sigma_{ng}^2 = \frac{1}{(C/G) \cdot H \cdot W} \sum_{c \in S_g} \sum_{h,w} (X_{nchw} - \mu_{ng})^2$
- Normalized: $\hat{X}_{nchw} = \frac{X_{nchw} - \mu_{ng}}{\sqrt{\sigma_{ng}^2 + \epsilon}}$ for $c \in S_g$
- Output: $Y_{nchw} = \gamma_c \cdot \hat{X}_{nchw} + \beta_c$

### Gradient Computations

**Gradient w.r.t. beta**:
$\frac{\partial L}{\partial \beta_c} = \sum_{n,h,w} \frac{\partial L}{\partial Y_{nchw}}$

**Gradient w.r.t. gamma**:
$\frac{\partial L}{\partial \gamma_c} = \sum_{n,h,w} \frac{\partial L}{\partial Y_{nchw}} \cdot \hat{X}_{nchw}$

**Gradient w.r.t. input** (most complex):
For channel `c` in group `g`:
$\frac{\partial L}{\partial X_{nchw}} = \frac{\gamma_c}{\sqrt{\sigma_{ng}^2 + \epsilon}} \left[ \frac{\partial L}{\partial Y_{nchw}} - \frac{1}{|S_g| \cdot H \cdot W}\left(\sum_{c' \in S_g} \frac{\partial L}{\partial \beta_{c'}} + \hat{X}_{nchw} \sum_{c' \in S_g} \frac{\partial L}{\partial \gamma_{c'}}\right) \right]$

## Algorithmic Strategy: Multi-Stage Group-wise Gradient Computation

The backward pass requires coordinated computation across groups with multiple reduction operations.

1.  **Pass 1: Compute Gamma and Beta Gradients**
    -   **Grid Scheduling**: Parallelize over channels (`C` dimension).
    -   **Reduction per Channel**: For each channel `c`, reduce across `N`, `H`, `W` dimensions:
        -   `grad_beta[c] = sum(grad_output[n, c, h, w])` over all `n, h, w`
        -   `grad_gamma[c] = sum(grad_output[n, c, h, w] * x_normalized[n, c, h, w])` over all `n, h, w`

2.  **Pass 2: Compute Group-wise Intermediate Values**
    -   **Grid Scheduling**: Parallelize over `(N, G)` pairs.
    -   **Group Reduction**: For each `(n, g)` pair:
        -   Sum `grad_beta` values for channels in group `g`
        -   Sum `grad_gamma` values for channels in group `g`
        -   These values are needed for the input gradient computation

3.  **Pass 3: Compute Input Gradients**
    -   **Grid Scheduling**: Parallelize over input tensor elements.
    -   **Per-Element Computation**: For each `(n, c, h, w)`:
        -   Identify which group `g` channel `c` belongs to
        -   Read the group-wise intermediate values from Pass 2
        -   Apply the complex input gradient formula

## Source Code Organization

-   [`groupnorm_bwd_xdl.cpp`](./groupnorm_bwd_xdl.cpp): The main example file. It sets up the forward pass results, output gradients, group configuration, and instantiates the `DeviceGroupnormBwd` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_groupnorm_bwd.hpp`](../../include/ck/tensor_operation/gpu/device/device_groupnorm_bwd.hpp): The high-level device interface for group normalization backward operations.
-   The underlying implementation coordinates multiple reduction and computation stages to efficiently handle the group-wise structure of the gradients.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/54_groupnorm_bwd
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
./groupnorm_bwd_xdl

# Run with verification, data initialization, and timing
./groupnorm_bwd_xdl 1 2 1
```

## Comparison with Other Normalization Backward Passes

| Normalization Type | Gradient Scope | Complexity | Memory Pattern |
|-------------------|----------------|------------|----------------|
| **BatchNorm** | Across batch for each channel | Medium | Channel-wise reductions |
| **LayerNorm** | Across features for each item | Medium | Per-sample reductions |
| **GroupNorm** | Across group for each (batch, group) | High | Group-wise reductions |
| **InstanceNorm** | Per channel per sample | Low | Independent computations |

## Applications in Computer Vision

Group normalization backward is particularly important for:

-   **Small Batch Training**: When batch sizes are too small for effective batch normalization
-   **Transfer Learning**: Fine-tuning pre-trained models with different batch sizes
-   **Object Detection**: Models like YOLO and R-CNN that benefit from batch-size independent normalization
-   **Segmentation Networks**: Dense prediction tasks where normalization stability is crucial
-   **Style Transfer**: Applications where group-wise feature normalization helps preserve style information

The group-wise structure provides a balance between the stability of batch normalization and the flexibility of layer normalization, making it valuable for many computer vision applications.
