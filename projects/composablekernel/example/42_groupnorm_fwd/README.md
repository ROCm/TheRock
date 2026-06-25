# Group Normalization Forward

This example demonstrates the forward pass of **Group Normalization (GroupNorm)**. GroupNorm is a normalization technique that acts as a bridge between Layer Normalization and Instance Normalization. It divides channels into groups and computes the mean and variance for normalization within each group. This makes its performance stable across a wide range of batch sizes, unlike BatchNorm.

## Mathematical Formulation

Given an input tensor `X` with shape `[N, C, H, W]` and a specified number of groups `G`:
The `C` channels are divided into `G` groups, with each group containing `C/G` channels. The normalization is performed independently for each group within each batch item.

For each batch item `n` and each group `g`:
1.  **Identify Channels**: Identify the set of channels belonging to group `g`. Let this set be $S_g$. The size of this set is $C' = C/G$.

2.  **Compute Mean**: The mean is calculated across the channels in the group and the spatial dimensions (`H`, `W`).
    $\mu_{ng} = \frac{1}{C' \cdot H \cdot W} \sum_{c \in S_g} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} X_{nchw}$

3.  **Compute Variance**: The variance is also calculated across the same dimensions.
    $\sigma_{ng}^2 = \frac{1}{C' \cdot H \cdot W} \sum_{c \in S_g} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} (X_{nchw} - \mu_{ng})^2$

4.  **Normalize**: The input is normalized using the computed mean and variance for its corresponding group. For any channel `c` in group `g`:
    $\hat{X}_{nchw} = \frac{X_{nchw} - \mu_{ng}}{\sqrt{\sigma_{ng}^2 + \epsilon}}$
    Where `epsilon` is a small constant for numerical stability.

5.  **Scale and Shift**: The normalized output is scaled by a learnable parameter `gamma` and shifted by a learnable parameter `beta`. Unlike BatchNorm, `gamma` and `beta` are applied per-channel, not per-group.
    $Y_{nchw} = \gamma_c \cdot \hat{X}_{nchw} + \beta_c$
    Both `gamma` and `beta` are vectors of shape `[C]`.

## Algorithmic Strategy: Two-Pass Parallel Reduction per Group

The implementation of GroupNorm is a parallel reduction problem, similar to LayerNorm and BatchNorm, but with a different scope for the reduction.

1.  **Grid Scheduling**: The `N * G` independent normalization problems (one for each batch item and each group) are distributed among the GPU's thread blocks. Each block is assigned one or more `(n, g)` pairs to normalize.

2.  **Pass 1: Compute Moments (Mean and Variance)**
    -   For an assigned `(n, g)` pair, the threads within a block cooperatively read the data for the channels in that group and the spatial dimensions.
    -   **Welford's Algorithm**: To compute mean and variance in a single pass with good numerical stability, Welford's online algorithm is used.
    -   **Intra-Block Reduction**: The threads perform a parallel reduction using shared memory to compute the final mean and variance for the `(n, g)` pair.
    -   The final mean and variance for each `(n, g)` pair are written to temporary arrays in global memory.

3.  **Pass 2: Normalize, Scale, and Shift**
    -   A second kernel (or a second stage in the same kernel after a grid-wide sync) is launched.
    -   Threads read the input data `X` again.
    -   For each element `X_nchw`, the thread identifies its group `g`, reads the corresponding mean `mu_ng` and variance `sigma_ng`, and applies the normalization formula.
    -   It then reads the per-channel `gamma_c` and `beta_c` values and applies the scale and shift.
    -   The final result `Y` is written to global memory.

Composable Kernel encapsulates this two-pass logic into a single, efficient `DeviceGroupnormFwd` operation.

## Source Code Organization

-   [`groupnorm_fwd_xdl.cpp`](./groupnorm_fwd_xdl.cpp): The main example file. It sets up the input tensor, `gamma` and `beta` vectors, the number of groups, and instantiates the `DeviceGroupnormFwd` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_groupnorm_fwd.hpp`](../../include/ck/tensor_operation/gpu/device/device_groupnorm_fwd.hpp): The high-level device interface for the GroupNorm forward pass.
-   The implementation internally uses a reduction kernel based on Welford's algorithm to compute the statistics and an elementwise kernel to apply the normalization.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/42_groupnorm_fwd
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
./groupnorm_fwd_xdl

# Run with verification, data initialization, and timing
./groupnorm_fwd_xdl 1 2 1
```

## Comparison of Normalization Layers

-   **BatchNorm**: Normalizes over `(N, H, W)`. Learns `gamma` and `beta` per channel `C`. Batch-size dependent.
-   **LayerNorm**: Normalizes over `(C, H, W)`. Learns `gamma` and `beta` per channel `C`. Batch-size independent.
-   **InstanceNorm**: Normalizes over `(H, W)`. Learns `gamma` and `beta` per channel `C`. A special case of GroupNorm where `G=C`.
-   **GroupNorm**: Normalizes over `(C/G, H, W)`. Learns `gamma` and `beta` per channel `C`. Batch-size independent.

GroupNorm's flexibility has made it popular in GANs and in Transformer-based vision models where batch sizes can be small.
