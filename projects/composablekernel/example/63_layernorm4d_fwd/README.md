# 4D Layer Normalization Forward

This example demonstrates the forward pass of **4D Layer Normalization**. This extends the layer normalization operation to 4-dimensional tensors, which is commonly used in computer vision applications where tensors have shape `[N, C, H, W]` and normalization is applied across the channel and spatial dimensions.

## Mathematical Formulation

Given a 4D input tensor `X` with shape `[N, C, H, W]`, 4D layer normalization computes an output tensor `Y` of the same shape. The normalization is performed independently for each batch item across the channel and spatial dimensions.

For each batch item `n` from `0` to `N-1`:
1.  **Compute Mean**: The mean is calculated across the channel (`C`) and spatial (`H`, `W`) dimensions.
    $\mu_n = \frac{1}{C \cdot H \cdot W} \sum_{c=0}^{C-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} X_{nchw}$

2.  **Compute Variance**: The variance is calculated across the same dimensions.
    $\sigma_n^2 = \frac{1}{C \cdot H \cdot W} \sum_{c=0}^{C-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} (X_{nchw} - \mu_n)^2$

3.  **Normalize**: The input is normalized using the computed mean and variance.
    $\hat{X}_{nchw} = \frac{X_{nchw} - \mu_n}{\sqrt{\sigma_n^2 + \epsilon}}$
    Where `epsilon` is a small constant for numerical stability.

4.  **Scale and Shift**: The normalized output is scaled by learnable parameter `gamma` and shifted by learnable parameter `beta`.
    $Y_{nchw} = \gamma_{chw} \cdot \hat{X}_{nchw} + \beta_{chw}$
    
    Note: The scale and shift parameters can have different granularities:
    - **Per-element**: `gamma` and `beta` have shape `[C, H, W]`
    - **Per-channel**: `gamma` and `beta` have shape `[C]` (broadcast over H, W)
    - **Global**: `gamma` and `beta` are scalars (broadcast over C, H, W)

## Algorithmic Strategy: Batch-Parallel Reduction with Spatial Aggregation

The implementation treats this as a parallel reduction problem with spatial aggregation for each batch item.

1.  **Grid Scheduling**: The `N` batch items are distributed among the GPU's thread blocks. Each block is assigned one or more batch items to normalize.

2.  **Spatial-Channel Reduction**: For each assigned batch item:
    -   **Cooperative Loading**: Threads within a block cooperatively read the 3D slice `X[n, :, :, :]` corresponding to their batch item.
    -   **Welford's Algorithm**: Use Welford's online algorithm to compute mean and variance across all `C × H × W` elements with good numerical stability.
    -   **Intra-Block Reduction**: Threads perform parallel reduction using shared memory to compute the final statistics for each batch item.

3.  **Normalization and Scale/Shift**: 
    -   **Elementwise Processing**: Each thread processes one or more elements of the batch item.
    -   **Apply Normalization**: Use the computed mean and variance to normalize each element.
    -   **Apply Scale/Shift**: Apply the appropriate `gamma` and `beta` values based on the parameterization choice.
    -   **Store Result**: Write the final normalized result to the output tensor.

## Source Code Organization

-   [`layernorm4d_fwd_xdl.cpp`](./layernorm4d_fwd_xdl.cpp): The main example file. It sets up the 4D input tensor, `gamma` and `beta` parameters, and instantiates the `DeviceLayernorm4dFwd` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_layernorm4d_fwd.hpp`](../../include/ck/tensor_operation/gpu/device/device_layernorm4d_fwd.hpp): The device interface for 4D layer normalization.
-   The underlying implementation uses reduction kernels optimized for the 4D tensor structure with efficient spatial dimension handling.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/63_layernorm4d_fwd
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
./layernorm4d_fwd_xdl

# Run with verification, data initialization, and timing
./layernorm4d_fwd_xdl 1 2 1
```

## Applications in Computer Vision

4D layer normalization has specific applications in computer vision tasks:

-   **Vision Transformers**: Some vision transformer variants apply layer normalization to 4D feature maps instead of flattening them.
-   **Style Transfer**: Normalizing feature maps across spatial and channel dimensions for style transfer applications.
-   **Feature Normalization**: Normalizing intermediate feature maps in CNNs for improved training stability.
-   **Attention Mechanisms**: Some spatial attention mechanisms benefit from normalized 4D feature representations.
-   **Multi-Scale Processing**: When processing features at different spatial scales, 4D layer normalization can provide consistent normalization.

## Comparison with Other Normalizations for 4D Tensors

| Normalization | Reduction Dimensions | Parameter Shape | Batch Dependence |
|---------------|---------------------|-----------------|------------------|
| **BatchNorm** | `[N, H, W]` per channel | `[C]` | Yes |
| **LayerNorm (2D)** | `[C, H, W]` per sample | `[C, H, W]` or `[C]` | No |
| **LayerNorm (4D)** | `[C, H, W]` per sample | `[C, H, W]` or variants | No |
| **InstanceNorm** | `[H, W]` per channel per sample | `[C]` | No |
| **GroupNorm** | Groups of channels per sample | `[C]` | No |

4D layer normalization provides batch-independent normalization while maintaining the spatial structure of the data, making it valuable for applications where spatial relationships are important.
