# 2D Max Pooling Backward

This example demonstrates the **backward pass of 2D max pooling**. This operation computes the gradient of the loss with respect to the input of a max pooling layer, which is essential for training convolutional neural networks that use max pooling for downsampling.

## Mathematical Formulation

The backward pass of max pooling propagates gradients only to the input positions that contributed to the maximum value in each pooling window.

Given:
- Input tensor `X` with shape `[N, C, H_in, W_in]`
- Output gradients `dL/dY` with shape `[N, C, H_out, W_out]`
- Pooling parameters: window size `(pool_h, pool_w)`, stride `(stride_h, stride_w)`, padding `(pad_h, pad_w)`

The backward pass computes input gradients `dL/dX` with the same shape as `X`.

For each pooling window, the gradient flows only to the position that had the maximum value:
$\frac{\partial L}{\partial X_{nchw}} = \sum_{\text{windows containing } (h,w)} \frac{\partial L}{\partial Y_{nch'w'}} \cdot \mathbf{1}[\text{argmax}_{(h'',w'')} X_{nch''w''} = (h,w)]$

Where the indicator function $\mathbf{1}[\cdot]$ is 1 if the position `(h,w)` was the argmax in its corresponding pooling window, and 0 otherwise.

## Algorithmic Strategy: Parallel Gradient Routing

The backward pass requires determining which input positions were selected during the forward pass and routing gradients accordingly.

1.  **Grid Scheduling**: The computation can be parallelized over either the input or output tensor elements, depending on the implementation strategy.

2.  **Argmax Information**: There are two main approaches to handle the argmax information:
    -   **Recomputation**: Recompute the argmax during the backward pass by examining each pooling window.
    -   **Stored Indices**: Use precomputed argmax indices from the forward pass (more memory efficient for multiple backward passes).

3.  **Gradient Routing Algorithm** (using recomputation approach):
    -   **Initialize**: Set all input gradients to zero.
    -   **For each output position**: Each thread processes one output gradient position `(n, c, h_out, w_out)`.
    -   **Find Input Window**: Calculate the corresponding input window based on stride and padding.
    -   **Recompute Argmax**: Find the position with the maximum value in the input window.
    -   **Route Gradient**: Add the output gradient to the input position that had the maximum value (using atomic operations if necessary).

4.  **Memory Access Optimization**: The kernel optimizes for:
    -   Coalesced access to gradient tensors
    -   Efficient atomic operations for gradient accumulation
    -   Minimal redundant computation of argmax positions

## Source Code Organization

-   [`maxpool2d_bwd_xdl.cpp`](./maxpool2d_bwd_xdl.cpp): The main example file. It sets up the input tensor, output gradients, pooling parameters, and instantiates the `DeviceMaxpool2dBwd` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_maxpool2d_bwd.hpp`](../../include/ck/tensor_operation/gpu/device/device_maxpool2d_bwd.hpp): The high-level device interface for 2D max pooling backward operations.
-   [`../../include/ck/tensor_operation/gpu/grid/gridwise_maxpool2d_bwd.hpp`](../../include/ck/tensor_operation/gpu/grid/gridwise_maxpool2d_bwd.hpp): The grid-wise kernel implementing the gradient routing algorithm.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/49_maxpool2d_bwd
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
./maxpool2d_bwd_xdl

# Run with verification, data initialization, and timing
./maxpool2d_bwd_xdl 1 2 1
```

## Computational Characteristics

Max pooling backward has unique characteristics compared to other CNN operations:

-   **Sparse Gradient Flow**: Unlike convolution or dense layers where gradients flow to all inputs, max pooling creates sparse gradient patterns where only selected input positions receive gradients.
-   **Memory-bound Operation**: The operation is typically memory-bound rather than compute-bound, as it involves reading gradients and writing results with minimal arithmetic.
-   **Atomic Operations**: When multiple output positions map to the same input position, atomic operations may be needed to correctly accumulate gradients.

## Relationship to Forward Pass

The backward pass must be consistent with the forward pass implementation:
- The same tie-breaking rules for equal maximum values
- Identical handling of padding and boundary conditions
- Consistent stride and window size interpretation

This ensures that the computed gradients correctly reflect the actual forward pass computation, which is essential for proper gradient-based optimization.
