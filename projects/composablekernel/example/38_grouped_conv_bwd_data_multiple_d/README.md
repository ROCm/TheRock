# Grouped Convolution Backward Data with Multiple Elementwise Inputs

This example demonstrates a **Grouped Convolution Backward Data Pass** fused with an elementwise operation that takes multiple auxiliary input tensors (`D` tensors). The backward data pass (also known as a transposed convolution or deconvolution) computes the gradient of the loss with respect to the convolution's *input* tensor. Fusing it with other operations is a powerful way to optimize the backward pass of a neural network.

## Mathematical Formulation

The operation computes the gradient with respect to the input (`GradIn`) of a grouped convolution, and then fuses the result with other tensors. For each group `g` from `0` to `G-1`:

1.  **Backward Data Convolution Stage**: A standard N-dimensional backward data convolution is performed for the group. This computes the gradient that should be propagated back to the input of the original forward-pass convolution.
    $GradIn_{temp[g]} = \text{ConvBwdData}(\text{GradOut}_{[g]}, \text{W}_{[g]})$
    Where `GradOut` is the gradient from the subsequent layer and `W` is the weight tensor from the forward pass.

2.  **Elementwise Stage**: The result of the backward convolution is combined with one or more auxiliary tensors ($D_{0[g]}, D_{1[g]}, \dots$) using a user-defined elementwise function `f`.
    $GradIn_{[g]} = f(GradIn_{temp[g]}, D_{0[g]}, D_{1[g]}, \dots)$

This fusion is particularly useful for operations like adding the gradient from a residual "skip" connection, which is a common pattern in modern network architectures. By fusing the addition, we avoid a separate kernel launch and a full read/write pass of the `GradIn` tensor.

## Algorithmic Strategy: Implicit Grouped GEMM with Fused Multi-D Epilogue

The implementation uses the implicit GEMM algorithm, but configured for the backward data pass.

1.  **Group Scheduling**: The `G` independent problems are distributed across the GPU's thread blocks. Each thread block is assigned to compute the fused backward convolution for one of the `G` groups.

2.  **Implicit GEMM for Backward Data**: The backward data convolution can be mathematically re-arranged to be equivalent to a forward convolution with transformed inputs and weights, which can then be solved with an implicit GEMM algorithm. Composable Kernel handles this transformation. A thread block executes the implicit GEMM for its assigned group, accumulating the `GradIn_temp` result in registers.

3.  **Fused Multi-D Epilogue**: Before writing the result to global memory, the epilogue performs the elementwise fusion:
    -   Threads load the corresponding tiles from the auxiliary `D` tensors for the assigned group.
    -   The user-defined elementwise function `f` is applied in registers to the computed gradient and the `D` tensor values.
    -   The final result `GradIn` for the group is written to global memory.

This strategy minimizes memory bandwidth by avoiding the materialization of the intermediate gradient tensor and maximizes parallelism by executing all groups concurrently.

## Source Code Organization

-   [`grouped_conv_bwd_data_multiple_d_xdl.cpp`](./grouped_conv_bwd_data_multiple_d_xdl.cpp): The main example file. It sets up the grouped backward convolution problem, including the multiple `D` tensors, and instantiates the `DeviceGroupedConvBwdDataMultipleD` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp`](../../include/ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp): The high-level device interface for this operation. It takes arrays of tensor descriptors, one for each group for each of the `D` tensors.
-   The underlying grid-wise kernel contains the logic to map thread blocks to groups and then execute the full implicit GEMM pipeline (formulated for backward data) with the fused multi-D epilogue for the assigned group.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/38_grouped_conv_bwd_data_multiple_d
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
./grouped_conv_bwd_data_multiple_d_xdl

# Run with verification, data initialization, and timing
./grouped_conv_bwd_data_multiple_d_xdl 1 2 1
```

## Applications in Backpropagation

Fusing operations into the backward pass is a critical optimization for training deep neural networks.

-   **Fused Residual Gradient**: In a residual block (`y = F(x) + x`), the gradient with respect to `x` is `dF/dx + dy/dx`. If `F` is a convolution, `dF/dx` is the output of the `ConvBwdData` operation. The `dy/dx` term (the gradient from the skip connection) can be passed as a `D` tensor and fused via an addition, computing the full gradient for `x` in a single kernel.
-   **Fused Gradient Clipping/Scaling**: The `D` tensors and the elementwise function `f` could be used to apply gradient scaling or other custom gradient processing steps directly to the output of the backward convolution, before the result is written back to memory.
