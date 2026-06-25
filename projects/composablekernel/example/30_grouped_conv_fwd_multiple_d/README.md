# Grouped Convolution Forward with Multiple Elementwise Inputs

This example demonstrates a **Grouped Convolution Forward Pass** fused with an elementwise operation that takes multiple auxiliary input tensors (`D` tensors). This is a powerful fusion that combines the parallel structure of grouped convolutions with the ability to merge subsequent elementwise layers, such as custom activations or residual connections, into a single kernel.

## Mathematical Formulation

This operation performs `G` independent fused convolution operations in parallel, where `G` is the group count. For each group `g` from `0` to `G-1`:

1.  **Convolution Stage**: A standard N-dimensional forward convolution is performed for the group.
    $C_{out[g]} = \text{Conv}(\text{In}_{[g]}, \text{W}_{[g]})$

2.  **Elementwise Stage**: The result of the convolution is combined with one or more auxiliary tensors ($D_{0[g]}, D_{1[g]}, \dots$) using a user-defined elementwise function `f`.
    $E_{[g]} = f(C_{out[g]}, D_{0[g]}, D_{1[g]}, \dots)$

The key optimization is that the intermediate convolution result, $C_{out[g]}$, is never written to global memory. It is computed and held in registers, then immediately consumed by the elementwise part of the kernel's epilogue before the final result `E` is stored.

## Algorithmic Strategy: Implicit Grouped GEMM with Fused Multi-D Epilogue

The implementation combines three core concepts: the implicit GEMM transformation for convolutions, the group-parallel scheduling of Grouped GEMM, and a multi-input fused epilogue.

1.  **Group Scheduling**: The `G` independent problems are distributed across the GPU's thread blocks. Each thread block is assigned to compute the fused convolution for one of the `G` groups.

2.  **Implicit GEMM Core**: Once a thread block is assigned a group `g`, it executes the convolution for that group using the implicit GEMM algorithm. This involves:
    -   Calculating the base memory addresses for the group's input tensors: $\text{In}_{[g]}, \text{W}_{[g]}, D_{0[g]}, \dots, E_{[g]}$.
    -   Performing a tiled GEMM, where tiles of the input `In` and weights `W` are read (with the `im2col` transformation happening on-the-fly) and the result is accumulated in registers.

3.  **Fused Multi-D Epilogue**: Before writing the result to global memory, the epilogue performs the elementwise fusion:
    -   Threads load the corresponding tiles from the auxiliary `D` tensors for the assigned group.
    -   The user-defined elementwise function `f` is applied in registers to the convolution result and the `D` tensor values.
    -   The final result `E` for the group is written to global memory.

This strategy is highly efficient as it minimizes memory bandwidth by avoiding the materialization of the intermediate convolution output and maximizes parallelism by executing all groups concurrently.

## Source Code Organization

-   [`grouped_conv_fwd_multiple_d_xdl.cpp`](./grouped_conv_fwd_multiple_d_xdl.cpp): The main example file. It sets up the grouped convolution problem, including the multiple `D` tensors, and instantiates the `DeviceGroupedConvFwdMultipleD` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_d.hpp`](../../include/ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_d.hpp): The high-level device interface for this operation. It takes arrays of tensor descriptors, one for each group for each of the `D` tensors.
-   The underlying grid-wise kernel contains the logic to map thread blocks to groups and then execute the full implicit GEMM pipeline with the fused multi-D epilogue for the assigned group.

## Build and Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build the Example
```bash
cd /path/to/composable_kernel/example/30_grouped_conv_fwd_multiple_d
mkdir build && cd build

cmake \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_PREFIX_PATH="/opt/rocm;${CK_INSTALL_PATH}" \
  ..

make -j
```

### Run the Example

```bash
arg1: verification (0=no, 1=yes)
arg2: initialization (0=no init, 1=integer value, 2=decimal value)
arg3: time kernel (0=no, 1=yes)
Following arguments (depending on number of spatial dims):
 Number of spatial dimensions (1=Conv1D, 2=Conv2D, 3=Conv3D)
 G, N, K, C,
 <filter spatial dimensions>, (ie Y, X for 2D)
 <input image spatial dimensions>, (ie Hi, Wi for 2D)
 <strides>, (ie Sy, Sx for 2D)
 <dilations>, (ie Dy, Dx for 2D)
 <left padding>, (ie LeftPy, LeftPx for 2D)
 <right padding>, (ie RightPy, RightPx for 2D)

./bin/example_grouped_conv_fwd_bias_relu_add_xdl_fp16 1 1 1
```

## Applications

This kernel is ideal for optimizing layers in modern CNNs that use grouped convolutions followed by complex activations or residual connections.

-   **Fused Residual Connections**: A common pattern is `Conv(x) + x`. This can be implemented by passing the input `x` as a `D` tensor and defining the elementwise function as `f(conv_out, d0) = conv_out + d0`. If this is a grouped convolution, this kernel is a perfect fit.
-   **Custom Gated Activations**: Some architectures use gated activations, such as `Conv_A(x) * sigmoid(Conv_B(x))`. While this kernel doesn't compute two convolutions, it can fuse one convolution with an elementwise multiplication against another tensor. For example, it could compute `Conv_A(x) * D0`, where `D0` is the pre-computed `sigmoid(Conv_B(x))`.
-   **Depthwise Separable Convolutions**: These layers consist of a depthwise convolution (a grouped convolution with `G = C`) followed by a pointwise convolution (`1x1` conv). If there is a residual connection or other elementwise operation after the depthwise stage, this kernel can fuse it directly, improving the performance of this widely used building block.
