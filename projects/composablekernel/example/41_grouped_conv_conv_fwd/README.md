# Fused Grouped-Convolution -> Convolution Forward

This example demonstrates a **fused Grouped Convolution followed by a standard Convolution**. This pattern is specifically designed to optimize Depthwise Separable Convolutions, a key building block in many modern, efficient convolutional neural networks like MobileNet.

A Depthwise Separable Convolution consists of two stages:
1.  **Depthwise Convolution**: A grouped convolution where the number of groups is equal to the number of input channels (`groups = in_channels`). Each filter is applied to exactly one input channel.
2.  **Pointwise Convolution**: A standard `1x1` convolution that projects the channels from the depthwise stage into a new output channel space.

Fusing these two stages into a single kernel can provide significant performance benefits.

## Mathematical Formulation

The operation computes a chain of two convolutions, where the first is a grouped convolution.

1.  **First Convolution (Conv0 - Grouped/Depthwise)**:
    $D_{temp} = \text{GroupedConv}(\text{In}, \text{W0})$
    Where `In` is the input tensor and `W0` are the weights for the grouped convolution.

2.  **Second Convolution (Conv1 - Pointwise)**:
    $Out = \text{Conv}(\text{D}_{temp}, \text{W1})$
    Where `D_temp` is the output of the first stage and `W1` are the weights for the second convolution (typically `1x1` filters).

The critical optimization is that the intermediate tensor `D_temp` is **never written to global memory**. It is produced and consumed entirely within the GPU's on-chip memory (registers and LDS/shared memory), saving a massive amount of memory bandwidth.

## Algorithmic Strategy: Fused Implicit GEMM-GEMM via Shared Memory

The implementation maps the two-stage convolution into a fused GEMM-GEMM problem, using shared memory as the communication buffer between the stages.

1.  **Grid Scheduling**: The problem is parallelized across the thread blocks of the GPU. Each thread block is assigned to compute a tile of the final output tensor `Out`.

2.  **Fused Execution within a Thread Block**: To compute its output tile, a thread block must perform both convolution stages for the corresponding input region.
    -   **Compute Conv0 Tile**: The thread block first computes a tile of the intermediate tensor, $D_{temp}$, using the implicit GEMM algorithm for the grouped convolution. The result of this computation is stored directly into a designated region of **shared memory (LDS)**.
    -   **Synchronization**: A block-wide synchronization (`__syncthreads()`) is performed. This is a critical step that ensures the *entire* tile of $D_{temp}$ is visible to all threads in the block before the second convolution begins.
    -   **Compute Conv1 Tile**: The threads then immediately start computing the second convolution. They use the intermediate tile stored in shared memory as the input for this second stage, applying the implicit GEMM algorithm for the pointwise convolution. The result is accumulated in registers.
    -   **Store Final Result**: Once a tile of the final output `Out` is computed, it is written to global memory.

This "producer-consumer" pattern within a thread block is highly efficient. It treats shared memory as a fast, programmable cache for the intermediate tensor, completely avoiding the slow round-trip to global HBM memory that would be required by two separate kernel calls.

## Source Code Organization

-   [`grouped_conv_conv_fwd_xdl.cpp`](./grouped_conv_conv_fwd_xdl.cpp): The main example file. It sets up the input tensor and the two weight tensors (W0 for grouped, W1 for standard) and instantiates the `DeviceGroupedConvConvFwd` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_grouped_conv_conv_fwd.hpp`](../../include/ck/tensor_operation/gpu/device/device_grouped_conv_conv_fwd.hpp): The high-level device interface for the fused convolution operation.
-   The underlying grid-wise kernel implements the complex fusion logic, managing the two implicit GEMM calculations and the data flow through shared memory.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/41_grouped_conv_conv_fwd
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
./grouped_conv_conv_fwd_xdl

# Run with verification, data initialization, and timing
./grouped_conv_conv_fwd_xdl 1 2 1
```

## Application to Efficient CNNs

As mentioned, this kernel is a direct, high-performance implementation of a **Depthwise Separable Convolution**. This architectural primitive is the foundation of many efficient CNNs, including:
-   **MobileNets (V1, V2, V3)**: Designed for high performance on mobile and edge devices.
-   **EfficientNets**: A family of models that systematically scale model depth, width, and resolution to achieve high accuracy with fewer parameters and FLOPs.
-   **Xception**: A model that takes the idea of separable convolutions to an extreme.

By providing a fused kernel for this common pattern, Composable Kernel allows developers to achieve significantly better performance for these models than would be possible by calling a library for the depthwise and pointwise stages separately.
