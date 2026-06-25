# Multiple Reductions

This example demonstrates a **Multiple Reduction** operation, where several different reduction computations (e.g., sum, average, max, min) are performed on the same input tensor in a single kernel launch. This is a highly efficient pattern when multiple statistics are needed for a tensor, as it requires only one read pass over the (potentially very large) input data.

## Mathematical Formulation

Given an input tensor `A`, this operation computes a set of output scalars or vectors, $\{R_0, R_1, \dots, R_N\}$, where each $R_i$ is the result of a different reduction operation applied to `A`.

$R_0 = \bigoplus_0 A$
$R_1 = \bigoplus_1 A$
...
$R_N = \bigoplus_N A$

Where $\bigoplus_i$ represents a distinct reduction operation, such as:
-   `sum`: $\sum_j A_j$
-   `avg`: $\frac{1}{N} \sum_j A_j$
-   `max`: $\max_j(A_j)$
-   `min`: $\min_j(A_j)$
-   `sum of squares`: $\sum_j A_j^2$

The reductions can be performed over the entire tensor to produce a scalar, or along specific dimensions to produce a lower-rank tensor.

## Algorithmic Strategy: Fused Parallel Reduction

The implementation uses a classic parallel reduction algorithm but extends it to handle multiple reduction functions simultaneously.

1.  **Grid Scheduling**: The input tensor is partitioned across the GPU's thread blocks. Each block is responsible for reducing a slice of the input data.

2.  **Intra-Block Reduction**:
    -   **Loading**: Threads within a block cooperatively load their assigned slice of the input tensor `A` into shared memory.
    -   **Fused Accumulation**: Each thread maintains a separate set of accumulators in its private registers, one for each of the `N` reduction operations being performed.
    -   As threads iterate through the data in shared memory, they update all of their accumulators simultaneously. For example, for each element `a`, a thread might update its `sum_accumulator += a`, `max_accumulator = max(max_accumulator, a)`, and `sum_sq_accumulator += a*a`.
    -   **Tree-Based Reduction**: After processing all elements in the slice, the threads perform a parallel reduction using shared memory. This is done *for each of the N reduction types*. For example, they first reduce all the `sum_accumulator` values to get the block's partial sum, then they reduce all the `max_accumulator` values to get the block's partial max, and so on.

3.  **Inter-Block Reduction**:
    -   Each thread block writes its `N` partial results (the block's partial sum, partial max, etc.) to `N` separate temporary arrays in global memory.
    -   A final, small reduction kernel is launched (or atomic operations are used) for each of the `N` temporary arrays to combine the partial results from all blocks into the final `N` output values.

The key to this kernel's efficiency is that the expensive part—reading the input tensor `A` from global memory—is only done once. All subsequent computations happen on-chip.

## Source Code Organization

-   [`multiple_reduce_xdl.cpp`](./multiple_reduce_xdl.cpp): The main example file. It sets up the input tensor and defines the multiple reduction operations to be performed. It then instantiates the `DeviceMultipleReduce` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_multiple_reduce.hpp`](../../include/ck/tensor_operation/gpu/device/device_multiple_reduce.hpp): The high-level device interface for the multiple reduction operation. It takes a tuple of structs, where each struct defines one of the reduction operations to be performed.
-   [`../../include/ck/tensor_operation/gpu/grid/gridwise_multiple_reduce.hpp`](../../include/ck/tensor_operation/gpu/grid/gridwise_multiple_reduce.hpp): The grid-wise kernel that implements the fused parallel reduction algorithm. It is heavily templated to generate the specific accumulation and reduction logic for the requested set of operations.

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/33_multiple_reduce
mkdir build && cd build

cmake \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_PREFIX_PATH="/opt/rocm;${CK_INSTALL_PATH}" \
  ..

make -j
```

### Run  ```example_dual_reduce_multiblock```

```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg2: time kernel (0=no, 1=yes) 
./bin/example_dual_reduce_multiblock -D 600,28,28,256 -v 1 2 1
```

Result
```
./bin/example_dual_reduce_multiblock -D 600,28,28,256 -v 1 2 1                        
launch_and_time_kernel: grid_dim {150, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 1.19529 ms, 201.499 GB/s, DeviceMultipleReduceBlockWise<256,M_C4_S1,K_C64_S1,InSrcVectorDim_1_InSrcVectorSize_1,OutDstVectorSize_1_1>
```

### Run ```example_dual_reduce_threadwise```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg2: time kernel (0=no, 1=yes)
./bin/example_dual_reduce_multiblock -D 8000,4,4,4 -v 1 2 1
```

Result
```
./bin/example_dual_reduce_threadwise -D 8000,4,4,4 -v 1 2 1
launch_and_time_kernel: grid_dim {32, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 0.01512 ms, 71.9577 GB/s, DeviceMultipleReduceThreadwise<256,M_C256_S1,K_C1_S4,InSrcVectorDim_1_InSrcVectorSize_2,OutDstVectorSize_1_1>
```

## Applications

This operation is extremely useful for computing statistics and implementing normalization layers.

-   **Normalization Layers**: Both Batch Normalization and Layer Normalization require computing the mean and variance of a tensor. Variance is defined as $\sigma^2 = E[X^2] - (E[X])^2$. This requires two statistics: the sum of elements (for the mean, $E[X]$) and the sum of squares of elements (for $E[X^2]$). This kernel can compute both in a single pass, making it a highly efficient way to calculate the moments needed for normalization.
-   **Data Analytics**: When analyzing a large dataset, one might want to compute its min, max, mean, and standard deviation all at once. This kernel can perform all the necessary underlying reductions in a single, efficient operation.
-   **Loss Function Components**: Some complex loss functions might involve multiple statistical properties of a model's output. This kernel can compute them efficiently.
