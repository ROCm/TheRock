# GEMM with K-Axis Splitting (Split-K GEMM)

This example demonstrates a General Matrix-Matrix Multiplication (GEMM) implemented with a **Split-K** algorithm. This is a technique used to increase the available parallelism for a single, large GEMM operation, which can lead to higher performance, especially on GPUs with a very large number of compute units.

## Mathematical Formulation

A standard GEMM computes the matrix product $C = A \times B$, where `A` has shape `[M, K]` and `B` has shape `[K, N]`. The computation is:
$C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}$

In a Split-K algorithm, the `K` dimension is split into `S` chunks of size `K_split = K / S`. The GEMM is then broken down into `S` smaller, partial GEMMs.

For each split `s` from `0` to `S-1`:
-   Let $A_s$ be the s-th slice of `A` along the K-axis (shape `[M, K_split]`).
-   Let $B_s$ be the s-th slice of `B` along the K-axis (shape `[K_split, N]`).
-   A partial product is computed: $C_s = A_s \times B_s$.

The final result `C` is the sum of all the partial products:
$C = \sum_{s=0}^{S-1} C_s = C_0 + C_1 + \dots + C_{S-1}$

## Algorithmic Strategy: Parallel Reduction of Partial GEMMs

The Split-K algorithm turns a single large GEMM into multiple smaller GEMMs whose results must be reduced (summed). This introduces a new axis of parallelism.

1.  **Splitting the K-Dimension**: The `K` dimension of the input matrices `A` and `B` is logically split into `S` parts. The `S` value is chosen by the kernel based on the problem size and hardware characteristics to expose a suitable amount of parallelism.

2.  **Parallel Partial GEMMs**: The `S` partial GEMMs are executed in parallel. The GPU's grid of thread blocks is now two-dimensional, mapping not only to the M and N dimensions of the output matrix `C`, but also to the `S` splits of the K dimension.
    -   A thread block is assigned to compute a tile of a *partial* product $C_s$.

3.  **Reduction of Partial Results**: The key challenge is how to sum the partial products $C_s$ efficiently.
    -   **Atomic Add**: The simplest method is for each block to compute its tile of $C_s$ and then use atomic add operations to accumulate its result directly into the final output matrix `C` in global memory. This is easy to implement but can suffer from high contention on the atomic operations, especially if many splits are trying to update the same memory location.
    -   **Two-Stage Reduction**: A more robust approach involves two stages:
        -   **Stage 1 (Partial Products)**: Each of the `S` parallel GEMMs writes its full partial product $C_s$ to a temporary workspace in global memory.
        -   **Stage 2 (Final Reduction)**: A separate reduction kernel is launched to sum the `S` partial products from the workspace into the final output matrix `C`.

Composable Kernel's implementation abstracts this complexity. The `DeviceGemmSplitK` interface handles the selection of the split factor `S`, the launch of the parallel partial GEMMs, and the final reduction step.

## Source Code Organization

-   [`splitk_gemm_xdl.cpp`](./splitk_gemm_xdl.cpp): The main example file. It sets up a standard GEMM problem and instantiates the `DeviceGemmSplitK` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_gemm_splitk.hpp`](../../include/ck/tensor_operation/gpu/device/device_gemm_splitk.hpp): The high-level device interface for the Split-K GEMM. It takes an additional `k_batch` parameter which controls the number of splits.
-   The underlying grid-wise kernel is modified to accept a `k_batch` index, so that each thread block knows which slice of the `A` and `B` matrices it is responsible for. It also includes the logic for the reduction (e.g., using atomic adds).

## Build and Run

### Prerequisites
Ensure the Composable Kernel library is built and installed.
```bash
cd /path/to/composable_kernel/build
make -j install
```

### Build the Example
```bash
cd /path/to/composable_kernel/example/35_splitK_gemm
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
./splitk_gemm_xdl

# Run with verification, data initialization, and timing
./splitk_gemm_xdl 1 2 1
```

## When is Split-K Useful?

Split-K is not always faster than a standard GEMM. It is most beneficial in specific scenarios:

-   **"Skinny" GEMMs**: For GEMMs where `M` and `N` are small but `K` is very large (e.g., `M=64, N=64, K=65536`). A standard GEMM might not generate enough parallel work to fill a large GPU. By splitting the large `K` dimension, we create many more independent work items, improving hardware utilization.
-   **Limited Shared Memory**: If a standard GEMM requires a very large tile size (and thus a large amount of shared memory) to be efficient, Split-K can be an alternative. It can use smaller tiles for the partial GEMMs, reducing the shared memory footprint per block.
-   **Load Balancing**: It can help with load balancing on heterogeneous hardware or in complex fused scenarios.

The trade-off is the overhead of the reduction step. The performance gain from increased parallelism must outweigh the cost of either atomic operations or writing and re-reading intermediate results.
