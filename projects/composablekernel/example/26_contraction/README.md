# Tensor Contraction

This example demonstrates a tensor contraction operation, which is a generalization of matrix multiplication to tensors of arbitrary rank (or number of dimensions). Tensor contractions are fundamental to many algorithms in physics, chemistry, and machine learning, particularly in the field of tensor networks.

## Mathematical Formulation

A tensor contraction sums the product of two tensors over a specified set of indices. It is most clearly expressed using Einstein summation notation (einsum).

For example, a standard matrix multiplication $C_{ik} = \sum_j A_{ij} B_{jk}$ is written in einsum notation as:
`ik = ij,jk`

A tensor contraction can involve more dimensions and more contracted indices. For instance, contracting a 3D tensor `A` with a 4D tensor `B`:
$D_{imn} = \sum_{j,k} A_{ijk} B_{kjmn}$
In einsum notation, this is:
`imn = ijk,kjmn`

Here, the `j` and `k` indices are the "contracted" or "summation" indices, while `i`, `m`, and `n` are the "free" or "output" indices.

Composable Kernel's contraction operation can perform any such contraction, provided there is a clear distinction between contracted indices and free indices for each tensor.

## Algorithmic Strategy: Mapping Contraction to GEMM

The dominant strategy for performing tensor contractions efficiently on GPUs is to reshape or "flatten" the input tensors into 2D matrices, perform a standard, highly-optimized GEMM, and then reshape the resulting matrix back into the desired output tensor shape.

1.  **Tensor-to-Matrix Reshaping**:
    -   The dimensions of each input tensor are partitioned into two sets: the contracted dimensions and the free (non-contracted) dimensions.
    -   The tensor is then treated as a 2D matrix by flattening all the free dimensions into the "row" dimension (M for tensor A, N for tensor B) and all the contracted dimensions into the "column" dimension (K).
    -   For example, in the contraction `imn = ijk,kjmn`:
        -   Tensor A (`ijk`): Free index is `i`, contracted indices are `jk`. It is reshaped into a matrix A' of shape `[i, (j*k)]`.
        -   Tensor B (`kjmn`): Free indices are `mn`, contracted indices are `kj`. It is reshaped into a matrix B' of shape `[(k*j), (m*n)]`.
        -   The GEMM computes `D' = A' x B'`. The resulting matrix D' has shape `[i, (m*n)]`.

2.  **High-Performance GEMM**: A standard, block-tiled GEMM kernel is used to perform the matrix multiplication `A' x B'`. This is the computationally intensive part of the operation.

3.  **Output Reshaping**: The resulting 2D matrix `D'` is then logically reshaped back into the desired multi-dimensional output tensor `D` of shape `[i, m, n]`.

Crucially, the reshaping operations are often *logical*. The data is not physically moved or transposed in global memory. Instead, the GEMM kernel is provided with a "tensor descriptor" that understands the original N-dimensional layout and can calculate the correct memory addresses for the flattened 2D view on the fly. This avoids costly data movement and is key to performance.

## Source Code Organization

-   [`contraction_xdl.cpp`](./contraction_xdl.cpp): The main example file. It defines the input tensors and their layouts, specifies the contraction indices, and instantiates the `DeviceContraction` operation.
-   [`../../include/ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp`](../../include/ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp): The high-level device interface for the contraction operation. It is highly generic and takes tensor descriptors that define the complex layouts and index mappings.
-   The device interface internally creates a plan to map the contraction to a GEMM, then calls a standard `DeviceGemm` instance to execute it. The intelligence lies in how the tensor descriptors are configured to present a 2D matrix view of the higher-dimensional tensor data to the underlying GEMM kernel.

## Build and Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build the Example
```bash
cd /path/to/composable_kernel/example/26_contraction
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
./contraction_xdl

# Run with verification, data initialization, and timing
./contraction_xdl 1 2 1
```

## Applications

Tensor contractions are the core computational primitive in a wide range of fields:

-   **Tensor Network Methods**: In physics and chemistry, methods like DMRG (Density Matrix Renormalization Group) and PEPS (Projected Entangled Pair States) use networks of interconnected tensors to represent complex quantum states. The simulation of these systems involves sequences of tensor contractions.
-   **High-Order Statistics**: In data analysis, computing higher-order moments (like skewness or kurtosis) can be expressed as tensor contractions.
-   **Relativistic Physics**: Many equations in general relativity are expressed in the language of tensors and involve contractions.
-   **Advanced Deep Learning Models**: Some research models, particularly in areas like quantum machine learning or geometric deep learning, use tensor contractions as a primary layer type, going beyond the capabilities of standard matrix multiplication.
