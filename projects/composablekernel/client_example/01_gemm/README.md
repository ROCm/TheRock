# Client Example: Basic GEMM

## Theory

This client example demonstrates a basic **GEMM (General Matrix Multiplication)** operation using the Composable Kernel library. GEMM is a core operation in linear algebra and deep learning, computing the product of two matrices and optionally adding a bias or scaling.

**Mathematical Formulation:**
$$
C = \alpha (A \times B) + \beta D
$$
- $A$: [M, K] input matrix
- $B$: [K, N] weight matrix
- $D$: [M, N] optional bias or residual
- $C$: [M, N] output
- $\alpha, \beta$: scalars (often 1.0, 0.0)

**Algorithmic Background:**
- The operation is implemented using a tiled/blocking strategy for memory efficiency.
- GEMM is the computational backbone for transformer attention, MLPs, and CNNs (via im2col).

## GEMM
General matrix multiplications operation. In CK GEMM operation is called as `DeviceGemm` and requires following types as template parameters:

* **ALayout** - A matrix layout (RowMajor/ColumnMajor).
* **BLayout** - B matrix layout (RowMajor/ColumnMajor).
* **CLayout** - B matrix layout (RowMajor/ColumnMajor).
* **ADataType** - A matrix data type.
* **BDataType** - B matrix data type.
* **CDataType** - B matrix data type.
* **AElementwiseOperation** - Fused operation on tensor A before GEMM.
* **BElementwiseOperation** - Fused operation on tensor B before GEMM.
* **CElementwiseOperation** - Fused operation on tensor C after GEMM.

For matrices with large K dimension `DeviceGemmSplitK` implementation is available. This implementation allows user to split K dimension between work groups. This implementation uses `AtomicAdd` operation on global memory, thus need to zero-out output buffer for correct results.

For fused operations with additional tensor there are `DeviceGemmMultipleABD` or `DeviceGemmMultipleD` operation which require following parameters:
* **DsLayout** - layouts for additional tensors for fused operations.
* **DsDataType** - data types for additional tensors for fused operations.

For `DeviceGemmMultipleABD` **ALayout**, **BLayout**, **ADataType** and **BDataType** user should pass a tuple.

List of the device operations in CK:

* **DeviceGemmDl** - Device operation with DL instructions.
* **DeviceGemmDpp** - Device operation with DL instructions with DPP instructions during data load.
* **DeviceGemmWmma_CShuffle** - Device operation with WMMA instructions with CShuffle optimization for more optimized data store.
* **DeviceGemm_Xdl_CShuffle_LdsDirectLoad** - Device operation with XDL instructions and CShuffle optimization for more optimized data store and direct load from global memory to shared memory.
* **DeviceGemm_Xdl_CShuffle** - Device operation with XDL instructions with CShuffle optimization for more optimized data store.
* **DeviceGemm_Xdl_CShuffleV2** - Device operation with XDL instructions with CShuffle optimization for more optimized data store. GEMM pipeline has been optimized compared to **DeviceGemm_Xdl_CShuffle**.
* **DeviceGemmXdlSkipBLds** - Device operation with XDL instructions. Load to shared memory has been skiped for B matrix.
* **DeviceGemm_Xdl_WaveletModel_CShuffle** - Device operation with XDL instructions with CShuffle optimization for more optimized data store. Producer and consumer scheme cooperation between waves in workgroup.
* **DeviceGemmXdl** - Device operation with XDL instructions.

Table of supported cases by instance factory with XDL instruction for Row/Row/Row, Row/Column/Row, Column/Row/Row or Column/Column/Row:

|       |Is supported|
|-------|---|
|bf16|&check;|
|fp16|&check;|
|fp32|&check;|
|int8|&check;|
|fp8 |&check;|

Table of supported cases by instance factory with WMMA instruction for Row/Row/Row, Row/Column/Row, Column/Row/Row or Column/Column/Row:

|       |Is supported|
|-------|---|
|bf16|&check;|
|fp16|&check;|
|fp32|&cross;|
|int8|&check;|
|fp8 |&cross;|

Table of supported cases by instance factory with DL instruction for Row/Row/Row, Row/Column/Row, Column/Row/Row or Column/Column/Row:

|       |Is supported|
|-------|---|
|bf16|&cross;|
|fp16|&check;|
|fp32|&check;|
|int8|&check;|
|fp8 |&cross;|

Table of supported cases by instance factory with fused output elementwise operation:

* **B Matrix Multiply + Add + Gelu** - bf16 (int8 for B matrix)
* **B Matrix Multiply + Add** - bf16 (int8 for B matrix)
* **B Matrix Multiply + Gelu** - bf16 (int8 for B matrix)
* **B Matrix Multiply** - bf16 (int8 for B matrix)

* **Add + Add + Gelu** - fp16
* **Add + Gelu** - fp16, bf16 (int8 for B matrix) for Row/Column/Row
* **Multiply** - fp16
* **Add + Multiply** - fp16
* **Add + Relu** - fp16 (int8 for B matrix)  for Row/Column/Row, bf16 (int8 for B matrix) for Row/Column/Row
* **Add + Silu** - fp16 (int8 for B matrix)  for Row/Column/Row, bf16 (int8 for B matrix) for Row/Column/Row
* **Add** - fp16 (int8 for B matrix)  for Row/Column/Row, bf16 (int8 for B matrix) for Row/Column/Row
* **Bilinear** - fp16, int8
* **Gelu** - fp16
* **Multiply + Add** - fp16 for Row/Column/Row and Row/Row/Row, fp16 (int8 for B matrix, fp32 for Bias) for Row/Column/Row and Row/Row/Row, 
* **Quantization** - int8

## GEMM V2 (Universal GEMM)
General matrix multiplications operation optimized for MI300 series. Operation is called as `DeviceGemmV2` and requires following types as template parameters:

* **ALayout** - A matrix layout (RowMajor/ColumnMajor).
* **BLayout** - B matrix layout (RowMajor/ColumnMajor).
* **CLayout** - B matrix layout (RowMajor/ColumnMajor).
* **ADataType** - A matrix data type.
* **BDataType** - B matrix data type.
* **CDataType** - B matrix data type.
* **AElementwiseOperation** - Fused operation on tensor A before GEMM.
* **BElementwiseOperation** - Fused operation on tensor B before GEMM.
* **CElementwiseOperation** - Fused operation on tensor C after GEMM.

This implementation allows user to split K dimension between work groups. This implementation requires AtomicAdd operation on global memory (output buffer must be set to zeroes if splitK parameter is larger than one).

List of the device operations for in CK:

* **DeviceGemm_Xdl_CShuffleV3** - Device operation with XDL instructions with CShuffle optimization for more optimized data store.
* **DeviceGemm_Xdl_CShuffleV3R1** - Device operation with XDL instructions with CShuffle optimization for more optimized data store. This implementation perform reduction on splitted K dimension after GEMM instead of AtomicAdd instruction. 

Table of supported cases by instance factory with XDL instruction for Row/Row/Row, Row/Column/Row, Column/Row/Row or Column/Column/Row:

|       |Is supported|
|-------|---|
|bf16|&check;|
|fp16|&check;|
|fp32|&cross;|
|int8|&cross;|
|fp8 (C bf16)|&check;|
|fp16 (A fp8)|&check;|
|fp16 (B fp8)|&check;|

## Others

* **DeviceGemm_dequantB** - GEMM with dequantization (implemented with WMMA instructions).
* **DeviceGemmMultipleD_ABScale** - GEMM with scale for A and B matrix.
* **DeviceGemmMultipleDLayernorm** - GEMM fused with layernorm.
* **DeviceGemmMultipleDMultipleR** - GEMM fused with reductions and custom global reductions operators.
* **DeviceGemmReduce** - GEMM fused with reduction.
* **DeviceGemm_Streamk_V2** - GEMM stream K implementation. Implementation allows to use reduction instead of AtomicAdd.
* **DeviceGemmStreamK** - GEMM stream K implementation using AtomicAdd.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/01_gemm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./gemm
```

## Source Code Structure

### Directory Layout
```
client_example/01_gemm/
├── gemm.cpp         # Main client example: sets up, runs, and verifies GEMM
├── CMakeLists.txt   # Build configuration for the example
```

### Key Functions

- **main()** (in `gemm.cpp`):  
  Sets up input matrices, configures GEMM parameters, launches the GEMM kernel, and verifies the result.
- **GEMM kernel invocation**:  
  Uses the Composable Kernel device API to launch the GEMM operation.

This client example provides a minimal, end-to-end demonstration of using Composable Kernel for matrix multiplication in a user application.
