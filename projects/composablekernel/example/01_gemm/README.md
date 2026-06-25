[Back to supported operations](../../../include/ck/README.md)
# Composable Kernel GEMM Example

## Introduction

GEMM (General Matrix Multiplication) is a fundamental operation in linear algebra and deep learning. It computes the product of two matrices, optionally adds a bias or residual, and is the core of many neural network layers (MLPs, attention, convolutions via im2col). This example demonstrates the flexible and high-performance GEMM API provided by Composable Kernel.

---

## Theory

**Mathematical Formulation:**
$$
C = \alpha (A \times B) + \beta D
$$
- $A$: [M, K] input matrix
- $B$: [K, N] weight matrix
- $D$: [M, N] optional bias/residual
- $C$: [M, N] output
- $\alpha, \beta$: scalars (often 1.0, 0.0)

GEMM is implemented using a tiled/blocking strategy to maximize data reuse and memory bandwidth. Modern GPU implementations use matrix core/XDL/MFMA instructions for high throughput. The operation is the computational backbone for transformer attention, MLPs, CNNs (via lowering), and more.

---

## CK GEMM API Overview

CK provides a highly composable GEMM API via the `DeviceGemm` family of device operations. These are highly templated to support a wide range of data types, layouts, and fused operations.

### Template Parameters

- **ALayout** - A matrix layout (RowMajor/ColumnMajor)
- **BLayout** - B matrix layout (RowMajor/ColumnMajor)
- **CLayout** - C matrix layout (RowMajor/ColumnMajor)
- **ADataType** - A matrix data type
- **BDataType** - B matrix data type
- **CDataType** - C matrix data type
- **AElementwiseOperation** - Fused operation on tensor A before GEMM
- **BElementwiseOperation** - Fused operation on tensor B before GEMM
- **CElementwiseOperation** - Fused operation on tensor C after GEMM

For large K dimension, use `DeviceGemmSplitK` to split K across workgroups (requires zeroing output buffer due to use of AtomicAdd).

For fused operations with additional tensors, use `DeviceGemmMultipleABD` or `DeviceGemmMultipleD`:
- **DsLayout** - layouts for additional tensors
- **DsDataType** - data types for additional tensors

For `DeviceGemmMultipleABD`, pass **ALayout**, **BLayout**, **ADataType**, **BDataType** as tuples.

---

## Supported GEMM Variants

- **DeviceGemm**: Standard GEMM
- **DeviceGemmSplitK**: Split-K GEMM for large K
- **DeviceGemmMultipleABD**: Fused GEMM with multiple A/B/D tensors
- **DeviceGemmMultipleD**: Fused GEMM with multiple D tensors

---

## Supported Device Operations

- **DeviceGemmDl**: DL instructions
- **DeviceGemmDpp**: DL instructions with DPP during data load
- **DeviceGemmWmma_CShuffle**: WMMA instructions with CShuffle optimization
- **DeviceGemm_Xdl_CShuffle_LdsDirectLoad**: XDL instructions, CShuffle, direct global-to-shared load
- **DeviceGemm_Xdl_CShuffle**: XDL instructions with CShuffle
- **DeviceGemm_Xdl_CShuffleV2**: XDL instructions, optimized pipeline vs. V1
- **DeviceGemmXdlSkipBLds**: XDL, skips shared memory load for B
- **DeviceGemm_Xdl_WaveletModel_CShuffle**: XDL, CShuffle, wavelet producer/consumer
- **DeviceGemmXdl**: XDL instructions

---

## Supported Data Types and Layouts

### XDL Instruction

|       |Is supported|
|-------|---|
|bf16   |✔️|
|fp16   |✔️|
|fp32   |✔️|
|int8   |✔️|
|fp8    |✔️|

### WMMA Instruction

|       |Is supported|
|-------|---|
|bf16   |✔️|
|fp16   |✔️|
|fp32   |❌|
|int8   |✔️|
|fp8    |❌|

### DL Instruction

|       |Is supported|
|-------|---|
|bf16   |❌|
|fp16   |✔️|
|fp32   |✔️|
|int8   |✔️|
|fp8    |❌|

---

## Supported Fused Elementwise Operations

- **B Matrix Multiply + Add + Gelu** - bf16 (int8 for B matrix)
- **B Matrix Multiply + Add** - bf16 (int8 for B matrix)
- **B Matrix Multiply + Gelu** - bf16 (int8 for B matrix)
- **B Matrix Multiply** - bf16 (int8 for B matrix)
- **Add + Add + Gelu** - fp16
- **Add + Gelu** - fp16, bf16 (int8 for B matrix) for Row/Column/Row
- **Multiply** - fp16
- **Add + Multiply** - fp16
- **Add + Relu** - fp16 (int8 for B matrix) for Row/Column/Row, bf16 (int8 for B matrix) for Row/Column/Row
- **Add + Silu** - fp16 (int8 for B matrix) for Row/Column/Row, bf16 (int8 for B matrix) for Row/Column/Row
- **Add** - fp16 (int8 for B matrix) for Row/Column/Row, bf16 (int8 for B matrix) for Row/Column/Row
- **Bilinear** - fp16, int8
- **Gelu** - fp16
- **Multiply + Add** - fp16 for Row/Column/Row and Row/Row/Row, fp16 (int8 for B matrix, fp32 for Bias) for Row/Column/Row and Row/Row/Row
- **Quantization** - int8

---

## GEMM V2 (Universal GEMM)

Optimized for MI300 series. Operation is called as `DeviceGemmV2` and uses similar template parameters as above.

- **ALayout**, **BLayout**, **CLayout**
- **ADataType**, **BDataType**, **CDataType**
- **AElementwiseOperation**, **BElementwiseOperation**, **CElementwiseOperation**

Split-K is supported (requires zeroing output buffer if splitK > 1).

### Device Operations

- **DeviceGemm_Xdl_CShuffleV3**: XDL with CShuffle optimization
- **DeviceGemm_Xdl_CShuffleV3R1**: XDL with CShuffle, reduction on split-K after GEMM

### Supported Types

|       |Is supported|
|-------|---|
|bf16   |✔️|
|fp16   |✔️|
|fp32   |❌|
|int8   |❌|
|fp8 (C bf16)|✔️|
|fp16 (A fp8)|✔️|
|fp16 (B fp8)|✔️|

---

## Other GEMM Extensions

- **DeviceGemm_dequantB**: GEMM with dequantization (WMMA)
- **DeviceGemmMultipleD_ABScale**: GEMM with scale for A and B
- **DeviceGemmMultipleDLayernorm**: GEMM fused with layernorm
- **DeviceGemmMultipleDMultipleR**: GEMM fused with reductions and custom global reductions
- **DeviceGemmReduce**: GEMM fused with reduction
- **DeviceGemm_Streamk_V2**: Stream K with reduction instead of AtomicAdd
- **DeviceGemmStreamK**: Stream K using AtomicAdd

---

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run

```bash
cd composable_kernel/example/01_gemm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (FP16)
./gemm_xdl_fp16 -M 4096 -N 4096 -K 4096 -v 1 -t 1
```

---

## Source Code Structure

```
example/01_gemm/
├── gemm_xdl_fp16.cpp         # Main example: sets up, runs, and verifies GEMM (FP16)
├── gemm_xdl_fp32.cpp         # Main example: FP32 variant
include/ck/tensor_operation/gpu/device/
│   └── device_gemm.hpp       # Device-level GEMM API (templated)
include/ck/tensor_operation/gpu/device/impl/
│   └── device_gemm_xdl.hpp   # XDL-based GEMM implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_gemm_xdl.hpp # Grid-level tiled GEMM kernel
include/ck/tensor_operation/gpu/block/
│   └── blockwise_gemm_xdl.hpp # Block-level tiled GEMM
library/reference_tensor_operation/cpu/
    └── reference_gemm.hpp    # CPU reference GEMM for correctness checking
```

### Key Classes and Functions

- **DeviceGemmXdl** (in `device_gemm.hpp`):  
  Main device API for launching GEMM kernels.  
- **GridwiseGemmXdl** (in `gridwise_gemm_xdl.hpp`):  
  Implements the tiled/blocking GEMM kernel for the GPU grid.
- **BlockwiseGemmXdl** (in `blockwise_gemm_xdl.hpp`):  
  Handles block-level computation and shared memory tiling.
- **reference_gemm** (in `reference_gemm.hpp`):  
  CPU implementation for result verification.

---

This example is the foundation for all matrix operations in Composable Kernel and is the basis for more advanced fused and batched operations.
