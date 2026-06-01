# Composable Kernel GEMM Bilinear Example

## Introduction

This example demonstrates GEMM (General Matrix Multiplication) fused with bilinear operations on auxiliary tensors using Composable Kernel. Bilinear fusion patterns are widely used in neural networks for gating, attention, and multimodal feature fusion, where the output of a matrix multiplication is combined elementwise with one or more additional tensors.

---

## Theory

**Mathematical Formulation:**
$$
F = \text{BilinearOp}(A \times B, D, E)
$$
- $A$: [M, K] input matrix
- $B$: [K, N] weight matrix
- $D$, $E$: [M, N] auxiliary tensors (or broadcastable)
- $F$: [M, N] output

**Examples:**
- Elementwise: $F = (A \times B) \odot D \odot E$
- Gated: $F = (A \times B) \odot \sigma(D) + E$
- Weighted: $F = \alpha (A \times B) + \beta (D \odot E)$

The GEMM result is kept in registers and combined with auxiliary tensors in the epilogue, avoiding intermediate writes to global memory. This pattern is common in attention, gating, and feature interaction layers.

---

## CK GEMM Bilinear API Overview

CK provides a composable API for GEMM with multiple auxiliary tensors via the `DeviceGemmMultipleD` operation.

### Template Parameters

- **ALayout** - A matrix layout (RowMajor/ColumnMajor)
- **BLayout** - B matrix layout (RowMajor/ColumnMajor)
- **DsLayout** - Layouts for auxiliary tensors (tuple)
- **ELayout** - Output matrix layout (RowMajor/ColumnMajor)
- **ADataType** - A matrix data type
- **BDataType** - B matrix data type
- **DsDataType** - Data types for auxiliary tensors (tuple)
- **EDataType** - Output matrix data type
- **AElementwiseOperation** - Fused operation on tensor A before GEMM
- **BElementwiseOperation** - Fused operation on tensor B before GEMM
- **CDEElementwiseOperation** - Fused operation on C, D, E after GEMM

### Supported Data Types and Layouts

- Supports fp16, int8, and other types depending on the device operation.
- Supports RowMajor and ColumnMajor layouts for all tensors.

### Supported Device Operations

- **DeviceGemmMultipleD**: Standard multi-tensor GEMM
- **DeviceGemmMultipleD_Bilinear**: GEMM with bilinear fusion in the epilogue

---

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run

```bash
cd composable_kernel/example/02_gemm_bilinear
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

```
### Run ```example_gemm_bilinear_xdl_fp16```

```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: time kernel (0=no, 1=yes)
#arg4 to 10: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD, StrideE
#arg11 to 12: alpha, beta
./bin/example_gemm_bilinear_xdl_fp16 1 1 1 3840 4096 4096 4096 4096 4096 4096 0.5 0.5
```

---

## Source Code Structure

```
example/02_gemm_bilinear/
├── gemm_bilinear_xdl.cpp         # Main example: sets up, runs, and verifies GEMM with bilinear fusion
├── gemm_bilinear_wmma_fp16.cpp   # WMMA FP16 variant
├── gemm_bilinear_wmma_int8.cpp   # WMMA int8 variant
include/ck/tensor_operation/gpu/device/
│   └── device_gemm_multiple_d.hpp       # Device-level API for multi-tensor GEMM
include/ck/tensor_operation/gpu/device/impl/
│   └── device_gemm_bilinear_impl.hpp    # Bilinear operation implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_gemm_multiple_d.hpp     # Grid-level multi-tensor GEMM kernel
include/ck/tensor_operation/gpu/element/
    └── element_wise_operation.hpp       # Elementwise operation definitions
```

### Key Classes and Functions

- **DeviceGemmMultipleD** (in `device_gemm_multiple_d.hpp`):  
  Device API for GEMM with multiple auxiliary tensors and fused epilogues.
- **gridwise_gemm_multiple_d** (in `gridwise_gemm_multiple_d.hpp`):  
  Implements the tiled/blocking GEMM kernel with multi-tensor epilogue.
- **element_wise_operation** (in `element_wise_operation.hpp`):  
  Defines bilinear and other elementwise operations.

---

This example demonstrates how Composable Kernel supports complex multi-tensor fusion patterns for advanced neural network architectures.
