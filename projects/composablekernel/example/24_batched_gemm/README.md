# Batched GEMM

## Theory

This example demonstrates **batched GEMM**: performing multiple independent matrix multiplications (all with the same shape) in a single kernel launch. Batched GEMM is used in multi-head attention, RNNs, and other models requiring parallel matrix multiplications.

**Mathematical Formulation:**
For $B$ batches:
$$
C_b = A_b \times B_b \quad \text{for} \quad b = 1, 2, ..., B
$$
- $A_b$: [M, K] input matrix for batch $b$
- $B_b$: [K, N] weight matrix for batch $b$
- $C_b$: [M, N] output matrix for batch $b$

**Algorithmic Background:**
- All matrices in the batch have the same shape and strides.
- The kernel launches a grid covering all batches, with each block assigned to a batch.
- Used for multi-head attention, parallel MLPs, and more.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/24_batched_gemm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./batched_gemm_xdl --verify=1 --time=1
```

## Source Code Structure

### Directory Layout
```
example/24_batched_gemm/
├── batched_gemm_xdl.cpp         # Main example: sets up, runs, and verifies batched GEMM
include/ck/tensor_operation/gpu/device/
│   └── device_batched_gemm_xdl.hpp       # Device-level batched GEMM API
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_batched_gemm_xdl.hpp     # Grid-level batched GEMM kernel
```

### Key Classes and Functions

- **DeviceBatchedGemmXdl** (in `device_batched_gemm_xdl.hpp`):  
  Device API for batched GEMM.
- **gridwise_batched_gemm_xdl** (in `gridwise_batched_gemm_xdl.hpp`):  
  Implements the tiled/blocking batched GEMM kernel.

This example demonstrates how Composable Kernel supports efficient parallel matrix multiplication for batched and multi-head workloads.
