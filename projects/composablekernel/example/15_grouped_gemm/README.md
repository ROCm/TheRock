# Grouped GEMM

## Theory

This example demonstrates **grouped GEMM**: performing multiple independent GEMM operations (with potentially different shapes) in a single kernel launch. Grouped GEMM is used in transformer models (e.g., multi-head attention), mixture-of-experts, and other architectures requiring heterogeneous batched matrix multiplications.

**Mathematical Formulation:**
For $G$ groups, each with its own $A_g$, $B_g$, $C_g$:
$$
C_g = A_g \times B_g \quad \text{for} \quad g = 1, 2, ..., G
$$
- $A_g$: [M_g, K_g] input matrix for group $g$
- $B_g$: [K_g, N_g] weight matrix for group $g$
- $C_g$: [M_g, N_g] output matrix for group $g$

**Algorithmic Background:**
- Each group can have different matrix sizes and strides.
- The kernel launches a grid covering all groups, with each block assigned to a group.
- Useful for variable-length sequences, multi-head attention, and expert routing.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/15_grouped_gemm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

```

### Run ```example_grouped_gemm_xdl```

```bash
#arg1: verification (0=no, 1=yes)
#arg2: initialization (0=no init, 1=integer value, 2=decimal value)
#arg3: run kernel # of times (>1)
./bin/example_grouped_gemm_xdl_fp16 0 1 5
```

## Source Code Structure

### Directory Layout
```
example/15_grouped_gemm/
├── grouped_gemm_xdl.cpp         # Main example: sets up, runs, and verifies grouped GEMM
include/ck/tensor_operation/gpu/device/
│   └── device_grouped_gemm_xdl.hpp       # Device-level grouped GEMM API
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_grouped_gemm_xdl.hpp     # Grid-level grouped GEMM kernel
```

### Key Classes and Functions

- **DeviceGroupedGemmXdl** (in `device_grouped_gemm_xdl.hpp`):  
  Device API for grouped GEMM.
- **gridwise_grouped_gemm_xdl** (in `gridwise_grouped_gemm_xdl.hpp`):  
  Implements the tiled/blocking grouped GEMM kernel.

This example demonstrates how Composable Kernel supports efficient heterogeneous batched matrix multiplication for advanced AI/ML workloads.
