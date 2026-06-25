# Parallel Softmax

## Theory

This example demonstrates **parallel softmax computation** over tensors. Softmax is a key operation in deep learning, especially in attention mechanisms and classification, converting logits into normalized probabilities.

**Mathematical Formulation:**
Given input $X$ and axis $a$:
$$
\text{softmax}(X)_i = \frac{\exp(X_i)}{\sum_j \exp(X_j)}
$$

**Algorithmic Background:**
- Softmax is implemented using a numerically stable algorithm:
  1. Subtract the maximum value for numerical stability.
  2. Exponentiate and sum.
  3. Normalize by the sum.
- Efficient parallel softmax requires careful reduction and memory access patterns.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/23_softmax
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j
```

```bash
# -D <xxx> : input 3-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg2: time kernel (0=no, 1=yes)
example_softmax_blockwise -D 4,128,2048 -v 1 1 1
```

Result
```
launch_and_time_kernel: grid_dim {64, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
Perf: 0.0242877 ms, 259.039 GB/s, DeviceReduceSoftmax<256,M_C8_S1,K_C32_S8,InSrcVectorDim_1_InSrcVectorSize_8_OutDstVectorSize_8>
```

## Source Code Structure

### Directory Layout
```
example/23_softmax/
├── softmax_xdl.cpp         # Main example: sets up, runs, and verifies softmax
include/ck/tensor_operation/gpu/device/
│   └── device_softmax.hpp       # Device-level softmax API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_softmax_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_softmax.hpp     # Grid-level softmax kernel
include/ck/tensor_operation/gpu/block/
    └── blockwise_softmax.hpp    # Block-level softmax
```

### Key Classes and Functions

- **DeviceSoftmax** (in `device_softmax.hpp`):  
  Device API for softmax.
- **gridwise_softmax** (in `gridwise_softmax.hpp`):  
  Implements the tiled/blocking softmax kernel.
- **blockwise_softmax** (in `blockwise_softmax.hpp`):  
  Handles block-level softmax and shared memory.

This example demonstrates how Composable Kernel implements efficient, numerically stable softmax for deep learning models.
