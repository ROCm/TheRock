# Parallel Reduction Operations

## Theory

This example demonstrates **parallel reduction operations** (e.g., sum, max, min, mean) over tensors. Reduction is a fundamental operation in deep learning for computing statistics (such as batch mean/variance), loss aggregation, and normalization.

**Mathematical Formulation:**
Given a tensor $X$ and a reduction axis $a$:
$$
Y = \text{reduce}_{a}(X)
$$
- For sum: $Y = \sum_{i \in a} X_i$
- For max: $Y = \max_{i \in a} X_i$
- For mean: $Y = \frac{1}{|a|} \sum_{i \in a} X_i$

**Algorithmic Background:**
- Reductions are implemented using parallel tree reduction or segmented reduction algorithms.
- Efficient reductions require careful memory access, synchronization, and sometimes numerically stable algorithms (e.g., Welford's for variance).

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/12_reduce
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j
```

## Run ```example_reduce_blockwise```

```bash
# -D <xxx> : input 3D/4D/5D tensor lengths
# -R <xxx> : reduce dimension ids
# -v <x> :   verification (0=no, 1=yes)
#arg1: data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64, 7: int4)
#arg2: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3: time kernel (0=no, 1=yes)
./bin/example_reduce_blockwise -D 16,64,32,960 -v 1 0 2 1
```

Expected Result:

```
./bin/example_reduce_blockwise -D 16,64,32,960 -v 1 0 2 1
launch_and_time_kernel: grid_dim {240, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 0.238063 ms, 264.285 GB/s, DeviceReduceBlockWise<256,M_C4_S1,K_C64_S1,InSrcVectorDim_0_InSrcVectorSize_1_OutDstVectorSize_1>
```

## Run ```example_reduce_multiblock_atomic_add```

```bash
# -D <xxx> : input 3D/4D/5D tensor lengths
# -R <xxx> : reduce dimension ids
# -v <x> :   verification (0=no, 1=yes)
#arg1: data type (0: fp32, 1: fp64)
#arg2: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3: time kernel (0=no, 1=yes)
./bin/example_reduce_multiblock_atomic_add -D 16,64,32,960 -v 1 0 2 0
```

Expected Result
```
./bin/example_reduce_multiblock_atomic_add -D 16,64,32,960 -v 1 0 2 0
Perf: 0 ms, inf GB/s, DeviceReduceMultiBlock<256,M_C4_S1,K_C64_S1,InSrcVectorDim_0_InSrcVectorSize_1_OutDstVectorSize_1>
echo $?
0
```

# Instructions for ```example_reduce_blockwise_two_call```

## Run ```example_reduce_blockwise_two_call```

```bash
#arg1:  verification (0=no, 1=yes(
#arg2:  initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3:  time kernel (0=no, 1=yes)
./bin/example_reduce_blockwise_two_call 1 2 1
```

Expected Result:

```
./bin/example_reduce_blockwise_two_call 1 2 1
launch_and_time_kernel: grid_dim {204800, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {6400, 1, 1}, block_dim {256, 1, 1}
Warm up 1 time
Start running 10 times...
Perf: 2.1791 ms, 771.42 GB/s, DeviceReduceBlockWise<256,M_C32_S1,K_C8_S1,InSrcVectorDim_1_InSrcVectorSize_1_OutDstVectorSize_1> => DeviceReduceBlockWise<256,M_C256_S1,K_C1_S1,InSrcVectorDim_1_InSrcVectorSize_1_OutDstVectorSize_1>
```

## Source Code Structure

### Directory Layout
```
example/12_reduce/
├── reduce_xdl.cpp         # Main example: sets up, runs, and verifies reduction
include/ck/tensor_operation/gpu/device/
│   └── device_reduce.hpp       # Device-level reduction API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_reduce_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
│   └── gridwise_reduce.hpp     # Grid-level reduction kernel
include/ck/tensor_operation/gpu/block/
    └── blockwise_reduce.hpp    # Block-level reduction
```

### Key Classes and Functions

- **DeviceReduce** (in `device_reduce.hpp`):  
  Device API for reductions.
- **gridwise_reduce** (in `gridwise_reduce.hpp`):  
  Implements the tiled/blocking reduction kernel.
- **blockwise_reduce** (in `blockwise_reduce.hpp`):  
  Handles block-level reduction and shared memory.

This example demonstrates how Composable Kernel implements efficient parallel reductions for deep learning and scientific computing.
