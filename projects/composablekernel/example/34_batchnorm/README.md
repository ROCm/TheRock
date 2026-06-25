# Batch Normalization Forward

## Theory

This example demonstrates **batch normalization forward pass**. Batch normalization is used in deep neural networks to normalize activations across the batch dimension, improving training stability and convergence.

**Mathematical Formulation:**
Given input $X[N, C, ...]$:
- Mean: $\mu_c = \frac{1}{N \cdot ...} \sum_{n,...} X_{n,c,...}$
- Variance: $\sigma^2_c = \frac{1}{N \cdot ...} \sum_{n,...} (X_{n,c,...} - \mu_c)^2$
- Normalized: $\hat{X}_{n,c,...} = \frac{X_{n,c,...} - \mu_c}{\sqrt{\sigma^2_c + \epsilon}}$
- Output: $Y_{n,c,...} = \gamma_c \hat{X}_{n,c,...} + \beta_c$

$\gamma_c$, $\beta_c$ are learnable scale and shift parameters per channel.

**Algorithmic Background:**
- Computes mean and variance per channel (across batch and spatial dimensions).
- Applies normalization and affine transformation.
- Used in CNNs, MLPs, and other deep learning models.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/example/34_batchnorm
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./batchnorm_fwd_xdl --verify=1 --time=1
```

## Run ```batchnorm forward nhwc```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1:  data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64)
#arg2: 1/0 to indicate whether to update the moving average and variance (0=no, 1=yes)
#arg3: 1/0 to indicate whether to save result mean/invVariance (0=no, 1=yes)
#arg4: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg5: time kernel (0=no, 1=yes) 
./bin/example_batchnorm_forward -D 128,16,16,1024 -v 1 0 0 1 2 1
```

Result 
```
./bin/example_batchnorm_forward -D 128,16,16,1024 -v 1 0 0 1 2 1
launch_and_time_kernel: grid_dim {64, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {120, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {120, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 2.08231 ms, 354.519 GB/s
```

Result
```
./bin/example_batchnorm_forward -D 128,16,16,1024 -v 1 0 1 0 2 0
echo $?
0
```

## Run ```batchnorm infer nhwc```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
#arg1:  data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64)
#arg2: initialization (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
#arg3: time kernel (0=no, 1=yes)
./bin/example_batchnorm_infer -D 128,16,16,1024 -v 1 0 2 1
```

Result
```
./bin/example_batchnorm_infer -D 128,16,16,1024 -v 1 0 2 1
launch_and_time_kernel: grid_dim {120, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 1.28235 ms, 523.329 GB/s
```

## Run ```batchnorm backward nhwc```
```bash
# -D <xxx> : input 4-d tensor lengths
# -v <x> :   verification (0=no, 1=yes)
Arg1: data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64)
Arg2 -- 1/0 to indicate whether to use saved mean and invVariance
Arg3 -- init method used for dy and bnScale (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)
Arg4 -- time kernel (0=no, 1=yes)
Arg5: use multi-block welford (0=n0, 1=yes)
./bin/example_batchnorm_backward -D 128,16,3,1024 -v 1 0 0 3 1 1
```

Result 
```
./bin/example_batchnorm_backward -D 128,16,3,1024 -v 1 0 0 3 1 1
launch_and_time_kernel: grid_dim {6144, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {6144, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
launch_and_time_kernel: grid_dim {6144, 1, 1}, block_dim {256, 1, 1} 
Warm up 1 time
Start running 10 times...
Perf: 0.411026 ms, 91.8702 GB/s
```

## Source Code Structure

### Directory Layout
```
example/34_batchnorm/
├── batchnorm_fwd_xdl.cpp         # Main example: sets up, runs, and verifies batchnorm
include/ck/tensor_operation/gpu/device/
│   └── device_batchnorm_fwd.hpp       # Device-level batchnorm API
include/ck/tensor_operation/gpu/device/impl/
│   └── device_batchnorm_fwd_impl.hpp  # Implementation
include/ck/tensor_operation/gpu/grid/
    └── gridwise_batchnorm_fwd.hpp     # Grid-level kernel
```

### Key Classes and Functions

- **DeviceBatchnormFwd** (in `device_batchnorm_fwd.hpp`):  
  Device API for batch normalization.
- **gridwise_batchnorm_fwd** (in `gridwise_batchnorm_fwd.hpp`):  
  Implements the tiled/blocking batchnorm kernel.

This example demonstrates how Composable Kernel implements efficient batch normalization for deep learning models.
