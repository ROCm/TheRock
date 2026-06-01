# Stream-K GEMM

This folder contains examples of Stream-K GEMMs using the ck_tile tile-programming implementation.

## build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# Compile the Stream-K kernels
make tile_example_streamk_gemm_basic -j
```
This will result in an executable `build/bin/tile_example_streamk_gemm_basic`

## example
```
args:
                 -m    m dimension (default:512)
                 -n    n dimension (default:512)
                 -k    k dimension (default:512)
          -a_layout    tensor A data layout (default: R)
          -b_layout    tensor B data layout (default: C)
          -c_layout    tensor C data layout (default: R)
-reduction_strategy    strategy for storing results in C tensor. atomic/reduction (default:atomic)
     -persistent_dp    persistent strategy for data-parallel section. Set to 0 for non-persistent or to 1 for persistent. (default:0)
          -stride_a    tensor A stride (default:0)
          -stride_b    tensor B stride (default:0)
          -stride_c    tensor C stride (default:0)
                 -v    validation strategy. 0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:1)
              -prec    data type. fp16/bf16/fp8/bf8 (default:fp16)
            -warmup    number of iterations before benchmarking the kernel (default:50)
            -repeat    number of iterations to benchmark the kernel (default:100)
             -timer    timing mode. gpu:gpu timer, cpu:cpu timer (default:gpu)
              -init    data initialization strategy. 0:random, 1:linear, 2:constant(1) (default:0)
       -flush_cache    flush the cache before running the kernel (default:true)
```
