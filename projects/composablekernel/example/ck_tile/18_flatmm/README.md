# FLATMM Matrix Multiplication with CK Tile

This example demonstrates FLATMM (flattened matrix multiplication) using the CK Tile programming model. FLATMM is a variant of GEMM optimized for certain memory layouts and batch processing patterns. Currently, it only supports the basic feature of the CK Tile FLATMM, but creates the placeholders for the future support on different FLATMM pipeline and different FLATMM modules. In the near future, we will gradually migrate all the FLATMM features from old CK to CK Tile.

---

## Algorithm and Math

Given:
- $A$: $[\text{batch}, M, K]$
- $B$: $[\text{batch}, K, N]$
- $C$: $[\text{batch}, M, N]$

For each batch $b$:
$$
C^{(b)} = A^{(b)} \times B^{(b)}
$$

- **FLATMM**: An alternative solution as the Preshuffled GEMM in /03_gemm


---

## Build & Run

```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# The basic pipeline method on the flatmm calculation
make tile_example_flatmm_basic -j
```
This will result in an executable `build/bin/tile_example_flatmm_basic`

### Arguments
```
args:
          -m    m dimension (default:256)
          -n    n dimension (default:256)
          -k    k dimension (default:128)
   -a_layout    A tensor data layout - Row by default (default:R)
   -b_layout    B tensor data layout - Row by default (default:C)
   -c_layout    C tensor data layout - Row by default (default:R)
   -stride_a    Tensor A stride (default:0)
   -stride_b    Tensor B stride (default:0)
   -stride_c    Tensor C stride (default:0)
          -v    0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:1)
       -prec    data type. fp16/bf16/fp8/bf8 (default:fp16)
     -warmup    number of iterations before benchmark the kernel (default:50)
     -repeat    number of iterations to benchmark the kernel (default:100)
      -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
    -split_k    splitK value (default:1)
       -init    0:random, 1:linear, 2:constant(1) (default:0)
  -warp_tile    0: 16x16, 1: 32x32, 2: 16x16x128 (950 only), 3: 32x32x64 (950 only) (default:0)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:flatmm_basic.json)
```

---

## Source Structure

- **Kernel**: [`flatmm_basic.hpp`](flatmm_basic.hpp) (tile-programming kernel template)
- **Executable**: [`flatmm_basic.cpp`](flatmm_basic.cpp)
- **Build**: `CMakeLists.txt`, `run_flatmm_example.inc`, `script/`

---

## Related CK Tile Examples

- [16_batched_gemm](../16_batched_gemm/README.md): Batched GEMM with tiles
- [03_gemm](../03_gemm/README.md): Single GEMM with tiles
- [17_grouped_gemm](../17_grouped_gemm/README.md): Grouped GEMM with tiles

For distribution, see [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
