# GEMM with CK Tile

This example demonstrates matrix multiplication (GEMM) using the CK Tile programming model, focusing on tile-based parallelism and modular kernel design.

---

## Algorithm and Math

GEMM computes:
$$
C = A \times B
$$
where $A$ is $[M, K]$, $B$ is $[N, K]$, and $C$ is $[M, N]$.

- **BlockTile GEMM**: Each Block Tile computes a tile of $C$ by loading tiles of $A$ and $B$, performing blockwise matrix multiply-accumulation, and writing results back with the epilogue.

---

## Tile Programming Model

- **Configuration**: The Configuration of how the kernel going to be initialized with Block Tile Dimension, Warps Layout, Warp Tile Dimension, and other improvements.
- **Block Tile**: Each block tile allocates in the compute unit of AMD GPU grabbing the .
- **Pipeline**: Modular design allows swapping different memory/computation pipelines (e.g., basic, memory-bound, compute).
- **Block GEMM**: Block Level implementation on how to coordinate the warps iteration and memory layout in block tile.
- **Warp GEMM**: Each Warp's GEMM Calculation
- **Epilogue**: Transferring the Accumulated result from register to global memory.

---

## Features

- **Flexible Layouts**: Supports row/column-major and custom strides for $A$, $B$, $C$.
- **Split K**: Split the Block Tile also on K Dimension and add it back after the matrix multiply-accumulation. Have a higher performance when M and N is small and K is large.
- **Preshuffled GEMM**: In inference task, shuffle the GEMM of B (weight) matrix in the warp layout and bypass the shared memory to do the GEMM calculation. Best performance solution for GEMM.
- **Precision**: Supports fp16, bf16, fp8, bf8, int4 (for B Matrix).
- **Validation**: CPU/GPU validation and error tolerance options.

---

## Build & Run

```bash
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# The basic pipeline method on the gemm calculation
make tile_example_gemm_basic -j`nproc`
# The memory bound pipeline on the gemm calculation
make tile_example_gemm_universal -j`nproc`
# The weight preshuffle pipeline on the gemm calculation
make tile_example_gemm_weight_preshuffle -j`nproc`
```
This will result in an executable `build/bin/tile_example_gemm_basic` & `build/bin/tile_example_gemm_universal`

## example
```
args:
          -m    m dimension (default:1024)
          -n    n dimension (default:2048)
          -k    k dimension (default:64)
   -a_layout    Tensor A data layout (default: R)
   -b_layout    Tensor B data layout (default: C)
   -c_layout    Tensor C data layout (default: R)
   -stride_a    Tensor A stride (default:0)
   -stride_b    Tensor B stride (default:0)
   -stride_c    Tensor C stride (default:0)
          -v    0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:2)
       -prec    data type. fp16/bf16/fp8/bf8 (default:fp16)
     -warmup    number of iterations before benchmark the kernel (default:50)
     -repeat    number of iterations to benchmark the kernel (default:100)
      -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
          -split_k    splitK value (default:1)
       -init    0:random, 1:linear, 2:constant(1) (default:0)
 -persistent    0:non-persistent, 1:persistent (default:0)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:gemm.json)
```


## Source Structure

- **Executables**: `gemm_basic.cpp`, `universal_gemm.cpp` (different kinds of GEMM implementation)
- **Utils**: `gemm_utils.hpp` (helper functions)
- **Build**: `CMakeLists.txt`, `run_gemm_example.inc`
- **Scripts**: `script/` (build and run helpers)

---

## Related CK Tile Examples

- [01_fmha](../01_fmha/README.md): Fused multi-head attention (FMHA)
- [18_flatmm](../18_flatmm/README.md): Preshuffled GEMM alternative solution
- [16_batched_gemm](../16_batched_gemm/README.md): Batched GEMM with tiles

For distribution, see `include/ck_tile/tile_program/tile_distribution/`.

---
[Back to CK Tile Examples](../README.md)
