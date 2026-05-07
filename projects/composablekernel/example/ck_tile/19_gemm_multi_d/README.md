# Multiple D GEMM with CK Tile

This example demonstrates GEMM with multiple D tensors (multi-output GEMM) using the CK Tile programming model. This is useful for fused operations where the GEMM output is combined with multiple side inputs (e.g., bias, residual, or other elementwise sources).

---

## Algorithm and Math

Given:
- $A$: $[M, K]$
- $B$: $[K, N]$
- $D_0, D_1, ..., D_n$: $[M, N]$ (multiple side inputs)
- $E$: $[M, N]$ (output)

The operation:
$$
E = f(A \times B, D_0, D_1, ..., D_n)
$$
where $f$ is a fused elementwise function (e.g., add, multiply, activation).

- **Tilewise Multi-D GEMM**: Each thread block processes a tile of $E$, loading corresponding tiles from $A$, $B$, and all $D_i$, performing blockwise GEMM and fused elementwise operations.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of $E$.
- **Pipeline**: Modular, supports different memory/computation pipelines and multi-D fusion.

---

## Features

- **Multiple D Inputs**: Supports arbitrary number of side inputs for fusion.
- **Flexible Layouts**: Supports row/column-major and custom strides for all tensors.
- **SplitK**: Supports K-batching for large K dimensions.
- **Validation**: GPU validation and benchmarking options.

---

## Build & Run

```
#in the root of ck_tile
mkdir build && cd build
#you can replace < arch> with the appropriate architecture(for example gfx90a or gfx942) or \
    leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
#The basic pipeline method on the gemm calculation
make tile_example_gemm_multi_d_fp16 -j
```
This will result in an executable `build/bin/tile_example_gemm_multi_d_fp16`

### Arguments

```
args:
          -m    m dimension (default:3840)
          -n    n dimension (default:4096)
          -k    k dimension (default:4096)
   -a_layout    A tensor data layout - Row by default (default:R)
   -b_layout    B tensor data layout - Col by default (default:C)
  -ds_layout    Ds tensor data layout - Row by default (default:R)
   -e_layout    E tensor data layout - Row by default (default:R)
   -stride_a    Tensor A stride (default:0)
   -stride_b    Tensor B stride (default:0)
  -stride_ds    Tensor Ds stride (default:0)
   -stride_e    Tensor E stride (default:0)
          -v    0. No validation, 1. Validation on GPU (default:1)
     -warmup    number of iterations before benchmark the kernel (default:50)
     -repeat    number of iterations to benchmark the kernel (default:100)
     -kbatch    kbatch for SplitK (default:1)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:cktile_gemm_multi_d_fp16.json)
```

---

## Source Structure

- **Kernel**: [`gemm_multi_d_fp16.hpp`](gemm_multi_d_fp16.hpp) (tile-programming kernel template)
- **Executable**: [`gemm_multi_d_fp16.cpp`](gemm_multi_d_fp16.cpp)
- **Utils**: [`utils.hpp`](utils.hpp)
- **Build**: `CMakeLists.txt`, `run_gemm_multi_d_fp16_example.inc`

---

## Related CK Tile Examples

- [03_gemm](../03_gemm/README.md): Single GEMM with tiles
- [16_batched_gemm](../16_batched_gemm/README.md): Batched GEMM with tiles
- [17_grouped_gemm](../17_grouped_gemm/README.md): Grouped GEMM with tiles

For distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
