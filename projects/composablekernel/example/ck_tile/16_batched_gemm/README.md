# Batched GEMM with CK Tile

This example demonstrates batched matrix multiplication (Batched GEMM) using the CK Tile programming model, enabling efficient parallel computation of multiple independent GEMMs in a single kernel launch.

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

- **Tilewise Batched GEMM**: Each thread block processes a tile of $C$ for a specific batch, loading corresponding tiles from $A$ and $B$, performing blockwise matrix multiply-accumulate, and writing results.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of $C$ for a given batch.
- **Pipeline**: Modular, supports different memory/computation pipelines.

---

## Features

- **Flexible Layouts**: Supports row/column-major and custom strides for $A$, $B$, $C$.
- **Batching**: Efficiently computes multiple GEMMs in parallel.
- **Precision**: Supports fp16, bf16, fp8, bf8.
- **Validation**: CPU/GPU validation and error tolerance options.

---

## Build & Run

```bash
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
make tile_example_batched_gemm -j
```

This will result in an executable `build/bin/tile_example_batched_gemm`

### Arguments

```bash
args:
               -m    m dimension (default:512)
               -n    n dimension (default:1024)
               -k    k dimension (default:2048)
         -stride_a    Tensor A stride (default:0)
         -stride_b    Tensor B stride (default:0)
         -stride_c    Tensor C stride (default:0)
         -a_layout    A tensor data layout - Row by default (default:R)
         -b_layout    B tensor data layout - Row by default (default:C)
         -c_layout    C tensor data layout - Row by default (default:R)
   -batch_stride_a    Batch A stride (default:1048576)
   -batch_stride_b    Batch B stride (default:2097152)
   -batch_stride_c    Batch C stride (default:524288)
      -batch_count    Batch count (default:8)
               -v    0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:2)
            -prec    data type. fp16/bf16/fp8/bf8 (default:fp16)
         -warmup    number of iterations before benchmark the kernel (default:50)
         -repeat    number of iterations to benchmark the kernel (default:100)
            -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
         -split_k    splitK value (default:1)
            -json    0: No Json, 1: Dump Results in Json format (default:0)
         -jsonfile    json file name to dump results (default:cktile_batched_gemm.json)
```

---

## Source Structure

- **Kernel**: [`batched_gemm.hpp`](batched_gemm.hpp) (tile-programming kernel template)
- **Executable**: [`batched_gemm.cpp`](batched_gemm.cpp)
- **Build**: `CMakeLists.txt`, `run_batched_gemm_example.inc`

---

## Related CK Tile Examples

- [03_gemm](../03_gemm/README.md): Single GEMM with tiles
- [15_fused_moe](../15_fused_moe/README.md): Fused MoE block (uses group/batched GEMM)
- [13_moe_sorting](../13_moe_sorting/README.md): MoE sorting for expert dispatch

For distribution, [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
