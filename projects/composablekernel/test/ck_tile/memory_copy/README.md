# Copy Kernel with CK Tile

This example demonstrates a basic copy kernel using the CK Tile programming model. It is designed as a minimal platform for new CK Tile kernel developers to test and understand tile-based data movement and memory hierarchy. Sample functional code for a simple
tile distribution for DRAM window and LDS window are provided and data is moved from DRAM to registers, registers to LDS, LDS to registers and finally data is moved to output DRAM window for a simple copy operation.

---

## Algorithm and Math

Given an input matrix $X$ of shape $[M, N]$, the copy kernel performs:
$$
Y_{i, j} = X_{i, j}
$$

- **Tilewise Copy**: Each thread block processes a tile (block) of the input, moving data from global memory (DRAM) to registers, registers to LDS (shared memory), LDS to registers, and finally to output DRAM.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of the input matrix.
- **Pipeline**: Simple, but can be extended for more complex memory patterns or fused operations.

---

## Features

- **Memory Hierarchy**: Illustrates DRAM, LDS, and register usage in CK Tile.
- **Minimal Example**: Ideal for learning and debugging tile-programming concepts.
- **Validation**: CPU validation and benchmarking options.

---

## Build & Run

```bash
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture 
# (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# Make the copy kernel executable
make test_copy -j
```
This will result in an executable `build/bin/test_copy_kernel`

### Arguments

```bash
args:
          -m        input matrix rows. (default 64)
          -n        input matrix cols. (default 8)
          -id       warp to use for computation. (default 0)
          -v        validation flag to check device results. (default 1)
          -prec     datatype precision to use. (default fp16)
          -warmup   no. of warmup iterations. (default 50)
          -repeat   no. of iterations for kernel execution time. (default 100)
```

---

## Source Structure

- **Kernel**: [`test_copy.hpp`](test_copy.hpp) (tile-programming kernel template)
- **Executable**: [`test_copy.cpp`](test_copy.cpp)
- **Build**: `CMakeLists.txt`

---

## Related CK Tile Examples

- [03_gemm](../03_gemm/README.md): GEMM with tiles
- [35_batched_transpose](../35_batched_transpose/README.md): Batched transpose with tiles
- [06_permute](../06_permute/README.md): Generic permutation with tiles

For distribution, see [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
