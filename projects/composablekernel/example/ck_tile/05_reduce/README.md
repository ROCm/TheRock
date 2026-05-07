# Reduction with CK Tile

This example demonstrates parallel reduction (sum, max, etc.) using the CK Tile programming model, a core operation for normalization, statistics, and aggregation in deep learning.

---

## Algorithm and Math

Given a tensor $X$ and a reduction axis, compute:
- **Sum**: $Y = \sum_i X_i$
- **Max**: $Y = \max_i X_i$
- **Mean**: $Y = \frac{1}{N} \sum_i X_i$

- **Tilewise Reduction**: Each thread block reduces a tile (block) of the input, using shared memory and register accumulation for efficiency.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (block) of the input tensor.
- **Pipeline**: Modular, can be extended for fused reductions or post-processing.

---

## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_reduce -j
./bin/tile_example_reduce -?
```

---

## Source Structure

- **Kernel**: `reduce.hpp` (tile-programming kernel template)
- **Executable**: `reduce.cpp` (argument parsing, kernel launch)
- **Build**: `CMakeLists.txt`

---

## Related CK Tile Examples

- [03_gemm](../03_gemm/README.md): GEMM with tiles
- [04_img2col](../04_img2col/README.md): im2col transformation
- [06_permute](../06_permute/README.md): Permutation with tiles

For distribution, see `include/ck_tile/tile_program/tile_distribution/`.

---
[Back to CK Tile Examples](../README.md)
