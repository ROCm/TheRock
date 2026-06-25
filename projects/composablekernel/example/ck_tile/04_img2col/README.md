# Image to Column (im2col) with CK Tile

This example demonstrates the im2col transformation using the CK Tile programming model, a key step for converting convolution into GEMM for efficient GPU execution.

---

## Algorithm and Math

Given an input image tensor $X$ and convolution kernel size, im2col rearranges sliding windows of $X$ into columns:
- For each patch, flatten and stack as a column in the output matrix.
- Enables convolution as matrix multiplication: $\text{im2col}(X) \times W$.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile (block of patches).
- **Pipeline**: Modular, can be extended for fused operations (e.g., quantization, activation).

---

## Build & Run

```bash
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
make tile_example_img2col -j
./bin/tile_example_img2col -?
```

---

## Source Structure

- **Kernel**: `image_to_column.hpp` (tile-programming kernel template)
- **Executable**: `image_to_column.cpp` (argument parsing, kernel launch)
- **Build**: `CMakeLists.txt`

---

## Related CK Tile Examples

- [03_gemm](../03_gemm/README.md): GEMM with tiles (im2col output as input)
- [05_reduce](../05_reduce/README.md): Reductions with tiles
- [06_permute](../06_permute/README.md): Permutation with tiles

For distribution, see `include/ck_tile/tile_program/tile_distribution/`.

---
[Back to CK Tile Examples](../README.md)
