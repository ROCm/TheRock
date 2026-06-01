# Batched Transpose (Block Transpose) with CK Tile

This example demonstrates a high-performance batched block transpose kernel using the CK Tile programming model, with a focus on architectures like gfx950. The kernel is optimized for tiled memory access and is suitable for layout conversions such as NCHW <-> NHWC in deep learning. This transpose load has some constraints in input tile distribution.

---

## Algorithm and Math

Given a batch of tensors $X$ of shape $[N, C, H, W]$, the block transpose operation rearranges axes to produce $Y$ of shape $[N, H, W, C]$ (NCHW to NHWC) or other permutations.

For each element:
$$
Y_{n, h, w, c} = X_{n, c, h, w}
$$

- **Blockwise Transpose**: Each thread block processes a tile (block) of the input, reading and writing in a coalesced, tiled fashion for optimal memory throughput.

---

## Tile Programming Model

- **Tiles**: Each thread block processes a tile of the input tensor for a given batch.
- **Policy**: [`transpose_policy.hpp`](transpose_policy.hpp) defines tile/block size and memory access patterns for optimal performance.

---

## Features

- **Optimized for Architecture**: Designed for architectures like gfx950 with constraints on input tile distribution.
- **Flexible Layouts**: Supports NCHW <-> NHWC and other axis permutations.
- **Batching**: Efficiently transposes multiple tensors in parallel.
- **Validation**: CPU validation and benchmarking options.

---

## Build & Run

```bash
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
# Make the transpose executable
make tile_example_transpose -j
```

This will result in an executable `build/bin/tile_example_transpose`

### Arguments

```bash
args:
          -N    input batch size (default:2)
          -C    input channel size. (default:64)
          -H    input height size. (default:1)
          -W    input width size. (default:64)
          -v    whether do CPU validation or not (default: 1)
  -layout_in    input tensor data layout - NCHW by default
 -layout_out    output tensor data layout - NHWC by default
       -seed    seed to be used, -1 means random every time (default:-1)
     -k_name    t to 1 will print kernel name (default:0)
```

---

## Source Structure

- **Kernel**: [`block_transpose.hpp`](block_transpose.hpp), [`batched_transpose_kernel.hpp`](batched_transpose_kernel.hpp), [`transpose_policy.hpp`](transpose_policy.hpp)
- **Executables**: [`transpose_example.cpp`](transpose_example.cpp), [`transpose_api.cpp`](transpose_api.cpp)
- **Build**: `CMakeLists.txt`

---

## Related CK Tile Examples

- [35_batched_transpose](../35_batched_transpose/README.md): Batched transpose with tiles
- [06_permute](../06_permute/README.md): Generic permutation with tiles
- [03_gemm](../03_gemm/README.md): GEMM with tiles

For tile engine and distribution, see [`include/ck_tile/tile_engine/`](../../../include/ck_tile/tile_engine/) and [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
