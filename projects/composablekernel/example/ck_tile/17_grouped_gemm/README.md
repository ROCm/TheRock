## Quick Tour for New Users

The `Grouped GEMM` operators are versions of GEMM that run multiple GEMM operations within a single kernel call. Each GEMM operation performs a matrix multiplication. Unlike regular batched GEMM operations where both matrices must be of the same size and have the same configuration, Grouped GEMM operations can take matrices with different sizes and configurations, making them more flexible for diverse workloads.

### Preshuffle and Persistence

The grouped GEMM examples include the following advanced optimization features:

#### Weight Preshuffle
Weight preshuffle is an optimization technique that reorganizes the B matrix (weights) in memory to improve data access patterns and reduce memory bandwidth requirements. This is particularly beneficial for inference workloads where the same weights are reused across multiple batches.

- **Implementation**: Available in `grouped_gemm_preshuffle.cpp` 
- **Configuration**: Uses `GemmConfigPreshuffleDecode` and `GemmConfigPreshufflePrefill` template configuration
- **Constraints**: Currently supports only A(Row major) + B(Column major) → C(Row major) layouts


#### Persistence Mode
Persistence mode is a GPU optimization where thread blocks remain active on the compute units to process multiple work items sequentially, reducing kernel launch overhead and improving occupancy.

- **Template Parameter**: Controlled by the `Persistent` boolean template parameter in `invoke_gemm`
- **Usage**: `invoke_gemm<ALayout, BLayout, CLayout, true>` enables persistence

#### Multi-D Operations
Multi-D operations extend the standard GEMM operation by supporting additional elementwise operations on the result tensor. This feature is particularly useful for workloads that require post-processing of the GEMM output.

- **Implementation**: Available in `grouped_gemm_multi_d.cpp`
- **Operation**: E = C × D₀ × D₁ (where C = A × B is the standard GEMM result)
- **Configuration**: Uses `GemmConfigV3`, `GemmConfigV4`, `GemmConfigMemory` template configuration with 2 D tensors
- **Data Types**: Supports fp16, bf16, fp8
- **Benefits**: Enables complex operations like scaling, activation functions, or other elementwise transformations in a single kernel call
- **Build Target**: `make tile_example_grouped_gemm_multi_d -j`

Multi-D operations supports both persistence and non-persistence modes.
Weight preshuffle supports only on non-persistence mode.

## Build
```
# in the root of ck_tile
mkdir build && cd build
../script/cmake-ck-dev.sh ../ <arch>
make tile_example_grouped_gemm -j
# The preshuffle example
make tile_example_grouped_gemm_preshuffle -j
# The multi-D operations example
make tile_example_grouped_gemm_multi_d -j
# The quant grouped gemm fp8 example
make tile_example_quant_grouped_gemm -j
```
Each example will result in an corresponding executable `build/bin/tile_example_grouped_gemm`, `build/bin/tile_example_grouped_gemm_preshuffle`, `build/bin/tile_example_grouped_gemm_multi_d`, and `build/bin/tile_example_quant_grouped_gemm`.


## example
```
args:
 -Ms          M dimensions - (Default: empty).
 -Ns          N dimensions - (Default: empty).
 -Ks          K dimensions - (Default: empty).
 -stride_As   Tensor A strides - (Default: empty).
 -stride_Bs   Tensor B strides - (Default: empty).
 -stride_Cs   Tensor C strides - (Default: empty).
 -a_layout    A tensor data layout - (Default: Row).
 -b_layout    B tensor data layout - (Default: Col).
 -c_layout    C tensor data layout - (Default: Row).
 -prec        data type. fp16/bf16/fp8 - (Default: fp16).
 -validate    0. No validation, 1. Validation on CPU. (Default: 1).
 -warmup      Number of iterations before benchmark the kernel. (Default: 10).
 -repeat      Number of iterations to benchmark the kernel. (Default: 100).
 -group_count Group count. (Default: 16).
 -kbatch      kbatch for SplitK (Default: 1).
 -json        0: No Json, 1: Dump Results in Json format (Default: 0).
 -jsonfile    json file name to dump results (Default: grouped_gemm.json).
```

If any of `Ms`, `Ns`, `Ks`, `stride_As`, `stride_Bs`, or `stride_Cs` are missing or their sizes
don't match `group_count`, the example generates defaults per group index `i` (0-based):

```text
M[i] = 256 + 256 * i
N[i] = 256 + 512 * i
K[i] = 512 + 384 * i

stride_A[i] = K[i]
stride_B[i] = K[i]
stride_C[i] = N[i]
```

## Source Structure

- **Kernel**: [`grouped_gemm.hpp`](grouped_gemm.hpp) (tile-programming kernel template)
- **Executables**: [`grouped_gemm.cpp`](grouped_gemm.cpp)
- **Build**: `CMakeLists.txt`, `run_grouped_gemm_example.inc`

---

## Related CK Tile Examples

- [16_batched_gemm](../16_batched_gemm/README.md): Batched GEMM with tiles
- [15_fused_moe](../15_fused_moe/README.md): Fused MoE block (uses grouped GEMM)
- [03_gemm](../03_gemm/README.md): Single GEMM with tiles

For distribution, see [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
