.. meta::
   :description: CK Tile memory swizzling with Morton ordering example
   :keywords: CK Tile, swizzling, Morton ordering, Z-order curve, GPU optimization

.. _ck_tile_swizzling_example:

**************************************
Memory Swizzling with Morton Ordering
**************************************

Overview
========

This chapter demonstrates a practical application of tensor descriptors for implementing memory swizzling patterns, specifically Morton ordering (Z-order curve) within tiles. Memory swizzling is used to optimize GPU memory access patterns and reduce :ref:`bank conflicts <ck_tile_lds_bank_conflicts>`. Morton ordering provides a space-filling curve that maintains spatial locality while enabling efficient parallel access. See :ref:`ck_tile_space_filling_curve` for more information about parallel access.

Morton ordering is widely used in:

- **GPU Texture Memory**: Optimizing cache efficiency for 2D texture access
- **Matrix Operations**: Reducing memory bank conflicts in shared memory
- **Image Processing**: Improving locality for block-based algorithms
- **Scientific Computing**: Enhancing data access patterns for stencil operations

Understanding Morton Ordering
=============================

Morton ordering interleaves the bits of 2D coordinates to create a 1D ordering that preserves spatial locality. For a 2D coordinate (y, x), we split each coordinate into its binary bits and interleave them:

- y = y₁y₀ (2 bits)
- x = x₁x₀ (2 bits)
- Morton index = y₁x₁y₀x₀ (4 bits)

This creates a Z-shaped traversal pattern within each tile:

.. code-block:: cpp

    // Morton encoding for 2D coordinates
    template<index_t NumBits = 2>
    __host__ __device__ index_t morton_encode_2d(index_t y, index_t x) {
        index_t result = 0;
        for (index_t i = 0; i < NumBits; ++i) {
            index_t bit_y = (y >> i) & 1;
            index_t bit_x = (x >> i) & 1;
            result |= (bit_y << (2*i + 1)) | (bit_x << (2*i));
        }
        return result;
    }
    
    // Morton decoding back to 2D coordinates
    template<index_t NumBits = 2>
    __host__ __device__ void morton_decode_2d(
        index_t morton_idx, 
        index_t& y, 
        index_t& x) 
    {
        y = 0;
        x = 0;
        for (index_t i = 0; i < NumBits; ++i) {
            y |= ((morton_idx >> (2*i + 1)) & 1) << i;
            x |= ((morton_idx >> (2*i)) & 1) << i;
        }
    }

Morton Pattern Analysis
-----------------------

The Morton index layout in a 4×4 tile follows this pattern:

.. code-block:: text

    Morton Index Layout:
     0  1  4  5
     2  3  6  7
     8  9 12 13
    10 11 14 15

Bit pattern breakdown:

.. code-block:: text

    (0,0) = (00, 00) →  0 = 0000
    (0,1) = (00, 01) →  1 = 0001
    (0,2) = (00, 10) →  4 = 0100
    (0,3) = (00, 11) →  5 = 0101
    (1,0) = (01, 00) →  2 = 0010
    (1,1) = (01, 01) →  3 = 0011
    (1,2) = (01, 10) →  6 = 0110
    (1,3) = (01, 11) →  7 = 0111

Stage 1: Tiling with UnmergeTransform
======================================

First, we split our texture into tiles using tensor descriptors (see :ref:`ck_tile_descriptors` and :ref:`ck_tile_transforms`). This creates a hierarchical structure: (Y_blk, y_in, X_blk, x_in).

.. code-block:: cpp

    template<index_t H, index_t W, index_t TileSize>
    struct TiledTextureDescriptor {
        static constexpr index_t NumTilesY = H / TileSize;
        static constexpr index_t NumTilesX = W / TileSize;
        
        // Original descriptor for H×W texture
        using BaseDesc = TensorDescriptor<
            Sequence<H, W>,
            Sequence<W, 1>  // Row-major layout
        >;
        
        // Stage 1: Split into tiles
        // Transform: [H, W] → [NumTilesY, TileSize, NumTilesX, TileSize]
        using TiledDesc = decltype(
            transform_tensor_descriptor(
                BaseDesc{},
                make_tuple(
                    make_unmerge_transform(Sequence<NumTilesY, TileSize>{}),
                    make_unmerge_transform(Sequence<NumTilesX, TileSize>{})
                ),
                Sequence<0>{},     // Y dimension
                Sequence<1>{},     // X dimension
                Sequence<0, 1>{},  // Y → (Y_blk, y_in)
                Sequence<2, 3>{}   // X → (X_blk, x_in)
            )
        );
    };

Example usage for an 8×8 texture with 4×4 tiles:

.. code-block:: cpp

    // Create tiled descriptor
    using TiledDesc8x8 = TiledTextureDescriptor<8, 8, 4>::TiledDesc;
    
    // Access pattern: iterate tile by tile
    template<typename DataType>
    __device__ void process_tiled_texture(const DataType* texture) {
        TiledDesc8x8 desc;
        
        // Process each tile
        for (index_t y_blk = 0; y_blk < 2; ++y_blk) {
            for (index_t x_blk = 0; x_blk < 2; ++x_blk) {
                // Process elements within tile
                for (index_t y_in = 0; y_in < 4; ++y_in) {
                    for (index_t x_in = 0; x_in < 4; ++x_in) {
                        // Calculate offset using descriptor
                        index_t offset = desc.calculate_offset({
                            y_blk, y_in, x_blk, x_in
                        });
                        
                        DataType value = texture[offset];
                        // Process value...
                    }
                }
            }
        }
    }

Stage 2: Morton Ordering with MergeTransform
============================================

The key insight is that MergeTransform enables Morton ordering by reordering and merging coordinate bits. The transformation involves:

1. Split coordinates into individual bits using UnmergeTransform
2. Reorder and merge bits using MergeTransform to create the Morton index

This leverages the coordinate transformation system described in :ref:`ck_tile_coordinate_systems`.

Mathematical Foundation
-----------------------

.. code-block:: cpp

    template<index_t TileSize = 4>
    struct MortonTransform {
        static_assert(TileSize == 4, "This example assumes 4x4 tiles");
        
        // Split 4 → (2, 2) for bit extraction
        using SplitTransform = UnmergeTransform<Sequence<2, 2>>;
        
        // Merge bits in Morton order: (y₀, x₀, y₁, x₁) → Morton
        using MortonMergeTransform = MergeTransform<Sequence<2, 2, 2, 2>>;
        
        // The merge operation computes:
        // morton_idx = y₁×8 + x₁×4 + y₀×2 + x₀
        // This matches the bit interleaving pattern!
    };

Complete Morton Implementation
------------------------------

Here's a complete implementation combining both stages:

.. code-block:: cpp

    template<typename DataType, index_t H, index_t W, index_t TileSize = 4>
    struct MortonSwizzledTexture {
        static constexpr index_t NumTilesY = H / TileSize;
        static constexpr index_t NumTilesX = W / TileSize;
        
        // Manual Morton implementation for reliability
        __device__ static void apply_morton_swizzling(
            const DataType* input,
            DataType* output)
        {
            // Process each tile
            for (index_t tile_y = 0; tile_y < NumTilesY; ++tile_y) {
                for (index_t tile_x = 0; tile_x < NumTilesX; ++tile_x) {
                    // Apply Morton ordering within tile
                    for (index_t morton_idx = 0; morton_idx < TileSize * TileSize; ++morton_idx) {
                        // Decode Morton index to tile coordinates
                        index_t y_in, x_in;
                        morton_decode_2d<2>(morton_idx, y_in, x_in);
                        
                        // Calculate global coordinates
                        index_t global_y = tile_y * TileSize + y_in;
                        index_t global_x = tile_x * TileSize + x_in;
                        
                        // Calculate linear indices
                        index_t src_idx = global_y * W + global_x;
                        index_t dst_idx = (tile_y * NumTilesX + tile_x) * TileSize * TileSize + morton_idx;
                        
                        output[dst_idx] = input[src_idx];
                    }
                }
            }
        }
    };

Memory Access Pattern Analysis
==============================

An analysis of the benefits of Morton ordering for different access patterns:

.. code-block:: cpp

    template<index_t TileSize = 4>
    struct AccessPatternAnalyzer {
        // Analyze spatial locality
        __host__ static void analyze_morton_locality() {
            printf("Morton Order Spatial Locality Analysis:\n");
            printf("Adjacent indices and their 2D distance:\n");
            
            for (index_t i = 0; i < TileSize * TileSize - 1; ++i) {
                index_t y1, x1, y2, x2;
                morton_decode_2d<2>(i, y1, x1);
                morton_decode_2d<2>(i + 1, y2, x2);
                
                index_t manhattan_dist = abs(y2 - y1) + abs(x2 - x1);
                printf("Morton %2d→%2d: (%d,%d)→(%d,%d), distance: %d\n",
                       i, i+1, y1, x1, y2, x2, manhattan_dist);
            }
        }
        
        // Compare cache line usage
        __host__ static void analyze_cache_efficiency() {
            constexpr index_t CacheLineSize = 128;  // bytes
            constexpr index_t ElementSize = sizeof(float);
            constexpr index_t ElementsPerCacheLine = CacheLineSize / ElementSize;
            
            printf("\nCache Efficiency Analysis:\n");
            printf("Cache line size: %d bytes (%d floats)\n", 
                   CacheLineSize, ElementsPerCacheLine);
            
            // Row-major access
            index_t row_major_lines = 0;
            for (index_t y = 0; y < TileSize; ++y) {
                for (index_t x = 0; x < TileSize; x += ElementsPerCacheLine) {
                    row_major_lines++;
                }
            }
            
            // Morton access
            index_t morton_lines = 0;
            index_t current_line = -1;
            for (index_t i = 0; i < TileSize * TileSize; ++i) {
                index_t y, x;
                morton_decode_2d<2>(i, y, x);
                index_t linear_idx = y * TileSize + x;
                index_t cache_line = linear_idx / ElementsPerCacheLine;
                
                if (cache_line != current_line) {
                    morton_lines++;
                    current_line = cache_line;
                }
            }
            
            printf("Row-major: %d cache lines\n", row_major_lines);
            printf("Morton: %d cache lines\n", morton_lines);
        }
    };

GPU Kernel Implementation
=========================

A complete GPU kernel using Morton ordering for optimized memory access:

.. code-block:: cpp

    template<typename DataType, 
             index_t BlockSize = 16,
             index_t TileSize = 4>
    __global__ void morton_optimized_kernel(
        const DataType* __restrict__ input,
        DataType* __restrict__ output,
        index_t H, index_t W)
    {
        // Shared memory with Morton layout
        __shared__ DataType smem[BlockSize * BlockSize];
        
        // Thread and block indices
        const index_t tid_x = threadIdx.x;
        const index_t tid_y = threadIdx.y;
        const index_t bid_x = blockIdx.x;
        const index_t bid_y = blockIdx.y;
        
        // Global position
        const index_t global_x = bid_x * BlockSize + tid_x;
        const index_t global_y = bid_y * BlockSize + tid_y;
        
        // Load to shared memory with coalescing
        if (global_x < W && global_y < H) {
            smem[tid_y * BlockSize + tid_x] = input[global_y * W + global_x];
        }
        __syncthreads();
        
        // Process tiles with Morton ordering
        constexpr index_t TilesPerBlock = BlockSize / TileSize;
        
        // Each thread processes one element in Morton order
        const index_t tile_id = (tid_y / TileSize) * TilesPerBlock + (tid_x / TileSize);
        const index_t morton_in_tile = (tid_y % TileSize) * TileSize + (tid_x % TileSize);
        
        // Decode Morton index
        index_t y_in_tile, x_in_tile;
        morton_decode_2d<2>(morton_in_tile, y_in_tile, x_in_tile);
        
        // Calculate position in shared memory
        const index_t tile_y = tile_id / TilesPerBlock;
        const index_t tile_x = tile_id % TilesPerBlock;
        const index_t smem_y = tile_y * TileSize + y_in_tile;
        const index_t smem_x = tile_x * TileSize + x_in_tile;
        
        // Process with Morton access pattern
        DataType value = smem[smem_y * BlockSize + smem_x];
        
        // Apply computation...
        value = compute_function(value);
        
        // Store result
        if (global_x < W && global_y < H) {
            output[global_y * W + global_x] = value;
        }
    }

Bank Conflict Reduction
=======================

Morton ordering is particularly effective for reducing shared memory bank conflicts (complementing the XOR preshuffle technique described in :ref:`ck_tile_lds_index_swapping`):

.. code-block:: cpp

    template<index_t WarpSize = 32>
    struct BankConflictAnalysis {
        static constexpr index_t NumBanks = 32;
        static constexpr index_t BankWidth = 4;  // bytes
        
        template<typename AccessPattern>
        __host__ static void analyze_bank_conflicts(
            const char* pattern_name,
            AccessPattern access_func)
        {
            index_t bank_access[NumBanks] = {0};
            
            // Simulate warp access
            for (index_t tid = 0; tid < WarpSize; ++tid) {
                index_t offset = access_func(tid);
                index_t bank = (offset * sizeof(float) / BankWidth) % NumBanks;
                bank_access[bank]++;
            }
            
            // Find maximum conflict
            index_t max_conflict = 0;
            for (index_t bank = 0; bank < NumBanks; ++bank) {
                max_conflict = max(max_conflict, bank_access[bank]);
            }
            
            printf("%s: %d-way bank conflict\n", pattern_name, max_conflict);
        }
        
        __host__ static void compare_access_patterns() {
            printf("Bank Conflict Analysis for 4x4 Tile Access:\n");
            
            // Row-major access
            analyze_bank_conflicts("Row-major", [](index_t tid) {
                return (tid / 4) * 4 + (tid % 4);
            });
            
            // Morton access
            analyze_bank_conflicts("Morton", [](index_t tid) {
                index_t y, x;
                morton_decode_2d<2>(tid % 16, y, x);
                return y * 4 + x;
            });
        }
    };

Practical Applications
======================

Real-world usage of Morton ordering in CK Tile:

**1. Texture Cache Optimization**

.. code-block:: cpp

    template<typename DataType>
    struct TextureCacheOptimized {
        static constexpr index_t TextureTileSize = 8;
        
        __device__ static DataType sample_2d_morton(
            const DataType* texture,
            float u, float v,
            index_t width, index_t height)
        {
            // Convert normalized coordinates to texel coordinates
            index_t x = u * width;
            index_t y = v * height;
            
            // Determine tile
            index_t tile_x = x / TextureTileSize;
            index_t tile_y = y / TextureTileSize;
            
            // Position within tile
            index_t x_in_tile = x % TextureTileSize;
            index_t y_in_tile = y % TextureTileSize;
            
            // Convert to Morton index
            index_t morton_idx = morton_encode_2d<3>(y_in_tile, x_in_tile);
            
            // Calculate final offset
            index_t tile_offset = (tile_y * (width / TextureTileSize) + tile_x) 
                                  * TextureTileSize * TextureTileSize;
            
            return texture[tile_offset + morton_idx];
        }
    };

**2. Matrix Multiplication with Swizzled Tiles**

For complete GEMM optimization techniques, see :ref:`ck_tile_gemm_optimization`.

.. code-block:: cpp

    template<typename DataType, index_t TileM, index_t TileN, index_t TileK>
    struct SwizzledGEMM {
        __device__ static void load_tile_morton(
            const DataType* matrix,
            DataType* tile,
            index_t row_offset,
            index_t col_offset,
            index_t ld)
        {
            // Load tile with Morton ordering for better LDS bank utilization
            #pragma unroll
            for (index_t i = 0; i < TileM * TileN; ++i) {
                index_t row_in_tile, col_in_tile;
                morton_decode_2d<3>(i, row_in_tile, col_in_tile);
                
                if (row_in_tile < TileM && col_in_tile < TileN) {
                    index_t global_row = row_offset + row_in_tile;
                    index_t global_col = col_offset + col_in_tile;
                    tile[i] = matrix[global_row * ld + global_col];
                }
            }
        }
    };

Summary
=======

Morton ordering with CK Tile provides memory optimization capabilities:

- **Spatial Locality**: Z-order curve maintains 2D locality in 1D memory layout
- **Bank Conflict Reduction**: Distributed access patterns across memory banks
- **Cache Efficiency**: Better utilization of cache lines for 2D access patterns
- **Mathematical Framework**: Tensor descriptors express swizzling cleanly
- **Practical Implementation**: Bit manipulation provides reliable results

Key implementation insights:

1. **MergeTransform** is essential for expressing Morton bit interleaving
2. **Manual bit manipulation** provides reliable and efficient implementation
3. **Tiling + Morton** combines hierarchical locality with local optimization
4. **GPU-specific tuning** adapts patterns to hardware characteristics

The tensor descriptor approach provides the mathematical framework for expressing these complex memory patterns, while practical implementations often use direct bit manipulation for efficiency and reliability.

For more examples of practical CK Tile usage, see :ref:`ck_tile_convolution_example`. For the underlying buffer and tensor abstractions, see :ref:`ck_tile_buffer_views` and :ref:`ck_tile_tensor_views`.
