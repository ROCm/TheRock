// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {

/**
 * @brief Tile copy shape configuration
 *
 * @tparam BlockWaves Number of waves along seq<M, N>
 * @tparam BlockTile Block size, seq<M, N>
 * @tparam WaveTile Wave size, seq<M, N>
 * @tparam ThreadTile Contiguous elements per thread along seq<M, N>
 */
template <typename BlockWaves, typename BlockTile, typename WaveTile, typename ThreadTile>
struct TileCopyShape
{
    // ThreadTile dimensions for memory operations
    static constexpr index_t ThreadTile_M = ThreadTile::at(number<0>{});
    static constexpr index_t ThreadTile_N = ThreadTile::at(number<1>{});

    // Wave tile dimensions
    static constexpr index_t WaveSize    = get_warp_size();
    static constexpr index_t Wave_Tile_N = WaveTile::at(number<1>{});
    static constexpr index_t Wave_Tile_M = ThreadTile_M * ThreadTile_N * WaveSize / Wave_Tile_N;

    // Block tile dimensions
    static constexpr index_t Block_Tile_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_Tile_N = BlockTile::at(number<1>{});

    // Waves per block configuration
    static constexpr index_t Waves_Per_Block_M = BlockWaves::at(number<0>{});
    static constexpr index_t Waves_Per_Block_N = BlockWaves::at(number<1>{});

    // Calculate wave repetition to cover entire block tile
    static constexpr index_t WaveRepetitionPerBlock_M =
        Block_Tile_M / (Waves_Per_Block_M * Wave_Tile_M);
    static constexpr index_t WaveRepetitionPerBlock_N =
        Block_Tile_N / (Waves_Per_Block_N * Wave_Tile_N);

    // Hardware configuration
    static constexpr index_t BlockSize = Waves_Per_Block_M * Waves_Per_Block_N * WaveSize;

    // Configuration validation
    static_assert(Block_Tile_M > 0 && Block_Tile_N > 0, "Block tile dimensions must be positive");
    static_assert(Wave_Tile_M > 0 && Wave_Tile_N > 0, "Wave tile dimensions must be positive");
    static_assert(ThreadTile_M > 0 && ThreadTile_N > 0, "ThreadTile dimensions must be positive");
    static_assert(Waves_Per_Block_M > 0 && Waves_Per_Block_N > 0,
                  "Waves per block must be positive");
    static_assert(Waves_Per_Block_M * Wave_Tile_M > 0,
                  "Invalid wave configuration for M dimension");
    static_assert(Waves_Per_Block_N * Wave_Tile_N > 0,
                  "Invalid wave configuration for N dimension");

    // Ensure wave tile dimensions align with wave size
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert(Wave_Tile_M / ThreadTile_M * Wave_Tile_N / ThreadTile_N == WaveSize,
                  "(Wave_Tile_M/ThreadTile_M) * (Wave_Tile_N/ThreadTile_N) != WaveSize");
#endif
};

/**
 * @brief Problem definition for tile copy operation
 */
template <typename XDataType_, typename BlockShape_>
struct TileCopyProblem
{
    using XDataType  = remove_cvref_t<XDataType_>;
    using BlockShape = remove_cvref_t<BlockShape_>;
};

/**
 * @brief Policy for tile copy operation
 */
template <typename Problem_>
struct TileCopyPolicy
{
    using Problem   = ck_tile::remove_cvref_t<Problem_>;
    using XDataType = typename Problem::XDataType;

    /**
     * @brief Create DRAM distribution for optimal memory access
     */
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeDRAMDistribution()
    {
        using S = typename Problem::BlockShape;

        constexpr index_t wave_size  = S::WaveSize;
        constexpr index_t block_size = S::BlockSize;

        // Distribution calculation to ensure all threads participate
        constexpr index_t N1 = S::ThreadTile_N;      // Elements per thread along N
        constexpr index_t N0 = S::Block_Tile_N / N1; // Threads needed along N

        constexpr index_t M2 = wave_size / N0;              // Threads per wave along M
        constexpr index_t M1 = block_size / wave_size;      // Waves possible along M
        constexpr index_t M0 = S::Block_Tile_M / (M1 * M2); // Wave iterations along M

        // Validate complete coverage
        static_assert(M0 * M1 * M2 * N0 * N1 == S::Block_Tile_M * S::Block_Tile_N,
                      "Tile distribution must cover entire block tile");

        constexpr auto outer_encoding =
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<N0, N1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{};
        return make_static_tile_distribution(outer_encoding);
    }
};

/**
 * @brief Direct copy kernel from global memory to global memory
 */
template <typename Problem_, typename Policy_>
struct TileCopyKernel
{
    using Problem   = ck_tile::remove_cvref_t<Problem_>;
    using XDataType = typename Problem::XDataType;
    using Policy    = ck_tile::remove_cvref_t<Policy_>;

    CK_TILE_DEVICE void operator()(const XDataType* p_x, XDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        // Calculate tile block origin and validate bounds
        // Use __builtin_amdgcn_readfirstlane to broadcast the same value to all threads in a wave
        // This saves VGPR usage by avoiding per-thread storage of the same value
        const auto tile_block_origin_m =
            __builtin_amdgcn_readfirstlane(get_block_id() * S::Block_Tile_M);
        if(tile_block_origin_m >= M)
        {
            return; // Early exit for out-of-bounds blocks
        }

        // Create tensor views for input and output
        const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(M, N), make_tuple(N, 1), number<S::ThreadTile_N>{}, number<1>{});

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_y, make_tuple(M, N), make_tuple(N, 1), number<S::ThreadTile_N>{}, number<1>{});

        // Create tile windows with DRAM distribution
        auto x_window = make_tile_window(x_m_n,
                                         make_tuple(S::Block_Tile_M, S::Block_Tile_N),
                                         {tile_block_origin_m, 0},
                                         Policy::template MakeDRAMDistribution<Problem>());

        auto y_window = make_tile_window(y_m_n,
                                         make_tuple(S::Block_Tile_M, S::Block_Tile_N),
                                         {tile_block_origin_m, 0},
                                         Policy::template MakeDRAMDistribution<Problem>());

        // Calculate iterations needed to cover N dimension
        // Note: This kernel uses data parallelism only in the M dimension.
        // Each block processes one tile in M dimension, but iterates through N dimension tiles.
        // This design choice is for simplicity and to avoid complex tile distribution.
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_Tile_N));

        // Get tile distribution for register tensor
        auto DramTileDist   = x_window.get_tile_distribution();
        using dram_reg_tile = decltype(make_static_distributed_tensor<XDataType>(DramTileDist));

        // Main copy loop - processes N dimension tiles sequentially within each block
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            dram_reg_tile dram_tile;

            // Direct copy implementation
            load_tile(dram_tile, x_window);
            store_tile(y_window, dram_tile);

            // Move to next N tile
            move_tile_window(x_window, {0, S::Block_Tile_N});
            move_tile_window(y_window, {0, S::Block_Tile_N});
        }
    }
};

/**
 * @brief Element-wise copy kernel for data transformation scenarios
 *
 * This kernel performs element-wise copy operations, allowing for data transformation
 * during the copy process. Useful when data needs to be processed or converted
 * between different formats.
 */
template <typename Problem_, typename Policy_>
struct ElementWiseTileCopyKernel
{
    using Problem   = ck_tile::remove_cvref_t<Problem_>;
    using XDataType = typename Problem::XDataType;
    using Policy    = ck_tile::remove_cvref_t<Policy_>;

    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;

    CK_TILE_HOST static auto BlockSize()
    {
        if(ck_tile::is_wave32())
        {
            return kBlockSize / 2;
        }
        else
        {
            return kBlockSize;
        }
    }
    CK_TILE_DEVICE void operator()(const XDataType* p_x, XDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        // Calculate block origin and validate bounds
        // Use __builtin_amdgcn_readfirstlane to broadcast the same value to all threads in a wave
        // This saves VGPR usage by avoiding per-thread storage of the same value
        const auto tile_block_origin_m =
            __builtin_amdgcn_readfirstlane(get_block_id() * S::Block_Tile_M);
        if(tile_block_origin_m >= M)
        {
            return; // Early exit for out-of-bounds blocks
        }

        // Create tensor views for input and output
        const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(M, N), make_tuple(N, 1), number<S::ThreadTile_N>{}, number<1>{});

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_y, make_tuple(M, N), make_tuple(N, 1), number<S::ThreadTile_N>{}, number<1>{});

        // Create tile windows with DRAM distribution
        auto x_window = make_tile_window(x_m_n,
                                         make_tuple(S::Block_Tile_M, S::Block_Tile_N),
                                         {tile_block_origin_m, 0},
                                         Policy::template MakeDRAMDistribution<Problem>());

        auto y_window = make_tile_window(y_m_n,
                                         make_tuple(S::Block_Tile_M, S::Block_Tile_N),
                                         {tile_block_origin_m, 0},
                                         Policy::template MakeDRAMDistribution<Problem>());

        // Calculate iterations needed to cover N dimension
        // Note: This kernel uses data parallelism only in the M dimension.
        // Each block processes one tile in M dimension, but iterates through N dimension tiles.
        // This design choice is for simplicity and to avoid complex tile distribution.
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_Tile_N));

        // Main element-wise copy loop - processes N dimension tiles sequentially within each block
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            // Element-wise copy implementation for data transformation
            const auto xa  = load_tile(x_window);
            auto y_compute = load_tile(y_window);

            constexpr auto spans = decltype(xa)::get_distributed_spans();

            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                    const auto x           = ck_tile::type_convert<XDataType>(xa[i_j_idx]);
                    y_compute(i_j_idx)     = x;
                });
            });

            store_tile(y_window, y_compute);

            // Move to next N tile
            move_tile_window(x_window, {0, S::Block_Tile_N});
            move_tile_window(y_window, {0, S::Block_Tile_N});
        }
    }
};

/**
 * @brief LDS-based copy kernel for data processing scenarios
 *
 * This kernel copies data from global memory to LDS and then to global memory,
 * useful when data needs to be processed or transformed during the copy operation.
 */
template <typename Problem_, typename Policy_>
struct TileCopyKernel_LDS
{
    using Problem   = ck_tile::remove_cvref_t<Problem_>;
    using XDataType = typename Problem::XDataType;
    using Policy    = ck_tile::remove_cvref_t<Policy_>;

    CK_TILE_DEVICE void operator()(const XDataType* p_x, XDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        // Calculate block origin and validate bounds
        // Use __builtin_amdgcn_readfirstlane to broadcast the same value to all threads in a wave
        // This saves VGPR usage by avoiding per-thread storage of the same value
        const auto tile_block_origin_m =
            __builtin_amdgcn_readfirstlane(get_block_id() * S::Block_Tile_M);
        if(tile_block_origin_m >= M)
        {
            return; // Early exit for out-of-bounds blocks
        }

        // LDS buffer allocation
        __shared__ XDataType x_lds_buffer[S::Block_Tile_Mmake * S::Block_Tile_N];

        // LDS tensor descriptor and view
        const auto x_lds_descriptor =
            make_naive_tensor_descriptor(make_tuple(S::Block_Tile_M, S::Block_Tile_N),
                                         make_tuple(S::Block_Tile_N, 1),
                                         number<S::ThreadTile_N>{},
                                         number<1>{});

        auto x_lds_view = make_tensor_view<address_space_enum::lds>(x_lds_buffer, x_lds_descriptor);

        // LDS windows with different distributions for optimal access patterns
        auto x_lds_write_window =
            make_tile_window(x_lds_view, make_tuple(S::Block_Tile_M, S::Block_Tile_N), {0, 0});

        auto x_lds_read_window = make_tile_window(x_lds_view,
                                                  make_tuple(S::Block_Tile_M, S::Block_Tile_N),
                                                  {0, 0},
                                                  Policy::template MakeDRAMDistribution<Problem>());

        // Global memory tensor views
        const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(M, N), make_tuple(N, 1), number<S::ThreadTile_N>{}, number<1>{});

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_y, make_tuple(M, N), make_tuple(N, 1), number<S::ThreadTile_N>{}, number<1>{});

        // Global memory tile windows
        auto x_window = make_tile_window(x_m_n,
                                         make_tuple(S::Block_Tile_M, S::Block_Tile_N),
                                         {tile_block_origin_m, 0},
                                         Policy::template MakeDRAMDistribution<Problem>());

        auto y_window = make_tile_window(
            y_m_n, make_tuple(S::Block_Tile_M, S::Block_Tile_N), {tile_block_origin_m, 0});

        // Calculate iterations needed to cover N dimension
        // Note: This kernel uses data parallelism only in the M dimension.
        // Each block processes one tile in M dimension, but iterates through N dimension tiles.
        // This design choice is for simplicity and to avoid complex tile distribution.
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_Tile_N));

        // Main copy loop with LDS staging - processes N dimension tiles sequentially within each
        // block
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            // Global memory to LDS
            auto dram_tile = load_tile(x_window);
            store_tile(x_lds_write_window, dram_tile);

            // Synchronize LDS access
            block_sync_lds();

            // LDS to global memory
            auto lds_tile = load_tile(x_lds_read_window);
            store_tile(y_window, lds_tile);

            // Move to next N tile
            move_tile_window(x_window, {0, S::Block_Tile_N});
            move_tile_window(y_window, {0, S::Block_Tile_N});
        }
    }
};

} // namespace ck_tile
