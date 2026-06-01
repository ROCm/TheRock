// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * GemmTilePartitioner allows customized mapping between a workgroup and the C-tile it computes.
 */

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

/**
 * @brief Class providing 2D workgroup index mapping into 2D output GEMM C-tile space.
 *
 */
template <typename BlockGemmShapeType>
struct GemmTile2DPartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShapeType>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST_DEVICE GemmTile2DPartitioner() noexcept = delete;
    CK_TILE_HOST_DEVICE GemmTile2DPartitioner([[maybe_unused]] index_t M,
                                              [[maybe_unused]] index_t N) noexcept;

    /**
     * @brief Calculates GEMM kernel grid size.
     *
     * @param M     GEMM's M dimension.
     * @param N     GEMM's N dimension.
     * @return dim3 Structure holding grid's X,Y and Z dimensions.
     */
    CK_TILE_HOST static auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> dim3
    {
        const index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        const index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        return dim3(GridDimX, GridDimY, 1);
    }

    /**
     * @brief Calculate number of loop iterations over GEMM's K dimension.
     *
     * @param K         GEMM's K dimension.
     * @return index_t  The number of loop iterations over K dimension.
     */
    CK_TILE_HOST_DEVICE static auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /**
     * @brief The function returns 2D output tile space.
     * @param [in] blockIdx is blockIdx.x
     * @param [in] blockIdy is blockIdx.y
     * @return Returns the output tile indexes.
     */

    /**
     * @brief Calculate workgroup 2D index mapping into 2D output C-tile space.
     *
     * @param blockIdx      WGP's X index.
     * @param blockIdy      WGP's Y index.
     * @return const tuple<index_t, index_t>    Tuple containing 2D output C-tile index.
     */
    CK_TILE_DEVICE static auto
    GetOutputTileIndex(index_t blockIdx, index_t blockIdy) noexcept -> const tuple<index_t, index_t>
    {
        const index_t iM = amd_wave_read_first_lane(blockIdx);
        const index_t iN = amd_wave_read_first_lane(blockIdy);
        return make_tuple(iM, iN);
    }
};

/**
 * @brief Class providing 1D WGP index mapping into 2D output C-tile space.
 *
 * @tparam BlockGemmShape_  A class providing basic GEMM parameters. \link TileGemmShape
 */
template <typename BlockGemmShape_>
struct GemmTile1DPartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST_DEVICE GemmTile1DPartitioner() noexcept = delete;

    /**
     * @brief Construct a new GemmTile1DPartitioner object.
     *
     * @param M     GEMM's M dimension.
     * @param N     GEMM's N dimension.
     */
    CK_TILE_HOST_DEVICE GemmTile1DPartitioner([[maybe_unused]] index_t M, index_t N) noexcept
    {
        N_ = N;
    }

    /**
     * @brief Calculates GEMM kernel grid size.
     *
     * @param M     GEMM's M dimension.
     * @param N     GEMM's N dimension.
     * @return dim3 Structure holding grid's X,Y and Z dimensions.
     */
    CK_TILE_HOST_DEVICE static auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> index_t
    {
        const index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        const index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        return GridDimX * GridDimY;
    }

    /**
     * @brief Calculate number of loop iterations over GEMM's K dimension.
     *
     * @param K         GEMM's K dimension.
     * @return index_t  The number of loop iterations over K dimension.
     */
    CK_TILE_HOST_DEVICE static auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /**
     * @brief Calculate workgroup 1D index mapping into 2D output C-tile space.
     *
     * @param blockIdx      WGP's index.
     * @return const tuple<index_t, index_t>    Tuple containing 2D output C-tile index.
     */
    CK_TILE_DEVICE static auto
    GetOutputTileIndex(index_t blockIdx) noexcept -> const tuple<index_t, index_t>
    {
        const index_t NBlocks = integer_divide_ceil(N_, NPerBlock);

        const index_t iM = amd_wave_read_first_lane(blockIdx / NBlocks);
        const index_t iN = amd_wave_read_first_lane(blockIdx - iM * NBlocks);
        return make_tuple(iM, iN);
    }

    private:
    CK_TILE_DEVICE static index_t N_;
};

/**
 * @brief `GemmTile1DPartitioner::GetOutputTileIndex`'s std::false specialization,
 * checking expression validity in-place for ill-formed.
 */
template <typename, typename = void>
struct HasFnOneArgImpl : std::false_type
{
};

/**
 * @brief `GemmTile1DPartitioner::GetOutputTileIndex`'s std::true specialization,
 * checking expression validity in-place for well-formed.
 * @note: `1` - a constant value indicating the number of parameters in the function.
 */
template <typename T>
struct HasFnOneArgImpl<T, std::void_t<decltype(std::declval<T>().GetOutputTileIndex(1))>>
    : std::true_type
{
};

/**
 * @brief Struct used to calculate offseted tile indexes.
 * @note: The struct supports the 1D-Partitioner mechanism,
 * enable-if `GetOutputTileIndex`-fn is std::true_type when `GetOutputTileIndex`-fn is well-formed,
 * otherwise std::false_type.
 */
template <typename TilePartitioner,
          typename = typename std::enable_if_t<HasFnOneArgImpl<TilePartitioner>{}>>
struct OffsettedTile1DPartitioner
{
    /**
     * @brief The function subtracts the block's start (offset) from 1D raw-indexes.
     * @param [in] block_start Workgroup offset.
     * @param [in] M           Gemm's M dimension.
     * @param [in] N           Gemm's N dimension.
     * @return Returns a `tuple` [Im, In] with shifted index.
     */
    [[nodiscard]] CK_TILE_DEVICE static auto GetOffsetedTileIndex(
        index_t block_start, index_t M, index_t N) noexcept -> const tuple<index_t, index_t>
    {
        const auto [iM, iN] = TilePartitioner{M, N}.GetOutputTileIndex(blockIdx.x - block_start);
        return make_tuple(iM, iN);
    }

    /**
     * @brief The function subtracts the block's start (offset) from a given block index.
     * @param [in] block_start Workgroup offset.
     * @param [in] M           Gemm's M dimension.
     * @param [in] N           Gemm's N dimension.
     * @param [in] block_idx   Current block index of the workgroup.
     * @return Returns a `tuple` [Im, In] with shifted index.
     */
    [[nodiscard]] CK_TILE_DEVICE static auto
    GetOffsetedTileIndex(index_t block_start, index_t M, index_t N, index_t block_idx) noexcept
        -> const tuple<index_t, index_t>
    {
        const auto [iM, iN] = TilePartitioner{M, N}.GetOutputTileIndex(block_idx - block_start);
        return make_tuple(iM, iN);
    }
};

/**
 * @brief Class mapping 1D block index into 2D output tile space.
 *
 * @note It groups spatially workgroups in order to better utilize caches.
 *       It is using grouped Rows of column-vectors WGP pattern. It's optimized
 *       for gfx94x-like multiple-die chip.
 *
 * @tparam GroupNum - The number of big groups.
 * @tparam M01      - The number of groups in M dim within spatially local WGPs,
 *
 */
template <typename BlockGemmShapeType, index_t GroupNum, index_t M01>
struct GemmSpatiallyLocalTilePartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShapeType>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST_DEVICE GemmSpatiallyLocalTilePartitioner() noexcept = delete;
    CK_TILE_HOST_DEVICE GemmSpatiallyLocalTilePartitioner(index_t M_, index_t N_) noexcept
        : M(M_), N(N_)
    {
    }

    /**
     * @brief Calculates GEMM kernel grid size.
     *
     * @param M     GEMM's M dimension.
     * @param N     GEMM's N dimension.
     * @return index_t A total number of workgroups.
     */
    CK_TILE_HOST_DEVICE static auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> index_t
    {
        const index_t GridDimX = integer_divide_ceil(M, MPerBlock);
        const index_t GridDimY = integer_divide_ceil(N, NPerBlock);
        return GridDimX * GridDimY;
    }

    /**
     * @brief Calculate number of loop iterations over GEMM's K dimension.
     *
     * @param K         GEMM's K dimension.
     * @return index_t  The number of loop iterations over K dimension.
     */
    CK_TILE_HOST_DEVICE static auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /**
     * @brief Calculate workgroup 1D index mapping into 2D output C-tile space.
     *
     * @param [in] block_1d_id      WGP's index.
     * @return const tuple<index_t, index_t>    Tuple containing 2D output C-tile index.
     */
    CK_TILE_DEVICE auto
    GetOutputTileIndex(index_t block_1d_id) noexcept -> const tuple<index_t, index_t>
    {
        const auto M0 = integer_divide_ceil(M, MPerBlock);
        const auto N0 = integer_divide_ceil(N, NPerBlock);

        if(M0 == 1)
        {
            return make_tuple(0, block_1d_id);
        }
        else if(N0 == 1)
        {
            return make_tuple(block_1d_id, 0);
        }
        // block_1d_id = block_1d_id % (M0 * N0); // swallow batch index
        else
        {
            const auto group_size    = integer_divide_ceil(M0 * N0, GroupNum);
            const auto big_group_num = GroupNum - (group_size * GroupNum - M0 * N0);
            const auto group_id_y    = block_1d_id / GroupNum;
            const auto group_id_x    = block_1d_id - group_id_y * GroupNum;
            const auto remap_block_1d_id =
                group_id_x <= big_group_num
                    ? group_id_x * group_size + group_id_y
                    : group_id_x * group_size + big_group_num - group_id_x + group_id_y;

            const index_t idx_M0 = remap_block_1d_id / N0;
            const index_t idx_N0 = remap_block_1d_id - idx_M0 * N0;

            const index_t M0_tmp     = M0 / M01;
            const index_t M0_mod_M01 = M0 - M0_tmp * M01;

            const auto M01_adapt = (idx_M0 < M0 - M0_mod_M01) ? M01 : M0_mod_M01;

            const index_t idx_M00          = idx_M0 / M01;
            const index_t idx_M01          = idx_M0 - idx_M00 * M01;
            const index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

            /**
             *                        idxN0
             *
             *           |<               mtx   N                 >|
             *
             *             NPerBlock   NPerBlock   NPerBlock   NPerBlock
             *                N_0         N_1        N_2         N_3
             *       -   |-----------|-----------|-----------|-----|-----|-
             *       ^   | -   -  0  |/---->  2  |           |     |     |
             *           | |   |     /     |     |           |     |     |  M_0  MPerBlock
             *           | M   |    /|     |     |           |     |     |
             *           |-0---|---/-|-----|-----|-----------|-----|-----|-
             *           | 1   |  /  |     |     |  blockid  |     |     |
             * idxM0     | |   | /   |     V     |     5     |     |     |  M_1  MPerBlock
             *           | -   V   1 |     -  3  |           |     |     |
             *           |-----------|-----------|-----------|-----|-----|-
             *    mtx M  |           |           |           |     |     |
             *           |           |           |           |     |     |  M_2  MPerBlock
             *           |           |           |           |     |     |
             *           |-----------|-----------|-----------|-----|-----|-
             *           |           |           |           |     |     |
             *           |           |           |           |     |     |  M_3  MPerBlock
             *           |           |           |           |     |     |
             *           |-----------|-----------|-----------|-----|-----|-
             *       V   |           |           |           |     |     |
             *       -   |-----------|-----------|-----------|-----|-----|- M_4  MPerBlock
             *           |           |           |           |     |     |
             *           |-----------|-----------|-----------|-----|-----|-
             *  Example:
             *   assume:
             *      M0 = 5
             *      N0 = 4
             *      block_1d_id = 5
             *      M01 = 2
             *
             *   idx_N0 = 1
             *   idx_M0 = 1
             *   M01_adapt = 2
             *   idx_M00 = 0
             *   idx_M01 = 1
             *   idx_N0_M01_local = 5
             *   output {1, 2}
             */

            const index_t N_out           = idx_N0_M01_local / M01_adapt;
            const index_t idx_loc_mod_M01 = idx_N0_M01_local - N_out * M01_adapt;

            return make_tuple(idx_loc_mod_M01 + idx_M00 * M01, N_out);
        }
    }

    private:
    index_t M;
    index_t N;
};
} // namespace ck_tile
