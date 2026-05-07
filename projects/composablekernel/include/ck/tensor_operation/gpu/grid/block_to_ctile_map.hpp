// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/math.hpp"
#include "ck/utility/number.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
#include <limits>
#include <stdlib.h>
#endif

namespace ck {

// Rows of column-vectors
template <index_t MPerBlock,
          index_t NPerBlock,
          typename CGridDesc_M_N,
          bool DeviceCTileIndexCheck = false>
struct BlockToCTileMap_M00_N0_M01
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ __device__ constexpr BlockToCTileMap_M00_N0_M01() = default;

    __host__ __device__ constexpr BlockToCTileMap_M00_N0_M01(const CGridDesc_M_N& c_grid_desc_m_n,
                                                             index_t M01 = 1)
        : M01_(M01), underlying_map_(GetBlockToCTileMap(c_grid_desc_m_n, M01))
    {
    }

    __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01_);

        const index_t grid_size = M00 * M01_ * N0;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        return underlying_map_.CalculateBottomIndex(idx_top);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ constexpr bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                                       const CTileDim& c_tile_dim) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return DefaultValidCTileIndex(c_tile_idx, c_tile_dim);
        else
            return true;
    }

    __host__ constexpr bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return true; // validity check moved to kernel

        const index_t M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        if(M0 % M01_ == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    private:
    __host__ __device__ static constexpr auto
    GetBlockToCTileMap(const CGridDesc_M_N& c_grid_desc_m_n, index_t M01)
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01);

        const auto m00_n0_m01_to_m0_n0_block_cluster_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_insert_transform(1),
                       make_unmerge_transform(make_tuple(M00, M01)),
                       make_pass_through_transform(make_tuple(N0))),
            make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}));

        const auto cblockid_to_m00_n0_m01_block_cluster_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(1, M00, N0, M01))),
            make_tuple(Sequence<0, 1, 2, 3>{}),
            make_tuple(Sequence<0>{}));

        const auto cblockid_to_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(m00_n0_m01_to_m0_n0_block_cluster_adaptor,
                                  cblockid_to_m00_n0_m01_block_cluster_adaptor);

        return cblockid_to_m0_n0_block_cluster_adaptor;
    }

    index_t M01_;
    using UnderlyingMap = decltype(GetBlockToCTileMap(CGridDesc_M_N{}, 1));
    UnderlyingMap underlying_map_;
};

// Rows of column-vectors
// This C-tile map dynamically adjusts M01 when C-tile index is out of range
template <index_t MPerBlock, index_t NPerBlock, typename CGridDesc_M_N = void>
struct BlockToCTileMap_M00_N0_M01Adapt;

template <index_t MPerBlock, index_t NPerBlock>
struct BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, void>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ constexpr BlockToCTileMap_M00_N0_M01Adapt() = default;

    __host__ __device__ constexpr BlockToCTileMap_M00_N0_M01Adapt(
        const BlockToCTileMap_M00_N0_M01Adapt&) = default;
    __host__ __device__ constexpr BlockToCTileMap_M00_N0_M01Adapt(
        BlockToCTileMap_M00_N0_M01Adapt&&) = default;
    __host__ __device__ constexpr BlockToCTileMap_M00_N0_M01Adapt&
    operator=(const BlockToCTileMap_M00_N0_M01Adapt&) = default;
    __host__ __device__ constexpr BlockToCTileMap_M00_N0_M01Adapt&
    operator=(BlockToCTileMap_M00_N0_M01Adapt&&) = default;

    __host__
        __device__ constexpr BlockToCTileMap_M00_N0_M01Adapt(index_t M, index_t N, index_t M01 = 8)
        : M_(M), N_(N), M01_(M01)
    {
#if 0
        if(get_thread_global_1d_id()==0){
            printf("Ctor called, M= %d, N= %d, M01 = %d\n", M_, N_, M01_);
        }
#endif
    }

    template <typename CGridDesc_M_N>
    __host__
        __device__ constexpr BlockToCTileMap_M00_N0_M01Adapt(const CGridDesc_M_N& c_grid_desc_m_n,
                                                             index_t M01 = 8)
        : BlockToCTileMap_M00_N0_M01Adapt(
              c_grid_desc_m_n.GetLength(I0), c_grid_desc_m_n.GetLength(I1), M01)
    {
    }

    __host__ __device__ static constexpr index_t CalculateGridSize(index_t M, index_t N)
    {
        const auto M0 = math::integer_divide_ceil(M, MPerBlock);
        const auto N0 = math::integer_divide_ceil(N, NPerBlock);

        return M0 * N0;
    }

    template <typename CGridDesc_M_N>
    __host__ static constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        return CalculateGridSize(c_grid_desc_m_n.GetLength(I0), c_grid_desc_m_n.GetLength(I1));
    }

    template <typename CGridDesc_M_N>
    __host__ __device__ constexpr bool
    CheckValidity(const CGridDesc_M_N& /* c_grid_desc_m_n */) const
    {
        return true;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        auto block_1d_id = idx_top[I0];

        const auto M0 = math::integer_divide_ceil(M_, MPerBlock);
        const auto N0 = math::integer_divide_ceil(N_, NPerBlock);

        block_1d_id = block_1d_id % (M0 * N0); // swallow batch index

        index_t idx_N0 = block_1d_id % N0;
        index_t idx_M0 = block_1d_id / N0;

        const auto M01_adapt = (idx_M0 < M0 - M0 % M01_) ? M01_ : M0 % M01_;

        index_t idx_M00          = idx_M0 / M01_;
        index_t idx_M01          = idx_M0 % M01_;
        index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

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

        return make_tuple(idx_N0_M01_local % M01_adapt + idx_M00 * M01_,
                          idx_N0_M01_local / M01_adapt);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ constexpr bool ValidCTileIndex(const CTileIdx& /* c_tile_idx */,
                                                       const CTileDim& /* c_tile_dim */) const
    {
        return true; // always valid provided that user gets grid size from CalculateGridSize()
    }

    private:
    index_t M_;
    index_t N_;
    index_t M01_;
};

// keep the redundant type argument for backward compatibility
template <index_t MPerBlock, index_t NPerBlock, typename CGridDesc_M_N>
struct BlockToCTileMap_M00_N0_M01Adapt : BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, void>
{
    using BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, void>::
        BlockToCTileMap_M00_N0_M01Adapt;
};

// Grouped Rows of column-vectors WGP mapping
// Optimized for gfx94x-like multipe-die chip

template <index_t GroupNum, index_t MPerBlock, index_t NPerBlock>
struct BlockToCTileMap_Grouped_M00_N0_M01Adapt
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ BlockToCTileMap_Grouped_M00_N0_M01Adapt() = default;
    __host__ __device__ BlockToCTileMap_Grouped_M00_N0_M01Adapt(index_t M,
                                                                index_t N,
                                                                index_t M01 = 8)
        : M_(M), N_(N), M01_(M01)
    {
    }

    __host__ __device__ static constexpr index_t CalculateGridSize(index_t M, index_t N)
    {
        const auto M0 = math::integer_divide_ceil(M, MPerBlock);
        const auto N0 = math::integer_divide_ceil(N, NPerBlock);

        return M0 * N0;
    }

    template <typename CGridDesc_M_N>
    __host__ bool CheckValidity(const CGridDesc_M_N& /* c_grid_desc_m_n */) const
    {
        return true;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        auto block_1d_id = idx_top[I0];

        const auto M0 = math::integer_divide_ceil(M_, MPerBlock);
        const auto N0 = math::integer_divide_ceil(N_, NPerBlock);

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
            const auto group_size    = math::integer_divide_ceil(M0 * N0, GroupNum);
            const auto big_group_num = GroupNum - (group_size * GroupNum - M0 * N0);
            auto group_id_x          = block_1d_id % GroupNum;
            auto group_id_y          = block_1d_id / GroupNum;
            auto remap_block_1d_id =
                group_id_x <= big_group_num
                    ? group_id_x * group_size + group_id_y
                    : group_id_x * group_size + big_group_num - group_id_x + group_id_y;

            index_t idx_N0 = remap_block_1d_id % N0;
            index_t idx_M0 = remap_block_1d_id / N0;

            const auto M01_adapt = (idx_M0 < M0 - M0 % M01_) ? M01_ : M0 % M01_;

            index_t idx_M00          = idx_M0 / M01_;
            index_t idx_M01          = idx_M0 % M01_;
            index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

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

            return make_tuple(idx_N0_M01_local % M01_adapt + idx_M00 * M01_,
                              idx_N0_M01_local / M01_adapt);
        }
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& /* c_tile_idx */,
                                             const CTileDim& /* c_tile_dim */) const
    {
        return true; // always valid provided that user gets grid size from CalculateGridSize()
    }

    private:
    index_t M_;
    index_t N_;
    index_t M01_;
};

// columns of row-vectors
// This C-tile map dynamically adjusts N01 when C-tile index is out of range
template <index_t MPerBlock, index_t NPerBlock, typename CGridDesc_M_N = void>
struct BlockToCTileMap_N00_M0_N01Adapt;

template <index_t MPerBlock, index_t NPerBlock>
struct BlockToCTileMap_N00_M0_N01Adapt<MPerBlock, NPerBlock, void>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ BlockToCTileMap_N00_M0_N01Adapt() = default;

    __host__ __device__ BlockToCTileMap_N00_M0_N01Adapt(const BlockToCTileMap_N00_M0_N01Adapt&) =
        default;
    __host__ __device__ BlockToCTileMap_N00_M0_N01Adapt(BlockToCTileMap_N00_M0_N01Adapt&&) =
        default;
    __host__ __device__ BlockToCTileMap_N00_M0_N01Adapt&
    operator=(const BlockToCTileMap_N00_M0_N01Adapt&) = default;
    __host__ __device__ BlockToCTileMap_N00_M0_N01Adapt&
    operator=(BlockToCTileMap_N00_M0_N01Adapt&&) = default;

    __host__ __device__ BlockToCTileMap_N00_M0_N01Adapt(index_t M, index_t N, index_t N01 = 8)
        : M_(M), N_(N), N01_(N01)
    {
#if 0
        if(get_thread_global_1d_id()==0){
            printf("Ctor called, M= %d, N= %d, N01 = %d\n", M_, N_, N01_);
        }
#endif
    }

    template <typename CGridDesc_M_N>
    __host__ __device__ BlockToCTileMap_N00_M0_N01Adapt(const CGridDesc_M_N& c_grid_desc_m_n,
                                                        index_t N01 = 8)
        : BlockToCTileMap_N00_M0_N01Adapt(
              c_grid_desc_m_n.GetLength(I0), c_grid_desc_m_n.GetLength(I1), N01)
    {
    }

    __host__ __device__ static constexpr index_t CalculateGridSize(index_t M, index_t N)
    {
        const auto M0 = math::integer_divide_ceil(M, MPerBlock);
        const auto N0 = math::integer_divide_ceil(N, NPerBlock);

        return M0 * N0;
    }

    template <typename CGridDesc_M_N>
    __host__ static constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        return CalculateGridSize(c_grid_desc_m_n.GetLength(I0), c_grid_desc_m_n.GetLength(I1));
    }

    template <typename CGridDesc_M_N>
    __host__ bool CheckValidity(const CGridDesc_M_N& /* c_grid_desc_m_n */) const
    {
        return true;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        auto block_1d_id = idx_top[I0];

        const auto M0 = math::integer_divide_ceil(M_, MPerBlock);
        const auto N0 = math::integer_divide_ceil(N_, NPerBlock);

        block_1d_id = block_1d_id % (M0 * N0); // swallow batch index

        index_t idx_M0 = block_1d_id % M0;
        index_t idx_N0 = block_1d_id / M0;

        const auto N01_adapt = (idx_N0 < N0 - N0 % N01_) ? N01_ : N0 % N01_;

        index_t idx_N00          = idx_N0 / N01_;
        index_t idx_N01          = idx_N0 % N01_;
        index_t idx_M0_N01_local = idx_M0 + idx_N01 * M0;

        /**
         *                        idxN0
         *
         *           |<               mtx   N                 >|
         *
         *                 |<---N01--->|
         *       -   |-----------|-----------|-----------|-----|-----|-
         *       ^   |     0 ---------->  1  |           |     |     |
         *           |           |   /       |           |     |     |  M_0  MPerBlock
         *           |           /           |           |     |     |
         *           |------/----------------|-----------|-----|-----|-
         *           |     |     |           |           |     |     |
         * idxM0     |     V     |           |           |     |     |  M_1  MPerBlock
         *           |     2 ---------->  3  |           |     |     |
         *           |-----------|-----------|-----------|-----|-----|-
         *    mtx M  |           |  blockid  |           |     |     |
         *           |           |    5      |           |     |     |  M_2  MPerBlock
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
         *             NPerBlock   NPerBlock   NPerBlock   NPerBlock
         *                 N_0         N_1        N_2         N_3
         *  Example:
         *   assume:
         *      N0 = 5
         *      M0 = 4
         *      block_1d_id = 5
         *      N01 = 2
         *
         *   idx_M0 = 1
         *   idx_N0 = 1
         *   N01_adapt = 2
         *   idx_N00 = 0
         *   idx_N01 = 1
         *   idx_M0_N01_local = 5
         *   output {2, 1}
         */

        return make_tuple(idx_M0_N01_local / N01_adapt,
                          idx_M0_N01_local % N01_adapt + idx_N00 * N01_);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& /* c_tile_idx */,
                                             const CTileDim& /* c_tile_dim */) const
    {
        return true; // always valid provided that user gets grid size from CalculateGridSize()
    }

    private:
    index_t M_;
    index_t N_;
    index_t N01_;
};

// 2D slices of column-vectors in 3D space
// This C-tile map dynamically adjusts M01 when C-tile index is out of range
template <index_t MPerBlock, index_t NPerBlock, typename CGridDesc_M_N>
struct BlockToCTileMap_KSplit_M00_N0_M01Adapt
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ __device__ BlockToCTileMap_KSplit_M00_N0_M01Adapt() = default;

    __host__ __device__ BlockToCTileMap_KSplit_M00_N0_M01Adapt(const CGridDesc_M_N& c_grid_desc_m_n,
                                                               index_t M01    = 8,
                                                               index_t KSplit = 1)
        : M01_(M01), KSplit_(KSplit), c_grid_desc_m_n_(c_grid_desc_m_n)
    {
    }

    __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const index_t grid_size = M0 * N0 * KSplit_;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        auto block_1d_id = idx_top[I0];

        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n_.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n_.GetLength(I1), NPerBlock);

        block_1d_id = block_1d_id % (M0 * N0 * KSplit_); // hide groups

        const index_t idx_ksplit = block_1d_id / (M0 * N0);
        block_1d_id              = block_1d_id % (M0 * N0);

        index_t idx_N0 = block_1d_id % N0;
        index_t idx_M0 = block_1d_id / N0;

        const auto M01_adapt = (idx_M0 < M0 - M0 % M01_) ? M01_ : M0 % M01_;

        index_t idx_M00          = idx_M0 / M01_;
        index_t idx_M01          = idx_M0 % M01_;
        index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

        return make_tuple(idx_ksplit,
                          idx_N0_M01_local % M01_adapt + idx_M00 * M01_,
                          idx_N0_M01_local / M01_adapt);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& /* c_tile_idx */,
                                             const CTileDim& /* c_tile_dim */) const
    {
        return true; // always valid provided that user gets grid size from CalculateGridSize()
    }

    __host__ constexpr bool CheckValidity(const CGridDesc_M_N& /* c_grid_desc_m_n */) const
    {
        return true;
    }

    private:
    index_t M01_;
    index_t KSplit_;
    CGridDesc_M_N c_grid_desc_m_n_;
};

// Blocks of row-vectors
template <index_t MPerBlock,
          index_t NPerBlock,
          typename CGridDesc_M_N,
          bool DeviceCTileIndexCheck = false>
struct BlockToCTileMap_M00_N00_M01_N01
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ __device__ BlockToCTileMap_M00_N00_M01_N01() = default;

    __host__ __device__ BlockToCTileMap_M00_N00_M01_N01(const CGridDesc_M_N& c_grid_desc_m_n,
                                                        index_t M01 = 1,
                                                        index_t N01 = 1)
        : M01_(M01), N01_(N01), underlying_map_(GetBlockToCTileMap(c_grid_desc_m_n, M01, N01))
    {
    }

    __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01_);
        const auto N00 = math::integer_divide_ceil(N0, N01_);

        const index_t grid_size = M00 * M01_ * N00 * N01_;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        return underlying_map_.CalculateBottomIndex(idx_top);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                             const CTileDim& c_tile_dim) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return DefaultValidCTileIndex(c_tile_idx, c_tile_dim);
        else
            return true;
    }

    __host__ constexpr bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return true; // validity check moved to kernel

        const index_t M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const index_t N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);
        if(M0 % M01_ == 0 && N0 % N01_ == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    private:
    __host__ __device__ static constexpr auto
    GetBlockToCTileMap(const CGridDesc_M_N& c_grid_desc_m_n, index_t M01, index_t N01)
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01);
        const auto N00 = math::integer_divide_ceil(N0, N01);

        const auto m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_insert_transform(1), // swallow the carry from lower dimensions
                           make_unmerge_transform(make_tuple(M00, M01)),
                           make_unmerge_transform(make_tuple(N00, N01))),
                make_tuple(Sequence<>{}, Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}));

        const auto cblockid_to_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(1, M00, N00, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto cblockid_to_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  cblockid_to_m00_m01_n00_n01_block_cluster_adaptor);

        return cblockid_to_m0_n0_block_cluster_adaptor;
    }

    index_t M01_, N01_;
    using UnderlyingMap = decltype(GetBlockToCTileMap(CGridDesc_M_N{}, 1, 1));
    UnderlyingMap underlying_map_;
};

// 2D slices of row-vectors in 3D space
template <index_t MPerBlock,
          index_t NPerBlock,
          typename CGridDesc_M_N,
          bool DeviceCTileIndexCheck = false>
struct BlockToCTileMap_KSplit_M00_N00_M01_N01
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    __host__ BlockToCTileMap_KSplit_M00_N00_M01_N01() = default;

    __host__ BlockToCTileMap_KSplit_M00_N00_M01_N01(const CGridDesc_M_N& c_grid_desc_m_n,
                                                    index_t M01    = 1,
                                                    index_t N01    = 1,
                                                    index_t KSplit = 1)
        : c_grid_desc_m_n_(c_grid_desc_m_n),
          M01_(M01),
          N01_(N01),
          KSplit_(KSplit),
          underlying_map_(GetBlockToCTileMap(c_grid_desc_m_n, M01, N01, KSplit))
    {
    }

    __host__ __device__ constexpr index_t
    CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01_);
        const auto N00 = math::integer_divide_ceil(N0, N01_);

        const index_t grid_size = M00 * M01_ * N00 * N01_ * KSplit_;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        static_assert(TopIdx::Size() == 1);

        return underlying_map_.CalculateBottomIndex(
            make_multi_index(idx_top[I0] % CalculateGridSize()));
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                             const CTileDim& c_tile_dim) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return DefaultValidCTileIndex(c_tile_idx, c_tile_dim);
        else
            return true;
    }

    __host__ constexpr bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        if constexpr(DeviceCTileIndexCheck)
            return true; // validity check moved to kernel

        const index_t M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const index_t N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);
        if(M0 % M01_ == 0 && N0 % N01_ == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    private:
    __device__ constexpr index_t CalculateGridSize() const
    {
        return CalculateGridSize(c_grid_desc_m_n_);
    }

    __host__ static constexpr auto GetBlockToCTileMap(const CGridDesc_M_N& c_grid_desc_m_n,
                                                      index_t M01,
                                                      index_t N01,
                                                      index_t KSplit)
    {
        const auto M0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I0), MPerBlock);
        const auto N0 = math::integer_divide_ceil(c_grid_desc_m_n.GetLength(I1), NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01);
        const auto N00 = math::integer_divide_ceil(N0, N01);

        const auto ksplit_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_pass_through_transform(KSplit),
                           make_unmerge_transform(make_tuple(M00, M01)),
                           make_unmerge_transform(make_tuple(N00, N01))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}));

        const auto c_blockid_to_ksplit_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(KSplit, M00, N00, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto c_blockid_to_ksplit_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(ksplit_m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  c_blockid_to_ksplit_m00_m01_n00_n01_block_cluster_adaptor);

        return c_blockid_to_ksplit_m0_n0_block_cluster_adaptor;
    }

    CGridDesc_M_N c_grid_desc_m_n_;
    index_t M01_, N01_, KSplit_;
    using UnderlyingMap = decltype(GetBlockToCTileMap(CGridDesc_M_N{}, 1, 1, 1));
    UnderlyingMap underlying_map_;
};

template <typename CTileIdx, typename CTileDim>
__host__ __device__ bool DefaultValidCTileIndex(const CTileIdx& c_tile_idx,
                                                const CTileDim& c_tile_dim)
{
    bool is_valid = false;

    const index_t m_block = c_tile_dim[Number<0>{}];
    const index_t n_block = c_tile_dim[Number<1>{}];

    if constexpr(CTileIdx::Size() == 2)
    {
        const index_t m_block_idx = c_tile_idx[Number<0>{}];
        const index_t n_block_idx = c_tile_idx[Number<1>{}];
        if(0 <= m_block_idx && m_block_idx < m_block && 0 <= n_block_idx && n_block_idx < n_block)
        {
            is_valid = true;
        }
    }
    else if constexpr(CTileIdx::Size() == 3)
    {
        const index_t ksplit_idx  = c_tile_idx[Number<0>{}];
        const index_t m_block_idx = c_tile_idx[Number<1>{}];
        const index_t n_block_idx = c_tile_idx[Number<2>{}];
        if(0 <= m_block_idx && m_block_idx < m_block && 0 <= n_block_idx && n_block_idx < n_block)
        {
            is_valid = true;
        }
        ignore = ksplit_idx;
    }

    return is_valid;
}

// This wrapper class is for grouped gemm where it subtracts blockIdx by a value so that the
// workgroups assigned to a given gemm problem have top index offsetted to range [0,
// grid_size_per_gemm]
template <typename UnderlyingBlockToCTileMap>
struct OffsettedBlockToCTileMap
{
    using underlying_type = UnderlyingBlockToCTileMap;

    __host__ __device__ OffsettedBlockToCTileMap() = default;
    __host__ __device__ OffsettedBlockToCTileMap(UnderlyingBlockToCTileMap block_to_ctile_map,
                                                 index_t block_start)
    {
        block_to_ctile_map_ = block_to_ctile_map;
        block_start_        = block_start;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        return block_to_ctile_map_.CalculateBottomIndex(
            make_multi_index(idx_top[Number<0>{}] - block_start_));
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                             const CTileDim& c_tile_dim) const
    {
        return block_to_ctile_map_.ValidCTileIndex(c_tile_idx, c_tile_dim);
    }

    template <typename CGridDesc_M_N>
    __host__ constexpr bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        return block_to_ctile_map_.CheckValidity(c_grid_desc_m_n);
    }

    template <typename CGridDesc_M_N>
    __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        return block_to_ctile_map_.CalculateGridSize(c_grid_desc_m_n);
    }

    __host__ __device__ constexpr index_t CalculateGridSize(index_t M, index_t N) const
    {
        return block_to_ctile_map_.CalculateGridSize(M, N);
    }

    UnderlyingBlockToCTileMap block_to_ctile_map_;
    index_t block_start_;
};
// second version with 2 offsets
template <typename UnderlyingBlockToCTileMap>
struct OffsettedBlockToCTileMap2
{
    using underlying_type = UnderlyingBlockToCTileMap;

    __host__ __device__ OffsettedBlockToCTileMap2(UnderlyingBlockToCTileMap block_to_ctile_map,
                                                  index_t group_offset,
                                                  index_t tile_offset)
        : block_to_ctile_map_{block_to_ctile_map},
          group_offset_{group_offset},
          tile_offset_{tile_offset}
    {
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        return block_to_ctile_map_.CalculateBottomIndex(
            make_multi_index(idx_top[Number<0>{}] + tile_offset_ - group_offset_));
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                             const CTileDim& c_tile_dim) const
    {
        return block_to_ctile_map_.ValidCTileIndex(c_tile_idx, c_tile_dim);
    }

    template <typename CGridDesc_M_N>
    __host__ constexpr bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        return block_to_ctile_map_.CheckValidity(c_grid_desc_m_n);
    }

    __host__ __device__ constexpr index_t CalculateGridSize(index_t M, index_t N) const
    {
        return block_to_ctile_map_.CalculateGridSize(M, N);
    }

    __device__ void UpdateTileOffset(index_t offset) { tile_offset_ = offset; }
    UnderlyingBlockToCTileMap block_to_ctile_map_;
    index_t group_offset_;
    index_t tile_offset_;
};

/**
 * @brief      Simple tile mapping which creates 3D grid of block of threads.
 *
 * @paragraph  Description
 *             This Block-to-C-tile-map creates a 3D grid (n_blocks, m_blocks, z_blocks) of thread
 *             blocks. The first 2D are regular 2D tiles created by division of output GEMM
 *             dimenions by corresponding tile size. The third dimension (Z) is a k-split dimension,
 *             which denotes the number of blocks we use to divide work on GEMM K dimension onto.
 *
 * @tparam     MPerBlock  Output block tile size in M dimension.
 * @tparam     NPerBlock  Output block tile size in N dimension.
 */
template <index_t MPerBlock, index_t NPerBlock>
struct BlockToCTileMap_3DGrid_KSplit
{

    __host__ __device__ BlockToCTileMap_3DGrid_KSplit() = default;

    __host__ __device__ constexpr auto
    CalculateGridSize(index_t M, index_t N, index_t k_split) const
    {
        // Create 3D grid
        const auto M0 = math::integer_divide_ceil(M, MPerBlock);
        const auto N0 = math::integer_divide_ceil(N, NPerBlock);
        return make_tuple(N0, M0, k_split);
    }

    template <typename TopIdx>
    __device__ constexpr auto CalculateBottomIndex(const TopIdx&) const
    {
        return make_tuple(blockIdx.z, blockIdx.y, blockIdx.x);
    }

    template <typename CTileIdx, typename CTileDim>
    __host__ __device__ bool ValidCTileIndex(const CTileIdx& /* c_tile_idx */,
                                             const CTileDim& /* c_tile_dim */) const
    {
        return true; // always valid provided that user gets grid size from CalculateGridSize()
    }

    template <typename CGridDesc_M_N>
    __host__ constexpr bool CheckValidity(const CGridDesc_M_N& /* c_grid_desc_m_n */) const
    {
        return true;
    }
};

enum StreamKReductionStrategy
{
    Atomic = 0, // sk block use atomic to do reduction
    Reduction,  // let some workgroup responsible for doing the reduction operation
};

template <uint32_t MPerBlock_,
          uint32_t NPerBlock_,
          uint32_t KPerBlock_,
          StreamKReductionStrategy ReductionStrategy_ = StreamKReductionStrategy::Atomic,
          uint32_t TileSwizzleSubM_                   = 8>
struct BlockToCTileMap_GemmStreamK
{
    static constexpr uint32_t min_k_iters_per_sk_block          = 2;
    static constexpr uint32_t MPerBlock                         = MPerBlock_;
    static constexpr uint32_t NPerBlock                         = NPerBlock_;
    static constexpr uint32_t KPerBlock                         = KPerBlock_;
    static constexpr StreamKReductionStrategy ReductionStrategy = ReductionStrategy_;
    static constexpr uint32_t tile_swizzle_sub_m                = TileSwizzleSubM_;

    //--------------------------------------
    // pass to device
    uint32_t sk_num_blocks;
    uint32_t sk_num_big_blocks;
    uint32_t dp_start_block_idx;
    uint32_t reduction_start_block_idx;
    uint32_t k_iters_per_big_block;
    MDiv2 n_tiles;
    MDiv k_iters_per_tile;
    MDiv eqav_tiles_big;    // for reduction
    MDiv eqav_tiles_little; // for reduction

    // MDiv tile_swizzle_sub_m_rem;
    //--------------------------------------

    // prefer construct on host
    BlockToCTileMap_GemmStreamK(uint32_t m,
                                uint32_t n,
                                uint32_t k,
                                uint32_t num_cu,
                                uint32_t occupancy,
                                uint32_t sk_blocks = 0xffffffff)
    {
        uint32_t num_tiles =
            math::integer_divide_ceil(m, MPerBlock) * math::integer_divide_ceil(n, NPerBlock);
        k_iters_per_tile = MDiv(math::integer_divide_ceil(k, KPerBlock));

        // one cu can hold one wg at one time, from the whole chip's point of view
        // if number of wg is same as num_cu, we call it 1 dispatch
        // if number of wg is 2x num_cu, we call it 2 dispatches.
        // one dispatch can deliver wg same as num_cu (full dispatch), or less than num_cu (partial
        // dispatch)
        //
        uint32_t full_dispatches         = num_tiles / num_cu;
        uint32_t full_dispatch_tiles     = full_dispatches * num_cu;
        uint32_t partial_dispatche_tiles = num_tiles - full_dispatch_tiles;

        uint32_t sk_occupancy = occupancy;
        uint32_t dp_tiles     = full_dispatch_tiles;
        uint32_t sk_tiles     = partial_dispatche_tiles;

        if(full_dispatches < occupancy)
        {
            // in this case, we allocate all blocks as sk blocks
            // sk_occupancy = occupancy - full_dispatches;
            sk_occupancy = 1; // TODO: single occ seems better
            dp_tiles     = full_dispatch_tiles;
            sk_tiles     = partial_dispatche_tiles;
        }
        else if((occupancy > 1) && (full_dispatches % occupancy == occupancy - 1))
        {
            // e.g. occupancy = 2, full_dispatches = 3, 5, 7 ...
            //      occupancy = 3, full_dispatches = 5, 8, 11 ...
            //      occupancy = 4, full_dispatches = 7, 11 ...
            sk_occupancy = 1; // left 1 slot for sk occupancy
            dp_tiles     = full_dispatch_tiles;
            sk_tiles     = partial_dispatche_tiles;
        }
        else
        {
            // others, we reduce 1 dispatch from dp, together with partial dispatch,
            // to construct sk dispatch
            sk_occupancy = occupancy - ((full_dispatches - 1) % occupancy);
            dp_tiles     = full_dispatch_tiles - num_cu;
            sk_tiles     = partial_dispatche_tiles + num_cu;
        }

        // uint32_t dp_iters_per_block = k_iters_per_tile.get();
        uint32_t sk_total_iters = k_iters_per_tile.get() * sk_tiles;
        uint32_t dp_num_blocks  = 0;

        {
            uint32_t min_sk_tiles = (sk_tiles >= num_cu) ? num_cu : (sk_tiles + 1);
            uint32_t max_sk_tiles =
                (sk_tiles >= num_cu) ? num_cu * sk_occupancy
                                     : math::min(num_cu, sk_total_iters / min_k_iters_per_sk_block);

            // if use dp for sk-block, how many iters do we need
            uint32_t dp_for_sk_iters = k_iters_per_tile.get();

            uint32_t best_sk_score =
                NumericLimits<int32_t>::Max(); // we need to find the smallest sk iters
            for(uint32_t tentative_sk_blocks = min_sk_tiles; tentative_sk_blocks < max_sk_tiles;
                tentative_sk_blocks++)
            {
                uint32_t tentative_sk_iters_per_block =
                    (sk_total_iters + tentative_sk_blocks - 1) / tentative_sk_blocks;
                uint32_t tentative_sk_iters = tentative_sk_iters_per_block;
                uint32_t sk_blocks_per_tile = (tentative_sk_blocks + sk_tiles - 1) / sk_tiles;

                // TODO: carefully adjust this parameter
                //       the more sk_blocks_per_tile, the worse the overhead
                uint32_t cross_sk_blocks_overhead = sk_blocks_per_tile;
                if(tentative_sk_blocks % sk_tiles != 0)
                {
                    // penalty for uneven divide
                    cross_sk_blocks_overhead +=
                        sk_blocks_per_tile * tentative_sk_iters_per_block / 50;
                }

                uint32_t tentative_sk_score = tentative_sk_iters + cross_sk_blocks_overhead;

                if(tentative_sk_score < best_sk_score)
                {
                    best_sk_score = tentative_sk_score;
                    sk_num_blocks = tentative_sk_blocks;
                }
            }

            if(best_sk_score >= dp_for_sk_iters)
            {
                sk_num_blocks = 0;
            }

            // give a chance to control num of sk blocks
            sk_num_blocks = sk_blocks != 0xffffffff ? sk_blocks : sk_num_blocks;

            if(sk_num_blocks == 0)
            {
                sk_num_big_blocks     = 0;
                k_iters_per_big_block = 0;

                dp_num_blocks      = num_tiles; // all tile to be dp block
                dp_start_block_idx = 0;
                sk_total_iters     = 0; // clear this tiles
            }
            else
            {
                // k_iters_per_sk_block is the floor of avg each ck block loop over tiles.
                // we need to decide how many iters for each sk block
                // let m = k_iters_per_sk_block
                // some of the sk block (little) will cover m iters, some (big) will cover m+1
                // we have
                // 1) l + b = sk_blocks
                // 2) l * m + b * (m + 1) = sk_total_iters
                //      => (l + b) * m + b = sk_total_iters
                //      => sk_blocks * m + b = sk_total_iters
                //      => b = sk_total_iters - m * sk_blocks
                //      NOTE: big could be zero
                uint32_t k_iters_per_sk_block = sk_total_iters / sk_num_blocks;
                sk_num_big_blocks     = sk_total_iters - k_iters_per_sk_block * sk_num_blocks;
                k_iters_per_big_block = k_iters_per_sk_block + 1;

                dp_num_blocks      = dp_tiles;
                dp_start_block_idx = (sk_num_blocks + num_cu - 1) / num_cu * num_cu;
            }
        }
        n_tiles                   = MDiv2(math::integer_divide_ceil(n, NPerBlock));
        reduction_start_block_idx = dp_start_block_idx + dp_num_blocks;

        if constexpr(ReductionStrategy == StreamKReductionStrategy::Reduction)
        {
            uint32_t upper_big    = math::lcm(k_iters_per_big_block, k_iters_per_tile.get());
            uint32_t upper_little = math::lcm(k_iters_per_big_block - 1, k_iters_per_tile.get());
            eqav_tiles_big        = MDiv(upper_big / k_iters_per_tile.get());
            eqav_tiles_little     = MDiv(upper_little / k_iters_per_tile.get());
        }

#if 0
        printf("cu:%d, occupancy:%d, grids:%d, num_tiles:%d, dp_tiles:%d, sk_num_big_blocks:%d, "
               "sk_num_blocks:%d, "
               "sk_total_iters:%d, dp_start_block_idx:%d, dp_iters_per_block:%d, dp_num_blocks:%d, "
               "k_iters_per_tile:%d, k_iters_per_big_block:%d, reduction_start_block_idx:%u, "
               "sk_tiles:%u, workspace(acc float):%u\n",
               num_cu,
               occupancy,
               get_grid_dims().x,
               num_tiles,
               dp_tiles,
               sk_num_big_blocks,
               sk_num_blocks,
               sk_total_iters,
               dp_start_block_idx,
               dp_iters_per_block,
               dp_num_blocks,
               k_iters_per_tile.get(),
               k_iters_per_big_block,
               reduction_start_block_idx,
               get_sk_tiles(),
               get_workspace_size(sizeof(float)));
#endif
    }

    __host__ __device__ uint32_t get_sk_total_iters() const
    {
        uint32_t sk_total_iters = sk_num_big_blocks * k_iters_per_big_block +
                                  (sk_num_blocks - sk_num_big_blocks) * (k_iters_per_big_block - 1);
        return sk_total_iters;
    }

    __host__ __device__ uint32_t get_sk_tiles() const
    {
        // tiles for sk
        uint32_t sk_total_iters = get_sk_total_iters();
        return k_iters_per_tile.div(sk_total_iters);
    }

    __host__ __device__ dim3 get_grid_dims() const
    {
        if constexpr(ReductionStrategy == StreamKReductionStrategy::Reduction)
        {
            return dim3(reduction_start_block_idx + get_sk_tiles(), 1, 1);
        }
        else
            return dim3(reduction_start_block_idx, 1, 1);
    }

    __device__ uint32_t get_block_idx() const
    {
        // TODO: swizzle block index for better locality
        return __builtin_amdgcn_readfirstlane(blockIdx.x);
    }

    __device__ void
    get_block_itr(uint32_t block_idx, uint32_t& iter_start, uint32_t& iter_end) const
    {
        if(block_idx < sk_num_big_blocks)
        {
            iter_start = block_idx * k_iters_per_big_block;
            iter_end   = iter_start + k_iters_per_big_block;
        }
        else if(block_idx < sk_num_blocks)
        {
            iter_start = (sk_num_big_blocks * k_iters_per_big_block) +
                         (block_idx - sk_num_big_blocks) * (k_iters_per_big_block - 1);
            iter_end = iter_start + (k_iters_per_big_block - 1);
        }
        else if(block_idx >= dp_start_block_idx)
        {
            uint32_t sk_total_iters     = get_sk_total_iters();
            uint32_t dp_iters_per_block = k_iters_per_tile.get();
            iter_start = sk_total_iters + (block_idx - dp_start_block_idx) * dp_iters_per_block;
            iter_end   = iter_start + dp_iters_per_block;
        }
    }

    __device__ uint32_t get_current_iter_length(uint32_t iter_start,
                                                uint32_t iter_end,
                                                uint32_t total_iter_length) const
    {
        uint32_t iter_length_mod, iter_length_quo /*unused*/;
        k_iters_per_tile.divmod(iter_end, iter_length_quo, iter_length_mod);
        uint32_t current_iter_length = math::min(
            iter_length_mod == 0 ? (iter_end - iter_start) : iter_length_mod, total_iter_length);
        return current_iter_length;
    }

    __device__ uint32_t get_tile_idx(uint32_t iter) const { return k_iters_per_tile.div(iter); }

    __device__ void
    get_tile_idx_with_offset(uint32_t iter, uint32_t& tile_idx, uint32_t& iter_offset) const
    {
        k_iters_per_tile.divmod(iter, tile_idx, iter_offset);
    }

    __device__ auto tile_to_spatial(uint32_t tile_idx, uint32_t m, uint32_t n) const
    {
        uint32_t m_tile_idx, n_tile_idx;
        uint32_t n_tiles_value = math::integer_divide_ceil(n, NPerBlock);
        n_tiles.divmod(tile_idx, n_tiles_value, m_tile_idx, n_tile_idx);

        // swizzle tile
        uint32_t m_tiles = math::integer_divide_ceil(m, MPerBlock);

        uint32_t tile_swizzle_sub_m_rem = m_tiles % tile_swizzle_sub_m;

        const auto sub_m_adapt = (m_tile_idx < (m_tiles - tile_swizzle_sub_m_rem))
                                     ? tile_swizzle_sub_m
                                     : tile_swizzle_sub_m_rem;

        uint32_t m_tile_idx_sub0, m_tile_idx_sub1;
        m_tile_idx_sub0 = m_tile_idx / tile_swizzle_sub_m;
        m_tile_idx_sub1 = m_tile_idx % tile_swizzle_sub_m;

        uint32_t tile_idx_local = n_tile_idx + m_tile_idx_sub1 * n_tiles_value;

        uint32_t m_tile_idx_with_adapt, n_tile_idx_with_adapt;

        n_tile_idx_with_adapt = tile_idx_local / sub_m_adapt;
        m_tile_idx_with_adapt = tile_idx_local % sub_m_adapt;
        return make_tuple(m_tile_idx_with_adapt + m_tile_idx_sub0 * tile_swizzle_sub_m,
                          n_tile_idx_with_adapt);
    }

    __host__ __device__ uint32_t get_workspace_size_for_acc(uint32_t acc_element_bytes) const
    {
        static constexpr uint32_t alignment = 128;
        uint32_t acc_buffer_bytes =
            MPerBlock * NPerBlock * get_total_acc_buffers() * acc_element_bytes;
        return (acc_buffer_bytes + alignment - 1) / alignment * alignment;
    }

    __host__ __device__ uint32_t get_workspace_size_for_semaphore() const
    {
        return get_sk_tiles() * sizeof(uint32_t);
    }

    __host__ __device__ uint32_t get_workspace_size(uint32_t acc_element_bytes) const
    {
        return get_workspace_size_for_acc(acc_element_bytes) + get_workspace_size_for_semaphore();
    }

    __host__ __device__ uint32_t get_tile_intersections(uint32_t tiles_,
                                                        const MDiv& eqav_tiles_) const
    {
        uint32_t tile_idx_       = tiles_ == 0 ? 0 : (tiles_ - 1);
        uint32_t max_eqav_tiles_ = eqav_tiles_.get() - 1;
        uint32_t quo_, rem_;
        eqav_tiles_.divmod(tile_idx_, quo_, rem_);
        return quo_ * max_eqav_tiles_ + rem_;
    }

    __host__ __device__ uint32_t get_tiles_cover_sk_block(uint32_t num_sk_blocks_,
                                                          uint32_t iters_per_sk_block_) const
    {
        return k_iters_per_tile.div(num_sk_blocks_ * iters_per_sk_block_ + k_iters_per_tile.get() -
                                    1);
    }

    __host__ __device__ uint32_t get_total_acc_buffers() const
    {
        uint32_t tiles_cover_big_blocks =
            get_tiles_cover_sk_block(sk_num_big_blocks, k_iters_per_big_block);
        uint32_t tiles_cover_little_blocks =
            get_tiles_cover_sk_block(sk_num_blocks - sk_num_big_blocks, k_iters_per_big_block - 1);

        uint32_t total_intersec_big =
            get_tile_intersections(tiles_cover_big_blocks, eqav_tiles_big);
        uint32_t total_intersec_little =
            get_tile_intersections(tiles_cover_little_blocks, eqav_tiles_little);

        return sk_num_blocks + total_intersec_big + total_intersec_little;
    }

    __device__ uint32_t get_acc_buffer_offset_from_tile(uint32_t tile_idx_) const
    {
        // TODO: from big to little
        uint32_t tiles_cover_big_blocks =
            get_tiles_cover_sk_block(sk_num_big_blocks, k_iters_per_big_block);
        if(tile_idx_ < tiles_cover_big_blocks)
        {
            uint32_t touched_sk_blocks =
                (tile_idx_ * k_iters_per_tile.get() + k_iters_per_big_block - 1) /
                k_iters_per_big_block;
            uint32_t current_intersec = get_tile_intersections(tile_idx_, eqav_tiles_big);
            return touched_sk_blocks + current_intersec;
        }
        else
        {
            uint32_t iters_per_little_sk_block = k_iters_per_big_block - 1;
            uint32_t tile_idx_little_reverse   = get_sk_tiles() - tile_idx_;
            uint32_t touched_sk_blocks =
                (tile_idx_little_reverse * k_iters_per_tile.get() + iters_per_little_sk_block - 1) /
                iters_per_little_sk_block;
            uint32_t current_intersec =
                get_tile_intersections(tile_idx_little_reverse, eqav_tiles_little);
            return get_total_acc_buffers() - (touched_sk_blocks + current_intersec);
        }
    }

    __device__ uint32_t get_acc_buffer_offset_from_block(uint32_t block_idx_) const
    {
        uint32_t iters_per_big_sk_block    = k_iters_per_big_block;
        uint32_t iters_per_little_sk_block = k_iters_per_big_block - 1;
        if(block_idx_ < sk_num_big_blocks)
        {
            uint32_t touched_tiles    = k_iters_per_tile.div(block_idx_ * iters_per_big_sk_block +
                                                          k_iters_per_tile.get() - 1);
            uint32_t current_intersec = get_tile_intersections(touched_tiles, eqav_tiles_big);
            return block_idx_ + current_intersec;
        }
        else
        {
            uint32_t block_idx_little_reverse = sk_num_blocks - block_idx_;
            uint32_t touched_tiles            = k_iters_per_tile.div(
                block_idx_little_reverse * iters_per_little_sk_block + k_iters_per_tile.get() - 1);
            uint32_t current_intersec = get_tile_intersections(touched_tiles, eqav_tiles_little);
            return get_total_acc_buffers() - (block_idx_little_reverse + current_intersec);
        }
    }
};

template <uint32_t MPerBlock_,
          uint32_t NPerBlock_,
          uint32_t KPerBlock_,
          StreamKReductionStrategy ReductionStrategy_ = StreamKReductionStrategy::Atomic,
          uint32_t TileSwizzleSubM_                   = 8,
          index_t GroupNum                            = 8,
          index_t M01_                                = 4>
struct BlockToCTileMap_GemmStreamK_v2
{
    static constexpr uint32_t min_k_iters_per_sk_block = 2;
    static constexpr uint32_t MPerBlock                = MPerBlock_;
    static constexpr uint32_t NPerBlock                = NPerBlock_;
    static constexpr uint32_t KPerBlock                = KPerBlock_;
    static constexpr uint32_t tile_swizzle_sub_m       = TileSwizzleSubM_;

    //--------------------------------------
    // pass to device
    mutable uint32_t sk_num_blocks;
    uint32_t sk_num_big_blocks;
    uint32_t dp_start_block_idx;
    uint32_t reduction_start_block_idx;
    uint32_t k_iters_per_big_block;
    MDiv2 n_tiles;
    MDiv k_iters_per_tile;
    MDiv equiv_tiles_big;    // for reduction
    MDiv equiv_tiles_little; // for reduction
    StreamKReductionStrategy reduction_strategy;

    // prefer construct on host
    __host__ __device__ BlockToCTileMap_GemmStreamK_v2(
        uint32_t m,
        uint32_t n,
        uint32_t k,
        uint32_t grid_size                           = 1,
        uint32_t streamk_sel                         = 1,
        StreamKReductionStrategy reduction_strategy_ = StreamKReductionStrategy::Atomic)
        : reduction_strategy(reduction_strategy_)
    {

        // total output tiles
        uint32_t num_tiles =
            math::integer_divide_ceil(m, MPerBlock) * math::integer_divide_ceil(n, NPerBlock);
        k_iters_per_tile = MDiv(math::integer_divide_ceil(k, KPerBlock));

        uint32_t dp_tiles, dp_num_blocks, sk_total_iters;

        // Ensure grid_size is at least 1 to avoid division by zero
        grid_size = math::max(grid_size, 1u);

        // default to regular DP GEMM if sk blocks == 0
        if(streamk_sel == 0)
        {
            sk_num_blocks         = 0;
            dp_tiles              = num_tiles;
            sk_num_big_blocks     = 0;
            k_iters_per_big_block = 0;

            dp_num_blocks      = num_tiles; // all tile to be dp block
            dp_start_block_idx = 0;
            sk_total_iters     = 0; // clear this tiles
        }
        // 2-tile sk + DP GEMM
        else
        {
            // check if there's enough work for DP+ stream-k
            bool bigEnough = num_tiles > grid_size;

            // Select between stream-k strategies
            // Add safety checks to prevent zero or negative values
            uint32_t sk_tiles = 0;
            if(streamk_sel == 1) // 1 tile stream-k
            {
                sk_tiles = bigEnough ? (num_tiles % grid_size) : num_tiles;

                // Ensure sk_tiles is at least 1
                sk_tiles = math::max(sk_tiles, 1u);
            }
            else if(streamk_sel == 2) // 2-tile stream-k
            {
                sk_tiles = bigEnough ? (grid_size + num_tiles % grid_size) : num_tiles;

                // Ensure sk_tiles is at least 1 but not more than num_tiles
                sk_tiles = math::min(math::max(sk_tiles, 1u), num_tiles);
            }
            else if(streamk_sel == 3) // 3-tile stream-k
            {
                sk_tiles = (num_tiles > (2 * grid_size)) ? (2 * grid_size + num_tiles % grid_size)
                                                         : num_tiles;

                // Ensure sk_tiles is at least 1 but not more than num_tiles
                sk_tiles = math::min(math::max(sk_tiles, 1u), num_tiles);
            }
            else if(streamk_sel == 4) // 4-tile stream-k
            {
                sk_tiles = (num_tiles > (3 * grid_size)) ? (3 * grid_size + num_tiles % grid_size)
                                                         : num_tiles;

                // Ensure sk_tiles is at least 1 but not more than num_tiles
                sk_tiles = math::min(math::max(sk_tiles, 1u), num_tiles);
            }

            sk_num_blocks = sk_tiles;
            // Remaining tiles are DP tiles
            dp_tiles = bigEnough ? (num_tiles - sk_tiles) : 0;

            sk_total_iters = k_iters_per_tile.get() * sk_tiles;

            // k_iters_per_sk_block is the floor of avg each ck block loop over tiles.
            // we need to decide how many iters for each sk block
            // let m = k_iters_per_sk_block
            // some of the sk block (little) will cover m iters, some (big) will cover m+1
            // we have
            // 1) l + b = sk_blocks
            // 2) l * m + b * (m + 1) = sk_total_iters
            //      => (l + b) * m + b = sk_total_iters
            //      => sk_blocks * m + b = sk_total_iters
            //      => b = sk_total_iters - m * sk_blocks
            //      NOTE: big could be zero

            // Add safety check for sk_num_blocks to prevent division by zero
            if(sk_num_blocks > 0)
            {
                uint32_t k_iters_per_sk_block = sk_total_iters / sk_num_blocks;
                sk_num_big_blocks     = sk_total_iters - k_iters_per_sk_block * sk_num_blocks;
                k_iters_per_big_block = k_iters_per_sk_block + 1;
            }
            else
            {
                // Fallback to default GEMM if no stream-k blocks
                sk_num_blocks         = 0;
                sk_num_big_blocks     = 0;
                k_iters_per_big_block = 0;
                dp_tiles              = num_tiles;
                dp_num_blocks         = num_tiles;
                dp_start_block_idx    = 0;
                sk_total_iters        = 0;
            }

            dp_num_blocks      = dp_tiles;
            dp_start_block_idx = sk_num_blocks;
        }

        n_tiles = MDiv2(math::integer_divide_ceil(n, NPerBlock));
        // Using multiple blocks for parallel reduction
        reduction_start_block_idx = dp_start_block_idx + dp_num_blocks;

        if(reduction_strategy == ck::StreamKReductionStrategy::Reduction)
        {
            // Add additional safety checks
            if(k_iters_per_big_block > 0 && k_iters_per_tile.get() > 0)
            {
                uint32_t upper_big = math::lcm(k_iters_per_big_block, k_iters_per_tile.get());
                uint32_t upper_little =
                    math::lcm(math::max(k_iters_per_big_block - 1, 1u), k_iters_per_tile.get());
                equiv_tiles_big    = MDiv(upper_big / k_iters_per_tile.get());
                equiv_tiles_little = MDiv(upper_little / k_iters_per_tile.get());
            }
            else
            {
                // Default safe values
                equiv_tiles_big    = MDiv(1);
                equiv_tiles_little = MDiv(1);
            }
        }
    }

    __host__ __device__ static constexpr index_t CalculateGridSize(index_t M, index_t N)
    {
        const auto M0 = math::integer_divide_ceil(M, MPerBlock);
        const auto N0 = math::integer_divide_ceil(N, NPerBlock);

        return M0 * N0;
    }
    __host__ __device__ uint32_t get_sk_total_iters() const
    {
        uint32_t sk_total_iters = sk_num_big_blocks * k_iters_per_big_block +
                                  (sk_num_blocks - sk_num_big_blocks) * (k_iters_per_big_block - 1);
        return sk_total_iters;
    }

    __host__ __device__ uint32_t get_sk_tiles() const
    {
        // tiles for sk
        uint32_t sk_total_iters = get_sk_total_iters();
        return k_iters_per_tile.div(sk_total_iters);
    }

    __host__ __device__ index_t get_grid_dims() const
    {
        if(reduction_strategy == StreamKReductionStrategy::Reduction)
        {
            // return dim3(reduction_start_block_idx + get_sk_tiles(), 1, 1);
            return reduction_start_block_idx + get_sk_tiles();
        }
        else
            return reduction_start_block_idx;
    }

    __device__ uint32_t get_block_idx() const
    {
        // TODO: swizzle block index for better locality
        return __builtin_amdgcn_readfirstlane(blockIdx.x);
    }

    __device__ void
    get_block_itr(uint32_t block_idx, uint32_t& iter_start, uint32_t& iter_end) const
    {
        if(block_idx < sk_num_big_blocks)
        {
            iter_start = block_idx * k_iters_per_big_block;
            iter_end   = iter_start + k_iters_per_big_block;
        }
        else if(block_idx < sk_num_blocks)
        {
            iter_start = (sk_num_big_blocks * k_iters_per_big_block) +
                         (block_idx - sk_num_big_blocks) * (k_iters_per_big_block - 1);
            iter_end = iter_start + (k_iters_per_big_block - 1);
        }
        else if(block_idx >= dp_start_block_idx)
        {
            uint32_t sk_total_iters     = get_sk_total_iters();
            uint32_t dp_iters_per_block = k_iters_per_tile.get();
            iter_start = sk_total_iters + (block_idx - dp_start_block_idx) * dp_iters_per_block;
            iter_end   = iter_start + dp_iters_per_block;
        }
    }

    __device__ uint32_t get_current_iter_length(uint32_t iter_start,
                                                uint32_t iter_end,
                                                uint32_t total_iter_length) const
    {
        uint32_t iter_length_mod, iter_length_quo /*unused*/;
        k_iters_per_tile.divmod(iter_end, iter_length_quo, iter_length_mod);
        uint32_t current_iter_length = math::min(
            iter_length_mod == 0 ? (iter_end - iter_start) : iter_length_mod, total_iter_length);
        return current_iter_length;
    }

    __device__ uint32_t get_tile_idx(uint32_t iter) const { return k_iters_per_tile.div(iter); }

    __device__ void
    get_tile_idx_with_offset(uint32_t iter, uint32_t& tile_idx, uint32_t& iter_offset) const
    {
        k_iters_per_tile.divmod(iter, tile_idx, iter_offset);
    }

    __device__ auto tile_to_spatial(uint32_t tile_idx, uint32_t m, uint32_t n) const
    {
        uint32_t m_tile_idx, n_tile_idx;
        uint32_t n_tiles_value = math::integer_divide_ceil(n, NPerBlock);
        n_tiles.divmod(tile_idx, n_tiles_value, m_tile_idx, n_tile_idx);

        // // swizzle tile
        uint32_t m_tiles = math::integer_divide_ceil(m, MPerBlock);

        uint32_t tile_swizzle_sub_m_rem = m_tiles % tile_swizzle_sub_m;

        const auto sub_m_adapt = (m_tile_idx < (m_tiles - tile_swizzle_sub_m_rem))
                                     ? tile_swizzle_sub_m
                                     : tile_swizzle_sub_m_rem;

        uint32_t m_tile_idx_sub0, m_tile_idx_sub1;
        m_tile_idx_sub0 = m_tile_idx / tile_swizzle_sub_m;
        m_tile_idx_sub1 = m_tile_idx % tile_swizzle_sub_m;

        uint32_t tile_idx_local = n_tile_idx + m_tile_idx_sub1 * n_tiles_value;

        uint32_t m_tile_idx_with_adapt, n_tile_idx_with_adapt;

        n_tile_idx_with_adapt = tile_idx_local / sub_m_adapt;
        m_tile_idx_with_adapt = tile_idx_local % sub_m_adapt;
        return make_tuple(m_tile_idx_with_adapt + m_tile_idx_sub0 * tile_swizzle_sub_m,
                          n_tile_idx_with_adapt);
    }

    __host__ __device__ uint32_t get_workspace_size_for_acc(uint32_t acc_element_bytes) const
    {
        static constexpr uint32_t alignment = 128;
        uint32_t acc_buffer_bytes =
            MPerBlock * NPerBlock * get_total_acc_buffers() * acc_element_bytes;
        return (acc_buffer_bytes + alignment - 1) / alignment * alignment;
    }

    __host__ __device__ uint32_t get_workspace_size_for_semaphore() const
    {
        return get_sk_tiles() * sizeof(uint32_t);
    }

    __host__ __device__ uint32_t get_workspace_size(uint32_t acc_element_bytes) const
    {
        return get_workspace_size_for_acc(acc_element_bytes) + get_workspace_size_for_semaphore();
    }

    __host__ __device__ uint32_t get_tile_intersections(uint32_t tiles_,
                                                        const MDiv& equiv_tiles_) const
    {
        uint32_t tile_idx_        = tiles_ == 0 ? 0 : (tiles_ - 1);
        uint32_t max_equiv_tiles_ = equiv_tiles_.get() - 1;
        uint32_t quo_, rem_;
        equiv_tiles_.divmod(tile_idx_, quo_, rem_);
        return quo_ * max_equiv_tiles_ + rem_;
    }

    __host__ __device__ uint32_t get_tiles_cover_sk_block(uint32_t num_sk_blocks_,
                                                          uint32_t iters_per_sk_block_) const
    {
        return k_iters_per_tile.div(num_sk_blocks_ * iters_per_sk_block_ + k_iters_per_tile.get() -
                                    1);
    }

    __host__ __device__ uint32_t get_total_acc_buffers() const
    {
        uint32_t tiles_cover_big_blocks =
            get_tiles_cover_sk_block(sk_num_big_blocks, k_iters_per_big_block);
        uint32_t tiles_cover_little_blocks =
            get_tiles_cover_sk_block(sk_num_blocks - sk_num_big_blocks, k_iters_per_big_block - 1);

        uint32_t total_intersec_big =
            get_tile_intersections(tiles_cover_big_blocks, equiv_tiles_big);
        uint32_t total_intersec_little =
            get_tile_intersections(tiles_cover_little_blocks, equiv_tiles_little);

        return sk_num_blocks + total_intersec_big + total_intersec_little;
    }

    __device__ uint32_t get_acc_buffer_offset_from_tile(uint32_t tile_idx_) const
    {
        // TODO: from big to little
        uint32_t tiles_cover_big_blocks =
            get_tiles_cover_sk_block(sk_num_big_blocks, k_iters_per_big_block);
        if(tile_idx_ < tiles_cover_big_blocks)
        {
            uint32_t touched_sk_blocks =
                (tile_idx_ * k_iters_per_tile.get() + k_iters_per_big_block - 1) /
                k_iters_per_big_block;
            uint32_t current_intersec = get_tile_intersections(tile_idx_, equiv_tiles_big);
            return touched_sk_blocks + current_intersec;
        }
        else
        {
            uint32_t iters_per_little_sk_block = k_iters_per_big_block - 1;
            uint32_t tile_idx_little_reverse   = get_sk_tiles() - tile_idx_;
            uint32_t touched_sk_blocks =
                (tile_idx_little_reverse * k_iters_per_tile.get() + iters_per_little_sk_block - 1) /
                iters_per_little_sk_block;
            uint32_t current_intersec =
                get_tile_intersections(tile_idx_little_reverse, equiv_tiles_little);
            return get_total_acc_buffers() - (touched_sk_blocks + current_intersec);
        }
    }

    __device__ uint32_t get_acc_buffer_offset_from_block(uint32_t block_idx_) const
    {
        uint32_t iters_per_big_sk_block    = k_iters_per_big_block;
        uint32_t iters_per_little_sk_block = k_iters_per_big_block - 1;
        if(block_idx_ < sk_num_big_blocks)
        {
            uint32_t touched_tiles    = k_iters_per_tile.div(block_idx_ * iters_per_big_sk_block +
                                                          k_iters_per_tile.get() - 1);
            uint32_t current_intersec = get_tile_intersections(touched_tiles, equiv_tiles_big);
            return block_idx_ + current_intersec;
        }
        else
        {
            uint32_t block_idx_little_reverse = sk_num_blocks - block_idx_;
            uint32_t touched_tiles            = k_iters_per_tile.div(
                block_idx_little_reverse * iters_per_little_sk_block + k_iters_per_tile.get() - 1);
            uint32_t current_intersec = get_tile_intersections(touched_tiles, equiv_tiles_little);
            return get_total_acc_buffers() - (block_idx_little_reverse + current_intersec);
        }
    }
};

} // namespace ck
