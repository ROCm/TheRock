// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

namespace ck_tile {
template <typename Problem_, typename Policy_ = PracticeGemmHostPolicy>
struct PracticeGemmHostPipeline
{
    using ADataType   = typename Problem_::ADataType;
    using BDataType   = typename Problem_::BDataType;
    using CDataType   = typename Problem_::CDataType;
    using AccDataType = typename Problem_::AccDataType;

    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using BlockTile = typename Problem::Shape::BlockTile;
    using WaveTile  = typename Problem::Shape::WaveTile;

    template <typename ADRAMTensorView, typename BDRAMTensorView, typename CDRAMTensorView>
    CK_TILE_DEVICE void operator()(const ADRAMTensorView& a_dram,
                                   const BDRAMTensorView& b_dram,
                                   CDRAMTensorView& c_dram) const
    {

        // Size of the entire problem
        const auto M = a_dram.get_tensor_descriptor().get_length(number<0>{}); // M x K
        const auto N = c_dram.get_tensor_descriptor().get_length(number<1>{}); // M x N
        const auto K = a_dram.get_tensor_descriptor().get_length(number<1>{}); // M x K

        // Size of the block tile
        const auto MPerBlock = BlockTile::at(number<0>{});
        const auto NPerBlock = BlockTile::at(number<1>{});
        const auto KPerBlock = BlockTile::at(number<2>{});

        // Number of block tile in the N direction to cover C (resultant) matrix
        const auto num_tile_n = integer_divide_ceil(N, NPerBlock);
        // Number of block tile in the M direction to cover C (resultant) matrix
        const auto num_tile_m = integer_divide_ceil(M, MPerBlock);

        // if(get_thread_id() == 0 && get_block_id() == 0)
        // {
        //     printf("num_tile_m: %d, num_tile_n: %d\n", num_tile_m, num_tile_n);
        //     printf("total number of tiles: %d\n", num_tile_m * num_tile_n);
        // }

        // Get block id
        const auto id_block =
            get_block_id(); // 0 to (M_block/BlockTile_M) * (N_block/BlockTile_N) - 1

        // Map block id to tile id
        const auto block2tile = Policy::MakeBlock2TileMap(num_tile_m, num_tile_n);

        const auto tile_id = block2tile(id_block);

        const auto tile_id_m = tile_id.at(number<0>{});
        const auto tile_id_n = tile_id.at(number<1>{});

        // if(get_thread_id() == 0 && get_block_id() == 15)
        // {
        //     printf("tile_id_m: %d, tile_id_n: %d\n", tile_id_m, tile_id_n);
        // }

        const auto tile_origin_m = tile_id_m * MPerBlock;
        const auto tile_origin_n = tile_id_n * NPerBlock;

        // create a tile window over dram for A and B
        const auto a_block_window = make_tile_window(
            a_dram, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {tile_origin_m, 0});

        const auto b_block_window = make_tile_window(
            b_dram, make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), {tile_origin_n, 0});

        constexpr auto block_gemm_pipeline =
            Policy::template GetPracticeGemmBlockPipeline<Problem>();

        int num_loops_k = integer_divide_ceil(K, KPerBlock);

        __shared__ char p_smem_char[block_gemm_pipeline.GetStaticLDSSize()];
        const auto c_block_tile =
            block_gemm_pipeline(a_block_window, b_block_window, num_loops_k, p_smem_char);
        auto c_window = make_tile_window(c_dram,
                                         make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                                         {tile_origin_m, tile_origin_n});
        store_tile(c_window, c_block_tile);
    }
};
} // namespace ck_tile
