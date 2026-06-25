// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include "ck_tile/core.hpp"
#include "host_level/practice_gemm_host_policy_agmem_bgmem_creg.hpp"
#include "host_level/practice_gemm_host_pipeline_agmem_bgmem_creg.hpp"

namespace ck_tile {

template <typename BlockTile_, typename WaveTile_>
struct PracticeGemmShape
{
    using BlockTile = remove_cvref_t<BlockTile_>;
    using WaveTile  = remove_cvref_t<WaveTile_>;

    static constexpr index_t BlockTile_M = BlockTile::at(number<0>{});
    static constexpr index_t BlockTile_N = BlockTile::at(number<1>{});
    static constexpr index_t BlockTile_K = BlockTile::at(number<2>{});

    static constexpr index_t WaveTile_M = WaveTile::at(number<0>{});
    static constexpr index_t WaveTile_N = WaveTile::at(number<1>{});
    static constexpr index_t WaveTile_K = WaveTile::at(number<2>{});

    // Thread block configuration
    static constexpr index_t kWarpSize  = 64;  // AMD GPU warp size (also called wavefront)
    static constexpr index_t kBlockSize = 256; // Total threads per block (4 warps × 64 threads)

    CK_TILE_HOST static std::string GetName()
    {
        // clang-format off
        return concat('_', "practice_gemm_shape",
                      concat('x', BlockTile_M, BlockTile_N, BlockTile_K),
                      concat('x', WaveTile_M, WaveTile_N, WaveTile_K));
        // clang-format on
    }
};

template <typename Problem_, typename Policy_>
struct PracticeGemmKernel
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    // Derive block size from the shape configuration
    static constexpr index_t kBlockSize = Problem::Shape::kBlockSize;

    CK_TILE_DEVICE void operator()(const typename Problem::ADataType* p_a,
                                   const typename Problem::BDataType* p_b,
                                   typename Problem::CDataType* p_c,
                                   const index_t M,
                                   const index_t N,
                                   const index_t K,
                                   const index_t stride_a,
                                   const index_t stride_b,
                                   const index_t stride_c) const
    {

        auto a_dram = make_naive_tensor_view<address_space_enum::global>(
            p_a, make_tuple(M, K), make_tuple(stride_a, 1), number<8>{}, number<1>{});

        auto b_dram = make_naive_tensor_view<address_space_enum::global>(
            p_b, make_tuple(N, K), make_tuple(stride_b, 1), number<8>{}, number<1>{});

        const auto c_dram = make_naive_tensor_view<address_space_enum::global>(
            p_c, make_tuple(M, N), make_tuple(stride_c, 1), number<8>{}, number<1>{});

        PracticeGemmHostPipeline<Problem, Policy>{}(a_dram, b_dram, c_dram);
    }
};

} // namespace ck_tile
