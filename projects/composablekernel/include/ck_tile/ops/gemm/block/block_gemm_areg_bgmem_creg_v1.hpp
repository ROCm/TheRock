// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bgmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"

namespace ck_tile {

// A is block distributed tensor
// B is block window on global memory
// C is block distributed tensor
// This will:
//   1. load B from global memory into shared memory and then
//   2. Call BlockGemmARegSGmemCRegV1
template <typename Problem_, typename Policy_ = BlockGemmARegBGmemCRegV1DefaultPolicy>
struct BlockGemmARegBGmemCRegV1
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // use BlockGemmARegBSmemCRegV1 as the underlying block-GEMM implementation
    using BlockGemmARegBGmemCRegImpl = BlockGemmARegBGmemCRegV1<
        BlockGemmProblem<ADataType, BDataType, CDataType, kBlockSize, BlockGemmShape>,
        BlockGemmARegBGmemCRegV1DefaultPolicy>;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return sizeof(BDataType) *
               Policy::template MakeBSmemBlockDescriptor<Problem>().get_element_space_size();
    }

    // C += A * B
    template <typename CBlockTensor, typename ABlockTensor, typename BBlockGmemWindowTmp>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensor& a_block_tensor,
                                   const BBlockGmemWindowTmp& b_block_gmem_window_tmp,
                                   void* smem_ptr) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cv_t<typename ABlockTensor::DataType>> &&
                std::is_same_v<BDataType, remove_cv_t<typename BBlockGmemWindowTmp::DataType>> &&
                std::is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
            "wrong!");

        constexpr index_t MPerBlock = ABlockTensor{}.get_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockGmemWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockTensor{}.get_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        const auto b_block_gmem_window =
            make_tile_window(b_block_gmem_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                             b_block_gmem_window_tmp.get_window_origin(),
                             Policy::template MakeBGmemTileDistribution<Problem>());

        // B LDS and LDS window
        auto b_block_smem = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<BDataType*>(smem_ptr),
            Policy::template MakeBSmemBlockDescriptor<Problem>());

        auto b_block_smem_window = make_tile_window(
            b_block_smem, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});

        // load B tile from global mem
        const auto b_block_tile = load_tile(b_block_gmem_window);

        // store B tile into shared mem
        store_tile(b_block_smem_window, b_block_tile);

        // wait for store_tile to finish
        block_sync_lds();

        // block GEMM
        BlockGemmARegBGmemCRegImpl{}(c_block_tensor, a_block_tensor, b_block_smem_window);
    }

    // C = A * B
    template <typename ABlockTensor, typename BBlockGmemWindowTmp>
    CK_TILE_DEVICE auto operator()(const ABlockTensor& a_block_tensor,
                                   const BBlockGmemWindowTmp& b_block_gmem_window_tmp,
                                   void* smem_ptr) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cv_t<typename ABlockTensor::DataType>> &&
                std::is_same_v<BDataType, remove_cv_t<typename BBlockGmemWindowTmp::DataType>>,
            "wrong!");

        constexpr index_t MPerBlock = ABlockTensor{}.get_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockGmemWindowTmp{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockTensor{}.get_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        const auto b_block_gmem_window =
            make_tile_window(b_block_gmem_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                             b_block_gmem_window_tmp.get_window_origin(),
                             Policy::template MakeBGmemTileDistribution<Problem>());

        // B LDS and LDS window
        auto b_block_smem = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<BDataType*>(smem_ptr),
            Policy::template MakeBSmemBlockDescriptor<Problem>());

        auto b_block_smem_window = make_tile_window(
            b_block_smem, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});

        // load B tile from global mem
        const auto b_block_tile = load_tile(b_block_gmem_window);

        // store B tile into shared mem
        store_tile(b_block_smem_window, b_block_tile);

        // wait for store_tile to finish
        block_sync_lds();

        // block GEMM
        return BlockGemmARegBGmemCRegImpl{}(a_block_tensor, b_block_smem_window);
    }
};

} // namespace ck_tile
