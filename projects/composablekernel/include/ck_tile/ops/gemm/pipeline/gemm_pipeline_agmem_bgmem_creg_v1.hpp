// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename Problem>
struct BaseGemmPipelineAGmemBGmemCRegV1
{
    static constexpr index_t PrefetchStages   = 1;
    static constexpr index_t PrefillStages    = 1;
    static constexpr index_t GlobalBufferNum  = 1;
    static constexpr bool UsePersistentKernel = false;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t) { return true; }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t)
    {
        return TailNumber::Empty;
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto TailHandler(const RunFunction& run_func, bool, TailNumber)
    {
        return run_func(bool_constant<true>{}, integral_constant<TailNumber, TailNumber::Empty>{});
    }
};

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem, typename Policy = UniversalGemmPipelineAgBgCrPolicy>
struct GemmPipelineAGmemBGmemCRegV1 : public BaseGemmPipelineAGmemBGmemCRegV1<Problem>
{
    using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType  = remove_cvref_t<typename Problem::CDataType>;

    using AElementWise   = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise   = remove_cvref_t<typename Problem::BElementWise>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using AsLayout = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    using I0 = number<0>;
    using I1 = number<1>;
    using I2 = number<2>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool Preshuffle = Problem::Preshuffle;

    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

    static constexpr index_t kLdsAlignmentInBytes = 16;

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "BASIC_V1";
        // clang-format on
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AGmemBGmemCRegV1", 
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock,  BlockSize),
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

    // For the basic gemm pipelien DoubleSmemBuffer set to be false naturally.
    static constexpr bool DoubleSmemBuffer = false;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_HOST_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        using ADramBlockWindowTmp =
            remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
        using BDramBlockWindowTmp =
            remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;

        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "wrong!");

        constexpr bool is_a_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        static_assert(is_a_col_major
                          ? (kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                             kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                          : (kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                             kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                      "A block window has incorrect lengths for defined ALayout!");
        static_assert(is_b_row_major
                          ? (kKPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                             kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                          : (kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                             kKPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                      "B block window has incorrect lengths for defined BLayout!");
        // A tile in LDS
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        constexpr index_t a_lds_block_space_size_aligned =
            integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(),
                                kLdsAlignmentInBytes) *
            kLdsAlignmentInBytes;

        // B tile in LDS
        BDataType* p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));

        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();

        auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

        // A DRAM tile window for load
        auto as_copy_dram_window = generate_tuple(
            [&](auto idx) {
                return make_tile_window(
                    a_dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                    make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                    a_dram_block_window_tmp[number<idx>{}].get_window_origin(),
                    Policy::template MakeADramTileDistribution<Problem>());
            },
            number<AsLayout::size()>{});

        // A LDS tile window for store
        auto a_copy_lds_window = make_tile_window(
            a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // B DRAM tile window for load
        auto bs_copy_dram_window = generate_tuple(
            [&](auto idx) {
                return make_tile_window(
                    b_dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                    make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                    b_dram_block_window_tmp[number<idx>{}].get_window_origin(),
                    Policy::template MakeBDramTileDistribution<Problem>());
            },
            number<BsLayout::size()>{});

        // B LDS tile window for store
        auto b_copy_lds_window = make_tile_window(
            b_lds_block, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // Tile distribution for load from lds
        constexpr auto a_lds_load_tile_distr =
            make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
        constexpr auto b_lds_load_tile_distr =
            make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

        // A LDS tile for block GEMM
        auto a_lds_gemm_window =
            make_tile_window(a_lds_block,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             a_lds_load_tile_distr);

        // B LDS tile for block GEMM
        auto b_lds_gemm_window =
            make_tile_window(b_lds_block,
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             b_lds_load_tile_distr);

        // Block GEMM
        auto block_gemm = BlockGemm();

        // Acc register tile
        auto c_block_tile = decltype(block_gemm(a_lds_gemm_window, b_lds_gemm_window)){};

        // prefetch
        // global read 0
        // Load tile — during value loading, an elementwise function is executed for each A0,
        // A1, … AN. The values A0, A1, … AN are read by the same thread.
        auto elementwise_As_res = load_tile_with_elementwise(as_copy_dram_window, a_element_func);

        // Load tile — during value loading, an elementwise function is executed for each B0,
        // B1, … BN. The values B0, B1, … BN are read by the same thread.
        auto elementwise_Bs_res = load_tile_with_elementwise(bs_copy_dram_window, b_element_func);

        {
            // move to 1
            // Move each A — the enhanced function move_tile_window is executed, which takes a tuple
            // as input.
            move_tile_window(as_copy_dram_window, {0, kKPerBlock});
            // Move each B — the enhanced function move_tile_window is executed, which takes a tuple
            // as input.
            move_tile_window(bs_copy_dram_window, {0, kKPerBlock});

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            if constexpr(is_a_col_major)
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                store_tile(a_copy_lds_window, a_shuffle_tmp);
            }
            else
            {
                store_tile(a_copy_lds_window, elementwise_As_res);
            }

            // LDS write 0
            if constexpr(is_b_row_major)
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                store_tile(b_copy_lds_window, b_shuffle_tmp);
            }
            else
            {
                store_tile(b_copy_lds_window, elementwise_Bs_res);
            }
        }

        index_t iCounter = num_loop - 1;
        while(iCounter > 0)
        {
            // global read i + 1
            elementwise_As_res = load_tile_with_elementwise(as_copy_dram_window, a_element_func);
            elementwise_Bs_res = load_tile_with_elementwise(bs_copy_dram_window, b_element_func);

            block_sync_lds();

            // GEMM i
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            block_sync_lds();

            // move to i + 2
            move_tile_window(as_copy_dram_window, {0, kKPerBlock});
            move_tile_window(bs_copy_dram_window, {0, kKPerBlock});

            // LDS write i + 1
            if constexpr(is_a_col_major)
            {
                auto a_shuffle_tmp_loop = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp_loop, elementwise_As_res);
                store_tile(a_copy_lds_window, a_shuffle_tmp_loop);
            }
            else
            {
                store_tile(a_copy_lds_window, elementwise_As_res);
            }

            // LDS write i + 1
            if constexpr(is_b_row_major)
            {
                auto b_shuffle_tmp_loop = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp_loop, elementwise_Bs_res);
                store_tile(b_copy_lds_window, b_shuffle_tmp_loop);
            }
            else
            {
                store_tile(b_copy_lds_window, elementwise_Bs_res);
            }

            iCounter--;
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 1
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
        }

        return c_block_tile;
    }

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return operator()(
            a_dram_block_window_tmp,
            [](auto& e, const ADataType & a) { e = a; },
            b_dram_block_window_tmp,
            [](auto& e, const BDataType & b) { e = b; },
            num_loop,
            p_smem);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return operator()(ck_tile::make_tuple(a_dram_block_window_tmp),
                          ck_tile::make_tuple(b_dram_block_window_tmp),
                          num_loop,
                          p_smem);
    }
};

} // namespace ck_tile
