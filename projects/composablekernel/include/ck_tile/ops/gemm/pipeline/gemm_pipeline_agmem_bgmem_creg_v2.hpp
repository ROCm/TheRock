// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2_default_policy.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename Problem>
struct BaseGemmPipelineAGmemBGmemCRegV2
{
    static constexpr index_t PrefetchStages   = 2;
    static constexpr index_t PrefillStages    = 1;
    static constexpr index_t GlobalBufferNum  = 1;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

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
template <typename Problem, typename Policy = GemmPipelineAGmemBGmemCRegV2DefaultPolicy>
struct GemmPipelineAGmemBGmemCRegV2 : public BaseGemmPipelineAGmemBGmemCRegV2<Problem>
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

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Problem::VectorSizeA;
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Problem::VectorSizeB;
    }
    static constexpr index_t GetVectorSizeC() { return Problem::VectorSizeC; }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool Preshuffle = Problem::Preshuffle;

    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

    // For the basic gemm pipelien DoubleSmemBuffer set to be false naturally.
    static constexpr bool DoubleSmemBuffer = false;

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "BASIC_V2";
        // clang-format on
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AGmemBGmemCRegV2",
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock, BlockSize));
        // clang-format on
    }
    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return integer_divide_ceil(sizeof(ADataType) *
                                       Policy::template MakeALdsBlockDescriptor<Problem>()
                                           .get_element_space_size() /
                                       APackedSize,
                                   16) *
                   16 +
               sizeof(BDataType) *
                   Policy::template MakeBLdsBlockDescriptor<Problem>().get_element_space_size() /
                   BPackedSize;
    }

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

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // A tile in LDS
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        constexpr index_t a_lds_block_space_size_aligned =
            integer_divide_ceil(
                sizeof(ADataType) * a_lds_block_desc.get_element_space_size() / APackedSize, 16) *
            16;

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
        auto a_copy_lds_window =
            make_tile_window(a_lds_block,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             as_copy_dram_window[number<0>{}].get_tile_distribution());

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
        auto b_copy_lds_window =
            make_tile_window(b_lds_block,
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             bs_copy_dram_window[number<0>{}].get_tile_distribution());

        // Block GEMM
        constexpr auto block_gemm = Policy::template GetBlockGemm<Problem>();

        // Tile distribution for load from lds
        constexpr auto a_lds_load_tile_distr =
            make_static_tile_distribution(decltype(block_gemm)::MakeABlockDistributionEncode());
        constexpr auto b_lds_load_tile_distr =
            make_static_tile_distribution(decltype(block_gemm)::MakeBBlockDistributionEncode());

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
            move_tile_window(as_copy_dram_window, {0, kKPerBlock});
            move_tile_window(bs_copy_dram_window, {0, kKPerBlock});

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            store_tile(a_copy_lds_window, elementwise_As_res);
            // global read 1
            elementwise_As_res = load_tile_with_elementwise(as_copy_dram_window, a_element_func);

            // LDS write 0
            store_tile(b_copy_lds_window, elementwise_Bs_res);
            // global read 1
            elementwise_Bs_res = load_tile_with_elementwise(bs_copy_dram_window, b_element_func);
        }

        index_t iCounter = num_loop - 2;

        do
        {
            block_sync_lds();

            // GEMM i
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            block_sync_lds();

            // move to i + 2
            move_tile_window(as_copy_dram_window, {0, kKPerBlock});
            move_tile_window(bs_copy_dram_window, {0, kKPerBlock});

            // LDS write i + 1
            store_tile(a_copy_lds_window, elementwise_As_res);
            // global read i + 2
            elementwise_As_res = load_tile_with_elementwise(as_copy_dram_window, a_element_func);

            // LDS write i + 1
            store_tile(b_copy_lds_window, elementwise_Bs_res);
            // global read i + 2
            elementwise_Bs_res = load_tile_with_elementwise(bs_copy_dram_window, b_element_func);

            iCounter--;

        } while(iCounter > 0);

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 2
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            block_sync_lds();

            // LDS write num_loop - 1
            store_tile(a_copy_lds_window, elementwise_As_res);

            store_tile(b_copy_lds_window, elementwise_Bs_res);

            block_sync_lds();

            // GEMM num_loop - 1
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
        }

        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
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
