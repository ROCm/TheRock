// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <sstream>

#include "ck_tile/core.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

// ToDo: Change the Pipeline to actual memory pipeline.
template <typename Problem, typename Policy = GemmAQuantPipelineAgBgCrDefaultPolicy>
struct AQuantGemmPipelineAgBgCrMem : public BaseGemmPipelineAgBgCrMem<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrMem<Problem>;
    using PipelineImplBase = GemmAQuantPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using AQDataType     = remove_cvref_t<typename Problem::AQDataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using QuantGroupSize = remove_cvref_t<typename Problem::AQuantGroupSize>;

    static_assert(QuantGroupSize::kM == 1, "no block for M supported yet!");
    static_assert(QuantGroupSize::kN == 1, "only M/K blocks for AQuant kernel!");

    using I0 = number<0>;
    using I1 = number<1>;
    using I2 = number<2>;

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    static constexpr index_t AQPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<AQDataType>>::PackedSize;

    using ALayout  = remove_cvref_t<typename Problem::ALayout>;
    using AQLayout = remove_cvref_t<typename Problem::AQLayout>;
    using BLayout  = remove_cvref_t<typename Problem::BLayout>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    static constexpr index_t BlockSize   = Problem::kBlockSize;
    static constexpr index_t MPerBlock   = BlockGemmShape::kM;
    static constexpr index_t NPerBlock   = BlockGemmShape::kN;
    static constexpr index_t KPerBlock   = BlockGemmShape::kK;
    static constexpr index_t KPerBlockAQ = BlockGemmShape::kK / QuantGroupSize::kK;

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }
    static constexpr index_t GetVectorSizeAQ()
    {
        return Policy::template GetVectorSizeAQ<Problem>();
    }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr bool PreshuffleQuant  = Problem::Traits::PreshuffleQuant;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    using Base::PrefetchStages;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});
        return concat('_', "aquant_pipeline_AgBgCrMem", 
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      BlockSize,
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', BlockGemm::WarpGemm::kM, BlockGemm::WarpGemm::kN, BlockGemm::WarpGemm::kK),
                      concat('x', kPadM, kPadN, kPadK), QuantGroupSize::GetName(),
                      Scheduler == GemmPipelineScheduler::Interwave ? "interwave" : "intrawave"); // else Intrawave
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST static std::string Print()
    {
        constexpr index_t MPerXDL = BlockGemm::WarpGemm::kM;
        constexpr index_t NPerXDL = BlockGemm::WarpGemm::kN;
        constexpr index_t KPerXDL = BlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK;

        constexpr index_t WaveSize = 64;
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});

        constexpr index_t A_LDS_Read_Width = GetSmemPackA();
        constexpr index_t B_LDS_Read_Width = GetSmemPackB();

        constexpr index_t A_LDS_Write_Width = GetSmemPackA();
        constexpr index_t B_LDS_Write_Width = GetSmemPackB();

        constexpr index_t A_Buffer_Load_Inst_Num =
            MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
        constexpr index_t B_Buffer_Load_Inst_Num =
            NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());
        constexpr index_t AQ_Buffer_Load_Inst_Num =
            MPerBlock * KPerBlockAQ / (BlockSize * GetVectorSizeAQ());

        constexpr index_t A_LDS_Write_Inst_Num =
            MPerBlock * KPerBlock / (BlockSize * A_LDS_Write_Width);
        constexpr index_t B_LDS_Write_Inst_Num =
            NPerBlock * KPerBlock / (BlockSize * B_LDS_Write_Width);

        constexpr index_t A_LDS_Read_Inst_Num =
            WaveNumN * MPerBlock * KPerBlock / (BlockSize * A_LDS_Read_Width);
        constexpr index_t B_LDS_Read_Inst_Num =
            WaveNumM * NPerBlock * KPerBlock / (BlockSize * B_LDS_Read_Width);

        constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                            (BlockSize / WaveSize) / (MPerXDL * NPerXDL * KPerXDL);

        auto str = std::stringstream{};

        str << "A/B vector size: " << GetVectorSizeA() << ", " << GetVectorSizeB() << ", "
            << "AQ vector size: " << GetVectorSizeAQ() << "\n"
            << "A/B LDS read/write width: " << A_LDS_Read_Width << ", " << B_LDS_Read_Width << "\n"
            << "A/B buffer load inst: " << A_Buffer_Load_Inst_Num << ", " << B_Buffer_Load_Inst_Num
            << ", " << "AQ buffer load inst: " << AQ_Buffer_Load_Inst_Num << "\n"
            << "A/B LDS write inst: " << A_LDS_Write_Inst_Num << ", " << B_LDS_Write_Inst_Num
            << "\n"
            << "A/B LDS read inst: " << A_LDS_Read_Inst_Num << ", " << B_LDS_Read_Inst_Num << "\n"
            << "C MFMA inst: " << C_MFMA_Inst_Num << "\n"
            << "QuantGroupSize: " << QuantGroupSize::GetName() << "\n"
            << "KPack: " << BlockGemm::Traits::KPack << "\n"
            << "PrefetchStages: " << PrefetchStages << "\n";
        return str.str();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Interwave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <typename ADramWindow, typename ABlockTile_, typename DramTileWindowStep>
        CK_TILE_DEVICE static void
        LoadAndConvertATile(ABlockTile_& a_block_tile,
                            ADramWindow& a_dram_window,
                            const DramTileWindowStep& dram_tile_window_step)
        {
            using DestDataType            = typename ABlockTile_::DataType;
            using SrcDataType             = typename ADramWindow::Base::TileWindowBase::DataType;
            constexpr index_t UnaryOpSize = 8;
            load_int4_tile<SrcDataType, DestDataType, UnaryOpSize>(a_block_tile, a_dram_window);
            move_tile_window(a_dram_window, dram_tile_window_step);
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename AQDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       const AQDramBlockWindowTmp& aq_dram_block_window_tmp,
                                       [[maybe_unused]] index_t m,
                                       index_t num_loop,
                                       void* p_smem) const
        {
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>> &&
                    std::is_same_v<AQDataType,
                                   remove_cvref_t<typename AQDramBlockWindowTmp::DataType>>,
                "A/B/AQ Dram block window should have the same data type as appropriate "
                "([A|B|AQ]DataType) defined in Problem definition!");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_aq_col_major =
                std::is_same_v<AQLayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(!PreshuffleQuant, "Memory pipeline does not support PreshuffleQuant!");

            static_assert(is_a_col_major
                              ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");

            // A/B tiles in LDS - using the same approach as regular gemm pipeline
            auto ab_lds_blocks = Base::template GetABLdsTensorViews<BDataType, BDataType>(p_smem);
            auto& a_lds_block  = ab_lds_blocks.at(I0{});
            auto& b_lds_block  = ab_lds_blocks.at(I1{});

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            auto a_windows =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);
            auto& a_copy_dram_window = a_windows.at(I0{});
            auto& a_copy_lds_window  = a_windows.at(I1{});
            auto& a_lds_gemm_window  = a_windows.at(I2{});

            auto b_windows =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);
            auto& b_copy_dram_window = b_windows.at(I0{});
            auto& b_copy_lds_window  = b_windows.at(I1{});
            auto& b_lds_gemm_window  = b_windows.at(I2{});

            auto aq_copy_dram_window = Base::GetAQDramLoadWindow(aq_dram_block_window_tmp);

            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ABlockTileDistr  = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr  = decltype(b_copy_dram_window.get_tile_distribution());
            using AQBlockTileDistr = decltype(aq_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<BDataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));
            using AQBlockTile =
                decltype(make_static_distributed_tensor<AQDataType>(AQBlockTileDistr{}));

            // Memory pipeline uses multiple prefetch stages
            tuple_array<ABlockTile, PrefetchStages> a_block_tiles;
            tuple_array<BBlockTile, PrefetchStages> b_block_tiles;
            tuple_array<AQBlockTile, PrefetchStages> aq_block_tiles;

            using ADramTileWindowStep  = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep  = typename BDramBlockWindowTmp::BottomTensorIndex;
            using AQDramTileWindowStep = typename AQDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr AQDramTileWindowStep aq_dram_tile_window_step =
                is_aq_col_major ? make_array(KPerBlockAQ, 0) : make_array(0, KPerBlockAQ);

            // Global prefetch initialization - DRAM to VGPRs
            LoadAndConvertATile(
                a_block_tiles.get(I0{}), a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(
                b_block_tiles.get(I0{}), b_copy_dram_window, b_dram_tile_window_step);
            Base::GlobalPrefetch(
                aq_block_tiles.get(I0{}), aq_copy_dram_window, aq_dram_tile_window_step);

            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS prefill - VGPRs to LDS
            if constexpr(is_a_col_major && !is_a_load_tr_v())
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_block_tiles.get(I0{}));
                Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window, a_block_tiles.get(I0{}), a_element_func);
            }
            if constexpr(is_b_row_major && !is_b_load_tr_v())
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, b_block_tiles.get(I0{}));
                Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
            }
            else
            {
                Base::LocalPrefill(b_copy_lds_window, b_block_tiles.get(I0{}), b_element_func);
            }
            // Additional prefetching for memory pipeline - DRAM to VGPRs
            static_for<1, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                LoadAndConvertATile(a_block_tiles.get(number<prefetch_idx>{}),
                                    a_copy_dram_window,
                                    a_dram_tile_window_step);
                Base::GlobalPrefetch(b_block_tiles.get(number<prefetch_idx>{}),
                                     b_copy_dram_window,
                                     b_dram_tile_window_step);
                Base::GlobalPrefetch(aq_block_tiles.get(number<prefetch_idx>{}),
                                     aq_copy_dram_window,
                                     aq_dram_tile_window_step);
            });

            // Main hot loop for memory pipeline
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                        block_sync_lds();
                        block_gemm.LocalPrefetch(
                            a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
                        block_gemm(c_block_tile,
                                   aq_block_tiles.get(number<prefetch_idx>{}),
                                   a_lds_gemm_window,
                                   b_lds_gemm_window);
                        block_sync_lds();
                        // Prepare next iteration data
                        if constexpr(is_a_col_major && !is_a_load_tr_v())
                        {
                            auto a_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                                Policy::template MakeShuffledARegTileDistribution<Problem>());
                            transpose_tile2d(
                                a_shuffle_tmp,
                                a_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}));
                            Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                a_copy_lds_window,
                                a_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}),
                                a_element_func);
                        }
                        if constexpr(is_b_row_major && !is_b_load_tr_v())
                        {
                            auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                                Policy::template MakeShuffledBRegTileDistribution<Problem>());
                            transpose_tile2d(
                                b_shuffle_tmp,
                                b_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}));
                            Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                        }
                        else
                        {
                            Base::LocalPrefill(
                                b_copy_lds_window,
                                b_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}),
                                b_element_func);
                        }

                        LoadAndConvertATile(a_block_tiles.get(number<prefetch_idx>{}),
                                            a_copy_dram_window,
                                            a_dram_tile_window_step);
                        Base::GlobalPrefetch(b_block_tiles.get(number<prefetch_idx>{}),
                                             b_copy_dram_window,
                                             b_dram_tile_window_step);
                        Base::GlobalPrefetch(aq_block_tiles.get(number<prefetch_idx>{}),
                                             aq_copy_dram_window,
                                             aq_dram_tile_window_step);
                    });

                    i += PrefetchStages;
                } while(i < (num_loop - PrefetchStages));
            }

            // Tail handling
            auto HotLoopTail = [&](auto tail_num) {
                static_for<0, tail_num - 1, 1>{}([&](auto prefetch_idx) {
                    block_sync_lds();
                    block_gemm.LocalPrefetch(
                        a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
                    block_gemm(c_block_tile,
                               aq_block_tiles.get(number<prefetch_idx>{}),
                               a_lds_gemm_window,
                               b_lds_gemm_window);
                    // no second block_sync_lds because it's interwave

                    if constexpr(is_a_col_major && !is_a_load_tr_v())
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp,
                                         a_block_tiles.get(number<prefetch_idx + 1>{}));
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window,
                                           a_block_tiles.get(number<prefetch_idx + 1>{}));
                    }
                    if constexpr(is_b_row_major && !is_b_load_tr_v())
                    {
                        auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp,
                                         b_block_tiles.get(number<prefetch_idx + 1>{}));
                        Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window,
                                           b_block_tiles.get(number<prefetch_idx + 1>{}));
                    }
                });

                block_sync_lds();
                block_gemm.LocalPrefetch(
                    a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
                block_gemm(c_block_tile,
                           aq_block_tiles.get(number<tail_num - 1>{}),
                           a_lds_gemm_window,
                           b_lds_gemm_window);
            };

            if constexpr(TailNum == TailNumber::One)
            {
                block_sync_lds();
                block_gemm.LocalPrefetch(
                    a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
                block_gemm(
                    c_block_tile, aq_block_tiles.get(I0{}), a_lds_gemm_window, b_lds_gemm_window);
            }
            else if constexpr(TailNum == TailNumber::Two)
            {
                HotLoopTail(number<2>{});
            }
            else if constexpr(TailNum == TailNumber::Three)
            {
                HotLoopTail(number<3>{});
            }
            else if constexpr(TailNum == TailNumber::Four)
            {
                HotLoopTail(number<4>{});
            }
            else if constexpr(TailNum == TailNumber::Five)
            {
                HotLoopTail(number<5>{});
            }
            else if constexpr(TailNum == TailNumber::Six)
            {
                HotLoopTail(number<6>{});
            }
            else if constexpr(TailNum == TailNumber::Seven)
            {
                HotLoopTail(number<7>{});
            }
            else if constexpr(TailNum == TailNumber::Full)
            {
                HotLoopTail(number<PrefetchStages>{});
            }
            return c_block_tile;
        }
    };

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AQDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const AQDramBlockWindowTmp& aq_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem,
                                   index_t m = 0) const
    {
        return PipelineImpl<GemmPipelineScheduler::Interwave>{}
            .template operator()<HasHotLoop, TailNum>(
                a_dram_block_window_tmp,
                [](const BDataType& a) { return a; },
                b_dram_block_window_tmp,
                [](const BDataType& b) { return b; },
                aq_dram_block_window_tmp,
                m,
                num_loop,
                p_smem);
    }
};

} // namespace ck_tile
