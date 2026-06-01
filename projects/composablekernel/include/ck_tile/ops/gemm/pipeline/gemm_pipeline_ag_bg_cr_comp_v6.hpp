// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6_default_policy.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem>
struct BaseGemmPipelineAgBgCrCompV6
{
    static constexpr index_t PrefetchStages  = 3;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 2;
    static constexpr index_t HotloopUnroll   = 2;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(num_loop % HotloopUnroll == 1)
        {
            return TailNumber::Odd;
        }
        else
        {
            return TailNumber::Even;
        }
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
    {
        // Handle all the valid cases.
        if(has_hot_loop)
        {
            if(tail_number == TailNumber::Odd)
            {
                return run_func(bool_constant<true>{},
                                integral_constant<TailNumber, TailNumber::Odd>{});
            }
            else if(tail_number == TailNumber::Even)
            {
                return run_func(bool_constant<true>{},
                                integral_constant<TailNumber, TailNumber::Even>{});
            }
        }
        else
        {
            if(tail_number == TailNumber::Odd)
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::Odd>{});
            }
            else if(tail_number == TailNumber::Even)
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::Even>{});
            }
        }
        // If execution reaches here, it's an invalid tail_number because it wasn't handled above.
#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_unreachable();
#else
        throw std::logic_error("Invalid TailNumber: Only TailNumber::Odd and TailNumber::Even are "
                               "supported in this pipeline context.");
#endif
    }
};

// Compute optimized pipeline
// GlobalPrefetchStages: 3
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 2
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV6DefaultPolicy>
struct GemmPipelineAgBgCrCompV6 : public BaseGemmPipelineAgBgCrCompV6<Problem>
{
    using Base      = BaseGemmPipelineAgBgCrCompV6<Problem>;
    using BasePImpl = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using AsDataType     = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType     = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using AElementWise = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise = remove_cvref_t<typename Problem::BElementWise>;

    using AsLayout = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    static_assert(!std::is_same_v<BDataType, pk_int4_t>, "Not implemented");

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

    using BlockGemm          = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

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

    static constexpr index_t KRepeat = BlockGemm::WarpGemm::kKPerThread / GetSmemPackA();

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    static constexpr auto Scheduler = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<BasePImpl::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<BasePImpl::is_b_load_tr>{};

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "COMPUTE_V6";
        // clang-format on
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AgBgCrCompV6", BlockSize,
                      concat('x', GetVectorSizeA(), GetVectorSizeB(),  GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK),
                      concat('_', KRepeat),
                      concat('_', DoubleSmemBuffer),
                      concat('_', Preshuffle));
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC()
    {
        return Policy::template IsTransposeC<Problem>();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public BasePImpl
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public BasePImpl
    {
        CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
        {
            constexpr index_t MPerXDL = BlockGemmShape::WarpTile::at(I0);
            constexpr index_t NPerXDL = BlockGemmShape::WarpTile::at(I1);
            constexpr index_t KPerXDL = BlockGemmShape::WarpTile::at(I2);

            constexpr index_t WaveSize = 64;
            constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0);
            constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1);

            constexpr index_t A_LDS_Read_Width = KPerXDL;
            constexpr index_t B_LDS_Read_Width = KPerXDL;

            constexpr index_t A_Buffer_Load_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
            constexpr index_t B_Buffer_Load_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());

            constexpr index_t A_LDS_Write_Inst_Num = MPerBlock * KPerBlock / (BlockSize * KPerXDL);
            constexpr index_t B_LDS_Write_Inst_Num = NPerBlock * KPerBlock / (BlockSize * KPerXDL);

            constexpr index_t A_LDS_Read_Inst_Num =
                WaveNumN * MPerBlock * KPerBlock / (BlockSize * KPerXDL);
            constexpr index_t B_LDS_Read_Inst_Num =
                WaveNumM * NPerBlock * KPerBlock / (BlockSize * KPerXDL);

            constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                                (BlockSize / WaveSize) /
                                                (MPerXDL * NPerXDL * KPerXDL);

            constexpr auto num_ds_read_inst_a =
                A_LDS_Read_Width * sizeof(ADataType) / APackedSize == 16 ? A_LDS_Read_Inst_Num
                                                                         : A_LDS_Read_Inst_Num / 2;
            constexpr auto num_ds_read_inst_b =
                B_LDS_Read_Width * sizeof(BDataType) / BPackedSize == 16 ? B_LDS_Read_Inst_Num
                                                                         : B_LDS_Read_Inst_Num / 2;

            constexpr auto mfma_cycle = NPerXDL == 16 ? 16 : 32;

            constexpr auto ds_read_a_issue_cycle =
                A_LDS_Read_Width * sizeof(ADataType) / APackedSize == 16 ? 8 : 4;
            constexpr auto ds_read_b_issue_cycle =
                B_LDS_Read_Width * sizeof(BDataType) / BPackedSize == 16 ? 8 : 4;

            constexpr auto ds_read_a_mfma_rate =
                (mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
            constexpr auto ds_read_b_mfma_rate =
                (mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);

            constexpr auto num_dsread_stage1_a = num_ds_read_inst_a / KRepeat * (KRepeat - 1);
            constexpr auto num_dsread_stage1_b = num_ds_read_inst_b / KRepeat * (KRepeat - 1);
            constexpr auto num_dsread_stage3_a = num_ds_read_inst_a / KRepeat;
            constexpr auto num_dsread_stage3_b = num_ds_read_inst_b / KRepeat;

            constexpr auto num_dsread_stage1_a_mfma =
                (num_dsread_stage1_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
            constexpr auto num_dsread_stage1_b_mfma =
                (num_dsread_stage1_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;
            constexpr auto num_dsread_stage3_a_mfma =
                (num_dsread_stage3_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
            constexpr auto num_dsread_stage3_b_mfma =
                (num_dsread_stage3_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;

            constexpr auto num_mfma_stage2 = C_MFMA_Inst_Num -
                                             num_ds_read_inst_a / ds_read_a_mfma_rate -
                                             num_ds_read_inst_b / ds_read_b_mfma_rate;
            constexpr auto num_mfma_per_issue =
                num_mfma_stage2 / (A_Buffer_Load_Inst_Num + B_Buffer_Load_Inst_Num);
            constexpr auto num_dswrite_per_issue_a = A_LDS_Write_Inst_Num / A_Buffer_Load_Inst_Num;
            constexpr auto num_dswrite_per_issue_b = B_LDS_Write_Inst_Num / B_Buffer_Load_Inst_Num;

            // stage 1
            static_for<0, num_dsread_stage1_a_mfma, 1>{}([&](auto i) {
                ignore = i;
                if constexpr((num_dsread_stage1_a - (i + 1) * ds_read_a_mfma_rate) >=
                             ds_read_a_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_dsread_stage1_a - (num_dsread_stage1_a_mfma - 1) * ds_read_a_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            static_for<0, num_dsread_stage1_b_mfma, 1>{}([&](auto i) {
                ignore = i;
                if constexpr((num_dsread_stage1_b - (i + 1) * ds_read_b_mfma_rate) >=
                             ds_read_b_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_dsread_stage1_b - (num_dsread_stage1_b_mfma - 1) * ds_read_b_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            // stage 2
            static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                static_for<0, num_dswrite_per_issue_a, 1>{}([&](auto idswrite) {
                    ignore = idswrite;
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(
                    0x008, num_mfma_per_issue - num_dswrite_per_issue_a, 0); // MFMA
            });
            static_for<0, B_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
                    ignore = idswrite;
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(
                    0x008, num_mfma_per_issue - num_dswrite_per_issue_b, 0); // MFMA
            });

            // stage 3
            static_for<0, num_dsread_stage3_a_mfma, 1>{}([&](auto i) {
                ignore = i;
                if constexpr((num_dsread_stage3_a - (i + 1) * ds_read_a_mfma_rate) >=
                             ds_read_a_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_dsread_stage3_a - (num_dsread_stage3_a_mfma - 1) * ds_read_a_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            static_for<0, num_dsread_stage3_b_mfma, 1>{}([&](auto i) {
                ignore = i;
                if constexpr((num_dsread_stage3_b - (i + 1) * ds_read_b_mfma_rate) >=
                             ds_read_b_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_dsread_stage3_b - (num_dsread_stage3_b_mfma - 1) * ds_read_b_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            __builtin_amdgcn_sched_barrier(0);
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* __restrict__ p_smem) const
        {
            // TODO: Add Multi A/B support
            static_assert(std::tuple_size<remove_cvref_t<AsDramBlockWindowTmp>>::value == 1,
                          "Multi A/B is not yet supported for this pipeline.");
            static_assert(std::tuple_size<remove_cvref_t<BsDramBlockWindowTmp>>::value == 1,
                          "Multi A/B is not yet supported for this pipeline.");

            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "Data Type conflict on A and B matrix input data type.");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(is_a_col_major
                              ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1])
                              : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1]),
                          "B block window has incorrect lengths for defined BLayout!");

            ////////////// LDS desc, window & register /////////////////
            using ALdsType =
                remove_cvref_t<decltype(BasePImpl::GetABLdsTensorViews(p_smem).at(I0))>;
            using BLdsType =
                remove_cvref_t<decltype(BasePImpl::GetABLdsTensorViews(p_smem).at(I1))>;
            auto&& ABLdsTensorViews = BasePImpl::GetABLdsTensorViews(p_smem);
            ALdsType& a_lds_block   = ABLdsTensorViews.at(I0);
            BLdsType& b_lds_block   = ABLdsTensorViews.at(I1);

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            using acopy_dram_type =
                remove_cvref_t<decltype(BasePImpl::GetAWindows(a_dram_block_window_tmp,
                                                               a_lds_block,
                                                               a_lds_load_tile_distr)
                                            .at(I0))>;
            using bcopy_dram_type =
                remove_cvref_t<decltype(BasePImpl::GetBWindows(b_dram_block_window_tmp,
                                                               b_lds_block,
                                                               b_lds_load_tile_distr)
                                            .at(I0))>;

            using a_copy_lds_window_type =
                remove_cvref_t<decltype(BasePImpl::GetAWindows(a_dram_block_window_tmp,
                                                               a_lds_block,
                                                               a_lds_load_tile_distr)
                                            .at(I1))>;
            using b_copy_lds_window_type =
                remove_cvref_t<decltype(BasePImpl::GetBWindows(b_dram_block_window_tmp,
                                                               b_lds_block,
                                                               b_lds_load_tile_distr)
                                            .at(I1))>;

            using a_lds_load_tile_distr_type =
                remove_cvref_t<decltype(BasePImpl::GetAWindows(a_dram_block_window_tmp,
                                                               a_lds_block,
                                                               a_lds_load_tile_distr)
                                            .at(I2))>;
            using b_lds_load_tile_distr_type =
                remove_cvref_t<decltype(BasePImpl::GetBWindows(b_dram_block_window_tmp,
                                                               b_lds_block,
                                                               b_lds_load_tile_distr)
                                            .at(I2))>;

            auto&& aWindows =
                BasePImpl::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);
            auto&& bWindows =
                BasePImpl::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            acopy_dram_type& a_copy_dram_window           = aWindows.at(I0);
            a_copy_lds_window_type& a_copy_lds_window     = aWindows.at(I1);
            a_lds_load_tile_distr_type& a_lds_gemm_window = aWindows.at(I2);

            // B DRAM tile window for load
            // B LDS tile window for store
            // B LDS tile for block GEMM
            bcopy_dram_type& b_copy_dram_window           = bWindows.at(I0);
            b_copy_lds_window_type& b_copy_lds_window     = bWindows.at(I1);
            b_lds_load_tile_distr_type& b_lds_gemm_window = bWindows.at(I2);

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ABlockTileDistr =
                decltype(a_copy_dram_window[number<0>{}].get_tile_distribution());
            using BBlockTileDistr =
                decltype(b_copy_dram_window[number<0>{}].get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

            ABlockTile a_block_tile[Base::GlobalBufferNum];
            BBlockTile b_block_tile[Base::GlobalBufferNum];

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            constexpr auto ALdsTileDistr = decltype(make_static_tile_distribution(
                BlockGemm::MakeABlockDistributionEncode())){};
            constexpr auto BLdsTileDistr = decltype(make_static_tile_distribution(
                BlockGemm::MakeBBlockDistributionEncode())){};

            using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
            using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));

            ALdsTile a_lds_tile;
            BLdsTile b_lds_tile;
            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // Global prefetch 1
            a_block_tile[I0] = load_tile_with_elementwise(a_copy_dram_window, a_element_func);
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
            b_block_tile[I0] = load_tile_with_elementwise(b_copy_dram_window, b_element_func);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // Local prefill 1
            if constexpr(is_a_col_major && !is_a_load_tr_v())
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_block_tile[I0]);
                BasePImpl::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
            }
            else
            {
                BasePImpl::LocalPrefill(a_copy_lds_window, a_block_tile[I0]);
            }
            if constexpr(is_b_row_major && !is_b_load_tr_v())
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, b_block_tile[I0]);
                BasePImpl::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
            }
            else
            {
                BasePImpl::LocalPrefill(b_copy_lds_window, b_block_tile[I0]);
            }

            // Global prefetch 2
            a_block_tile[I0] = load_tile_with_elementwise(a_copy_dram_window, a_element_func);
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
            b_block_tile[I0] = load_tile_with_elementwise(b_copy_dram_window, b_element_func);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            // Global prefetch 3
            a_block_tile[I1] = load_tile_with_elementwise(a_copy_dram_window, a_element_func);
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
            b_block_tile[I1] = load_tile_with_elementwise(b_copy_dram_window, b_element_func);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            block_sync_lds();

            // Local prefetch 1
            BasePImpl::LocalPrefetch(a_lds_tile, a_lds_gemm_window, is_a_load_tr_v);
            BasePImpl::LocalPrefetch(b_lds_tile, b_lds_gemm_window, is_b_load_tr_v);

            if(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    auto LoopFunc = [&](auto vmem_buf_idx) {
                        static_for<0, KRepeat, 1>{}([&](auto k0) {
                            if constexpr(k0 == (KRepeat - 1))
                            {
                                block_sync_lds();

                                // Local prefill 2
                                if constexpr(is_a_col_major && !is_a_load_tr_v())
                                {
                                    auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                                        Policy::template MakeShuffledARegTileDistribution<
                                            Problem>());
                                    transpose_tile2d(a_shuffle_tmp, a_block_tile[vmem_buf_idx]);
                                    BasePImpl::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                                }
                                else
                                {
                                    BasePImpl::LocalPrefill(a_copy_lds_window,
                                                            a_block_tile[vmem_buf_idx]);
                                }
                                if constexpr(is_b_row_major && !is_b_load_tr_v())
                                {
                                    auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                                        Policy::template MakeShuffledBRegTileDistribution<
                                            Problem>());
                                    transpose_tile2d(b_shuffle_tmp, b_block_tile[vmem_buf_idx]);
                                    BasePImpl::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                                }
                                else
                                {
                                    BasePImpl::LocalPrefill(b_copy_lds_window,
                                                            b_block_tile[vmem_buf_idx]);
                                }

                                // Global prefetch 4
                                a_block_tile[vmem_buf_idx] =
                                    load_tile_with_elementwise(a_copy_dram_window, a_element_func);
                                move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
                                b_block_tile[vmem_buf_idx] =
                                    load_tile_with_elementwise(b_copy_dram_window, b_element_func);
                                move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

                                block_sync_lds();
                            }
                            block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                            // Local prefetch 2
                            BasePImpl::LocalPrefetch(a_lds_tile, a_lds_gemm_window, is_a_load_tr_v);
                            BasePImpl::LocalPrefetch(b_lds_tile, b_lds_gemm_window, is_b_load_tr_v);
                        });

                        HotLoopScheduler();
                    };

                    LoopFunc(I0);
                    LoopFunc(I1);

                    i += Base::HotloopUnroll;
                } while(i < (num_loop - Base::PrefetchStages));
            }

            auto ReadWriteCompFunc = [&](auto vmem_buf_idx) {
                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    if constexpr(k0 == (KRepeat - 1))
                    {
                        block_sync_lds();

                        // Local prefill 3
                        if constexpr(is_a_col_major && !is_a_load_tr_v())
                        {
                            auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                                Policy::template MakeShuffledARegTileDistribution<Problem>());
                            transpose_tile2d(a_shuffle_tmp, a_block_tile[vmem_buf_idx]);
                            BasePImpl::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                        }
                        else
                        {
                            BasePImpl::LocalPrefill(a_copy_lds_window, a_block_tile[vmem_buf_idx]);
                        }
                        if constexpr(is_b_row_major && !is_b_load_tr_v())
                        {
                            auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                                Policy::template MakeShuffledBRegTileDistribution<Problem>());
                            transpose_tile2d(b_shuffle_tmp, b_block_tile[vmem_buf_idx]);
                            BasePImpl::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                        }
                        else
                        {
                            BasePImpl::LocalPrefill(b_copy_lds_window, b_block_tile[vmem_buf_idx]);
                        }

                        block_sync_lds();
                    }

                    block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                    BasePImpl::LocalPrefetch(a_lds_tile, a_lds_gemm_window, is_a_load_tr_v);
                    BasePImpl::LocalPrefetch(b_lds_tile, b_lds_gemm_window, is_b_load_tr_v);
                });

                HotLoopScheduler();
            };

            auto ReadCompFunc = [&]() {
                static_for<0, KRepeat - 1, 1>{}([&]() {
                    __syncthreads();
                    block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                    // Local prefetch 4
                    BasePImpl::LocalPrefetch(a_lds_tile, a_lds_gemm_window, is_a_load_tr_v);
                    BasePImpl::LocalPrefetch(b_lds_tile, b_lds_gemm_window, is_b_load_tr_v);

                    __syncthreads();
                });

                block_gemm(c_block_tile, a_lds_tile, b_lds_tile);

                HotLoopScheduler();
            };

            if constexpr(TailNum == TailNumber::Odd)
            {
                ReadWriteCompFunc(I0);
                ReadWriteCompFunc(I1);
                ReadCompFunc();
            }
            else if constexpr(TailNum == TailNumber::Even)
            {
                ReadWriteCompFunc(I0);
                ReadCompFunc();
            }

            return c_block_tile;
        }
    };

    public:
    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* __restrict__ p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                a_element_func,
                b_dram_block_window_tmp,
                b_element_func,
                num_loop,
                p_smem);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const index_t num_loop,
                                   void* __restrict__ p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                [](auto& e, const ADataType& a) { e = a; },
                b_dram_block_window_tmp,
                [](auto& e, const BDataType& b) { e = b; },
                num_loop,
                p_smem);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* __restrict__ p_smem) const
    {
        return operator()(ck_tile::make_tuple(a_dram_block_window_tmp),
                          a_element_func,
                          ck_tile::make_tuple(b_dram_block_window_tmp),
                          b_element_func,
                          num_loop,
                          p_smem);
    }
};
} // namespace ck_tile
