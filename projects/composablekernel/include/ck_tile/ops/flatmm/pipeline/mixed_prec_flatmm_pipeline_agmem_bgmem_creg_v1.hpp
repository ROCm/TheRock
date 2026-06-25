// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp"
#include "ck_tile/ops/flatmm/pipeline/mixed_prec_flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_      = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                      = true,
          TailNumber TailNum_                   = TailNumber::Full,
          amd_buffer_coherence_enum BMemNTType_ = amd_buffer_coherence_enum::coherence_default,
          bool BPreShufflePermute_              = false,
          typename ComputeDataType_             = ADataType_>
struct F16xMXF4FlatmmPipelineProblem : FlatmmPipelineProblem<ADataType_,
                                                             ADataType_,
                                                             CDataType_,
                                                             BlockGemmShape_,
                                                             Traits_,
                                                             Scheduler_,
                                                             HasHotLoop_,
                                                             TailNum_,
                                                             BMemNTType_,
                                                             BPreShufflePermute_,
                                                             ComputeDataType_>
{
    using BlockGemmShape = BlockGemmShape_;

    using QuantType = BDataType_;

    static constexpr index_t flatNPerWarp = BlockGemmShape::flatNPerWarp;

    static constexpr int MXF4ScaleGranularityK = 32;

    static constexpr int ContinuousKPerThread      = 32; // it's fixed for fp4
    static constexpr int ContinuousScaleNPerThread = 2;  // it's fixed for fp4
    static constexpr int ContinuousScaleKPerThread = 2;  // it's fixed for fp4
    static constexpr index_t flatKPerWarp          = 64 * ContinuousKPerThread;
};

template <typename Problem, typename PipelinePolicy = F16xMXF4FlatmmPipelineAgBgCrPolicy>
struct F16xMXF4FlatmmPipelineAGmemBGmemCRegV1
    : FlatmmPipelineAGmemBGmemCRegV1<Problem, PipelinePolicy>
{
    using Underlying = FlatmmPipelineAGmemBGmemCRegV1<Problem, PipelinePolicy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::QuantType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>; // TileFlatmmShape

    using ComputeType = ADataType;
    static_assert(sizeof(ADataType) >= sizeof(BDataType));

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockFlatmm =
        remove_cvref_t<decltype(PipelinePolicy::template GetBlockFlatmm<Problem>())>;

    static constexpr auto config =
        BlockFlatmm::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    static constexpr index_t DsWritePreIssue = 3; // default 2, ds write at MIter - 2
    static constexpr index_t DsReadPreload   = 2; // default 2, preload 2 ds read

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t WaveSize  = get_warp_size();

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr index_t flatKPerWarp = Problem::flatKPerWarp;
    static constexpr index_t flatNPerWarp = Problem::flatNPerWarp;

    static constexpr index_t GetVectorSizeA() { return Problem::VectorSizeA; }
    static constexpr index_t GetVectorSizeB() { return 32; /* fixed for fp4 shuffle layout*/ }
    static constexpr index_t GetVectorSizeC() { return Problem::VectorSizeC; }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr index_t kLdsAlignmentInBytes = 16;
    static constexpr index_t NumWaveGroups        = Problem::NumWaveGroups;
    static constexpr bool UsePersistentKernel     = Problem::Traits::UsePersistentKernel;

    static constexpr auto I0   = number<0>();
    static constexpr auto I1   = number<1>();
    static constexpr auto I2   = number<2>();
    static constexpr auto idxM = I0;
    static constexpr auto idxN = I1;
    static constexpr auto idxK = I2;
    using BlockTile            = remove_cvref_t<typename BlockGemmShape::BlockTile>;
    using BlockWarps           = remove_cvref_t<typename BlockGemmShape::BlockWarps>;
    using WarpTile             = remove_cvref_t<typename BlockGemmShape::WarpTile>;

    static constexpr index_t MWarp = config.template at<1>();
    static constexpr index_t NWarp = config.template at<2>();

    static constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
    static constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);
    static constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

    static constexpr index_t KFlatPerBlockPerIter = flatKPerWarp;
    static constexpr index_t NFlatPerBlockPerIter = flatNPerWarp;

    static constexpr index_t MPerBlockPerIter = kMPerBlock / MIterPerWarp;
    static constexpr index_t KPerBlockPerIter = kKPerBlock / KIterPerWarp;

    static constexpr int MXFP4PackedSize = 2;
    static constexpr index_t AK1         = Problem::VectorLoadSize / sizeof(ADataType);
    static constexpr index_t BK1 = Problem::VectorLoadSize / sizeof(BDataType) * MXFP4PackedSize;
    static constexpr index_t m_preload = (MIterPerWarp * KIterPerWarp >= DsReadPreload)
                                             ? DsReadPreload
                                             : MIterPerWarp * KIterPerWarp;

    static constexpr int ContinuousKPerThread      = Problem::ContinuousKPerThread;
    static constexpr int ContinuousScaleNPerThread = Problem::ContinuousScaleNPerThread;
    static constexpr int ContinuousScaleKPerThread = Problem::ContinuousScaleKPerThread;

    static constexpr int ScaleKFlatPerWarp =
        ContinuousScaleNPerThread * ContinuousScaleKPerThread * get_warp_size();

    static constexpr int XDLK_PerThread =
        WarpTile::at(I2) / (get_warp_size() / WarpTile::at(I1)); // 8

    static constexpr int XDL_PerWeightK = 4;                                          // 4
    static constexpr int XDL_PerScaleK  = XDL_PerWeightK * ContinuousScaleKPerThread; // 4
    static constexpr int XDL_PerScaleN  = ContinuousScaleNPerThread;                  // 2
    static_assert(XDL_PerScaleK % XDL_PerWeightK == 0);
    static_assert(KIterPerWarp % XDL_PerScaleK == 0);
    static_assert(NIterPerWarp % XDL_PerScaleN == 0);

    static constexpr int MXFP4KPerWarp = KIterPerWarp / XDL_PerWeightK;
    static constexpr int ScaleKPerWarp = KIterPerWarp / XDL_PerScaleK;
    static constexpr int ScaleNPerWarp = NIterPerWarp / XDL_PerScaleN;

    static constexpr int MXFP4K_PerScaleK = MXFP4KPerWarp / ScaleKPerWarp;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;

#ifdef __gfx942__
    static constexpr index_t mfma_per_wg = 2;
#else
    static constexpr index_t mfma_per_wg = 1;
#endif
    static constexpr index_t dsread_per_wg =
        WG::kM * WG::kK * sizeof(ADataType) / WaveSize / Problem::VectorLoadSize;
    static_assert((WG::kM * WG::kK * sizeof(ADataType) / WaveSize) % Problem::VectorLoadSize == 0);

    static constexpr index_t dsread_num_perK  = dsread_per_wg * MIterPerWarp;
    static constexpr index_t dswrite_num_perK = dsread_num_perK / (MWarp * NWarp);
    static constexpr index_t dswrite_rep    = (dswrite_num_perK + MIterPerWarp - 1) / MIterPerWarp;
    static constexpr index_t Aload_num_perK = dswrite_num_perK;
    static constexpr index_t Aload_rep      = dswrite_rep;
    static constexpr index_t Bload_num_perK = kNPerBlock * WG::kK / NWarp / BK1 / WaveSize;
    static constexpr index_t ScaleBload_K1  = ContinuousScaleNPerThread * ContinuousScaleKPerThread;
    static constexpr index_t ScaleBload_num =
        kNPerBlock * kKPerBlock / NWarp / 32 / ScaleBload_K1 /
        WaveSize; // BlockN * BlockK / NWarp / ScalePerK / ScaleB_K1 / wavesize
    static constexpr index_t Bload_total_num =
        Bload_num_perK * KIterPerWarp + ScaleBload_num + 0X3f0;
    static constexpr index_t KPerScaleLoad = KIterPerWarp / ScaleBload_num;
    static constexpr index_t HalfMIter     = (MIterPerWarp + 1) / 2;
    static constexpr index_t Bload_rep     = (Bload_num_perK + HalfMIter - 1) / HalfMIter;

    static constexpr index_t mfma_perM_perK = NIterPerWarp * mfma_per_wg;
    static constexpr index_t dswrite_mIter  = (DsWritePreIssue - 1) % MIterPerWarp;
    static constexpr index_t dswrite_kIter  = (DsWritePreIssue - 1) / MIterPerWarp;

    // For the basic gemm pipelien DoubleSmemBuffer set to be false naturally.
    static constexpr bool DoubleSmemBuffer = false;

    static constexpr auto BMemNTType         = Problem::BMemNTType;
    static constexpr bool BPreShufflePermute = Problem::BPreShufflePermute;

    CK_TILE_HOST_DEVICE static constexpr auto
    SchedulerPerM(index_t dsread_perM, index_t dswrite_perM, index_t load_perM)
    {
#if CKTILE_FLATMM_USE_BUFFER_LOAD_LDS
        // GFX950 use BUFFER_LOAD_LDS to fill lds_buffer_A.
        // There is no separate DS_WRITE instruction at all.
        dswrite_perM = 0;
#endif
        // Init inst order
        index_t max_data_inst   = dsread_perM > load_perM
                                      ? (dsread_perM > dswrite_perM ? dsread_perM : dswrite_perM)
                                      : (load_perM > dswrite_perM ? load_perM : dswrite_perM);
        index_t sum_data_inst   = dsread_perM + load_perM + dswrite_perM;
        index_t round_data_inst = (sum_data_inst + mfma_perM_perK - 1) / mfma_perM_perK;

        index_t inst_order[NIterPerWarp * 10];
        _Pragma("unroll") for(int idx = 0; idx < NIterPerWarp * 10; idx++) { inst_order[idx] = 0; }

        index_t index = 0;
        _Pragma("unroll") for(int j = 0; j < max_data_inst; j++)
        {
            if(dswrite_perM > j)
            {
                inst_order[index] = 1;
                index++;
            }
            if(load_perM > j)
            {
                inst_order[index] = 2;
                index++;
            }
            if(dsread_perM > j)
            {
                inst_order[index] = 3;
                index++;
            }
        }

        // Schedule IGLP
        _Pragma("unroll") for(int j = 0; j < mfma_perM_perK; j++)
        {
            index_t inst_idx = 0;
            if(j == 0)
                ;
            else if(j == 1)
                inst_idx = mfma_perM_perK == 2 ? 1 : mfma_perM_perK - 2;
            else if(j == 2)
                inst_idx = mfma_perM_perK - 1;
            else
                inst_idx = mfma_perM_perK - j;

            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

            _Pragma("unroll") for(int r = 0; r < round_data_inst; r++)
            {
                if(r % 2 == 0)
                {
                    if(inst_order[inst_idx + r * mfma_perM_perK] == 1)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    }
                    if(inst_order[inst_idx + r * mfma_perM_perK] == 2)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    }
                    if(inst_order[inst_idx + r * mfma_perM_perK] == 3)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                }
                else
                {
                    if(inst_order[(r + 1) * mfma_perM_perK - 1 - inst_idx] == 1)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    }
                    if(inst_order[(r + 1) * mfma_perM_perK - 1 - inst_idx] == 2)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    }
                    if(inst_order[(r + 1) * mfma_perM_perK - 1 - inst_idx] == 3)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                }
            }
        }
    }
    CK_TILE_HOST_DEVICE static constexpr auto HotLoopScheduler()
    {
        // Keypoint of pipeline optimize is workload balance in time
        // instruction schedule example(128X256X256, 1X4, 16X16X128):
        // Iter MNK     MFMA    ds_read ds_write    A_load  b_load
        // -1   M6N0:   57      -       8           -       -
        // -1   M6N1:   58      1       -           -       -
        // -1   M6N2:   59      -       -           7       -
        // -1   M6N3:   60      2       -           -       -
        // -1   M7N0:   61      -       -           -       -
        // -1   M7N1:   62      3       -           -       -
        // -1   M7N2:   63      -       -           8       -
        // -1   M7N3:   64      4       -           -       -
        //  0   M0N0K0:  1      -       -           -       1
        //  0   M0N1:    2      5       -           -       -
        //  0   M0N2:    3      -       -           -       2
        //  0   M0N3:    4      6       -           -       -
        //  0   M1N0:    5      -       -           -       3
        //  0   M1N1:    6      7       -           -       -
        //  0   M1N2:    7      -       -           -       4
        //  0   M1N3:    8      8       -           -       -
        //  0   M2N0:    9      -       -           -       5
        //  0   M2N1:   10      9       -           -       -
        //  0   M2N2:   11      -       -           -       6
        //  0   M2N3:   12     10       -           -       -
        //  0   M3N0:   13      -       1           -       7
        //  0   M3N1:   14     11       -           -       -
        //  0   M3N2:   15      -       -           -       8
        //  0   M3N3:   16     12       -           -       -
        //  0   M4N0:   17      -       2           -       -
        //  0   M4N1:   18     13       -           -       -
        //  0   M4N2:   19      -       -           1       -
        //  0   M4N3:   20     14       -           -       -
        //  0   M5N0:   21      -       3           -       -
        //  0   M5N1:   22     15       -           -       -
        //  0   M5N2:   23      -       -           2       -
        //  0   M5N3:   24     16       -           -       -
        //  0   M6N0:   25      -       4           -       -
        //  0   M6N1:   26     17       -           -       -
        //  0   M6N2:   27      -       -           3       -
        //  0   M6N3:   28     18       -           -       -
        //  0   M7N0:   29      -       -           -       -
        //  0   M7N1:   30     19       -           -       -
        //  0   M7N2:   31      -       -           4       -
        //  0   M7N3:   32     20       -           -       -
        //  0   M0N0K1: 33      -       -           -       9
        //  0   M0N1:   34     21       -           -       -
        //  0   M0N2:   35      -       -           -       10
        //  0   M0N3:   36     22       -           -       -
        //  0   M1N0:   37      -       -           -       11
        //  0   M1N1:   38     23       -           -       -
        //  0   M1N2:   39      -       -           -       12
        //  0   M1N3:   40     24       -           -       -
        //  0   M2N0:   41      -       -           -       13
        //  0   M2N1:   42     25       -           -       -
        //  0   M2N2:   43      -       -           -       14
        //  0   M2N3:   44     26       -           -       -
        //  0   M3N0:   45      -       5           -       15
        //  0   M3N1:   46     27       -           -       -
        //  0   M3N2:   47      -       -           -       16
        //  0   M3N3:   48     28       -           -       -
        //  0   M4N0:   49      -       6           -       -
        //  0   M4N1:   50     29       -           -       -
        //  0   M4N2:   51      -       -           5       -
        //  0   M4N3:   52     30       -           -       -
        //  0   M5N0:   53      -       7           -       -
        //  0   M5N1:   54     31       -           -       -
        //  0   M5N2:   55      -       -           6       -
        //  0   M5N3:   56     32       -           -       -
        //  0   M6N0:   57      -       8           -       -
        //  0   M6N1:   58      1       -           -       -
        //  0   M6N2:   59      -       -           7       -
        //  0   M6N3:   60      2       -           -       -
        //  0   M7N0:   61      -       -           -       -
        //  0   M7N1:   62      3       -           -       -
        //  0   M7N2:   63      -       -           8       -
        //  0   M7N3:   64      4       -           -       -

        _Pragma("unroll") for(int kIter = 0; kIter < KIterPerWarp; kIter++)
        {
            _Pragma("unroll") for(int mIter = 0; mIter < MIterPerWarp; mIter++)
            {
                index_t dsread_perM  = 0;
                index_t dswrite_perM = 0;
                index_t load_perM    = 0;

                // Calculate ds_read number per M
                dsread_perM = dsread_per_wg;

                // Calculate buffer_load number per M
                if(mIter < HalfMIter)
                {
                    load_perM =
                        ((Aload_num_perK - (MIterPerWarp - 1 - mIter) * Aload_rep) > 0 ? Aload_rep
                                                                                       : 0) +
                        ((Bload_num_perK - (HalfMIter - 1 - mIter) * Bload_rep) > 0 ? Bload_rep
                                                                                    : 0);
                }
                else
                {
                    load_perM = (Aload_num_perK - (MIterPerWarp - 1 - mIter) * Aload_rep) > 0
                                    ? Aload_rep
                                    : 0;
                }
                if((kIter % KPerScaleLoad == 0) && (mIter == 0))
                {
                    load_perM = load_perM + 1;
                }
                SchedulerPerM(dsread_perM, dswrite_perM, load_perM);
            }
        }
        // Add Aload when Aload data > needed
        if(Aload_num_perK == 0)
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
        __builtin_amdgcn_sched_barrier(0);
    }

    CK_TILE_HOST_DEVICE static constexpr auto Last2ndHotLoopScheduler()
    {
        _Pragma("unroll") for(int kIter = 0; kIter < KIterPerWarp; kIter++)
        {
            _Pragma("unroll") for(int mIter = 0; mIter < MIterPerWarp; mIter++)
            {
                index_t dsread_perM  = 0;
                index_t dswrite_perM = 0;
                index_t load_perM    = 0;

                // Calculate ds_read number per M
                dsread_perM = dsread_per_wg;

                // Calculate buffer_load number per M
                if(mIter < HalfMIter)
                {
                    load_perM =
                        ((Bload_num_perK - (HalfMIter - 1 - mIter) * Bload_rep) > 0 ? Bload_rep
                                                                                    : 0);
                }
                SchedulerPerM(dsread_perM, dswrite_perM, load_perM);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
    }

    CK_TILE_HOST_DEVICE static constexpr auto LastHotLoopScheduler()
    {
        _Pragma("unroll") for(int kIter = 0; kIter < KIterPerWarp; kIter++)
        {
            _Pragma("unroll") for(int mIter = 0; mIter < MIterPerWarp; mIter++)
            {
                index_t dsread_perM  = 0;
                index_t dswrite_perM = 0;
                index_t load_perM    = 0;

                // Calculate ds_read number per M
                if((kIter * MIterPerWarp + mIter) < (KIterPerWarp * MIterPerWarp - m_preload))
                    dsread_perM = dsread_per_wg;

                SchedulerPerM(dsread_perM, dswrite_perM, load_perM);
            }
        }
        // __builtin_amdgcn_sched_barrier(0);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetADramTileDistribution()
    {
        return PipelinePolicy::template MakeADramTileDistribution<Problem>();
    }

    template <typename ADramBlockWindowTmp,
              typename AElementFunction,
              typename BFlatBlockWindowTmp,
              typename DequantBFlatWindow>
    CK_TILE_HOST_DEVICE auto operator()(ADramBlockWindowTmp a_copy_dram_window_,
                                        const AElementFunction& a_element_func,
                                        const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                        const DequantBFlatWindow& scale_b_flat_window,
                                        const index_t num_loop,
                                        const index_t k_padded_zeros,
                                        void* p_smem_ping,
                                        void* p_smem_pong) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}],
                      "wrong!");
        static_assert(kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto MIter_2nd_last = (MIterPerWarp >= 2) ? MIterPerWarp - 2 : MIterPerWarp - 1;
        const index_t iMWarp          = get_warp_id() / NWarp;

        using CWarpDstr   = typename WG::CWarpDstr;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        __builtin_amdgcn_sched_barrier(0);

        auto a_copy_dram_window = replace_bottom_tensor_view(
            PipelinePolicy::template TransformF16xF4_ATensorView<Problem>(
                a_copy_dram_window_.get_bottom_tensor_view()),
            a_copy_dram_window_);

        // A tile in LDS
        ADataType* p_a_lds_ping = static_cast<ADataType*>(p_smem_ping);
        ADataType* p_a_lds_pong = static_cast<ADataType*>(p_smem_pong);

        constexpr auto write_a_lds_block_desc =
            PipelinePolicy::template MakeF16xF4_WriteALdsBlockDescriptor<Problem>();
        constexpr auto read_a_lds_block_desc =
            PipelinePolicy::template MakeF16xF4_ReadALdsBlockDescriptor<Problem>();

        auto write_a_lds_block_ping =
            make_tensor_view<address_space_enum::lds>(p_a_lds_ping, write_a_lds_block_desc);
        auto write_a_lds_block_pong =
            make_tensor_view<address_space_enum::lds>(p_a_lds_pong, write_a_lds_block_desc);
        auto read_a_lds_block_ping =
            make_tensor_view<address_space_enum::lds>(p_a_lds_ping, read_a_lds_block_desc);
        auto read_a_lds_block_pong =
            make_tensor_view<address_space_enum::lds>(p_a_lds_pong, read_a_lds_block_desc);

        auto a_copy_lds_window_ping =
            make_tile_window(write_a_lds_block_ping,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             PipelinePolicy::template MakeADramTileDistribution<Problem>());
        auto a_copy_lds_window_pong =
            make_tile_window(write_a_lds_block_pong,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             PipelinePolicy::template MakeADramTileDistribution<Problem>());

        // ping-pong window for A LDS
        auto a_warp_window_ping_tmp =
            make_tile_window(read_a_lds_block_ping,
                             make_tuple(number<WG::kM>{}, number<WG::kK>{}),
                             {iMWarp * WG::kM, 0},
                             PipelinePolicy::template MakeF16xF4_ALDS_TileDistribution<Problem>());
        auto a_warp_window_pong_tmp =
            make_tile_window(read_a_lds_block_pong,
                             make_tuple(number<WG::kM>{}, number<WG::kK>{}),
                             {iMWarp * WG::kM, 0},
                             PipelinePolicy::template MakeF16xF4_ALDS_TileDistribution<Problem>());

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_ping_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows_ping;

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_pong_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows_pong;

        auto A_Lds_Stride = 8;
        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows_ping(mIter)(kIter) = a_warp_window_ping_tmp;
                a_warp_windows_pong(mIter)(kIter) = a_warp_window_pong_tmp;

                auto weight_k_idx  = kIter / number<XDL_PerWeightK>{};
                auto weight_k_rank = kIter % number<XDL_PerWeightK>{};
                move_tile_window(
                    a_warp_windows_ping(mIter)(kIter),
                    {mIter * MPerBlockPerIter,
                     weight_k_rank * A_Lds_Stride + weight_k_idx * XDL_PerWeightK * WG::kK});
                move_tile_window(
                    a_warp_windows_pong(mIter)(kIter),
                    {mIter * MPerBlockPerIter,
                     weight_k_rank * A_Lds_Stride + weight_k_idx * XDL_PerWeightK * WG::kK});
            });
        });

        // Block GEMM
        auto block_flatmm = BlockFlatmm();
        // Acc register tile
        auto c_block_tile = block_flatmm.MakeCBlockTile();

        // B flat DRAM window for load
        auto b_flat_distribution =
            PipelinePolicy::template MakeFp4BFlatDramTileDistribution<Problem>();
        auto scale_b_flat_distribution =
            PipelinePolicy::template MakeFp4ScaleBFlatDramTileDistribution<Problem>();

        auto b_flat_dram_window = make_tile_window(
            b_flat_dram_block_window_tmp.get_bottom_tensor_view(), // from kernel gemm_pad_views
            make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
            b_flat_dram_block_window_tmp.get_window_origin(),
            b_flat_distribution);

        auto scale_b_flat_dram_window = make_tile_window(
            scale_b_flat_window.get_bottom_tensor_view(), // from kernel gemm_pad_views
            make_tuple(number<flatNPerWarp>{}, number<ScaleKFlatPerWarp>{}),
            scale_b_flat_window.get_window_origin(),
            scale_b_flat_distribution);

        using MXFP4_Buffer = decltype(load_tile(b_flat_dram_window));
        // use v4i32 as the data type between basicblock to avoid unpack and repack operation.
        using V4UInt_Buffer = thread_buffer<uint32_t, XDL_PerWeightK>;
        union UnionB
        {
            V4UInt_Buffer u = 0;
            MXFP4_Buffer mxfp4;
        } ub;

        // pingpong buffer for B
        statically_indexed_array<
            statically_indexed_array<decltype(b_flat_dram_window), MXFP4KPerWarp>,
            NIterPerWarp>
            b_flat_dram_windows;
        statically_indexed_array<statically_indexed_array<V4UInt_Buffer, MXFP4KPerWarp>,
                                 NIterPerWarp>
            b_warp_tensor_ping;
        statically_indexed_array<statically_indexed_array<V4UInt_Buffer, MXFP4KPerWarp>,
                                 NIterPerWarp>
            b_warp_tensor_pong;

        statically_indexed_array<
            statically_indexed_array<decltype(scale_b_flat_dram_window), ScaleKPerWarp>,
            ScaleNPerWarp>
            scale_b_flat_dram_windows;
        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(scale_b_flat_dram_window)), ScaleKPerWarp>,
            ScaleNPerWarp>
            scale_b_warp_tensor_ping;
        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(scale_b_flat_dram_window)), ScaleKPerWarp>,
            ScaleNPerWarp>
            scale_b_warp_tensor_pong;

        using ABlockTile = decltype(load_tile(a_copy_dram_window));
        ABlockTile a_block_tile;

        enum
        {
            PrefillBeforeGemm = 1,
            PrefillAfterGemm  = 2,
            PrefillAlways     = PrefillBeforeGemm | PrefillAfterGemm,
        };
#if CKTILE_FLATMM_USE_BUFFER_LOAD_LDS
        auto prefill_lds_a_stage1 =
            [&]([[maybe_unused]] auto lds_tile_a, auto dram_tile_a, auto prefill_location) {
                // global -> lds
                if constexpr(prefill_location & PrefillAfterGemm)
                    async_load_tile(lds_tile_a, dram_tile_a);
            };
        auto prefill_lds_a_stage2 = [&](auto lds_tile_a) {
            // async_load_fence();
            // __builtin_amdgcn_s_waitcnt(0x03fc);
            // data has been stored in lds, no need more operation.
            static_assert(std::is_same_v<AElementFunction, identity>,
                          "buffer_load_lds don't support element func fot A before mfma");
        };
#else
        auto prefill_lds_a_stage1 =
            [&]([[maybe_unused]] auto lds_tile_a, auto dram_tile_a, auto prefill_location) {
                // global -> vgpr
                if constexpr(prefill_location & PrefillBeforeGemm)
                    a_block_tile = load_tile(dram_tile_a);
            };
        auto prefill_lds_a_stage2 = [&]([[maybe_unused]] auto lds_tile_a) {
            // vgpr -> lds
            auto a_block_tile_transformed = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(lds_tile_a, a_block_tile_transformed);
        };
#endif

        // HEAD
        // Prefetch A0
        prefill_lds_a_stage1(a_copy_lds_window_ping, a_copy_dram_window, number<PrefillAlways>{});

        // move A window to next k
        move_tile_window(a_copy_dram_window, {0, kKPerBlock});

        // prefetch B
        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, MXFP4KPerWarp, 1>{}([&](auto kIter) {
                if constexpr(nIter % XDL_PerScaleN == 0 && kIter % MXFP4K_PerScaleK == 0)
                {
                    auto scale_n_iter = nIter / number<XDL_PerScaleN>{};
                    auto scale_k_iter = kIter / number<MXFP4K_PerScaleK>{};

                    scale_b_flat_dram_windows(scale_n_iter)(scale_k_iter) =
                        scale_b_flat_dram_window;
                    move_tile_window(
                        scale_b_flat_dram_windows(scale_n_iter)(scale_k_iter),
                        {scale_n_iter * NFlatPerBlockPerIter, scale_k_iter * ScaleKFlatPerWarp});
                    scale_b_warp_tensor_ping(scale_n_iter)(scale_k_iter) =
                        load_tile(scale_b_flat_dram_windows(scale_n_iter)(scale_k_iter));
                }
                auto packed_n_idx  = nIter / number<ContinuousScaleNPerThread>{};
                auto packed_n_rank = nIter % number<ContinuousScaleNPerThread>{};

                b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;
                move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                 {packed_n_idx * ContinuousScaleNPerThread * NFlatPerBlockPerIter +
                                      packed_n_rank,
                                  kIter * KFlatPerBlockPerIter});

                ub.mxfp4                         = load_tile(b_flat_dram_windows(nIter)(kIter));
                b_warp_tensor_ping(nIter)(kIter) = ub.u;
            });
        });
        // move B window to next flat K
        move_tile_window(b_flat_dram_window, {0, MXFP4KPerWarp * KFlatPerBlockPerIter});
        move_tile_window(scale_b_flat_dram_window, {0, ScaleKPerWarp * ScaleKFlatPerWarp});

        prefill_lds_a_stage2(a_copy_lds_window_ping);

        __builtin_amdgcn_sched_barrier(0);

        // Prefetch A1
        prefill_lds_a_stage1(a_copy_lds_window_pong, a_copy_dram_window, number<PrefillAlways>{});
        // move A window to next k
        move_tile_window(a_copy_dram_window, {0, kKPerBlock});

        // initialize C
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        __builtin_amdgcn_s_waitcnt(Bload_total_num);
        block_sync_lds();

        // preload A00,A10... from lds
        statically_indexed_array<decltype(load_tile(a_warp_windows_ping(number<0>{})(number<0>{}))),
                                 m_preload>
            a_warp_tensor;

        static_for<0, m_preload, 1>{}([&](auto loadIter) {
            constexpr auto mIter = loadIter % MIterPerWarp;
            constexpr auto kIter = loadIter / MIterPerWarp;
            a_warp_tensor(loadIter) =
                load_tile(a_warp_windows_ping(number<mIter>{})(number<kIter>{}));
        });
        __builtin_amdgcn_sched_barrier(0);

        statically_indexed_array<typename WG::BWarpTensor, NIterPerWarp> dequant_B_n;

        auto dequant_mxfp4 = [&](const auto& quant_weight_tensor,
                                 const auto& scale_tensor,
                                 auto xdl_nIter,
                                 auto xdl_kIter) {
            auto quant_idx_k = xdl_kIter % number<XDL_PerWeightK>{};

            auto scale_idx_n  = xdl_nIter % number<XDL_PerScaleN>{};
            auto scale_idx_k  = (xdl_kIter % number<XDL_PerScaleK>{}) / number<XDL_PerWeightK>{};
            auto scale_offset = scale_idx_n + scale_idx_k * number<XDL_PerScaleN>{};

            auto scale = scale_tensor.get_thread_buffer()[scale_offset];

            constexpr int ScalarCnt      = WG::BWarpTensor::get_thread_buffer_size();
            constexpr int PackedCnt      = ScalarCnt / MXFP4PackedSize;
            constexpr int float_mantissa = 23;

            uint32_t uscale = uint32_t(scale.data) << float_mantissa;

            using ComputeV2Type =
                std::conditional_t<std::is_same_v<ComputeType, half_t>, fp16x2_t, bf16x2_t>;

#if defined(__gfx950__)
            auto pk_mxfp4x4_to_compute_v2 = [](auto pk_mxfp4x4, float fscale, auto byte_idx) {
                if constexpr(std::is_same_v<ComputeType, half_t>)
                {
                    return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(
                        pk_mxfp4x4, fscale, int(byte_idx));
                }
                else if constexpr(std::is_same_v<ComputeType, bf16_t>)
                {
                    return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(
                        pk_mxfp4x4, fscale, int(byte_idx));
                }
                else
                {
                    static_assert(sizeof(pk_mxfp4x4) == 0, "unsupported compute type");
                }
            };
            static_for<0, PackedCnt, 1>{}([&](auto i) {
                dequant_B_n[xdl_nIter].get_thread_buffer().template set_as<ComputeV2Type>(
                    i,
                    pk_mxfp4x4_to_compute_v2(
                        quant_weight_tensor[quant_idx_k], bit_cast<float>(uscale), i));
            });
#else
            auto pk_mxfp4_to_compute_v2 = [](auto pk_mxfp4, float fscale) {
                if constexpr(std::is_same_v<ComputeType, half_t>)
                {
                    return pk_fp4_to_fp16x2(pk_mxfp4, fscale);
                }
                else if constexpr(std::is_same_v<ComputeType, bf16_t>)
                {
                    return pk_fp4_to_bf16x2(pk_mxfp4, fscale);
                }
                else
                {
                    static_assert(sizeof(pk_mxfp4) == 0, "unsupported compute type");
                }
            };
            static_for<0, PackedCnt, 1>{}([&](auto i) {
                dequant_B_n[xdl_nIter].get_thread_buffer().template set_as<ComputeV2Type>(
                    i,
                    pk_mxfp4_to_compute_v2(
                        bit_cast<thread_buffer<pk_fp4_t, 4>>(quant_weight_tensor[quant_idx_k])
                            .at(i),
                        bit_cast<float>(uscale)));
            });
#endif
        };

        // MAIN LOOP
        index_t iCounter = (num_loop - 1) / 2;
        while(iCounter > 0)
        {
            // prefetch B(2i+1)
            static_for<0, MXFP4KPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    if constexpr(nIter % XDL_PerScaleN == 0 && kIter % MXFP4K_PerScaleK == 0)
                    {
                        auto scale_n_iter = nIter / number<XDL_PerScaleN>{};
                        auto scale_k_iter = kIter / number<MXFP4K_PerScaleK>{};

                        scale_b_flat_dram_windows(scale_n_iter)(scale_k_iter) =
                            scale_b_flat_dram_window;

                        move_tile_window(scale_b_flat_dram_windows(scale_n_iter)(scale_k_iter),
                                         {scale_n_iter * NFlatPerBlockPerIter,
                                          scale_k_iter * ScaleKFlatPerWarp});

                        scale_b_warp_tensor_pong(scale_n_iter)(scale_k_iter) =
                            load_tile(scale_b_flat_dram_windows(scale_n_iter)(scale_k_iter));
                    }

                    auto packed_n_idx  = nIter / number<ContinuousScaleNPerThread>{};
                    auto packed_n_rank = nIter % number<ContinuousScaleNPerThread>{};

                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(
                        b_flat_dram_windows(nIter)(kIter),
                        {packed_n_idx * ContinuousScaleNPerThread * NFlatPerBlockPerIter +
                             packed_n_rank,
                         kIter * KFlatPerBlockPerIter});

                    ub.mxfp4                         = load_tile(b_flat_dram_windows(nIter)(kIter));
                    b_warp_tensor_pong(nIter)(kIter) = ub.u;
                });
            });

            // Prefill A(2i+1)
            prefill_lds_a_stage2(a_copy_lds_window_pong);

            // Prefetch A(2i+2)
            prefill_lds_a_stage1(
                a_copy_lds_window_ping, a_copy_dram_window, number<PrefillBeforeGemm>{});
            // GEMM 2i
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        if constexpr(mIter == 0)
                            dequant_mxfp4(
                                b_warp_tensor_ping(nIter)(kIter / number<XDL_PerWeightK>{}),
                                scale_b_warp_tensor_ping(nIter / number<XDL_PerScaleN>{})(
                                    kIter / number<XDL_PerScaleK>{}),
                                nIter,
                                kIter);

                        // warp GEMM
                        WG{}(c_warp_tensor, a_warp_tensor(number<AwarpIter>{}), dequant_B_n[nIter]);

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_ping(number<AmIter>{})(number<AkIter>{}));
                    }

                    // barrier
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        __builtin_amdgcn_s_waitcnt(Bload_total_num);
                        block_sync_lds();
                    }
                });
            });
            prefill_lds_a_stage1(
                a_copy_lds_window_ping, a_copy_dram_window, number<PrefillAfterGemm>{});

            // move A window to next k
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            // move B window to next flat K
            move_tile_window(b_flat_dram_window, {0, MXFP4KPerWarp * KFlatPerBlockPerIter});
            move_tile_window(scale_b_flat_dram_window, {0, ScaleKPerWarp * ScaleKFlatPerWarp});

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor(loadIter) =
                    load_tile(a_warp_windows_pong(number<mIter>{})(number<kIter>{}));
            });
            HotLoopScheduler();

            // Next K

            // prefetch B(2i+2)
            static_for<0, MXFP4KPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    if constexpr(nIter % XDL_PerScaleN == 0 && kIter % MXFP4K_PerScaleK == 0)
                    {
                        auto scale_n_iter = nIter / number<XDL_PerScaleN>{};
                        auto scale_k_iter = kIter / number<MXFP4K_PerScaleK>{};

                        scale_b_flat_dram_windows(scale_n_iter)(scale_k_iter) =
                            scale_b_flat_dram_window;

                        move_tile_window(scale_b_flat_dram_windows(scale_n_iter)(scale_k_iter),
                                         {scale_n_iter * NFlatPerBlockPerIter,
                                          scale_k_iter * ScaleKFlatPerWarp});

                        scale_b_warp_tensor_ping(scale_n_iter)(scale_k_iter) =
                            load_tile(scale_b_flat_dram_windows(scale_n_iter)(scale_k_iter));
                    }

                    auto packed_n_idx  = nIter / number<ContinuousScaleNPerThread>{};
                    auto packed_n_rank = nIter % number<ContinuousScaleNPerThread>{};

                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;
                    move_tile_window(
                        b_flat_dram_windows(nIter)(kIter),
                        {packed_n_idx * ContinuousScaleNPerThread * NFlatPerBlockPerIter +
                             packed_n_rank,
                         kIter * KFlatPerBlockPerIter});

                    ub.mxfp4                         = load_tile(b_flat_dram_windows(nIter)(kIter));
                    b_warp_tensor_ping(nIter)(kIter) = ub.u;
                });
            });

            // Prefill A(2i+2)
            prefill_lds_a_stage2(a_copy_lds_window_ping);

            // Prefetch A(2i+3)
            prefill_lds_a_stage1(
                a_copy_lds_window_pong, a_copy_dram_window, number<PrefillBeforeGemm>{});

            // GEMM 2i+1
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;
                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        if constexpr(mIter == 0)
                            dequant_mxfp4(
                                b_warp_tensor_pong(nIter)(kIter / number<XDL_PerWeightK>{}),
                                scale_b_warp_tensor_pong(nIter / number<XDL_PerScaleN>{})(
                                    kIter / number<XDL_PerScaleK>{}),
                                nIter,
                                kIter);

                        // warp GEMM
                        WG{}(c_warp_tensor, a_warp_tensor(number<AwarpIter>{}), dequant_B_n[nIter]);

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_pong(number<AmIter>{})(number<AkIter>{}));
                    }

                    // barrier
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        __builtin_amdgcn_s_waitcnt(Bload_total_num);
                        block_sync_lds();
                    }
                });
            });
            prefill_lds_a_stage1(
                a_copy_lds_window_pong, a_copy_dram_window, number<PrefillAfterGemm>{});

            // move A window to next k
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            // move B window to next flat K
            move_tile_window(b_flat_dram_window, {0, MXFP4KPerWarp * KFlatPerBlockPerIter});
            move_tile_window(scale_b_flat_dram_window, {0, ScaleKPerWarp * ScaleKFlatPerWarp});

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor(loadIter) =
                    load_tile(a_warp_windows_ping(number<mIter>{})(number<kIter>{}));
            });
            HotLoopScheduler();

            iCounter--;
        }

        // TAIL
        if constexpr(TailNum == TailNumber::Even)
        {
            // prefetch B(loopK)
            const int b_k_off = b_flat_dram_window.get_tile_distribution().calculate_index()[I1] /
                                ContinuousKPerThread / WG::kN * ContinuousKPerThread;
            static_for<0, MXFP4KPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    if constexpr(nIter % XDL_PerScaleN == 0 && kIter % MXFP4K_PerScaleK == 0)
                    {
                        auto scale_n_iter = nIter / number<XDL_PerScaleN>{};
                        auto scale_k_iter = kIter / number<MXFP4K_PerScaleK>{};

                        scale_b_flat_dram_windows(scale_n_iter)(scale_k_iter) =
                            scale_b_flat_dram_window;

                        move_tile_window(scale_b_flat_dram_windows(scale_n_iter)(scale_k_iter),
                                         {scale_n_iter * NFlatPerBlockPerIter,
                                          scale_k_iter * ScaleKFlatPerWarp});

                        scale_b_warp_tensor_pong(scale_n_iter)(scale_k_iter) =
                            load_tile(scale_b_flat_dram_windows(scale_n_iter)(scale_k_iter));
                    }
                });

                const int b_k_off_inter = kIter * kKPerBlock / MXFP4KPerWarp + b_k_off;
                if(b_k_off_inter < kKPerBlock - k_padded_zeros)
                {
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        auto packed_n_idx  = nIter / number<ContinuousScaleNPerThread>{};
                        auto packed_n_rank = nIter % number<ContinuousScaleNPerThread>{};

                        b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                        move_tile_window(
                            b_flat_dram_windows(nIter)(kIter),
                            {packed_n_idx * ContinuousScaleNPerThread * NFlatPerBlockPerIter +
                                 packed_n_rank,
                             kIter * KFlatPerBlockPerIter});

                        ub.mxfp4 = load_tile(b_flat_dram_windows(nIter)(kIter));
                        b_warp_tensor_pong(nIter)(kIter) = ub.u;
                    });
                }
            });

            // Prefill A(loopK)
            prefill_lds_a_stage2(a_copy_lds_window_pong);

            // GEMM loopK-1
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        if constexpr(mIter == 0)
                            dequant_mxfp4(
                                b_warp_tensor_ping(nIter)(kIter / number<XDL_PerWeightK>{}),
                                scale_b_warp_tensor_ping(nIter / number<XDL_PerScaleN>{})(
                                    kIter / number<XDL_PerScaleK>{}),
                                nIter,
                                kIter);

                        // warp GEMM
                        WG{}(c_warp_tensor, a_warp_tensor(number<AwarpIter>{}), dequant_B_n[nIter]);

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_ping(number<AmIter>{})(number<AkIter>{}));
                    }

                    // barrier
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        __builtin_amdgcn_s_waitcnt(Bload_total_num);
                        block_sync_lds();
                    }
                });
            });

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor(loadIter) =
                    load_tile(a_warp_windows_pong(number<mIter>{})(number<kIter>{}));
            });

            __builtin_amdgcn_sched_barrier(0);
            // Last2ndHotLoopScheduler();

            // GEMM loopK
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                if(kIter * WG::kK < kKPerBlock - k_padded_zeros)
                {
                    static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                        constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                            // read C warp tensor from C block tensor
                            CWarpTensor c_warp_tensor;

                            c_warp_tensor.get_thread_buffer() =
                                c_block_tile.get_y_sliced_thread_data(
                                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                            if constexpr(mIter == 0)
                                dequant_mxfp4(
                                    b_warp_tensor_pong(nIter)(kIter / number<XDL_PerWeightK>{}),
                                    scale_b_warp_tensor_pong(nIter / number<XDL_PerScaleN>{})(
                                        kIter / number<XDL_PerScaleK>{}),
                                    nIter,
                                    kIter);

                            // warp GEMM
                            WG{}(c_warp_tensor,
                                 a_warp_tensor(number<AwarpIter>{}),
                                 dequant_B_n[nIter]);

                            // write C warp tensor into C block tensor
                            c_block_tile.set_y_sliced_thread_data(
                                merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                c_warp_tensor.get_thread_buffer());
                        });
                        if constexpr((kIter * MIterPerWarp + mIter) <
                                     (KIterPerWarp * MIterPerWarp - m_preload))
                        {
                            constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                            constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                            a_warp_tensor(number<AwarpIter>{}) =
                                load_tile(a_warp_windows_pong(number<AmIter>{})(number<AkIter>{}));
                        }
                        // barrier
                        // if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                        // {
                        //     block_sync_lds();
                        // }
                    });
                }
            });
            LastHotLoopScheduler();
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            // GEMM loopK
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        if constexpr(mIter == 0)
                            dequant_mxfp4(
                                b_warp_tensor_ping(nIter)(kIter / number<XDL_PerWeightK>{}),
                                scale_b_warp_tensor_ping(nIter / number<XDL_PerScaleN>{})(
                                    kIter / number<XDL_PerScaleK>{}),
                                nIter,
                                kIter);
                        // warp GEMM
                        WG{}(c_warp_tensor, a_warp_tensor(number<AwarpIter>{}), dequant_B_n[nIter]);

                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                    // preload next A from lds
                    if constexpr((kIter * MIterPerWarp + mIter) <
                                 (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor(number<AwarpIter>{}) =
                            load_tile(a_warp_windows_ping(number<AmIter>{})(number<AkIter>{}));
                    }

                    // barrier
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        __builtin_amdgcn_s_waitcnt(Bload_total_num);
                        block_sync_lds();
                    }
                });
            });
            LastHotLoopScheduler();
        }

        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp,
              typename BFlatBlockWindowTmp,
              typename DequantBFlatWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   const DequantBFlatWindow& scale_b_flat_window,
                                   const index_t num_loop,
                                   const index_t k_padded_zeros,
                                   void* p_smem_ping,
                                   void* p_smem_pong) const
    {
        return operator()(a_dram_block_window_tmp,
                          identity{},
                          b_flat_dram_block_window_tmp,
                          scale_b_flat_window,
                          num_loop,
                          k_padded_zeros,
                          p_smem_ping,
                          p_smem_pong);
    }

    template <typename ADramBlockWindowTmp,
              typename BFlatBlockWindowTmp,
              typename DequantBFlatWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   const DequantBFlatWindow& scale_b_flat_window,
                                   const index_t num_loop,
                                   void* p_smem_ping,
                                   void* p_smem_pong) const
    {
        return operator()(a_dram_block_window_tmp,
                          identity{},
                          b_flat_dram_block_window_tmp,
                          scale_b_flat_window,
                          num_loop,
                          0,
                          p_smem_ping,
                          p_smem_pong);
    }
};

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_      = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                      = true,
          TailNumber TailNum_                   = TailNumber::Full,
          amd_buffer_coherence_enum BMemNTType_ = amd_buffer_coherence_enum::coherence_default,
          bool BPreShufflePermute_              = false,
          typename ComputeDataType_             = ADataType_>
struct F8xMXF4FlatmmPipelineProblem : FlatmmPipelineProblem<ADataType_,
                                                            BDataType_,
                                                            CDataType_,
                                                            BlockGemmShape_,
                                                            Traits_,
                                                            Scheduler_,
                                                            HasHotLoop_,
                                                            TailNum_,
                                                            BMemNTType_,
                                                            BPreShufflePermute_,
                                                            ComputeDataType_>
{
    using BlockGemmShape = BlockGemmShape_;

    // using QuantType = BDataType_;

    static constexpr index_t flatNPerWarp = BlockGemmShape::flatNPerWarp;

    static constexpr int ScaleGranularityK = 32;

    static constexpr int ContinuousKPerThread = 32; // it's fixed for fp4
    static constexpr int MXdlPack             = 2;  // it's fixed for fp4
    static constexpr int NXdlPack             = 2;  // it's fixed for fp4
    static constexpr int KXdlPack             = 2;
    // static constexpr index_t flatKPerWarp = BlockGemmShape::flatKPerWarp * KXdlPack;
    static constexpr index_t flatKPerWarp = get_warp_size() * ContinuousKPerThread;
};

template <typename Problem, typename PipelinePolicy = F8xMXF4FlatmmPipelineAgBgCrPolicy>
struct F8xMXF4FlatmmPipelineAGmemBGmemCRegV1
    : FlatmmPipelineAGmemBGmemCRegV1<Problem, PipelinePolicy>
{
    using Underlying = FlatmmPipelineAGmemBGmemCRegV1<Problem, PipelinePolicy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>; // TileFlatmmShape

    using ComputeType = ADataType;
    static_assert(sizeof(ADataType) >= sizeof(BDataType));

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockFlatmm =
        remove_cvref_t<decltype(PipelinePolicy::template GetBlockFlatmm<Problem>())>;

    static constexpr auto config =
        BlockFlatmm::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    static constexpr index_t DsWritePreIssue = 3; // default 2, ds write at MIter - 2
    static constexpr index_t DsReadPreload   = 4; // default 4 for MXFP4 (MXdlPack * KXdlPack)

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t WaveSize  = get_warp_size();

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr index_t flatKPerWarp = Problem::flatKPerWarp;
    static constexpr index_t flatNPerWarp = Problem::flatNPerWarp;

    static constexpr index_t GetVectorSizeA() { return 32; } /* fixed for fp4 shuffle layout*/
    static constexpr index_t GetVectorSizeB() { return 32; } /* fixed for fp4 shuffle layout*/
    static constexpr index_t GetVectorSizeC() { return Problem::VectorSizeC; }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    // static constexpr index_t kLdsAlignmentInBytes = 16;
    static constexpr index_t NumWaveGroups    = Problem::NumWaveGroups;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    static constexpr auto I0   = number<0>();
    static constexpr auto I1   = number<1>();
    static constexpr auto I2   = number<2>();
    static constexpr auto idxM = I0;
    static constexpr auto idxN = I1;
    static constexpr auto idxK = I2;
    using BlockTile            = remove_cvref_t<typename BlockGemmShape::BlockTile>;
    using BlockWarps           = remove_cvref_t<typename BlockGemmShape::BlockWarps>;
    using WarpTile             = remove_cvref_t<typename BlockGemmShape::WarpTile>;

    static constexpr index_t MWarp = config.template at<1>();
    static constexpr index_t NWarp = config.template at<2>();

    static constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
    static constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);
    static constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

    static constexpr index_t KFlatPerBlockPerIter = flatKPerWarp;
    static constexpr index_t NFlatPerBlockPerIter = flatNPerWarp;

    static constexpr index_t MPerBlockPerIter = kMPerBlock / MIterPerWarp;
    static constexpr index_t KPerBlockPerIter = kKPerBlock / KIterPerWarp;

    static constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
    static constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;

    static constexpr index_t MXdlPack          = Problem::MXdlPack;
    static constexpr index_t NXdlPack          = Problem::NXdlPack;
    static constexpr index_t KXdlPack          = Problem::KXdlPack;
    static constexpr index_t ScaleGranularityK = Problem::ScaleGranularityK;

    static constexpr index_t AK1 =
        Problem::VectorLoadSize / sizeof(ADataType) * APackedSize; // 16 / 1 = 16
    static constexpr index_t BK1 =
        Problem::VectorLoadSize / sizeof(BDataType) * BPackedSize; // 16 / 1 * 2 = 32

    static constexpr index_t m_preload = (MIterPerWarp * KIterPerWarp >= DsReadPreload)
                                             ? DsReadPreload
                                             : MIterPerWarp * KIterPerWarp;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;

    static constexpr index_t mfma_per_wg = 1; // 950 only

    static constexpr index_t dsread_per_wg =
        WG::kM * WG::kK / AK1 / WaveSize;                     // 16 * 128 / 16 / 64 = 2
    static_assert((WG::kM * WG::kK) % (AK1 * WaveSize) == 0); // 16 * 128 %  16 * 64

    static constexpr index_t dsread_num_perK  = dsread_per_wg * MIterPerWarp;
    static constexpr index_t dswrite_num_perK = dsread_num_perK / NWarp;
    static constexpr index_t dswrite_rep    = (dswrite_num_perK + MIterPerWarp - 1) / MIterPerWarp;
    static constexpr index_t Aload_num_perK = dswrite_num_perK;
    static constexpr index_t Aload_rep      = dswrite_rep;

    static constexpr index_t Bload_num_perK = kNPerBlock * WG::kK / NWarp / BK1 / WaveSize;
    static constexpr index_t Bload_num      = Bload_num_perK * KIterPerWarp;
    static constexpr index_t ScaleBload_num =
        kNPerBlock * kKPerBlock / NWarp / ScaleGranularityK / NXdlPack / KXdlPack / WaveSize;
    static constexpr index_t ScaleAload_num =
        kMPerBlock * kKPerBlock / MWarp / ScaleGranularityK / MXdlPack / KXdlPack / WaveSize;

    // static constexpr index_t KPerScaleLoad = KIterPerWarp / ScaleBload_num;
    static constexpr index_t HalfMIter = (MIterPerWarp + 1) / 2;
    static constexpr index_t Bload_rep = (Bload_num_perK + HalfMIter - 1) / HalfMIter;

    static constexpr index_t mfma_perM_perK = NIterPerWarp * mfma_per_wg;
    static constexpr index_t dswrite_mIter  = (DsWritePreIssue - 1) % MIterPerWarp;
    static constexpr index_t dswrite_kIter  = (DsWritePreIssue - 1) / MIterPerWarp;

    // For the basic gemm pipelien DoubleSmemBuffer set to be false naturally.
    static constexpr bool DoubleSmemBuffer = false;

    static constexpr auto BMemNTType         = Problem::BMemNTType;
    static constexpr bool BPreShufflePermute = Problem::BPreShufflePermute;

    CK_TILE_HOST_DEVICE static constexpr auto
    SchedulerPerM(index_t dsread_perM, index_t dswrite_perM, index_t load_perM)
    {
        // Init inst order
        index_t max_data_inst   = dsread_perM > load_perM
                                      ? (dsread_perM > dswrite_perM ? dsread_perM : dswrite_perM)
                                      : (load_perM > dswrite_perM ? load_perM : dswrite_perM);
        index_t sum_data_inst   = dsread_perM + load_perM + dswrite_perM;
        index_t round_data_inst = (sum_data_inst + mfma_perM_perK - 1) / mfma_perM_perK;

        index_t inst_order[NIterPerWarp * 10];
        _Pragma("unroll") for(int idx = 0; idx < NIterPerWarp * 10; idx++) { inst_order[idx] = 0; }

        index_t index = 0;
        _Pragma("unroll") for(int j = 0; j < max_data_inst; j++)
        {
            if(dswrite_perM > j)
            {
                inst_order[index] = 1;
                index++;
            }
            if(load_perM > j)
            {
                inst_order[index] = 2;
                index++;
            }
            if(dsread_perM > j)
            {
                inst_order[index] = 3;
                index++;
            }
        }

        // Schedule IGLP
        _Pragma("unroll") for(int j = 0; j < mfma_perM_perK; j++)
        {
            index_t inst_idx = 0;
            if(j == 0)
                ;
            else if(j == 1)
                inst_idx = mfma_perM_perK == 2 ? 1 : mfma_perM_perK - 2;
            else if(j == 2)
                inst_idx = mfma_perM_perK - 1;
            else
                inst_idx = mfma_perM_perK - j;

            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

            _Pragma("unroll") for(int r = 0; r < round_data_inst; r++)
            {
                if(r % 2 == 0)
                {
                    if(inst_order[inst_idx + r * mfma_perM_perK] == 1)
                    {
                        // __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    }
                    if(inst_order[inst_idx + r * mfma_perM_perK] == 2)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    }
                    if(inst_order[inst_idx + r * mfma_perM_perK] == 3)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                }
                else
                {
                    if(inst_order[(r + 1) * mfma_perM_perK - 1 - inst_idx] == 1)
                    {
                        // __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    }
                    if(inst_order[(r + 1) * mfma_perM_perK - 1 - inst_idx] == 2)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    }
                    if(inst_order[(r + 1) * mfma_perM_perK - 1 - inst_idx] == 3)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                }
            }
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto HotLoopScheduler()
    {
        // Keypoint of pipeline optimize is workload balance in time
        // instruction schedule example(128X256X256, 1X4, 16X16X128):
        // Iter MNK     MFMA    ds_read ds_write    A_load  b_load
        // -1   M6N0:   57      -       8           -       -
        // -1   M6N1:   58      1       -           -       -
        // -1   M6N2:   59      -       -           7       -
        // -1   M6N3:   60      2       -           -       -
        // -1   M7N0:   61      -       -           -       -
        // -1   M7N1:   62      3       -           -       -
        // -1   M7N2:   63      -       -           8       -
        // -1   M7N3:   64      4       -           -       -
        //  0   M0N0K0:  1      -       -           -       1
        //  0   M0N1:    2      5       -           -       -
        //  0   M0N2:    3      -       -           -       2
        //  0   M0N3:    4      6       -           -       -
        //  0   M1N0:    5      -       -           -       3
        //  0   M1N1:    6      7       -           -       -
        //  0   M1N2:    7      -       -           -       4
        //  0   M1N3:    8      8       -           -       -
        //  0   M2N0:    9      -       -           -       5
        //  0   M2N1:   10      9       -           -       -
        //  0   M2N2:   11      -       -           -       6
        //  0   M2N3:   12     10       -           -       -
        //  0   M3N0:   13      -       1           -       7
        //  0   M3N1:   14     11       -           -       -
        //  0   M3N2:   15      -       -           -       8
        //  0   M3N3:   16     12       -           -       -
        //  0   M4N0:   17      -       2           -       -
        //  0   M4N1:   18     13       -           -       -
        //  0   M4N2:   19      -       -           1       -
        //  0   M4N3:   20     14       -           -       -
        //  0   M5N0:   21      -       3           -       -
        //  0   M5N1:   22     15       -           -       -
        //  0   M5N2:   23      -       -           2       -
        //  0   M5N3:   24     16       -           -       -
        //  0   M6N0:   25      -       4           -       -
        //  0   M6N1:   26     17       -           -       -
        //  0   M6N2:   27      -       -           3       -
        //  0   M6N3:   28     18       -           -       -
        //  0   M7N0:   29      -       -           -       -
        //  0   M7N1:   30     19       -           -       -
        //  0   M7N2:   31      -       -           4       -
        //  0   M7N3:   32     20       -           -       -
        //  0   M0N0K1: 33      -       -           -       9
        //  0   M0N1:   34     21       -           -       -
        //  0   M0N2:   35      -       -           -       10
        //  0   M0N3:   36     22       -           -       -
        //  0   M1N0:   37      -       -           -       11
        //  0   M1N1:   38     23       -           -       -
        //  0   M1N2:   39      -       -           -       12
        //  0   M1N3:   40     24       -           -       -
        //  0   M2N0:   41      -       -           -       13
        //  0   M2N1:   42     25       -           -       -
        //  0   M2N2:   43      -       -           -       14
        //  0   M2N3:   44     26       -           -       -
        //  0   M3N0:   45      -       5           -       15
        //  0   M3N1:   46     27       -           -       -
        //  0   M3N2:   47      -       -           -       16
        //  0   M3N3:   48     28       -           -       -
        //  0   M4N0:   49      -       6           -       -
        //  0   M4N1:   50     29       -           -       -
        //  0   M4N2:   51      -       -           5       -
        //  0   M4N3:   52     30       -           -       -
        //  0   M5N0:   53      -       7           -       -
        //  0   M5N1:   54     31       -           -       -
        //  0   M5N2:   55      -       -           6       -
        //  0   M5N3:   56     32       -           -       -
        //  0   M6N0:   57      -       8           -       -
        //  0   M6N1:   58      1       -           -       -
        //  0   M6N2:   59      -       -           7       -
        //  0   M6N3:   60      2       -           -       -
        //  0   M7N0:   61      -       -           -       -
        //  0   M7N1:   62      3       -           -       -
        //  0   M7N2:   63      -       -           8       -
        //  0   M7N3:   64      4       -           -       -

        _Pragma("unroll") for(int kIter = 0; kIter < KIterPerWarp; kIter++)
        {
            _Pragma("unroll") for(int mIter = 0; mIter < MIterPerWarp; mIter++)
            {
                index_t dsread_perM  = 0;
                index_t dswrite_perM = 0;
                index_t load_perM    = 0;

                // Calculate ds_read number per M
                dsread_perM = dsread_per_wg;

                // Calculate ds_write number per M
                if(mIter == 0)
                {
                    dswrite_perM =
                        (dswrite_num_perK - (MIterPerWarp - DsWritePreIssue) * dswrite_rep) > 0
                            ? dswrite_num_perK - (MIterPerWarp - DsWritePreIssue) * dswrite_rep
                            : 0;
                }
                else if(mIter >= MIterPerWarp - DsWritePreIssue + 1)
                {
                    dswrite_perM = 0;
                }
                else
                {
                    dswrite_perM = (dswrite_num_perK -
                                    (MIterPerWarp - DsWritePreIssue - mIter) * dswrite_rep) > 0
                                       ? dswrite_rep
                                       : 0;
                }
                // Add ds write when ds write data > needed
                if(dswrite_num_perK == 0 && kIter == (KIterPerWarp - 1 - dswrite_kIter))
                {
                    if(mIter == MIterPerWarp - 1 - dswrite_mIter)
                        dswrite_perM = 1;
                }

                // Calculate buffer_load number per M
                if(mIter < HalfMIter)
                {
                    load_perM =
                        ((Aload_num_perK - (MIterPerWarp - 1 - mIter) * Aload_rep) > 0 ? Aload_rep
                                                                                       : 0) +
                        ((Bload_num_perK - (HalfMIter - 1 - mIter) * Bload_rep) > 0 ? Bload_rep
                                                                                    : 0);
                }
                else
                {
                    load_perM = (Aload_num_perK - (MIterPerWarp - 1 - mIter) * Aload_rep) > 0
                                    ? Aload_rep
                                    : 0;
                }
                // if((kIter % KPerScaleLoad == 0) && (mIter == 0))
                // {
                //     load_perM = load_perM + 1;
                // }
                SchedulerPerM(dsread_perM, dswrite_perM, load_perM);
            }
        }
        // Add Aload when Aload data > needed
        if(Aload_num_perK == 0)
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
        __builtin_amdgcn_sched_barrier(0);
    }

    CK_TILE_HOST_DEVICE static constexpr auto Last2ndHotLoopScheduler()
    {
        _Pragma("unroll") for(int kIter = 0; kIter < KIterPerWarp; kIter++)
        {
            _Pragma("unroll") for(int mIter = 0; mIter < MIterPerWarp; mIter++)
            {
                index_t dsread_perM  = 0;
                index_t dswrite_perM = 0;
                index_t load_perM    = 0;

                // Calculate ds_read number per M
                dsread_perM = dsread_per_wg;

                // Calculate ds_write number per M
                if(mIter == 0)
                {
                    dswrite_perM =
                        (dswrite_num_perK - (MIterPerWarp - DsWritePreIssue) * dswrite_rep) > 0
                            ? dswrite_num_perK - (MIterPerWarp - DsWritePreIssue) * dswrite_rep
                            : 0;
                }
                else if(mIter >= MIterPerWarp - DsWritePreIssue + 1)
                {
                    dswrite_perM = 0;
                }
                else
                {
                    dswrite_perM = (dswrite_num_perK -
                                    (MIterPerWarp - DsWritePreIssue - mIter) * dswrite_rep) > 0
                                       ? dswrite_rep
                                       : 0;
                }
                // Add ds write when ds write data > needed
                if(dswrite_num_perK == 0 && kIter == (KIterPerWarp - 1 - dswrite_kIter))
                {
                    if(mIter == MIterPerWarp - 1 - dswrite_mIter)
                        dswrite_perM = 1;
                }

                // Calculate buffer_load number per M
                if(mIter < HalfMIter)
                {
                    load_perM =
                        ((Bload_num_perK - (HalfMIter - 1 - mIter) * Bload_rep) > 0 ? Bload_rep
                                                                                    : 0);
                }
                SchedulerPerM(dsread_perM, dswrite_perM, load_perM);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
    }

    CK_TILE_HOST_DEVICE static constexpr auto LastHotLoopScheduler()
    {
        _Pragma("unroll") for(int kIter = 0; kIter < KIterPerWarp; kIter++)
        {
            _Pragma("unroll") for(int mIter = 0; mIter < MIterPerWarp; mIter++)
            {
                index_t dsread_perM  = 0;
                index_t dswrite_perM = 0;
                index_t load_perM    = 0;

                // Calculate ds_read number per M
                if((kIter * MIterPerWarp + mIter) < (KIterPerWarp * MIterPerWarp - m_preload))
                    dsread_perM = dsread_per_wg;

                SchedulerPerM(dsread_perM, dswrite_perM, load_perM);
            }
        }
        // __builtin_amdgcn_sched_barrier(0);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetADramTileDistribution()
    {
        return PipelinePolicy::template MakeADramTileDistribution<Problem>();
    }

    template <typename ADramBlockWindowTmp,
              typename BFlatBlockWindowTmp,
              typename ScaleADramBlockWindowTmp,
              typename ScaleBDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_copy_dram_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   const ScaleADramBlockWindowTmp& scale_a_window,
                                   const ScaleBDramBlockWindowTmp& scale_b_window,
                                   index_t num_loop,
                                   void* __restrict__ p_smem_ping,
                                   void* __restrict__ p_smem_pong) const
    {
#ifndef __gfx950__
        static_assert(false, "Only gfx950 is supported for MXFP4 flatmm pipeline now.");
#endif
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}],
                      "wrong!");
        static_assert(kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // constexpr auto MIter_2nd_last = max(0, MIterPerWarp - 2);
        static_assert(NWarp == 4);

        using CWarpDstr   = typename WG::CWarpDstr;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        auto a_dram_window = replace_bottom_tensor_view(
            PipelinePolicy::template MakeMXFP4_AAsyncLoadDramDescriptor<Problem>(
                // PipelinePolicy::template Make_F8AAsyncLoadDramDescriptor<Problem>(
                a_copy_dram_window_tmp.get_bottom_tensor_view()),
            a_copy_dram_window_tmp);

        __builtin_amdgcn_sched_barrier(0);

        // A tile in LDS
        ADataType* p_a_lds_ping = static_cast<ADataType*>(p_smem_ping);
        ADataType* p_a_lds_pong = static_cast<ADataType*>(p_smem_pong);

        constexpr auto a_lds_block_desc =
            PipelinePolicy::template MakeMXFP4_ALdsBlockDescriptor<Problem>();
        // PipelinePolicy::template MakeF8_ReadALdsBlockDescriptor<Problem>();

        constexpr auto a_load_lds_block_desc =
            PipelinePolicy::template MakeMXFP4_ALdsBlockDescriptor<Problem>();
        // PipelinePolicy::template MakeF8_WriteALdsBlockDescriptor<Problem>();

        auto a_lds_block_ping_load =
            make_tensor_view<address_space_enum::lds>(p_a_lds_ping, a_load_lds_block_desc);
        auto a_lds_block_pong_load =
            make_tensor_view<address_space_enum::lds>(p_a_lds_pong, a_load_lds_block_desc);

        auto a_lds_block_ping =
            make_tensor_view<address_space_enum::lds>(p_a_lds_ping, a_lds_block_desc);
        auto a_lds_block_pong =
            make_tensor_view<address_space_enum::lds>(p_a_lds_pong, a_lds_block_desc);

        auto a_store_lds_window_ping = make_tile_window(
            a_lds_block_ping_load, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});
        auto a_store_lds_window_pong = make_tile_window(
            a_lds_block_pong_load, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // ping-pong window for A LDS
        auto a_warp_window_ping =
            make_tile_window(a_lds_block_ping,
                             make_tuple(number<WG::kM>{}, number<WG::kK>{}),
                             {0, 0},
                             PipelinePolicy::template MakeMXF4_ALDS_TileDistribution<Problem>());
        auto a_warp_window_pong =
            make_tile_window(a_lds_block_pong,
                             make_tuple(number<WG::kM>{}, number<WG::kK>{}),
                             {0, 0},
                             PipelinePolicy::template MakeMXF4_ALDS_TileDistribution<Problem>());

        // Block GEMM
        auto block_flatmm = BlockFlatmm();
        // Acc register tile
        auto c_block_tile = block_flatmm.MakeCBlockTile();

        // B flat DRAM window for load

        // pingpong buffer for B
        auto b_flat_dram_windows = generate_tuple(
            [&](auto nIter) {
                constexpr auto packed_n_idx  = nIter / number<NXdlPack>{};
                constexpr auto packed_n_rank = nIter % number<NXdlPack>{};
                auto window_i                = make_tile_window(
                    b_flat_dram_block_window_tmp.get_bottom_tensor_view(),
                    make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
                    b_flat_dram_block_window_tmp.get_window_origin(),
                    PipelinePolicy::template MakeMXFP4_BFlatDramTileDistribution<Problem>());
                move_tile_window(
                    window_i,
                    {number<packed_n_idx * NXdlPack * NFlatPerBlockPerIter + packed_n_rank>{},
                     number<0>{}});
                return window_i;
            },
            number<NIterPerWarp>{});
        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(b_flat_dram_windows(I0))), KIterPerWarp>,
            NIterPerWarp>
            b_warp_tensor_ping, b_warp_tensor_pong;

        // pingpong buffer for Scale A and Scale B
        auto scale_a_dram_window = make_tile_window(
            scale_a_window.get_bottom_tensor_view(),
            make_tuple(number<MWarp * WG::kM>{}, number<64 / WG::kM>{}),
            scale_a_window.get_window_origin(),
            PipelinePolicy::template MakeMXFP4_ScaleA_FlatDramTileDistribution<Problem>());

        auto scale_b_dram_window = make_tile_window(
            scale_b_window.get_bottom_tensor_view(),
            make_tuple(number<NWarp * WG::kN>{}, number<64 / WG::kN>{}),
            scale_b_window.get_window_origin(),
            PipelinePolicy::template MakeMXFP4_ScaleB_DramTileDistribution<Problem>());

        // ping pong buffer for scale A
        statically_indexed_array<
            statically_indexed_array<decltype(scale_a_dram_window), KIterPerWarp / KXdlPack>,
            MIterPerWarp / MXdlPack>
            scale_a_dram_windows;
        statically_indexed_array<statically_indexed_array<decltype(load_tile(scale_a_dram_window)),
                                                          KIterPerWarp / KXdlPack>,
                                 MIterPerWarp / MXdlPack>
            scale_a_tile_tensor_ping;
        statically_indexed_array<statically_indexed_array<decltype(load_tile(scale_a_dram_window)),
                                                          KIterPerWarp / KXdlPack>,
                                 MIterPerWarp / MXdlPack>
            scale_a_tile_tensor_pong;

        // ping pong buffer for scale B
        statically_indexed_array<
            statically_indexed_array<decltype(scale_b_dram_window), KIterPerWarp / KXdlPack>,
            NIterPerWarp / NXdlPack>
            scale_b_dram_windows;
        statically_indexed_array<statically_indexed_array<decltype(load_tile(scale_b_dram_window)),
                                                          KIterPerWarp / KXdlPack>,
                                 NIterPerWarp / NXdlPack>
            scale_b_tile_tensor_ping;
        statically_indexed_array<statically_indexed_array<decltype(load_tile(scale_b_dram_window)),
                                                          KIterPerWarp / KXdlPack>,
                                 NIterPerWarp / NXdlPack>
            scale_b_tile_tensor_pong;

        auto async_load_tile_ = [](auto lds, auto dram) {
            async_load_tile(lds, dram, number<-1>{}, true_type{}, false_type{});
        };

        // HEAD
        // Prefetch A0
        async_load_tile_(a_store_lds_window_ping, a_dram_window);
        move_tile_window(a_dram_window, {0, kKPerBlock});

        // prefetch B
        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_tensor_ping(nIter)(kIter) = load_tile_with_offset(
                    b_flat_dram_windows(nIter), number<kIter * KFlatPerBlockPerIter>{});
            });
            // move B window to next flat K
            move_tile_window(b_flat_dram_windows(nIter), {0, KIterPerWarp * KFlatPerBlockPerIter});
        });

        // prefetch Scale A
        static_for<0, MIterPerWarp / MXdlPack, 1>{}([&](auto mIter_pack) {
            static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                scale_a_dram_windows(mIter_pack)(kIter_pack) = scale_a_dram_window;
                move_tile_window(scale_a_dram_windows(mIter_pack)(kIter_pack),
                                 {mIter_pack * MWarp * WG::kM, kIter_pack * (64 / WG::kM)});

                scale_a_tile_tensor_ping(mIter_pack)(kIter_pack) =
                    load_tile(scale_a_dram_windows(mIter_pack)(kIter_pack));
            });
        });
        // move Scale A window to next K
        move_tile_window(scale_a_dram_window, {0, kKPerBlock / (32 * KXdlPack)});

        // prefetch Scale B
        static_for<0, NIterPerWarp / NXdlPack, 1>{}([&](auto nIter_pack) {
            static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                scale_b_dram_windows(nIter_pack)(kIter_pack) = scale_b_dram_window;
                move_tile_window(scale_b_dram_windows(nIter_pack)(kIter_pack),
                                 {nIter_pack * NWarp * WG::kN, kIter_pack * (64 / WG::kN)});

                scale_b_tile_tensor_ping(nIter_pack)(kIter_pack) =
                    load_tile(scale_b_dram_windows(nIter_pack)(kIter_pack));
            });
        });
        // move Scale B window to next K
        move_tile_window(scale_b_dram_window, {0, kKPerBlock / (32 * KXdlPack)});
        __builtin_amdgcn_sched_barrier(0);

        // Prefetch A1
        if constexpr(HasHotLoop || TailNum == TailNumber::Even)
        {
            async_load_tile_(a_store_lds_window_pong, a_dram_window);
            move_tile_window(a_dram_window, {0, kKPerBlock});
        }
        // initialize C
        clear_tile(c_block_tile);

        statically_indexed_array<decltype(load_tile(a_warp_window_pong)), m_preload> a_warp_tensor;

        // preload A00,A10... from lds
        s_waitcnt_barrier</*vmcnt*/ dswrite_num_perK>();
        static_for<0, m_preload, 1>{}([&](auto loadIter) {
            constexpr auto mIter = loadIter % MXdlPack;
            constexpr auto kIter = loadIter / MXdlPack;

            a_warp_tensor(loadIter) = load_tile_with_offset(
                a_warp_window_ping, tuple<number<mIter * WG::kM>, number<kIter * WG::kK>>{});
        });
        __builtin_amdgcn_sched_barrier(0);

        // MAIN LOOP
        auto main_body_implx2 = [&]() mutable {
            // prefetch B(2i+1)
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    b_warp_tensor_pong(nIter)(kIter) = load_tile_with_offset(
                        b_flat_dram_windows(nIter), number<kIter * KFlatPerBlockPerIter>{});
                    if constexpr(kIter == KIterPerWarp - 1)
                        move_tile_window(b_flat_dram_windows(nIter),
                                         {0, BlockGemmShape::flatKPerBlock});
                });
            });

            // prefetch Scale A and Scale B (2i+1)
            static_for<0, MIterPerWarp / MXdlPack, 1>{}([&](auto mIter_pack) {
                static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                    scale_a_dram_windows(mIter_pack)(kIter_pack) = scale_a_dram_window;
                    move_tile_window(scale_a_dram_windows(mIter_pack)(kIter_pack),
                                     {mIter_pack * MWarp * WG::kM, kIter_pack * (64 / WG::kM)});

                    scale_a_tile_tensor_pong(mIter_pack)(kIter_pack) =
                        load_tile(scale_a_dram_windows(mIter_pack)(kIter_pack));
                });
            });

            static_for<0, NIterPerWarp / NXdlPack, 1>{}([&](auto nIter_pack) {
                static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                    scale_b_dram_windows(nIter_pack)(kIter_pack) = scale_b_dram_window;
                    move_tile_window(scale_b_dram_windows(nIter_pack)(kIter_pack),
                                     {nIter_pack * NWarp * WG::kN, kIter_pack * (64 / WG::kN)});

                    scale_b_tile_tensor_pong(nIter_pack)(kIter_pack) =
                        load_tile(scale_b_dram_windows(nIter_pack)(kIter_pack));
                });
            });

            // GEMM 2i
            static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                static_for<0, MIterPerWarp / MXdlPack, 1>{}([&](auto mIter_pack) {
                    static_for<0, NIterPerWarp / NXdlPack, 1>{}([&](auto nIter_pack) {
                        static_for<0, KXdlPack, 1>{}([&](auto ikxdl) {
                            static_for<0, MXdlPack, 1>{}([&](auto imxdl) {
                                constexpr auto AwarpIter = imxdl + ikxdl * MXdlPack;
                                constexpr auto m_iter    = mIter_pack * MXdlPack + imxdl;
                                constexpr auto k_iter    = kIter_pack * KXdlPack + ikxdl;
                                static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                                    constexpr auto n_iter = nIter_pack * NXdlPack + inxdl;

                                    // read C warp tensor from C block tensor
                                    CWarpTensor c_warp_tensor;
                                    c_warp_tensor.get_thread_buffer() =
                                        c_block_tile.get_y_sliced_thread_data(
                                            merge_sequences(sequence<m_iter, n_iter>{},
                                                            c_warp_y_index_zeros),
                                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                                    //  warp GEMM
                                    WG{}.template
                                    // operator()<MXdlPack + imxdl, ikxdl * NXdlPack + inxdl>(
                                    operator()<ikxdl * MXdlPack + imxdl, ikxdl * NXdlPack + inxdl>(
                                        c_warp_tensor,
                                        a_warp_tensor(number<AwarpIter>{}),
                                        b_warp_tensor_ping(nIter_pack * number<NXdlPack>{} + inxdl)(
                                            kIter_pack * number<KXdlPack>{} + ikxdl),
                                        scale_a_tile_tensor_ping(mIter_pack)(kIter_pack)
                                            .get_thread_buffer()[0],
                                        scale_b_tile_tensor_ping(nIter_pack)(kIter_pack)
                                            .get_thread_buffer()[0]);

                                    // write C warp tensor into C block tensor
                                    c_block_tile.set_y_sliced_thread_data(
                                        merge_sequences(sequence<m_iter, n_iter>{},
                                                        c_warp_y_index_zeros),
                                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                        c_warp_tensor.get_thread_buffer());
                                });
                                // preload next A from lds
                                constexpr auto addr =
                                    m_iter % 2 + k_iter * 2 + m_iter / 2 * 4 + m_preload;
                                if constexpr(addr < (KIterPerWarp * MIterPerWarp) &&
                                             (nIter_pack == NIterPerWarp / NXdlPack - 1))
                                {
                                    constexpr auto AmIter              = addr % 2 + addr / 4 * 2;
                                    constexpr auto AkIter              = addr / 2 % 2;
                                    a_warp_tensor(number<AwarpIter>{}) = load_tile_with_offset(
                                        a_warp_window_ping,
                                        tuple<number<AmIter * WG::kM>, number<AkIter * WG::kK>>{});
                                }
                            });
                        });
                    });
                });
            });
            // barrier as ds_load A(2i) and buffer_load_lds A(2i + 1) finished
            s_waitcnt< // vmcnt
                Bload_num + ScaleAload_num + ScaleBload_num>();
            block_sync_lds();

            // Prefetch A(2i+2)
            async_load_tile_(a_store_lds_window_ping, a_dram_window);
            move_tile_window(a_dram_window, {0, kKPerBlock});

            // move B window to next flat K
            move_tile_window(scale_a_dram_window, {0, kKPerBlock / (32 * KXdlPack)});
            move_tile_window(scale_b_dram_window, {0, kKPerBlock / (32 * KXdlPack)});

            // preload A(2i+1)
            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter    = loadIter % MXdlPack;
                constexpr auto kIter    = loadIter / MXdlPack;
                a_warp_tensor(loadIter) = load_tile_with_offset(
                    a_warp_window_pong, tuple<number<mIter * WG::kM>, number<kIter * WG::kK>>{});
            });
            HotLoopScheduler();

            ////////////////////////////// Next K //////////////////////////////

            // prefetch B(2i+2)
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    b_warp_tensor_ping(nIter)(kIter) = load_tile_with_offset(
                        b_flat_dram_windows(nIter), number<kIter * KFlatPerBlockPerIter>{});
                    if constexpr(kIter == KIterPerWarp - 1)
                        move_tile_window(b_flat_dram_windows(nIter),
                                         {0, BlockGemmShape::flatKPerBlock});
                });
            });

            // prefetch Scale A and Scale B (2i+2)
            static_for<0, MIterPerWarp / MXdlPack, 1>{}([&](auto mIter_pack) {
                static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                    scale_a_dram_windows(mIter_pack)(kIter_pack) = scale_a_dram_window;
                    move_tile_window(scale_a_dram_windows(mIter_pack)(kIter_pack),
                                     {mIter_pack * MWarp * WG::kM, kIter_pack * (64 / WG::kM)});

                    scale_a_tile_tensor_ping(mIter_pack)(kIter_pack) =
                        load_tile(scale_a_dram_windows(mIter_pack)(kIter_pack));
                });
            });

            static_for<0, NIterPerWarp / NXdlPack, 1>{}([&](auto nIter_pack) {
                static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                    scale_b_dram_windows(nIter_pack)(kIter_pack) = scale_b_dram_window;
                    move_tile_window(scale_b_dram_windows(nIter_pack)(kIter_pack),
                                     {nIter_pack * NWarp * WG::kN, kIter_pack * (64 / WG::kN)});

                    scale_b_tile_tensor_ping(nIter_pack)(kIter_pack) =
                        load_tile(scale_b_dram_windows(nIter_pack)(kIter_pack));
                });
            });

            // GEMM 2i+1
            static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                static_for<0, MIterPerWarp / MXdlPack, 1>{}([&](auto mIter_pack) {
                    static_for<0, NIterPerWarp / NXdlPack, 1>{}([&](auto nIter_pack) {
                        static_for<0, KXdlPack, 1>{}([&](auto ikxdl) {
                            static_for<0, MXdlPack, 1>{}([&](auto imxdl) {
                                constexpr auto AwarpIter = imxdl + ikxdl * MXdlPack;
                                static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                                    // read C warp tensor from C block tensor
                                    CWarpTensor c_warp_tensor;
                                    c_warp_tensor.get_thread_buffer() =
                                        c_block_tile.get_y_sliced_thread_data(
                                            merge_sequences(
                                                sequence<mIter_pack * MXdlPack + imxdl,
                                                         nIter_pack * NXdlPack + inxdl>{},
                                                c_warp_y_index_zeros),
                                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                                    // warp GEMM
                                    WG{}.template
                                    // operator()<MXdlPack + imxdl, ikxdl * NXdlPack + inxdl>(
                                    operator()<ikxdl * MXdlPack + imxdl, ikxdl * NXdlPack + inxdl>(
                                        c_warp_tensor,
                                        a_warp_tensor(number<AwarpIter>{}),
                                        b_warp_tensor_pong(nIter_pack * number<NXdlPack>{} + inxdl)(
                                            kIter_pack * number<KXdlPack>{} + ikxdl),
                                        scale_a_tile_tensor_pong(mIter_pack)(kIter_pack)
                                            .get_thread_buffer()[0], // scale A
                                        scale_b_tile_tensor_pong(nIter_pack)(kIter_pack)
                                            .get_thread_buffer()[0]); // scale B

                                    // write C warp tensor into C block tensor
                                    c_block_tile.set_y_sliced_thread_data(
                                        merge_sequences(sequence<mIter_pack * MXdlPack + imxdl,
                                                                 nIter_pack * NXdlPack + inxdl>{},
                                                        c_warp_y_index_zeros),
                                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                        c_warp_tensor.get_thread_buffer());
                                });
                                // preload next A from lds
                                constexpr auto addr = (mIter_pack * MXdlPack + imxdl) % 2 +
                                                      (kIter_pack * KXdlPack + ikxdl) * 2 +
                                                      (mIter_pack * MXdlPack + imxdl) / 2 * 4 +
                                                      m_preload;
                                if constexpr(addr < (KIterPerWarp * MIterPerWarp) &&
                                             (nIter_pack == NIterPerWarp / NXdlPack - 1))
                                {
                                    constexpr auto AmIter              = addr % 2 + addr / 4 * 2;
                                    constexpr auto AkIter              = addr / 2 % 2;
                                    a_warp_tensor(number<AwarpIter>{}) = load_tile_with_offset(
                                        a_warp_window_pong,
                                        tuple<number<AmIter * WG::kM>, number<AkIter * WG::kK>>{});
                                }
                            });
                        });
                    });
                });
            });
            // barrier as ds_load A(2i + 1) and buffer_load_lds A(2i + 2) finished
            s_waitcnt< // vmcnt
                Bload_num + ScaleAload_num + ScaleBload_num>();
            block_sync_lds();

            // Prefetch A(2i+3)
            async_load_tile_(a_store_lds_window_pong, a_dram_window);
            move_tile_window(a_dram_window, {0, kKPerBlock});
            // move B window to next flat K
            move_tile_window(scale_a_dram_window, {0, kKPerBlock / (32 * KXdlPack)});
            move_tile_window(scale_b_dram_window, {0, kKPerBlock / (32 * KXdlPack)});

            // preload A(2i+2)
            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter    = loadIter % MXdlPack;
                constexpr auto kIter    = loadIter / MXdlPack;
                a_warp_tensor(loadIter) = load_tile_with_offset(
                    a_warp_window_ping, tuple<number<mIter * WG::kM>, number<kIter * WG::kK>>{});
            });
            HotLoopScheduler();
        };

        if constexpr(HasHotLoop)
        {
            index_t iCounter = (num_loop - 1) / 2;
            while(iCounter > 0)
            {
                main_body_implx2();
                iCounter--;
            }
        }

        // TAIL
        if constexpr(TailNum == TailNumber::Even)
        {
            // prefetch B(loopK)
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    b_warp_tensor_pong(nIter)(kIter) = load_tile_with_offset(
                        b_flat_dram_windows(nIter),
                        make_tuple(number<0>{}, number<kIter * KFlatPerBlockPerIter>{}));
                });
            });

            // prefetch Scale A and Scale B (2i+1)
            static_for<0, MIterPerWarp / MXdlPack, 1>{}([&](auto mIter_pack) {
                static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                    scale_a_dram_windows(mIter_pack)(kIter_pack) = scale_a_dram_window;
                    move_tile_window(scale_a_dram_windows(mIter_pack)(kIter_pack),
                                     {mIter_pack * MWarp * WG::kM, kIter_pack * (64 / WG::kM)});

                    scale_a_tile_tensor_pong(mIter_pack)(kIter_pack) =
                        load_tile(scale_a_dram_windows(mIter_pack)(kIter_pack));
                });
            });
            static_for<0, NIterPerWarp / NXdlPack, 1>{}([&](auto nIter_pack) {
                static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                    scale_b_dram_windows(nIter_pack)(kIter_pack) = scale_b_dram_window;
                    move_tile_window(scale_b_dram_windows(nIter_pack)(kIter_pack),
                                     {nIter_pack * NWarp * WG::kN, kIter_pack * (64 / WG::kN)});

                    scale_b_tile_tensor_pong(nIter_pack)(kIter_pack) =
                        load_tile(scale_b_dram_windows(nIter_pack)(kIter_pack));
                });
            });

            // GEMM loopK-1
            static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                static_for<0, MIterPerWarp / MXdlPack, 1>{}([&](auto mIter_pack) {
                    static_for<0, NIterPerWarp / NXdlPack, 1>{}([&](auto nIter_pack) {
                        static_for<0, KXdlPack, 1>{}([&](auto ikxdl) {
                            static_for<0, MXdlPack, 1>{}([&](auto imxdl) {
                                constexpr auto AwarpIter = imxdl + ikxdl * MXdlPack;
                                static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                                    // read C warp tensor from C block tensor
                                    CWarpTensor c_warp_tensor;
                                    c_warp_tensor.get_thread_buffer() =
                                        c_block_tile.get_y_sliced_thread_data(
                                            merge_sequences(
                                                sequence<mIter_pack * MXdlPack + imxdl,
                                                         nIter_pack * NXdlPack + inxdl>{},
                                                c_warp_y_index_zeros),
                                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                                    // warp GEMM
                                    WG{}.template
                                    operator()<ikxdl * MXdlPack + imxdl, ikxdl * NXdlPack + inxdl>(
                                        c_warp_tensor,
                                        a_warp_tensor(number<AwarpIter>{}),
                                        b_warp_tensor_ping(nIter_pack * number<NXdlPack>{} + inxdl)(
                                            kIter_pack * number<KXdlPack>{} + ikxdl),
                                        scale_a_tile_tensor_ping(mIter_pack)(kIter_pack)
                                            .get_thread_buffer()[0], // scale A
                                        scale_b_tile_tensor_ping(nIter_pack)(kIter_pack)
                                            .get_thread_buffer()[0]); // scale B

                                    // write C warp tensor into C block tensor
                                    c_block_tile.set_y_sliced_thread_data(
                                        merge_sequences(sequence<mIter_pack * MXdlPack + imxdl,
                                                                 nIter_pack * NXdlPack + inxdl>{},
                                                        c_warp_y_index_zeros),
                                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                        c_warp_tensor.get_thread_buffer());
                                });
                                // preload next A from lds
                                constexpr auto addr = (mIter_pack * MXdlPack + imxdl) % 2 +
                                                      (kIter_pack * KXdlPack + ikxdl) * 2 +
                                                      (mIter_pack * MXdlPack + imxdl) / 2 * 4 +
                                                      m_preload;
                                if constexpr(addr < (KIterPerWarp * MIterPerWarp) &&
                                             (nIter_pack == NIterPerWarp / NXdlPack - 1))
                                {
                                    constexpr auto AmIter              = addr % 2 + addr / 4 * 2;
                                    constexpr auto AkIter              = addr / 2 % 2;
                                    a_warp_tensor(number<AwarpIter>{}) = load_tile_with_offset(
                                        a_warp_window_ping,
                                        tuple<number<AmIter * WG::kM>, number<AkIter * WG::kK>>{});
                                }
                            });
                        });
                    });
                });
            });
            // barrier as ds_load A(2i) and buffer_load_lds A(2i + 1) finished
            s_waitcnt< // vmcnt
                Bload_num + ScaleAload_num + ScaleBload_num>();
            block_sync_lds();

            // preload A(2i+1)
            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter    = loadIter % MXdlPack;
                constexpr auto kIter    = loadIter / MXdlPack;
                a_warp_tensor(loadIter) = load_tile_with_offset(
                    a_warp_window_pong, tuple<number<mIter * WG::kM>, number<kIter * WG::kK>>{});
            });

            // Last2ndHotLoopScheduler();

            // GEMM loopK
            static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                static_for<0, MIterPerWarp / MXdlPack, 1>{}([&](auto mIter_pack) {
                    static_for<0, NIterPerWarp / NXdlPack, 1>{}([&](auto nIter_pack) {
                        static_for<0, KXdlPack, 1>{}([&](auto ikxdl) {
                            static_for<0, MXdlPack, 1>{}([&](auto imxdl) {
                                constexpr auto AwarpIter = imxdl + ikxdl * MXdlPack;
                                static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                                    // read C warp tensor from C block tensor
                                    CWarpTensor c_warp_tensor;
                                    c_warp_tensor.get_thread_buffer() =
                                        c_block_tile.get_y_sliced_thread_data(
                                            merge_sequences(
                                                sequence<mIter_pack * MXdlPack + imxdl,
                                                         nIter_pack * NXdlPack + inxdl>{},
                                                c_warp_y_index_zeros),
                                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                                    // warp GEMM
                                    WG{}.template
                                    operator()<ikxdl * MXdlPack + imxdl, ikxdl * NXdlPack + inxdl>(
                                        //  operator()<MXdlPack + imxdl, ikxdl * NXdlPack + inxdl>(
                                        c_warp_tensor,
                                        a_warp_tensor(number<AwarpIter>{}),
                                        b_warp_tensor_pong(nIter_pack * number<NXdlPack>{} + inxdl)(
                                            kIter_pack * number<KXdlPack>{} + ikxdl),
                                        scale_a_tile_tensor_pong(mIter_pack)(kIter_pack)
                                            .get_thread_buffer()[0], // scale A
                                        scale_b_tile_tensor_pong(nIter_pack)(kIter_pack)
                                            .get_thread_buffer()[0]); // scale B

                                    // write C warp tensor into C block tensor
                                    c_block_tile.set_y_sliced_thread_data(
                                        merge_sequences(sequence<mIter_pack * MXdlPack + imxdl,
                                                                 nIter_pack * NXdlPack + inxdl>{},
                                                        c_warp_y_index_zeros),
                                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                        c_warp_tensor.get_thread_buffer());
                                });
                                // preload next A from lds
                                constexpr auto addr = (mIter_pack * MXdlPack + imxdl) % 2 +
                                                      (kIter_pack * KXdlPack + ikxdl) * 2 +
                                                      (mIter_pack * MXdlPack + imxdl) / 2 * 4 +
                                                      m_preload;
                                if constexpr(addr < (KIterPerWarp * MIterPerWarp) &&
                                             (nIter_pack == NIterPerWarp / NXdlPack - 1))
                                {
                                    constexpr auto AmIter              = addr % 2 + addr / 4 * 2;
                                    constexpr auto AkIter              = addr / 2 % 2;
                                    a_warp_tensor(number<AwarpIter>{}) = load_tile_with_offset(
                                        a_warp_window_pong,
                                        tuple<number<AmIter * WG::kM>, number<AkIter * WG::kK>>{});
                                }
                            });
                        });
                    });
                });
            });
            // LastHotLoopScheduler();
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            // GEMM loopK
            static_for<0, KIterPerWarp / KXdlPack, 1>{}([&](auto kIter_pack) {
                static_for<0, MIterPerWarp / MXdlPack, 1>{}([&](auto mIter_pack) {
                    static_for<0, NIterPerWarp / NXdlPack, 1>{}([&](auto nIter_pack) {
                        static_for<0, KXdlPack, 1>{}([&](auto ikxdl) {
                            static_for<0, MXdlPack, 1>{}([&](auto imxdl) {
                                constexpr auto AwarpIter = imxdl + ikxdl * MXdlPack;
                                constexpr auto m_iter    = mIter_pack * MXdlPack + imxdl;
                                constexpr auto k_iter    = kIter_pack * KXdlPack + ikxdl;
                                static_for<0, NXdlPack, 1>{}([&](auto inxdl) {
                                    constexpr auto n_iter = nIter_pack * NXdlPack + inxdl;

                                    // read C warp tensor from C block tensor
                                    CWarpTensor c_warp_tensor;
                                    c_warp_tensor.get_thread_buffer() =
                                        c_block_tile.get_y_sliced_thread_data(
                                            merge_sequences(sequence<m_iter, n_iter>{},
                                                            c_warp_y_index_zeros),
                                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                                    //  warp GEMM
                                    WG{}.template
                                    // operator()<MXdlPack + imxdl, ikxdl * NXdlPack + inxdl>(
                                    operator()<ikxdl * MXdlPack + imxdl, ikxdl * NXdlPack + inxdl>(
                                        c_warp_tensor,
                                        a_warp_tensor(number<AwarpIter>{}),
                                        b_warp_tensor_ping(nIter_pack * number<NXdlPack>{} + inxdl)(
                                            kIter_pack * number<KXdlPack>{} + ikxdl),
                                        scale_a_tile_tensor_ping(mIter_pack)(kIter_pack)
                                            .get_thread_buffer()[0],
                                        scale_b_tile_tensor_ping(nIter_pack)(kIter_pack)
                                            .get_thread_buffer()[0]);

                                    // write C warp tensor into C block tensor
                                    c_block_tile.set_y_sliced_thread_data(
                                        merge_sequences(sequence<m_iter, n_iter>{},
                                                        c_warp_y_index_zeros),
                                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                        c_warp_tensor.get_thread_buffer());
                                });
                                // preload next A from lds
                                constexpr auto addr =
                                    m_iter % 2 + k_iter * 2 + m_iter / 2 * 4 + m_preload;
                                if constexpr(addr < (KIterPerWarp * MIterPerWarp) &&
                                             (nIter_pack == NIterPerWarp / NXdlPack - 1))
                                {
                                    constexpr auto AmIter              = addr % 2 + addr / 4 * 2;
                                    constexpr auto AkIter              = addr / 2 % 2;
                                    a_warp_tensor(number<AwarpIter>{}) = load_tile_with_offset(
                                        a_warp_window_ping,
                                        tuple<number<AmIter * WG::kM>, number<AkIter * WG::kK>>{});
                                }
                            });
                        });
                    });
                });
            });
            // barrier as ds_load A(2i) and buffer_load_lds A(2i + 1) finished
            s_waitcnt< // vmcnt
                Bload_num + ScaleAload_num + ScaleBload_num>();
            block_sync_lds();
        }
        else
        {
            static_assert(false, "Wrong TailNum");
        }
        return c_block_tile;
    }
};

} // namespace ck_tile
