// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {

template <typename Problem>
struct BaseFlatmmPipelineAGmemBGmemCRegV1
{
    static constexpr index_t PrefetchStages = 2;

    CK_TILE_HOST static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        return num_loop % 2 == 0 ? TailNumber::Even : TailNumber::Odd;
    }

    CK_TILE_HOST static constexpr amd_buffer_coherence_enum
    GetBMemNTType(index_t M, index_t N, index_t K)
    {
        ck_tile::ignore = N;
        ck_tile::ignore = K;
        if(M <= 416)
        {
#if defined(__gfx942__) || defined(__gfx950__)
            return ck_tile::amd_buffer_coherence_enum::WAVE_NT1;
#endif
        }
        return ck_tile::amd_buffer_coherence_enum::coherence_default;
    }

    template <bool DispatchHotloop = false, TailNumber tail_num, typename RunFunction>
    CK_TILE_HOST_DEVICE static auto TailHandler(const RunFunction& run_func, bool has_hot_loop)
    {
        if constexpr(!DispatchHotloop)
            return run_func(bool_constant<true>{}, integral_constant<TailNumber, tail_num>{});
        else if(has_hot_loop)
            return run_func(bool_constant<true>{}, integral_constant<TailNumber, tail_num>{});
        else
            return run_func(bool_constant<false>{}, integral_constant<TailNumber, tail_num>{});
    }

    template <bool DispatchHotloop = false, typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_num)
    {
        if(TailNumber::Even == tail_num)
            return TailHandler<DispatchHotloop, TailNumber::Even>(run_func, has_hot_loop);
        else if(TailNumber::Odd == tail_num)
            return TailHandler<DispatchHotloop, TailNumber::Odd>(run_func, has_hot_loop);
        else
        {
            assert(("Wrong TailNumber!", false));
            return TailHandler<DispatchHotloop, TailNumber::Even>(run_func, has_hot_loop);
        }
    }
};

template <typename Problem, typename PipelinePolicy = UniversalFlatmmPipelineAgBgCrPolicy>
struct FlatmmPipelineAGmemBGmemCRegV1
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>; // TileFlatmmShape

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

    static constexpr index_t flatKPerWarp = BlockGemmShape::flatKPerWarp;
    static constexpr index_t flatNPerWarp = BlockGemmShape::flatNPerWarp;

    static constexpr index_t GetVectorSizeA() { return Problem::VectorSizeA; }
    static constexpr index_t GetVectorSizeB() { return Problem::VectorSizeB; }
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

    static constexpr index_t K1        = Problem::VectorLoadSize / sizeof(ADataType);
    static constexpr index_t m_preload = (MIterPerWarp * KIterPerWarp >= DsReadPreload)
                                             ? DsReadPreload
                                             : MIterPerWarp * KIterPerWarp;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;

/*
defined(USING_MFMA_16x16x32) && defined(ENABLE_FP8) // mi300 fp8 16c 0.5*K1
defined(USING_MFMA_32x32x16) && defined(ENABLE_FP8) // mi300 fp8 32c 0.5*K1
defined(USING_MFMA_16x16x16) && defined(ENABLE_FP16) // mi300 fp16 16c 0.5*K1
defined(USING_MFMA_32x32x8) && defined(ENABLE_FP16) // mi300 fp16 32c 0.5*K1

defined(USING_MFMA_16x16x128) && defined(ENABLE_FP8) // mi350 fp8 32c 2*K1
defined(USING_MFMA_32x32x64) && defined(ENABLE_FP8) // mi350 fp8 64c 2*K1
defined(USING_MFMA_16x16x32) && defined(ENABLE_FP16) // mi350 fp16 16c 1*K1
defined(USING_MFMA_32x32x16) && defined(ENABLE_FP16) // mi350 fp16 32c 1*K1

defined(USING_MFMA_16x16x128) && defined(ENABLE_FP4) // mi350 fp4 16c 1*K1
defined(USING_MFMA_32x32x64) && defined(ENABLE_FP4) // mi350 fp4 32c 1*K1
*/

// #if (defined(USING_MFMA_16x16x32_F8) ||  \
//     defined(USING_MFMA_32x32x16_F8) ||  \
//     defined(USING_MFMA_16x16x16_F16) || \
//     defined(USING_MFMA_32x32x8_F16)) // K1 per Mfma = 0.5
//     static constexpr auto mfma_per_wg = 2;
//     static constexpr auto dsread_per_wg = 1;
// #elif (defined(USING_MFMA_16x16x32_F16) || \
//     defined(USING_MFMA_32x32x16_F16) ||   \
//     defined(USING_MFMA_16x16x128_F4) ||   \
//     defined(USING_MFMA_32x32x64_F4)) // K1 per Mfma = 1
//     static constexpr auto mfma_per_wg = 1;
//     static constexpr auto dsread_per_wg = 1;
// #elif (defined(USING_MFMA_16x16x128_F8) || \
//     defined(USING_MFMA_32x32x64_F8)) // K1 per Mfma = 2
//     static constexpr auto mfma_per_wg = 1;
//     static constexpr auto dsread_per_wg = 2;
// #endif
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
    static constexpr index_t Bload_num_perK = kNPerBlock * WG::kK / NWarp / K1 / WaveSize;
    static constexpr index_t HalfMIter      = (MIterPerWarp + 1) / 2;
    static constexpr index_t Bload_rep      = (Bload_num_perK + HalfMIter - 1) / HalfMIter;

    static constexpr index_t mfma_perM_perK = NIterPerWarp * mfma_per_wg;
    static constexpr index_t dswrite_mIter  = (DsWritePreIssue - 1) % MIterPerWarp;
    static constexpr index_t dswrite_kIter  = (DsWritePreIssue - 1) / MIterPerWarp;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AGmemBGmemCRegV1", 
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock,  BlockSize),
                      concat('x', WG::kM, WG::kN, WG::kK),
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

    // For the basic gemm pipelien DoubleSmemBuffer set to be false naturally.
    static constexpr bool DoubleSmemBuffer = false;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return PipelinePolicy::template GetSmemSize<Problem>();
    }

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
#pragma unroll
        for(int idx = 0; idx < NIterPerWarp * 10; idx++)
        {
            inst_order[idx] = 0;
        }

        index_t index = 0;
#pragma unroll
        for(int j = 0; j < max_data_inst; j++)
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
#pragma unroll
        for(int j = 0; j < mfma_perM_perK; j++)
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

#pragma unroll
            for(int r = 0; r < round_data_inst; r++)
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

#pragma unroll
        for(int kIter = 0; kIter < KIterPerWarp; kIter++)
        {
#pragma unroll
            for(int mIter = 0; mIter < MIterPerWarp; mIter++)
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
#pragma unroll
        for(int kIter = 0; kIter < KIterPerWarp; kIter++)
        {
#pragma unroll
            for(int mIter = 0; mIter < MIterPerWarp; mIter++)
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
#pragma unroll
        for(int kIter = 0; kIter < KIterPerWarp; kIter++)
        {
#pragma unroll
            for(int mIter = 0; mIter < MIterPerWarp; mIter++)
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

    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp, typename AElementFunction>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                        index_t num_loop,
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

        // A tile in LDS
        ADataType* p_a_lds_ping = static_cast<ADataType*>(p_smem_ping);
        ADataType* p_a_lds_pong = static_cast<ADataType*>(p_smem_pong);

        constexpr auto a_lds_block_desc =
            PipelinePolicy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block_ping =
            make_tensor_view<address_space_enum::lds>(p_a_lds_ping, a_lds_block_desc);
        auto a_lds_block_pong =
            make_tensor_view<address_space_enum::lds>(p_a_lds_pong, a_lds_block_desc);

        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             PipelinePolicy::template MakeADramTileDistribution<Problem>());

        auto a_copy_lds_window_ping =
            make_tile_window(a_lds_block_ping,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             PipelinePolicy::template MakeADramTileDistribution<Problem>());

        auto a_copy_lds_window_pong =
            make_tile_window(a_lds_block_pong,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             PipelinePolicy::template MakeADramTileDistribution<Problem>());

        // ping-pong window for A LDS
        auto a_warp_window_ping_tmp =
            make_tile_window(a_lds_block_ping,
                             make_tuple(number<WG::kM>{}, number<WG::kK>{}),
                             {iMWarp * WG::kM, 0},
                             PipelinePolicy::template MakeALDS_WarpTileDistribution<Problem>());

        auto a_warp_window_pong_tmp =
            make_tile_window(a_lds_block_pong,
                             make_tuple(number<WG::kM>{}, number<WG::kK>{}),
                             {iMWarp * WG::kM, 0},
                             PipelinePolicy::template MakeALDS_WarpTileDistribution<Problem>());

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_ping_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows_ping;

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_pong_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows_pong;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows_ping(mIter)(kIter) = a_warp_window_ping_tmp;
                a_warp_windows_pong(mIter)(kIter) = a_warp_window_pong_tmp;

                move_tile_window(a_warp_windows_ping(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
                move_tile_window(a_warp_windows_pong(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        // Block GEMM
        auto block_flatmm = BlockFlatmm();
        // Acc register tile
        auto c_block_tile = block_flatmm.MakeCBlockTile();

        // B flat DRAM window for load
        auto b_flat_distribution =
            PipelinePolicy::template MakeBFlatDramTileDistribution<Problem>();
        auto b_flat_dram_window = // tile_window_with_static_distribution
            make_tile_window(
                b_flat_dram_block_window_tmp.get_bottom_tensor_view(), // from kernel gemm_pad_views
                make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
                b_flat_dram_block_window_tmp.get_window_origin(),
                b_flat_distribution);

        // pingpong buffer for B
        statically_indexed_array<
            statically_indexed_array<decltype(b_flat_dram_window), KIterPerWarp>,
            NIterPerWarp>
            b_flat_dram_windows;

        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(b_flat_dram_window)), KIterPerWarp>,
            NIterPerWarp>
            b_warp_tensor_ping;

        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(b_flat_dram_window)), KIterPerWarp>,
            NIterPerWarp>
            b_warp_tensor_pong;

        // HEAD
        // Prefetch A0
        auto a_block_tile = load_tile(a_copy_dram_window);
        // move A window to next k
        move_tile_window(a_copy_dram_window, {0, kKPerBlock});

        // prefetch B
        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                 {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                b_warp_tensor_ping(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
            });
        });
        // move B window to next flat K
        move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

        auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
        store_tile(a_copy_lds_window_ping, a_block_tile_tmp);
        __builtin_amdgcn_sched_barrier(0);

        // Prefetch A1
        a_block_tile = load_tile(a_copy_dram_window);
        // move A window to next k
        move_tile_window(a_copy_dram_window, {0, kKPerBlock});

        // initialize C
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

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

        // MAIN LOOP
        index_t iCounter = (num_loop - 1) / 2;
        while(iCounter > 0)
        {
            // prefetch B(2i+1)
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor_pong(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });

            // Prefill A(2i+1)
            a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window_pong, a_block_tile_tmp);

            // Prefetch A(2i+2)
            a_block_tile = load_tile(a_copy_dram_window);
            // move A window to next k
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

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

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor(number<AwarpIter>{}),
                             b_warp_tensor_ping(nIter)(kIter));

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
                        block_sync_lds();
                    }
                });
            });

            // move B window to next flat K
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter = loadIter % MIterPerWarp;
                constexpr auto kIter = loadIter / MIterPerWarp;
                a_warp_tensor(loadIter) =
                    load_tile(a_warp_windows_pong(number<mIter>{})(number<kIter>{}));
            });
            HotLoopScheduler();

            // Next K

            // prefetch B(2i+2)
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor_ping(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });

            // Prefill A(2i+2)
            a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window_ping, a_block_tile_tmp);

            // Prefetch A(2i+3)
            a_block_tile = load_tile(a_copy_dram_window);
            // move A window to next k
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

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

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor(number<AwarpIter>{}),
                             b_warp_tensor_pong(nIter)(kIter));

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
                        block_sync_lds();
                    }
                });
            });

            // move B window to next flat K
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

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
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor_pong(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });

            // Prefill A(loopK)
            a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window_pong, a_block_tile_tmp);

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

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor(number<AwarpIter>{}),
                             b_warp_tensor_ping(nIter)(kIter));

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

            Last2ndHotLoopScheduler();

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

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor(number<AwarpIter>{}),
                             b_warp_tensor_pong(nIter)(kIter));

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
                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        block_sync_lds();
                    }
                });
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

                        // warp GEMM
                        WG{}(c_warp_tensor,
                             a_warp_tensor(number<AwarpIter>{}),
                             b_warp_tensor_ping(nIter)(kIter));

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
                        block_sync_lds();
                    }
                });
            });
            LastHotLoopScheduler();
        }

        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem_ping,
                                   void* p_smem_pong) const
    {
        return operator()(
            a_dram_block_window_tmp,
            [](const ADataType & a) { return a; },
            b_flat_dram_block_window_tmp,
            num_loop,
            p_smem_ping,
            p_smem_pong);
    }
};

} // namespace ck_tile
