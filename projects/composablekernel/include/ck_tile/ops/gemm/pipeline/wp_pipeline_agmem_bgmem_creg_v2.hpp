// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/load_interleaved_pk_type.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp"

namespace ck_tile {

template <typename Problem>
struct BaseWeightPreshufflePipelineAGmemBGmemCRegV2
{
    static constexpr index_t PrefetchStages   = 2;
    static constexpr index_t PrefillStages    = 1;
    static constexpr index_t GlobalBufferNum  = 1;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        return num_loop % 2 == 0 ? TailNumber::Even : TailNumber::Odd;
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
    {
        if(has_hot_loop)
        {
            if(tail_number == TailNumber::Odd)
            {
                return run_func(bool_constant<true>{},
                                integral_constant<TailNumber, TailNumber::Odd>{});
            }
            else // Even tail number
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
            else // Even tail number
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::Even>{});
            }
        }
    }
};

template <typename Problem, typename PipelinePolicy = UniversalWeightPreshufflePipelineAgBgCrPolicy>
struct WeightPreshufflePipelineAGmemBGmemCRegV2
    : public BaseWeightPreshufflePipelineAGmemBGmemCRegV2<Problem>
{
    using Base             = BaseWeightPreshufflePipelineAGmemBGmemCRegV2<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, PipelinePolicy>;

    using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType  = remove_cvref_t<typename Problem::CDataType>;

    using AElementWise   = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise   = remove_cvref_t<typename Problem::BElementWise>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>; // TileFlatmmShape

    using AsLayout = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    using BlockWeightPreshuffle =
        remove_cvref_t<decltype(PipelinePolicy::template GetBlockWeightPreshuffle<Problem>())>;

    static constexpr index_t DsWritePreIssue = 3; // default 2, ds write at MIter - 2
    static constexpr index_t DsReadPreload   = 2; // default 2, preload 2 ds read

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t WaveSize  = get_warp_size();

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    // bogus variables to compile grouped gemm (to be removed)
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t kflatKPerBlock = BlockGemmShape::flatKPerBlock;

    static constexpr index_t flatKPerWarp = BlockGemmShape::flatKPerWarp;
    static constexpr index_t flatNPerWarp = BlockGemmShape::flatNPerWarp;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return PipelinePolicy::template GetVectorSizeA<Problem, IsWave32Host>();
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return PipelinePolicy::template GetVectorSizeB<Problem, IsWave32Host>();
    }

    static constexpr index_t GetVectorSizeC()
    {
        return PipelinePolicy::template GetVectorSizeC<Problem>();
    }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr index_t kLdsAlignmentInBytes = 16;
    static constexpr index_t NumWaveGroups        = Problem::NumWaveGroups;

    static constexpr auto I0   = number<0>();
    static constexpr auto I1   = number<1>();
    static constexpr auto I2   = number<2>();
    static constexpr auto idxM = I0;
    static constexpr auto idxN = I1;
    static constexpr auto idxK = I2;
    using BlockTile            = remove_cvref_t<typename BlockGemmShape::BlockTile>;
    using BlockWarps           = remove_cvref_t<typename BlockGemmShape::BlockWarps>;
    using WarpTile             = remove_cvref_t<typename BlockGemmShape::WarpTile>;

    static constexpr index_t MWarp = BlockWarps::at(I0);
    static constexpr index_t NWarp = BlockWarps::at(I1);

    static constexpr index_t WarpTileM = WarpTile::at(I0);
    static constexpr index_t WarpTileN = WarpTile::at(I1);
    static constexpr index_t WarpTileK = WarpTile::at(I2);

    static constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpTileM);
    static constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpTileN);
    static constexpr index_t KIterPerWarp = kKPerBlock / WarpTileK;

    static constexpr index_t KFlatPerBlockPerIter = flatKPerWarp;
    static constexpr index_t NFlatPerBlockPerIter = flatNPerWarp;

    static constexpr index_t MPerBlockPerIter = kMPerBlock / MIterPerWarp;
    static constexpr index_t KPerBlockPerIter = kKPerBlock / KIterPerWarp;

    static constexpr index_t K1        = Problem::VectorLoadSize / sizeof(ADataType);
    static constexpr index_t m_preload = (MIterPerWarp * KIterPerWarp >= DsReadPreload)
                                             ? DsReadPreload
                                             : MIterPerWarp * KIterPerWarp;

#ifdef __gfx942__
    static constexpr index_t mfma_per_wg = 2;
#else
    static constexpr index_t mfma_per_wg = 1;
#endif
    static constexpr index_t dsread_per_wg = max(
        index_t(WarpTileM * WarpTileK * sizeof(ADataType) / WaveSize / Problem::VectorLoadSize), 1);
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert((WarpTileM * WarpTileK * sizeof(ADataType) * MIterPerWarp / WaveSize) %
                      Problem::VectorLoadSize ==
                  0);
#endif
    static constexpr index_t dsread_num_perK = WarpTileM * WarpTileK * sizeof(ADataType) *
                                               MIterPerWarp / WaveSize / Problem::VectorLoadSize;
    static constexpr index_t dswrite_num_perK = dsread_num_perK / (MWarp * NWarp);
    static constexpr index_t dswrite_rep    = (dswrite_num_perK + MIterPerWarp - 1) / MIterPerWarp;
    static constexpr index_t Aload_num_perK = dswrite_num_perK;
    static constexpr index_t Aload_rep      = dswrite_rep;
    static constexpr index_t Bload_num_perK = kNPerBlock * WarpTileK / NWarp / K1 / WaveSize;
    static constexpr index_t HalfMIter      = (MIterPerWarp + 1) / 2;
    static constexpr index_t Bload_rep      = (Bload_num_perK + HalfMIter - 1) / HalfMIter;

    static constexpr index_t mfma_perM_perK = NIterPerWarp * mfma_per_wg;
    static constexpr index_t dswrite_mIter  = (DsWritePreIssue - 1) % MIterPerWarp;
    static constexpr index_t dswrite_kIter  = (DsWritePreIssue - 1) / MIterPerWarp;

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "PRESHUFFLE_V2";
        // clang-format on
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AGmemBGmemCRegV2", 
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock,  BlockSize),
                      concat('x', WarpTileM, WarpTileN, WarpTileK),
                      concat('x', GetVectorSizeA(), GetVectorSizeB()),
                      concat('x', kPadM, kPadN, kPadK));

        // clang-format on
    }

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static constexpr index_t Preshuffle = Problem::Preshuffle;
    using Base::UsePersistentKernel;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size = PipelinePolicy::template GetSmemSize<Problem>();
        return DoubleSmemBuffer ? 2 * smem_size : smem_size;
    }

    // dsread_perM: how many LDS reads want to issue in this M-iter
    // dswrite_perM: how many LDS writes you want to do this M-iter
    // load_perM: how many global loads VMEM want to do in this M-iter
    CK_TILE_HOST_DEVICE static constexpr auto
    SchedulerPerM(index_t dsread_perM, index_t dswrite_perM, index_t load_perM)
    {

        // Init inst order
        index_t max_data_inst   = dsread_perM > load_perM
                                      ? (dsread_perM > dswrite_perM ? dsread_perM : dswrite_perM)
                                      : (load_perM > dswrite_perM ? load_perM : dswrite_perM);
        index_t sum_data_inst   = dsread_perM + load_perM + dswrite_perM;
        index_t round_data_inst = ck_tile::integer_divide_ceil(sum_data_inst, mfma_perM_perK);

        constexpr int kOrderCap       = NIterPerWarp * 10;
        index_t inst_order[kOrderCap] = {};
        index_t index                 = 0;
#pragma unroll
        // round-robin
        // Index:   0 1 2 3 4 5 ...
        // Value:   1 2 3 1 2 3 ...
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

    struct PipelineImpl : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename BFlatBlockWindowTmp,
                  typename AElementFunction,
                  typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                                !is_detected<is_tuple, BFlatBlockWindowTmp>::value,
                                            bool>* = nullptr,
                  index_t UnaryOpSize_             = 8>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                       [[maybe_unused]] const AElementFunction& a_element_func,
                                       const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                       index_t num_loop,
                                       void* p_smem) const
        {
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>>,
                "wrong!");

            static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}],
                          "wrong!");
            static_assert(kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                          "wrong!");

            // A tile in LDS
            constexpr index_t smem_size = PipelinePolicy::template GetSmemSize<Problem>();

            constexpr auto a_lds_block_desc =
                PipelinePolicy::template MakeALdsBlockDescriptor<Problem>();

            auto a_lds_blocks = generate_tuple(
                [&](auto i) {
                    ADataType* p_a_lds = static_cast<ADataType*>(
                        static_cast<void*>(static_cast<char*>(p_smem) + smem_size * i.value));
                    return make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);
                },
                number<2>{});

            constexpr auto a_lds_load_tile_distr = make_static_tile_distribution(
                BlockWeightPreshuffle::MakeABlockDistributionEncode());
            auto&& windows_result =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_blocks, a_lds_load_tile_distr);
            auto&& a_copy_dram_window = windows_result.template get<0>();
            auto&& a_lds_windows      = windows_result.template get<1>();
            auto a_copy_lds_windows   = generate_tuple(
                [&](auto i) -> decltype(auto) { return a_lds_windows[i].template at<0>(); },
                number<2>{});
            // Block GEMM
            auto block_weight_preshuffle = BlockWeightPreshuffle();
            // Acc register tile
            auto c_block_tile = block_weight_preshuffle.MakeCBlockTile();

            auto a_load_windows = generate_tuple(
                [&](auto i) -> decltype(auto) {
                    return block_weight_preshuffle.MakeALoadWindows(a_copy_lds_windows[i]);
                },
                number<2>{});

            // B flat DRAM window for load
            auto b_flat_distribution =
                PipelinePolicy::template MakeBFlatDramTileDistribution<Problem>();
            auto b_flat_dram_window = // tile_window_with_static_distribution
                make_tile_window(b_flat_dram_block_window_tmp
                                     .get_bottom_tensor_view(), // from kernel gemm_pad_views
                                 make_tuple(number<flatNPerWarp * NIterPerWarp>{},
                                            number<flatKPerWarp * KIterPerWarp>{}),
                                 b_flat_dram_block_window_tmp.get_window_origin(),
                                 b_flat_distribution);

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BFlatBlockWindowTmp::BottomTensorIndex;
            constexpr ADramTileWindowStep a_dram_tile_window_step = make_array(0, kKPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step = make_array(0, kflatKPerBlock);

            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));

            using BTypeToUse =
                std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
            using BBlockTile =
                decltype(make_static_distributed_tensor<BTypeToUse>(b_flat_distribution));

            ABlockTile a_global_tile;
            BBlockTile b_global_tile[2];

            // // Prefetch A0
            Base::GlobalPrefetch(a_global_tile, a_copy_dram_window, a_dram_tile_window_step);

            Base::template GlobalPrefetch<BDataType, BTypeToUse, UnaryOpSize_>(
                b_global_tile[0], b_flat_dram_window, b_dram_tile_window_step);

            // Prefill A0
            Base::LocalPrefill(a_copy_lds_windows[I0], a_global_tile);

            // Prefetch A1
            Base::GlobalPrefetch(a_global_tile, a_copy_dram_window, a_dram_tile_window_step);

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            block_sync_lds();

            // preload A00,A10 from lds
            block_weight_preshuffle.LocalPrefetch(a_load_windows[I0]);

            __builtin_amdgcn_sched_barrier(0);
            // MAIN LOOP
            if constexpr(HasHotLoop)
            {
                index_t i_global_read = amd_wave_read_first_lane(2);
                do
                {
                    {
                        Base::template GlobalPrefetch<BDataType, BTypeToUse, UnaryOpSize_>(
                            b_global_tile[1], b_flat_dram_window, b_dram_tile_window_step);
                        Base::LocalPrefill(a_copy_lds_windows[I1], a_global_tile);
                        Base::GlobalPrefetch(
                            a_global_tile, a_copy_dram_window, a_dram_tile_window_step);
                        block_weight_preshuffle(c_block_tile,
                                                a_load_windows[I0],
                                                b_global_tile[0],
                                                b_flat_distribution);

                        block_weight_preshuffle.LocalPrefetch(a_load_windows[I1]);
                        HotLoopScheduler();
                    }
                    {
                        Base::template GlobalPrefetch<BDataType, BTypeToUse, UnaryOpSize_>(
                            b_global_tile[0], b_flat_dram_window, b_dram_tile_window_step);
                        Base::LocalPrefill(a_copy_lds_windows[I0], a_global_tile);
                        Base::GlobalPrefetch(
                            a_global_tile, a_copy_dram_window, a_dram_tile_window_step);
                        block_weight_preshuffle(c_block_tile,
                                                a_load_windows[I1],
                                                b_global_tile[1],
                                                b_flat_distribution);

                        block_weight_preshuffle.LocalPrefetch(a_load_windows[I0]);
                        HotLoopScheduler();
                    }
                    i_global_read += 2;
                } while(i_global_read < num_loop);
            }

            // tail
            if constexpr(TailNum == TailNumber::Even)
            {
                {
                    Base::template GlobalPrefetch<BDataType, BTypeToUse, UnaryOpSize_>(
                        b_global_tile[1], b_flat_dram_window, b_dram_tile_window_step);
                    Base::LocalPrefill(a_copy_lds_windows[I1], a_global_tile);
                    block_weight_preshuffle(
                        c_block_tile, a_load_windows[I0], b_global_tile[0], b_flat_distribution);
                    block_sync_lds();
                    block_weight_preshuffle.LocalPrefetch(a_load_windows[I1]);
                    Last2ndHotLoopScheduler();
                }
                {
                    block_weight_preshuffle(
                        c_block_tile, a_load_windows[I1], b_global_tile[1], b_flat_distribution);
                    LastHotLoopScheduler();
                }
            }
            else if constexpr(TailNum == TailNumber::Odd)
            {
                block_weight_preshuffle(
                    c_block_tile, a_load_windows[I0], b_global_tile[0], b_flat_distribution);
                LastHotLoopScheduler();
            }

            return c_block_tile;
        }
    };

    // called from universal gemm kernel
    template <typename ADramBlockWindowTmp,
              typename BFlatBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BFlatBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   [[maybe_unused]] const AElementFunction& a_element_func,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   [[maybe_unused]] const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const auto has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp[number<0>{}],
                a_element_func,
                b_flat_dram_block_window_tmp[number<0>{}],
                num_loop,
                p_smem);
        };
        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    // called from general gemm kernel
    template <typename ADramBlockWindowTmp,
              typename BFlatBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BFlatBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const auto has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            constexpr auto PassThrough = [](const ADataType& a) { return a; };
            return PipelineImpl{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                PassThrough,
                b_flat_dram_block_window_tmp,
                num_loop,
                p_smem);
        };
        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    // called from grouped gemm kernel
    template <typename ADramBlockWindowTmp,
              typename BFlatBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BFlatBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   index_t num_loop,
                                   TailNumber tail_number,
                                   void* __restrict__ p_smem) const
    {
        const auto has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto RunPipeline  = [&](auto hot_loop_, auto tail_num_) {
            constexpr auto PassThrough = [](const auto& x) { return x; };
            return PipelineImpl{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                PassThrough,
                b_flat_dram_block_window_tmp,
                num_loop,
                p_smem);
        };
        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }
};

} // namespace ck_tile
