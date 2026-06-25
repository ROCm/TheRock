// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp"

#define DS_READ_A_PREFETCH_STAGES 2

template <typename T>
constexpr auto compute_stage_loads(T total_loads, T stages)
{
    return std::make_pair((total_loads + stages - 1) / stages, // ceil
                          total_loads / stages                 // floor
    );
}

namespace ck {

// Compute optimized pipeline
// GlobalPrefetchStages: 2
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 1

template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
          index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename ComputeDataType,
          typename AccDataType,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPacks>
struct BlockwiseGemmXdlops_pipeline_bpreshuffle_v3
{
};

template <index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename ComputeDataType,
          typename AccDataType,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack
          // ,bool TransposeC //disable transposec right now...
          >
struct BlockwiseGemmXdlops_pipeline_bpreshuffle_v3<BlockGemmPipelineScheduler::Intrawave,
                                                   BlockSize,
                                                   ADataType,
                                                   BDataType,
                                                   ComputeDataType,
                                                   AccDataType,
                                                   ATileDesc,
                                                   BTileDesc,
                                                   AMmaTileDesc,
                                                   BMmaTileDesc,
                                                   ABlockTransferSrcScalarPerVector,
                                                   BBlockTransferSrcScalarPerVector,
                                                   MPerBlock,
                                                   NPerBlock,
                                                   KPerBlock,
                                                   MPerXDL,
                                                   NPerXDL,
                                                   MRepeat,
                                                   NRepeat,
                                                   KPack>
    : BlockwiseGemmXdlops_pipeline_base<BlockSize,
                                        ADataType,
                                        BDataType,
                                        ComputeDataType,
                                        AccDataType,
                                        ATileDesc,
                                        BTileDesc,
                                        AMmaTileDesc,
                                        BMmaTileDesc,
                                        ABlockTransferSrcScalarPerVector,
                                        BBlockTransferSrcScalarPerVector,
                                        MPerBlock,
                                        NPerBlock,
                                        KPerBlock,
                                        MPerXDL,
                                        NPerXDL,
                                        MRepeat,
                                        NRepeat,
                                        KPack>

{
    using Base = BlockwiseGemmXdlops_pipeline_base<BlockSize,
                                                   ADataType,
                                                   BDataType,
                                                   ComputeDataType,
                                                   AccDataType,
                                                   ATileDesc,
                                                   BTileDesc,
                                                   AMmaTileDesc,
                                                   BMmaTileDesc,
                                                   ABlockTransferSrcScalarPerVector,
                                                   BBlockTransferSrcScalarPerVector,
                                                   MPerBlock,
                                                   NPerBlock,
                                                   KPerBlock,
                                                   MPerXDL,
                                                   NPerXDL,
                                                   MRepeat,
                                                   NRepeat,
                                                   KPack>;
    using Base::A_K1;
    using Base::B_K1;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::KGroup;
    using Base::KRepeat;
    using Base::xdlops_gemm;
    using typename Base::HotLoopInstList;

    using Base::a_block_desc_m0_m1_m2_k;
    using Base::CalculateCThreadOriginDataIndex;
    using Base::CalculateCThreadOriginDataIndex8D;
    using Base::GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4;
    using Base::GetCThreadBuffer;
    using Base::GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4;
    using Base::MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::MWaves;
    using Base::WaveSize;

    static constexpr index_t PrefetchStages        = 2;
    static constexpr index_t PrefillStages         = 1;
    static constexpr index_t GlobalBufferNum       = 1;
    static constexpr index_t HotloopLocalBufSwitch = MRepeat % 2 == 0 ? 0 : 1;

    template <typename TileDesc_M0_M1_M2_K>
    __host__ __device__ static constexpr auto MakeAGemmMmaTileDescriptor(const TileDesc_M0_M1_M2_K&)
    {
        constexpr index_t M0 = TileDesc_M0_M1_M2_K{}.GetLength(Number<0>{});
        constexpr index_t M1 = TileDesc_M0_M1_M2_K{}.GetLength(Number<1>{});
        constexpr index_t M2 = TileDesc_M0_M1_M2_K{}.GetLength(Number<2>{});
        constexpr index_t K2 = KPack / KGroup;
        constexpr index_t K1 = WaveSize / NPerXDL;
        constexpr index_t K0 = KRepeat * KGroup;

        return transform_tensor_descriptor(
            TileDesc_M0_M1_M2_K{},
            make_tuple(
                make_pass_through_transform(Number<M0>{}),
                make_pass_through_transform(Number<M1>{}),
                make_pass_through_transform(Number<M2>{}),
                make_unmerge_transform(make_tuple(Number<K0>{}, Number<K1>{}, Number<K2>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4, 5>{}));
    }

    static constexpr auto a_block_desc_m0_m1_m2_k0_k1_k2 =
        MakeAGemmMmaTileDescriptor(a_block_desc_m0_m1_m2_k);

    __host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
    {
        return num_loop % 2 == 0 ? TailNumber::Even : TailNumber::Odd;
    }

    __device__ static constexpr auto HotLoopScheduler()
    {
        constexpr auto num_ds_read_inst_a =
            HotLoopInstList::A_LDS_Read_Width * sizeof(ADataType) == 16
                ? HotLoopInstList::A_LDS_Read_Inst_Num
                : HotLoopInstList::A_LDS_Read_Inst_Num / 2;

        constexpr auto num_ds_write_inst_a = HotLoopInstList::A_LDS_Write_Inst_Num;

        constexpr auto num_buffer_load_inst_a = HotLoopInstList::A_Buffer_Load_Inst_Num;
        constexpr auto num_buffer_load_inst_b = HotLoopInstList::B_Buffer_Load_Inst_Num;

        static_assert(num_buffer_load_inst_a == num_ds_write_inst_a);

        constexpr auto num_mfma_inst = HotLoopInstList::C_MFMA_Inst_Num;
        constexpr auto mfma_cycle    = HotLoopInstList::C_MFMA_Inst_Cycle;

        constexpr auto ds_read_a_issue_cycle =
            HotLoopInstList::A_LDS_Read_Width * sizeof(ADataType) == 16 ? 8 : 4;
        constexpr auto ds_read_a_mfma_rate =
            math::integer_divide_ceil(mfma_cycle - 4, 2 * ds_read_a_issue_cycle);

        constexpr auto num_total_stages = MRepeat;

        // Group num_mfma_perstage num_ds_read_a_perstage
        // since we want to reuse a local register buffer
        constexpr auto num_mfma_perstage      = num_mfma_inst / MRepeat;
        constexpr auto num_ds_read_a_perstage = num_ds_read_inst_a / MRepeat;

        constexpr auto num_ds_read_a_mfma_perstage =
            math::integer_divide_ceil(num_ds_read_a_perstage, ds_read_a_mfma_rate);

        constexpr auto total_buffer_loads = num_buffer_load_inst_a + num_buffer_load_inst_b;
        constexpr auto stages_available   = MRepeat - DS_READ_A_PREFETCH_STAGES;

        constexpr auto stage_loads = compute_stage_loads(total_buffer_loads, stages_available);

        constexpr auto buffer_load_perstage_more = stage_loads.first;
        constexpr auto buffer_load_perstage_less = stage_loads.second;

        constexpr auto buffer_load_stages_more = total_buffer_loads % stages_available;

        constexpr auto buffer_b_heavy_loads = buffer_load_perstage_more * buffer_load_stages_more;
        constexpr auto buffer_b_remaining =
            num_buffer_load_inst_b - buffer_load_perstage_more * buffer_load_stages_more;

        constexpr auto buffer_load_b_stages =
            buffer_b_heavy_loads > num_buffer_load_inst_b
                ? num_buffer_load_inst_b / buffer_load_perstage_more
                : (buffer_load_stages_more + buffer_b_remaining / buffer_load_perstage_less);

        constexpr auto buffer_load_a_stages =
            num_total_stages - DS_READ_A_PREFETCH_STAGES - buffer_load_b_stages;

        static_assert(buffer_load_a_stages > 0,
                      "The buffer load a stages should always have a value over 0.");

        constexpr auto buffer_load_issue_point_interval_more =
            math::integer_divide_ceil(num_mfma_perstage, buffer_load_perstage_more);
        constexpr auto buffer_load_issue_point_interval_less =
            buffer_load_perstage_less == 0
                ? INT32_MAX
                : math::integer_divide_ceil(num_mfma_perstage, buffer_load_perstage_less);
        constexpr auto buffer_load_issue_point_a = num_mfma_perstage >= 3 ? 1 : 0;

        // B global read
        static_for<0, buffer_load_b_stages, 1>{}([&](auto i) {
            static_for<0, num_mfma_perstage, 1>{}([&](auto imfma) {
                __builtin_amdgcn_sched_group_barrier(SCHED_GROUP_MFMA, 1, 0);

                if constexpr(((i < buffer_load_stages_more) &&
                              (imfma % buffer_load_issue_point_interval_more == 0)) ||
                             ((i >= buffer_load_stages_more) &&
                              (imfma % buffer_load_issue_point_interval_less == 0)))
                {
                    __builtin_amdgcn_sched_group_barrier(SCHED_GROUP_VMEM, 1, 0);
                }

                if constexpr(imfma >= (num_mfma_perstage - num_ds_read_a_mfma_perstage))
                {
                    __builtin_amdgcn_sched_group_barrier(
                        SCHED_GROUP_LDS_READ, ds_read_a_mfma_rate, 0);
                }
            });
        });

        // A global read + A local write
        static_for<0, buffer_load_a_stages, 1>{}([&](auto i) {
            static_for<0, num_mfma_perstage, 1>{}([&](auto imfma) {
                __builtin_amdgcn_sched_group_barrier(SCHED_GROUP_MFMA, 1, 0);
                if constexpr((((i + buffer_load_b_stages) < buffer_load_stages_more) &&
                              (imfma % buffer_load_issue_point_interval_more == 0)) ||
                             (((i + buffer_load_b_stages) >= buffer_load_stages_more) &&
                              (imfma % buffer_load_issue_point_interval_less == 0)))
                {
                    __builtin_amdgcn_sched_group_barrier(SCHED_GROUP_LDS_WRITE, 1, 0);
                }
                if constexpr((((i + buffer_load_b_stages) < buffer_load_stages_more) &&
                              (imfma % buffer_load_issue_point_interval_more ==
                               buffer_load_issue_point_a)) ||
                             (((i + buffer_load_b_stages) >= buffer_load_stages_more) &&
                              (imfma % buffer_load_issue_point_interval_less ==
                               buffer_load_issue_point_a)))
                {
                    __builtin_amdgcn_sched_group_barrier(SCHED_GROUP_VMEM, 1, 0);
                }
                if constexpr(imfma >= (num_mfma_perstage - num_ds_read_a_mfma_perstage))
                {
                    __builtin_amdgcn_sched_group_barrier(
                        SCHED_GROUP_LDS_READ, ds_read_a_mfma_rate, 0);
                }
            });
        });

        // lds synchronization, prefetch next loop local A
        static_for<0, DS_READ_A_PREFETCH_STAGES, 1>{}([&](auto i) {
            ignore = i;
            static_for<0, num_mfma_perstage, 1>{}([&](auto imfma) {
                __builtin_amdgcn_sched_group_barrier(SCHED_GROUP_MFMA, 1, 0);
                if constexpr(imfma >= (num_mfma_perstage - num_ds_read_a_mfma_perstage))
                {
                    __builtin_amdgcn_sched_group_barrier(
                        SCHED_GROUP_LDS_READ, ds_read_a_mfma_rate, 0);
                }
            });
        });
    }

    template <bool HasMainLoop,
              TailNumber TailNum,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename CThreadBuffer>
    __device__ void Run(const AGridDesc& a_grid_desc,
                        const ABlockDesc& a_block_desc,
                        ABlockTransfer& a_blockwise_copy,
                        const AGridBuffer& a_grid_buf,
                        ABlockBuffer& a_block_buf,
                        const ABlockTransferStep& a_block_copy_step,
                        const BGridDesc& b_grid_desc,
                        BBlockTransfer& b_blockwise_copy,
                        const BGridBuffer& b_grid_buf,
                        BBlockBuffer& b_block_buf,
                        const BBlockTransferStep& b_block_copy_step,
                        CThreadBuffer& c_thread_buf,
                        index_t num_loop) const
    {
        ignore = b_block_buf;
        __builtin_amdgcn_sched_barrier(0);
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataType>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataType>(
            b_thread_desc_.GetElementSpaceSize());

        StaticallyIndexedArray<decltype(b_thread_buf), Number<2>{}> b_thread_bufs;
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0);

        // Global prefetch A1 B1
        b_blockwise_copy.Run(b_grid_desc,
                             b_grid_buf,
                             b_block_desc_n0_n1_k0_k1,
                             b_block_origin_idx,
                             b_thread_bufs(I0));
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        __builtin_amdgcn_sched_barrier(0);

        // Local prefill A1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(I0));

        // Global prefetch A2
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

        // Local prefetch A1
        block_sync_lds();
        static_for<0, DS_READ_A_PREFETCH_STAGES, 1>{}([&](auto m0) {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, KGroup, 1>{}([&](auto kg0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k0_k1_k2,
                                       make_tuple(m0, I0, I0, Number<k0 * KGroup + kg0>{}, I0, I0),
                                       a_block_buf.At(I0),
                                       a_thread_desc_,
                                       make_tuple(m0, I0, I0, k0, I0, Number<kg0 * A_K1>{}),
                                       a_thread_buf);
                });
            });
        });

        // Initialize C
        c_thread_buf.Clear();

        __builtin_amdgcn_sched_barrier(0);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                auto LoopFunc = [&](auto mfma_reg_buf, auto local_read_buf) {
                    b_blockwise_copy.Run(b_grid_desc,
                                         b_grid_buf,
                                         b_block_desc_n0_n1_k0_k1,
                                         b_block_origin_idx,
                                         b_thread_bufs(local_read_buf));
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(local_read_buf));
                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, KRepeat, 1>{}([&](auto k0) {
                            static_for<0, NRepeat, 1>{}([&](auto n0) {
                                vector_type<ComputeDataType, KPack> a_thread_vec;
                                vector_type<ComputeDataType, KPack> b_thread_vec;

                                static_for<0, KPack, 1>{}([&](auto ik) {
                                    a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                            make_tuple((m0 + HotloopLocalBufSwitch * mfma_reg_buf) %
                                                           2,
                                                       I0,
                                                       I0,
                                                       k0,
                                                       I0,
                                                       ik))>{}];
                                    b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                        b_thread_bufs[mfma_reg_buf]
                                                     [Number<b_thread_desc_.CalculateOffset(
                                                         make_tuple(n0, I0, k0, ik))>{}];
                                });

                                using mfma_input_type =
                                    typename vector_type<ComputeDataType,
                                                         xdlops_gemm.K1PerXdlops>::type;

                                constexpr index_t c_offset =
                                    c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                                xdlops_gemm.Run(
                                    a_thread_vec.template AsType<mfma_input_type>(),
                                    b_thread_vec.template AsType<mfma_input_type>(),
                                    c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                            });
                        });

                        if constexpr(m0.value == (MRepeat - 2))
                        {
                            block_sync_lds();

                            static_for<0, KRepeat, 1>{}([&](auto k0) {
                                static_for<0, KGroup, 1>{}([&](auto kg0) {
                                    a_thread_copy_.Run(
                                        a_block_desc_m0_m1_m2_k0_k1_k2,
                                        make_tuple(Number<0>{},
                                                   I0,
                                                   I0,
                                                   Number<k0 * KGroup + kg0>{},
                                                   I0,
                                                   I0),
                                        a_block_buf.At(local_read_buf),
                                        a_thread_desc_,
                                        make_tuple(
                                            Number<(m0 + 2 + HotloopLocalBufSwitch * mfma_reg_buf) %
                                                   2>{},
                                            I0,
                                            I0,
                                            k0,
                                            I0,
                                            Number<kg0 * A_K1>{}),
                                        a_thread_buf);
                                });
                            });
                        }
                        else if constexpr(m0.value == (MRepeat - 1))
                        {
                            static_for<0, KRepeat, 1>{}([&](auto k0) {
                                static_for<0, KGroup, 1>{}([&](auto kg0) {
                                    a_thread_copy_.Run(
                                        a_block_desc_m0_m1_m2_k0_k1_k2,
                                        make_tuple(Number<(m0 + 2) % MRepeat>{},
                                                   I0,
                                                   I0,
                                                   Number<k0 * KGroup + kg0>{},
                                                   I0,
                                                   I0),
                                        a_block_buf.At(local_read_buf),
                                        a_thread_desc_,
                                        make_tuple(
                                            Number<(m0 + 2 + HotloopLocalBufSwitch * mfma_reg_buf) %
                                                   2>{},
                                            I0,
                                            I0,
                                            k0,
                                            I0,
                                            Number<kg0 * A_K1>{}),
                                        a_thread_buf);
                                });
                            });
                        }
                        else
                        {
                            static_for<0, KRepeat, 1>{}([&](auto k0) {
                                static_for<0, KGroup, 1>{}([&](auto kg0) {
                                    a_thread_copy_.Run(
                                        a_block_desc_m0_m1_m2_k0_k1_k2,
                                        make_tuple(Number<(m0 + 2) % MRepeat>{},
                                                   I0,
                                                   I0,
                                                   Number<k0 * KGroup + kg0>{},
                                                   I0,
                                                   I0),
                                        a_block_buf.At(mfma_reg_buf),
                                        a_thread_desc_,
                                        make_tuple(
                                            Number<(m0 + 2 + HotloopLocalBufSwitch * mfma_reg_buf) %
                                                   2>{},
                                            I0,
                                            I0,
                                            k0,
                                            I0,
                                            Number<kg0 * A_K1>{}),
                                        a_thread_buf);
                                });
                            });
                        }
                    });
                    HotLoopScheduler();
                };

                LoopFunc(I0, I1);
                LoopFunc(I1, I0);

                i += 2;
            } while(i < (num_loop - 2));
        }
        // tail
        if constexpr(TailNum == TailNumber::Even)
        {
            b_blockwise_copy.Run(b_grid_desc,
                                 b_grid_buf,
                                 b_block_desc_n0_n1_k0_k1,
                                 b_block_origin_idx,
                                 b_thread_bufs(I1));
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(I1));

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<ComputeDataType, KPack> a_thread_vec;
                        vector_type<ComputeDataType, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0 % 2, I0, I0, k0, I0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                b_thread_bufs[I0][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataType, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });

                if constexpr(m0.value == (MRepeat - 2))
                {
                    block_sync_lds();

                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        static_for<0, KGroup, 1>{}([&](auto kg0) {
                            a_thread_copy_.Run(
                                a_block_desc_m0_m1_m2_k0_k1_k2,
                                make_tuple(
                                    Number<0>{}, I0, I0, Number<k0 * KGroup + kg0>{}, I0, I0),
                                a_block_buf.At(I1),
                                a_thread_desc_,
                                make_tuple(
                                    Number<(m0 + 2) % 2>{}, I0, I0, k0, I0, Number<kg0 * A_K1>{}),
                                a_thread_buf);
                        });
                    });
                }
                else if constexpr(m0.value == (MRepeat - 1))
                {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        static_for<0, KGroup, 1>{}([&](auto kg0) {
                            a_thread_copy_.Run(
                                a_block_desc_m0_m1_m2_k0_k1_k2,
                                make_tuple(Number<(m0 + 2) % MRepeat>{},
                                           I0,
                                           I0,
                                           Number<k0 * KGroup + kg0>{},
                                           I0,
                                           I0),
                                a_block_buf.At(I1),
                                a_thread_desc_,
                                make_tuple(
                                    Number<(m0 + 2) % 2>{}, I0, I0, k0, I0, Number<kg0 * A_K1>{}),
                                a_thread_buf);
                        });
                    });
                }
                else
                {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        static_for<0, KGroup, 1>{}([&](auto kg0) {
                            a_thread_copy_.Run(
                                a_block_desc_m0_m1_m2_k0_k1_k2,
                                make_tuple(Number<(m0 + 2) % MRepeat>{},
                                           I0,
                                           I0,
                                           Number<k0 * KGroup + kg0>{},
                                           I0,
                                           I0),
                                a_block_buf.At(I0),
                                a_thread_desc_,
                                make_tuple(
                                    Number<(m0 + 2) % 2>{}, I0, I0, k0, I0, Number<kg0 * A_K1>{}),
                                a_thread_buf);
                        });
                    });
                }
            });

            HotLoopScheduler();

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<ComputeDataType, KPack> a_thread_vec;
                        vector_type<ComputeDataType, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(make_tuple(
                                    (m0 + HotloopLocalBufSwitch) % 2, I0, I0, k0, I0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                b_thread_bufs[I1][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataType, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });

                if constexpr(m0.value < (MRepeat - 2))
                {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        static_for<0, KGroup, 1>{}([&](auto kg0) {
                            a_thread_copy_.Run(
                                a_block_desc_m0_m1_m2_k0_k1_k2,
                                make_tuple(
                                    Number<m0 + 2>{}, I0, I0, Number<k0 * KGroup + kg0>{}, I0, I0),
                                a_block_buf.At(I1),
                                a_thread_desc_,
                                make_tuple(Number<(m0 + 2 + HotloopLocalBufSwitch) % 2>{},
                                           I0,
                                           I0,
                                           k0,
                                           I0,
                                           Number<kg0 * A_K1>{}),
                                a_thread_buf);
                        });
                    });
                }
            });

            HotLoopScheduler();
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<ComputeDataType, KPack> a_thread_vec;
                        vector_type<ComputeDataType, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0 % 2, I0, I0, k0, I0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                b_thread_bufs[I0][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataType, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });

                if constexpr(m0.value < (MRepeat - 2))
                {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        static_for<0, KGroup, 1>{}([&](auto kg0) {
                            a_thread_copy_.Run(
                                a_block_desc_m0_m1_m2_k0_k1_k2,
                                make_tuple(
                                    Number<m0 + 2>{}, I0, I0, Number<k0 * KGroup + kg0>{}, I0, I0),
                                a_block_buf.At(I0),
                                a_thread_desc_,
                                make_tuple(
                                    Number<(m0 + 2) % 2>{}, I0, I0, k0, I0, Number<kg0 * A_K1>{}),
                                a_thread_buf);
                        });
                    });
                }
            });
        }
    }

    protected:
    // MRepeat MWave MLane KRepeat KLane KPack
    // KRepeat -> MRepeat-> Mwave->KLane->MLane->KPack
    // Reduce the vgpr usage here.
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(I2, I1, I1, Number<KRepeat>{}, I1, Number<KPack>{}));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<ADataType,
                                                         ComputeDataType,
                                                         decltype(a_block_desc_m0_m1_m2_k0_k1_k2),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, 1, 1, KPack / KGroup>,
                                                         Sequence<0, 1, 2, 3, 4, 5>,
                                                         5,
                                                         A_K1,
                                                         A_K1>;

    AThreadCopy a_thread_copy_{Base::CalculateAThreadOriginDataIndex6D()};

    static constexpr auto b_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<NRepeat>{}, I1, Number<KRepeat>{}, Number<KPack>{}));

    static constexpr BTileDesc b_block_desc_n0_n1_k0_k1;

    using Base::c_thread_desc_;
};

} // namespace ck
