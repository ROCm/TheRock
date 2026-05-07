// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp"

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
          typename ComputeTypeA,
          typename ComputeTypeB,
          typename AccDataType,
          typename AWmmaTileDesc,
          typename BWmmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          index_t KInner,
          bool TransposeC = false,
          bool BSkipLDS   = false>
struct BlockwiseGemmWmmaops_pipeline_v3
{
};

template <index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename ComputeTypeA,
          typename ComputeTypeB,
          typename AccDataType,
          typename AWmmaTileDesc,
          typename BWmmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          index_t KInner,
          bool TransposeC>
struct BlockwiseGemmWmmaops_pipeline_v3<BlockGemmPipelineScheduler::Intrawave,
                                        BlockSize,
                                        ADataType,
                                        BDataType,
                                        ComputeTypeA,
                                        ComputeTypeB,
                                        AccDataType,
                                        AWmmaTileDesc,
                                        BWmmaTileDesc,
                                        ABlockTransferSrcScalarPerVector,
                                        BBlockTransferSrcScalarPerVector,
                                        MPerBlock,
                                        NPerBlock,
                                        KPerBlock,
                                        MPerWmma,
                                        NPerWmma,
                                        MRepeat,
                                        NRepeat,
                                        KPack,
                                        KInner,
                                        TransposeC,
                                        false>
    : BlockwiseGemmWmmaops_pipeline_base<BlockSize,
                                         ADataType,
                                         BDataType,
                                         ComputeTypeA,
                                         ComputeTypeB,
                                         AccDataType,
                                         AWmmaTileDesc,
                                         BWmmaTileDesc,
                                         ABlockTransferSrcScalarPerVector,
                                         BBlockTransferSrcScalarPerVector,
                                         MPerBlock,
                                         NPerBlock,
                                         KPerBlock,
                                         MPerWmma,
                                         NPerWmma,
                                         MRepeat,
                                         NRepeat,
                                         KPack,
                                         KInner,
                                         TransposeC>
{
    using Base = BlockwiseGemmWmmaops_pipeline_base<BlockSize,
                                                    ADataType,
                                                    BDataType,
                                                    ComputeTypeA,
                                                    ComputeTypeB,
                                                    AccDataType,
                                                    AWmmaTileDesc,
                                                    BWmmaTileDesc,
                                                    ABlockTransferSrcScalarPerVector,
                                                    BBlockTransferSrcScalarPerVector,
                                                    MPerBlock,
                                                    NPerBlock,
                                                    KPerBlock,
                                                    MPerWmma,
                                                    NPerWmma,
                                                    MRepeat,
                                                    NRepeat,
                                                    KPack,
                                                    KInner,
                                                    TransposeC>;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::I3;

    using Base::A_K1;
    using Base::A_KRow;
    using Base::B_K1;
    using Base::B_KRow;
    using Base::KRepeat;
    using Base::WmmaK;

    using Base::wmma_gemm;
    using typename Base::HotLoopInstList;

    using Base::CalculateCThreadOriginDataIndex;
    using Base::
        GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs;
    using Base::GetCThreadBuffer;
    using Base::
        GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs;
    using Base::
        GetCThreadDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs;

    using Base::a_block_desc_k0_m0_m1_m2_k1;
    using Base::b_block_desc_k0_n0_n1_n2_k1;

    using typename Base::Empty;

    static constexpr index_t PrefetchStages  = 2;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    __host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
    {
        if(BlockHasHotloop(num_loop))
        {
            return TailNumber::Full;
        }
        else
        {
            if(num_loop == 1)
            {
                return TailNumber::Odd;
            }
            else
            {
                return TailNumber::Even;
            }
        }
    }

    __device__ static constexpr auto HotLoopScheduler()
    {
        // TODO: Calculation of the number of instructions may require changes for WMMA
        /*
        // A/B split schedule
        // compiler is likely to use ds_read2 when instruction width smaller than 16bytes
        constexpr auto num_ds_read_inst_a =
            HotLoopInstList::A_LDS_Read_Width * sizeof(ADataType) == 16
                ? HotLoopInstList::A_LDS_Read_Inst_Num
                : HotLoopInstList::A_LDS_Read_Inst_Num / 2;
        constexpr auto num_ds_read_inst_b =
            HotLoopInstList::B_LDS_Read_Width * sizeof(BDataType) == 16
                ? HotLoopInstList::B_LDS_Read_Inst_Num
                : HotLoopInstList::B_LDS_Read_Inst_Num / 2;

        constexpr auto num_ds_write_inst_a = HotLoopInstList::A_LDS_Write_Inst_Num;
        constexpr auto num_ds_write_inst_b = HotLoopInstList::B_LDS_Write_Inst_Num;

        constexpr auto num_buffer_load_inst_a = HotLoopInstList::A_Buffer_Load_Inst_Num;
        constexpr auto num_buffer_load_inst_b = HotLoopInstList::B_Buffer_Load_Inst_Num;

        constexpr auto num_wmma_inst = HotLoopInstList::C_WMMA_Inst_Num;

        constexpr auto wmma_cycle = NPerWmma == 16 ? 16 : 32;
        constexpr auto ds_read_a_issue_cycle =
            HotLoopInstList::A_LDS_Read_Width * sizeof(ADataType) == 16 ? 8 : 4;
        constexpr auto ds_read_b_issue_cycle =
            HotLoopInstList::B_LDS_Read_Width * sizeof(BDataType) == 16 ? 8 : 4;
        constexpr auto ds_read_a_wmma_rate =
            (wmma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
        constexpr auto ds_read_b_wmma_rate =
            (wmma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);

        constexpr auto num_dsread_a_wmma =
            (num_ds_read_inst_a + ds_read_a_wmma_rate - 1) / ds_read_a_wmma_rate;
        constexpr auto num_dsread_b_wmma =
            (num_ds_read_inst_b + ds_read_b_wmma_rate - 1) / ds_read_b_wmma_rate;

        // stage 1
        // Separate this part?
        // constexpr auto num_wmma_per_ds_read = sizeof(ComputeDataType) / sizeof(ADataType) >
        //                                               sizeof(ComputeDataType) / sizeof(BDataType)
        //                                           ? sizeof(ComputeDataType) / sizeof(ADataType)
        //                                           : sizeof(ComputeDataType) / sizeof(BDataType);
        constexpr auto num_wmma_stage1 = num_wmma_inst - (num_dsread_a_wmma + num_dsread_b_wmma);
        constexpr auto num_wmma_per_issue =
            num_wmma_stage1 / (num_buffer_load_inst_a + num_buffer_load_inst_b);
        constexpr auto num_dswrite_per_issue_a = num_ds_write_inst_a / num_buffer_load_inst_a;
        constexpr auto num_dswrite_per_issue_b = num_ds_write_inst_b / num_buffer_load_inst_b;

        static_for<0, num_buffer_load_inst_a, 1>{}([&](auto i) {
            ignore = i;
            static_for<0, num_dswrite_per_issue_a, 1>{}([&](auto idswrite) {
                ignore = idswrite;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // WMMA
            });
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(
                0x008, num_wmma_per_issue - num_dswrite_per_issue_a, 0); // WMMA
        });
        static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
            ignore = i;
            static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
                ignore = idswrite;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // WMMA
            });
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(
                0x008, num_wmma_per_issue - num_dswrite_per_issue_b, 0); // WMMA
        });

        // stage 2
        static_for<0, num_dsread_a_wmma, 1>{}([&](auto i) {
            if constexpr((num_ds_read_inst_a - (i + 1) * ds_read_a_wmma_rate) >=
                         ds_read_a_wmma_rate)
            {
                __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_wmma_rate, 0); // DS read
            }
            else
            {
                __builtin_amdgcn_sched_group_barrier(0x100,
                                                     num_ds_read_inst_a - (num_dsread_a_wmma - 1) *
                                                                              ds_read_a_wmma_rate,
                                                     0); // DS read
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // WMMA
        });

        static_for<0, num_dsread_b_wmma, 1>{}([&](auto i) {
            if constexpr((num_ds_read_inst_b - (i + 1) * ds_read_b_wmma_rate) >=
                         ds_read_b_wmma_rate)
            {
                __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_wmma_rate, 0); // DS read
            }
            else
            {
                __builtin_amdgcn_sched_group_barrier(0x100,
                                                     num_ds_read_inst_b - (num_dsread_b_wmma - 1) *
                                                                              ds_read_b_wmma_rate,
                                                     0); // DS read
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // WMMA
        });
        */
    }

    template <typename ABlockBuffer,
              typename AThreadBuffer,
              typename BBlockBuffer,
              typename BThreadBuffer,
              typename BScaleStruct>
    __device__ inline void LocalLoad(ABlockBuffer& a_block_buf,
                                     AThreadBuffer& a_thread_buf,
                                     BBlockBuffer& b_block_buf,
                                     BThreadBuffer& b_thread_buf,
                                     BScaleStruct& b_scale_struct) const
    {
        static_for<0, KRepeat, 1>{}([&](auto k0) {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
                                   make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                   a_block_buf,
                                   a_thread_desc_,
                                   make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                   a_thread_buf);
            });

            if constexpr(ck::is_same_v<BScaleStruct, Empty>)
            {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(b_block_desc_k0_n0_n1_n2_k1,
                                       make_tuple(I0, n0, k0, I0, I0, I0, I0),
                                       b_block_buf,
                                       b_thread_desc_,
                                       make_tuple(I0, n0, k0, I0, I0, I0, I0),
                                       b_thread_buf);
                });
            }
            else
            {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(b_block_desc_k0_n0_n1_n2_k1,
                                       make_tuple(I0, n0, k0, I0, I0, I0, I0),
                                       b_block_buf,
                                       b_scale_struct.scale_thread_bufs(
                                           I0)[Number<n0 * BScaleStruct::num_scale_k_block +
                                                      k0 / BScaleStruct::num_scale_krepeat>{}],
                                       b_thread_desc_,
                                       make_tuple(I0, n0, k0, I0, I0, I0, I0),
                                       b_thread_buf);
                });
            }
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
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename CThreadBuffer,
              typename AScaleStruct,
              typename BScaleStruct,
              typename enable_if<ck::is_same_v<AScaleStruct, Empty>, bool>::type = false>
    __device__ void Run(const AGridDesc& a_grid_desc,
                        const ABlockDesc& a_block_desc,
                        ABlockTransfer& a_blockwise_copy,
                        const AGridBuffer& a_grid_buf,
                        ABlockBuffer& a_block_buf,
                        const ABlockTransferStep& a_block_copy_step,
                        const BGridDesc& b_grid_desc,
                        const BBlockDesc& b_block_desc,
                        BBlockTransfer& b_blockwise_copy,
                        const BGridBuffer& b_grid_buf,
                        BBlockBuffer& b_block_buf,
                        const BBlockTransferStep& b_block_copy_step,
                        CThreadBuffer& c_thread_buf,
                        AScaleStruct&,
                        BScaleStruct& b_scale_struct,
                        index_t num_loop,
                        index_t num_loop_per_scale) const
    {
        __builtin_amdgcn_sched_barrier(0);

        constexpr index_t KPerWaveBlock = wmma_gemm.GetKPerWaveBlk();

        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeB>(
            b_thread_desc_.GetElementSpaceSize());

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Scales global load
        b_scale_struct.template GlobalLoad<0>(num_loop_per_scale == 1);

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // Global prefetch 2, perform when at least 2 loops exist.
        if constexpr(TailNum == TailNumber::Even || TailNum == TailNumber::Full)
        {
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        }

        // Initialize C
        c_thread_buf.Clear();

        // Local prefetch 1
        block_sync_lds();

        LocalLoad(a_block_buf, a_thread_buf, b_block_buf, b_thread_buf, b_scale_struct);

        __builtin_amdgcn_sched_barrier(0);

        // Main body, perform when at least 3 loops exist.
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                block_sync_lds();

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                b_scale_struct.template GlobalLoad<0>((i + 2) % num_loop_per_scale == 0);

                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            static_for<0, KInner, 1>{}([&](auto k_inner) {
                                vector_type<ComputeTypeA, KPack / A_KRow / KInner> a_thread_vec;
                                vector_type<ComputeTypeB, KPack / B_KRow / KInner> b_thread_vec;

                                static_for<0, KPack / A_KRow / KInner, 1>{}([&](auto ik) {
                                    constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                    a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                            make_tuple(Number<kk / A_K1>{},
                                                       m0,
                                                       k0,
                                                       I0,
                                                       I0,
                                                       I0,
                                                       Number<kk % A_K1>{}))>{}];
                                });
                                static_for<0, KPack / B_KRow / KInner, 1>{}([&](auto ik) {
                                    constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                    b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                        b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                            make_tuple(Number<kk / B_K1>{},
                                                       n0,
                                                       k0,
                                                       I0,
                                                       I0,
                                                       I0,
                                                       Number<kk % B_K1>{}))>{}];
                                });

                                using wmma_input_type_a =
                                    typename vector_type<ComputeTypeA, WmmaK / A_KRow>::type;
                                using wmma_input_type_b =
                                    typename vector_type<ComputeTypeB, WmmaK / B_KRow>::type;

                                constexpr index_t c_offset =
                                    c_thread_desc_.CalculateOffset(make_tuple(m0, n0, I0));

                                wmma_gemm.Run(
                                    a_thread_vec.template AsType<wmma_input_type_a>(),
                                    b_thread_vec.template AsType<wmma_input_type_b>(),
                                    c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                            });
                        });
                    });
                });

                block_sync_lds();

                LocalLoad(a_block_buf, a_thread_buf, b_block_buf, b_thread_buf, b_scale_struct);

                HotLoopScheduler();
                __builtin_amdgcn_sched_barrier(0);

                i += 1;
            } while(i < (num_loop - 2));
        }

        // Pre-tail, perform when at least 2 loops exist.
        if constexpr(TailNum == TailNumber::Even || TailNum == TailNumber::Full)
        {
            block_sync_lds();

            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

            // No RunRead or MoveSrcSliceWindow here, already finished them all!

            b_scale_struct.template GlobalLoad<0>(num_loop % num_loop_per_scale == 0);

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        static_for<0, KInner, 1>{}([&](auto k_inner) {
                            vector_type<ComputeTypeA, KPack / A_KRow / KInner> a_thread_vec;
                            vector_type<ComputeTypeB, KPack / B_KRow / KInner> b_thread_vec;

                            static_for<0, KPack / A_KRow / KInner, 1>{}([&](auto ik) {
                                constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(Number<kk / A_K1>{},
                                                   m0,
                                                   k0,
                                                   I0,
                                                   I0,
                                                   I0,
                                                   Number<kk % A_K1>{}))>{}];
                            });
                            static_for<0, KPack / B_KRow / KInner, 1>{}([&](auto ik) {
                                constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(Number<kk / B_K1>{},
                                                   n0,
                                                   k0,
                                                   I0,
                                                   I0,
                                                   I0,
                                                   Number<kk % B_K1>{}))>{}];
                            });

                            using wmma_input_type_a =
                                typename vector_type<ComputeTypeA, WmmaK / A_KRow>::type;
                            using wmma_input_type_b =
                                typename vector_type<ComputeTypeB, WmmaK / B_KRow>::type;

                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, I0));

                            wmma_gemm.Run(a_thread_vec.template AsType<wmma_input_type_a>(),
                                          b_thread_vec.template AsType<wmma_input_type_b>(),
                                          c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        });
                    });
                });
            });

            block_sync_lds();

            LocalLoad(a_block_buf, a_thread_buf, b_block_buf, b_thread_buf, b_scale_struct);

            HotLoopScheduler();
            __builtin_amdgcn_sched_barrier(0);
        }

        // Tail, always perform.
        {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        static_for<0, KInner, 1>{}([&](auto k_inner) {
                            vector_type<ComputeTypeA, KPack / A_KRow / KInner> a_thread_vec;
                            vector_type<ComputeTypeB, KPack / B_KRow / KInner> b_thread_vec;

                            static_for<0, KPack / A_KRow / KInner, 1>{}([&](auto ik) {
                                constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(Number<kk / A_K1>{},
                                                   m0,
                                                   k0,
                                                   I0,
                                                   I0,
                                                   I0,
                                                   Number<kk % A_K1>{}))>{}];
                            });
                            static_for<0, KPack / B_KRow / KInner, 1>{}([&](auto ik) {
                                constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(Number<kk / B_K1>{},
                                                   n0,
                                                   k0,
                                                   I0,
                                                   I0,
                                                   I0,
                                                   Number<kk % B_K1>{}))>{}];
                            });

                            using wmma_input_type_a =
                                typename vector_type<ComputeTypeA, WmmaK / A_KRow>::type;
                            using wmma_input_type_b =
                                typename vector_type<ComputeTypeB, WmmaK / B_KRow>::type;

                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, I0));

                            wmma_gemm.Run(a_thread_vec.template AsType<wmma_input_type_a>(),
                                          b_thread_vec.template AsType<wmma_input_type_b>(),
                                          c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                        });
                    });
                });
            });
            // Let's leak last WMMA block to epilogue region, cover the potential lds-shuffle
            // latency
            // __builtin_amdgcn_sched_barrier(0);
        }
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
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename CThreadBuffer,
              typename AScaleStruct,
              typename BScaleStruct,
              typename enable_if<!ck::is_same_v<AScaleStruct, Empty> &&
                                     !ck::is_same_v<BScaleStruct, Empty>,
                                 bool>::type = false>
    __device__ void Run(const AGridDesc& a_grid_desc,
                        const ABlockDesc& a_block_desc,
                        ABlockTransfer& a_blockwise_copy,
                        const AGridBuffer& a_grid_buf,
                        ABlockBuffer& a_block_buf,
                        const ABlockTransferStep& a_block_copy_step,
                        const BGridDesc& b_grid_desc,
                        const BBlockDesc& b_block_desc,
                        BBlockTransfer& b_blockwise_copy,
                        const BGridBuffer& b_grid_buf,
                        BBlockBuffer& b_block_buf,
                        const BBlockTransferStep& b_block_copy_step,
                        CThreadBuffer& c_thread_buf,
                        AScaleStruct& a_scale_struct,
                        BScaleStruct& b_scale_struct,
                        index_t num_loop,
                        index_t num_loop_per_scale) const
    {
        __builtin_amdgcn_sched_barrier(0);

        constexpr index_t KPerWaveBlock = wmma_gemm.GetKPerWaveBlk();
        static constexpr auto NumScaleKBlock =
            Number<ck::math::max(AScaleStruct::num_slice_k, BScaleStruct::num_slice_k)>{};

        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeB>(
            b_thread_desc_.GetElementSpaceSize());

        using CScaleStruct  = typename Base::template CScale<AScaleStruct, BScaleStruct>;
        auto c_scale_struct = CScaleStruct{};

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Scales global load
        a_scale_struct.template GlobalLoad<0>(num_loop_per_scale == 1);
        b_scale_struct.template GlobalLoad<0>(num_loop_per_scale == 1);

        c_scale_struct.Load(a_scale_struct, b_scale_struct);

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // Global prefetch 2, perform when at least 2 loops exist.
        if constexpr(TailNum == TailNumber::Even || TailNum == TailNumber::Full)
        {
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        }

        // Initialize C
        c_thread_buf.Clear();

        // Local prefetch 1
        block_sync_lds();

        auto local_load_func = [&]() {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
                                       make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                       a_block_buf,
                                       a_thread_desc_,
                                       make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                       a_thread_buf);
                });
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(b_block_desc_k0_n0_n1_n2_k1,
                                       make_tuple(I0, n0, k0, I0, I0, I0, I0),
                                       b_block_buf,
                                       b_thread_desc_,
                                       make_tuple(I0, n0, k0, I0, I0, I0, I0),
                                       b_thread_buf);
                });
            });
        };

        local_load_func();

        __builtin_amdgcn_sched_barrier(0);

        // Main body, perform when at least 3 loops exist.
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                block_sync_lds();

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_scale_struct.template GlobalLoad<0>((i + 2) % num_loop_per_scale == 0);
                b_scale_struct.template GlobalLoad<0>((i + 2) % num_loop_per_scale == 0);

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        static_for<0, NumScaleKBlock, 1>{}([&](auto kscale0) {
                            c_scale_struct.Clear();
                            static_for<0, KRepeat / NumScaleKBlock, 1>{}([&](auto k0) {
                                static_for<0, KInner, 1>{}([&](auto k_inner) {
                                    vector_type<ComputeTypeA, KPack / A_KRow / KInner> a_thread_vec;
                                    vector_type<ComputeTypeB, KPack / B_KRow / KInner> b_thread_vec;

                                    static_for<0, KPack / A_KRow / KInner, 1>{}([&](auto ik) {
                                        constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                        constexpr index_t k_index =
                                            kscale0 * (KRepeat / NumScaleKBlock) + k0;
                                        a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                            a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                                make_tuple(Number<kk / A_K1>{},
                                                           m0,
                                                           k_index,
                                                           I0,
                                                           I0,
                                                           I0,
                                                           Number<kk % A_K1>{}))>{}];
                                    });
                                    static_for<0, KPack / B_KRow / KInner, 1>{}([&](auto ik) {
                                        constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                        constexpr index_t k_index =
                                            kscale0 * (KRepeat / NumScaleKBlock) + k0;
                                        b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                            b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                                make_tuple(Number<kk / B_K1>{},
                                                           n0,
                                                           k_index,
                                                           I0,
                                                           I0,
                                                           I0,
                                                           Number<kk % B_K1>{}))>{}];
                                    });

                                    using wmma_input_type_a =
                                        typename vector_type<ComputeTypeA, WmmaK / A_KRow>::type;
                                    using wmma_input_type_b =
                                        typename vector_type<ComputeTypeB, WmmaK / B_KRow>::type;

                                    wmma_gemm.Run(a_thread_vec.template AsType<wmma_input_type_a>(),
                                                  b_thread_vec.template AsType<wmma_input_type_b>(),
                                                  c_scale_struct.c_thread_buf_per_scale
                                                      .GetVectorTypeReference(Number<0>{}));
                                });
                            });
                            c_scale_struct.template UpdateCThreadBuf<kscale0, m0, n0>(c_thread_buf);
                        });
                    });
                });

                c_scale_struct.Load(a_scale_struct, b_scale_struct);
                block_sync_lds();

                local_load_func();

                HotLoopScheduler();
                __builtin_amdgcn_sched_barrier(0);

                i += 1;
            } while(i < (num_loop - 2));
        }

        // Pre-tail, perform when at least 2 loops exist.
        if constexpr(TailNum == TailNumber::Even || TailNum == TailNumber::Full)
        {
            block_sync_lds();

            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

            // No RunRead or MoveSrcSliceWindow here, already finished them all!
            a_scale_struct.template GlobalLoad<0>(num_loop % num_loop_per_scale == 0);
            b_scale_struct.template GlobalLoad<0>(num_loop % num_loop_per_scale == 0);

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, NumScaleKBlock, 1>{}([&](auto kscale0) {
                        c_scale_struct.Clear();
                        static_for<0, KRepeat / NumScaleKBlock, 1>{}([&](auto k0) {
                            static_for<0, KInner, 1>{}([&](auto k_inner) {
                                vector_type<ComputeTypeA, KPack / A_KRow / KInner> a_thread_vec;
                                vector_type<ComputeTypeB, KPack / B_KRow / KInner> b_thread_vec;

                                static_for<0, KPack / A_KRow / KInner, 1>{}([&](auto ik) {
                                    constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                    constexpr index_t k_index =
                                        kscale0 * (KRepeat / NumScaleKBlock) + k0;
                                    a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                            make_tuple(Number<kk / A_K1>{},
                                                       m0,
                                                       k_index,
                                                       I0,
                                                       I0,
                                                       I0,
                                                       Number<kk % A_K1>{}))>{}];
                                });
                                static_for<0, KPack / B_KRow / KInner, 1>{}([&](auto ik) {
                                    constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                    constexpr index_t k_index =
                                        kscale0 * (KRepeat / NumScaleKBlock) + k0;
                                    b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                        b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                            make_tuple(Number<kk / B_K1>{},
                                                       n0,
                                                       k_index,
                                                       I0,
                                                       I0,
                                                       I0,
                                                       Number<kk % B_K1>{}))>{}];
                                });

                                using wmma_input_type_a =
                                    typename vector_type<ComputeTypeA, WmmaK / A_KRow>::type;
                                using wmma_input_type_b =
                                    typename vector_type<ComputeTypeB, WmmaK / B_KRow>::type;

                                wmma_gemm.Run(
                                    a_thread_vec.template AsType<wmma_input_type_a>(),
                                    b_thread_vec.template AsType<wmma_input_type_b>(),
                                    c_scale_struct.c_thread_buf_per_scale.GetVectorTypeReference(
                                        Number<0>{}));
                            });
                        });
                        c_scale_struct.template UpdateCThreadBuf<kscale0, m0, n0>(c_thread_buf);
                    });
                });
            });

            c_scale_struct.Load(a_scale_struct, b_scale_struct);
            block_sync_lds();

            local_load_func();

            HotLoopScheduler();
            __builtin_amdgcn_sched_barrier(0);
        }

        // Tail, always perform.
        {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, NumScaleKBlock, 1>{}([&](auto kscale0) {
                        c_scale_struct.Clear();
                        static_for<0, KRepeat / NumScaleKBlock, 1>{}([&](auto k0) {
                            vector_type<ComputeTypeA, KPack / A_KRow / KInner> a_thread_vec;
                            vector_type<ComputeTypeB, KPack / B_KRow / KInner> b_thread_vec;
                            static_for<0, KInner, 1>{}([&](auto k_inner) {
                                static_for<0, KPack / A_KRow / KInner, 1>{}([&](auto ik) {
                                    constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                    constexpr index_t k_index =
                                        kscale0 * (KRepeat / NumScaleKBlock) + k0;
                                    a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                            make_tuple(Number<kk / A_K1>{},
                                                       m0,
                                                       k_index,
                                                       I0,
                                                       I0,
                                                       I0,
                                                       Number<kk % A_K1>{}))>{}];
                                });
                                static_for<0, KPack / B_KRow / KInner, 1>{}([&](auto ik) {
                                    constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                    constexpr index_t k_index =
                                        kscale0 * (KRepeat / NumScaleKBlock) + k0;
                                    b_thread_vec.template AsType<ComputeTypeB>()(ik) =
                                        b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                            make_tuple(Number<kk / B_K1>{},
                                                       n0,
                                                       k_index,
                                                       I0,
                                                       I0,
                                                       I0,
                                                       Number<kk % B_K1>{}))>{}];
                                });

                                using wmma_input_type_a =
                                    typename vector_type<ComputeTypeA, WmmaK / A_KRow>::type;
                                using wmma_input_type_b =
                                    typename vector_type<ComputeTypeB, WmmaK / B_KRow>::type;

                                wmma_gemm.Run(
                                    a_thread_vec.template AsType<wmma_input_type_a>(),
                                    b_thread_vec.template AsType<wmma_input_type_b>(),
                                    c_scale_struct.c_thread_buf_per_scale.GetVectorTypeReference(
                                        Number<0>{}));
                            });
                        });
                        c_scale_struct.template UpdateCThreadBuf<kscale0, m0, n0>(c_thread_buf);
                    });
                });
            });
            // Let's leak last WMMA block to epilogue region, cover the potential lds-shuffle
            // latency
            // __builtin_amdgcn_sched_barrier(0);
        }
    }

    protected:
    using Base::a_thread_copy_;
    using Base::a_thread_desc_;
    using Base::b_thread_copy_;
    using Base::b_thread_desc_;
    using Base::c_thread_desc_;
};

} // namespace ck
