// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp"

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
struct BlockwiseGemmXdlops_pipeline_v1_ab_scale
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
struct BlockwiseGemmXdlops_pipeline_v1_ab_scale<BlockGemmPipelineScheduler::Intrawave,
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
                                        KPack,
                                        true>

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
                                                   KPack,
                                                   true>;
    using Base::A_K1;
    using Base::B_K1;
    using Base::I0;
    using Base::I1;
    using Base::KRepeat;
    using Base::xdlops_gemm;
    using typename Base::HotLoopInstList;

    using Base::CalculateCThreadOriginDataIndex;
    using Base::CalculateCThreadOriginDataIndex8D;
    using Base::GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4;
    using Base::GetCThreadBuffer;
    using Base::GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4;
    using Base::GetWaveIdx;
    using Base::MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;

    using Base::a_block_desc_m0_m1_m2_k;
    using Base::b_block_desc_n0_n1_n2_k;

    using ComputeDataTypeBuf = typename Base::ComputeDataTypeBuf;

    static constexpr index_t AMmaKStride = xdlops_gemm.K0PerXdlops * KPack;
    static constexpr index_t BMmaKStride = xdlops_gemm.K0PerXdlops * KPack;

    static constexpr index_t PrefetchStages  = 2;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    // Force mfma not cross the scaleblock
    __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];

        const auto xdlops_a_idx = xdlops_gemm.CalculateAThreadOriginDataIndex();

        return make_tuple(0, waveId_m, xdlops_a_idx[I1], KPack * xdlops_a_idx[I0]);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_n = wave_idx[I1];

        const auto xdlops_b_idx = xdlops_gemm.CalculateBThreadOriginDataIndex();

        return make_tuple(0, waveId_n, xdlops_b_idx[I1], KPack * xdlops_b_idx[I0]);
    }

    __host__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
    {
        return num_loop == 1 ? TailNumber::Odd : TailNumber::Full;
    }

    __device__ static constexpr auto HotLoopScheduler()
    {
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

        constexpr auto num_mfma_inst = HotLoopInstList::C_MFMA_Inst_Num;

        constexpr auto mfma_cycle = HotLoopInstList::C_MFMA_Inst_Cycle;
        constexpr auto ds_read_a_issue_cycle =
            HotLoopInstList::A_LDS_Read_Width * sizeof(ADataType) == 16 ? 8 : 4;
        constexpr auto ds_read_b_issue_cycle =
            HotLoopInstList::B_LDS_Read_Width * sizeof(BDataType) == 16 ? 8 : 4;
        constexpr auto ds_read_a_mfma_rate =
            (mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
        constexpr auto ds_read_b_mfma_rate =
            (mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);

        constexpr auto num_dsread_a_mfma =
            (num_ds_read_inst_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
        constexpr auto num_dsread_b_mfma =
            (num_ds_read_inst_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;

        // stage 1
        // Separate this part?
        // constexpr auto num_mfma_per_ds_read = sizeof(ComputeDataTypeBuf) / sizeof(ADataType) >
        //                                               sizeof(ComputeDataTypeBuf) /
        //                                               sizeof(BDataType)
        //                                           ? sizeof(ComputeDataTypeBuf) /
        //                                           sizeof(ADataType) : sizeof(ComputeDataTypeBuf)
        //                                           / sizeof(BDataType);
        constexpr auto num_mfma_stage1 = num_mfma_inst - (num_dsread_a_mfma + num_dsread_b_mfma);
        constexpr auto num_mfma_per_issue =
            num_mfma_stage1 / (num_buffer_load_inst_a + num_buffer_load_inst_b);
        constexpr auto num_dswrite_per_issue_a = num_ds_write_inst_a / num_buffer_load_inst_a;
        constexpr auto num_dswrite_per_issue_b = num_ds_write_inst_b / num_buffer_load_inst_b;

        static_for<0, num_buffer_load_inst_a, 1>{}([&](auto i) {
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
        static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
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

        // stage 2
        static_for<0, num_dsread_a_mfma, 1>{}([&](auto i) {
            if constexpr((num_ds_read_inst_a - (i + 1) * ds_read_a_mfma_rate) >=
                         ds_read_a_mfma_rate)
            {
                __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
            }
            else
            {
                __builtin_amdgcn_sched_group_barrier(0x100,
                                                     num_ds_read_inst_a - (num_dsread_a_mfma - 1) *
                                                                              ds_read_a_mfma_rate,
                                                     0); // DS read
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
        });

        static_for<0, num_dsread_b_mfma, 1>{}([&](auto i) {
            if constexpr((num_ds_read_inst_b - (i + 1) * ds_read_b_mfma_rate) >=
                         ds_read_b_mfma_rate)
            {
                __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0); // DS read
            }
            else
            {
                __builtin_amdgcn_sched_group_barrier(0x100,
                                                     num_ds_read_inst_b - (num_dsread_b_mfma - 1) *
                                                                              ds_read_b_mfma_rate,
                                                     0); // DS read
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
        });
    }

    template <bool HasMainLoop,
              int NumKBlockPerScale,
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
              typename CScaleThreadDesc,
              typename CThreadBuffer,
              typename AScaleGridBuffer,
              typename AScaleGridDesc,
              typename AScaleThreadDesc,
              typename AScaleThreadTransfer,
              typename AScaleThreadTransferStep,
              typename BScaleGridBuffer,
              typename BScaleGridDesc,
              typename BScaleThreadDesc,
              typename BScaleThreadTransfer,
              typename BScaleThreadTransferStep>
    __device__ void Run(
        // ABlockCopy
        const AGridDesc& a_grid_desc,
        const ABlockDesc& a_block_desc,
        ABlockTransfer& a_blockwise_copy,
        const AGridBuffer& a_grid_buf,
        ABlockBuffer& a_block_buf,
        const ABlockTransferStep& a_block_copy_step,
        // BBlockCopy
        const BGridDesc& b_grid_desc,
        const BBlockDesc& b_block_desc,
        BBlockTransfer& b_blockwise_copy,
        const BGridBuffer& b_grid_buf,
        BBlockBuffer& b_block_buf,
        const BBlockTransferStep& b_block_copy_step,
        // CThread
        const CScaleThreadDesc& c_scale_thread_desc,
        CThreadBuffer& c_thread_buf,
        // AScaleThreadCopy
        const AScaleGridDesc& a_scale_grid_desc,
        const AScaleThreadDesc& a_scale_thread_desc,
        AScaleThreadTransfer& a_scale_thread_copy,
        const AScaleGridBuffer& a_scale_grid_buf,
        const AScaleThreadTransferStep& a_scale_thread_copy_step,
        // BScaleThreadCopy
        const BScaleGridDesc& b_scale_grid_desc,
        const BScaleThreadDesc& b_scale_thread_desc,
        BScaleThreadTransfer& b_scale_thread_copy,
        const BScaleGridBuffer& b_scale_grid_buf,
        const BScaleThreadTransferStep& b_scale_thread_copy_step,
        // num_loop
        index_t num_loop) const
    {
        __builtin_amdgcn_sched_barrier(0);
        // assume kperblock = scaleblockk
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
            b_thread_desc_.GetElementSpaceSize());
        auto a_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            a_scale_thread_desc.GetElementSpaceSize());
        auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            b_scale_thread_desc.GetElementSpaceSize());
        auto c_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            c_scale_thread_desc.GetElementSpaceSize());

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        static_for<0, MRepeat, 1>{}([&](auto m0) {
            a_scale_thread_copy.Run(a_scale_grid_desc,
                                    a_scale_grid_buf,
                                    a_scale_thread_desc,
                                    make_tuple(m0, I0),
                                    a_scale_thread_buf);
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<0>{}));
        });

        if constexpr(NumKBlockPerScale == 1)
        {
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<2>{}));
        }
        else
        {
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<1>{}));
        }

        b_scale_thread_copy.Run(b_scale_grid_desc,
                                b_scale_grid_buf,
                                b_scale_thread_desc,
                                make_tuple(I0, I0),
                                b_scale_thread_buf);

        b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc, b_scale_thread_copy_step);

        constexpr auto num_scale_k_block = CScaleThreadDesc{}.GetLength(Number<0>{});
        constexpr auto num_scale_m_block = CScaleThreadDesc{}.GetLength(Number<1>{});
        constexpr auto num_scale_n_block = CScaleThreadDesc{}.GetLength(Number<2>{});

        static_for<0, num_scale_m_block, 1>{}([&](auto m0) {
            static_for<0, num_scale_n_block, 1>{}([&](auto n0) {
                static_for<0, num_scale_k_block, 1>{}([&](auto k0) {
                    constexpr index_t c_offset =
                        CScaleThreadDesc{}.CalculateOffset(make_tuple(k0, m0, n0));
                    constexpr index_t a_offset =
                        AScaleThreadDesc{}.CalculateOffset(make_tuple(m0, k0));
                    constexpr index_t b_offset =
                        BScaleThreadDesc{}.CalculateOffset(make_tuple(n0, k0));

                    c_scale_thread_buf(Number<c_offset>{}) =
                        a_scale_thread_buf[Number<a_offset>{}] *
                        b_scale_thread_buf[Number<b_offset>{}];
                });
            });
        });

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // Global prefetch 2
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        static_for<0, MRepeat, 1>{}([&](auto m0) {
            a_scale_thread_copy.Run(a_scale_grid_desc,
                                    a_scale_grid_buf,
                                    a_scale_thread_desc,
                                    make_tuple(m0, I0),
                                    a_scale_thread_buf);
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<0>{}));
        });

        if constexpr(NumKBlockPerScale == 1)
        {
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<2>{}));
        }
        else
        {
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<1>{}));
        }

        b_scale_thread_copy.Run(b_scale_grid_desc,
                                b_scale_grid_buf,
                                b_scale_thread_desc,
                                make_tuple(I0, I0),
                                b_scale_thread_buf);

        b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc, b_scale_thread_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                  AccDataType,
                                  1,
                                  xdlops_gemm.GetRegSizePerXdlops(),
                                  true>
            c_thread_buf_per_scale;

        // Local prefetch 1
        block_sync_lds();
        static_for<0, KRepeat, 1>{}([&](auto k0) {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                   make_tuple(m0, I0, I0, Number<k0 * AMmaKStride>{}),
                                   a_block_buf,
                                   a_thread_desc_,
                                   make_tuple(m0, I0, k0, I0),
                                   a_thread_buf);
            });
            static_for<0, NRepeat, 1>{}([&](auto n0) {
                b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                   make_tuple(n0, I0, I0, Number<k0 * BMmaKStride>{}),
                                   b_block_buf,
                                   b_thread_desc_,
                                   make_tuple(n0, I0, k0, I0),
                                   b_thread_buf);
            });
        });

        __builtin_amdgcn_sched_barrier(0);

        // main body
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

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        static_for<0, num_scale_k_block, 1>{}([&](auto kscale0) {
                            static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                    .template AsType<AccDataType>()(Number<t>{}) = 0;
                            });
                            static_for<0, KRepeat / num_scale_k_block, 1>{}([&](auto k0) {
                                vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                                vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                                static_for<0, KPack, 1>{}([&](auto ik) {
                                    a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                            make_tuple(m0,
                                                       I0,
                                                       kscale0 * KRepeat / num_scale_k_block + k0,
                                                       ik))>{}];
                                    b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                        b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                            make_tuple(n0,
                                                       I0,
                                                       kscale0 * KRepeat / num_scale_k_block + k0,
                                                       ik))>{}];
                                });

                                using mfma_input_type =
                                    typename vector_type<ComputeDataTypeBuf,
                                                         xdlops_gemm.K1PerXdlops>::type;

                                xdlops_gemm.template Run<>(
                                    a_thread_vec.template AsType<mfma_input_type>(),
                                    b_thread_vec.template AsType<mfma_input_type>(),
                                    c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{}));
                            });
                            static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                                constexpr index_t c_offset =
                                    c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                                constexpr index_t cscale_offset =
                                    CScaleThreadDesc{}.CalculateOffset(
                                        make_tuple(kscale0, m0, n0 * num_scale_n_block / NRepeat));

                                c_thread_buf(Number<c_offset>{}) +=
                                    c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                        .template AsType<AccDataType>()[Number<t>{}] *
                                    type_convert<AccDataType>(
                                        c_scale_thread_buf[Number<cscale_offset>{}]);
                            });
                        });
                    });
                });

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, num_scale_n_block, 1>{}([&](auto n0) {
                        static_for<0, num_scale_k_block, 1>{}([&](auto k0) {
                            constexpr index_t c_offset =
                                CScaleThreadDesc{}.CalculateOffset(make_tuple(k0, m0, n0));
                            constexpr index_t a_offset =
                                AScaleThreadDesc{}.CalculateOffset(make_tuple(m0, k0));
                            constexpr index_t b_offset =
                                BScaleThreadDesc{}.CalculateOffset(make_tuple(n0, k0));

                            c_scale_thread_buf(Number<c_offset>{}) =
                                a_scale_thread_buf[Number<a_offset>{}] *
                                b_scale_thread_buf[Number<b_offset>{}];
                        });
                    });
                });

                block_sync_lds();
                static_for<0, KRepeat, 1>{}([&](auto k) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                           make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                           a_block_buf,
                                           a_thread_desc_,
                                           make_tuple(m0, I0, k, I0),
                                           a_thread_buf);
                    });
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                           b_block_buf,
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k, I0),
                                           b_thread_buf);
                    });
                });

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_scale_thread_copy.Run(a_scale_grid_desc,
                                            a_scale_grid_buf,
                                            a_scale_thread_desc,
                                            make_tuple(m0, I0),
                                            a_scale_thread_buf);
                    a_scale_thread_copy.MoveSrcSliceWindow(
                        a_scale_grid_desc, a_scale_thread_copy_step.At(Number<0>{}));
                });

                if constexpr(NumKBlockPerScale == 1)
                {
                    a_scale_thread_copy.MoveSrcSliceWindow(
                        a_scale_grid_desc, a_scale_thread_copy_step.At(Number<2>{}));
                }
                else
                {
                    a_scale_thread_copy.MoveSrcSliceWindow(
                        a_scale_grid_desc, a_scale_thread_copy_step.At(Number<1>{}));
                }

                b_scale_thread_copy.Run(b_scale_grid_desc,
                                        b_scale_grid_buf,
                                        b_scale_thread_desc,
                                        make_tuple(I0, I0),
                                        b_scale_thread_buf);

                b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc, b_scale_thread_copy_step);
                HotLoopScheduler();
                __builtin_amdgcn_sched_barrier(0);
                i += 1;
            } while(i < (num_loop - 2));
        }

        // tail
        if constexpr(TailNum == TailNumber::Full)
        {
            block_sync_lds();
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, num_scale_k_block, 1>{}([&](auto kscale0) {
                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                            c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                .template AsType<AccDataType>()(Number<t>{}) = 0;
                        });
                        static_for<0, KRepeat / num_scale_k_block, 1>{}([&](auto k0) {
                            vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                            vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   ik))>{}];
                                b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   ik))>{}];
                            });

                            using mfma_input_type =
                                typename vector_type<ComputeDataTypeBuf,
                                                     xdlops_gemm.K1PerXdlops>::type;

                            xdlops_gemm.template Run<>(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{}));
                        });
                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                            constexpr index_t cscale_offset = CScaleThreadDesc{}.CalculateOffset(
                                make_tuple(kscale0, m0, n0 * num_scale_n_block / NRepeat));

                            c_thread_buf(Number<c_offset>{}) +=
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                    .template AsType<AccDataType>()[Number<t>{}] *
                                type_convert<AccDataType>(
                                    c_scale_thread_buf[Number<cscale_offset>{}]);
                        });
                    });
                });
            });

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, num_scale_n_block, 1>{}([&](auto n0) {
                    static_for<0, num_scale_k_block, 1>{}([&](auto k0) {
                        constexpr index_t c_offset =
                            CScaleThreadDesc{}.CalculateOffset(make_tuple(k0, m0, n0));
                        constexpr index_t a_offset =
                            AScaleThreadDesc{}.CalculateOffset(make_tuple(m0, k0));
                        constexpr index_t b_offset =
                            BScaleThreadDesc{}.CalculateOffset(make_tuple(n0, k0));

                        c_scale_thread_buf(Number<c_offset>{}) =
                            a_scale_thread_buf[Number<a_offset>{}] *
                            b_scale_thread_buf[Number<b_offset>{}];
                    });
                });
            });

            block_sync_lds();
            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf,
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_buf);
                });
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                       make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                       b_block_buf,
                                       b_thread_desc_,
                                       make_tuple(n0, I0, k, I0),
                                       b_thread_buf);
                });
            });

            HotLoopScheduler();
            __builtin_amdgcn_sched_barrier(0);

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, num_scale_k_block, 1>{}([&](auto kscale0) {
                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                            c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                .template AsType<AccDataType>()(Number<t>{}) = 0;
                        });
                        static_for<0, KRepeat / num_scale_k_block, 1>{}([&](auto k0) {
                            vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                            vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   ik))>{}];
                                b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   ik))>{}];
                            });

                            using mfma_input_type =
                                typename vector_type<ComputeDataTypeBuf,
                                                     xdlops_gemm.K1PerXdlops>::type;

                            xdlops_gemm.template Run<>(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{}));
                        });
                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                            constexpr index_t cscale_offset = CScaleThreadDesc{}.CalculateOffset(
                                make_tuple(kscale0, m0, n0 * num_scale_n_block / NRepeat));

                            c_thread_buf(Number<c_offset>{}) +=
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                    .template AsType<AccDataType>()[Number<t>{}] *
                                type_convert<AccDataType>(
                                    c_scale_thread_buf[Number<cscale_offset>{}]);
                        });
                    });
                });
            });
            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, num_scale_k_block, 1>{}([&](auto kscale0) {
                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                            c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                .template AsType<AccDataType>()(Number<t>{}) = 0;
                        });
                        static_for<0, KRepeat / num_scale_k_block, 1>{}([&](auto k0) {
                            vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                            vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   ik))>{}];
                                b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   ik))>{}];
                            });

                            using mfma_input_type =
                                typename vector_type<ComputeDataTypeBuf,
                                                     xdlops_gemm.K1PerXdlops>::type;

                            xdlops_gemm.template Run<>(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{}));
                        });
                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                            constexpr index_t cscale_offset = CScaleThreadDesc{}.CalculateOffset(
                                make_tuple(kscale0, m0, n0 * num_scale_n_block / NRepeat));

                            c_thread_buf(Number<c_offset>{}) +=
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                    .template AsType<AccDataType>()[Number<t>{}] *
                                type_convert<AccDataType>(
                                    c_scale_thread_buf[Number<cscale_offset>{}]);
                        });
                    });
                });
            });
            __builtin_amdgcn_sched_barrier(0);
        }
    }

    protected:
    using Base::a_thread_desc_;
    using Base::b_thread_desc_;
    using Base::c_thread_desc_;
    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<ADataType,
                                                         ComputeDataTypeBuf,
                                                         decltype(a_block_desc_m0_m1_m2_k),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, KPack>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         A_K1,
                                                         A_K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<BDataType,
                                                         ComputeDataTypeBuf,
                                                         decltype(b_block_desc_n0_n1_n2_k),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, KPack>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         B_K1,
                                                         B_K1>;

    AThreadCopy a_thread_copy_{CalculateAThreadOriginDataIndex()};
    BThreadCopy b_thread_copy_{CalculateBThreadOriginDataIndex()};
};

} // namespace ck
