// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp"

namespace ck {

// Compute optimimal pipeline with highest resource request
// GlobalPrefetchStages: 3
// LocalPreFillStages: 2
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 2

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
struct BlockwiseGemmXdlops_pipeline_v4
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
struct BlockwiseGemmXdlops_pipeline_v4<BlockGemmPipelineScheduler::Intrawave,
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
    using Base::MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;

    using Base::a_block_desc_m0_m1_m2_k;
    using Base::b_block_desc_n0_n1_n2_k;

    using Base::AMmaKStride;
    using Base::BMmaKStride;

    using ComputeDataTypeBuf = typename Base::ComputeDataTypeBuf;

    static constexpr index_t PrefetchStages  = 3;
    static constexpr index_t PrefillStages   = 2;
    static constexpr index_t GlobalBufferNum = 1;
    static constexpr index_t HotloopUnroll   = 2;

    __host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
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

    __device__ static constexpr void HotLoopScheduler()
    {
        // TODO: Take data type into consideration as pipe ver 3
        // A-B splited schedule
        constexpr auto num_ds_read_inst_a =
            HotLoopInstList::A_LDS_Read_Width * sizeof(ADataType) == 16
                ? HotLoopInstList::A_LDS_Read_Inst_Num
                : HotLoopInstList::A_LDS_Read_Inst_Num / 2;
        constexpr auto num_ds_read_inst_b =
            HotLoopInstList::B_LDS_Read_Width * sizeof(BDataType) == 16
                ? HotLoopInstList::B_LDS_Read_Inst_Num
                : HotLoopInstList::B_LDS_Read_Inst_Num / 2;

        constexpr auto num_issue_a = HotLoopInstList::A_Buffer_Load_Inst_Num;
        constexpr auto num_dswrite_per_issue_a =
            (HotLoopInstList::A_LDS_Write_Inst_Num + num_issue_a - 1) / num_issue_a;
        constexpr auto num_dsread_per_issue_a = num_ds_read_inst_a / num_issue_a;

        constexpr auto num_issue_b = HotLoopInstList::B_Buffer_Load_Inst_Num;
        constexpr auto num_dswrite_per_issue_b =
            (HotLoopInstList::B_LDS_Write_Inst_Num + num_issue_b - 1) / num_issue_b;
        constexpr auto num_dsread_per_issue_b = num_ds_read_inst_b / num_issue_b;

        constexpr auto num_mfma_per_issue =
            HotLoopInstList::C_MFMA_Inst_Num / (num_issue_a + num_issue_b);

        static_for<0, num_issue_a, 1>{}([&](auto i) {
            ignore = i;
            static_for<0, num_dsread_per_issue_a, 1>{}([&](auto idsread) {
                ignore = idsread;
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            static_for<0, num_dswrite_per_issue_a, 1>{}([&](auto idswrite) {
                ignore = idswrite;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008,
                                                 num_mfma_per_issue - num_dsread_per_issue_a -
                                                     num_dswrite_per_issue_a,
                                                 0); // MFMA
        });

        static_for<0, num_issue_b, 1>{}([&](auto i) {
            ignore = i;
            static_for<0, num_dsread_per_issue_b, 1>{}([&](auto idsread) {
                ignore = idsread;
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
                ignore = idswrite;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008,
                                                 num_mfma_per_issue - num_dsread_per_issue_a -
                                                     num_dswrite_per_issue_b,
                                                 0); // MFMA
        });
        __builtin_amdgcn_sched_barrier(0);
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
              typename CThreadBuffer>
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
                        index_t num_loop) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
            b_thread_desc_.GetElementSpaceSize());

        StaticallyIndexedArray<decltype(a_thread_buf), Number<2>{}> a_thread_bufs;
        StaticallyIndexedArray<decltype(b_thread_buf), Number<2>{}> b_thread_bufs;

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(I0));
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(I0));

        // Local prefetch 1
        block_sync_lds();
        static_for<0, KRepeat, 1>{}([&](auto k) {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                   make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                   a_block_buf.At(I0),
                                   a_thread_desc_,
                                   make_tuple(m0, I0, k, I0),
                                   a_thread_bufs(I0));
            });
            static_for<0, NRepeat, 1>{}([&](auto n0) {
                b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                   make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                   b_block_buf.At(I0),
                                   b_thread_desc_,
                                   make_tuple(n0, I0, k, I0),
                                   b_thread_bufs(I0));
            });
        });

        // Global prefetch 2
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Local prefill 2
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(I1));
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(I1));

        // Global prefetch 3
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            // This hot loop has two legacy loopover, to implement the double local buffer strategy
            do
            {
                auto LoopFunc = [&](auto lds_read_buf,
                                    auto lds_read_reg_buf,
                                    auto lds_write_buf,
                                    auto mfma_reg_buf) {
                    block_sync_lds();

                    static_for<0, KRepeat, 1>{}([&](auto k) {
                        static_for<0, MRepeat, 1>{}([&](auto m0) {
                            a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                               make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                               a_block_buf.At(lds_read_buf),
                                               a_thread_desc_,
                                               make_tuple(m0, I0, k, I0),
                                               a_thread_bufs(lds_read_reg_buf));
                        });
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                               make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                               b_block_buf.At(lds_read_buf),
                                               b_thread_desc_,
                                               make_tuple(n0, I0, k, I0),
                                               b_thread_bufs(lds_read_reg_buf));
                        });
                    });

                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(lds_write_buf));
                    b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(lds_write_buf));

                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                    b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        static_for<0, MRepeat, 1>{}([&](auto m0) {
                            static_for<0, NRepeat, 1>{}([&](auto n0) {
                                vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                                vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                                static_for<0, KPack, 1>{}([&](auto ik) {
                                    a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                        a_thread_bufs[mfma_reg_buf]
                                                     [Number<a_thread_desc_.CalculateOffset(
                                                         make_tuple(m0, I0, k0, ik))>{}];
                                    b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                        b_thread_bufs[mfma_reg_buf]
                                                     [Number<b_thread_desc_.CalculateOffset(
                                                         make_tuple(n0, I0, k0, ik))>{}];
                                });

                                using mfma_input_type =
                                    typename vector_type<ComputeDataTypeBuf,
                                                         xdlops_gemm.K1PerXdlops>::type;

                                constexpr index_t c_offset =
                                    c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                                xdlops_gemm.Run(
                                    a_thread_vec.template AsType<mfma_input_type>(),
                                    b_thread_vec.template AsType<mfma_input_type>(),
                                    c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                            });
                        });
                    });

                    HotLoopScheduler();
                };

                LoopFunc(I1, I1, I0, I0);
                LoopFunc(I0, I0, I1, I1);

                i += HotloopUnroll;
            } while(i < (num_loop - PrefetchStages));
        }

        auto ReadWriteCompFunc = [&](auto lds_read_buf,
                                     auto lds_read_reg_buf,
                                     auto lds_write_buf,
                                     auto mfma_reg_buf) {
            block_sync_lds();

            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf.At(lds_read_buf),
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_bufs(lds_read_reg_buf));
                });
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                       make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                       b_block_buf.At(lds_read_buf),
                                       b_thread_desc_,
                                       make_tuple(n0, I0, k, I0),
                                       b_thread_bufs(lds_read_reg_buf));
                });
            });

            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(lds_write_buf));
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(lds_write_buf));

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                        vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                a_thread_bufs[mfma_reg_buf][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                b_thread_bufs[mfma_reg_buf][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataTypeBuf, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            HotLoopScheduler();
        };

        auto ReadCompFunc = [&](auto lds_read_buf, auto lds_read_reg_buf, auto mfma_reg_buf) {
            block_sync_lds();

            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf.At(lds_read_buf),
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_bufs(lds_read_reg_buf));
                });
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                       make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                       b_block_buf.At(lds_read_buf),
                                       b_thread_desc_,
                                       make_tuple(n0, I0, k, I0),
                                       b_thread_bufs(lds_read_reg_buf));
                });
            });

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                        vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                a_thread_bufs[mfma_reg_buf][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                b_thread_bufs[mfma_reg_buf][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataTypeBuf, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            HotLoopScheduler();
        };

        auto CompFunc = [&](auto mfma_reg_buf) {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                        vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                a_thread_bufs[mfma_reg_buf][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                b_thread_bufs[mfma_reg_buf][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataTypeBuf, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });
        };
        // tail
        if constexpr(TailNum == TailNumber::Odd)
        {
            ReadWriteCompFunc(I1, I1, I0, I0);
            ReadCompFunc(I0, I0, I1);
            CompFunc(I0);
        }
        else if constexpr(TailNum == TailNumber::Even)
        {
            ReadCompFunc(I1, I1, I0);
            CompFunc(I1);
        }
    }

    protected:
    using Base::a_thread_copy_;
    using Base::a_thread_desc_;
    using Base::b_thread_copy_;
    using Base::b_thread_desc_;
    using Base::c_thread_desc_;
};

// Compute optimimal pipeline with highest resource request
// Implementation with direct load
// GlobalPrefetchStages: 3
// LocalPreFillStages: 2
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 2

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
struct BlockwiseGemmXdlopsDirectLoad_pipeline_v4
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
struct BlockwiseGemmXdlopsDirectLoad_pipeline_v4<BlockGemmPipelineScheduler::Intrawave,
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
    using Base::MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;

    using Base::a_block_desc_m0_m1_m2_k;
    using Base::b_block_desc_n0_n1_n2_k;

    using Base::AMmaKStride;
    using Base::BMmaKStride;

    using ComputeDataTypeBuf = typename Base::ComputeDataTypeBuf;

    static constexpr index_t PrefetchStages  = 2;
    static constexpr index_t PrefillStages   = 2;
    static constexpr index_t GlobalBufferNum = 1;
    static constexpr index_t HotloopUnroll   = 2;

    __host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
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

    __device__ static constexpr void HotLoopScheduler()
    {
        // TODO: Take data type into consideration as pipe ver 3
        // A-B splited schedule
        constexpr auto num_ds_read_inst_a =
            HotLoopInstList::A_LDS_Read_Width * sizeof(ADataType) == 16
                ? HotLoopInstList::A_LDS_Read_Inst_Num
                : HotLoopInstList::A_LDS_Read_Inst_Num / 2;
        constexpr auto num_ds_read_inst_b =
            HotLoopInstList::B_LDS_Read_Width * sizeof(BDataType) == 16
                ? HotLoopInstList::B_LDS_Read_Inst_Num
                : HotLoopInstList::B_LDS_Read_Inst_Num / 2;

        constexpr auto num_issue_a             = HotLoopInstList::A_Buffer_Load_Inst_Num;
        constexpr auto num_dswrite_per_issue_a = 0;
        constexpr auto num_dsread_per_issue_a  = num_ds_read_inst_a / num_issue_a;

        constexpr auto num_issue_b             = HotLoopInstList::B_Buffer_Load_Inst_Num;
        constexpr auto num_dswrite_per_issue_b = 0;
        constexpr auto num_dsread_per_issue_b  = num_ds_read_inst_b / num_issue_b;

        constexpr auto num_mfma_per_issue =
            HotLoopInstList::C_MFMA_Inst_Num / (num_issue_a + num_issue_b);

        static_for<0, num_issue_a, 1>{}([&](auto i) {
            ignore = i;
            static_for<0, num_dsread_per_issue_a, 1>{}([&](auto idsread) {
                ignore = idsread;
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            static_for<0, num_dswrite_per_issue_a, 1>{}([&](auto idswrite) {
                ignore = idswrite;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008,
                                                 num_mfma_per_issue - num_dsread_per_issue_a -
                                                     num_dswrite_per_issue_a,
                                                 0); // MFMA
        });

        static_for<0, num_issue_b, 1>{}([&](auto i) {
            ignore = i;
            static_for<0, num_dsread_per_issue_b, 1>{}([&](auto idsread) {
                ignore = idsread;
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
                ignore = idswrite;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008,
                                                 num_mfma_per_issue - num_dsread_per_issue_a -
                                                     num_dswrite_per_issue_b,
                                                 0); // MFMA
        });
        __builtin_amdgcn_sched_barrier(0);
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
              typename CThreadBuffer>
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
                        index_t num_loop) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
            b_thread_desc_.GetElementSpaceSize());

        StaticallyIndexedArray<decltype(a_thread_buf), Number<2>{}> a_thread_bufs;
        StaticallyIndexedArray<decltype(b_thread_buf), Number<2>{}> b_thread_bufs;

        // Global prefetch 1
        a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf.At(I0));
        b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf.At(I0));

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        block_sync_lds_direct_load();

        // Local prefetch 1
        static_for<0, KRepeat, 1>{}([&](auto k) {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                   make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                   a_block_buf.At(I0),
                                   a_thread_desc_,
                                   make_tuple(m0, I0, k, I0),
                                   a_thread_bufs(I0));
            });
            static_for<0, NRepeat, 1>{}([&](auto n0) {
                b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                   make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                   b_block_buf.At(I0),
                                   b_thread_desc_,
                                   make_tuple(n0, I0, k, I0),
                                   b_thread_bufs(I0));
            });
        });

        // Global prefetch 2
        a_blockwise_copy.Run(a_grid_desc, a_grid_buf, a_block_desc, a_block_buf.At(I1));
        b_blockwise_copy.Run(b_grid_desc, b_grid_buf, b_block_desc, b_block_buf.At(I1));

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            // This hot loop has two legacy loopover, to implement the double local buffer strategy
            do
            {
                auto LoopFunc = [&](auto lds_read_buf,
                                    auto lds_read_reg_buf,
                                    auto lds_write_buf,
                                    auto mfma_reg_buf) {
                    block_sync_lds_direct_load();

                    static_for<0, KRepeat, 1>{}([&](auto k) {
                        static_for<0, MRepeat, 1>{}([&](auto m0) {
                            a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                               make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                               a_block_buf.At(lds_read_buf),
                                               a_thread_desc_,
                                               make_tuple(m0, I0, k, I0),
                                               a_thread_bufs(lds_read_reg_buf));
                        });
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                               make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                               b_block_buf.At(lds_read_buf),
                                               b_thread_desc_,
                                               make_tuple(n0, I0, k, I0),
                                               b_thread_bufs(lds_read_reg_buf));
                        });
                    });

                    a_blockwise_copy.Run(
                        a_grid_desc, a_grid_buf, a_block_desc, a_block_buf.At(lds_write_buf));
                    b_blockwise_copy.Run(
                        b_grid_desc, b_grid_buf, b_block_desc, b_block_buf.At(lds_write_buf));

                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        static_for<0, MRepeat, 1>{}([&](auto m0) {
                            static_for<0, NRepeat, 1>{}([&](auto n0) {
                                vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                                vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                                static_for<0, KPack, 1>{}([&](auto ik) {
                                    a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                        a_thread_bufs[mfma_reg_buf]
                                                     [Number<a_thread_desc_.CalculateOffset(
                                                         make_tuple(m0, I0, k0, ik))>{}];
                                    b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                        b_thread_bufs[mfma_reg_buf]
                                                     [Number<b_thread_desc_.CalculateOffset(
                                                         make_tuple(n0, I0, k0, ik))>{}];
                                });

                                using mfma_input_type =
                                    typename vector_type<ComputeDataTypeBuf,
                                                         xdlops_gemm.K1PerXdlops>::type;

                                constexpr index_t c_offset =
                                    c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                                xdlops_gemm.Run(
                                    a_thread_vec.template AsType<mfma_input_type>(),
                                    b_thread_vec.template AsType<mfma_input_type>(),
                                    c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                            });
                        });
                    });

                    HotLoopScheduler();
                };

                LoopFunc(I1, I1, I0, I0);
                LoopFunc(I0, I0, I1, I1);

                i += HotloopUnroll;
            } while(i < (num_loop - PrefetchStages));
        }

        auto ReadWriteCompFunc = [&](auto lds_read_buf,
                                     auto lds_read_reg_buf,
                                     auto lds_write_buf,
                                     auto mfma_reg_buf) {
            block_sync_lds_direct_load();

            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf.At(lds_read_buf),
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_bufs(lds_read_reg_buf));
                });
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                       make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                       b_block_buf.At(lds_read_buf),
                                       b_thread_desc_,
                                       make_tuple(n0, I0, k, I0),
                                       b_thread_bufs(lds_read_reg_buf));
                });
            });

            a_blockwise_copy.Run(
                a_grid_desc, a_grid_buf, a_block_desc, a_block_buf.At(lds_write_buf));
            b_blockwise_copy.Run(
                b_grid_desc, b_grid_buf, b_block_desc, b_block_buf.At(lds_write_buf));

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                        vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                a_thread_bufs[mfma_reg_buf][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                b_thread_bufs[mfma_reg_buf][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataTypeBuf, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            HotLoopScheduler();
        };

        auto ReadCompFunc = [&](auto lds_read_buf, auto lds_read_reg_buf, auto mfma_reg_buf) {
            block_sync_lds_direct_load();

            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf.At(lds_read_buf),
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_bufs(lds_read_reg_buf));
                });
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                       make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                       b_block_buf.At(lds_read_buf),
                                       b_thread_desc_,
                                       make_tuple(n0, I0, k, I0),
                                       b_thread_bufs(lds_read_reg_buf));
                });
            });

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                        vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                a_thread_bufs[mfma_reg_buf][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                b_thread_bufs[mfma_reg_buf][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataTypeBuf, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            HotLoopScheduler();
        };

        auto CompFunc = [&](auto mfma_reg_buf) {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                        vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                a_thread_bufs[mfma_reg_buf][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                b_thread_bufs[mfma_reg_buf][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataTypeBuf, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });
        };
        // tail
        if constexpr(TailNum == TailNumber::Odd)
        {
            ReadWriteCompFunc(I1, I1, I0, I0);
            ReadCompFunc(I0, I0, I1);
            CompFunc(I0);
        }
        else if constexpr(TailNum == TailNumber::Even)
        {
            ReadCompFunc(I1, I1, I0);
            CompFunc(I1);
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
