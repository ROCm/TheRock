// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp"

namespace ck {

// Maximum Global Memory throughput pipeline with >=32KB data in fly
// GlobalPrefetchStages: >=2
// LocalPreFillStages: 1
// LocalPreFetchStages: 0
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
struct BlockwiseGemmXdlops_pipeline_v2_b_scale
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
struct BlockwiseGemmXdlops_pipeline_v2_b_scale<BlockGemmPipelineScheduler::Intrawave,
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
    using Base::KRepeat;
    using Base::xdlops_gemm;

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
    using Base::WaveSize;

    using ComputeDataTypeBuf = typename Base::ComputeDataTypeBuf;

    static constexpr index_t WgpPerCU =
        (4 * WaveSize / BlockSize) >= 1 ? 4 * WaveSize / BlockSize : 1;
    static constexpr index_t FullMemBandPrefetchStages = math::integer_divide_ceil(
        32768 / WgpPerCU,
        (MPerBlock * sizeof(ADataType) + NPerBlock * sizeof(BDataType)) * KPerBlock);
    static constexpr index_t PrefetchStages =
        FullMemBandPrefetchStages >= 2
            ? FullMemBandPrefetchStages <= 8 ? FullMemBandPrefetchStages : 8
            : 2;

    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = PrefetchStages;

    __host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
    {
        if(num_loop % PrefetchStages == 1)
        {
            return TailNumber::One;
        }
        else if(num_loop % PrefetchStages == 2)
        {
            return TailNumber::Two;
        }
        else if(num_loop % PrefetchStages == 3)
        {
            return TailNumber::Three;
        }
        else if(num_loop % PrefetchStages == 4)
        {
            return TailNumber::Four;
        }
        else if(num_loop % PrefetchStages == 5)
        {
            return TailNumber::Five;
        }
        else if(num_loop % PrefetchStages == 6)
        {
            return TailNumber::Six;
        }
        else if(num_loop % PrefetchStages == 7)
        {
            return TailNumber::Seven;
        }
        else
        {
            return TailNumber::Full;
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

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

        // Global prefetch [2, PrefetchStages]
        static_for<1, PrefetchStages, 1>{}([&](auto iprefetch) {
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, iprefetch);
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, iprefetch);

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        });

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                static_for<0, PrefetchStages, 1>{}([&](auto iprefetch) {
                    // -------------------------------------------------------------------------------------------
                    block_sync_lds();
                    static_for<0, KRepeat, 1>{}([&](auto k) {
                        static_for<0, MRepeat, 1>{}([&](auto m0) {
                            a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                               make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                               a_block_buf,
                                               a_thread_desc_,
                                               make_tuple(m0, I0, k, I0),
                                               a_thread_buf);
                            static_for<0, NRepeat, 1>{}([&](auto n0) {
                                b_thread_copy_.Run(
                                    b_block_desc_n0_n1_n2_k,
                                    make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                    b_block_buf,
                                    b_thread_desc_,
                                    make_tuple(n0, I0, k, I0),
                                    b_thread_buf);
                            });
                        });
                    });

                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        static_for<0, MRepeat, 1>{}([&](auto m0) {
                            static_for<0, NRepeat, 1>{}([&](auto n0) {
                                vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                                vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                                static_for<0, KPack, 1>{}([&](auto ik) {
                                    a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                            make_tuple(m0, I0, k0, ik))>{}];
                                    b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                        b_thread_buf[Number<b_thread_desc_.CalculateOffset(
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

                    block_sync_lds();
                    a_blockwise_copy.RunWrite(
                        a_block_desc, a_block_buf, Number<(iprefetch + 1) % PrefetchStages>{});
                    b_blockwise_copy.RunWrite(
                        b_block_desc, b_block_buf, Number<(iprefetch + 1) % PrefetchStages>{});

                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, iprefetch);
                    b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, iprefetch);

                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                });

                i += PrefetchStages;
            } while(i < (num_loop - PrefetchStages));
        }

        // tail

        auto LoopTailFunc = [&](auto tail_num) {
            static_for<1, tail_num, 1>{}([&](auto iprefetch) {
                block_sync_lds();
                static_for<0, KRepeat, 1>{}([&](auto k) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                           make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                           a_block_buf,
                                           a_thread_desc_,
                                           make_tuple(m0, I0, k, I0),
                                           a_thread_buf);
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                               make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                               b_block_buf,
                                               b_thread_desc_,
                                               make_tuple(n0, I0, k, I0),
                                               b_thread_buf);
                        });
                    });
                });

                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                            vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0, I0, k0, ik))>{}];
                                b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
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

                block_sync_lds();
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, iprefetch);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, iprefetch);
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
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                           b_block_buf,
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k, I0),
                                           b_thread_buf);
                    });
                });
            });

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                        vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
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

        if constexpr(TailNum == TailNumber::One)
        {
            block_sync_lds();
            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf,
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_buf);
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                           b_block_buf,
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k, I0),
                                           b_thread_buf);
                    });
                });
            });

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                        vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                b_thread_buf[Number<b_thread_desc_.CalculateOffset(
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
        }
        else if constexpr(TailNum == TailNumber::Two)
        {
            LoopTailFunc(Number<2>{});
        }
        else if constexpr(TailNum == TailNumber::Three)
        {
            LoopTailFunc(Number<3>{});
        }
        else if constexpr(TailNum == TailNumber::Four)
        {
            LoopTailFunc(Number<4>{});
        }
        else if constexpr(TailNum == TailNumber::Five)
        {
            LoopTailFunc(Number<5>{});
        }
        else if constexpr(TailNum == TailNumber::Six)
        {
            LoopTailFunc(Number<6>{});
        }
        else if constexpr(TailNum == TailNumber::Seven)
        {
            LoopTailFunc(Number<7>{});
        }
        else if constexpr(TailNum == TailNumber::Full)
        {
            LoopTailFunc(Number<PrefetchStages>{});
        }
    }

    protected:
    using Base::a_thread_copy_;
    using Base::a_thread_desc_;
    using Base::b_thread_copy_;
    using Base::b_thread_desc_;
    using Base::c_thread_desc_;
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
struct BlockwiseGemmXdlops_pipeline_v2_b_scale<BlockGemmPipelineScheduler::Interwave,
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
    using Base::KPerThread;
    using Base::xdlops_gemm;

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
    using Base::WaveSize;

    using ComputeDataTypeBuf = typename Base::ComputeDataTypeBuf;

    static constexpr index_t NumMacClusters = CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING_MAC_CLUSTERS;
    static constexpr index_t KPerInnerLoop  = math::max(KPerThread / NumMacClusters, KPack);
    static constexpr index_t KRepeat        = KPerThread / KPerInnerLoop;

    static constexpr index_t WgpPerCU =
        (4 * WaveSize / BlockSize) >= 1 ? 4 * WaveSize / BlockSize : 1;
    static constexpr index_t FullMemBandPrefetchStages = math::integer_divide_ceil(
        32768 / WgpPerCU,
        (MPerBlock * sizeof(ADataType) + NPerBlock * sizeof(BDataType)) * KPerBlock);
    static constexpr index_t PrefetchStages =
        FullMemBandPrefetchStages >= 2
            ? FullMemBandPrefetchStages <= 8 ? FullMemBandPrefetchStages : 8
            : 2;

    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = PrefetchStages;

    __host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
    {
        if(num_loop % PrefetchStages == 1)
        {
            return TailNumber::One;
        }
        else if(num_loop % PrefetchStages == 2)
        {
            return TailNumber::Two;
        }
        else if(num_loop % PrefetchStages == 3)
        {
            return TailNumber::Three;
        }
        else if(num_loop % PrefetchStages == 4)
        {
            return TailNumber::Four;
        }
        else if(num_loop % PrefetchStages == 5)
        {
            return TailNumber::Five;
        }
        else if(num_loop % PrefetchStages == 6)
        {
            return TailNumber::Six;
        }
        else if(num_loop % PrefetchStages == 7)
        {
            return TailNumber::Seven;
        }
        else
        {
            return TailNumber::Full;
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
              typename BScaleGridBuffer,
              typename BScaleGridDesc,
              typename BScaleThreadDesc,
              typename BScaleThreadTransfer,
              typename BScaleThreadTransferStep>
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
                        const BScaleGridDesc& b_scale_grid_desc,
                        // BScaleThreadCopy
                        const BScaleThreadDesc& b_scale_thread_desc,
                        BScaleThreadTransfer& b_scale_thread_copy,
                        const BScaleGridBuffer& b_scale_grid_buf,
                        const BScaleThreadTransferStep& b_scale_thread_copy_step,
                        // num loop
                        index_t num_loop,
                        index_t num_loop_per_scale) const
    {
        ignore = num_loop_per_scale;

        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
            b_thread_desc_.GetElementSpaceSize());

        auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
            b_scale_thread_desc.GetElementSpaceSize());

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        static_for<0, NRepeat, 1>{}([&](auto n0) {
            b_scale_thread_copy.Run(b_scale_grid_desc,
                                    b_scale_grid_buf,
                                    b_scale_thread_desc,
                                    make_tuple(n0, I0),
                                    b_scale_thread_buf);

            b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                   b_scale_thread_copy_step.At(Number<0>{}));
        });
        b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                               b_scale_thread_copy_step.At(Number<1>{}));

        // Initialize C
        c_thread_buf.Clear();

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

        // Global prefetch [2, PrefetchStages]
        static_for<1, PrefetchStages, 1>{}([&](auto iprefetch) {
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, iprefetch);
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, iprefetch);

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        });

        auto c_thread_buf_per_scale = remove_cvref_t<decltype(c_thread_buf)>(); // need?

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                static_for<0, PrefetchStages, 1>{}([&](auto iprefetch) {
                    // -------------------------------------------------------------------------------------------
                    block_sync_lds();
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        static_for<0, MRepeat, 1>{}([&](auto m0) {
                            a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                               make_tuple(m0, I0, I0, Number<k0 * KPerInnerLoop>{}),
                                               a_block_buf,
                                               a_thread_desc_,
                                               make_tuple(m0, I0, k0, I0),
                                               a_thread_buf);
                            static_for<0, NRepeat, 1>{}([&](auto n0) {
                                b_thread_copy_.Run(
                                    b_block_desc_n0_n1_n2_k,
                                    make_tuple(n0, I0, I0, Number<k0 * KPerInnerLoop>{}),
                                    b_block_buf,
                                    b_thread_desc_,
                                    make_tuple(n0, I0, k0, I0),
                                    b_thread_buf);
                            });
                        });
                        __builtin_amdgcn_sched_barrier(0);
                        // NOTE: Synchronize threads in a workgroup at the start of each MAC
                        // cluster, but except the first, as we can shorten non-MAC cluster a bit
                        // and there's no observable negative impact. The desired effect is waves in
                        // a workgroup executing MAC in sync. This avoids some out-of-sync waves
                        // hijacking MAC resource from other workgroups and reducing the chance of
                        // latency hiding by waiting for the rest of the workgroup at the eventual
                        // sync point.
                        if constexpr(k0.value != 0 || KRepeat == 1)
                        {
                            __builtin_amdgcn_s_barrier();
                            __builtin_amdgcn_sched_barrier(0);
                        }
                        static_for<0, KPerInnerLoop, KPack>{}([&](auto k_) {
                            static_for<0, MRepeat, 1>{}([&](auto m0) {
                                static_for<0, NRepeat, 1>{}([&](auto n0) {
                                    vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                                    vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                                    static_for<0, KPack, 1>{}([&](auto ik) {
                                        a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                            a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                                make_tuple(m0, I0, k0, k_ + ik))>{}];
                                        b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                            b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                                make_tuple(n0, I0, k0, k_ + ik))>{}];
                                    });

                                    using mfma_input_type =
                                        typename vector_type<ComputeDataTypeBuf,
                                                             xdlops_gemm.K1PerXdlops>::type;

                                    constexpr index_t c_offset =
                                        c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                                    // The block_sync_lds() here performs double duty:
                                    // A) safeguard against data hazard because barrier from
                                    // blockwise_gemm is moved here B) reduce VMEM FIFO congestion
                                    // by applying small delays to different wavefronts It is
                                    // performed near the end of MAC cluster to minimize lgkmcnt
                                    // penalty
                                    if constexpr(k0.value == KRepeat - 1 &&
                                                 k_.value == KPerInnerLoop - KPack &&
                                                 m0.value == MRepeat - 1 && n0.value == NRepeat - 1)
                                    {
                                        __builtin_amdgcn_sched_barrier(0);
                                        block_sync_lds();
                                        __builtin_amdgcn_sched_barrier(0);
                                    }
                                    xdlops_gemm.Run(
                                        a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                                    if constexpr(k_.value == 0 && m0.value == 0 && n0.value == 0)
                                    {
                                        __builtin_amdgcn_sched_barrier(0);
                                        __builtin_amdgcn_s_setprio(1);
                                        __builtin_amdgcn_sched_barrier(0);
                                    }
                                });

                                // static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t)
                                // {
                                //     constexpr index_t c_offset =
                                //         c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                                //     c_thread_buf(Number<c_offset>{}) +=
                                //         c_thread_buf_per_scale[Number<t>{}] *
                                //         type_convert<AccDataType>(b_scale_thread_buf[n0]);
                                // });
                            });
                        });
                        __builtin_amdgcn_sched_barrier(0);
                        __builtin_amdgcn_s_setprio(0);
                        __builtin_amdgcn_sched_barrier(0);
                    });

                    // static_for<0, NRepeat, 1>{}([&](auto n0) {
                    //         b_scale_thread_copy.Run(b_scale_grid_desc,
                    //                                 b_scale_grid_buf,
                    //                                 b_scale_thread_desc,
                    //                                 make_tuple(n0, I0),
                    //                                 b_scale_thread_buf);

                    //         b_scale_thread_copy.MoveSrcSliceWindow(
                    //         b_scale_grid_desc, b_scale_thread_copy_step.At(Number<0>{}));
                    //     });
                    // b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                    //                                    b_scale_thread_copy_step.At(Number<1>{}));

                    // block_sync_lds();
                    a_blockwise_copy.RunWrite(
                        a_block_desc, a_block_buf, Number<(iprefetch + 1) % PrefetchStages>{});
                    b_blockwise_copy.RunWrite(
                        b_block_desc, b_block_buf, Number<(iprefetch + 1) % PrefetchStages>{});

                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, iprefetch);
                    b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, iprefetch);

                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                });
                i += PrefetchStages;
            } while(i < (num_loop - PrefetchStages));
        }

        // tail

        auto LoopTailFunc = [&](auto tail_num) {
            static_for<1, tail_num, 1>{}([&](auto iprefetch) {
                block_sync_lds();
                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                           make_tuple(m0, I0, I0, Number<k0 * KPerInnerLoop>{}),
                                           a_block_buf,
                                           a_thread_desc_,
                                           make_tuple(m0, I0, k0, I0),
                                           a_thread_buf);
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                               make_tuple(n0, I0, I0, Number<k0 * KPerInnerLoop>{}),
                                               b_block_buf,
                                               b_thread_desc_,
                                               make_tuple(n0, I0, k0, I0),
                                               b_thread_buf);
                        });
                    });

                    __builtin_amdgcn_sched_barrier(0);
                    if constexpr(k0.value != 0 || KRepeat == 1)
                    {
                        __builtin_amdgcn_s_barrier();
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    static_for<0, KPerInnerLoop, KPack>{}([&](auto k_) {
                        static_for<0, MRepeat, 1>{}([&](auto m0) {
                            static_for<0, NRepeat, 1>{}([&](auto n0) {
                                vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                                vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                                static_for<0, KPack, 1>{}([&](auto ik) {
                                    a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                            make_tuple(m0, I0, k0, k_ + ik))>{}];
                                    b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                        b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                            make_tuple(n0, I0, k0, k_ + ik))>{}];
                                });

                                using mfma_input_type =
                                    typename vector_type<ComputeDataTypeBuf,
                                                         xdlops_gemm.K1PerXdlops>::type;

                                constexpr index_t c_offset =
                                    c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                                if constexpr(k0.value == KRepeat - 1 &&
                                             k_.value == KPerInnerLoop - KPack &&
                                             m0.value == MRepeat - 1 && n0.value == NRepeat - 1)
                                {
                                    __builtin_amdgcn_sched_barrier(0);
                                    block_sync_lds();
                                    __builtin_amdgcn_sched_barrier(0);
                                }
                                xdlops_gemm.Run(
                                    a_thread_vec.template AsType<mfma_input_type>(),
                                    b_thread_vec.template AsType<mfma_input_type>(),
                                    c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                                if constexpr(k_.value == 0 && m0.value == 0 && n0.value == 0)
                                {
                                    __builtin_amdgcn_sched_barrier(0);
                                    __builtin_amdgcn_s_setprio(1);
                                    __builtin_amdgcn_sched_barrier(0);
                                }
                            });

                            // static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                            //     constexpr index_t c_offset =
                            //         c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                            //     c_thread_buf(Number<c_offset>{}) +=
                            //         c_thread_buf_per_scale[Number<t>{}] *
                            //         type_convert<AccDataType>(b_scale_thread_buf[n0]);
                            // });
                        });
                    });
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_setprio(0);
                    __builtin_amdgcn_sched_barrier(0);
                });

                // static_for<0, NRepeat, 1>{}([&](auto n0) {
                //     b_scale_thread_copy.Run(b_scale_grid_desc,
                //                             b_scale_grid_buf,
                //                             b_scale_thread_desc,
                //                             make_tuple(n0, I0),
                //                             b_scale_thread_buf);

                //     b_scale_thread_copy.MoveSrcSliceWindow(
                //         b_scale_grid_desc, b_scale_thread_copy_step.At(Number<0>{}));
                // });
                // b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                //                                        b_scale_thread_copy_step.At(Number<1>{}));

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, iprefetch);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, iprefetch);
            });
            block_sync_lds();
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k0 * KPerInnerLoop>{}),
                                       a_block_buf,
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k0, I0),
                                       a_thread_buf);
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<k0 * KPerInnerLoop>{}),
                                           b_block_buf,
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k0, I0),
                                           b_thread_buf);
                    });
                });

                __builtin_amdgcn_sched_barrier(0);
                if constexpr(k0.value != 0 || KRepeat == 1)
                {
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                }
                static_for<0, KPerInnerLoop, KPack>{}([&](auto k_) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                            vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0, I0, k0, k_ + ik))>{}];
                                b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0, I0, k0, k_ + ik))>{}];
                            });

                            using mfma_input_type =
                                typename vector_type<ComputeDataTypeBuf,
                                                     xdlops_gemm.K1PerXdlops>::type;

                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                            if constexpr(k0.value == KRepeat - 1 &&
                                         k_.value == KPerInnerLoop - KPack &&
                                         m0.value == MRepeat - 1 && n0.value == NRepeat - 1)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                block_sync_lds();
                                __builtin_amdgcn_sched_barrier(0);
                            }
                            xdlops_gemm.Run(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                            if constexpr(k_.value == 0 && m0.value == 0 && n0.value == 0)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                __builtin_amdgcn_s_setprio(1);
                                __builtin_amdgcn_sched_barrier(0);
                            }
                        });

                        // static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                        //     constexpr index_t c_offset =
                        //         c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                        //     c_thread_buf(Number<c_offset>{}) +=
                        //         c_thread_buf_per_scale[Number<t>{}] *
                        //         type_convert<AccDataType>(b_scale_thread_buf[n0]);
                        // });
                    });
                });
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
            });
        };

        if constexpr(TailNum == TailNumber::One)
        {
            block_sync_lds();
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k0 * KPerInnerLoop>{}),
                                       a_block_buf,
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k0, I0),
                                       a_thread_buf);
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<k0 * KPerInnerLoop>{}),
                                           b_block_buf,
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k0, I0),
                                           b_thread_buf);
                    });
                });

                __builtin_amdgcn_sched_barrier(0);
                if constexpr(k0.value != 0 || KRepeat == 1)
                {
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                }
                static_for<0, KPerInnerLoop, KPack>{}([&](auto k_) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                            vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0, I0, k0, k_ + ik))>{}];
                                b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                                    b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0, I0, k0, k_ + ik))>{}];
                            });

                            using mfma_input_type =
                                typename vector_type<ComputeDataTypeBuf,
                                                     xdlops_gemm.K1PerXdlops>::type;

                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                            if constexpr(k0.value == KRepeat - 1 &&
                                         k_.value == KPerInnerLoop - KPack &&
                                         m0.value == MRepeat - 1 && n0.value == NRepeat - 1)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                block_sync_lds();
                                __builtin_amdgcn_sched_barrier(0);
                            }
                            xdlops_gemm.Run(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                            if constexpr(k_.value == 0 && m0.value == 0 && n0.value == 0)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                __builtin_amdgcn_s_setprio(1);
                                __builtin_amdgcn_sched_barrier(0);
                            }
                        });

                        // static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                        //     constexpr index_t c_offset =
                        //         c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                        //     c_thread_buf(Number<c_offset>{}) +=
                        //         c_thread_buf_per_scale[Number<t>{}] *
                        //         type_convert<AccDataType>(b_scale_thread_buf[n0]);
                        // });
                    });
                });
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
            });
        }
        else if constexpr(TailNum == TailNumber::Two)
        {
            LoopTailFunc(Number<2>{});
        }
        else if constexpr(TailNum == TailNumber::Three)
        {
            LoopTailFunc(Number<3>{});
        }
        else if constexpr(TailNum == TailNumber::Four)
        {
            LoopTailFunc(Number<4>{});
        }
        else if constexpr(TailNum == TailNumber::Five)
        {
            LoopTailFunc(Number<5>{});
        }
        else if constexpr(TailNum == TailNumber::Six)
        {
            LoopTailFunc(Number<6>{});
        }
        else if constexpr(TailNum == TailNumber::Seven)
        {
            LoopTailFunc(Number<7>{});
        }
        else if constexpr(TailNum == TailNumber::Full)
        {
            LoopTailFunc(Number<PrefetchStages>{});
        }
    }

    protected:
    // K->M loopover
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor(
        make_tuple(Number<MRepeat>{}, I1, Number<KRepeat>{}, Number<KPerInnerLoop>{}),
        make_tuple(Number<KPerInnerLoop>{},
                   Number<KRepeat * MRepeat * KPerInnerLoop>{},
                   Number<MRepeat * KPerInnerLoop>{},
                   I1));

    static constexpr auto b_thread_desc_ = make_naive_tensor_descriptor(
        make_tuple(Number<NRepeat>{}, I1, Number<KRepeat>{}, Number<KPerInnerLoop>{}),
        make_tuple(Number<KPerInnerLoop>{},
                   Number<KRepeat * NRepeat * KPerInnerLoop>{},
                   Number<NRepeat * KPerInnerLoop>{},
                   I1));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<ADataType,
                                                         ComputeDataTypeBuf,
                                                         decltype(a_block_desc_m0_m1_m2_k),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, KPerInnerLoop>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         A_K1,
                                                         A_K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<BDataType,
                                                         ComputeDataTypeBuf,
                                                         decltype(b_block_desc_n0_n1_n2_k),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, KPerInnerLoop>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         B_K1,
                                                         B_K1>;

    AThreadCopy a_thread_copy_{Base::CalculateAThreadOriginDataIndex()};
    BThreadCopy b_thread_copy_{Base::CalculateBThreadOriginDataIndex()};
    using Base::c_thread_desc_;
};

} // namespace ck
