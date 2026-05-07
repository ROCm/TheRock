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
struct BlockwiseGemmXdlops_pipeline_v2_ab_scale
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
struct BlockwiseGemmXdlops_pipeline_v2_ab_scale<BlockGemmPipelineScheduler::Intrawave,
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

    __host__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
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
        index_t num_loop,
        index_t num_loop_per_scale) const
    {
        // assume kperblock = scaleblockk
        ignore            = num_loop_per_scale;
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataTypeBuf>(
            b_thread_desc_.GetElementSpaceSize());
        auto a_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            a_scale_thread_desc.GetElementSpaceSize());
        auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            b_scale_thread_desc.GetElementSpaceSize());

        // Global prefetch 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

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

        if(num_loop_per_scale == 1)
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

        // Local prefill 1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

        // Initialize C
        c_thread_buf.Clear();

        // Global prefetch [2, PrefetchStages]
        static_for<1, PrefetchStages, 1>{}([&](auto iprefetch) {
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, iprefetch);
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, iprefetch);

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        });

        auto c_thread_buf_per_scale = remove_cvref_t<decltype(c_thread_buf)>();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                static_for<0, PrefetchStages, 1>{}([&](auto iprefetch) {
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
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            c_thread_buf_per_scale.Clear();
                            static_for<0, KRepeat, 1>{}([&](auto k0) {
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

                                xdlops_gemm.template Run<>(
                                    a_thread_vec.template AsType<mfma_input_type>(),
                                    b_thread_vec.template AsType<mfma_input_type>(),
                                    c_thread_buf_per_scale.GetVectorTypeReference(I0));
                            });
                            static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                                constexpr index_t c_offset =
                                    c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                                c_thread_buf(Number<c_offset>{}) +=
                                    c_thread_buf_per_scale[Number<t>{}] *
                                    type_convert<AccDataType>(a_scale_thread_buf[m0]) *
                                    type_convert<AccDataType>(b_scale_thread_buf[I0]);
                            });
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

                    if(num_loop_per_scale == 1)
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

                    b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                           b_scale_thread_copy_step);

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

                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        c_thread_buf_per_scale.Clear();
                        static_for<0, KRepeat, 1>{}([&](auto k0) {
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

                            xdlops_gemm.template Run<>(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf_per_scale.GetVectorTypeReference(I0));
                        });
                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                            c_thread_buf(Number<c_offset>{}) +=
                                c_thread_buf_per_scale[Number<t>{}] *
                                type_convert<AccDataType>(a_scale_thread_buf[m0]) *
                                type_convert<AccDataType>(b_scale_thread_buf[I0]);
                        });
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

                if(num_loop_per_scale == 1)
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

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    c_thread_buf_per_scale.Clear();
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
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

                        xdlops_gemm.template Run<>(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf_per_scale.GetVectorTypeReference(I0));
                    });
                    static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                        c_thread_buf(Number<c_offset>{}) +=
                            c_thread_buf_per_scale[Number<t>{}] *
                            type_convert<AccDataType>(a_scale_thread_buf[m0]) *
                            type_convert<AccDataType>(b_scale_thread_buf[I0]);
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

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    c_thread_buf_per_scale.Clear();
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
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

                        xdlops_gemm.template Run<>(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf_per_scale.GetVectorTypeReference(I0));
                    });
                    static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, t));
                        c_thread_buf(Number<c_offset>{}) +=
                            c_thread_buf_per_scale[Number<t>{}] *
                            type_convert<AccDataType>(a_scale_thread_buf[m0]) *
                            type_convert<AccDataType>(b_scale_thread_buf[I0]);
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

} // namespace ck
