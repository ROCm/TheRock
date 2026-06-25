// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp"

namespace ck {

// Naive pipeline with lowest resource request per WGP

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
struct BlockwiseGemmWmmaops_pipeline_v1
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
struct BlockwiseGemmWmmaops_pipeline_v1<BlockGemmPipelineScheduler::Intrawave,
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
    // GlobalPrefetchStages: 1
    // LocalPreFillStages: 1
    // LocalPreFetchStages: 0
    // LocalSharedMemoryBuffer: 1
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
    using typename Base::HotLoopInstList;

    using Base::A_K1;
    using Base::A_KRow;
    using Base::B_K1;
    using Base::B_KRow;
    using Base::KRepeat;
    using Base::WmmaK;

    using Base::wmma_gemm;

    using Base::CalculateCThreadOriginDataIndex;
    using Base::
        GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs;
    using Base::GetCThreadBuffer;
    using Base::
        GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs;

    using Base::a_block_desc_k0_m0_m1_m2_k1;
    using Base::b_block_desc_k0_n0_n1_n2_k1;

    using typename Base::Empty;

    static constexpr index_t PrefetchStages  = 1;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    static bool BlockHasHotloop(index_t num_loop) { return num_loop > PrefetchStages; }

    static TailNumber BlockLoopTailNum(index_t num_loop)
    {
        ignore = num_loop;
        return TailNumber::Full;
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

        // Initialize C
        c_thread_buf.Clear();

        auto blockwise_gemm_func = [&]() {
            // Local load
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
                                       make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                       a_block_buf,
                                       a_thread_desc_,
                                       make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                       a_thread_buf);
                    if constexpr(m0 == I0)
                    {
                        if constexpr(ck::is_same<BScaleStruct, Empty>::value == true)
                        {
                            static_for<0, NRepeat, 1>{}([&](auto n0) {
                                b_thread_copy_.Run(b_block_desc_k0_n0_n1_n2_k1,
                                                   make_tuple(I0, n0, k0, I0, I0, I0, I0),
                                                   b_block_buf,
                                                   b_thread_desc_,
                                                   make_tuple(I0, n0, I0, I0, I0, I0, I0),
                                                   b_thread_buf);
                            });
                        }
                        else
                        {
                            static_for<0, NRepeat, 1>{}([&](auto n0) {
                                b_thread_copy_.Run(
                                    b_block_desc_k0_n0_n1_n2_k1,
                                    make_tuple(I0, n0, k0, I0, I0, I0, I0),
                                    b_block_buf,
                                    b_scale_struct.scale_thread_bufs(
                                        I0)[Number<n0 * BScaleStruct::num_scale_k_block +
                                                   k0 / BScaleStruct::num_scale_krepeat>{}],
                                    b_thread_desc_,
                                    make_tuple(I0, n0, I0, I0, I0, I0, I0),
                                    b_thread_buf);
                            });
                        }
                    }

                    static_for<0, KInner, 1>{}([&](auto k_inner) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            vector_type<ComputeTypeA, KPack / A_KRow / KInner> a_thread_vec;
                            vector_type<ComputeTypeB, KPack / B_KRow / KInner> b_thread_vec;

                            static_for<0, KPack / A_KRow / KInner, 1>{}([&](auto ik) {
                                constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(Number<kk / A_K1>{},
                                                   I0,
                                                   I0,
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
                                                   I0,
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
        };

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                block_sync_lds();
                blockwise_gemm_func();

                block_sync_lds();
                b_scale_struct.template GlobalLoad<0>((i + 2) % num_loop_per_scale == 0);
                if constexpr(ck::is_same<BScaleStruct, Empty>::value == false)
                {
                    block_sync_lds();
                }
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                constexpr index_t num_ds_write_inst =
                    HotLoopInstList::A_LDS_Write_Inst_Num + HotLoopInstList::B_LDS_Write_Inst_Num;

                constexpr index_t num_buffer_load_inst = HotLoopInstList::A_Buffer_Load_Inst_Num +
                                                         HotLoopInstList::B_Buffer_Load_Inst_Num;
                static_for<0, num_buffer_load_inst, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                });
                static_for<0, KRepeat, 1>{}([&](auto) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                        if constexpr(m0 == I0)
                        {
                            static_for<0, NRepeat, 1>{}([&](auto) {
                                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                            });
                        }
                        static_for<0, KInner, 1>{}([&](auto) {
                            static_for<0, NRepeat, 1>{}([&](auto) {
                                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // WMMA
                            });
                        });
                    });
                });
                static_for<0, num_ds_write_inst, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                });

                i += 1;
            } while(i < (num_loop - 1));
        }

        // tail
        if constexpr(TailNum == TailNumber::Full)
        {
            block_sync_lds();
            blockwise_gemm_func();
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
        constexpr index_t KPerWaveBlock = wmma_gemm.GetKPerWaveBlk();
        static constexpr auto NumScaleKBlock =
            Number<ck::math::max(AScaleStruct::num_slice_k, BScaleStruct::num_slice_k)>{};

        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeA>(
            Base::a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeB>(
            Base::b_thread_desc_.GetElementSpaceSize());

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

        // Initialize C
        c_thread_buf.Clear();

        auto blockwise_gemm_func = [&]() {
            // Local load
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    Base::a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
                                             make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                             a_block_buf,
                                             Base::a_thread_desc_,
                                             make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                             a_thread_buf);
                });
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    Base::b_thread_copy_.Run(b_block_desc_k0_n0_n1_n2_k1,
                                             make_tuple(I0, n0, k0, I0, I0, I0, I0),
                                             b_block_buf,
                                             Base::b_thread_desc_,
                                             make_tuple(I0, n0, k0, I0, I0, I0, I0),
                                             b_thread_buf);
                });
            });

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
                                        a_thread_buf[Number<Base::a_thread_desc_.CalculateOffset(
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
                                        b_thread_buf[Number<Base::b_thread_desc_.CalculateOffset(
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
        };

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_scale_struct.template GlobalLoad<0>((i + 2) % num_loop_per_scale == 0);
                b_scale_struct.template GlobalLoad<0>((i + 2) % num_loop_per_scale == 0);

                block_sync_lds();
                blockwise_gemm_func();

                block_sync_lds();
                c_scale_struct.Load(a_scale_struct, b_scale_struct);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                i += 1;
            } while(i < (num_loop - 1));
        }

        // tail
        if constexpr(TailNum == TailNumber::Full)
        {
            block_sync_lds();
            blockwise_gemm_func();
        }
    }

    protected:
    // A[MRepeat, I1, I1, KPack]
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<KPack / A_K1 / A_KRow>{}, I1, I1, I1, I1, I1, Number<A_K1>{}));

    // B[NRepeat, N1, N2, KPack]
    static constexpr auto b_thread_desc_ = make_naive_tensor_descriptor_packed(make_tuple(
        Number<KPack / B_K1 / B_KRow>{}, Number<NRepeat>{}, I1, I1, I1, I1, Number<B_K1>{}));

    using AThreadCopy =
        ThreadwiseTensorSliceTransfer_v4<ADataType,
                                         ComputeTypeA,
                                         decltype(a_block_desc_k0_m0_m1_m2_k1),
                                         decltype(a_thread_desc_),
                                         Sequence<KPack / A_K1 / A_KRow, 1, 1, 1, 1, 1, A_K1>,
                                         Sequence<0, 1, 2, 3, 4, 5, 6>,
                                         6,
                                         A_K1,
                                         A_K1>;

    using BThreadCopy =
        ThreadwiseTensorSliceTransfer_v4<BDataType,
                                         ComputeTypeB,
                                         decltype(b_block_desc_k0_n0_n1_n2_k1),
                                         decltype(b_thread_desc_),
                                         Sequence<KPack / B_K1 / B_KRow, 1, 1, 1, 1, 1, B_K1>,
                                         Sequence<0, 1, 2, 3, 4, 5, 6>,
                                         6,
                                         B_K1,
                                         B_K1>;

    AThreadCopy a_thread_copy_{Base::CalculateAThreadOriginDataIndex()};
    BThreadCopy b_thread_copy_{Base::CalculateBThreadOriginDataIndex()};
    using Base::c_thread_desc_;
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
struct BlockwiseGemmWmmaops_pipeline_v1<BlockGemmPipelineScheduler::Interwave,
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
    // GlobalPrefetchStages: 1
    // LocalPreFillStages: 1
    // LocalPreFetchStages: 0
    // LocalSharedMemoryBuffer: 1
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

    using Base::A_K1;
    using Base::A_KRow;
    using Base::B_K1;
    using Base::B_KRow;
    using Base::KRepeat;
    using Base::WmmaK;

    using Base::wmma_gemm;

    using Base::CalculateCThreadOriginDataIndex;
    using Base::
        GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs;
    using Base::GetCThreadBuffer;
    using Base::
        GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs;

    using Base::a_block_desc_k0_m0_m1_m2_k1;
    using Base::b_block_desc_k0_n0_n1_n2_k1;

    using typename Base::Empty;

    static constexpr index_t NumKClusters      = CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING_MAC_CLUSTERS;
    static constexpr index_t KRepeatPerCluster = math::max(KRepeat / NumKClusters, 1);

    static constexpr index_t PrefetchStages  = 1;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    static bool BlockHasHotloop(index_t num_loop) { return num_loop > PrefetchStages; }

    static TailNumber BlockLoopTailNum(index_t num_loop)
    {
        ignore = num_loop;
        return TailNumber::Full;
    }

    template <typename AScaleStruct, typename BScaleStruct>
    struct KLoopParams
    {
        static constexpr auto KRepeatNoScale = 1;
        static constexpr auto NumScaleKBlock =
            Number<ck::math::max(AScaleStruct::num_slice_k, BScaleStruct::num_slice_k)>{};
        static constexpr auto KRepeatPerNumScaleKBlock = KRepeatPerCluster / NumScaleKBlock;
    };

    template <>
    struct KLoopParams<Empty, Empty>
    {
        static constexpr index_t KRepeatNoScale           = KRepeatPerCluster;
        static constexpr index_t NumScaleKBlock           = 1;
        static constexpr index_t KRepeatPerNumScaleKBlock = 1;
    };

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

        // Initialize C
        c_thread_buf.Clear();

        auto blockwise_gemm_func = [&]() {
            static_for<0, KRepeat, KRepeatPerCluster>{}([&](auto k0_offset) {
                static_for<0, KRepeatPerCluster, 1>{}([&](auto k0_inner) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
                                           make_tuple(I0, m0, k0_offset + k0_inner, I0, I0, I0, I0),
                                           a_block_buf,
                                           a_thread_desc_,
                                           make_tuple(I0, m0, k0_inner, I0, I0, I0, I0),
                                           a_thread_buf);
                    });
                    if constexpr(ck::is_same<BScaleStruct, Empty>::value == true)
                    {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(
                                b_block_desc_k0_n0_n1_n2_k1,
                                make_tuple(I0, n0, k0_offset + k0_inner, I0, I0, I0, I0),
                                b_block_buf,
                                b_thread_desc_,
                                make_tuple(I0, n0, k0_inner, I0, I0, I0, I0),
                                b_thread_buf);
                        });
                    }
                    else
                    {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(
                                b_block_desc_k0_n0_n1_n2_k1,
                                make_tuple(I0, n0, k0_offset + k0_inner, I0, I0, I0, I0),
                                b_block_buf,
                                b_scale_struct.scale_thread_bufs(I0)[Number<
                                    n0 * BScaleStruct::num_scale_k_block +
                                    (k0_offset + k0_inner) / BScaleStruct::num_scale_krepeat>{}],
                                b_thread_desc_,
                                make_tuple(I0, n0, k0_inner, I0, I0, I0, I0),
                                b_thread_buf);
                        });
                    }
                });

                __builtin_amdgcn_sched_barrier(0);
                // NOTE: Synchronize threads in a workgroup at the start of each MAC cluster,
                // but except the first, as we can shorten non-MAC cluster a bit and there's no
                // observable negative impact. The desired effect is waves in a workgroup
                // executing MAC in sync. This avoids some out-of-sync waves hijacking MAC
                // resource from other workgroups and reducing the chance of latency hiding by
                // waiting for the rest of the workgroup at the eventual sync point.
                if constexpr(k0_offset != 0 || KRepeat == 1)
                {
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                }
                static_for<0, KRepeatPerCluster, 1>{}([&](auto k0_inner) {
                    static_for<0, KInner, 1>{}([&](auto k_inner) {
                        static_for<0, MRepeat, 1>{}([&](auto m0) {
                            static_for<0, NRepeat, 1>{}([&](auto n0) {
                                vector_type<ComputeTypeA, KPack / A_KRow / KInner> a_thread_vec;
                                vector_type<ComputeTypeB, KPack / B_KRow / KInner> b_thread_vec;

                                static_for<0, KPack / A_KRow / KInner, 1>{}([&](auto ik) {
                                    constexpr index_t kk = ik + k_inner * KPerWaveBlock;
                                    a_thread_vec.template AsType<ComputeTypeA>()(ik) =
                                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                            make_tuple(Number<kk / A_K1>{},
                                                       m0,
                                                       k0_inner,
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
                                                       k0_inner,
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

                                // The block_sync_lds() here performs double duty:
                                // A) safeguard against data hazard.
                                // B) reduce VMEM FIFO congestion by applying small delays to
                                // different wavefronts.
                                // It is performed near the end of MAC cluster to minimize lgkmcnt
                                // penalty
                                if constexpr(k0_offset + k0_inner == KRepeat - 1 &&
                                             m0 == MRepeat - 1 && n0 == NRepeat - 1)
                                {
                                    __builtin_amdgcn_sched_barrier(0);
                                    block_sync_lds();
                                    __builtin_amdgcn_sched_barrier(0);
                                }
                                wmma_gemm.Run(
                                    a_thread_vec.template AsType<wmma_input_type_a>(),
                                    b_thread_vec.template AsType<wmma_input_type_b>(),
                                    c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                                if constexpr(k0_inner == 0 && m0 == 0 && n0 == 0)
                                {
                                    __builtin_amdgcn_sched_barrier(0);
                                    __builtin_amdgcn_s_setprio(1);
                                    __builtin_amdgcn_sched_barrier(0);
                                }
                            });
                        });
                    });
                });

                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
            });
        };

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                block_sync_lds();
                blockwise_gemm_func();

                b_scale_struct.template GlobalLoad<0>((i + 2) % num_loop_per_scale == 0);
                if constexpr(ck::is_same<BScaleStruct, Empty>::value == false)
                {
                    block_sync_lds();
                }
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                i += 1;
            } while(i < (num_loop - 1));
        }

        // tail
        if constexpr(TailNum == TailNumber::Full)
        {
            block_sync_lds();
            blockwise_gemm_func();
        }
    }

    protected:
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor(make_tuple(Number<KPack / A_K1 / A_KRow>{},
                                                Number<MRepeat>{},
                                                Number<KRepeatPerCluster>{},
                                                I1,
                                                I1,
                                                I1,
                                                Number<A_K1>{}),
                                     make_tuple(Number<A_K1>{},
                                                Number<KPack / A_KRow>{},
                                                Number<KPack / A_KRow * MRepeat>{},
                                                I0,
                                                I0,
                                                I0,
                                                I1));

    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor(make_tuple(Number<KPack / B_K1 / B_KRow>{},
                                                Number<NRepeat>{},
                                                Number<KRepeatPerCluster>{},
                                                I1,
                                                I1,
                                                I1,
                                                Number<B_K1>{}),
                                     make_tuple(Number<B_K1>{},
                                                Number<KPack / B_KRow>{},
                                                Number<KPack / B_KRow * NRepeat>{},
                                                I0,
                                                I0,
                                                I0,
                                                I1));

    using AThreadCopy =
        ThreadwiseTensorSliceTransfer_v4<ADataType,
                                         ComputeTypeA,
                                         decltype(a_block_desc_k0_m0_m1_m2_k1),
                                         decltype(a_thread_desc_),
                                         Sequence<KPack / A_K1 / A_KRow, 1, 1, 1, 1, 1, A_K1>,
                                         Sequence<0, 1, 2, 3, 4, 5, 6>,
                                         6,
                                         A_K1,
                                         A_K1>;

    using BThreadCopy =
        ThreadwiseTensorSliceTransfer_v4<BDataType,
                                         ComputeTypeB,
                                         decltype(b_block_desc_k0_n0_n1_n2_k1),
                                         decltype(b_thread_desc_),
                                         Sequence<KPack / B_K1 / B_KRow, 1, 1, 1, 1, 1, B_K1>,
                                         Sequence<0, 1, 2, 3, 4, 5, 6>,
                                         6,
                                         B_K1,
                                         B_K1>;

    AThreadCopy a_thread_copy_{Base::CalculateAThreadOriginDataIndex()};
    BThreadCopy b_thread_copy_{Base::CalculateBThreadOriginDataIndex()};
    using Base::c_thread_desc_;
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
struct BlockwiseGemmWmmaops_pipeline_v1<BlockGemmPipelineScheduler::Intrawave,
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
                                        true>
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
    // GlobalPrefetchStages: 2
    // LocalPreFillStages: 1
    // LocalPreFetchStages: 1
    // LocalSharedMemoryBuffer: 1
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
    using Base::MWaves;
    using Base::WaveSize;
    using typename Base::HotLoopInstList;

    using Base::A_K1;
    using Base::A_KRow;
    using Base::B_K1;
    using Base::B_KRow;
    using Base::KRepeat;
    using Base::WmmaK;

    using Base::wmma_gemm;

    using Base::CalculateCThreadOriginDataIndex;
    using Base::
        GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs;
    using Base::GetCThreadBuffer;
    using Base::
        GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs;

    using Base::a_block_desc_k0_m0_m1_m2_k1;
    using Base::b_block_desc_k0_n0_n1_n2_k1;

    using typename Base::Empty;

    static constexpr index_t PrefetchStages  = 2;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 2;

    static bool BlockHasHotloop(index_t num_loop) { return num_loop > PrefetchStages; }

    static TailNumber BlockLoopTailNum(index_t num_loop)
    {
        return num_loop % 2 == 0 ? TailNumber::Even : TailNumber::Odd;
    }

    __device__ static constexpr auto HotLoopScheduler()
    {
        constexpr auto num_ds_read_inst_a     = HotLoopInstList::A_LDS_Read_Inst_Num;
        constexpr auto num_buffer_load_inst_a = HotLoopInstList::A_Buffer_Load_Inst_Num;
        constexpr auto num_buffer_load_inst_b = HotLoopInstList::B_Buffer_Load_Inst_Num * MWaves;
        constexpr auto wmma_interleave        = 2;
        // B global
        static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
            ignore = i;
            if constexpr(MPerBlock >= 128 && NPerBlock >= 128)
            {
                __builtin_amdgcn_sched_group_barrier(0x008, 2 * wmma_interleave, 0);
            }
            else
            {
                __builtin_amdgcn_sched_group_barrier(0x008, wmma_interleave, 0);
            }
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
        });

        // A global
        static_for<0, num_buffer_load_inst_a, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // WMMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // WMMA
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
        });

        // A local
        static_for<0, num_ds_read_inst_a, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // WMMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
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
                        const BBlockDesc&,
                        BBlockTransfer& b_blockwise_copy,
                        const BGridBuffer& b_grid_buf,
                        BBlockBuffer&,
                        const BBlockTransferStep& b_block_copy_step,
                        CThreadBuffer& c_thread_buf,
                        AScaleStruct&,
                        BScaleStruct&,
                        index_t num_loop,
                        index_t) const
    {
        __builtin_amdgcn_sched_barrier(0);
        constexpr index_t KPerWaveBlock = wmma_gemm.GetKPerWaveBlk();

        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeB>(
            b_thread_desc_.GetElementSpaceSize());

        StaticallyIndexedArray<decltype(b_thread_buf), Number<2>{}> b_thread_bufs;
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);

        // Global prefetch A1 B1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.Run(b_grid_desc,
                             b_grid_buf,
                             b_block_desc_k0_n0_n1_n2_k1,
                             b_block_origin_idx,
                             b_thread_bufs(I0));

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        __builtin_amdgcn_sched_barrier(0);

        // Local prefill A1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

        // Global prefetch A2
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

        // Local prefetch A1
        block_sync_lds();
        static_for<0, MRepeat, 1>{}([&](auto m0) {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
                                   make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                   a_block_buf,
                                   a_thread_desc_,
                                   make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                   a_thread_buf);
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
                auto LoopFunc = [&](auto wmma_reg_buf, auto local_read_buf) {
                    b_blockwise_copy.Run(b_grid_desc,
                                         b_grid_buf,
                                         b_block_desc_k0_n0_n1_n2_k1,
                                         b_block_origin_idx,
                                         b_thread_bufs(local_read_buf));

                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                    block_sync_lds();

                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, wmma_reg_buf);

                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, local_read_buf);
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            static_for<0, KRepeat, 1>{}([&](auto k0) {
                                vector_type<ComputeTypeA, KPack / A_KRow / KInner> a_thread_vec;
                                vector_type<ComputeTypeB, KPack / B_KRow / KInner> b_thread_vec;
                                static_for<0, KInner, 1>{}([&](auto k_inner) {
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
                                            b_thread_bufs[wmma_reg_buf]
                                                         [Number<b_thread_desc_.CalculateOffset(
                                                             make_tuple(Number<kk / B_K1>{},
                                                                        I0,
                                                                        I0,
                                                                        n0,
                                                                        I0,
                                                                        k0,
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

                    // loop prefetch copy
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, KRepeat, 1>{}([&](auto k0) {
                            a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
                                               make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                               a_block_buf,
                                               a_thread_desc_,
                                               make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                               a_thread_buf);
                        });
                    });

                    HotLoopScheduler();
                    __builtin_amdgcn_sched_barrier(0);
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
                                 b_block_desc_k0_n0_n1_n2_k1,
                                 b_block_origin_idx,
                                 b_thread_bufs(I1));

            block_sync_lds();

            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        vector_type<ComputeTypeA, KPack / A_KRow / KInner> a_thread_vec;
                        vector_type<ComputeTypeB, KPack / B_KRow / KInner> b_thread_vec;
                        static_for<0, KInner, 1>{}([&](auto k_inner) {
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
                                    b_thread_bufs[I0][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(Number<kk / B_K1>{},
                                                   I0,
                                                   I0,
                                                   n0,
                                                   I0,
                                                   k0,
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

            // tail Local Prefetch A1
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
                                       make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                       a_block_buf,
                                       a_thread_desc_,
                                       make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                       a_thread_buf);
                });
            });

            __builtin_amdgcn_sched_barrier(0);

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        vector_type<ComputeTypeA, KPack / A_KRow / KInner> a_thread_vec;
                        vector_type<ComputeTypeB, KPack / B_KRow / KInner> b_thread_vec;
                        static_for<0, KInner, 1>{}([&](auto k_inner) {
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
                                    b_thread_bufs[I1][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(Number<kk / B_K1>{},
                                                   I0,
                                                   I0,
                                                   n0,
                                                   I0,
                                                   k0,
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
        else if constexpr(TailNum == TailNumber::Odd)
        {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        vector_type<ComputeTypeA, KPack / A_KRow / KInner> a_thread_vec;
                        vector_type<ComputeTypeB, KPack / B_KRow / KInner> b_thread_vec;
                        static_for<0, KInner, 1>{}([&](auto k_inner) {
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
                                    b_thread_bufs[I0][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(Number<kk / B_K1>{},
                                                   I0,
                                                   I0,
                                                   n0,
                                                   I0,
                                                   k0,
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
                        const BBlockDesc&,
                        BBlockTransfer& b_blockwise_copy,
                        const BGridBuffer& b_grid_buf,
                        BBlockBuffer&,
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

        StaticallyIndexedArray<decltype(b_thread_buf), Number<2>{}> b_thread_bufs;
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);

        using CScaleStruct  = typename Base::template CScale<AScaleStruct, BScaleStruct>;
        auto c_scale_struct = CScaleStruct{};

        auto gemm_core_func = [&](auto reg_buf) {
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
                                        b_thread_bufs[reg_buf]
                                                     [Number<b_thread_desc_.CalculateOffset(
                                                         make_tuple(Number<kk / B_K1>{},
                                                                    I0,
                                                                    I0,
                                                                    n0,
                                                                    I0,
                                                                    k_index,
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
        };

        auto a_local_prefetch_func = [&]() {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    a_thread_copy_.Run(a_block_desc_k0_m0_m1_m2_k1,
                                       make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                       a_block_buf,
                                       a_thread_desc_,
                                       make_tuple(I0, m0, k0, I0, I0, I0, I0),
                                       a_thread_buf);
                });
            });
        };

        // Global prefetch A1 B1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.Run(b_grid_desc,
                             b_grid_buf,
                             b_block_desc_k0_n0_n1_n2_k1,
                             b_block_origin_idx,
                             b_thread_bufs(I0));

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Scales global load
        a_scale_struct.template GlobalLoad<0>(num_loop_per_scale == 1);
        b_scale_struct.template GlobalLoad<0>(num_loop_per_scale == 1);

        __builtin_amdgcn_sched_barrier(0);

        c_scale_struct.Load(a_scale_struct, b_scale_struct);

        // Local prefill A1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

        // Global prefetch A2
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

        // Local prefetch A1
        block_sync_lds();
        a_local_prefetch_func();

        // Initialize C
        c_thread_buf.Clear();

        __builtin_amdgcn_sched_barrier(0);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                auto LoopFunc = [&](auto wmma_reg_buf, auto local_read_buf) {
                    b_blockwise_copy.Run(b_grid_desc,
                                         b_grid_buf,
                                         b_block_desc_k0_n0_n1_n2_k1,
                                         b_block_origin_idx,
                                         b_thread_bufs(local_read_buf));

                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                    block_sync_lds();

                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, wmma_reg_buf);

                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, local_read_buf);
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

                    a_scale_struct.template GlobalLoad<0>(
                        (i + 2 + wmma_reg_buf) % num_loop_per_scale == 0);
                    b_scale_struct.template GlobalLoad<0>(
                        (i + 2 + wmma_reg_buf) % num_loop_per_scale == 0);

                    gemm_core_func(wmma_reg_buf);

                    block_sync_lds();

                    // loop prefetch copy
                    a_local_prefetch_func();

                    c_scale_struct.Load(a_scale_struct, b_scale_struct);

                    // HotLoopScheduler();
                    __builtin_amdgcn_sched_barrier(0);
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
                                 b_block_desc_k0_n0_n1_n2_k1,
                                 b_block_origin_idx,
                                 b_thread_bufs(I1));

            block_sync_lds();

            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

            a_scale_struct.template GlobalLoad<0>(num_loop % num_loop_per_scale == 0);
            b_scale_struct.template GlobalLoad<0>(num_loop % num_loop_per_scale == 0);

            gemm_core_func(I0);

            block_sync_lds();

            // tail Local Prefetch A1
            a_local_prefetch_func();

            c_scale_struct.Load(a_scale_struct, b_scale_struct);

            __builtin_amdgcn_sched_barrier(0);

            gemm_core_func(I1);
            // Let's leak last WMMA block to epilogue region, cover the potential lds-shuffle
            // latency
            // __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            gemm_core_func(I0);
        }
    }

    protected:
    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(Number<KPack / B_K1 / B_KRow>{},
                                                       I1,
                                                       I1,
                                                       Number<NRepeat>{},
                                                       I1,
                                                       Number<KRepeat>{},
                                                       Number<B_K1>{}));

    using Base::a_thread_copy_;
    using Base::a_thread_desc_;
    using Base::c_thread_desc_;
};

} // namespace ck
