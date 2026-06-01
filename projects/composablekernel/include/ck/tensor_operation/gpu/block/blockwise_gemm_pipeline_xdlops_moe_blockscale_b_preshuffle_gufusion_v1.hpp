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
          index_t MScaleBlock,
          index_t NScaleBlock,
          index_t KScaleBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPacks>
struct BlockwiseGemmXdlops_pipeline_moe_blockscale_bpreshuffle_gufusion_v1
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
          index_t MScaleBlock,
          index_t NScaleBlock,
          index_t KScaleBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack
          // ,bool TransposeC //disable transposec right now...
          >
struct BlockwiseGemmXdlops_pipeline_moe_blockscale_bpreshuffle_gufusion_v1<
    BlockGemmPipelineScheduler::Intrawave,
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
    MScaleBlock,
    NScaleBlock,
    KScaleBlock,
    MPerXDL,
    NPerXDL,
    MRepeat,
    NRepeat,
    KPack> : BlockwiseGemmXdlops_pipeline_base<BlockSize,
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
    using Base::NWaves;
    using Base::WaveSize;

    static constexpr index_t PrefetchStages  = 2;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 2;

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
        constexpr auto num_ds_read_inst_a     = HotLoopInstList::A_LDS_Read_Inst_Num;
        constexpr auto num_buffer_load_inst_a = HotLoopInstList::A_Buffer_Load_Inst_Num;
        constexpr auto num_buffer_load_inst_b =
            HotLoopInstList::B_Buffer_Load_Inst_Num * MWaves * 2;
        constexpr auto mfma_interleave = MPerXDL == 32 ? 1 : 2;
        // B global
        static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
            ignore = i;
            if constexpr(MPerBlock >= 128 && NPerBlock >= 64)
            {
                __builtin_amdgcn_sched_group_barrier(0x008, 2 * mfma_interleave, 0);
            }
            else
            {
                __builtin_amdgcn_sched_group_barrier(0x008, mfma_interleave, 0);
            }
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
        });

        // A global
        static_for<0, num_buffer_load_inst_a, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
        });

        // A local
        static_for<0, MPerXDL == 32 ? num_ds_read_inst_a / 2 : num_ds_read_inst_a, 1>{}(
            [&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                     // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, MPerXDL == 32 ? 2 : 1, 0); // DS read
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
        BBlockTransfer& b_blockwise_copy_up,
        const BGridBuffer& b_grid_buf,
        const BGridBuffer& b_grid_buf_up,
        BBlockBuffer& b_block_buf,
        const BBlockTransferStep& b_block_copy_step,
        // CThread
        const CScaleThreadDesc& c_scale_thread_desc,
        CThreadBuffer& c_thread_buf,
        CThreadBuffer& c_thread_buf_up,
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
        BScaleThreadTransfer& b_scale_thread_copy_up,
        const BScaleGridBuffer& b_scale_grid_buf,
        const BScaleGridBuffer& b_scale_grid_buf_up,
        const BScaleThreadTransferStep& b_scale_thread_copy_step,
        // num_loop
        index_t num_loop) const
    {
        ignore = b_block_desc;
        ignore = b_block_buf;
        // __builtin_amdgcn_sched_barrier(0);
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataType>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataType>(
            b_thread_desc_.GetElementSpaceSize());

        StaticallyIndexedArray<decltype(b_thread_buf), Number<2>{}> b_thread_bufs;
        StaticallyIndexedArray<decltype(b_thread_buf), Number<2>{}> b_thread_bufs_up;
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0);

        auto a_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            a_scale_thread_desc.GetElementSpaceSize());
        auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            b_scale_thread_desc.GetElementSpaceSize());
        auto b_scale_thread_buf_up = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            b_scale_thread_desc.GetElementSpaceSize());
        auto c_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            c_scale_thread_desc.GetElementSpaceSize());
        auto c_scale_thread_buf_up = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            c_scale_thread_desc.GetElementSpaceSize());

        // Global prefetch A1 B1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
        b_blockwise_copy.Run(b_grid_desc,
                             b_grid_buf,
                             b_block_desc_n0_n1_k0_k1,
                             b_block_origin_idx,
                             b_thread_bufs(I0));
        b_blockwise_copy_up.Run(b_grid_desc,
                                b_grid_buf_up,
                                b_block_desc_n0_n1_k0_k1,
                                b_block_origin_idx,
                                b_thread_bufs_up(I0));

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        b_blockwise_copy_up.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        a_scale_thread_copy.Run(a_scale_grid_desc,
                                a_scale_grid_buf,
                                a_scale_thread_desc,
                                make_tuple(I0, I0),
                                a_scale_thread_buf);

        if constexpr(NumKBlockPerScale == 1)
        {
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<1>{}));
        }
        else
        {
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<0>{}));
        }

        b_scale_thread_copy.Run(b_scale_grid_desc,
                                b_scale_grid_buf,
                                b_scale_thread_desc,
                                make_tuple(I0, I0),
                                b_scale_thread_buf);

        b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc, b_scale_thread_copy_step);

        b_scale_thread_copy_up.Run(b_scale_grid_desc,
                                   b_scale_grid_buf_up,
                                   b_scale_thread_desc,
                                   make_tuple(I0, I0),
                                   b_scale_thread_buf_up);

        b_scale_thread_copy_up.MoveSrcSliceWindow(b_scale_grid_desc, b_scale_thread_copy_step);

        // __builtin_amdgcn_sched_barrier(0);

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
                    c_scale_thread_buf_up(Number<c_offset>{}) =
                        a_scale_thread_buf[Number<a_offset>{}] *
                        b_scale_thread_buf_up[Number<b_offset>{}];
                });
            });
        });

        // Local prefill A1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);

        // Global prefetch A2
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

        a_scale_thread_copy.Run(a_scale_grid_desc,
                                a_scale_grid_buf,
                                a_scale_thread_desc,
                                make_tuple(I0, I0),
                                a_scale_thread_buf);

        if constexpr(NumKBlockPerScale == 1)
        {
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<1>{}));
        }
        else
        {
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<0>{}));
        }

        b_scale_thread_copy.Run(b_scale_grid_desc,
                                b_scale_grid_buf,
                                b_scale_thread_desc,
                                make_tuple(I0, I0),
                                b_scale_thread_buf);

        b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc, b_scale_thread_copy_step);

        b_scale_thread_copy_up.Run(b_scale_grid_desc,
                                   b_scale_grid_buf_up,
                                   b_scale_thread_desc,
                                   make_tuple(I0, I0),
                                   b_scale_thread_buf_up);

        b_scale_thread_copy_up.MoveSrcSliceWindow(b_scale_grid_desc, b_scale_thread_copy_step);

        StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                  AccDataType,
                                  1,
                                  xdlops_gemm.GetRegSizePerXdlops(),
                                  true>
            c_thread_buf_per_scale;
        StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                  AccDataType,
                                  1,
                                  xdlops_gemm.GetRegSizePerXdlops(),
                                  true>
            c_thread_buf_per_scale_up;

        // Local prefetch A1
        block_sync_lds();
        static_for<0, MRepeat, 1>{}([&](auto m0) {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, KGroup, 1>{}([&](auto kg0) {
                    a_thread_copy_.Run(
                        a_block_desc_m0_m1_m2_k0_k1_k2,
                        make_tuple(m0, I0, I0, Number<k0 * KGroup + kg0>{}, I0, I0),
                        a_block_buf,
                        a_thread_desc_,
                        make_tuple(m0, I0, I0, k0, I0, Number<kg0 * KPack / KGroup>{}),
                        a_thread_buf);
                });
            });
        });

        // Initialize C
        c_thread_buf.Clear();
        c_thread_buf_up.Clear();

        // __builtin_amdgcn_sched_barrier(0);

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

                    b_blockwise_copy_up.Run(b_grid_desc,
                                            b_grid_buf_up,
                                            b_block_desc_n0_n1_k0_k1,
                                            b_block_origin_idx,
                                            b_thread_bufs_up(local_read_buf));
                    b_blockwise_copy_up.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                    block_sync_lds();
                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, mfma_reg_buf);

                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, local_read_buf);
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            static_for<0, num_scale_k_block, 1>{}([&](auto kscale0) {
                                static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                                    c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                        .template AsType<AccDataType>()(Number<t>{}) = 0;
                                    c_thread_buf_per_scale_up.GetVectorTypeReference(Number<0>{})
                                        .template AsType<AccDataType>()(Number<t>{}) = 0;
                                });
                                vector_type<AccDataType, 2> c_scale_thread_vec;
                                vector_type<AccDataType, 2> c_scale_thread_vec_up;
                                constexpr index_t cscale_offset =
                                    CScaleThreadDesc{}.CalculateOffset(
                                        make_tuple(kscale0, m0, n0 * num_scale_n_block / NRepeat));

                                c_scale_thread_vec.template AsType<AccDataType>()(Number<0>{}) =
                                    c_scale_thread_buf[Number<cscale_offset>{}];
                                c_scale_thread_vec.template AsType<AccDataType>()(Number<1>{}) =
                                    c_scale_thread_buf[Number<cscale_offset>{}];
                                c_scale_thread_vec_up.template AsType<AccDataType>()(Number<0>{}) =
                                    c_scale_thread_buf_up[Number<cscale_offset>{}];
                                c_scale_thread_vec_up.template AsType<AccDataType>()(Number<1>{}) =
                                    c_scale_thread_buf_up[Number<cscale_offset>{}];

                                static_for<0, KRepeat / num_scale_k_block, 1>{}([&](auto k0) {
                                    vector_type<ComputeDataType, KPack> a_thread_vec;
                                    vector_type<ComputeDataType, KPack> b_thread_vec;
                                    vector_type<ComputeDataType, KPack> b_thread_vec_up;

                                    static_for<0, KPack, 1>{}([&](auto ik) {
                                        a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                            a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                                make_tuple(m0,
                                                           I0,
                                                           I0,
                                                           kscale0 * KRepeat / num_scale_k_block +
                                                               k0,
                                                           I0,
                                                           ik))>{}];
                                        b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                            b_thread_bufs[mfma_reg_buf][Number<
                                                b_thread_desc_.CalculateOffset(make_tuple(
                                                    n0,
                                                    I0,
                                                    kscale0 * KRepeat / num_scale_k_block + k0,
                                                    ik))>{}];
                                        b_thread_vec_up.template AsType<ComputeDataType>()(ik) =
                                            b_thread_bufs_up[mfma_reg_buf][Number<
                                                b_thread_desc_.CalculateOffset(make_tuple(
                                                    n0,
                                                    I0,
                                                    kscale0 * KRepeat / num_scale_k_block + k0,
                                                    ik))>{}];
                                    });

                                    using mfma_input_type =
                                        typename vector_type<ComputeDataType,
                                                             xdlops_gemm.K1PerXdlops>::type;

                                    xdlops_gemm.template Run<>(
                                        a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{}));
                                    xdlops_gemm.template Run<>(
                                        a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec_up.template AsType<mfma_input_type>(),
                                        c_thread_buf_per_scale_up.GetVectorTypeReference(
                                            Number<0>{}));
                                });

                                constexpr index_t c_offset =
                                    c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                                static_for<0, xdlops_gemm.GetRegSizePerXdlops() / 2, 1>{}(
                                    [&](auto t) {
                                        using pk_fma_type =
                                            typename vector_type<AccDataType, 2>::type;

                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                            .template AsType<pk_fma_type>()(t) =
                                            __builtin_elementwise_fma(
                                                c_thread_buf_per_scale
                                                    .GetVectorTypeReference(Number<0>{})
                                                    .template AsType<pk_fma_type>()[t],
                                                c_scale_thread_vec
                                                    .template AsType<pk_fma_type>()[Number<0>{}],
                                                c_thread_buf
                                                    .GetVectorTypeReference(Number<c_offset>{})
                                                    .template AsType<pk_fma_type>()[t]);
                                        c_thread_buf_up.GetVectorTypeReference(Number<c_offset>{})
                                            .template AsType<pk_fma_type>()(t) =
                                            __builtin_elementwise_fma(
                                                c_thread_buf_per_scale_up
                                                    .GetVectorTypeReference(Number<0>{})
                                                    .template AsType<pk_fma_type>()[t],
                                                c_scale_thread_vec_up
                                                    .template AsType<pk_fma_type>()[Number<0>{}],
                                                c_thread_buf_up
                                                    .GetVectorTypeReference(Number<c_offset>{})
                                                    .template AsType<pk_fma_type>()[t]);
                                    });
                            });
                        });
                    });

                    block_sync_lds();

                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, KRepeat, 1>{}([&](auto k0) {
                            static_for<0, KGroup, 1>{}([&](auto kg0) {
                                a_thread_copy_.Run(
                                    a_block_desc_m0_m1_m2_k0_k1_k2,
                                    make_tuple(m0, I0, I0, Number<k0 * KGroup + kg0>{}, I0, I0),
                                    a_block_buf,
                                    a_thread_desc_,
                                    make_tuple(m0, I0, I0, k0, I0, Number<kg0 * KPack / KGroup>{}),
                                    a_thread_buf);
                            });
                        });
                    });

                    HotLoopScheduler();
                    __builtin_amdgcn_sched_barrier(0);

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
                                c_scale_thread_buf_up(Number<c_offset>{}) =
                                    a_scale_thread_buf[Number<a_offset>{}] *
                                    b_scale_thread_buf_up[Number<b_offset>{}];
                            });
                        });
                    });

                    a_scale_thread_copy.Run(a_scale_grid_desc,
                                            a_scale_grid_buf,
                                            a_scale_thread_desc,
                                            make_tuple(I0, I0),
                                            a_scale_thread_buf);

                    if constexpr(NumKBlockPerScale == 1)
                    {
                        a_scale_thread_copy.MoveSrcSliceWindow(
                            a_scale_grid_desc, a_scale_thread_copy_step.At(Number<1>{}));
                    }
                    else
                    {
                        a_scale_thread_copy.MoveSrcSliceWindow(
                            a_scale_grid_desc, a_scale_thread_copy_step.At(Number<0>{}));
                    }

                    b_scale_thread_copy.Run(b_scale_grid_desc,
                                            b_scale_grid_buf,
                                            b_scale_thread_desc,
                                            make_tuple(I0, I0),
                                            b_scale_thread_buf);

                    b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                           b_scale_thread_copy_step);
                    b_scale_thread_copy_up.Run(b_scale_grid_desc,
                                               b_scale_grid_buf_up,
                                               b_scale_thread_desc,
                                               make_tuple(I0, I0),
                                               b_scale_thread_buf_up);

                    b_scale_thread_copy_up.MoveSrcSliceWindow(b_scale_grid_desc,
                                                              b_scale_thread_copy_step);
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

            b_blockwise_copy_up.Run(b_grid_desc,
                                    b_grid_buf_up,
                                    b_block_desc_n0_n1_k0_k1,
                                    b_block_origin_idx,
                                    b_thread_bufs_up(I1));
            block_sync_lds();
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, num_scale_k_block, 1>{}([&](auto kscale0) {
                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                            c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                .template AsType<AccDataType>()(Number<t>{}) = 0;
                            c_thread_buf_per_scale_up.GetVectorTypeReference(Number<0>{})
                                .template AsType<AccDataType>()(Number<t>{}) = 0;
                        });
                        vector_type<AccDataType, 2> c_scale_thread_vec;
                        vector_type<AccDataType, 2> c_scale_thread_vec_up;
                        constexpr index_t cscale_offset = CScaleThreadDesc{}.CalculateOffset(
                            make_tuple(kscale0, m0, n0 * num_scale_n_block / NRepeat));

                        c_scale_thread_vec.template AsType<AccDataType>()(Number<0>{}) =
                            c_scale_thread_buf[Number<cscale_offset>{}];
                        c_scale_thread_vec.template AsType<AccDataType>()(Number<1>{}) =
                            c_scale_thread_buf[Number<cscale_offset>{}];
                        c_scale_thread_vec_up.template AsType<AccDataType>()(Number<0>{}) =
                            c_scale_thread_buf_up[Number<cscale_offset>{}];
                        c_scale_thread_vec_up.template AsType<AccDataType>()(Number<1>{}) =
                            c_scale_thread_buf_up[Number<cscale_offset>{}];

                        static_for<0, KRepeat / num_scale_k_block, 1>{}([&](auto k0) {
                            vector_type<ComputeDataType, KPack> a_thread_vec;
                            vector_type<ComputeDataType, KPack> b_thread_vec;
                            vector_type<ComputeDataType, KPack> b_thread_vec_up;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0,
                                                   I0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   I0,
                                                   ik))>{}];
                                b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                    b_thread_bufs[I0][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   ik))>{}];
                                b_thread_vec_up.template AsType<ComputeDataType>()(ik) =
                                    b_thread_bufs_up[I0][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   ik))>{}];
                            });

                            using mfma_input_type =
                                typename vector_type<ComputeDataType,
                                                     xdlops_gemm.K1PerXdlops>::type;

                            xdlops_gemm.template Run<>(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{}));
                            xdlops_gemm.template Run<>(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec_up.template AsType<mfma_input_type>(),
                                c_thread_buf_per_scale_up.GetVectorTypeReference(Number<0>{}));
                        });
                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        static_for<0, xdlops_gemm.GetRegSizePerXdlops() / 2, 1>{}([&](auto t) {
                            using pk_fma_type = typename vector_type<AccDataType, 2>::type;

                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                .template AsType<pk_fma_type>()(t) = __builtin_elementwise_fma(
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                    .template AsType<pk_fma_type>()[t],
                                c_scale_thread_vec.template AsType<pk_fma_type>()[Number<0>{}],
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                    .template AsType<pk_fma_type>()[t]);
                            c_thread_buf_up.GetVectorTypeReference(Number<c_offset>{})
                                .template AsType<pk_fma_type>()(t) = __builtin_elementwise_fma(
                                c_thread_buf_per_scale_up.GetVectorTypeReference(Number<0>{})
                                    .template AsType<pk_fma_type>()[t],
                                c_scale_thread_vec_up.template AsType<pk_fma_type>()[Number<0>{}],
                                c_thread_buf_up.GetVectorTypeReference(Number<c_offset>{})
                                    .template AsType<pk_fma_type>()[t]);
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
                        c_scale_thread_buf_up(Number<c_offset>{}) =
                            a_scale_thread_buf[Number<a_offset>{}] *
                            b_scale_thread_buf_up[Number<b_offset>{}];
                    });
                });
            });

            block_sync_lds();

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, KGroup, 1>{}([&](auto kg0) {
                        a_thread_copy_.Run(
                            a_block_desc_m0_m1_m2_k0_k1_k2,
                            make_tuple(m0, I0, I0, Number<k0 * KGroup + kg0>{}, I0, I0),
                            a_block_buf,
                            a_thread_desc_,
                            make_tuple(m0, I0, I0, k0, I0, Number<kg0 * KPack / KGroup>{}),
                            a_thread_buf);
                    });
                });
            });

            // __builtin_amdgcn_sched_barrier(0);

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, num_scale_k_block, 1>{}([&](auto kscale0) {
                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                            c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                .template AsType<AccDataType>()(Number<t>{}) = 0;
                            c_thread_buf_per_scale_up.GetVectorTypeReference(Number<0>{})
                                .template AsType<AccDataType>()(Number<t>{}) = 0;
                        });
                        vector_type<AccDataType, 2> c_scale_thread_vec;
                        vector_type<AccDataType, 2> c_scale_thread_vec_up;
                        constexpr index_t cscale_offset = CScaleThreadDesc{}.CalculateOffset(
                            make_tuple(kscale0, m0, n0 * num_scale_n_block / NRepeat));

                        c_scale_thread_vec.template AsType<AccDataType>()(Number<0>{}) =
                            c_scale_thread_buf[Number<cscale_offset>{}];
                        c_scale_thread_vec.template AsType<AccDataType>()(Number<1>{}) =
                            c_scale_thread_buf[Number<cscale_offset>{}];
                        c_scale_thread_vec_up.template AsType<AccDataType>()(Number<0>{}) =
                            c_scale_thread_buf_up[Number<cscale_offset>{}];
                        c_scale_thread_vec_up.template AsType<AccDataType>()(Number<1>{}) =
                            c_scale_thread_buf_up[Number<cscale_offset>{}];

                        static_for<0, KRepeat / num_scale_k_block, 1>{}([&](auto k0) {
                            vector_type<ComputeDataType, KPack> a_thread_vec;
                            vector_type<ComputeDataType, KPack> b_thread_vec;
                            vector_type<ComputeDataType, KPack> b_thread_vec_up;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0,
                                                   I0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   I0,
                                                   ik))>{}];
                                b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                    b_thread_bufs[I1][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   ik))>{}];
                                b_thread_vec_up.template AsType<ComputeDataType>()(ik) =
                                    b_thread_bufs_up[I1][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   ik))>{}];
                            });

                            using mfma_input_type =
                                typename vector_type<ComputeDataType,
                                                     xdlops_gemm.K1PerXdlops>::type;

                            xdlops_gemm.template Run<>(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{}));
                            xdlops_gemm.template Run<>(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec_up.template AsType<mfma_input_type>(),
                                c_thread_buf_per_scale_up.GetVectorTypeReference(Number<0>{}));
                        });
                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        static_for<0, xdlops_gemm.GetRegSizePerXdlops() / 2, 1>{}([&](auto t) {
                            using pk_fma_type = typename vector_type<AccDataType, 2>::type;

                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                .template AsType<pk_fma_type>()(t) = __builtin_elementwise_fma(
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                    .template AsType<pk_fma_type>()[t],
                                c_scale_thread_vec.template AsType<pk_fma_type>()[Number<0>{}],
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                    .template AsType<pk_fma_type>()[t]);
                            c_thread_buf_up.GetVectorTypeReference(Number<c_offset>{})
                                .template AsType<pk_fma_type>()(t) = __builtin_elementwise_fma(
                                c_thread_buf_per_scale_up.GetVectorTypeReference(Number<0>{})
                                    .template AsType<pk_fma_type>()[t],
                                c_scale_thread_vec_up.template AsType<pk_fma_type>()[Number<0>{}],
                                c_thread_buf_up.GetVectorTypeReference(Number<c_offset>{})
                                    .template AsType<pk_fma_type>()[t]);
                        });
                    });
                });
            });
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, num_scale_k_block, 1>{}([&](auto kscale0) {
                        static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                            c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                .template AsType<AccDataType>()(Number<t>{}) = 0;
                            c_thread_buf_per_scale_up.GetVectorTypeReference(Number<0>{})
                                .template AsType<AccDataType>()(Number<t>{}) = 0;
                        });
                        vector_type<AccDataType, 2> c_scale_thread_vec;
                        vector_type<AccDataType, 2> c_scale_thread_vec_up;
                        constexpr index_t cscale_offset = CScaleThreadDesc{}.CalculateOffset(
                            make_tuple(kscale0, m0, n0 * num_scale_n_block / NRepeat));

                        c_scale_thread_vec.template AsType<AccDataType>()(Number<0>{}) =
                            c_scale_thread_buf[Number<cscale_offset>{}];
                        c_scale_thread_vec.template AsType<AccDataType>()(Number<1>{}) =
                            c_scale_thread_buf[Number<cscale_offset>{}];
                        c_scale_thread_vec_up.template AsType<AccDataType>()(Number<0>{}) =
                            c_scale_thread_buf_up[Number<cscale_offset>{}];
                        c_scale_thread_vec_up.template AsType<AccDataType>()(Number<1>{}) =
                            c_scale_thread_buf_up[Number<cscale_offset>{}];

                        static_for<0, KRepeat / num_scale_k_block, 1>{}([&](auto k0) {
                            vector_type<ComputeDataType, KPack> a_thread_vec;
                            vector_type<ComputeDataType, KPack> b_thread_vec;
                            vector_type<ComputeDataType, KPack> b_thread_vec_up;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                    a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0,
                                                   I0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   I0,
                                                   ik))>{}];
                                b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                    b_thread_bufs[I0][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   ik))>{}];
                                b_thread_vec_up.template AsType<ComputeDataType>()(ik) =
                                    b_thread_bufs_up[I0][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0,
                                                   I0,
                                                   kscale0 * KRepeat / num_scale_k_block + k0,
                                                   ik))>{}];
                            });

                            using mfma_input_type =
                                typename vector_type<ComputeDataType,
                                                     xdlops_gemm.K1PerXdlops>::type;

                            xdlops_gemm.template Run<>(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec.template AsType<mfma_input_type>(),
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{}));
                            xdlops_gemm.template Run<>(
                                a_thread_vec.template AsType<mfma_input_type>(),
                                b_thread_vec_up.template AsType<mfma_input_type>(),
                                c_thread_buf_per_scale_up.GetVectorTypeReference(Number<0>{}));
                        });
                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        static_for<0, xdlops_gemm.GetRegSizePerXdlops() / 2, 1>{}([&](auto t) {
                            using pk_fma_type = typename vector_type<AccDataType, 2>::type;

                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                .template AsType<pk_fma_type>()(t) = __builtin_elementwise_fma(
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                    .template AsType<pk_fma_type>()[t],
                                c_scale_thread_vec.template AsType<pk_fma_type>()[Number<0>{}],
                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                    .template AsType<pk_fma_type>()[t]);
                            c_thread_buf_up.GetVectorTypeReference(Number<c_offset>{})
                                .template AsType<pk_fma_type>()(t) = __builtin_elementwise_fma(
                                c_thread_buf_per_scale_up.GetVectorTypeReference(Number<0>{})
                                    .template AsType<pk_fma_type>()[t],
                                c_scale_thread_vec_up.template AsType<pk_fma_type>()[Number<0>{}],
                                c_thread_buf_up.GetVectorTypeReference(Number<c_offset>{})
                                    .template AsType<pk_fma_type>()[t]);
                        });
                    });
                });
            });
        }
    }

    protected:
    // MRepeat MWave MLane KRepeat KLane KPack
    // KRepeat -> MRepeat-> Mwave->KLane->MLane->KPack
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, I1, I1, Number<KRepeat>{}, I1, Number<KPack>{}));

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
