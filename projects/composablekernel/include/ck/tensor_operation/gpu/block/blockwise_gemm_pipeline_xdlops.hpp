// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/xdlops_gemm.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

// Double LDS buffer
// Prefetech 2 stage
// Local prefetch 1 stage

namespace ck {

template <index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t ABufferLoadWidth,
          index_t BBufferLoadWidth,
          index_t ALDSWriteWidth,
          index_t BLDSWriteWidth,
          index_t ALDSReadWidth,
          index_t BLDSReadWidth,
          index_t MRepeat,
          index_t NRepeat,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t KPerXDL>
struct BlockwiseGemmXdlops_pipeline_hotloop_inst
{
    static constexpr index_t WaveNumM = MPerBlock / (MRepeat * MPerXDL);
    static constexpr index_t WaveNumN = NPerBlock / (NRepeat * NPerXDL);
    static constexpr index_t WaveSize = BlockSize / (WaveNumM * WaveNumN);

    static constexpr index_t A_Buffer_Load_Inst_Num =
        MPerBlock * KPerBlock / (BlockSize * ABufferLoadWidth);
    static constexpr index_t B_Buffer_Load_Inst_Num =
        NPerBlock * KPerBlock / (BlockSize * BBufferLoadWidth);

    static constexpr index_t A_LDS_Write_Inst_Num =
        MPerBlock * KPerBlock / (BlockSize * ALDSWriteWidth);
    static constexpr index_t B_LDS_Write_Inst_Num =
        NPerBlock * KPerBlock / (BlockSize * BLDSWriteWidth);

    static constexpr index_t A_LDS_Read_Inst_Num =
        WaveNumN * MPerBlock * KPerBlock / (BlockSize * ALDSReadWidth);
    static constexpr index_t B_LDS_Read_Inst_Num =
        WaveNumM * MPerBlock * KPerBlock / (BlockSize * BLDSReadWidth);

    static constexpr index_t C_MFMA_Inst_Num =
        MPerBlock * NPerBlock * KPerBlock / (BlockSize / WaveSize) / (MPerXDL * NPerXDL * KPerXDL);

    static constexpr auto Print()
    {
        printf(" Blk/Wave Size: %d, %d, M/N/K PerBlk: %d, %d, %d, M/N/K PerXdl: %d, %d, %d\n",
               BlockSize,
               WaveSize,
               MPerBlock,
               NPerBlock,
               KPerBlock,
               MPerXDL,
               NPerXDL,
               KPerXDL);

        printf(" A/B buffer load inst: %d, %d\n A/B LDS write inst: %d, %d\n A/B LDS read inst: "
               "%d, %d\n C MFMA inst: %d\n",
               A_Buffer_Load_Inst_Num,
               B_Buffer_Load_Inst_Num,
               A_LDS_Write_Inst_Num,
               B_LDS_Write_Inst_Num,
               A_LDS_Read_Inst_Num,
               B_LDS_Read_Inst_Num,
               C_MFMA_Inst_Num);
    }
};

template <
    index_t BlockSize,
    typename FloatAB,
    typename FloatAcc,
    typename ATileDesc,
    typename BTileDesc,
    typename AMmaTileDesc,
    typename BMmaTileDesc,
    index_t MPerBlock,
    index_t NPerBlock,
    index_t KPerBlock,
    index_t MPerXDL,
    index_t NPerXDL,
    index_t MRepeat,
    index_t NRepeat,
    index_t KPack,
    bool TransposeC = false,
    index_t AMmaKStride =
        KPack * XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, FloatAB, TransposeC>{}.K0PerXdlops,
    index_t BMmaKStride =
        KPack * XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, FloatAB, TransposeC>{}.K0PerXdlops>
struct BlockwiseGemmXdlops_pipeline_v4
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t MWaves   = MPerBlock / (MRepeat * MPerXDL);
    static constexpr index_t NWaves   = NPerBlock / (NRepeat * NPerXDL);
    static constexpr index_t WaveSize = BlockSize / MWaves / NWaves;

    static constexpr index_t A_K0 = ATileDesc{}.GetLength(I0);
    static constexpr index_t B_K0 = BTileDesc{}.GetLength(I0);
    static constexpr index_t A_K1 = ATileDesc{}.GetLength(I2);
    static constexpr index_t B_K1 = BTileDesc{}.GetLength(I2);

    static constexpr auto xdlops_gemm =
        XdlopsGemm<FloatAB, MPerXDL, NPerXDL, KPack, FloatAB, TransposeC>{};

    static constexpr index_t KPerThread = KPerBlock / xdlops_gemm.K0PerXdlops;
    static constexpr index_t KRepeat    = KPerThread / KPack;

    using HotLoopInstList = BlockwiseGemmXdlops_pipeline_hotloop_inst<BlockSize,
                                                                      MPerBlock,
                                                                      NPerBlock,
                                                                      KPerBlock,
                                                                      A_K1,
                                                                      B_K1,
                                                                      A_K1,
                                                                      B_K1,
                                                                      KPack,
                                                                      KPack,
                                                                      MRepeat,
                                                                      NRepeat,
                                                                      MPerXDL,
                                                                      NPerXDL,
                                                                      xdlops_gemm.KPerXdlops>;

    static_assert(KPerThread % KPack == 0,
                  "Wrong KPack setting; try increasing KPerThread or decreasing KPack");

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              FloatAcc,
                              MRepeat * NRepeat,
                              xdlops_gemm.GetRegSizePerXdlops(),
                              true>
        c_thread_buf_;

    __host__ __device__ constexpr auto& GetCThreadBuffer() { return c_thread_buf_; }

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = ThisThreadBlock::GetThreadId();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MWaves, NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

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

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static auto
    CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk(xdlops_i, blk_i);

        constexpr auto mrepeat_mwave_mperxdl_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerXDL))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto nrepeat_nwave_nperxdl_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        const index_t c_thread_m = mrepeat_mwave_mperxdl_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_idx[I0]))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperxdl_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_idx[I1]))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    template <index_t m0, index_t n0, index_t xdlops_i, index_t blk_i>
    __device__ static auto
    CalculateCThreadOriginDataIndex8D(Number<m0>, Number<n0>, Number<xdlops_i>, Number<blk_i>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = xdlops_gemm.GetBeginOfThreadBlk4D(xdlops_i, blk_i);

        return make_tuple(
            m0, n0, waveId_m, waveId_n, blk_idx[I0], blk_idx[I1], blk_idx[I2], blk_idx[I3]);
    }

    using Tuple4 = decltype(CalculateAThreadOriginDataIndex());

    __host__ __device__
    BlockwiseGemmXdlops_pipeline_v4(Tuple4 a_origin = CalculateAThreadOriginDataIndex(),
                                    Tuple4 b_origin = CalculateBThreadOriginDataIndex())
        : a_thread_copy_(a_origin), b_thread_copy_(b_origin)
    {
#if defined(__HIP_DEVICE_COMPILE__)
        static_assert(AMmaTileDesc::IsKnownAtCompileTime() && BMmaTileDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerXDL * MRepeat) == 0 && NPerBlock % (NPerXDL * NRepeat) == 0,
                      "wrong!");
#endif
        // HotLoopInstList::Print();
    }

    // transposed XDL output supporting C_xdl' = B_xdl' * A_xdl'
    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, N, M0, M1, M2));
    }

    // XDL output supporting C_xdl = A_xdl * B_xdl
    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    __host__ __device__ static constexpr auto GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_m0_m1_m2_n_tblk_lens = xdlops_gemm.GetCM0M1M2NThreadBlkLengths();

        constexpr auto M0 = c_m0_m1_m2_n_tblk_lens[I0];
        constexpr auto M1 = c_m0_m1_m2_n_tblk_lens[I1];
        constexpr auto M2 = c_m0_m1_m2_n_tblk_lens[I2];
        constexpr auto N  = c_m0_m1_m2_n_tblk_lens[I3];

        return make_naive_tensor_descriptor_packed(
            make_tuple(I1, Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M0, M1, M2, N));
    }

    // transposed XDL output supporting C_xdl' = B_xdl' * A_xdl'
    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    // XDL output supporting C_xdl = A_xdl * B_xdl
    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_block_desc_m0_n0_m1_n1_m2_n2);
    }

    __host__ __device__ static constexpr auto GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2()
    {
        constexpr auto c_block_desc_g_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerXDL>{},
                                                           Number<NPerXDL>{}));

        return xdlops_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
            c_block_desc_g_m0_n0_m1_n1_m2_n2);
    }

    template <typename CGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto c_grid_desc_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(M / (MWaves * MPerXDL), MWaves, MPerXDL)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerXDL), NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

        return xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m0_n0_m1_n1_m2_n2);
    }

    template <typename CGridDesc_G_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc_G_M_N& c_grid_desc_g_m_n)
    {
        const auto G = c_grid_desc_g_m_n.GetLength(I0);
        const auto M = c_grid_desc_g_m_n.GetLength(I1);
        const auto N = c_grid_desc_g_m_n.GetLength(I2);

        const auto c_grid_desc_g_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_g_m_n,
            make_tuple(make_pass_through_transform(G),
                       make_unmerge_transform(make_tuple(M / (MWaves * MPerXDL), MWaves, MPerXDL)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerXDL), NWaves, NPerXDL))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3, 5>{}, Sequence<2, 4, 6>{}));

        return xdlops_gemm.MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
            c_grid_desc_g_m0_n0_m1_n1_m2_n2);
    }

    __device__ static constexpr auto HotLoopScheduler()
    {
        // schedule
        constexpr auto num_ds_read_inst =
            HotLoopInstList::A_LDS_Read_Inst_Num + HotLoopInstList::B_LDS_Read_Inst_Num;
        constexpr auto num_ds_write_inst =
            HotLoopInstList::A_LDS_Write_Inst_Num + HotLoopInstList::B_LDS_Write_Inst_Num;
        ;
        constexpr auto num_buffer_load_inst =
            HotLoopInstList::A_Buffer_Load_Inst_Num + HotLoopInstList::B_Buffer_Load_Inst_Num;
        ;
        constexpr auto num_mfma_inst = HotLoopInstList::C_MFMA_Inst_Num;

        constexpr auto num_issue = num_buffer_load_inst;

        static_for<0, num_issue, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(
                0x100, num_ds_read_inst / num_buffer_load_inst, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);      // MFMA
            __builtin_amdgcn_sched_group_barrier(
                0x200, num_ds_write_inst / num_buffer_load_inst, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);       // MFMA
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);       // VMEM read
            __builtin_amdgcn_sched_group_barrier(
                0x008, num_mfma_inst / num_buffer_load_inst - 3, 0); // MFMA
        });
    }

    template <index_t stage>
    __device__ static constexpr auto TailScheduler()
    {
    }

    template <>
    __device__ constexpr auto TailScheduler<1>()
    {
        // schedule
        constexpr auto num_ds_read_inst =
            HotLoopInstList::A_LDS_Read_Inst_Num + HotLoopInstList::B_LDS_Read_Inst_Num;
        constexpr auto num_ds_write_inst =
            HotLoopInstList::A_LDS_Write_Inst_Num + HotLoopInstList::B_LDS_Write_Inst_Num;
        ;
        constexpr auto num_mfma_inst = HotLoopInstList::C_MFMA_Inst_Num;

        constexpr auto num_issue = num_ds_write_inst;

        static_for<0, num_issue, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(
                0x100, num_ds_read_inst / num_ds_write_inst - 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(
                0x008, num_mfma_inst / num_ds_write_inst - 3, 0); // MFMA
        });
    }

    template <>
    __device__ constexpr auto TailScheduler<2>()
    {
        // schedule
        constexpr auto num_ds_read_inst =
            HotLoopInstList::A_LDS_Read_Inst_Num + HotLoopInstList::B_LDS_Read_Inst_Num;
        constexpr auto num_mfma_inst = HotLoopInstList::C_MFMA_Inst_Num;

        constexpr auto num_issue = num_ds_read_inst;

        static_for<0, num_issue, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(
                0x008, num_mfma_inst / num_ds_read_inst, 0); // MFMA
        });
    }

    static constexpr AMmaTileDesc a_block_desc_m0_m1_m2_k;
    static constexpr BMmaTileDesc b_block_desc_n0_n1_n2_k;

    template <bool HasMainLoop,
              index_t TailNum,
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
        __builtin_amdgcn_sched_barrier(0);
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            b_thread_desc_.GetElementSpaceSize());

        StaticallyIndexedArray<decltype(a_thread_buf), Number<2>{}> a_thread_bufs;
        StaticallyIndexedArray<decltype(b_thread_buf), Number<2>{}> b_thread_bufs;
        // Inst List:
        // ds_read_b128: 16
        // ds_write_b128: 8
        // buffer_load_dwordx4: 16
        // v_mfma: 0
        // -------------------------------------------------------------------------------------------

        // Global prefetch 1th, Fill Ping LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(I0));
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(I0));

        // Local prefetch 1th, Fill Ping Reg
        block_sync_lds();
        static_for<0, KRepeat, 1>{}([&](auto k) {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                   make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                   a_block_buf.At(I0),
                                   a_thread_desc_,
                                   make_tuple(m0, I0, k, I0),
                                   a_thread_bufs(I0));
                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                       make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                       b_block_buf.At(I0),
                                       b_thread_desc_,
                                       make_tuple(n0, I0, k, I0),
                                       b_thread_bufs(I0));
                });
            });
        });

        // Global prefetch 2th, Fill Pong LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(I1));
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(I1));

        // Global prefetch 3rd
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
                // -------------------------------------------------------------------------------------------
                using PingP1 = Number<0>;
                using PongP1 = Number<1>;
                // MFMA: Ping Reg
                // DS_WRITE: To Ping LDS
                // DS_READ: Pong LDS to Pong Reg
                block_sync_lds();

                static_for<0, KRepeat, 1>{}([&](auto k) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                           make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                           a_block_buf.At(PongP1{}),
                                           a_thread_desc_,
                                           make_tuple(m0, I0, k, I0),
                                           a_thread_bufs(PongP1{}));
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                               make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                               b_block_buf.At(PongP1{}),
                                               b_thread_desc_,
                                               make_tuple(n0, I0, k, I0),
                                               b_thread_bufs(PongP1{}));
                        });
                    });
                });

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(PingP1{}));
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(PingP1{}));

                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            vector_type<FloatAB, KPack> a_thread_vec;
                            vector_type<FloatAB, KPack> b_thread_vec;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<FloatAB>()(ik) =
                                    a_thread_bufs[PingP1{}][Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0, I0, k0, ik))>{}];
                                b_thread_vec.template AsType<FloatAB>()(ik) =
                                    b_thread_bufs[PingP1{}][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0, I0, k0, ik))>{}];
                            });

                            using mfma_input_type =
                                typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

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
                __builtin_amdgcn_sched_barrier(0);

                // -------------------------------------------------------------------------------------------
                using PingP2 = Number<1>;
                using PongP2 = Number<0>;
                // MFMA: Pong Reg
                // DS_WRITE: To Pong LDS
                // DS_READ: Ping LDS to Ping Reg
                block_sync_lds();

                static_for<0, KRepeat, 1>{}([&](auto k) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                           make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                           a_block_buf.At(PongP2{}),
                                           a_thread_desc_,
                                           make_tuple(m0, I0, k, I0),
                                           a_thread_bufs(PongP2{}));
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                               make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                               b_block_buf.At(PongP2{}),
                                               b_thread_desc_,
                                               make_tuple(n0, I0, k, I0),
                                               b_thread_bufs(PongP2{}));
                        });
                    });
                });

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(PingP2{}));
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(PingP2{}));

                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            vector_type<FloatAB, KPack> a_thread_vec;
                            vector_type<FloatAB, KPack> b_thread_vec;

                            static_for<0, KPack, 1>{}([&](auto ik) {
                                a_thread_vec.template AsType<FloatAB>()(ik) =
                                    a_thread_bufs[PingP2{}][Number<a_thread_desc_.CalculateOffset(
                                        make_tuple(m0, I0, k0, ik))>{}];
                                b_thread_vec.template AsType<FloatAB>()(ik) =
                                    b_thread_bufs[PingP2{}][Number<b_thread_desc_.CalculateOffset(
                                        make_tuple(n0, I0, k0, ik))>{}];
                            });

                            using mfma_input_type =
                                typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

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
                __builtin_amdgcn_sched_barrier(0);

                i += 2;
            } while(i < (num_loop - 3));
        }

        // tail
        if constexpr(TailNum == 3)
        {
            using PingP1 = Number<0>;
            using PongP1 = Number<1>;
            // MFMA: Ping Reg
            // DS_WRITE: To Ping LDS
            // DS_READ: Pong LDS to Pong Reg
            block_sync_lds();

            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf.At(PongP1{}),
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_bufs(PongP1{}));
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                           b_block_buf.At(PongP1{}),
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k, I0),
                                           b_thread_bufs(PongP1{}));
                    });
                });
            });

            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(PingP1{}));
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf.At(PingP1{}));

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_bufs[PingP1{}][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_bufs[PingP1{}][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            TailScheduler<1>();
            __builtin_amdgcn_sched_barrier(0);

            // -------------------------------------------------------------------------------------------
            using PingP2 = Number<1>;
            using PongP2 = Number<0>;
            // MFMA: Pong Reg
            // DS_WRITE: To Pong LDS
            // DS_READ: Ping LDS to Ping Reg
            block_sync_lds();

            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf.At(PongP2{}),
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_bufs(PongP2{}));
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                           b_block_buf.At(PongP2{}),
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k, I0),
                                           b_thread_bufs(PongP2{}));
                    });
                });
            });

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_bufs[PingP2{}][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_bufs[PingP2{}][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            TailScheduler<2>();
            __builtin_amdgcn_sched_barrier(0);

            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_bufs[PongP2{}][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_bufs[PongP2{}][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            // 64 v_mfma
            __builtin_amdgcn_sched_group_barrier(0x008, 64, 0); // MFMA
            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr(TailNum == 2)
        {
            using PingP1 = Number<0>;
            using PongP1 = Number<1>;
            // MFMA: Ping Reg
            // DS_WRITE: To Ping LDS
            // DS_READ: Pong LDS to Pong Reg
            block_sync_lds();

            static_for<0, KRepeat, 1>{}([&](auto k) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                       make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                       a_block_buf.At(PongP1{}),
                                       a_thread_desc_,
                                       make_tuple(m0, I0, k, I0),
                                       a_thread_bufs(PongP1{}));
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                           make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                           b_block_buf.At(PongP1{}),
                                           b_thread_desc_,
                                           make_tuple(n0, I0, k, I0),
                                           b_thread_bufs(PongP1{}));
                    });
                });
            });

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_bufs[PingP1{}][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_bufs[PingP1{}][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            TailScheduler<2>();
            __builtin_amdgcn_sched_barrier(0);

            // -------------------------------------------------------------------------------------------
            using PingP2 = Number<1>;
            // MFMA: Pong Reg
            // DS_WRITE: To Pong LDS
            // DS_READ: Ping LDS to Ping Reg

            static_for<0, KRepeat, 1>{}([&](auto k0) {
                static_for<0, MRepeat, 1>{}([&](auto m0) {
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        vector_type<FloatAB, KPack> a_thread_vec;
                        vector_type<FloatAB, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<FloatAB>()(ik) =
                                a_thread_bufs[PingP2{}][Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0, I0, k0, ik))>{}];
                            b_thread_vec.template AsType<FloatAB>()(ik) =
                                b_thread_bufs[PingP2{}][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<FloatAB, xdlops_gemm.K1PerXdlops>::type;

                        constexpr index_t c_offset =
                            c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                        xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type>(),
                                        b_thread_vec.template AsType<mfma_input_type>(),
                                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                    });
                });
            });

            // 64 v_mfma
            __builtin_amdgcn_sched_group_barrier(0x008, 64, 0); // MFMA
            __builtin_amdgcn_sched_barrier(0);
        }
    }

    protected:
    // M1, N1 as double buffer index
    // Read buffer + Compute buffer
    // A[M0, M1, M2, KPack]
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor(
        make_tuple(Number<MRepeat>{}, I1, Number<KRepeat>{}, Number<KPack>{}),
        make_tuple(
            Number<KPack>{}, Number<KRepeat * MRepeat * KPack>{}, Number<MRepeat * KPack>{}, I1));

    // B[N0, N1, N2, KPack]
    static constexpr auto b_thread_desc_ = make_naive_tensor_descriptor(
        make_tuple(Number<NRepeat>{}, I1, Number<KRepeat>{}, Number<KPack>{}),
        make_tuple(
            Number<KPack>{}, Number<KRepeat * NRepeat * KPack>{}, Number<NRepeat * KPack>{}, I1));

    // C[M, N, NumRegXdlops]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, xdlops_gemm.GetRegSizePerXdlops()));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(a_block_desc_m0_m1_m2_k),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, KPack>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         A_K1,
                                                         A_K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatAB,
                                                         FloatAB,
                                                         decltype(b_block_desc_n0_n1_n2_k),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, KPack>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         B_K1,
                                                         B_K1>;

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
