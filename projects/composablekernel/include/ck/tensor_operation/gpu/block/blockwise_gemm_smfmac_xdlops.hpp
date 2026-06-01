// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/xdlops_gemm.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

template <index_t MNXdlPerWave, index_t MNWaves, index_t MNPerXdl, typename TileDesc_K0_MN_K1>
__host__ __device__ static constexpr auto
MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K(const TileDesc_K0_MN_K1&)
{
    constexpr index_t K0 = TileDesc_K0_MN_K1{}.GetLength(Number<0>{});
    constexpr index_t K1 = TileDesc_K0_MN_K1{}.GetLength(Number<2>{});

    return transform_tensor_descriptor(
        TileDesc_K0_MN_K1{},
        make_tuple(make_merge_transform_v3_division_mod(make_tuple(Number<K0>{}, Number<K1>{})),
                   make_unmerge_transform(
                       make_tuple(Number<MNXdlPerWave>{}, Number<MNWaves>{}, Number<MNPerXdl>{}))),
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
        make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
}

template <index_t BlockSize,
          typename FloatA,
          typename FloatB,
          typename FloatAcc,
          typename AK0MK1BlockDesc,
          typename BK0NK1BlockDesc,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack,
          typename ComputeTypeA = FloatA,
          typename ComputeTypeB = FloatB>
struct BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t MWaves   = MPerBlock / (MRepeat * MPerXDL);
    static constexpr index_t NWaves   = NPerBlock / (NRepeat * NPerXDL);
    static constexpr index_t WaveSize = BlockSize / MWaves / NWaves;

    static constexpr index_t MPerBlock = AK0MK1BlockDesc{}.GetLength(I1);
    static constexpr index_t NPerBlock = BK0NK1BlockDesc{}.GetLength(I1);
    static constexpr index_t KPerBlock =
        BK0NK1BlockDesc{}.GetLength(I0) * BK0NK1BlockDesc{}.GetLength(I2);

    static constexpr index_t A_K0 = AK0MK1BlockDesc{}.GetLength(I0);
    static constexpr index_t B_K0 = BK0NK1BlockDesc{}.GetLength(I0);
    static constexpr index_t A_K1 = AK0MK1BlockDesc{}.GetLength(I2);
    static constexpr index_t B_K1 = BK0NK1BlockDesc{}.GetLength(I2);

    static constexpr auto xdlops_gemm =
        SparseXdlopsGemm<ComputeTypeA, MPerXDL, NPerXDL, KPack, ComputeTypeB>{};

    static constexpr index_t KPerThread = KPerBlock / xdlops_gemm.K0PerXdlops;

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
        const auto wave_idx     = GetWaveIdx();
        const auto waveId_m     = wave_idx[I0];
        const auto xdlops_a_idx = xdlops_gemm.CalculateAThreadOriginDataIndex();

        return make_tuple(0, waveId_m, xdlops_a_idx[I1], KPerThread * xdlops_a_idx[I0]);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto wave_idx     = GetWaveIdx();
        const auto waveId_n     = wave_idx[I1];
        const auto xdlops_b_idx = xdlops_gemm.CalculateBThreadOriginDataIndex();

        return make_tuple(0, waveId_n, xdlops_b_idx[I1], KPerThread * xdlops_b_idx[I0]);
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

        return make_tuple(Number<m0>{},
                          Number<n0>{},
                          waveId_m,
                          waveId_n,
                          blk_idx[I0],
                          blk_idx[I1],
                          blk_idx[I2],
                          blk_idx[I3]);
    }

    __host__ __device__ BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1()
    {
#if defined(__HIP_DEVICE_COMPILE__)
        static_assert(AK0MK1BlockDesc::IsKnownAtCompileTime() &&
                          BK0NK1BlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(MPerBlock % (MPerXDL * MRepeat) == 0,
                      "MPerBlock must be divisible by MPerXDL * MRepeat");
        static_assert(NPerBlock % (NPerXDL * NRepeat) == 0,
                      "NPerBlock must be divisible by NPerXDL * NRepeat");

        static_assert(
            KPack % (16 * sizeof(ComputeTypeA)) == 0,
            "KPack must be divisbile by number of elements processed in single smfmac instruction");
#endif
    }

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

    __host__ __device__ static constexpr auto MakeABlockDescriptor_M0_M1_M2_K()
    {
        return transform_tensor_descriptor(
            AK0MK1BlockDesc{},
            make_tuple(
                make_merge_transform_v3_division_mod(make_tuple(Number<A_K0>{}, Number<A_K1>{})),
                make_unmerge_transform(
                    make_tuple(Number<MRepeat>{}, Number<MWaves>{}, Number<MPerXDL>{}))),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
    }

    __host__ __device__ static constexpr auto MakeBBlockDescriptor_N0_N1_N2_K()
    {
        return transform_tensor_descriptor(
            BK0NK1BlockDesc{},
            make_tuple(
                make_merge_transform_v3_division_mod(make_tuple(Number<B_K0>{}, Number<B_K1>{})),
                make_unmerge_transform(
                    make_tuple(Number<NRepeat>{}, Number<NWaves>{}, Number<NPerXDL>{}))),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
    }

    static constexpr auto a_block_desc_m0_m1_m2_k = MakeABlockDescriptor_M0_M1_M2_K();
    static constexpr auto b_block_desc_n0_n1_n2_k = MakeBBlockDescriptor_N0_N1_N2_K();

    // Prepares data in a_thread_buf by squeezing values by ommiting zeros to adjust it to 2:4
    // structural sparsity. The indexes of non-zero elements are stored in idx_buf and used later in
    // smfmac instruction
    template <typename AThreadBuf, typename IdxBuf, int32_t num_elems>
    __device__ void SetIdxSqueezeA(AThreadBuf& a_thread_buf, IdxBuf& idx_buf)
    {
        static constexpr int32_t bit_clear_masks[4] = {0b11, 0b1100, 0b110000, 0b11000000};
        static constexpr int32_t processed_elems    = 16 / sizeof(ComputeTypeA);

        static_for<0, num_elems, processed_elems>{}([&](auto i) {
            constexpr int idx_reg_num  = i / (16 * sizeof(ComputeTypeA));
            constexpr int idx_reg_part = (i % 32) / processed_elems;

            vector_type<ComputeTypeA, processed_elems> a_thread_vec;
            static_for<0, processed_elems, 1>{}([&](auto j) {
                a_thread_vec.template AsType<ComputeTypeA>()(j) = a_thread_buf
                    [Number<a_thread_desc_.CalculateOffset(make_tuple(0, 0, 0, i + j))>{}];
            });

            uint8_t idx = 0b11101110; // set to last 2 elems for both 4-elems subgroups by default
            for(int j = 0; j < processed_elems; j += 4)
            {
                int32_t a_pos                 = idx_reg_part * processed_elems + j;
                int32_t nonzero_pos           = 0;
                ComputeTypeA nonzero_elems[2] = {a_thread_vec[j + 2], a_thread_vec[j + 3]};
                for(int k = 0; k < 3; k += 1)
                {
                    if(a_thread_vec[j + k] != 0.0f)
                    {
                        nonzero_elems[nonzero_pos] = a_thread_vec[j + k];
                        idx &= ~bit_clear_masks[j / 2 + nonzero_pos];
                        idx |= k << 2 * (j / 2 + nonzero_pos);
                        ++nonzero_pos;
                    }
                }
                a_thread_vec[j / 2]     = nonzero_elems[0];
                a_thread_vec[j / 2 + 1] = nonzero_elems[1];
            }
            IdxBuf[idx_reg_num].AsType<int8x4_t>()[Number<idx_reg_part>{}] = idx;

            static_for<0, processed_elems / 2, 1>{}([&](auto j) {
                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                    make_tuple(0, 0, 0, i / 2 + j))>{}] = a_thread_vec[j];
            });
        });
    }

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeA>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeTypeB>(
            b_thread_desc_.GetElementSpaceSize());
        static constexpr int32_t elems_per_idx = 16 * sizeof(ComputeTypeA);
        auto idx_buf = make_static_buffer<AddressSpaceEnum::Vgpr, int32_t>(
            (a_thread_desc_.GetElementSpaceSize() + elems_per_idx - 1) / elems_per_idx);

        static_for<0, MRepeat, 1>{}([&](auto m0) {
            // read A
            a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                               make_tuple(m0, I0, I0, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I0, I0, I0),
                               a_thread_buf);

            SetIdxSqueezeA(a_thread_buf, idx_buf, a_thread_desc_.GetElementSpaceSize());

            static_for<0, NRepeat, 1>{}([&](auto n0) {
                // read B
                b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                   make_tuple(n0, I0, I0, I0),
                                   b_block_buf,
                                   b_thread_desc_,
                                   make_tuple(I0, I0, I0, I0),
                                   b_thread_buf);

                static_for<0, KPerThread, KPack>{}([&](auto k) {
                    // a_thread_vec is smaller because it's structurally sparse 2:4
                    vector_type<ComputeTypeA, KPack / 2> a_thread_vec;
                    vector_type<ComputeTypeB, KPack> b_thread_vec;
                    vector_type<int32_t, KPack / elems_per_idx> idx_vec;

                    static_for<0, KPack / 2, 1>{}([&](auto i) {
                        a_thread_vec.template AsType<ComputeTypeA>()(i) =
                            a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                make_tuple(0, 0, 0, k / 2 + i))>{}];
                    });

                    static_for<0, KPack, 1>{}([&](auto i) {
                        b_thread_vec.template AsType<ComputeTypeB>()(2 * i) = b_thread_buf
                            [Number<b_thread_desc_.CalculateOffset(make_tuple(0, 0, 0, k + i))>{}];
                    });

                    static_for<0, KPack / elems_per_idx, 1>{}([&](auto i) {
                        idx_vec.template AsType<int32_t>()(i) = idx_buf[k / elems_per_idx + i];
                    });

                    // A is smaller because it's structurally sparse 2:4
                    using mfma_input_type_a =
                        typename vector_type<ComputeTypeA, xdlops_gemm.K1PerXdlops / 2>::type;
                    using mfma_input_type_b =
                        typename vector_type<ComputeTypeB, xdlops_gemm.K1PerXdlops>::type;
                    using mfma_input_type_idx = typename vector_type<int32_t, 1>::type;

                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                    xdlops_gemm.Run(a_thread_vec.template AsType<mfma_input_type_a>(),
                                    b_thread_vec.template AsType<mfma_input_type_b>(),
                                    idx_vec.template AsType<mfma_input_type_idx>(),
                                    c_thread_buf.GetVectorTypeReference(Number<c_offset>{}));
                });
            });
        });
    }

    protected:
    // A[M0, M1, M2, KPerThread]
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I1, Number<KPerThread>{}));

    // B[N0, N1, N2, KPerThread]
    static constexpr auto b_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1, I1, I1, Number<KPerThread>{}));

    // C[M, N, NumRegXdlops]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, xdlops_gemm.GetRegSizePerXdlops()));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatA,
                                                         ComputeTypeA,
                                                         decltype(a_block_desc_m0_m1_m2_k),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, KPerThread>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         A_K1,
                                                         A_K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<FloatB,
                                                         ComputeTypeB,
                                                         decltype(b_block_desc_n0_n1_n2_k),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, KPerThread>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         B_K1,
                                                         B_K1>;

    AThreadCopy a_thread_copy_{CalculateAThreadOriginDataIndex()};
    BThreadCopy b_thread_copy_{CalculateBThreadOriginDataIndex()};
};

} // namespace ck
