// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/dpp_gemm.hpp"

namespace ck {

/**
 * Blockwise GEMM that uses DPP instruction modifier to limit the amount of data loaded for each
 * thread by sharing the data between threads in a lanegroup.
 *
 * In every iteration, each wave calculates a C tile of size `MPerDpp` * `NPerDpp`, there are
 * `MRepeat` iterations for `M` dimension and `NRepeat` for `N` one.
 * In total, the algorithm runs using
 * `MPerBlock / (MRepeat * MPerDpp) * NPerBlock / (NRepeat * NPerDpp)` waves.
 */
template <index_t BlockSize,
          typename ABDataType,
          typename AccDataType,
          typename AK0MK1BlockDesc,
          typename BK0NK1BlockDesc,
          index_t MPerDpp,
          index_t NPerDpp,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack>
struct BlockwiseGemmDpp_ak0mak1_bk0nbk1_m0n0m1n1m2n2
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t MPerBlock = AK0MK1BlockDesc{}.GetLength(I1);
    static constexpr index_t NPerBlock = BK0NK1BlockDesc{}.GetLength(I1);
    static constexpr index_t KPerBlock =
        BK0NK1BlockDesc{}.GetLength(I0) * BK0NK1BlockDesc{}.GetLength(I2);

    static constexpr index_t MWaves   = MPerBlock / (MRepeat * MPerDpp);
    static constexpr index_t NWaves   = NPerBlock / (NRepeat * NPerDpp);
    static constexpr index_t WaveSize = BlockSize / MWaves / NWaves;

    static constexpr index_t A_K0 = AK0MK1BlockDesc{}.GetLength(I0);
    static constexpr index_t B_K0 = BK0NK1BlockDesc{}.GetLength(I0);
    static constexpr index_t A_K1 = AK0MK1BlockDesc{}.GetLength(I2);
    static constexpr index_t B_K1 = BK0NK1BlockDesc{}.GetLength(I2);

    static constexpr auto dpp_gemm = DppGemm<ABDataType, MPerDpp, NPerDpp, KPack>{};

    static constexpr index_t KPerThread = KPerBlock / dpp_gemm.K0PerDpp;

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              AccDataType,
                              MRepeat * NRepeat,
                              dpp_gemm.GetRegSizePerDpp(),
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

    __device__ static auto CalculateAThreadOriginDataIndex_M0_M1_M2_K()
    {
        const auto wave_idx    = GetWaveIdx();
        const auto waveId_m    = wave_idx[I0];
        const auto dpp_a_idx   = dpp_gemm.CalculateAThreadOriginDataIndex_K_M();
        const auto dpp_a_idx_k = dpp_a_idx[I0];
        const auto dpp_a_idx_m = dpp_a_idx[I1];
        return make_tuple(0, waveId_m, dpp_a_idx_m, KPerThread * dpp_a_idx_k);
    }

    __device__ static auto CalculateBThreadOriginDataIndex_N0_N1_N2_K()
    {
        const auto wave_idx    = GetWaveIdx();
        const auto waveId_n    = wave_idx[I1];
        const auto dpp_b_idx   = dpp_gemm.CalculateBThreadOriginDataIndex_K_N();
        const auto dpp_b_idx_k = dpp_b_idx[I0];
        const auto dpp_b_idx_n = dpp_b_idx[I1];
        return make_tuple(0, waveId_n, dpp_b_idx_n, KPerThread * dpp_b_idx_k);
    }

    template <index_t m0, index_t n0>
    __device__ static auto CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>)
    {
        const auto wave_idx = GetWaveIdx();
        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx      = dpp_gemm.GetBeginOfThreadBlk();
        const auto blk_m_offset = blk_idx[I0];
        const auto blk_n_offset = blk_idx[I1];

        constexpr auto mrepeat_mwave_MPerDpp_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerDpp))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto nrepeat_nwave_NPerDpp_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerDpp))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        const index_t c_thread_m = mrepeat_mwave_MPerDpp_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_m_offset))[I0];
        const index_t c_thread_n = nrepeat_nwave_NPerDpp_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_n_offset))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    __host__ __device__ BlockwiseGemmDpp_ak0mak1_bk0nbk1_m0n0m1n1m2n2()
    {
        static_assert(AK0MK1BlockDesc::IsKnownAtCompileTime() &&
                          BK0NK1BlockDesc::IsKnownAtCompileTime(),
                      "Wrong! Block descriptors should be known at the time of compilation.");

#if defined(__HIP_DEVICE_COMPILE__)
        // Host wave size can be different than the device one and this assert could fail for host,
        // but it does matter only for device.
        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");
#endif

        static_assert(MPerBlock % (MPerDpp * MRepeat) == 0,
                      "Invalid parameters. MPerBlock must be divisible by MPerDpp * MRepeat.");
        static_assert(NPerBlock % (NPerDpp * NRepeat) == 0,
                      "Invalid parameters. NPerBlock must be divisible by NPerDpp * NRepeat.");
    }

    __host__ __device__ static constexpr auto GetCThreadDescriptor_M0_N0_M1_N1_M2_N2()
    {
        constexpr auto c_m_n_tblk_lens = dpp_gemm.GetCMNThreadBlkLengths();
        constexpr auto M               = c_m_n_tblk_lens[I0];
        constexpr auto N               = c_m_n_tblk_lens[I1];

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M, N));
    }

    __host__ __device__ static constexpr auto GetCThreadDescriptor_G_M0_N0_M1_N1_M2_N2()
    {
        constexpr auto c_m_n_tblk_lens = dpp_gemm.GetCMNThreadBlkLengths();
        constexpr auto M               = c_m_n_tblk_lens[I0];
        constexpr auto N               = c_m_n_tblk_lens[I1];

        return make_naive_tensor_descriptor_packed(
            make_tuple(I1, Number<MRepeat>{}, Number<NRepeat>{}, I1, I1, M, N));
    }

    __host__ __device__ static constexpr auto GetCBlockDescriptor_M0_N0_M1_N1_M2_N2()
    {
        constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerDpp>{},
                                                           Number<NPerDpp>{}));

        return c_block_desc_m0_n0_m1_n1_m2_n2;
    }

    __host__ __device__ static constexpr auto GetCBlockDescriptor_G_M0_N0_M1_N1_M2_N2()
    {
        constexpr auto c_block_desc_g_m0_n0_m1_n1_m2_n2 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<MRepeat>{},
                                                           Number<NRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<NWaves>{},
                                                           Number<MPerDpp>{},
                                                           Number<NPerDpp>{}));
        return c_block_desc_g_m0_n0_m1_n1_m2_n2;
    }

    template <typename CGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_N2(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto c_grid_desc_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(M / (MWaves * MPerDpp), MWaves, MPerDpp)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerDpp), NWaves, NPerDpp))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

        return c_grid_desc_m0_n0_m1_n1_m2_n2;
    }

    template <typename CGridDesc_G_M_N>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_G_M0_N0_M1_N1_M2_N2(const CGridDesc_G_M_N& c_grid_desc_g_m_n)
    {
        const auto G = c_grid_desc_g_m_n.GetLength(I0);
        const auto M = c_grid_desc_g_m_n.GetLength(I1);
        const auto N = c_grid_desc_g_m_n.GetLength(I2);

        const auto c_grid_desc_g_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
            c_grid_desc_g_m_n,
            make_tuple(make_pass_through_transform(G),
                       make_unmerge_transform(make_tuple(M / (MWaves * MPerDpp), MWaves, MPerDpp)),
                       make_unmerge_transform(make_tuple(N / (NWaves * NPerDpp), NWaves, NPerDpp))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3, 5>{}, Sequence<2, 4, 6>{}));

        return c_grid_desc_g_m0_n0_m1_n1_m2_n2;
    }

    __host__ __device__ static constexpr auto MakeABlockDescriptor_M0_M1_M2_K()
    {
        return transform_tensor_descriptor(
            AK0MK1BlockDesc{},
            make_tuple(
                make_merge_transform_v3_division_mod(make_tuple(Number<A_K0>{}, Number<A_K1>{})),
                make_unmerge_transform(
                    make_tuple(Number<MRepeat>{}, Number<MWaves>{}, Number<MPerDpp>{}))),
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
                    make_tuple(Number<NRepeat>{}, Number<NWaves>{}, Number<NPerDpp>{}))),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
    }

    static constexpr auto a_block_desc_m0_m1_m2_k = MakeABlockDescriptor_M0_M1_M2_K();
    static constexpr auto b_block_desc_n0_n1_n2_k = MakeBBlockDescriptor_N0_N1_N2_K();

    template <typename ABlockBuffer, typename BBlockBuffer, typename CThreadBuffer>
    __device__ void Run(const ABlockBuffer& a_block_buf,
                        const BBlockBuffer& b_block_buf,
                        CThreadBuffer& c_thread_buf) const
    {
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ABDataType>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ABDataType>(
            b_thread_desc_.GetElementSpaceSize());

        static_for<0, MRepeat, 1>{}([&](auto m0) {
            // read A
            a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                               make_tuple(m0, I0, I0, I0),
                               a_block_buf,
                               a_thread_desc_,
                               make_tuple(I0, I0, I0, I0),
                               a_thread_buf);

            static_for<0, NRepeat, 1>{}([&](auto n0) {
                // read B
                b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                   make_tuple(n0, I0, I0, I0),
                                   b_block_buf,
                                   b_thread_desc_,
                                   make_tuple(I0, I0, I0, I0),
                                   b_thread_buf);

                static_for<0, KPerThread, KPack>{}([&](auto k) {
                    vector_type<ABDataType, KPack> a_thread_vec;
                    vector_type<ABDataType, KPack> b_thread_vec;

                    static_for<0, KPack, 1>{}([&](auto i) {
                        a_thread_vec.template AsType<ABDataType>()(i) = a_thread_buf
                            [Number<a_thread_desc_.CalculateOffset(make_tuple(0, 0, 0, k + i))>{}];
                        b_thread_vec.template AsType<ABDataType>()(i) = b_thread_buf
                            [Number<b_thread_desc_.CalculateOffset(make_tuple(0, 0, 0, k + i))>{}];
                    });

                    using dpp_input_type =
                        typename vector_type<ABDataType, dpp_gemm.K1PerDpp>::type;

                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                    dpp_gemm.Run(a_thread_vec.template AsType<dpp_input_type>(),
                                 b_thread_vec.template AsType<dpp_input_type>(),
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

    // C[M, N, NumRegDpp]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, dpp_gemm.GetRegSizePerDpp()));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<ABDataType,
                                                         ABDataType,
                                                         decltype(a_block_desc_m0_m1_m2_k),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, KPerThread>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         A_K1,
                                                         A_K1>;

    using BThreadCopy = ThreadwiseTensorSliceTransfer_v4<ABDataType,
                                                         ABDataType,
                                                         decltype(b_block_desc_n0_n1_n2_k),
                                                         decltype(b_thread_desc_),
                                                         Sequence<1, 1, 1, KPerThread>,
                                                         Sequence<0, 1, 2, 3>,
                                                         3,
                                                         B_K1,
                                                         B_K1>;

    AThreadCopy a_thread_copy_{CalculateAThreadOriginDataIndex_M0_M1_M2_K()};
    BThreadCopy b_thread_copy_{CalculateBThreadOriginDataIndex_N0_N1_N2_K()};
};

} // namespace ck
