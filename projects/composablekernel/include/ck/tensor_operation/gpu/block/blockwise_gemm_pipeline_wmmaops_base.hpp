// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/wmma_gemm.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

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
          bool TransposeC = false>
struct BlockwiseGemmWmmaops_pipeline_base
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t WaveSize = 32;

    static constexpr index_t MWaves = MPerBlock / (MRepeat * MPerWmma);
    static constexpr index_t NWaves = NPerBlock / (NRepeat * NPerWmma);

#if defined(__gfx12__)
    static constexpr index_t A_KRow = 2;
    static constexpr index_t B_KRow = 2;
#else
    static constexpr index_t A_KRow = 1;
    static constexpr index_t B_KRow = 1;
#endif

    static constexpr auto wmma_gemm = WmmaGemm<ComputeTypeA,
                                               ComputeTypeB,
                                               AccDataType,
                                               MPerWmma,
                                               NPerWmma,
                                               KPack / KInner,
                                               TransposeC>{};

    static constexpr index_t KPerThread = wmma_gemm.wmma_instr.k_per_blk * KInner;
    static constexpr index_t A_K1       = ck::math::min(AWmmaTileDesc{}.GetLength(I6), KPerThread);
    static constexpr index_t B_K1       = ck::math::min(BWmmaTileDesc{}.GetLength(I6), KPerThread);

    static_assert(KPack % (A_K1 * A_KRow) == 0, "wrong!");
    static_assert(KPack % (B_K1 * B_KRow) == 0, "wrong!");
    static constexpr index_t KRepeat = KPerBlock / KPack;

    static constexpr auto WmmaK = Number<wmma_gemm.wmma_instr.k_per_wmma>{};

    using HotLoopInstList =
        ck::BlockwiseGemmWmmaops_pipeline_hotloop_inst<BlockSize,
                                                       MPerBlock,
                                                       NPerBlock,
                                                       KPerBlock,
                                                       ABlockTransferSrcScalarPerVector,
                                                       BBlockTransferSrcScalarPerVector,
                                                       A_K1,
                                                       B_K1,
                                                       A_K1,
                                                       B_K1,
                                                       MRepeat,
                                                       NRepeat,
                                                       MPerWmma,
                                                       NPerWmma,
                                                       wmma_gemm.wmma_instr.k_per_wmma>;

    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                              AccDataType,
                              MRepeat * NRepeat,
                              wmma_gemm.GetRegSizePerWmma(),
                              true>
        c_thread_buf_;

    struct Empty
    {
        __device__ Empty() {};
        template <index_t NBuffer>
        __device__ void GlobalLoad(bool cond)
        {
            ignore = NBuffer;
            ignore = cond;
        }
    };

    template <index_t ScaleSliceSizeMN,
              index_t ScaleSliceStrideMN,
              index_t ScaleSliceSizeK,
              index_t NumberOfBuffers,
              index_t RegSizePerWmma,
              typename GridDesc,
              typename ThreadCopy,
              typename GridBuffer,
              typename ThreadStaticBuffer,
              typename ThreadDesc>
    struct ABScale
    {
        __device__ ABScale(GridDesc scale_grid_desc_,
                           ThreadCopy scale_thread_copy_,
                           GridBuffer scale_grid_buf_)
            : scale_thread_copy(scale_thread_copy_),
              scale_grid_desc(scale_grid_desc_),
              scale_grid_buf(scale_grid_buf_) {};

        static constexpr index_t num_scale_k_block = ThreadDesc{}.GetLength(Number<1>{});
        static constexpr index_t num_scale_krepeat = KRepeat / num_scale_k_block;

        static constexpr index_t num_slice_mn      = ScaleSliceSizeMN;
        static constexpr index_t num_slice_k       = ScaleSliceSizeK;
        static constexpr index_t reg_size_per_wmma = RegSizePerWmma;

        static constexpr auto scale_thread_desc = ThreadDesc{};

        static constexpr auto scale_thread_copy_step =
            make_tuple(make_multi_index(ScaleSliceStrideMN, 0),
                       make_multi_index(-ScaleSliceSizeMN / RegSizePerWmma * ScaleSliceStrideMN, 0),
                       make_multi_index(-ScaleSliceSizeMN / RegSizePerWmma * ScaleSliceStrideMN,
                                        ScaleSliceSizeK));

        template <index_t NBuffer>
        __device__ void GlobalLoad(bool cond)
        {
            static_for<0, ScaleSliceSizeMN / RegSizePerWmma, 1>{}([&](auto m0) {
                scale_thread_copy.Run(scale_grid_desc,
                                      scale_grid_buf,
                                      scale_thread_desc,
                                      make_tuple(m0 * Number<RegSizePerWmma>{}, Number<0>{}),
                                      scale_thread_bufs(Number<NBuffer>{}));

                scale_thread_copy.MoveSrcSliceWindow(scale_grid_desc,
                                                     scale_thread_copy_step.At(Number<0>{}));
            });

            if(cond)
            {
                scale_thread_copy.MoveSrcSliceWindow(scale_grid_desc,
                                                     scale_thread_copy_step.At(Number<2>{}));
            }
            else
            {
                scale_thread_copy.MoveSrcSliceWindow(scale_grid_desc,
                                                     scale_thread_copy_step.At(Number<1>{}));
            }
        }

        ThreadCopy scale_thread_copy;
        GridDesc scale_grid_desc;
        GridBuffer scale_grid_buf;
        StaticallyIndexedArray<ThreadStaticBuffer, Number<NumberOfBuffers>{}> scale_thread_bufs;
    };

    template <typename AScaleStruct, typename BScaleStruct>
    struct CScale
    {
        __device__ CScale() {}

        static constexpr auto reg_size_per_wmma =
            ck::is_same_v<BScaleStruct, Empty> && ck::is_same_v<AScaleStruct, Empty>
                ? 1
                : wmma_gemm.GetRegSizePerWmma();
        static constexpr auto c_scale_thread_desc = make_naive_tensor_descriptor_packed(make_tuple(
            Number<ck::math::max(AScaleStruct::num_slice_k, BScaleStruct::num_slice_k)>{},
            Number<AScaleStruct::num_slice_mn>{},
            Number<BScaleStruct::num_slice_mn>{}));
        using CScaleThreadDesc                    = decltype(c_scale_thread_desc);
        static constexpr auto num_scale_k_block   = CScaleThreadDesc{}.GetLength(Number<0>{});
        static constexpr auto num_scale_m_block   = CScaleThreadDesc{}.GetLength(Number<1>{});
        static constexpr auto num_scale_n_block   = CScaleThreadDesc{}.GetLength(Number<2>{});
        using ThreadStaticBuffer = decltype(make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            c_scale_thread_desc.GetElementSpaceSize()));

        __device__ void Load(AScaleStruct& a_scale_struct, BScaleStruct& b_scale_struct)
        {
            using AScaleThreadDesc = decltype(AScaleStruct::scale_thread_desc);
            using BScaleThreadDesc = decltype(BScaleStruct::scale_thread_desc);

            static_for<0, num_scale_m_block, 1>{}([&](auto m0) {
                static_for<0, num_scale_n_block, 1>{}([&](auto n0) {
                    static_for<0, num_scale_k_block, 1>{}([&](auto k0) {
                        constexpr index_t c_offset =
                            CScaleThreadDesc{}.CalculateOffset(make_tuple(k0, m0, n0));
                        constexpr index_t a_offset =
                            AScaleThreadDesc{}.CalculateOffset(make_tuple(m0, k0));
                        constexpr index_t b_offset =
                            BScaleThreadDesc{}.CalculateOffset(make_tuple(n0, k0));

                        c_scale_thread_bufs(I0)(Number<c_offset>{}) =
                            a_scale_struct.scale_thread_bufs(I0)[Number<a_offset>{}] *
                            b_scale_struct.scale_thread_bufs(I0)[Number<b_offset>{}];
                    });
                });
            });
        }

        __device__ void Clear()
        {
            static_for<0, reg_size_per_wmma, 1>{}([&](auto t) {
                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                    .template AsType<AccDataType>()(Number<t>{}) = 0;
            });
        }

        template <index_t k_index, index_t m_index, index_t n_index, typename CThreadBuf>
        __device__ void UpdateCThreadBuf(CThreadBuf& c_thread_buf)
        {
            static_for<0, reg_size_per_wmma, 1>{}([&](auto t) {
                constexpr index_t c_offset =
                    c_thread_desc_.CalculateOffset(make_tuple(m_index, n_index, t));
                constexpr index_t cscale_offset = CScaleThreadDesc{}.CalculateOffset(make_tuple(
                    k_index,
                    (m_index * num_scale_m_block / MRepeat) % num_scale_m_block +
                        (Number<t / (reg_size_per_wmma / AScaleStruct::reg_size_per_wmma)>{}) %
                            AScaleStruct::reg_size_per_wmma,
                    (n_index * num_scale_n_block / NRepeat) % num_scale_n_block));
                c_thread_buf(Number<c_offset>{}) +=
                    c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                        .template AsType<AccDataType>()[Number<t>{}] *
                    type_convert<AccDataType>(c_scale_thread_bufs(I0)[Number<cscale_offset>{}]);
            });
        }

        StaticallyIndexedArray<ThreadStaticBuffer, Number<1>{}> c_scale_thread_bufs;
        StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, AccDataType, 1, reg_size_per_wmma, true>
            c_thread_buf_per_scale;
    };

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

        const auto wmma_a_idx = wmma_gemm.CalculateAThreadOriginDataIndex();

#if defined(__gfx12__)
        const auto wmma_krow = wmma_gemm.GetSubGroupId();
#else
        const auto wmma_krow = 0;
#endif

        return make_tuple(0, 0, 0, waveId_m, wmma_krow, wmma_a_idx, 0);
    }

    __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_n = wave_idx[I1];

        const auto wmma_b_idx = wmma_gemm.CalculateBThreadOriginDataIndex();

#if defined(__gfx12__)
        const auto wmma_krow = wmma_gemm.GetSubGroupId();
#else
        const auto wmma_krow = 0;
#endif

        return make_tuple(0, 0, 0, waveId_n, wmma_krow, wmma_b_idx, 0);
    }

    template <index_t m0, index_t n0>
    __device__ static auto CalculateCThreadOriginDataIndex(Number<m0>, Number<n0>)
    {
        const auto wave_idx = GetWaveIdx();

        const auto waveId_m = wave_idx[I0];
        const auto waveId_n = wave_idx[I1];

        const auto blk_idx = wmma_gemm.GetBeginOfThreadBlk();

        constexpr auto mrepeat_mwave_mperwmma_to_m_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(MRepeat, MWaves, MPerWmma))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto nrepeat_nwave_nperwmma_to_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(NRepeat, NWaves, NPerWmma))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        const index_t c_thread_m = mrepeat_mwave_mperwmma_to_m_adaptor.CalculateBottomIndex(
            make_tuple(m0, waveId_m, blk_idx[I0]))[I0];
        const index_t c_thread_n = nrepeat_nwave_nperwmma_to_n_adaptor.CalculateBottomIndex(
            make_tuple(n0, waveId_n, blk_idx[I1]))[I0];

        return make_tuple(c_thread_m, c_thread_n);
    }

    using Tuple7 = decltype(CalculateAThreadOriginDataIndex());

    /**
     * @brief Constructor for BlockwiseGemmWmmaops_pipeline_base.
     *
     * This constructor initializes the thread copy objects for matrices A and B.
     * It also performs several compile-time checks to ensure the correctness of the
     * matrix tile descriptors.
     *
     * @param a_origin The origin data index for matrix A.
     * @param b_origin The origin data index for matrix B.
     *
     * @note The constructor includes static assertions to ensure that:
     * - The matrix tile descriptors for A and B are known at compile-time.
     * - The number of threads in the thread block matches the product of MWaves, NWaves, and
     * WaveSize.
     * - The dimensions of the block are divisible by the product of the corresponding WMMA and
     * repeat dimensions.
     */
    __host__ __device__
    BlockwiseGemmWmmaops_pipeline_base(Tuple7 a_origin = CalculateAThreadOriginDataIndex(),
                                       Tuple7 b_origin = CalculateBThreadOriginDataIndex())
        : a_thread_copy_(a_origin), b_thread_copy_(b_origin)
    {
        static_assert(AWmmaTileDesc::IsKnownAtCompileTime() &&
                          BWmmaTileDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == MWaves * NWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize");

        static_assert(MPerBlock % (MPerWmma * MRepeat) == 0 &&
                          NPerBlock % (NPerWmma * NRepeat) == 0,
                      "wrong!");
    }

    // transposed WMMA output C' = B' * A'
    __host__ __device__ static constexpr auto
    GetCThreadDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs()
    {
        constexpr auto c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens =
            wmma_gemm.GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths();

        constexpr auto NAccVgprs = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I2];

        return make_naive_tensor_descriptor_packed(
            //        |MRepeat            |MWave |MSubGroup |NRepeat           |NWave
            //        |NThreadPerSubGroup |MAccVgprs
            make_tuple(Number<MRepeat>{}, I1, I1, Number<NRepeat>{}, I1, I1, NAccVgprs));
    }

    static constexpr auto MAccVgprs =
        wmma_gemm.GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths()[I2];

    __host__ __device__ static constexpr auto
    GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs()
    {
        constexpr auto c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens =
            wmma_gemm.GetCMSubGroupNThreadPerSubGroupMAccVgprsThreadBlkLengths();

        constexpr auto AccStride = c_msubgroup_nthreadpersubgroup_maccvgprs_tblk_lens[I3];
        return make_naive_tensor_descriptor(
            //        |MRepeat           |MWave |MSubGroup |NRepeat           |NWave
            //        |NThreadPerSubGroup |MAccVgprs
            make_tuple(Number<MRepeat>{}, I1, I1, Number<NRepeat>{}, I1, I1, MAccVgprs),
            make_tuple(Number<NRepeat>{} * MAccVgprs * AccStride,
                       Number<NRepeat>{} * MAccVgprs * AccStride,
                       Number<NRepeat>{} * MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       MAccVgprs * AccStride,
                       AccStride));
    }

    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs()
    {
        constexpr auto c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma =
            make_naive_tensor_descriptor_packed(make_tuple(Number<MRepeat>{},
                                                           Number<MWaves>{},
                                                           Number<MPerWmma>{},
                                                           Number<NRepeat>{},
                                                           Number<NWaves>{},
                                                           Number<NPerWmma>{}));

        return wmma_gemm
            .MakeCDesc_MBlockxRepeat_MWave_MSubGroup_NBlockxRepeat_NWave_NThreadPerSubGroup_MAccVgprs(
                c_block_desc_mrepeat_mwave_mperwmma_nrepeat_nwave_nperwmma);
    }

    // Describe how data allocated in thread copy src buffer
    // M0_M1_M2 = MRepeat_MWave_MPerWmma, N0_N1_N2 = NRepeat_NWave_NPerWmma
    static constexpr AWmmaTileDesc a_block_desc_k0_m0_m1_m2_k1;
    static constexpr BWmmaTileDesc b_block_desc_k0_n0_n1_n2_k1;

    protected:
    static constexpr auto a_thread_desc_ =
        make_naive_tensor_descriptor(make_tuple(Number<KPack / A_K1 / A_KRow>{},
                                                Number<MRepeat>{},
                                                Number<KRepeat>{},
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
                                                Number<KRepeat>{},
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

    // C[M, N, NumRegWmma]
    static constexpr auto c_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MRepeat>{}, Number<NRepeat>{}, wmma_gemm.GetRegSizePerWmma()));

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

    AThreadCopy a_thread_copy_;
    BThreadCopy b_thread_copy_;
};

} // namespace ck
