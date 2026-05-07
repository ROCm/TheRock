// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_selector.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

// Currently we do not have a elegant way to put single lds buffer & double lds buffer pipe in same
// kernel function Blockers:
// 1. Two separted declaration of __shared__ pointer is the key to make sure data access operate on
// two lds chunks.
// 2. Occupied __shared__ won't release until whole shader end, a.k.a AB and C may not use same lds
// buffer when we declare __shared__ inside blkgemmpipe
template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Even>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    // __attribute__((amdgpu_waves_per_eu(1, 1)))
    kernel_gemm_xdl_cshuffle_v3_b_preshuffle(typename GridwiseGemm::Argument karg)
{
#if defined(__gfx9__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        // Full K needed for matrix B
        const index_t Kt = karg.K;

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg);

        const index_t num_k_per_block = GridwiseGemm::CalculateBK0Shuffled(karg.K);
        const index_t k_id            = blockIdx.z * num_k_per_block;

        GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            karg.p_a_grid + splitk_batch_offset.a_k_split_offset,
            karg.p_b_grid,
            karg.p_c_grid + splitk_batch_offset.c_reduce_offset,
            p_shared,
            karg,
            k_id,
            Kt);
    }
#else
    ignore = karg;
#endif // end of if (defined(__gfx9__))
}

template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Even>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    // __attribute__((amdgpu_waves_per_eu(1, 1)))
    kernel_gemm_xdl_cshuffle_v3_b_preshuffle_2lds(typename GridwiseGemm::Argument karg)
{
#if defined(__gfx9__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        // Pass two lds pointer is the key to tell compiler that ds_read/write
        // operate on different lds chunk at same time without order dependecy
        __shared__ char p_shared_0[GridwiseGemm::GetSharedMemoryNumberOfByte()];
        __shared__ char p_shared_1[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        // Full K needed for matrix B
        const index_t Kt = karg.K;

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg);

        const index_t num_k_per_block = GridwiseGemm::CalculateBK0Shuffled(karg.K);
        const index_t k_id            = blockIdx.z * num_k_per_block;

        GridwiseGemm::template Run_2Lds<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            karg.p_a_grid + splitk_batch_offset.a_k_split_offset,
            karg.p_b_grid,
            karg.p_c_grid + splitk_batch_offset.c_reduce_offset,
            p_shared_0,
            p_shared_1,
            karg,
            k_id,
            Kt);
    }
#else
    ignore = karg;
#endif // end of if (defined(__gfx9__))
}

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct GridwiseGemm_xdl_cshuffle_v3_b_preshuffle
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // K1 should be Number<...>
    static constexpr auto AK0Number = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0Number = Number<KPerBlock / BK1Value>{};
    static constexpr auto AK1Number = Number<AK1Value>{};
    static constexpr auto BK1Number = Number<BK1Value>{};

    // Use singal rate mfma instruction for this special case A (f8_t) * B (pk_i4_t)
    // See example gemm_xdl_fp8_pk_i4_bpreshuffle_v3
    // TODO: explore optimization opportunity by using new mfma instructions on gfx950
    static constexpr auto lcm_AK1_BK1         = math::lcm(AK1Number, BK1Number);
    static constexpr bool is_single_rate_mfma = true;
    static constexpr auto is_scale_mfma       = false;
    static constexpr auto mfma                = MfmaSelector<ComputeTypeA,
                                                             MPerXdl,
                                                             NPerXdl,
                                                             ComputeTypeA,
                                                             is_single_rate_mfma,
                                                             is_scale_mfma>{};
    static constexpr index_t KPack = math::max(lcm_AK1_BK1, mfma.selected_mfma.k_per_blk);
    static constexpr index_t KLane = mfma.GetKPerXdlops() / mfma.GetK1PerXdlops();

    static constexpr index_t KRepeat = KPerBlock / KLane / KPack;
    static constexpr index_t NLane   = NPerXdl;
    static constexpr index_t NWave   = NPerBlock / NPerXdl / NXdlPerWave;

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t APackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<ADataType>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    static constexpr index_t BPackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<BDataType>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    __host__ static auto CalculateGridSize(index_t M, index_t N, index_t KBatch)
    {
        return std::make_tuple(Block2CTileMap::CalculateGridSize(M, N), 1, KBatch);
    }

    __host__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ __device__ static auto CalculateBN0Shuffled(index_t N)
    {
        return math::integer_divide_ceil(N, NLane);
    }

    __host__ __device__ static auto CalculateBK0Shuffled(index_t K)
    {
        return math::integer_divide_ceil(K, KLane * KPack);
    }

    __host__ static auto CalculateKPadded(index_t K)
    {
        return math::integer_divide_ceil(K, KPerBlock) * KPerBlock;
    }

    __host__ static auto CalculateAK0Padded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / AK1Value);
    }

    __host__ static auto CalculateBK0Padded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / BK1Value);
    }

    __host__ static auto CalculateKPadded(index_t K, index_t K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * KPerBlock;
    }

    __host__ static auto CalculateKRead(index_t K, index_t K_Batch = 1)
    {
        constexpr auto KReadVec = math::lcm(AK1Number, BK1Number);
        auto K_t                = K_Batch * KReadVec;
        return (K + K_t - 1) / K_t * KReadVec;
    }

    __host__ static auto CalculateMBlock(index_t M)
    {
        return math::integer_divide_ceil(M, MPerBlock);
    }

    __host__ static auto CalculateNBlock(index_t N)
    {
        return math::integer_divide_ceil(N, NPerBlock);
    }

    template <index_t MNXdlPerWave, index_t MNWaves, index_t MNPerXdl, typename TileDesc_K0_MN_K1>
    __host__ __device__ static constexpr auto MakeGemmMmaTileDescriptor(const TileDesc_K0_MN_K1&)
    {
        constexpr index_t K0 = TileDesc_K0_MN_K1{}.GetLength(Number<0>{});
        constexpr index_t K1 = TileDesc_K0_MN_K1{}.GetLength(Number<2>{});

        return transform_tensor_descriptor(
            TileDesc_K0_MN_K1{},
            make_tuple(make_merge_transform_v3_division_mod(make_tuple(Number<K0>{}, Number<K1>{})),
                       make_unmerge_transform(make_tuple(
                           Number<MNXdlPerWave>{}, Number<MNWaves>{}, Number<MNPerXdl>{}))),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
    }

    __host__ __device__ static auto MakeAGridDescriptor_AK0_M_AK1(
        index_t M, index_t MPad, index_t K, index_t KPad, index_t StrideA, index_t AK0)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        if constexpr(GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both M and K
            const auto a_grid_desc_m_k =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_right_pad_transform(M, MPad - M),
                                                       make_right_pad_transform(K, KPad - K)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_pass_through_transform(MPad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad M, but not K
            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_right_pad_transform(M, MPad - M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad K, but not M
            const auto a_grid_desc_m_k = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else
        {
            // not pad M or K
            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
    }

    __host__ __device__ static auto MakeBGridDescriptor_Preshuffled(index_t N0, index_t K0)
    {
        constexpr index_t MWave           = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t WaveSize        = BlockSize / (MWave * NWave);
        constexpr index_t NkSwizzleNumber = Number<WaveSize * KPack>{};
        return make_naive_tensor_descriptor(
            make_tuple(N0 / NWave, NWave, K0, NkSwizzleNumber),
            make_tuple(NWave * K0 * NkSwizzleNumber, K0 * NkSwizzleNumber, NkSwizzleNumber, I1));
    }

    __host__ __device__ static auto MakeBGridDescriptor_BK0_N_BK1(
        index_t K, index_t KPad, index_t N, index_t NPad, index_t StrideB, index_t BK0)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(StrideB, I1));
            }
        }();

        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        static_assert(!(is_same_v<remove_cvref_t<ADataType>, pk_i4_t> &&
                        GemmSpec != GemmSpecialization::Default),
                      "pk_i4_t does not support padding");

        if constexpr(GemmSpec == GemmSpecialization::NKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both N and K
            const auto b_grid_desc_n_k =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_right_pad_transform(N, NPad - N),
                                                       make_right_pad_transform(K, KPad - K)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_pass_through_transform(NPad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad N, but not K
            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad K, but not N
            const auto b_grid_desc_n_k = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_pass_through_transform(N), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else
        {
            if constexpr(!PermuteB)
            {
                // not pad N or K
                const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                    b_grid_desc_nraw_kraw,
                    make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                               make_pass_through_transform(N)),
                    make_tuple(Sequence<1>{}, Sequence<0>{}),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

                return b_grid_desc_bk0_n_bk1;
            }
            else
            {
                // Pre-shuffled Weight
                // BGlobal[K / KPerBlock, N, KPerBlock / K1, K1] -> BTile[K / K1, N, K1]
                constexpr index_t BK01 = KPerBlock / BK1Value;
                const index_t BK0_     = StrideB / BK1Value;
                const index_t BK00     = BK0_ / BK01;

                const auto b_grid_desc_bk00_n_bk01_bk1_permute =
                    make_naive_tensor_descriptor_packed(make_tuple(BK00, N, BK01, BK1Value));

                const auto b_grid_desc_bk0_n_bk1_permute = transform_tensor_descriptor(
                    b_grid_desc_bk00_n_bk01_bk1_permute,
                    make_tuple(make_merge_transform(make_tuple(BK00, BK01)),
                               make_pass_through_transform(make_tuple(N)),
                               make_pass_through_transform(BK1Value)),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                return b_grid_desc_bk0_n_bk1_permute;
            }
        }
    }

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeAMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

        return MakeGemmMmaTileDescriptor<MXdlPerWave, MWaves, MPerXdl>(ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeBMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        // constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);

        // return MakeGemmMmaTileDescriptor<NXdlPerWave, NWaves, NPerXdl>(BBlockDesc_BK0_N_BK1{});

        return MakeGemmMmaTileDescriptor<NXdlPerWave, NWave, NPerXdl>(BBlockDesc_BK0_N_BK1{});
    }

    __host__ __device__ static auto
    MakeCGridDescriptor_M_N(index_t M, index_t MPad, index_t N, index_t NPad, index_t StrideC)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

        // pad M and N
        return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                           make_tuple(make_right_pad_transform(M, MPad - M),
                                                      make_right_pad_transform(N, NPad - N)),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}));
#if 0
        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        if constexpr(GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M and N
            return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                               make_tuple(make_right_pad_transform(M, MPad - M),
                                                          make_right_pad_transform(N, NPad - N)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_right_pad_transform(M, MPad - M), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad N, but not M
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_pass_through_transform(M), make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad M or N
            return c_grid_desc_mraw_nraw;
        }
#endif
    }

    struct Problem
    {
        __host__ Problem(index_t M_,
                         index_t N_,
                         index_t K_,
                         index_t StrideA_,
                         index_t StrideB_,
                         index_t StrideC_,
                         index_t KBatch_)
            : M{M_},
              N{N_},
              K{K_},
              StrideA{StrideA_},
              StrideB{StrideB_},
              StrideC{StrideC_},
              KBatch{KBatch_},
              MPadded{CalculateMPadded(M_)},
              NPadded{CalculateNPadded(N_)},
              KRead{CalculateKRead(K_, KBatch_)},
              KPadded{CalculateKPadded(K_, KBatch_)},
              AK0{CalculateAK0Padded(K_, KBatch_)},
              BK0{CalculateBK0Padded(K_, KBatch_)},
              MBlock{CalculateMBlock(M_)},
              NBlock{CalculateNBlock(N_)}
        {
        }

        __host__ void Print() const
        {
            std::cout << "problem {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "SA:" << StrideA << ", " << "SB:" << StrideB << ", " << "SC:" << StrideC
                      << ", " << "MP:" << MPadded << ", " << "NP:" << NPadded << ", "
                      << "KRead:" << KRead << ", " << "KP:" << KPadded << ", " << "AK0:" << AK0
                      << ", " << "BK0:" << BK0 << ", " << "MBlock: " << MBlock << ", "
                      << "NBlock: " << NBlock << "}" << std::endl;
        }

        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        index_t KBatch;
        index_t MPadded;
        index_t NPadded;
        index_t KRead;
        index_t KPadded;
        index_t AK0;
        index_t BK0;
        index_t MBlock;
        index_t NBlock;
    };

    // Argument
    struct Argument : public tensor_operation::device::BaseArgument, public Problem
    {
        __host__ Argument(const ADataType* p_a_grid_,
                          const BDataType* p_b_grid_,
                          CDataType* p_c_grid_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          index_t StrideA_,
                          index_t StrideB_,
                          index_t StrideC_,
                          index_t k_batch_,
                          bool is_reduce_ = false)
            : Problem{M_, N_, K_, StrideA_, StrideB_, StrideC_, k_batch_},
              p_a_grid{p_a_grid_},
              p_b_grid{p_b_grid_},
              p_c_grid{p_c_grid_},
              is_reduce(is_reduce_)
        {
        }

        __host__ __device__ inline bool IsReduceAdd() const
        {
            return (Problem::KBatch > 1) && is_reduce;
        }

        __host__ __device__ inline bool IsAtomicAdd() const
        {
            return (Problem::KBatch > 1) && (!is_reduce);
        }

        const ADataType* p_a_grid;
        const BDataType* p_b_grid;
        CDataType* p_c_grid;
        bool is_reduce;
    };

    struct SplitKBatchOffset
    {

        __device__ SplitKBatchOffset(Argument& karg)
        {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = blockIdx.z * karg.KRead / APackedSize;
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = blockIdx.z * karg.KRead * karg.StrideA;
            }

            if(blockIdx.z < static_cast<uint32_t>(karg.KBatch - 1))
            {
                karg.K = karg.KRead;
            }
            else
            {
                karg.K = karg.K - karg.KRead * (karg.KBatch - 1);
            }

            if(karg.IsReduceAdd())
            {
                c_reduce_offset = blockIdx.z * karg.M * karg.N;
            }
            else
            {
                c_reduce_offset = 0;
            }
        }

        index_t a_k_split_offset;
        index_t c_reduce_offset;
    };

    __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        constexpr index_t MWave    = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t WaveSize = BlockSize / (MWave * NWave);
        // A matrix in LDS memory, dst of blockwise copy
        if constexpr(ABlockLdsExtraM)
        {
            return make_naive_tensor_descriptor(
                make_tuple(AK0Number, Number<MPerBlock>{}, AK1Number),
                make_tuple(AK1Number, Number<KPerBlock + ABlockLdsExtraM>{}, I1));
        }
        // xor tensor transformation request more unnecessary vgpr usage, would cause register spill
        // in some cases.
        else if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            constexpr auto a_lds_block_desc =
                make_naive_tensor_descriptor(make_tuple(AK0Number, Number<MPerBlock>{}, AK1Number),
                                             make_tuple(AK1Number, Number<KPerBlock>{}, I1));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(make_xor_with_modulo_transform(
                               make_tuple(Number<MPerBlock>{}, Number<AK0Number>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}));

            return a_lds_block_desc_permuted;
        }
        else // ColumnMajor A
        {
            // kfold and mpair dimension is not always required.
            // more dimension in merge_transform increase the difficulty of generating immarg offset
            // for compiler.
            constexpr auto M0 = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I1);
            constexpr auto M1 = MPerBlock / M0;

            constexpr auto KThreadWrite     = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I0);
            constexpr auto K0PerThreadWrite = AK0Number / KThreadWrite;
            constexpr auto KThreadRead      = WaveSize / MPerXdl;
            constexpr auto K0PerThreadRead  = AK0Number / KThreadRead;

            constexpr auto kfold = (AK1Number * M0 * sizeof(ADataType) > 128)
                                       ? 1
                                       : 128 / (AK1Number * M0 * sizeof(ADataType));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=mpair<=n0
            constexpr auto mpair = (AK1Number * MPerXdl * sizeof(ADataType) > 128)
                                       ? 1
                                       : ((128 / (AK1Number * MPerXdl * sizeof(ADataType))) > M0
                                              ? M0
                                              : 128 / (AK1Number * MPerXdl * sizeof(ADataType)));

            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Number<KThreadWrite / kfold / KThreadReadPerm>{},
                           Number<K0PerThreadWrite>{},
                           Number<KThreadReadPerm * M1>{},
                           Number<kfold * M0 / mpair>{},
                           Number<mpair>{},
                           AK1Number));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_xor_with_modulo_transform(
                        make_tuple(Number<KThreadReadPerm * M1>{}, Number<kfold * M0 / mpair>{})),
                    make_pass_through_transform(Number<mpair>{}),
                    make_pass_through_transform(AK1Number)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

            constexpr auto a_lds_block_desc_unmerged = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(Number<KThreadReadPerm>{}, Number<M1>{})),
                    make_unmerge_transform(make_tuple(Number<kfold>{}, Number<M0 / mpair>{})),
                    make_pass_through_transform(Number<mpair>{}),
                    make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<1>{},
                           Sequence<2>{},
                           Sequence<0, 3>{},
                           Sequence<4, 5>{},
                           Sequence<6>{},
                           Sequence<7>{}));

            constexpr auto a_lds_block_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<KThreadReadPerm>{},
                                          Number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          Number<kfold>{},
                                          Number<K0PerThreadWrite>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<M0 / mpair>{}, Number<mpair>{}, Number<M1>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0, 1, 4, 2>{}, Sequence<5, 6, 3>{}, Sequence<7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return a_lds_block_desc_ak0_m_ak1;
        }
    }

    __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // K0 -> N0/NWave -> NWave -> KLane -> NLane -> KPack
        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<NXdlPerWave>{},
                       I1,
                       Number<KRepeat>{},
                       Number<BK1Value>{})); //??? BK1Value same as KPack?
    }

    __device__ static constexpr auto GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    using BlockwiseGemmPipe =
        remove_cvref_t<decltype(BlockGemmBPreshufflePipeline_Selector<
                                BlkGemmPipelineVer,
                                BlkGemmPipeSched,
                                BlockSize,
                                ADataType,
                                BDataType,
                                ComputeTypeA,
                                AccDataType,
                                decltype(GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()),
                                decltype(GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()),
                                decltype(MakeAMmaTileDescriptor_M0_M1_M2_K(
                                    GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1())),
                                decltype(MakeBMmaTileDescriptor_N0_N1_N2_K(
                                    GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1())),
                                ABlockTransferSrcScalarPerVector,
                                BBlockTransferSrcScalarPerVector,
                                MPerBlock,
                                NPerBlock,
                                KPerBlock,
                                MPerXdl,
                                NPerXdl,
                                MXdlPerWave,
                                NXdlPerWave,
                                KPack>())>;

    __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        // LDS allocation for C shuffle in LDS
        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

        constexpr auto c_block_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();

        return math::max(a_block_space_size_aligned * sizeof(ADataType) / APackedSize,
                         c_block_size * sizeof(CShuffleDataType));
    }

    IS_VALID_COMPILATION_PARAMETER_IMPL(CDataType)

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    __host__ static constexpr bool CheckValidity(const Argument& karg)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        if constexpr(NXdlPerWave % CShuffleNXdlPerWavePerShuffle != 0)
        {
            return false;
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding) &&
                     !(is_same<tensor_layout::gemm::RowMajor, ALayout>::value))
        {
            if(!(karg.M % MPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M value is not a multiple of MPerBlock! M: " << karg.M << " "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding) &&
                     (is_same<tensor_layout::gemm::RowMajor, BLayout>::value))
        {
            if(!(karg.N % NPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N value is not a multiple of NPerBlock! N: " << karg.N << " "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::KPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {

            auto K_t = karg.KBatch * KPerBlock;
            if(!(karg.K % K_t == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K value is not a multiple of K_Batch * K0PerBlock * K1! K: "
                              << karg.K << " " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            constexpr auto KReadVec = math::lcm(AK1Number, BK1Number);
            auto K_t                = karg.KBatch * KReadVec;
            auto KReadPadSplited    = math::integer_divide_ceil(karg.K, K_t) * KReadVec;
            if((KReadPadSplited * (karg.KBatch - 1)) >= karg.K)
            {
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            if(karg.K % ABlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K (" << karg.K
                              << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                              << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(karg.M % ABlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M (" << karg.M
                              << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                              << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            if(karg.N % BBlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N (" << karg.N
                              << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                              << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(karg.K % BBlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K (" << karg.K
                              << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                              << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
        {
            if(karg.N % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N (" << karg.N
                              << ") value is not a multiple of "
                                 "CShuffleBlockTransferScalarPerVector_NPerBlock ("
                              << CShuffleBlockTransferScalarPerVector_NPerBlock << " )! "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(karg.M % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M (" << karg.M
                              << ") value is not a multiple of "
                                 "CShuffleBlockTransferScalarPerVector_NPerBlock ("
                              << CShuffleBlockTransferScalarPerVector_NPerBlock << " )! "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(!(is_same<remove_cvref_t<CDataType>, half_t>::value ||
                       is_same<remove_cvref_t<CDataType>, float>::value ||
                       is_same<remove_cvref_t<CDataType>, bhalf_t>::value ||
                       is_same<remove_cvref_t<CDataType>, int32_t>::value))
        {
            if(!karg.IsReduceAdd())
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << " KBatch: " << karg.KBatch << " > 1 is not support yet" << __FILE__
                              << ":" << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                if(karg.KBatch > 1)
                {
                    return false;
                }
            }
        }

        // check gridwise gemm pipeline
        const auto num_k_loop = karg.AK0 / (KPerBlock / AK1Value);

        if(num_k_loop <= BlockwiseGemmPipe::PrefetchStages)
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return BlockwiseGemmPipe::BlockHasHotloop(num_loop);
    }

    __host__ static constexpr TailNumber CalculateKBlockLoopTailNum(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return BlockwiseGemmPipe::BlockLoopTailNum(num_loop);
    }

    template <typename CGridDesc>
    __host__ __device__ static constexpr auto MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const CGridDesc& c_grid_desc_m_n, index_t MBlock, index_t NBlock)
    {
        const auto c_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return c_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    // if arch = gfx942
    using Block2CTileMap = BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock>;
    // using Block2CTileMap = BlockToCTileMap_3DGrid_KSplit<MPerBlock, NPerBlock>;

    template <typename AGridDesc_AK0_M_K1,
              typename BGridDesc_BPreshuffled,
              typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run(const ADataType* p_a_grid,
                               const BDataType* p_b_grid,
                               CDataType* p_c_grid,
                               void* p_shared,
                               const Problem& problem,
                               const AGridDesc_AK0_M_K1& a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BPreshuffled& b_grid_desc_bpreshuffled,
                               const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                               const index_t k_id)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bpreshuffled.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        const AElementwiseOperation a_element_op{};
        // const BElementwiseOperation b_element_op{};
        const CElementwiseOperation c_element_op{};

        // divide block work by [M, N]
        const auto block_2_ctile_map = Block2CTileMap{problem.M, problem.N, 4};

        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);
        const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

        // N0, K0, Blocksize*KPack
        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NXdlPerWave);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0Number, MPerBlock, AK1Number>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                ADataType,
                                                ADataType,
                                                decltype(a_grid_desc_ak0_m_ak1),
                                                decltype(a_block_desc_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<0, 1, 2>,
                                                ABlockTransferSrcVectorDim,
                                                2,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_AK1,
                                                1,
                                                1,
                                                AThreadTransferSrcResetCoordinateAfterRun,
                                                true,
                                                BlockwiseGemmPipe::GlobalBufferNum>(
                a_grid_desc_ak0_m_ak1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // B matrix threadwise copy, using threadwiseTensorSliceTransfer_v2
        auto b_block_buf = make_static_buffer<AddressSpaceEnum::Vgpr, BDataType>(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        auto b_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
            BDataType,
            BDataType,
            decltype(b_grid_desc_bpreshuffled),
            decltype(b_block_desc_bk0_n_bk1),
            Sequence<Number<NXdlPerWave>{}, I1, Number<KRepeat>{}, Number<BK1Value>{}>,
            Sequence<1, 2, 0, 3>,
            3,
            BBlockTransferSrcScalarPerVector,
            BThreadTransferSrcResetCoordinateAfterRun,
            true>(b_grid_desc_bpreshuffled,
                  make_multi_index(n_block_data_idx_on_grid,
                                   get_warp_local_1d_id() % NWave,
                                   k_id,
                                   KPack * (get_thread_local_1d_id() % WarpSize)));

        // LDS allocation for A and B: be careful of alignment

        // Cast after lds
        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ADataType*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1Number, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(0, 0, KRepeat, 0);

        // Blockwise GEMM pipeline
        static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
        auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
        auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            KPerBlock);

        blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, TailNum>(a_grid_desc_ak0_m_ak1,
                                                                         a_block_desc_ak0_m_ak1,
                                                                         a_blockwise_copy,
                                                                         a_grid_buf,
                                                                         a_block_buf,
                                                                         a_block_slice_copy_step,
                                                                         b_grid_desc_bpreshuffled,
                                                                         b_blockwise_copy,
                                                                         b_grid_buf,
                                                                         b_block_buf,
                                                                         b_block_slice_copy_step,
                                                                         c_thread_buf,
                                                                         num_k_block_main_loop);

        // shuffle C and write out
        {
            static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                              NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);

            // TODO: hacky, fix it!
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm_pipeline.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            // TODO: hacky, fix it!
            // c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp is only used to get lengths
            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                blockwise_gemm_pipeline.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I4);
            constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I5);
            constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I6);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I7);

            constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<CShuffleDataType*>(p_shared),
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 = transform_tensor_descriptor(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per shuffle
                        M1,                                      // M1 = MWave
                        M2,                                      // M2 * M3 * M4 = MPerXdl
                        M3,
                        M4)),
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per shuffle
                        N1,                                      // N1 = NWave
                        N2))),                                   // N2 = NPerXdl
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4, 5, 6>{}, Sequence<>{}, Sequence<1, 3, 7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm_pipeline.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   CShuffleDataType,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   Sequence<CShuffleMXdlPerWavePerShuffle,
                                                            CShuffleNXdlPerWavePerShuffle,
                                                            I1,
                                                            I1,
                                                            M2,
                                                            I1,
                                                            M4,
                                                            I1>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     m_thread_data_on_block_idx[I3],
                                     m_thread_data_on_block_idx[I4],
                                     n_thread_data_on_block_idx[I2]),
                    ck::tensor_operation::element_wise::PassThrough{}};

            // shuffle: blockwise copy C from LDS to global
            auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // ThreadGroup
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                         1,
                         CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                CShuffleDataType,     // typename SrcData,
                CDataType,            // typename DstData,
                decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 c_grid_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(block_m_id, 0, block_n_id, 0),
                 c_element_op};

            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c_vgpr =
                SpaceFillingCurve<Sequence<MXdlPerWave, NXdlPerWave, 1, 1, M2, 1, M4, 1>,
                                  Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                  Sequence<CShuffleMXdlPerWavePerShuffle,
                                           CShuffleNXdlPerWavePerShuffle,
                                           1,
                                           1,
                                           M2,
                                           1,
                                           M4,
                                           1>>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_c_global =
                SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                           1,
                                           CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                c_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                              sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                              c_thread_buf,
                                              c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                              c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                c_shuffle_block_copy_lds_to_global.Run(
                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                    c_shuffle_block_buf,
                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                    c_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                    // move on C
                    c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        c_grid_desc_mblock_mperblock_nblock_nperblock, c_global_step);
                }
            });
        }
    }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run(const ADataType* p_a_grid,
                               const BDataType* p_b_grid,
                               CDataType* p_c_grid,
                               void* p_shared,
                               const Problem& problem,
                               const index_t k_id,
                               const index_t Kt)
    {
        index_t BN0Shuffled              = CalculateBN0Shuffled(problem.N);
        index_t BK0Shuffled              = CalculateBK0Shuffled(Kt);
        const auto a_grid_desc_ak0_m_ak1 = MakeAGridDescriptor_AK0_M_AK1(
            problem.M, problem.MPadded, problem.K, problem.KPadded, problem.StrideA, problem.AK0);
        const auto b_grid_desc_bpreshuffled =
            MakeBGridDescriptor_Preshuffled(BN0Shuffled, BK0Shuffled);
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideC);
        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                c_grid_desc_m_n, problem.MBlock, problem.NBlock);

        Run<decltype(a_grid_desc_ak0_m_ak1),
            decltype(b_grid_desc_bpreshuffled),
            decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
            HasMainKBlockLoop,
            CGlobalMemoryDataOperation,
            TailNum>(p_a_grid,
                     p_b_grid,
                     p_c_grid,
                     p_shared,
                     problem,
                     a_grid_desc_ak0_m_ak1,
                     b_grid_desc_bpreshuffled,
                     c_grid_desc_mblock_mperblock_nblock_nperblock,
                     k_id);
    }

    template <typename AGridDesc_AK0_M_K1,
              typename BGridDesc_BPreshuffled,
              typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run_2Lds(const ADataType* p_a_grid,
                                    const BDataType* p_b_grid,
                                    CDataType* p_c_grid,
                                    void* p_shared_0,
                                    void* p_shared_1,
                                    const Problem& problem,
                                    const AGridDesc_AK0_M_K1& a_grid_desc_ak0_m_ak1,
                                    const BGridDesc_BPreshuffled& b_grid_desc_bpreshuffled,
                                    const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                        c_grid_desc_mblock_mperblock_nblock_nperblock,
                                    const index_t k_id)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bpreshuffled.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        const AElementwiseOperation a_element_op{};
        // const BElementwiseOperation b_element_op{};
        const CElementwiseOperation c_element_op{};

        // divide block work by [M, N]
        const auto block_2_ctile_map = Block2CTileMap{problem.M, problem.N, 4};

        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);
        const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

        // N0, K0, Blocksize*KPack
        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NXdlPerWave);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0Number, MPerBlock, AK1Number>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                ADataType,
                                                ADataType,
                                                decltype(a_grid_desc_ak0_m_ak1),
                                                decltype(a_block_desc_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<0, 1, 2>,
                                                ABlockTransferSrcVectorDim,
                                                2,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_AK1,
                                                1,
                                                1,
                                                AThreadTransferSrcResetCoordinateAfterRun,
                                                true,
                                                2>(
                a_grid_desc_ak0_m_ak1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        // Thread-wise copy
        // K0 -> N0/NWave -> NWave -> KLane -> NLane -> KPack
        auto b_block_buf_ping = make_static_buffer<AddressSpaceEnum::Vgpr, BDataType>(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());
        auto b_block_buf_pong = make_static_buffer<AddressSpaceEnum::Vgpr, BDataType>(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());
        auto b_block_bufs = make_tuple(b_block_buf_ping, b_block_buf_pong);

        auto b_blockwise_copy = ThreadwiseTensorSliceTransfer_v2<
            BDataType,
            BDataType,
            decltype(b_grid_desc_bpreshuffled),
            decltype(b_block_desc_bk0_n_bk1),
            Sequence<Number<NXdlPerWave>{}, I1, Number<KRepeat>{}, Number<BK1Value>{}>,
            Sequence<1, 2, 0, 3>,
            3,
            BBlockTransferSrcScalarPerVector,
            BThreadTransferSrcResetCoordinateAfterRun,
            true>(b_grid_desc_bpreshuffled,
                  make_multi_index(n_block_data_idx_on_grid,
                                   get_warp_local_1d_id() % NWave,
                                   k_id,
                                   KPack * (get_thread_local_1d_id() % WarpSize)));

        // LDS allocation for A and B: be careful of alignment
        auto a_block_buf_ping = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ADataType*>(p_shared_0), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto a_block_buf_pong = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ADataType*>(p_shared_1), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto a_block_bufs = make_tuple(a_block_buf_ping, a_block_buf_pong);

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1Number, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(0, 0, KRepeat, 0);

        // Blockwise GEMM pipeline
        static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
        auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
        auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            KPerBlock);

        blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, TailNum>(a_grid_desc_ak0_m_ak1,
                                                                         a_block_desc_ak0_m_ak1,
                                                                         a_blockwise_copy,
                                                                         a_grid_buf,
                                                                         a_block_bufs,
                                                                         a_block_slice_copy_step,
                                                                         b_grid_desc_bpreshuffled,
                                                                         b_blockwise_copy,
                                                                         b_grid_buf,
                                                                         b_block_bufs,
                                                                         b_block_slice_copy_step,
                                                                         c_thread_buf,
                                                                         num_k_block_main_loop);

        // shuffle C and write out
        {
            static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                              NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);

            // TODO: hacky, fix it!
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm_pipeline.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            // TODO: hacky, fix it!
            // c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp is only used to get lengths
            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                blockwise_gemm_pipeline.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I4);
            constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I5);
            constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I6);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I7);

            constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<CShuffleDataType*>(p_shared_0),
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 = transform_tensor_descriptor(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per shuffle
                        M1,                                      // M1 = MWave
                        M2,                                      // M2 * M3 * M4 = MPerXdl
                        M3,
                        M4)),
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per shuffle
                        N1,                                      // N1 = NWave
                        N2))),                                   // N2 = NPerXdl
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4, 5, 6>{}, Sequence<>{}, Sequence<1, 3, 7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm_pipeline.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   CShuffleDataType,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   Sequence<CShuffleMXdlPerWavePerShuffle,
                                                            CShuffleNXdlPerWavePerShuffle,
                                                            I1,
                                                            I1,
                                                            M2,
                                                            I1,
                                                            M4,
                                                            I1>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     m_thread_data_on_block_idx[I3],
                                     m_thread_data_on_block_idx[I4],
                                     n_thread_data_on_block_idx[I2]),
                    ck::tensor_operation::element_wise::PassThrough{}};

            // shuffle: blockwise copy C from LDS to global
            auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // ThreadGroup
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                         1,
                         CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                CShuffleDataType,     // typename SrcData,
                CDataType,            // typename DstData,
                decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 c_grid_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(block_m_id, 0, block_n_id, 0),
                 c_element_op};

            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c_vgpr =
                SpaceFillingCurve<Sequence<MXdlPerWave, NXdlPerWave, 1, 1, M2, 1, M4, 1>,
                                  Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                  Sequence<CShuffleMXdlPerWavePerShuffle,
                                           CShuffleNXdlPerWavePerShuffle,
                                           1,
                                           1,
                                           M2,
                                           1,
                                           M4,
                                           1>>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_c_global =
                SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                           1,
                                           CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                c_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                              sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                              c_thread_buf,
                                              c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                              c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                c_shuffle_block_copy_lds_to_global.Run(
                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                    c_shuffle_block_buf,
                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                    c_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                    // move on C
                    c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        c_grid_desc_mblock_mperblock_nblock_nperblock, c_global_step);
                }
            });
        }
    }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run_2Lds(const ADataType* p_a_grid,
                                    const BDataType* p_b_grid,
                                    CDataType* p_c_grid,
                                    void* p_shared_0,
                                    void* p_shared_1,
                                    const Problem& problem,
                                    const index_t k_id,
                                    const index_t Kt)
    {
        index_t BN0Shuffled              = CalculateBN0Shuffled(problem.N);
        index_t BK0Shuffled              = CalculateBK0Shuffled(Kt);
        const auto a_grid_desc_ak0_m_ak1 = MakeAGridDescriptor_AK0_M_AK1(
            problem.M, problem.MPadded, problem.K, problem.KPadded, problem.StrideA, problem.AK0);
        const auto b_grid_desc_bpreshuffled =
            MakeBGridDescriptor_Preshuffled(BN0Shuffled, BK0Shuffled);
        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideC);

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                c_grid_desc_m_n, problem.MBlock, problem.NBlock);

        Run_2Lds<decltype(a_grid_desc_ak0_m_ak1),
                 decltype(b_grid_desc_bpreshuffled),
                 decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                 HasMainKBlockLoop,
                 CGlobalMemoryDataOperation,
                 TailNum>(p_a_grid,
                          p_b_grid,
                          p_c_grid,
                          p_shared_0,
                          p_shared_1,
                          problem,
                          a_grid_desc_ak0_m_ak1,
                          b_grid_desc_bpreshuffled,
                          c_grid_desc_mblock_mperblock_nblock_nperblock,
                          k_id);
    }
};

} // namespace ck
