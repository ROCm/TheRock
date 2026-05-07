// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename CGridDesc_M_N,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
#if CK_USE_WAVES_PER_EU
    __attribute__((amdgpu_waves_per_eu(CK_MIN_WAVES_PER_EU, CK_MAX_WAVES_PER_EU)))
#endif
    kernel_gemm_xdlops_v2r3(const FloatAB* __restrict__ p_a_grid,
                            const FloatAB* __restrict__ p_b_grid,
                            FloatC* __restrict__ p_c_grid,
                            const AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1,
                            const BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1,
                            const CGridDesc_M_N c_grid_desc_m_n)
{
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx94__) || defined(__gfx11__) || \
    defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                      p_b_grid,
                                                      p_c_grid,
                                                      p_shared,
                                                      a_grid_desc_k0_m_k1,
                                                      b_grid_desc_k0_n_k1,
                                                      c_grid_desc_m_n);
    }
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_n_k1;
    ignore = c_grid_desc_m_n;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename GridwiseGemm, bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
#if CK_USE_WAVES_PER_EU
    __attribute__((amdgpu_waves_per_eu(CK_MIN_WAVES_PER_EU, CK_MAX_WAVES_PER_EU)))
#endif
    kernel_gemm_xdlops_v2r3(const typename GridwiseGemm::Argument karg)
{
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx94__) || defined(__gfx11__) || \
    defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

        const auto a_grid_desc_k0_m_k1 =
            amd_wave_read_first_lane(GridwiseGemm::MakeAGridDescriptor_K0_M_K1(
                karg.M, karg.MPadded, karg.K, karg.K0, karg.StrideA));
        const auto b_grid_desc_k0_n_k1 =
            amd_wave_read_first_lane(GridwiseGemm::MakeBGridDescriptor_K0_N_K1(
                karg.K, karg.N, karg.NPadded, karg.K0, karg.StrideB));
        const auto c_grid_desc_m_n = amd_wave_read_first_lane(GridwiseGemm::MakeCGridDescriptor_M_N(
            karg.M, karg.MPadded, karg.N, karg.NPadded, karg.StrideC));

        GridwiseGemm::template Run<HasMainKBlockLoop>(karg.p_a_grid,
                                                      karg.p_b_grid,
                                                      karg.p_c_grid,
                                                      p_shared,
                                                      a_grid_desc_k0_m_k1,
                                                      b_grid_desc_k0_n_k1,
                                                      c_grid_desc_m_n);
    }
#else
    ignore = karg;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t K1Value,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          bool BBlockLdsExtraN,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          index_t NumGemmKPrefetchStage = 1,
          LoopScheduler LoopSched       = make_default_loop_scheduler(),
          PipelineVersion PipelineVer   = PipelineVersion::v1>
struct GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3
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
    static constexpr bool is_single_rate_mfma =
        (((is_same<FloatAB, half_t>::value || is_same<FloatAB, bhalf_t>::value) && K1Value <= 4) ||
         (is_same<FloatAB, int8_t>::value && K1Value <= 8) ||
         ((is_same<FloatAB, f8_t>::value || is_same<FloatAB, bf8_t>::value) && K1Value < 32))
            ? true
            : false;
    static constexpr auto is_scale_mfma = false;
    static constexpr auto K1            = Number<math::max(
        K1Value,
        MfmaSelector<FloatAB, MPerXdl, NPerXdl, FloatAB, is_single_rate_mfma, is_scale_mfma>::
            selected_mfma.k_per_blk)>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using ElementDataTypeAB = conditional_t<is_same_v<FloatAB, ck::tf32_t>, float, FloatAB>;

    __host__ static auto CalculateGridSize(index_t M, index_t N)
    {
        return std::make_tuple(Block2CTileMap::CalculateGridSize(M, N), 1, 1);
    }

    template <typename CGridDesc_M_N>
    __host__ static auto CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        return std::make_tuple(Block2CTileMap::CalculateGridSize(c_grid_desc_m_n), 1, 1);
    }

    template <typename>
    __host__ static auto CalculateGridSize(index_t M, index_t N)
    {
        return std::make_tuple(Block2CTileMap::CalculateGridSize(M, N), 1, 1);
    }

    __host__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_divide_ceil(M, MPerBlock) * MPerBlock;
    }

    __host__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_divide_ceil(N, NPerBlock) * NPerBlock;
    }

    __host__ static auto CalculateK0(index_t K) { return math::integer_divide_ceil(K, K1Value); }

    // Argument
    struct Problem
    {
        __host__ Problem(index_t M_,
                         index_t N_,
                         index_t K_,
                         index_t StrideA_,
                         index_t StrideB_,
                         index_t StrideC_)
            : M{M_},
              N{N_},
              K{K_},
              StrideA{StrideA_},
              StrideB{StrideB_},
              StrideC{StrideC_},
              MPadded{CalculateMPadded(M_)},
              NPadded{CalculateNPadded(N_)},
              K0{CalculateK0(K_)}
        {
        }

        __host__ void Print() const
        {
            std::cout << "problem {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "SA:" << StrideA << ", " << "SB:" << StrideB << ", " << "SC:" << StrideC
                      << ", " << "MP:" << MPadded << ", " << "NP:" << NPadded << ", " << "K0:" << K0
                      << "}" << std::endl;
        }

        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        index_t MPadded;
        index_t NPadded;
        index_t K0;
    };

    // Argument
    struct Argument : public Problem, public tensor_operation::device::BaseArgument
    {
        __host__ Argument(const ElementDataTypeAB* p_a_grid_,
                          const ElementDataTypeAB* p_b_grid_,
                          FloatC* p_c_grid_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          index_t StrideA_,
                          index_t StrideB_,
                          index_t StrideC_)
            : Problem{M_, N_, K_, StrideA_, StrideB_, StrideC_},
              p_a_grid{p_a_grid_},
              p_b_grid{p_b_grid_},
              p_c_grid{p_c_grid_}
        {
        }

        const ElementDataTypeAB* p_a_grid;
        const ElementDataTypeAB* p_b_grid;
        FloatC* p_c_grid;
    };

    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>())>;

    // denorm test fix, required to work around fp16 mfma issue
    // we convert fp16->fp32->bf16 and execute bf16 mfma instruction
    // when mfma if fixed, remove this section and update
    // FloatABAdjusted -> FloatAB throughout this file
#if CK_GFX90A_DENORM_WORKAROUND
    using FloatABAdjusted = conditional_t<is_same_v<FloatAB, ck::half_t>, ck::bhalf_t, FloatAB>;
#else
    using FloatABAdjusted = FloatAB;
#endif

    __host__ __device__ static constexpr auto GetABlockDescriptor_K0PerBlock_MPerBlock_K1()
    {
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_k0_m_k1 = [&]() {
            if constexpr(ABlockLdsExtraM)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    make_tuple(Number<MPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);
            }
        }();

        return a_block_desc_k0_m_k1;
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_K0PerBlock_NPerBlock_K1()
    {
        constexpr auto max_lds_align = K1;

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_k0_n_k1 = [&]() {
            if constexpr(BBlockLdsExtraN)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    make_tuple(Number<NPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1), max_lds_align);
            }
        }();

        return b_block_desc_k0_n_k1;
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_k0_m_k1 = GetABlockDescriptor_K0PerBlock_MPerBlock_K1();

        constexpr auto b_block_desc_k0_n_k1 = GetBBlockDescriptor_K0PerBlock_NPerBlock_K1();

        constexpr auto max_lds_align = K1;

        constexpr auto a_block_space_size_aligned =
            math::integer_least_multiple(a_block_desc_k0_m_k1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned =
            math::integer_least_multiple(b_block_desc_k0_n_k1.GetElementSpaceSize(), max_lds_align);

        return (a_block_space_size_aligned + b_block_space_size_aligned) *
               sizeof(ElementDataTypeAB);
    }

    template <
        InMemoryDataOperationEnum CGlobalMemoryDataOperation_ = InMemoryDataOperationEnum::Set>
    __device__ static bool constexpr IsValidCompilationParameter()
    {
        return ck::tensor_operation::device::IsValidGemmCompilationParameter<
            BlockSize,
            MPerBlock,
            NPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            FloatC,
            CGlobalMemoryDataOperation>();
    }

    template <typename AGridDesc_K0_M_K1, typename BGridDesc_K0_N_K1, typename CGridDesc_M_N>
    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1,
                  const BGridDesc_K0_N_K1& b_grid_desc_k0_n_k1,
                  const CGridDesc_M_N& c_grid_desc_m_n)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        const auto M  = a_grid_desc_k0_m_k1.GetLength(I1);
        const auto N  = b_grid_desc_k0_n_k1.GetLength(I1);
        const auto K0 = a_grid_desc_k0_m_k1.GetLength(I0);

        if(!(M == c_grid_desc_m_n.GetLength(I0) && N == c_grid_desc_m_n.GetLength(I1) &&
             K0 == b_grid_desc_k0_n_k1.GetLength(I0) && K1 == a_grid_desc_k0_m_k1.GetLength(I2) &&
             K1 == b_grid_desc_k0_n_k1.GetLength(I2)))
            return false;

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K0 % K0PerBlock == 0))
            return false;

        // check gridwise gemm pipeline
        const auto num_k_loop = K0 / K0PerBlock;

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ static constexpr bool CheckValidity(const Problem& problem)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        // check gridwise gemm pipeline
        const auto num_k_loop = math::integer_divide_ceil(problem.K0, K0PerBlock);
        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = math::integer_divide_ceil(K, K0PerBlock * K1);

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    template <typename CGridDesc>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(const CGridDesc& c_grid_desc_m_n)
    {
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_k0_m_k1 = [&]() {
            if constexpr(ABlockLdsExtraM)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1),
                    make_tuple(Number<MPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, K1), max_lds_align);
            }
        }();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_k0_n_k1 = [&]() {
            if constexpr(BBlockLdsExtraN)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1),
                    make_tuple(Number<NPerBlock + 1>{} * K1, K1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<NPerBlock>{}, K1), max_lds_align);
            }
        }();

        using BlockwiseGemm =
            BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                ElementDataTypeAB,
                                                                ElementDataTypeAB,
                                                                FloatAcc,
                                                                decltype(a_block_desc_k0_m_k1),
                                                                decltype(b_block_desc_k0_n_k1),
                                                                MPerXdl,
                                                                NPerXdl,
                                                                MXdlPerWave,
                                                                NXdlPerWave,
                                                                K1,
                                                                FloatABAdjusted,
                                                                FloatABAdjusted>;

        return BlockwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m_n);
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    using Block2CTileMap = BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock>;

    template <bool HasMainKBlockLoop,
              typename AGridDesc_K0_M_K1,
              typename BGridDesc_K0_N_K1,
              typename CGridDesc_M_N>
    __device__ static void Run(const ElementDataTypeAB* p_a_grid,
                               const ElementDataTypeAB* p_b_grid,
                               FloatC* p_c_grid,
                               void* __restrict__ p_shared,
                               const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1,
                               const BGridDesc_K0_N_K1& b_grid_desc_k0_n_k1,
                               const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m_n);

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_k0_m_k1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_k0_n_k1.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetElementSpaceSize());

        const AElementwiseOperation a_element_op{};
        const BElementwiseOperation b_element_op{};
        const CElementwiseOperation c_element_op{};

        const auto block_2_ctile_map =
            Block2CTileMap{c_grid_desc_m_n.GetLength(I0), c_grid_desc_m_n.GetLength(I1)};

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0),
                          c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1))))
        {
            return;
        }

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_k0_m_k1 = GetABlockDescriptor_K0PerBlock_MPerBlock_K1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_k0_n_k1 = GetBBlockDescriptor_K0PerBlock_NPerBlock_K1();

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<K0PerBlock, MPerBlock, K1>,
                                                ABlockTransferThreadClusterLengths_K0_M_K1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                ElementDataTypeAB,
                                                ElementDataTypeAB,
                                                decltype(a_grid_desc_k0_m_k1),
                                                decltype(a_block_desc_k0_m_k1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                ABlockTransferSrcVectorDim,
                                                2,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_K1,
                                                1,
                                                1,
                                                AThreadTransferSrcResetCoordinateAfterRun,
                                                true,
                                                NumGemmKPrefetchStage>(
                a_grid_desc_k0_m_k1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_k0_m_k1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<K0PerBlock, NPerBlock, K1>,
                                                BBlockTransferThreadClusterLengths_K0_N_K1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                ElementDataTypeAB,
                                                ElementDataTypeAB,
                                                decltype(b_grid_desc_k0_n_k1),
                                                decltype(b_block_desc_k0_n_k1),
                                                BBlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                BBlockTransferSrcVectorDim,
                                                2,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_K1,
                                                1,
                                                1,
                                                BThreadTransferSrcResetCoordinateAfterRun,
                                                true,
                                                NumGemmKPrefetchStage>(
                b_grid_desc_k0_n_k1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_k0_n_k1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check
        auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
            BlockSize,
            ElementDataTypeAB,
            ElementDataTypeAB,
            FloatAcc,
            decltype(a_block_desc_k0_m_k1),
            decltype(b_block_desc_k0_n_k1),
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            K1,
            LoopSched,
            FloatABAdjusted,
            FloatABAdjusted>();

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned =
            math::integer_least_multiple(a_block_desc_k0_m_k1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ElementDataTypeAB*>(p_shared), a_block_desc_k0_m_k1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ElementDataTypeAB*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_k0_n_k1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(K0PerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(K0PerBlock, 0, 0);

        // gridwise GEMM pipeline
        const auto K0                       = a_grid_desc_k0_m_k1.GetLength(I0);
        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(K0 / K0PerBlock);

        GridwiseGemmPipe::template Run<HasMainKBlockLoop>(a_grid_desc_k0_m_k1,
                                                          a_block_desc_k0_m_k1,
                                                          a_blockwise_copy,
                                                          a_grid_buf,
                                                          a_block_buf,
                                                          a_block_slice_copy_step,
                                                          b_grid_desc_k0_n_k1,
                                                          b_block_desc_k0_n_k1,
                                                          b_blockwise_copy,
                                                          b_grid_buf,
                                                          b_block_buf,
                                                          b_block_slice_copy_step,
                                                          blockwise_gemm,
                                                          c_thread_buf,
                                                          num_k_block_main_loop);

        // output: register to global memory
        {
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4);
            constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
            constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_grid =
                m_block_data_idx_on_grid + c_thread_mtx_on_block[I0];

            const index_t n_thread_data_on_grid =
                n_block_data_idx_on_grid + c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_grid_idx =
                m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_grid));

            const auto n_thread_data_on_grid_to_n0_n1_n2_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_grid_idx =
                n_thread_data_on_grid_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_grid));

            auto c_thread_copy =
                ThreadwiseTensorSliceTransfer_v1r3<FloatAcc,
                                                   FloatC,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   decltype(c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                                   CElementwiseOperation,
                                                   Sequence<M0, N0, I1, I1, M2, I1, M4, I1>,
                                                   CThreadTransferSrcDstAccessOrder,
                                                   CThreadTransferSrcDstVectorDim,
                                                   CThreadTransferDstScalarPerVector,
                                                   CGlobalMemoryDataOperation,
                                                   1,
                                                   true>{
                    c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                    make_multi_index(m_thread_data_on_grid_idx[I0],
                                     n_thread_data_on_grid_idx[I0],
                                     m_thread_data_on_grid_idx[I1],
                                     n_thread_data_on_grid_idx[I1],
                                     m_thread_data_on_grid_idx[I2],
                                     m_thread_data_on_grid_idx[I3],
                                     m_thread_data_on_grid_idx[I4],
                                     n_thread_data_on_grid_idx[I2]),
                    c_element_op};

            c_thread_copy.Run(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                              c_thread_buf,
                              c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                              c_grid_buf);
        }
    }
};

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t K1Value,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          bool BBlockLdsExtraN,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          index_t NumGemmKPrefetchStage = 1,
          LoopScheduler LoopSched       = make_default_loop_scheduler(),
          PipelineVersion PipelineVer   = PipelineVersion::v1>
struct GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3_ext
    : GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<BlockSize,
                                              FloatAB,
                                              FloatAcc,
                                              FloatC,
                                              CGlobalMemoryDataOperation,
                                              AElementwiseOperation,
                                              BElementwiseOperation,
                                              CElementwiseOperation,
                                              MPerBlock,
                                              NPerBlock,
                                              K0PerBlock,
                                              MPerXdl,
                                              NPerXdl,
                                              K1Value,
                                              MXdlPerWave,
                                              NXdlPerWave,
                                              ABlockTransferThreadClusterLengths_K0_M_K1,
                                              ABlockTransferThreadClusterArrangeOrder,
                                              ABlockTransferSrcAccessOrder,
                                              ABlockTransferSrcVectorDim,
                                              ABlockTransferSrcScalarPerVector,
                                              ABlockTransferDstScalarPerVector_K1,
                                              AThreadTransferSrcResetCoordinateAfterRun,
                                              ABlockLdsExtraM,
                                              BBlockTransferThreadClusterLengths_K0_N_K1,
                                              BBlockTransferThreadClusterArrangeOrder,
                                              BBlockTransferSrcAccessOrder,
                                              BBlockTransferSrcVectorDim,
                                              BBlockTransferSrcScalarPerVector,
                                              BBlockTransferDstScalarPerVector_K1,
                                              BThreadTransferSrcResetCoordinateAfterRun,
                                              BBlockLdsExtraN,
                                              CThreadTransferSrcDstAccessOrder,
                                              CThreadTransferSrcDstVectorDim,
                                              CThreadTransferDstScalarPerVector,
                                              NumGemmKPrefetchStage,
                                              LoopSched,
                                              PipelineVer>
{
    using Parent =
        GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<BlockSize,
                                                FloatAB,
                                                FloatAcc,
                                                FloatC,
                                                CGlobalMemoryDataOperation,
                                                AElementwiseOperation,
                                                BElementwiseOperation,
                                                CElementwiseOperation,
                                                MPerBlock,
                                                NPerBlock,
                                                K0PerBlock,
                                                MPerXdl,
                                                NPerXdl,
                                                K1Value,
                                                MXdlPerWave,
                                                NXdlPerWave,
                                                ABlockTransferThreadClusterLengths_K0_M_K1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                ABlockTransferSrcAccessOrder,
                                                ABlockTransferSrcVectorDim,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_K1,
                                                AThreadTransferSrcResetCoordinateAfterRun,
                                                ABlockLdsExtraM,
                                                BBlockTransferThreadClusterLengths_K0_N_K1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                BBlockTransferSrcAccessOrder,
                                                BBlockTransferSrcVectorDim,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_K1,
                                                BThreadTransferSrcResetCoordinateAfterRun,
                                                BBlockLdsExtraN,
                                                CThreadTransferSrcDstAccessOrder,
                                                CThreadTransferSrcDstVectorDim,
                                                CThreadTransferDstScalarPerVector,
                                                NumGemmKPrefetchStage,
                                                LoopSched,
                                                PipelineVer>;

    using typename Parent::GridwiseGemmPipe;
    using typename Parent::Problem;

    using Parent::I1;

    using Parent::K1;

    __device__ static auto
    MakeAGridDescriptor_K0_M_K1(index_t M, index_t MPad, index_t K, index_t K0, index_t StrideA)
    {
        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
        {
            const auto K0Pad = math::integer_divide_ceil(K0, K0PerBlock) * K0PerBlock;
            const auto KPad  = K0Pad * K1Value;

            const auto a_grid_desc_m_kpad = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return transform_tensor_descriptor(
                a_grid_desc_m_kpad,
                make_tuple(make_unmerge_transform(make_tuple(K0Pad, K1Value)),
                           make_right_pad_transform(M, MPad - M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding)
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Value)),
                           make_right_pad_transform(M, MPad - M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Value)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    __device__ static auto
    MakeBGridDescriptor_K0_N_K1(index_t K, index_t N, index_t NPad, index_t K0, index_t StrideB)
    {
        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
        {
            const auto K0Pad = math::integer_divide_ceil(K0, K0PerBlock) * K0PerBlock;
            const auto KPad  = K0Pad * K1Value;

            const auto b_grid_desc_kpad_n = transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_right_pad_transform(K, KPad - K), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return transform_tensor_descriptor(
                b_grid_desc_kpad_n,
                make_tuple(make_unmerge_transform(make_tuple(K0Pad, K1Value)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }

        else if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding)
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Value)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Value)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    __device__ static auto
    MakeCGridDescriptor_M_N(index_t M, index_t MPad, index_t N, index_t NPad, index_t StrideC)
    {
        const auto c_grid_desc_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
        {
            return transform_tensor_descriptor(c_grid_desc_m_n,
                                               make_tuple(make_right_pad_transform(M, MPad - M),
                                                          make_right_pad_transform(N, NPad - N)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_pass_through_transform(M), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    IS_VALID_COMPILATION_PARAMETER_IMPL(FloatC)

    __host__ static constexpr bool CheckValidity(const Problem& problem)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {
            if(!(problem.M % MPerBlock == 0))
            {
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {
            if(!(problem.N % NPerBlock == 0))
            {
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::KPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {
            if(!(problem.K0 % K0PerBlock == 0))
            {
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            if(problem.K % ABlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }
        else
        {
            if(problem.M % ABlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            if(problem.N % BBlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }
        else
        {
            if(problem.K % BBlockTransferSrcScalarPerVector != 0)
            {
                return false;
            }
        }

        // check gridwise gemm pipeline
        const auto num_k_loop = math::integer_divide_ceil(problem.K0, K0PerBlock);

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }
};

} // namespace ck
