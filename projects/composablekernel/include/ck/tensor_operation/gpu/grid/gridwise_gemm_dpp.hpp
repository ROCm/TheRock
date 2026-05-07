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
#include "ck/tensor_operation/gpu/block/blockwise_gemm_dpp.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseGemm, bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
#if CK_USE_WAVES_PER_EU
    __attribute__((amdgpu_waves_per_eu(CK_MIN_WAVES_PER_EU, CK_MAX_WAVES_PER_EU)))
#endif
    kernel_gemm_dpp(const typename GridwiseGemm::Argument karg)
{
#if(defined(__gfx103__) || defined(__gfx11__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    const auto a_grid_desc_ak0_m_ak1 = amd_wave_read_first_lane(
        GridwiseGemm::MakeAGridDescriptor_AK0_M_AK1(karg.M, karg.K, karg.AK0, karg.StrideA));
    const auto b_grid_desc_bk0_n_bk1 = amd_wave_read_first_lane(
        GridwiseGemm::MakeBGridDescriptor_BK0_N_BK1(karg.K, karg.N, karg.BK0, karg.StrideB));
    const auto c_grid_desc_m_n = amd_wave_read_first_lane(
        GridwiseGemm::MakeCGridDescriptor_M_N(karg.M, karg.N, karg.StrideC));

    GridwiseGemm::template Run<HasMainKBlockLoop>(karg.p_a_grid,
                                                  karg.p_b_grid,
                                                  karg.p_c_grid,
                                                  p_shared,
                                                  a_grid_desc_ak0_m_ak1,
                                                  b_grid_desc_bk0_n_bk1,
                                                  c_grid_desc_m_n);
#else
    ignore = karg;
#endif
}

template <index_t BlockSize,
          typename ABDataType,
          typename AccDataType,
          typename CDataType,
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
          index_t KPerBlock,
          index_t MPerDpp,
          index_t NPerDpp,
          index_t AK1Value,
          index_t BK1Value,
          index_t MDppPerWave,
          index_t NDppPerWave,
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
          PipelineVersion PipelineVer   = PipelineVersion::v1>
struct GridwiseGemm_ak0mak1_bk0nbk1_mn_dpp
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto AK1         = Number<AK1Value>{};
    static constexpr auto BK1         = Number<BK1Value>{};
    static constexpr auto AK0PerBlock = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0PerBlock = Number<KPerBlock / BK1Value>{};

    static constexpr auto max_lds_align = math::lcm(AK1, BK1);

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;
    // return block_id to C matrix tile idx (m0, n0) mapping
    using Block2CTileMap = BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock>;

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

    __host__ static auto CalculateAK0(index_t K) { return math::integer_divide_floor(K, AK1Value); }
    __host__ static auto CalculateBK0(index_t K) { return math::integer_divide_floor(K, BK1Value); }

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
              AK0{CalculateAK0(K)},
              BK0{CalculateBK0(K)}
        {
        }

        __host__ void Print() const
        {
            std::cout << "problem {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "SA:" << StrideA << ", " << "SB:" << StrideB << ", " << "SC:" << StrideC
                      << ", " << "MP:" << MPadded << ", " << "NP:" << NPadded << ", "
                      << "AK0:" << AK0 << ", " << "BK0:" << BK0 << "}" << std::endl;
        }

        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        index_t MPadded;
        index_t NPadded;
        index_t AK0;
        index_t BK0;
    };

    // Argument
    struct Argument : public Problem, public tensor_operation::device::BaseArgument
    {
        __host__ Argument(const ABDataType* p_a_grid_,
                          const ABDataType* p_b_grid_,
                          CDataType* p_c_grid_,
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

        const ABDataType* p_a_grid;
        const ABDataType* p_b_grid;
        CDataType* p_c_grid;
    };

    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage>())>;

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = [&]() {
            if constexpr(ABlockLdsExtraM)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<AK0PerBlock>{}, Number<MPerBlock>{}, AK1),
                    make_tuple(Number<MPerBlock + 1>{} * AK1, AK1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<AK0PerBlock>{}, Number<MPerBlock>{}, AK1), max_lds_align);
            }
        }();

        return a_block_desc_ak0_m_ak1;
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = [&]() {
            if constexpr(BBlockLdsExtraN)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<BK0PerBlock>{}, Number<NPerBlock>{}, BK1),
                    make_tuple(Number<NPerBlock + 1>{} * BK1, BK1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<BK0PerBlock>{}, Number<NPerBlock>{}, BK1), max_lds_align);
            }
        }();

        return b_block_desc_bk0_n_bk1;
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);
        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        return (a_block_space_size_aligned + b_block_space_size_aligned) * sizeof(ABDataType);
    }

    __host__ static constexpr bool CheckValidity(const Problem& problem)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(AK1)>>::value,
                      "Wrong! AK1 must be known at the time of compilation.");
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(BK1)>>::value,
                      "Wrong! BK1 must be known at the time of compilation.");

        static_assert(
            MPerBlock % (MPerDpp * MDppPerWave) == 0,
            "Invalid tuning parameters! MPerBlock must be divisible by MPerDpp * MDppPerWave.");
        static_assert(
            NPerBlock % (NPerDpp * NDppPerWave) == 0,
            "Invalid tuning parameters! NPerBlock must be divisible by NPerDpp * NDppPerWave.");

        static_assert(
            KPerBlock % AK1Value == 0 && KPerBlock % BK1Value == 0,
            "Invalid tuning parameters! KPerBlock must be divisible by both AK1 and BK1.");

        static_assert(AK1Value % ABlockTransferDstScalarPerVector_K1 == 0,
                      "Invalid tuning parameters! AK1Value must be divisible by "
                      "ABlockTransferDstScalarPerVector_K1");

        static_assert(BK1Value % BBlockTransferDstScalarPerVector_K1 == 0,
                      "Invalid tuning parameters! BK1Value must be divisible by "
                      "BBlockTransferDstScalarPerVector_K1");

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

        if(problem.K % KPerBlock != 0)
        {
            return false;
        }

        // check gridwise gemm pipeline
        const auto num_k_loop = problem.K / KPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        return true;
    }

    __host__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const auto num_loop = K / KPerBlock;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    template <typename CGridDesc>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_N2(const CGridDesc& c_grid_desc_m_n)
    {
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        constexpr index_t KPack = math::max(
            math::lcm(AK1, BK1), DppSelector<ABDataType, MPerDpp, NPerDpp>::selected_dpp.k_per_dpp);

        using BlockwiseGemm =
            BlockwiseGemmDpp_ak0mak1_bk0nbk1_m0n0m1n1m2n2<BlockSize,
                                                          ABDataType,
                                                          AccDataType,
                                                          decltype(a_block_desc_ak0_m_ak1),
                                                          decltype(b_block_desc_bk0_n_bk1),
                                                          MPerDpp,
                                                          NPerDpp,
                                                          MDppPerWave,
                                                          NDppPerWave,
                                                          KPack>;

        return BlockwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_N2(c_grid_desc_m_n);
    }

    static constexpr auto matrix_padder =
        ck::tensor_operation::device::MatrixPadder<GemmSpec, index_t, index_t, index_t>{
            MPerBlock, NPerBlock, KPerBlock};

    __device__ static auto
    MakeAGridDescriptor_AK0_M_AK1(index_t M, index_t K, index_t AK0, index_t StrideA)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        const auto a_grid_desc_m_k = matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
        return transform_tensor_descriptor(
            a_grid_desc_m_k,
            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                       make_pass_through_transform(M)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    __device__ static auto
    MakeBGridDescriptor_BK0_N_BK1(index_t K, index_t N, index_t BK0, index_t StrideB)
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

        const auto b_grid_desc_n_k = matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
        return transform_tensor_descriptor(
            b_grid_desc_n_k,
            make_tuple(make_pass_through_transform(N),
                       make_unmerge_transform(make_tuple(BK0, BK1Value))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}));
    }

    __device__ static auto MakeCGridDescriptor_M_N(index_t M, index_t N, index_t StrideC)
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

        return matrix_padder.PadCDescriptor_M_N(c_grid_desc_mraw_nraw);
    }

    template <bool HasMainKBlockLoop,
              typename AGridDesc_AK0_M_AK1,
              typename BGridDesc_BK0_N_BK1,
              typename CGridDesc_M_N>
    __device__ static void Run(const ABDataType* __restrict__ p_a_grid,
                               const ABDataType* __restrict__ p_b_grid,
                               CDataType* __restrict__ p_c_grid,
                               void* __restrict__ p_shared,
                               const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
                               const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto c_grid_desc_m0_n0_m1_n1_m2_n2 =
            MakeCGridDescriptor_M0_N0_M1_N1_M2_N2(c_grid_desc_m_n);

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_m0_n0_m1_n1_m2_n2.GetElementSpaceSize());

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
               make_tuple(c_grid_desc_m0_n0_m1_n1_m2_n2.GetLength(I0),
                          c_grid_desc_m0_n0_m1_n1_m2_n2.GetLength(I1))))
        {
            return;
        }

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);
        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0PerBlock, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_K0_M_K1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                ABDataType,
                                                ABDataType,
                                                decltype(a_grid_desc_ak0_m_ak1),
                                                decltype(a_block_desc_ak0_m_ak1),
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
                a_grid_desc_ak0_m_ak1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<BK0PerBlock, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_K0_N_K1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                ABDataType,
                                                ABDataType,
                                                decltype(b_grid_desc_bk0_n_bk1),
                                                decltype(b_block_desc_bk0_n_bk1),
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
                b_grid_desc_bk0_n_bk1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[AK0PerBlock, MPerBlock] is in LDS
        //     b_mtx[BK0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        constexpr index_t KPack = math::max(
            math::lcm(AK1, BK1), DppSelector<ABDataType, MPerDpp, NPerDpp>::selected_dpp.k_per_dpp);
        auto blockwise_gemm =
            BlockwiseGemmDpp_ak0mak1_bk0nbk1_m0n0m1n1m2n2<BlockSize,
                                                          ABDataType,
                                                          AccDataType,
                                                          decltype(a_block_desc_ak0_m_ak1),
                                                          decltype(b_block_desc_bk0_n_bk1),
                                                          MPerDpp,
                                                          NPerDpp,
                                                          MDppPerWave,
                                                          NDppPerWave,
                                                          KPack>();

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ABDataType*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ABDataType*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(AK0PerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(BK0PerBlock, 0, 0);

        // gridwise GEMM pipeline
        const auto AK0 = a_grid_desc_ak0_m_ak1.GetLength(I0);
        // (AK0 / AK0PerBlock) is always equal to (BK0 / BK0PerBlock)
        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(AK0 / AK0PerBlock);

        GridwiseGemmPipe::template Run<HasMainKBlockLoop>(a_grid_desc_ak0_m_ak1,
                                                          a_block_desc_ak0_m_ak1,
                                                          a_blockwise_copy,
                                                          a_grid_buf,
                                                          a_block_buf,
                                                          a_block_slice_copy_step,
                                                          b_grid_desc_bk0_n_bk1,
                                                          b_block_desc_bk0_n_bk1,
                                                          b_blockwise_copy,
                                                          b_grid_buf,
                                                          b_block_buf,
                                                          b_block_slice_copy_step,
                                                          blockwise_gemm,
                                                          c_thread_buf,
                                                          num_k_block_main_loop);

        // output: register to global memory
        {
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_n2 =
                blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2();

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2 =
                blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_n2.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_n2.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2.GetLength(I4);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2.GetLength(I5);

            constexpr auto MPerThread = c_thread_desc_m0_n0_m1_n1_m2_n2.GetLength(I4);
            constexpr auto NPerThread = c_thread_desc_m0_n0_m1_n1_m2_n2.GetLength(I5);

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0);

            const index_t m_thread_data_on_grid =
                m_block_data_idx_on_grid + c_thread_mtx_on_block[I0];

            const index_t n_thread_data_on_grid =
                n_block_data_idx_on_grid + c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_grid_to_m0_m1_m2_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(M0, M1, M2))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_grid_idx =
                m_thread_data_on_grid_to_m0_m1_m2_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_grid));

            const auto n_thread_data_on_grid_to_n0_n1_n2_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_grid_idx =
                n_thread_data_on_grid_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_grid));

            auto c_thread_copy =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   CDataType,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_n2),
                                                   decltype(c_grid_desc_m0_n0_m1_n1_m2_n2),
                                                   CElementwiseOperation,
                                                   Sequence<M0, N0, I1, I1, MPerThread, NPerThread>,
                                                   CThreadTransferSrcDstAccessOrder,
                                                   CThreadTransferSrcDstVectorDim,
                                                   CThreadTransferDstScalarPerVector,
                                                   CGlobalMemoryDataOperation,
                                                   1,
                                                   true>{
                    c_grid_desc_m0_n0_m1_n1_m2_n2,
                    make_multi_index(m_thread_data_on_grid_idx[I0],
                                     n_thread_data_on_grid_idx[I0],
                                     m_thread_data_on_grid_idx[I1],
                                     n_thread_data_on_grid_idx[I1],
                                     m_thread_data_on_grid_idx[I2],
                                     n_thread_data_on_grid_idx[I2]),
                    c_element_op};

            c_thread_copy.Run(c_thread_desc_m0_n0_m1_n1_m2_n2,
                              make_tuple(I0, I0, I0, I0, I0, I0),
                              c_thread_buf,
                              c_grid_desc_m0_n0_m1_n1_m2_n2,
                              c_grid_buf);
        }
    }
};

} // namespace ck
