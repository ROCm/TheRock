// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r2.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

namespace ck {

// GEMM:
//   input : A0[M, K], A1[M, K]
//   input : B0[N, K], B1[N, K]
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   D0, D1, ... and E have the same layout
template <typename AsDataType,
          typename BsDataType,
          typename AComputeDataType_,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          index_t NumGemmKPrefetchStage,
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
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched,
          PipelineVersion PipelineVer = PipelineVersion::v1,
          typename BComputeDataType_  = AComputeDataType_>
struct GridwiseGemmMultipleABD_xdl_cshuffle
{
    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // K1 should be Number<...>
    static constexpr auto AK1         = Number<AK1Value>{};
    static constexpr auto BK1         = Number<BK1Value>{};
    static constexpr auto AK0PerBlock = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0PerBlock = Number<KPerBlock / BK1Value>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>())>;

#if CK_GFX90A_DENORM_WORKAROUND
    using AComputeDataType =
        conditional_t<is_same_v<AComputeDataType_, ck::half_t>, ck::bhalf_t, AComputeDataType_>;
    using BComputeDataType =
        conditional_t<is_same_v<BComputeDataType_, ck::half_t>, ck::bhalf_t, BComputeDataType_>;
#else
    // Element data type is used in LDS and registers. ComputeDataType_ is inside mfma, eg tf32.
    using AElementDataType =
        conditional_t<is_same_v<AComputeDataType_, ck::tf32_t>, float, AComputeDataType_>;
    using BElementDataType =
        conditional_t<is_same_v<BComputeDataType_, ck::tf32_t>, float, BComputeDataType_>;
#endif

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(AK0PerBlock, Number<MPerBlock>{}, AK1),
            make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1, AK1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(BK0PerBlock, Number<NPerBlock>{}, BK1),
            make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1, BK1, I1));
    }

    __host__ __device__ static constexpr auto
    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    static constexpr auto MakeAsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using ADataType = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;

                return static_cast<const ADataType*>(nullptr);
            },
            Number<NumATensor>{});
    }

    static constexpr auto MakeBsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using BDataType = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;

                return static_cast<const BDataType*>(nullptr);
            },
            Number<NumBTensor>{});
    }

    // ck::Tuple<const D0DataType*, const D1DataType*, ...>
    static constexpr auto MakeDsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1, BK1);

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        // LDS allocation for C shuffle in LDS
        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

        constexpr auto c_block_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();

        return math::max(a_block_space_size_aligned * sizeof(AElementDataType) +
                             b_block_space_size_aligned * sizeof(BElementDataType),
                         c_block_size * sizeof(CShuffleDataType));
    }

    // A desc for source in blockwise copy
    template <typename AGridDesc_M_K>
    __host__ __device__ static constexpr auto
    MakeDefaultAGridDescriptor_AK0_M_AK1(const AGridDesc_M_K& a_grid_desc_m_k)
    {
        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);

        const auto AK0 = K / AK1;

        return transform_tensor_descriptor(a_grid_desc_m_k,
                                           make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                      make_pass_through_transform(M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename AsGridDesc_M_K>
    __host__ __device__ static constexpr auto
    MakeDefaultAsGridDescriptor_AK0_M_AK1(const AsGridDesc_M_K& as_grid_desc_m_k)
    {
        return generate_tuple(
            [&](auto i) { return MakeDefaultAGridDescriptor_AK0_M_AK1(as_grid_desc_m_k[i]); },
            Number<NumATensor>{});
    }

    // B desc for source in blockwise copy
    template <typename BGridDesc_N_K>
    __host__ __device__ static constexpr auto
    MakeDefaultBGridDescriptor_BK0_N_BK1(const BGridDesc_N_K& b_grid_desc_n_k)
    {
        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = b_grid_desc_n_k.GetLength(I1);

        const auto BK0 = K / BK1;

        return transform_tensor_descriptor(b_grid_desc_n_k,
                                           make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                      make_pass_through_transform(N)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename BsGridDesc_N_K>
    __host__ __device__ static constexpr auto
    MakeDefaultBsGridDescriptor_BK0_N_BK1(const BsGridDesc_N_K& bs_grid_desc_n_k)
    {
        return generate_tuple(
            [&](auto i) { return MakeDefaultBGridDescriptor_BK0_N_BK1(bs_grid_desc_n_k[i]); },
            Number<NumBTensor>{});
    }

    // E desc for destination in blockwise copy
    template <typename EGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const EGridDesc_M_N& e_grid_desc_m_n)
    {
        const auto M = e_grid_desc_m_n.GetLength(I0);
        const auto N = e_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        const auto e_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            e_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return e_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // Ds desc for source in blockwise copy
    template <typename DsGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const DsGridDesc_M_N& ds_grid_desc_m_n)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(ds_grid_desc_m_n[i]);
            },
            Number<NumDTensor>{});
    }

    // return block_id to E matrix tile idx (m0, n0) mapping
    template <typename EGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2ETileMap(const EGridDesc_M_N& e_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, EGridDesc_M_N>(
            e_grid_desc_m_n);
    }

    IS_VALID_COMPILATION_PARAMETER_IMPL(EDataType)

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename AsGridDesc_M_K,
              typename BsGridDesc_N_K,
              typename DsGridDesc_M_N,
              typename EGridDesc_M_N,
              typename Block2ETileMap>
    __host__ __device__ static constexpr bool CheckValidity(const AsGridDesc_M_K& as_grid_desc_m_k,
                                                            const BsGridDesc_N_K& bs_grid_desc_n_k,
                                                            const DsGridDesc_M_N& ds_grid_desc_m_n,
                                                            const EGridDesc_M_N& e_grid_desc_m_n,
                                                            const Block2ETileMap& block_2_etile_map)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        static_assert(KPerBlock % AK1Value == 0 && KPerBlock % BK1Value == 0,
                      "KPerBlock must be divisible by AK1Value and BK1Value!");

        const auto M  = as_grid_desc_m_k[I0].GetLength(I0);
        const auto N  = bs_grid_desc_n_k[I0].GetLength(I0);
        const auto AK = as_grid_desc_m_k[I0].GetLength(I1);
        const auto BK = bs_grid_desc_n_k[I0].GetLength(I1);

        // check consistency of desc
        if(!(M == e_grid_desc_m_n.GetLength(I0) && N == e_grid_desc_m_n.GetLength(I1) && AK == BK))
        {
            return false;
        }

        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        bool valid = true;
        static_for<0, NumATensor, 1>{}([&](auto i) {
            using ADataType = remove_cvref_t<tuple_element_t<i.value, AsDataType>>;
            valid =
                valid && (as_grid_desc_m_k[i].GetElementSpaceSize() * sizeof(ADataType) <= TwoGB);
            valid = valid && (M == as_grid_desc_m_k[i].GetLength(I0) &&
                              AK == as_grid_desc_m_k[i].GetLength(I1));
        });

        static_for<0, NumBTensor, 1>{}([&](auto i) {
            using BDataType = remove_cvref_t<tuple_element_t<i.value, BsDataType>>;
            valid =
                valid && (bs_grid_desc_n_k[i].GetElementSpaceSize() * sizeof(BDataType) <= TwoGB);
            valid = valid && (N == bs_grid_desc_n_k[i].GetLength(I0) &&
                              BK == bs_grid_desc_n_k[i].GetLength(I1));
        });

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            valid = valid && (M == ds_grid_desc_m_n[i].GetLength(I0) &&
                              N == ds_grid_desc_m_n[i].GetLength(I1));
        });

        if(!valid)
        {
            return false;
        }

        // check tile size
        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && AK % KPerBlock == 0))
        {
            return false;
        }

        // check gridwise gemm pipeline
        const auto num_k_loop = AK / KPerBlock;

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        // check block-to-E-tile
        if(!block_2_etile_map.CheckValidity(e_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        // check tensor size: cannot be larger than 2GB each

        if(!(e_grid_desc_m_n.GetElementSpaceSize() * sizeof(EDataType) <= TwoGB))
        {
            return false;
        }

        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    using AsGridPointer = decltype(MakeAsGridPointer());
    using BsGridPointer = decltype(MakeBsGridPointer());
    using DsGridPointer = decltype(MakeDsGridPointer());

    template <typename ALayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto
    MakeAGridDescriptor_M_K(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        constexpr auto matrix_padder =
            ck::tensor_operation::device::MatrixPadder<GemmSpec, index_t, index_t, index_t>{
                MPerBlock, NPerBlock, KPerBlock};

        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA));
            }
        }();

        return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
    }

    template <typename AsLayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto MakeAsGridDescriptor_M_K(
#ifdef CK_CODE_GEN_RTC
        const ck::Array<index_t, NumATensor>& MRaws,
        const ck::Array<index_t, NumATensor>& KRaws,
        const ck::Array<index_t, NumATensor>& AsStride
#else
        const std::array<index_t, NumATensor>& MRaws,
        const std::array<index_t, NumATensor>& KRaws,
        const std::array<index_t, NumATensor>& AsStride
#endif
    )
    {
        return generate_tuple(
            [&](auto i) {
                using ALayout = remove_cvref_t<tuple_element_t<i.value, AsLayout>>;

                return MakeAGridDescriptor_M_K<ALayout, GemmSpec>(MRaws[i], KRaws[i], AsStride[i]);
            },
            Number<NumATensor>{});
    }

    template <typename BLayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto
    MakeBGridDescriptor_N_K(const index_t NRaw, const index_t KRaw, const index_t StrideB)
    {
        constexpr auto matrix_padder =
            ck::tensor_operation::device::MatrixPadder<GemmSpec, index_t, index_t, index_t>{
                MPerBlock, NPerBlock, KPerBlock};

        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
    }

    template <typename BsLayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto MakeBsGridDescriptor_N_K(
#ifdef CK_CODE_GEN_RTC
        const ck::Array<index_t, NumBTensor>& NRaws,
        const ck::Array<index_t, NumBTensor>& KRaws,
        const ck::Array<index_t, NumBTensor>& BsStride
#else
        const std::array<index_t, NumBTensor>& NRaws,
        const std::array<index_t, NumBTensor>& KRaws,
        const std::array<index_t, NumBTensor>& BsStride
#endif
    )
    {
        return generate_tuple(
            [&](auto i) {
                using BLayout = remove_cvref_t<tuple_element_t<i.value, BsLayout>>;

                return MakeBGridDescriptor_N_K<BLayout, GemmSpec>(NRaws[i], KRaws[i], BsStride[i]);
            },
            Number<NumBTensor>{});
    }

    template <typename ELayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto
    MakeEGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideE)
    {
        constexpr auto matrix_padder =
            ck::tensor_operation::device::MatrixPadder<GemmSpec, index_t, index_t, index_t>{
                MPerBlock, NPerBlock, KPerBlock};
        const auto e_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideE, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideE));
            }
        }();

        return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
    }

    template <typename DsLayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto MakeDsGridDescriptor_M_N(
#ifdef CK_CODE_GEN_RTC
        const ck::Array<index_t, NumDTensor>& MRaws,
        const ck::Array<index_t, NumDTensor>& NRaws,
        const ck::Array<index_t, NumDTensor>& DsStride
#else
        const std::array<index_t, NumDTensor>& MRaws,
        const std::array<index_t, NumDTensor>& NRaws,
        const std::array<index_t, NumDTensor>& DsStride
#endif
    )
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return MakeEGridDescriptor_M_N<DLayout, GemmSpec>(MRaws[i], NRaws[i], DsStride[i]);
            },
            Number<NumDTensor>{});
    }

    __device__ __host__ static constexpr auto GetMPerBlock() { return MPerBlock; }

    template <bool HasMainKBlockLoop,
              typename AsGridDesc_AK0_M_AK1,
              typename BsGridDesc_BK0_N_BK1,
              typename DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename Block2ETileMap>
    __device__ static void Run(AsGridPointer p_as_grid,
                               BsGridPointer p_bs_grid,
                               DsGridPointer p_ds_grid,
                               EDataType* __restrict__ p_e_grid,
                               void* __restrict__ p_shared,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const CDEElementwiseOperation& cde_element_op,
                               const AsGridDesc_AK0_M_AK1 as_grid_desc_ak0_m_ak1,
                               const BsGridDesc_BK0_N_BK1 bs_grid_desc_bk0_n_bk1,
                               const DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   ds_grid_desc_mblock_mperblock_nblock_nperblock,
                               const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   e_grid_desc_mblock_mperblock_nblock_nperblock,
                               const Block2ETileMap& block_2_etile_map)
    {
        const auto as_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_as_grid[i], as_grid_desc_ak0_m_ak1[i].GetElementSpaceSize());
            },
            Number<NumATensor>{});

        const auto bs_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_bs_grid[i], bs_grid_desc_bk0_n_bk1[i].GetElementSpaceSize());
            },
            Number<NumBTensor>{});

        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[i],
                    ds_grid_desc_mblock_mperblock_nblock_nperblock[i].GetElementSpaceSize());
            },
            Number<NumDTensor>{});

        auto e_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e_grid, e_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_etile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_etile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }
        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1, BK1);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        const auto idx_as_block_begin =
            generate_tuple([&](auto) { return make_multi_index(0, m_block_data_idx_on_grid, 0); },
                           Number<NumATensor>{});

        auto a_blockwise_copy = ThreadGroupTensorSliceTransfer_v7r2<
            ThisThreadBlock,
            AsDataType,
            Tuple<AElementDataType>,
            decltype(as_grid_desc_ak0_m_ak1),
            decltype(tie(a_block_desc_ak0_m_ak1)),
            AElementwiseOperation,
            Sequence<static_cast<index_t>(InMemoryDataOperationEnum::Set)>,
            Sequence<AK0PerBlock, MPerBlock, AK1>,
            ABlockTransferThreadClusterLengths_AK0_M_AK1,
            ABlockTransferThreadClusterArrangeOrder,
            ABlockTransferSrcAccessOrder,
            Sequence<1, 0, 2>,
            ABlockTransferSrcVectorDim,
            2,
            ABlockTransferSrcScalarPerVector,
            ABlockTransferDstScalarPerVector_AK1,
            uniform_sequence_gen_t<NumATensor, AThreadTransferSrcResetCoordinateAfterRun>,
            Sequence<true>>{as_grid_desc_ak0_m_ak1,
                            idx_as_block_begin,
                            tie(a_block_desc_ak0_m_ak1),
                            make_tuple(make_multi_index(0, 0, 0)),
                            a_element_op};

        const auto idx_bs_block_begin =
            generate_tuple([&](auto) { return make_multi_index(0, n_block_data_idx_on_grid, 0); },
                           Number<NumBTensor>{});

        auto b_blockwise_copy = ThreadGroupTensorSliceTransfer_v7r2<
            ThisThreadBlock,
            BsDataType,
            Tuple<BElementDataType>,
            decltype(bs_grid_desc_bk0_n_bk1),
            decltype(tie(b_block_desc_bk0_n_bk1)),
            BElementwiseOperation,
            Sequence<static_cast<index_t>(InMemoryDataOperationEnum::Set)>,
            Sequence<BK0PerBlock, NPerBlock, BK1>,
            BBlockTransferThreadClusterLengths_BK0_N_BK1,
            BBlockTransferThreadClusterArrangeOrder,
            BBlockTransferSrcAccessOrder,
            Sequence<1, 0, 2>,
            BBlockTransferSrcVectorDim,
            2,
            BBlockTransferSrcScalarPerVector,
            BBlockTransferDstScalarPerVector_BK1,
            uniform_sequence_gen_t<NumBTensor, BThreadTransferSrcResetCoordinateAfterRun>,
            Sequence<true>>{bs_grid_desc_bk0_n_bk1,
                            idx_bs_block_begin,
                            tie(b_block_desc_bk0_n_bk1),
                            make_tuple(make_multi_index(0, 0, 0)),
                            b_element_op};

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check
        constexpr auto lcm_AK1_BK1 = math::lcm(AK1, BK1);
        constexpr bool is_single_rate_mfma =
            (((is_same<AComputeDataType_, half_t>::value ||
               is_same<AComputeDataType_, bhalf_t>::value) &&
              lcm_AK1_BK1 <= 4) ||
             (is_same<AComputeDataType_, int8_t>::value && lcm_AK1_BK1 <= 8) ||
             ((is_same<AComputeDataType_, f8_t>::value ||
               is_same<AComputeDataType_, bf8_t>::value) &&
              lcm_AK1_BK1 < 32))
                ? true
                : false;
        static constexpr auto is_scale_mfma = false;
        constexpr index_t KPack             = math::max(lcm_AK1_BK1,
                                            MfmaSelector<AComputeDataType_,
                                                                     MPerXdl,
                                                                     NPerXdl,
                                                                     BComputeDataType_,
                                                                     is_single_rate_mfma,
                                                                     is_scale_mfma>::selected_mfma.k_per_blk);

        auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
            BlockSize,
            AElementDataType,
            BElementDataType,
            AccDataType,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_block_desc_bk0_n_bk1),
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            LoopSched,
            AComputeDataType_,
            BComputeDataType_>();

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<AElementDataType*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<BElementDataType*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1, 0, 0);

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (as_grid_desc_ak0_m_ak1[I0].GetLength(I0) * as_grid_desc_ak0_m_ak1[I0].GetLength(I2)) /
            KPerBlock);

        // gridwise GEMM pipeline
        const auto gridwise_gemm_pipeline =
            GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>();

        gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(as_grid_desc_ak0_m_ak1,
                                                               a_block_desc_ak0_m_ak1,
                                                               a_blockwise_copy,
                                                               as_grid_buf,
                                                               a_block_buf,
                                                               a_block_slice_copy_step,
                                                               bs_grid_desc_bk0_n_bk1,
                                                               b_block_desc_bk0_n_bk1,
                                                               b_blockwise_copy,
                                                               bs_grid_buf,
                                                               b_block_buf,
                                                               b_block_slice_copy_step,
                                                               blockwise_gemm,
                                                               c_thread_buf,
                                                               num_k_block_main_loop);

        // shuffle C and write out
        {
            static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                              NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
            constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

            // TODO: hacky, fix it!
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

            // TODO: hacky, fix it!
            // c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp is only used to get lengths
            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

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
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

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

            // tuple of reference to C/Ds tensor descriptors
            const auto c_ds_desc_refs = concat_tuple_of_reference(
                tie(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                generate_tie([&](auto i) -> const auto& // return type should be reference
                             { return ds_grid_desc_mblock_mperblock_nblock_nperblock[i]; },
                             Number<NumDTensor>{}));

            // tuple of reference to C/Ds tensor descriptors
            const auto c_ds_buf_refs = concat_tuple_of_reference(
                tie(c_shuffle_block_buf),
                generate_tie([&](auto i) -> const auto& // return type should be reference
                             { return ds_grid_buf[i]; },
                             Number<NumDTensor>{}));

            // tuple of starting index of C/Ds blockwise copy
            const auto idx_c_ds_block_begin = container_concat(
                make_tuple(make_multi_index(0, 0, 0, 0)),
                generate_tuple(
                    [&](auto) {
                        return make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0);
                    },
                    Number<NumDTensor>{}));

            // blockwise copy C/D/E between LDS and global
            auto cde_block_copy_lds_and_global = ThreadGroupTensorSliceTransfer_v7r2<
                ThisThreadBlock,
                decltype(container_concat(make_tuple(CShuffleDataType{}), DsDataType{})),
                Tuple<EDataType>,
                decltype(c_ds_desc_refs),
                decltype(tie(e_grid_desc_mblock_mperblock_nblock_nperblock)),
                CDEElementwiseOperation,
                Sequence<static_cast<index_t>(EGlobalMemoryDataOperation)>, // FIXME: make Sequence
                                                                            // support arbitray type
                Sequence<1,
                         CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                         1,
                         CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                Sequence<0, 1, 2, 3>, // typename SrcDimAccessOrder,
                Sequence<0, 1, 2, 3>, // typename DstDimAccessOrder,
                3,                    // index_t SrcVectorDim,
                3,                    // index_t DstVectorDim,
                CDEShuffleBlockTransferScalarPerVector_NPerBlock,
                CDEShuffleBlockTransferScalarPerVector_NPerBlock,
                sequence_merge_t<
                    Sequence<true>,
                    uniform_sequence_gen_t<NumDTensor,
                                           false>>, // ThreadTransferSrcResetCoordinateAfterRunFlags
                Sequence<false>>                    // ThreadTransferDstResetCoordinateAfterRunFlags
                {c_ds_desc_refs,
                 idx_c_ds_block_begin,
                 tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                 make_tuple(make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0)),
                 cde_element_op};

            // space filling curve for threadwise C in VGPR before shuffle
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

            // space filling curve for shuffled blockwise C/D/E
            constexpr auto sfc_cde_block =
                SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                           1,
                                           CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_cde_block.GetNumOfAccess(), "wrong!");

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
                cde_block_copy_lds_and_global.Run(
                    c_ds_desc_refs,
                    c_ds_buf_refs,
                    tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                    tie(e_grid_buf));

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto cde_lds_and_global_step =
                        sfc_cde_block.GetForwardStep(access_id);

                    // move on Ds
                    static_for<0, NumDTensor, 1>{}([&](auto i) {
                        cde_block_copy_lds_and_global.MoveSrcSliceWindow(
                            c_ds_desc_refs, i + I1, cde_lds_and_global_step);
                    });

                    // move on E
                    cde_block_copy_lds_and_global.MoveDstSliceWindow(
                        tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                        I0,
                        cde_lds_and_global_step);
                }
            });
        }
    }

    template <bool HasMainKBlockLoop,
              GemmSpecialization GemmSpec,
              typename AsLayout,
              typename BsLayout,
              typename DsLayout,
              typename ELayout,
              typename Block2ETileMap>
    __device__ static void Run(AsGridPointer p_as_grid,
                               BsGridPointer p_bs_grid,
                               DsGridPointer p_ds_grid,
                               void* __restrict__ p_e_grid_,
                               void* __restrict__ p_shared,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const CDEElementwiseOperation& cde_element_op,
                               const index_t M,
                               const index_t N,
                               const index_t K,
#ifdef CK_CODE_GEN_RTC
                               const ck::Array<index_t, NumATensor> StrideAs,
                               const ck::Array<index_t, NumBTensor> StrideBs,
                               const ck::Array<index_t, NumDTensor> StrideDs,
#else
                               const std::array<index_t, NumATensor> StrideAs,
                               const std::array<index_t, NumBTensor> StrideBs,
                               const std::array<index_t, NumDTensor> StrideDs,
#endif
                               const index_t StrideE,
                               const Block2ETileMap& block_2_etile_map)
    {
        using AsGridDesc_M_K =
            remove_cvref_t<decltype(MakeAsGridDescriptor_M_K<AsLayout, GemmSpec>({}, {}, {}))>;
        using BsGridDesc_N_K =
            remove_cvref_t<decltype(MakeBsGridDescriptor_N_K<BsLayout, GemmSpec>({}, {}, {}))>;
        using DsGridDesc_M_N =
            remove_cvref_t<decltype(MakeDsGridDescriptor_M_N<DsLayout, GemmSpec>({}, {}, {}))>;

        const auto p_e_grid = reinterpret_cast<EDataType*>(p_e_grid_);

        AsGridDesc_M_K as_grid_desc_m_k;
        BsGridDesc_N_K bs_grid_desc_n_k;
        DsGridDesc_M_N ds_grid_desc_m_n;

        static_for<0, NumATensor, 1>{}([&](auto j) {
            using ALayout = remove_cvref_t<tuple_element_t<j.value, AsLayout>>;

            as_grid_desc_m_k(j) = MakeAGridDescriptor_M_K<ALayout, GemmSpec>(M, K, StrideAs[j]);
        });

        static_for<0, NumBTensor, 1>{}([&](auto j) {
            using BLayout = remove_cvref_t<tuple_element_t<j.value, BsLayout>>;

            bs_grid_desc_n_k(j) = MakeBGridDescriptor_N_K<BLayout, GemmSpec>(N, K, StrideBs[j]);
        });

        static_for<0, NumDTensor, 1>{}([&](auto j) {
            using DLayout = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;

            ds_grid_desc_m_n(j) = MakeEGridDescriptor_M_N<DLayout, GemmSpec>(M, N, StrideDs[j]);
        });

        const auto e_grid_desc_m_n = MakeEGridDescriptor_M_N<ELayout, GemmSpec>(M, N, StrideE);

        // tensor descriptors for block/thread-wise copy
        const auto as_grid_desc_ak0_m_ak1 = MakeDefaultAsGridDescriptor_AK0_M_AK1(as_grid_desc_m_k);

        const auto bs_grid_desc_bk0_n_bk1 = MakeDefaultBsGridDescriptor_BK0_N_BK1(bs_grid_desc_n_k);

        const auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(ds_grid_desc_m_n);

        const auto e_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(e_grid_desc_m_n);

        Run<HasMainKBlockLoop>(p_as_grid,
                               p_bs_grid,
                               p_ds_grid,
                               p_e_grid,
                               p_shared,
                               a_element_op,
                               b_element_op,
                               cde_element_op,
                               as_grid_desc_ak0_m_ak1,
                               bs_grid_desc_bk0_n_bk1,
                               ds_grid_desc_mblock_mperblock_nblock_nperblock,
                               e_grid_desc_mblock_mperblock_nblock_nperblock,
                               block_2_etile_map);
    }
};

} // namespace ck
