// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_layernorm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_gemm_multiple_d_welford_first_half_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_welford_second_half_layernorm2d.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {

template <typename GridwiseGemmWelford,
          typename ABDataType,
          typename DsPointer,
          typename EMeanVarDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename MeanVarGridDescriptor_MBlock_MPerBlock_NBlock,
          typename CountGridDescriptor_MBlock_MPerBlock_NBlock,
          typename Block2ETileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_multiple_d_welford_first_half_xdl_cshuffle(
        const ABDataType* __restrict__ p_a_grid,
        const ABDataType* __restrict__ p_b_grid,
        DsPointer p_ds_grid,
        EMeanVarDataType* __restrict__ p_e_grid,
        EMeanVarDataType* __restrict__ p_welford_mean_grid,
        EMeanVarDataType* __restrict__ p_welford_var_grid,
        int32_t* __restrict__ p_welford_count_grid,
        const AElementwiseOperation a_element_op,
        const BElementwiseOperation b_element_op,
        const CDEElementwiseOperation cde_element_op,
        const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
        const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
        const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock,
        const MeanVarGridDescriptor_MBlock_MPerBlock_NBlock
            mean_var_grid_desc_mblock_mperblock_nblock,
        const CountGridDescriptor_MBlock_MPerBlock_NBlock count_grid_desc_mblock_mperblock_nblock,
        const Block2ETileMap block_2_etile_map,
        index_t NRaw)
{
#if defined(__gfx9__) || defined(__gfx12__)
    if constexpr(GridwiseGemmWelford::template IsValidCompilationParameter<>())
    {
        __shared__ char p_shared[GridwiseGemmWelford::GetSharedMemoryNumberOfByte()];

        GridwiseGemmWelford::template Run<HasMainKBlockLoop>(
            p_a_grid,
            p_b_grid,
            p_ds_grid,
            p_e_grid,
            p_welford_mean_grid,
            p_welford_var_grid,
            p_welford_count_grid,
            p_shared,
            a_element_op,
            b_element_op,
            cde_element_op,
            a_grid_desc_ak0_m_ak1,
            b_grid_desc_bk0_n_bk1,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            e_grid_desc_mblock_mperblock_nblock_nperblock,
            mean_var_grid_desc_mblock_mperblock_nblock,
            count_grid_desc_mblock_mperblock_nblock,
            block_2_etile_map,
            NRaw);
    }
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = p_welford_mean_grid;
    ignore = p_welford_var_grid;
    ignore = p_welford_count_grid;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = mean_var_grid_desc_mblock_mperblock_nblock;
    ignore = count_grid_desc_mblock_mperblock_nblock;
    ignore = block_2_etile_map;
    ignore = NRaw;
#endif
}

template <typename GridwiseWelfordLayernorm,
          typename EMeanVarDataType,
          typename HDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename ComputeDataType,
          typename EHGridDesc_M_N,
          typename LayernormMeanVarGridDesc_M_NBlock,
          typename LayernormCountGridDesc_M_NBlock,
          typename GammaBetaGridDesc_N,
          typename HElementwiseOperation>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_welford_layernorm2d_second_half(
        const EMeanVarDataType* __restrict__ p_e_grid,
        const EMeanVarDataType* __restrict__ p_in_welford_mean_grid,
        const EMeanVarDataType* __restrict__ p_in_welford_var_grid,
        const int32_t* __restrict__ p_in_welford_count_grid,
        const GammaDataType* __restrict__ p_gamma_grid,
        const BetaDataType* __restrict__ p_beta_grid,
        HDataType* __restrict__ p_h_grid,
        const EHGridDesc_M_N e_grid_desc_m_n,
        const EHGridDesc_M_N h_grid_desc_m_n,
        const LayernormMeanVarGridDesc_M_NBlock mean_var_grid_desc_m_nblock,
        const LayernormCountGridDesc_M_NBlock count_grid_desc_m_nblock,
        const GammaBetaGridDesc_N gamma_grid_desc_n,
        const GammaBetaGridDesc_N beta_grid_desc_n,
        index_t numMeanVarCountBlockTileIteration_N,
        index_t NBlockClusterLength,
        ComputeDataType epsilon,
        HElementwiseOperation h_element_op)
{
    GridwiseWelfordLayernorm::Run(p_e_grid,
                                  p_in_welford_mean_grid,
                                  p_in_welford_var_grid,
                                  p_in_welford_count_grid,
                                  p_gamma_grid,
                                  p_beta_grid,
                                  p_h_grid,
                                  e_grid_desc_m_n,
                                  h_grid_desc_m_n,
                                  mean_var_grid_desc_m_nblock,
                                  count_grid_desc_m_nblock,
                                  gamma_grid_desc_n,
                                  beta_grid_desc_n,
                                  numMeanVarCountBlockTileIteration_N,
                                  NBlockClusterLength,
                                  epsilon,
                                  h_element_op);
}

} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

// GEMM:
//   input : A[M, K]
//   input : B[N, K]
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   output : H[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
//   H = layernorm(E)
// Assume:
//   D0, D1, ... and E have the same layout
//   Calculate mean & variance along N dimension in layernorm(E)
template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename HLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EMeanVarDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename HDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename HElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename PostShuffleThreadClusterSize_M_N,
          index_t PostShuffleScalarPerVector,
          typename LayernormThreadClusterSize_M_N,
          index_t LayernormThreadSliceSize_M,
          LoopScheduler LoopSched     = make_default_loop_scheduler(),
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct DeviceGemmMultipleDLayernorm_Xdl_CShuffle
    : public DeviceGemmMultipleDLayernorm<ALayout,
                                          BLayout,
                                          DsLayout,
                                          HLayout,
                                          ADataType,
                                          BDataType,
                                          DsDataType,
                                          GammaDataType,
                                          BetaDataType,
                                          HDataType,
                                          AElementwiseOperation,
                                          BElementwiseOperation,
                                          CDEElementwiseOperation,
                                          HElementwiseOperation>
{
    // EDataType, MeanDataType and VarDataType must be the same.
    // eg. M, N, K = [1, 1, 1],
    // in case of layernorm, divisor = 1 / sqrt(var + 1e-5) = 316.227783
    // if (x - mean) != 0, (x - mean) * divisor * gamma might be too large
    // However, (x - mean) * divisor * gamma should be 0 in this case

    using DeviceOp = DeviceGemmMultipleDLayernorm_Xdl_CShuffle;
    using ELayout  = HLayout;

    GET_NXDL_PER_WAVE_IMPL
    static constexpr auto NXdlPerWave64 = GetNXdlPerWave<true>();
    static constexpr auto NXdlPerWave32 = GetNXdlPerWave<false>();

    static constexpr index_t NumDTensor                  = DsDataType::Size();
    static constexpr index_t LayernormHDstVectorSize     = PostShuffleScalarPerVector;
    static constexpr index_t LayernormGammaSrcVectorSize = PostShuffleScalarPerVector;
    static constexpr index_t LayernormBetaSrcVectorSize  = PostShuffleScalarPerVector;
    static constexpr index_t LayernormESrcVectorSize     = PostShuffleScalarPerVector;
    static constexpr index_t LayernormThreadSliceSize_N  = PostShuffleScalarPerVector;
    using LayernormBlockTileSize_M_N =
        Sequence<LayernormThreadClusterSize_M_N::At(0) * LayernormThreadSliceSize_M,
                 LayernormThreadClusterSize_M_N::At(1) * LayernormThreadSliceSize_N>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    static auto MakeAGridDescriptor_M_K(index_t MRaw, index_t KRaw, index_t StrideA)
    {
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

    static auto MakeBGridDescriptor_N_K(index_t KRaw, index_t NRaw, index_t StrideB)
    {
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

    template <typename DoPads, index_t MPerTile, index_t NPerTile>
    static auto MakeEHGridDescriptor_M_N(index_t M, index_t N, index_t Stride)
    {
        // Only support row major for E and H
        const auto grid_desc_m_n =
            make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(Stride, I1));
        return PadTensorDescriptor(grid_desc_m_n, make_tuple(MPerTile, NPerTile), DoPads{});
    }

    static auto MakeDsGridDescriptor_M_N(const std::array<index_t, NumDTensor>& MRaws,
                                         const std::array<index_t, NumDTensor>& NRaws,
                                         const std::array<index_t, NumDTensor>& DsStride)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                static_assert(is_same<tensor_layout::gemm::RowMajor, DLayout>::value);

                return DeviceOp::
                    MakeEHGridDescriptor_M_N<Sequence<true, true>, MPerBlock, NPerBlock>(
                        MRaws[i], NRaws[i], DsStride[i]);
            },
            Number<NumDTensor>{});
    }

    template <typename DoPads, index_t MPerTile, index_t NPerTile>
    static auto MakeMeanVarDescriptor_M_N(index_t M, index_t N)
    {
        const auto grid_desc_m_n =
            make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(N, I1));
        return PadTensorDescriptor(grid_desc_m_n, make_tuple(MPerTile, NPerTile), DoPads{});
    }

    template <typename DoPads, index_t MPerTile, index_t NPerTile>
    static auto MakeCountDescriptor_M_N(index_t M, index_t N)
    {
        // We will broadcast [N] to [M, N] in this descriptor
        // Hence, 1st stride is 0
        const auto grid_desc_m_n =
            make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I0, I1));
        return PadTensorDescriptor(grid_desc_m_n, make_tuple(MPerTile, NPerTile), DoPads{});
    }

    template <index_t XPerTile>
    static auto MakeDescriptor_X(index_t X)
    {
        const auto grid_desc_x = make_naive_tensor_descriptor_packed(make_tuple(X));
        return PadTensorDescriptor(grid_desc_x, make_tuple(XPerTile), Sequence<true>{});
    }

    using AGridDesc_M_K  = decltype(MakeAGridDescriptor_M_K(1, 1, 1));
    using BGridDesc_N_K  = decltype(MakeBGridDescriptor_N_K(1, 1, 1));
    using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({}, {}, {}))>;
    // We have to separate mean var descriptor for gemm and layernorm bacause of different grid
    // layout(different padding)
    using GemmMeanVarGridDesc_M_NBlock =
        decltype(MakeMeanVarDescriptor_M_N<Sequence<true, false>, MPerBlock, NPerBlock>(1, 1));

    using GemmCountGridDesc_M_NBlock =
        decltype(MakeCountDescriptor_M_N<Sequence<true, false>, MPerBlock, NPerBlock>(1, 1));

    using LayernormMeanVarGridDesc_M_NBlock =
        decltype(MakeMeanVarDescriptor_M_N<Sequence<true, true>,
                                           LayernormBlockTileSize_M_N::At(0),
                                           LayernormBlockTileSize_M_N::At(1)>(1, 1));

    using LayernormCountGridDesc_M_NBlock =
        decltype(MakeCountDescriptor_M_N<Sequence<true, true>,
                                         LayernormBlockTileSize_M_N::At(0),
                                         LayernormBlockTileSize_M_N::At(1)>(1, 1));

    using GammaBetaGridDesc_N = decltype(MakeDescriptor_X<LayernormBlockTileSize_M_N::At(1)>(1));
    using EHGridDesc_M_N = decltype(MakeEHGridDescriptor_M_N<Sequence<true, true>, 1, 1>(1, 1, 1));

    template <index_t NXdlPerWave_>
    using GridwiseGemmWelfordBase = GridwiseGemmMultipleDWelfordFirstHalf_xdl_cshuffle<
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EMeanVarDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        AGridDesc_M_K,
        BGridDesc_N_K,
        DsGridDesc_M_N,
        EHGridDesc_M_N,
        GemmMeanVarGridDesc_M_NBlock,
        GemmCountGridDesc_M_NBlock,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave_,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        PostShuffleThreadClusterSize_M_N,
        PostShuffleScalarPerVector,
        LoopSched,
        PipelineVer>;
    using GridwiseGemmWelford64 = GridwiseGemmWelfordBase<math::max(NXdlPerWave64, 1)>;
    using GridwiseGemmWelford32 = GridwiseGemmWelfordBase<NXdlPerWave32>;

    using Block2ETileMap = typename GridwiseGemmWelford64::DefaultBlock2ETileMap;

    using GridwiseWelfordLayernorm =
        GridwiseWelfordSecondHalfLayernorm2d<EMeanVarDataType,
                                             HDataType,
                                             GammaDataType,
                                             BetaDataType,
                                             AccDataType,
                                             EHGridDesc_M_N,
                                             LayernormMeanVarGridDesc_M_NBlock,
                                             LayernormCountGridDesc_M_NBlock,
                                             GammaBetaGridDesc_N,
                                             HElementwiseOperation,
                                             BlockSize,
                                             LayernormThreadClusterSize_M_N::At(I0),
                                             LayernormThreadClusterSize_M_N::At(I1),
                                             LayernormThreadSliceSize_M,
                                             LayernormThreadSliceSize_N,
                                             LayernormESrcVectorSize,
                                             LayernormHDstVectorSize,
                                             LayernormGammaSrcVectorSize,
                                             LayernormBetaSrcVectorSize>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a_grid,
                 const void* p_b_grid,
                 std::array<const void*, NumDTensor> p_ds_grid,
                 const void* p_gamma_grid,
                 const void* p_beta_grid,
                 void* p_h_grid,
                 index_t MRaw,
                 index_t NRaw,
                 index_t KRaw,
                 index_t StrideA,
                 index_t StrideB,
                 std::array<index_t, NumDTensor> StrideDs,
                 index_t StrideH,
                 double epsilon,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op,
                 HElementwiseOperation h_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a_grid)},
              p_b_grid_{static_cast<const BDataType*>(p_b_grid)},
              p_ds_grid_{},
              p_workspace_e_grid_{nullptr},
              p_workspace_mean_{nullptr},
              p_workspace_var_{nullptr},
              p_workspace_count_{nullptr},
              p_gamma_grid_{static_cast<const GammaDataType*>(p_gamma_grid)},
              p_beta_grid_{static_cast<const BetaDataType*>(p_beta_grid)},
              p_h_grid_{static_cast<HDataType*>(p_h_grid)},
              a_grid_desc_m_k_{DeviceOp::MakeAGridDescriptor_M_K(MRaw, KRaw, StrideA)},
              b_grid_desc_n_k_{DeviceOp::MakeBGridDescriptor_N_K(KRaw, NRaw, StrideB)},
              ds_grid_desc_m_n_{},
              gemm_e_grid_desc_m_n_{
                  DeviceOp::MakeEHGridDescriptor_M_N<Sequence<true, true>, MPerBlock, NPerBlock>(
                      MRaw, NRaw, StrideH)},
              layernorm_e_grid_desc_m_n_{
                  DeviceOp::MakeEHGridDescriptor_M_N<Sequence<true, true>,
                                                     LayernormBlockTileSize_M_N::At(0),
                                                     LayernormBlockTileSize_M_N::At(1)>(
                      MRaw, NRaw, StrideH)},
              gemm_mean_var_grid_desc_m_nblock_{},
              gemm_count_grid_desc_m_nblock_{},
              layernorm_mean_var_grid_desc_m_nblock_{},
              layernorm_count_grid_desc_m_nblock_{},
              gamma_grid_desc_n_{
                  DeviceOp::MakeDescriptor_X<LayernormBlockTileSize_M_N::At(1)>(NRaw)},
              beta_grid_desc_n_{
                  DeviceOp::MakeDescriptor_X<LayernormBlockTileSize_M_N::At(1)>(NRaw)},
              h_grid_desc_m_n_{
                  DeviceOp::MakeEHGridDescriptor_M_N<Sequence<true, true>,
                                                     LayernormBlockTileSize_M_N::At(0),
                                                     LayernormBlockTileSize_M_N::At(1)>(
                      MRaw, NRaw, StrideH)},
              a_grid_desc_ak0_m_ak1_{
                  GridwiseGemmWelford64::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k_)},
              b_grid_desc_bk0_n_bk1_{
                  GridwiseGemmWelford64::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k_)},
              block_2_etile_map_{
                  GridwiseGemmWelford64::MakeDefaultBlock2ETileMap(gemm_e_grid_desc_m_n_)},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              h_element_op_{h_element_op},
              MRaw_{MRaw},
              NRaw_{NRaw},
              KRaw_{KRaw},
              gemm_nblock_{math::integer_divide_ceil(NRaw, NPerBlock)},
              epsilon_{static_cast<AccDataType>(epsilon)}
        {
            // We don't need to pad in N dimension in gemm for mean/var/count. Set NPerTile 1.
            gemm_mean_var_grid_desc_m_nblock_ =
                DeviceOp::MakeMeanVarDescriptor_M_N<Sequence<true, false>, MPerBlock, 1>(
                    MRaw, gemm_nblock_);

            gemm_count_grid_desc_m_nblock_ =
                DeviceOp::MakeCountDescriptor_M_N<Sequence<true, false>, MPerBlock, 1>(
                    MRaw, gemm_nblock_);

            layernorm_mean_var_grid_desc_m_nblock_ =
                DeviceOp::MakeMeanVarDescriptor_M_N<Sequence<true, true>,
                                                    LayernormBlockTileSize_M_N::At(0),
                                                    LayernormBlockTileSize_M_N::At(1)>(
                    MRaw, gemm_nblock_);

            layernorm_count_grid_desc_m_nblock_ =
                DeviceOp::MakeCountDescriptor_M_N<Sequence<true, true>,
                                                  LayernormBlockTileSize_M_N::At(0),
                                                  LayernormBlockTileSize_M_N::At(1)>(MRaw,
                                                                                     gemm_nblock_);

            // populate pointer, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds_grid[i]);

                // D desc
                ds_grid_desc_m_n_(i) =
                    DeviceOp::MakeEHGridDescriptor_M_N<Sequence<true, true>, MPerBlock, NPerBlock>(
                        MRaw, NRaw, StrideDs[i]);
            });

            // populate desc for Ds/E/mean/var/count
            if(get_warp_size() == 64)
            {
                if constexpr(NXdlPerWave64 > 0)
                {
                    if(GridwiseGemmWelford64::CheckValidity(a_grid_desc_m_k_,
                                                            b_grid_desc_n_k_,
                                                            ds_grid_desc_m_n_,
                                                            gemm_e_grid_desc_m_n_,
                                                            block_2_etile_map_))
                    {
                        ds_grid_desc_mblock_mperblock_nblock_nperblock_ = GridwiseGemmWelford64::
                            MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                                ds_grid_desc_m_n_);

                        e_grid_desc_mblock_mperblock_nblock_nperblock_ = GridwiseGemmWelford64::
                            MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                                gemm_e_grid_desc_m_n_);

                        gemm_mean_var_grid_desc_mblock_mperblock_nblock_ = GridwiseGemmWelford64::
                            MakeMeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock(
                                gemm_mean_var_grid_desc_m_nblock_);

                        gemm_count_grid_desc_mblock_mperblock_nblock_ = GridwiseGemmWelford64::
                            MakeMeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock(
                                gemm_count_grid_desc_m_nblock_);
                    }
                }
            }
            else
            {
                if constexpr(NXdlPerWave32 > 0)
                {
                    if(GridwiseGemmWelford32::CheckValidity(a_grid_desc_m_k_,
                                                            b_grid_desc_n_k_,
                                                            ds_grid_desc_m_n_,
                                                            gemm_e_grid_desc_m_n_,
                                                            block_2_etile_map_))
                    {
                        ds_grid_desc_mblock_mperblock_nblock_nperblock_ = GridwiseGemmWelford32::
                            MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                                ds_grid_desc_m_n_);

                        e_grid_desc_mblock_mperblock_nblock_nperblock_ = GridwiseGemmWelford32::
                            MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                                gemm_e_grid_desc_m_n_);

                        gemm_mean_var_grid_desc_mblock_mperblock_nblock_ = GridwiseGemmWelford32::
                            MakeMeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock(
                                gemm_mean_var_grid_desc_m_nblock_);

                        gemm_count_grid_desc_mblock_mperblock_nblock_ = GridwiseGemmWelford32::
                            MakeMeanVarCountGridDescriptor_MBlock_MPerBlock_NBlock(
                                gemm_count_grid_desc_m_nblock_);
                    }
                }
            }
        }

        void Print() const
        {
            std::cout << "A[M, K]: " << a_grid_desc_m_k_ << std::endl;
            std::cout << "B[N, K]: " << b_grid_desc_n_k_ << std::endl;
            static_for<0, NumDTensor, 1>{}(
                [&](auto i) { std::cout << "Ds[M, N]: " << ds_grid_desc_m_n_[i] << std::endl; });
            std::cout << "E[M, N]: " << gemm_e_grid_desc_m_n_ << std::endl;
            std::cout << "H[M, N]: " << h_grid_desc_m_n_ << std::endl;
        }

        //  private:
        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemmWelford64::DsGridPointer p_ds_grid_;
        void* p_workspace_e_grid_;
        void* p_workspace_mean_;
        void* p_workspace_var_;
        void* p_workspace_count_;
        const GammaDataType* p_gamma_grid_;
        const BetaDataType* p_beta_grid_;
        HDataType* p_h_grid_;

        // tensor descriptors for problem definiton
        AGridDesc_M_K a_grid_desc_m_k_;
        BGridDesc_N_K b_grid_desc_n_k_;
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EHGridDesc_M_N gemm_e_grid_desc_m_n_;
        EHGridDesc_M_N layernorm_e_grid_desc_m_n_;
        GemmMeanVarGridDesc_M_NBlock gemm_mean_var_grid_desc_m_nblock_;
        GemmCountGridDesc_M_NBlock gemm_count_grid_desc_m_nblock_;
        LayernormMeanVarGridDesc_M_NBlock layernorm_mean_var_grid_desc_m_nblock_;
        LayernormCountGridDesc_M_NBlock layernorm_count_grid_desc_m_nblock_;
        GammaBetaGridDesc_N gamma_grid_desc_n_;
        GammaBetaGridDesc_N beta_grid_desc_n_;
        EHGridDesc_M_N h_grid_desc_m_n_;

        // tensor descriptors for block/thread-wise copy
        typename GridwiseGemmWelford64::DefaultAGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        typename GridwiseGemmWelford64::DefaultBGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        typename GridwiseGemmWelford64::DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            ds_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemmWelford64::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock_;
        typename GridwiseGemmWelford64::MeanVarGridDescriptor_MBlock_MPerBlock_NBlock
            gemm_mean_var_grid_desc_mblock_mperblock_nblock_;
        typename GridwiseGemmWelford64::CountGridDescriptor_MBlock_MPerBlock_NBlock
            gemm_count_grid_desc_mblock_mperblock_nblock_;

        // block-to-e-tile map
        Block2ETileMap block_2_etile_map_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
        HElementwiseOperation h_element_op_;

        index_t MRaw_;
        index_t NRaw_;
        index_t KRaw_;
        index_t gemm_nblock_;
        AccDataType epsilon_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;
        template <typename GridwiseGemmWelford>
        float RunImp(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float avg_time = 0;

            if(!GridwiseGemmWelford::CheckValidity(arg.a_grid_desc_m_k_,
                                                   arg.b_grid_desc_n_k_,
                                                   arg.ds_grid_desc_m_n_,
                                                   arg.gemm_e_grid_desc_m_n_,
                                                   arg.block_2_etile_map_))
            {
                throw std::runtime_error("wrong! GridwiseGemmWelford has invalid setting");
            }
            if(arg.p_workspace_e_grid_ == nullptr || arg.p_workspace_mean_ == nullptr ||
               arg.p_workspace_var_ == nullptr || arg.p_workspace_count_ == nullptr)
                throw std::runtime_error("wrong! WorkSpace pointer has not been set");

            index_t grid_size = arg.block_2_etile_map_.CalculateGridSize(arg.gemm_e_grid_desc_m_n_);

            const auto M = arg.h_grid_desc_m_n_.GetLength(I0);
            const auto N = arg.h_grid_desc_m_n_.GetLength(I1);
            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel_gemm_welford =
                    kernel_gemm_multiple_d_welford_first_half_xdl_cshuffle<
                        GridwiseGemmWelford,
                        ADataType, // TODO: distiguish A/B datatype
                        typename GridwiseGemmWelford::DsGridPointer,
                        EMeanVarDataType,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CDEElementwiseOperation,
                        typename GridwiseGemmWelford::DefaultAGridDesc_AK0_M_AK1,
                        typename GridwiseGemmWelford::DefaultBGridDesc_BK0_N_BK1,
                        typename GridwiseGemmWelford::
                            DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                        typename GridwiseGemmWelford::
                            EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                        typename GridwiseGemmWelford::MeanVarGridDescriptor_MBlock_MPerBlock_NBlock,
                        typename GridwiseGemmWelford::CountGridDescriptor_MBlock_MPerBlock_NBlock,
                        typename GridwiseGemmWelford::DefaultBlock2ETileMap,
                        has_main_loop>;

                const auto kernel_welford_layernorm =
                    kernel_welford_layernorm2d_second_half<GridwiseWelfordLayernorm,
                                                           EMeanVarDataType,
                                                           HDataType,
                                                           GammaDataType,
                                                           BetaDataType,
                                                           AccDataType,
                                                           EHGridDesc_M_N,
                                                           LayernormMeanVarGridDesc_M_NBlock,
                                                           LayernormCountGridDesc_M_NBlock,
                                                           GammaBetaGridDesc_N,
                                                           HElementwiseOperation>;

                avg_time +=
                    launch_and_time_kernel(stream_config,
                                           kernel_gemm_welford,
                                           dim3(grid_size),
                                           dim3(BlockSize),
                                           0,
                                           arg.p_a_grid_,
                                           arg.p_b_grid_,
                                           arg.p_ds_grid_,
                                           static_cast<EMeanVarDataType*>(arg.p_workspace_e_grid_),
                                           static_cast<EMeanVarDataType*>(arg.p_workspace_mean_),
                                           static_cast<EMeanVarDataType*>(arg.p_workspace_var_),
                                           static_cast<int32_t*>(arg.p_workspace_count_),
                                           arg.a_element_op_,
                                           arg.b_element_op_,
                                           arg.cde_element_op_,
                                           arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                           arg.gemm_mean_var_grid_desc_mblock_mperblock_nblock_,
                                           arg.gemm_count_grid_desc_mblock_mperblock_nblock_,
                                           arg.block_2_etile_map_,
                                           arg.NRaw_);

                index_t MBlockClusterLength =
                    math::integer_divide_ceil(M, LayernormBlockTileSize_M_N::At(0));
                index_t NBlockClusterLength =
                    math::integer_divide_ceil(N, LayernormBlockTileSize_M_N::At(1));
                grid_size = MBlockClusterLength * NBlockClusterLength;

                index_t numMeanVarCountBlockTileIteration_N = math::integer_divide_ceil(
                    arg.gemm_nblock_, LayernormThreadClusterSize_M_N::At(I1));

                avg_time += launch_and_time_kernel(
                    stream_config,
                    kernel_welford_layernorm,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    static_cast<EMeanVarDataType*>(arg.p_workspace_e_grid_),
                    static_cast<const EMeanVarDataType*>(arg.p_workspace_mean_),
                    static_cast<const EMeanVarDataType*>(arg.p_workspace_var_),
                    static_cast<const int32_t*>(arg.p_workspace_count_),
                    arg.p_gamma_grid_,
                    arg.p_beta_grid_,
                    arg.p_h_grid_,
                    arg.layernorm_e_grid_desc_m_n_,
                    arg.h_grid_desc_m_n_,
                    arg.layernorm_mean_var_grid_desc_m_nblock_,
                    arg.layernorm_count_grid_desc_m_nblock_,
                    arg.gamma_grid_desc_n_,
                    arg.beta_grid_desc_n_,
                    numMeanVarCountBlockTileIteration_N,
                    NBlockClusterLength,
                    arg.epsilon_,
                    arg.h_element_op_);

                return avg_time;
            };

            if(GridwiseGemmWelford::CalculateHasMainKBlockLoop(K))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }
        using GridwiseGemm32 = GridwiseGemmWelford32;
        using GridwiseGemm64 = GridwiseGemmWelford64;
        INVOKER_RUN_IMPL
        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    size_t GetWorkSpaceSize(const BaseArgument* pArg) const override
    {
        const Argument* pArg_ = dynamic_cast<const Argument*>(pArg);

        size_t workspace_size = 0;

        int gemm_welford_size = pArg_->MRaw_ * pArg_->gemm_nblock_;

        // workspace for welford intermediate mean
        workspace_size += gemm_welford_size * sizeof(EMeanVarDataType) + 64;

        // workspace for welford intermediate variance
        workspace_size += gemm_welford_size * sizeof(EMeanVarDataType) + 64;

        // workspace for welford intermediate count
        workspace_size += pArg_->gemm_nblock_ * sizeof(int32_t) + 64;

        if constexpr(!is_same_v<EMeanVarDataType, HDataType>)
            workspace_size += pArg_->MRaw_ * pArg_->NRaw_ * sizeof(EMeanVarDataType);

        return (workspace_size);
    };

    void SetWorkSpacePointer(BaseArgument* pArg,
                             void* p_workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        Argument* pArg_ = dynamic_cast<Argument*>(pArg);

        pArg_->p_workspace_ = p_workspace;

        int gemm_welford_size = pArg_->MRaw_ * pArg_->gemm_nblock_;

        // setup buffer used for intermediate welford mean
        pArg_->p_workspace_mean_ = static_cast<char*>(pArg_->p_workspace_);

        index_t mean_space_sz = gemm_welford_size * sizeof(EMeanVarDataType);
        mean_space_sz         = math::integer_least_multiple(mean_space_sz, 64);

        // setup buffer used for intermediate welford varirance
        pArg_->p_workspace_var_ = reinterpret_cast<char*>(pArg_->p_workspace_mean_) + mean_space_sz;

        index_t variance_space_sz = gemm_welford_size * sizeof(EMeanVarDataType);
        variance_space_sz         = math::integer_least_multiple(variance_space_sz, 64);

        // setup buffer used for intermediate welford count
        pArg_->p_workspace_count_ =
            reinterpret_cast<char*>(pArg_->p_workspace_var_) + variance_space_sz;

        index_t count_space_sz = gemm_welford_size * sizeof(int32_t);
        count_space_sz         = math::integer_least_multiple(count_space_sz, 64);

        if constexpr(!is_same_v<EMeanVarDataType, HDataType>)
            pArg_->p_workspace_e_grid_ =
                reinterpret_cast<char*>(pArg_->p_workspace_count_) + count_space_sz;
        else
            pArg_->p_workspace_e_grid_ = static_cast<void*>(pArg_->p_h_grid_);
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_xdl_wmma_supported<ADataType, BDataType, MPerXDL, NPerXDL>())
        {
            return false;
        }
        // check vector load/store
        {
            using Row = ck::tensor_layout::gemm::RowMajor;
            using Col = ck::tensor_layout::gemm::ColumnMajor;

            // check vector load of A
            if constexpr(is_same_v<ALayout, Row> && ABlockTransferSrcVectorDim == 2)
            {
                if(arg.KRaw_ % ABlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
            else if constexpr(is_same_v<ALayout, Col> && ABlockTransferSrcVectorDim == 1)
            {
                // FIXME: not rigorous
                if(arg.MRaw_ % ABlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            // check vector laod of B
            if constexpr(is_same_v<BLayout, Col> && BBlockTransferSrcVectorDim == 2)
            {
                if(arg.KRaw_ % BBlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
            else if constexpr(is_same_v<BLayout, Row> && BBlockTransferSrcVectorDim == 1)
            {
                // FIXME: not rigorous
                if(arg.NRaw_ % BBlockTransferSrcScalarPerVector != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }

            // check vector load of Ds
            // only support RowMajor for now
            bool all_valid = true;

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                if constexpr(!is_same_v<DLayout, Row>)
                {
                    all_valid = false;
                }
            });

            if(!all_valid)
            {
                return false;
            }

            // check vector store of E
            // E and H only support RowMajor for now
            if constexpr(is_same_v<ELayout, Row> && is_same_v<HLayout, Row>)
            {
                if(arg.NRaw_ % PostShuffleScalarPerVector != 0 ||
                   arg.NRaw_ % LayernormGammaSrcVectorSize != 0 ||
                   arg.NRaw_ % LayernormBetaSrcVectorSize != 0 ||
                   arg.NRaw_ % LayernormHDstVectorSize != 0)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        if(get_warp_size() == 64)
        {
            if constexpr(NXdlPerWave64 > 0)
            {
                return GridwiseGemmWelford64::CheckValidity(arg.a_grid_desc_m_k_,
                                                            arg.b_grid_desc_n_k_,
                                                            arg.ds_grid_desc_m_n_,
                                                            arg.gemm_e_grid_desc_m_n_,
                                                            arg.block_2_etile_map_);
            }
            else
            {
                return false;
            }
        }
        else
        {
            if constexpr(NXdlPerWave32 > 0)
            {
                return GridwiseGemmWelford32::CheckValidity(arg.a_grid_desc_m_k_,
                                                            arg.b_grid_desc_n_k_,
                                                            arg.ds_grid_desc_m_n_,
                                                            arg.gemm_e_grid_desc_m_n_,
                                                            arg.block_2_etile_map_);
            }
            else
            {
                return false;
            }
        }
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             std::array<const void*, NumDTensor> p_ds,
                             const void* p_gamma,
                             const void* p_beta,
                             void* p_h,
                             index_t MRaw,
                             index_t NRaw,
                             index_t KRaw,
                             index_t StrideA,
                             index_t StrideB,
                             std::array<index_t, NumDTensor> StrideDs,
                             index_t StrideH,
                             double epsilon,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op,
                             HElementwiseOperation h_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_gamma,
                        p_beta,
                        p_h,
                        MRaw,
                        NRaw,
                        KRaw,
                        StrideA,
                        StrideB,
                        StrideDs,
                        StrideH,
                        epsilon,
                        a_element_op,
                        b_element_op,
                        cde_element_op,
                        h_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      std::array<const void*, NumDTensor> p_ds,
                                                      const void* p_gamma,
                                                      const void* p_beta,
                                                      void* p_h,
                                                      index_t MRaw,
                                                      index_t NRaw,
                                                      index_t KRaw,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      std::array<index_t, NumDTensor> StrideDs,
                                                      index_t StrideH,
                                                      double epsilon,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CDEElementwiseOperation cde_element_op,
                                                      HElementwiseOperation h_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_gamma,
                                          p_beta,
                                          p_h,
                                          MRaw,
                                          NRaw,
                                          KRaw,
                                          StrideA,
                                          StrideB,
                                          StrideDs,
                                          StrideH,
                                          epsilon,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op,
                                          h_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<LoopScheduler, std::string> LoopSchedToString{
            {LoopScheduler::Default, "Default"}, {LoopScheduler::Interwave, "Interwave"}};

        std::map<PipelineVersion, std::string> PipelineVersionToString{{PipelineVersion::v1, "v1"},
                                                                       {PipelineVersion::v2, "v2"}};

        // clang-format off
        str << "DeviceGemmMultipleDLayernorm_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << getGemmSpecializationString(GemmSpec) << ", "
            << PostShuffleThreadClusterSize_M_N::At(I0) << ", "
            << PostShuffleThreadClusterSize_M_N::At(I1) << ", "
            << LayernormThreadClusterSize_M_N::At(I0) << ", "
            << LayernormThreadClusterSize_M_N::At(I1) << ", "
            << LayernormThreadSliceSize_M
            << ">"
            << " LoopScheduler: "
            << LoopSchedToString[LoopSched] << ", "
            << "PipelineVersion: "
            << PipelineVersionToString[PipelineVer];
        // clang-format on

        return str.str();
    }
}; // namespace device

} // namespace device
} // namespace tensor_operation
} // namespace ck
