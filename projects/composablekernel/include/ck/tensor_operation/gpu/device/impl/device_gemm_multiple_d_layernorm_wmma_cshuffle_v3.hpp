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
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_welford_second_half_layernorm2d.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename EMeanVarDataType,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_gemm_multiple_d_welford_first_half_wmma_cshuffle_v3(
        typename GridwiseGemm::Argument karg,
        EMeanVarDataType* __restrict__ p_welford_mean_grid,
        EMeanVarDataType* __restrict__ p_welford_var_grid,
        int32_t* __restrict__ p_welford_count_grid)
{
#if(defined(__gfx11__) || defined(__gfx12__))
#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using e_data_type = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(EGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<e_data_type, ck::half_t> ||
                    std::is_same_v<e_data_type, ck::bhalf_t>)))
    {
#endif
        constexpr index_t LDS_size = GridwiseGemm::template GetSharedMemoryNumberOfByte<
            typename GridwiseGemm::EpilogueWelfordCShuffle>();

        __shared__ char p_shared[LDS_size];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        auto epilogue_args = typename GridwiseGemm::EpilogueWelfordCShuffle(
            p_welford_mean_grid, p_welford_var_grid, p_welford_count_grid, karg.M, karg.N);

        GridwiseGemm::template Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, TailNum>(
            p_shared, splitk_batch_offset, karg, epilogue_args);

#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
    ignore = p_welford_mean_grid;
    ignore = p_welford_var_grid;
    ignore = p_welford_count_grid;
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

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename HLayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename HDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename EMeanVarDataType, // LayerNorm
          typename GammaDataType,    // LayerNorm
          typename BetaDataType,     // LayerNorm
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename HElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
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
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEShuffleBlockTransferScalarPerVector,
          typename LayernormThreadClusterSize_M_N,
          index_t LayernormThreadSliceSize_M,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = HDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3
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
    using DeviceOp = DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3;

    static constexpr index_t NumDTensor                  = DsDataType::Size();
    static constexpr index_t LayernormHDstVectorSize     = CDEShuffleBlockTransferScalarPerVector;
    static constexpr index_t LayernormGammaSrcVectorSize = CDEShuffleBlockTransferScalarPerVector;
    static constexpr index_t LayernormBetaSrcVectorSize  = CDEShuffleBlockTransferScalarPerVector;
    static constexpr index_t LayernormESrcVectorSize     = CDEShuffleBlockTransferScalarPerVector;
    static constexpr index_t LayernormThreadSliceSize_N  = CDEShuffleBlockTransferScalarPerVector;

    using LayernormBlockTileSize_M_N =
        Sequence<LayernormThreadClusterSize_M_N::At(0) * LayernormThreadSliceSize_M,
                 LayernormThreadClusterSize_M_N::At(1) * LayernormThreadSliceSize_N>;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using CDEShuffleBlockTransferScalarPerVectors =
        Sequence<CDEShuffleBlockTransferScalarPerVector,
                 CDEShuffleBlockTransferScalarPerVector,
                 CDEShuffleBlockTransferScalarPerVector>;

    // GEMM + Welford 1st part kernel
    using GridwiseGemmWelford = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        DsLayout,
        HLayout,
        Tuple<ADataType>,
        Tuple<BDataType>,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EMeanVarDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
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
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        PermuteA,
        PermuteB>;

    // Welford 2nd part kernel
    template <typename DoPads, index_t MPerTile, index_t NPerTile>
    static auto MakeEHGridDescriptor_M_N(index_t M, index_t N, index_t Stride)
    {
        // Only support row major for E and H
        const auto grid_desc_m_n =
            make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(Stride, I1));
        return PadTensorDescriptor(grid_desc_m_n, make_tuple(MPerTile, NPerTile), DoPads{});
    }

    template <index_t XPerTile>
    static auto MakeDescriptor_X(index_t X)
    {
        const auto grid_desc_x = make_naive_tensor_descriptor_packed(make_tuple(X));
        return PadTensorDescriptor(grid_desc_x, make_tuple(XPerTile), Sequence<true>{});
    }

    using LayernormMeanVarGridDesc_M_NBlock =
        decltype(GridwiseGemmWelford::EpilogueWelfordCShuffle::template MakeMeanVarDescriptor_M_N<
                 Sequence<true, true>,
                 LayernormBlockTileSize_M_N::At(0),
                 LayernormBlockTileSize_M_N::At(1)>(1, 1));

    using LayernormCountGridDesc_M_NBlock =
        decltype(GridwiseGemmWelford::EpilogueWelfordCShuffle::template MakeCountDescriptor_M_N<
                 Sequence<true, true>,
                 LayernormBlockTileSize_M_N::At(0),
                 LayernormBlockTileSize_M_N::At(1)>(1, 1));

    using GammaBetaGridDesc_N = decltype(MakeDescriptor_X<LayernormBlockTileSize_M_N::At(1)>(1));
    using EHGridDesc_M_N = decltype(MakeEHGridDescriptor_M_N<Sequence<true, true>, 1, 1>(1, 1, 1));

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
              layernorm_e_grid_desc_m_n_{
                  DeviceOp::MakeEHGridDescriptor_M_N<Sequence<true, true>,
                                                     LayernormBlockTileSize_M_N::At(0),
                                                     LayernormBlockTileSize_M_N::At(1)>(
                      MRaw, NRaw, StrideH)},
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
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              h_element_op_{h_element_op},
              MRaw_{MRaw},
              NRaw_{NRaw},
              KRaw_{KRaw},
              StrideA_{StrideA},
              StrideB_{StrideB},
              StrideDs_{StrideDs},
              StrideH_{StrideH},
              gemm_nblock_{math::integer_divide_ceil(NRaw, NPerBlock)},
              epsilon_{static_cast<AccDataType>(epsilon)}
        {
            static_for<0, NumDTensor, 1>{}([&](auto i) { p_ds_grid_[i] = p_ds_grid[i]; });

            layernorm_mean_var_grid_desc_m_nblock_ =
                GridwiseGemmWelford::EpilogueWelfordCShuffle::template MakeMeanVarDescriptor_M_N<
                    Sequence<true, true>,
                    LayernormBlockTileSize_M_N::At(0),
                    LayernormBlockTileSize_M_N::At(1)>(MRaw, gemm_nblock_);

            layernorm_count_grid_desc_m_nblock_ =
                GridwiseGemmWelford::EpilogueWelfordCShuffle::template MakeCountDescriptor_M_N<
                    Sequence<true, true>,
                    LayernormBlockTileSize_M_N::At(0),
                    LayernormBlockTileSize_M_N::At(1)>(MRaw, gemm_nblock_);
        }

        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        std::array<const void*, NumDTensor> p_ds_grid_;
        void* p_workspace_e_grid_;
        void* p_workspace_mean_;
        void* p_workspace_var_;
        void* p_workspace_count_;
        const GammaDataType* p_gamma_grid_;
        const BetaDataType* p_beta_grid_;
        HDataType* p_h_grid_;

        // tensor descriptors (Welford second half)
        EHGridDesc_M_N layernorm_e_grid_desc_m_n_;
        LayernormMeanVarGridDesc_M_NBlock layernorm_mean_var_grid_desc_m_nblock_;
        LayernormCountGridDesc_M_NBlock layernorm_count_grid_desc_m_nblock_;
        GammaBetaGridDesc_N gamma_grid_desc_n_;
        GammaBetaGridDesc_N beta_grid_desc_n_;
        EHGridDesc_M_N h_grid_desc_m_n_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
        HElementwiseOperation h_element_op_;

        index_t MRaw_;
        index_t NRaw_;
        index_t KRaw_;
        index_t StrideA_;
        index_t StrideB_;
        std::array<index_t, NumDTensor> StrideDs_;
        index_t StrideH_;
        index_t gemm_nblock_;
        AccDataType epsilon_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            typename GridwiseGemmWelford::Argument gemm_arg{
                std::array<const void*, 1>{arg.p_a_grid_},
                std::array<const void*, 1>{arg.p_b_grid_},
                arg.p_ds_grid_,
                static_cast<EMeanVarDataType*>(arg.p_workspace_e_grid_),
                arg.MRaw_,
                arg.NRaw_,
                arg.KRaw_,
                std::array<index_t, 1>{arg.StrideA_}, // StrideAs
                std::array<index_t, 1>{arg.StrideB_}, // StrideBs
                arg.StrideDs_,                        // StrideDs
                arg.StrideH_,                         // StrideE
                I1,                                   // kbatch
                arg.a_element_op_,
                arg.b_element_op_,
                arg.cde_element_op_};

            if(stream_config.log_level_ > 0)
            {
                gemm_arg.Print();
                GridwiseGemmWelford::BlockwiseGemmPipe::HotLoopInstList::Print();
            }

            if(!GridwiseGemmWelford::CheckValidity(gemm_arg))
            {
                throw std::runtime_error("wrong! GridwiseGemmWelford has invalid setting");
            }

            if(arg.p_workspace_e_grid_ == nullptr || arg.p_workspace_mean_ == nullptr ||
               arg.p_workspace_var_ == nullptr || arg.p_workspace_count_ == nullptr)
                throw std::runtime_error("wrong! WorkSpace pointer has not been set");

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) =
                GridwiseGemmWelford::CalculateGridSize(arg.MRaw_, arg.NRaw_, 1);

            float ave_time = 0;

            index_t K_split = (arg.KRaw_ + KPerBlock - 1) / KPerBlock * KPerBlock;

            const bool has_main_k_block_loop =
                GridwiseGemmWelford::CalculateHasMainKBlockLoop(K_split);

            const auto Run = [&](const auto& kernel_gemm_welford_first_half) {
                // Note: cache flushing not supported

                const auto kernel_welford_second_half =
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

                // First kernel launch: GEMM + Welford first part
                ave_time +=
                    launch_and_time_kernel(stream_config,
                                           kernel_gemm_welford_first_half,
                                           dim3(gdx, gdy, gdz),
                                           dim3(BlockSize),
                                           0,
                                           gemm_arg,
                                           static_cast<EMeanVarDataType*>(arg.p_workspace_mean_),
                                           static_cast<EMeanVarDataType*>(arg.p_workspace_var_),
                                           static_cast<int32_t*>(arg.p_workspace_count_));

                // Second kernel launch: Welford second part
                const auto M = arg.h_grid_desc_m_n_.GetLength(I0);
                const auto N = arg.h_grid_desc_m_n_.GetLength(I1);

                index_t MBlockClusterLength =
                    math::integer_divide_ceil(M, LayernormBlockTileSize_M_N::At(0));
                index_t NBlockClusterLength =
                    math::integer_divide_ceil(N, LayernormBlockTileSize_M_N::At(1));

                auto grid_size = MBlockClusterLength * NBlockClusterLength;

                index_t numMeanVarCountBlockTileIteration_N = math::integer_divide_ceil(
                    arg.gemm_nblock_, LayernormThreadClusterSize_M_N::At(I1));

                ave_time += launch_and_time_kernel(
                    stream_config,
                    kernel_welford_second_half,
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
            };

            constexpr index_t minimum_occupancy = []() {
                if constexpr(BlkGemmPipeSched == BlockGemmPipelineScheduler::Interwave)
                {
                    return 2;
                }
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    return (MPerBlock * NPerBlock / BlockSize <= 128) ? 2 : 1;
                }
                else
                {
                    return 1;
                }
            }();

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    const auto kernel = kernel_gemm_multiple_d_welford_first_half_wmma_cshuffle_v3<
                        GridwiseGemmWelford,
                        EMeanVarDataType,
                        true,
                        InMemoryDataOperationEnum::Set,
                        minimum_occupancy>;
                    Run(kernel);
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    const auto kernel = kernel_gemm_multiple_d_welford_first_half_wmma_cshuffle_v3<
                        GridwiseGemmWelford,
                        EMeanVarDataType,
                        false,
                        InMemoryDataOperationEnum::Set,
                        minimum_occupancy>;
                    Run(kernel);
                }
            }

            return ave_time;
        }

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
        workspace_size += gemm_welford_size * sizeof(EMeanVarDataType) + 128;

        // workspace for welford intermediate variance
        workspace_size += gemm_welford_size * sizeof(EMeanVarDataType) + 128;

        // workspace for welford intermediate count
        workspace_size += pArg_->gemm_nblock_ * sizeof(int32_t) + 128;

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
        mean_space_sz         = math::integer_least_multiple(mean_space_sz, 128);

        // setup buffer used for intermediate welford variance
        pArg_->p_workspace_var_ = reinterpret_cast<char*>(pArg_->p_workspace_mean_) + mean_space_sz;

        index_t variance_space_sz = gemm_welford_size * sizeof(EMeanVarDataType);
        variance_space_sz         = math::integer_least_multiple(variance_space_sz, 128);

        // setup buffer used for intermediate welford count
        pArg_->p_workspace_count_ =
            reinterpret_cast<char*>(pArg_->p_workspace_var_) + variance_space_sz;

        index_t count_space_sz = gemm_welford_size * sizeof(int32_t);
        count_space_sz         = math::integer_least_multiple(count_space_sz, 128);

        if constexpr(!is_same_v<EMeanVarDataType, HDataType>)
            pArg_->p_workspace_e_grid_ =
                reinterpret_cast<char*>(pArg_->p_workspace_count_) + count_space_sz;
        else
            pArg_->p_workspace_e_grid_ = static_cast<void*>(pArg_->p_h_grid_);
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_gfx11_supported() && !ck::is_gfx12_supported())
        {
            return false;
        }

        // No need to check for splitK because we force KBatch = 1 (no support)

        if constexpr(std::is_same_v<ComputeTypeA, f8_t> || std::is_same_v<ComputeTypeA, bf8_t> ||
                     std::is_same_v<ComputeTypeB, f8_t> || std::is_same_v<ComputeTypeB, bf8_t>)
        {
            if(ck::is_gfx11_supported())
            {
                return false;
            }
        }

        if((arg.KRaw_ % AK1 != 0 || arg.KRaw_ % BK1 != 0) &&
           !(GemmSpec == GemmSpecialization::MKPadding ||
             GemmSpec == GemmSpecialization::NKPadding ||
             GemmSpec == GemmSpecialization::MNKPadding ||
             GemmSpec == GemmSpecialization::KPadding))
        {
            return false;
        }

        typename GridwiseGemmWelford::Argument gemm_arg{
            std::array<const void*, 1>{arg.p_a_grid_},
            std::array<const void*, 1>{arg.p_b_grid_},
            arg.p_ds_grid_,
            static_cast<EMeanVarDataType*>(arg.p_workspace_e_grid_),
            arg.MRaw_,
            arg.NRaw_,
            arg.KRaw_,
            std::array<index_t, 1>{arg.StrideA_}, // StrideAs
            std::array<index_t, 1>{arg.StrideB_}, // StrideBs
            arg.StrideDs_,                        // StrideDs
            arg.StrideH_,                         // StrideE
            I1,                                   // kbatch
            arg.a_element_op_,
            arg.b_element_op_,
            arg.cde_element_op_};

        const auto a_grid_desc_ak0_m_ak1 =
            GridwiseGemmWelford::MakeAsGridDescriptor_AK0_M_AK1(gemm_arg.M,
                                                                gemm_arg.MPadded,
                                                                gemm_arg.K,
                                                                gemm_arg.KPadded,
                                                                gemm_arg.StrideAs,
                                                                gemm_arg.AK0);
        const auto b_grid_desc_bk0_n_bk1 =
            GridwiseGemmWelford::MakeBsGridDescriptor_BK0_N_BK1(gemm_arg.K,
                                                                gemm_arg.KPadded,
                                                                gemm_arg.N,
                                                                gemm_arg.NPadded,
                                                                gemm_arg.StrideBs,
                                                                gemm_arg.BK0);

        const auto M = a_grid_desc_ak0_m_ak1[I0].GetLength(I1);
        const auto N = b_grid_desc_bk0_n_bk1[I0].GetLength(I1);
        const auto K =
            a_grid_desc_ak0_m_ak1[I0].GetLength(I0) * a_grid_desc_ak0_m_ak1[I0].GetLength(I2);

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0))
        {
            return false;
        }

        return GridwiseGemmWelford::CheckValidity(gemm_arg);
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

        std::map<BlockGemmPipelineScheduler, std::string> BlkGemmPipelineSchedulerToString{
            {BlockGemmPipelineScheduler::Intrawave, "Intrawave"},
            {BlockGemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<BlockGemmPipelineVersion, std::string> BlkGemmPipelineVersionToString{
            {BlockGemmPipelineVersion::v1, "v1"},
            {BlockGemmPipelineVersion::v2, "v2"},
            {BlockGemmPipelineVersion::v3, "v3"},
            {BlockGemmPipelineVersion::v4, "v4"},
            {BlockGemmPipelineVersion::v5, "v5"}};

        // clang-format off
        str << "DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3"
            << ">"
            << "BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << "WaveTile: "
            << MPerWmma << "x"<<NPerWmma << ", "
            << "WaveMap: "
            << MRepeat << "x" << NRepeat << ", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector << "x" << BBlockTransferSrcScalarPerVector << ", "
            << "GemmSpec: "
            << getGemmSpecializationString(GemmSpec) << ", "
            << "VmemWriteThreadCluster: "
            << CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(I1) << ", "
            << CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(I3) << ", "
            << "LayerNormThreadCluster: "
            << LayernormThreadClusterSize_M_N::At(I0) << ", "
            << LayernormThreadClusterSize_M_N::At(I1) << ", "
            << "LayerNormThreadSliceSize: "
            << LayernormThreadSliceSize_M << ", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseGemmWelford::BlockwiseGemmPipe::PrefetchStages << ", "
            << "KPack: "
            << GridwiseGemmWelford::KPack;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
