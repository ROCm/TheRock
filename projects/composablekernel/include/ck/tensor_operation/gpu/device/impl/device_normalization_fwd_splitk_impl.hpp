// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization_fwd.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/normalization/gridwise_normalization_splitk_1st.hpp"
#include "ck/tensor_operation/gpu/grid/normalization/gridwise_normalization_splitk_2nd.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
template <typename GridwiseWelford,
          typename XDataType,
          typename WorkspaceMeanVarDataType,
          typename ComputeDataType,
          typename XGridDesc_M_K,
          typename MeanVarGridDesc_M_KBlock>
__global__ void
kernel_normalizationSplitK1st(const XGridDesc_M_K x_grid_desc_m_k,
                              const MeanVarGridDesc_M_KBlock mean_var_grid_desc_m_kblock,
                              index_t num_k_block_tile_iteration,
                              const XDataType* const __restrict__ p_x_global,
                              WorkspaceMeanVarDataType* const __restrict__ p_welford_mean,
                              WorkspaceMeanVarDataType* const __restrict__ p_welford_variance,
                              int32_t* const __restrict__ p_welford_count)
{
    GridwiseWelford::Run(x_grid_desc_m_k,
                         mean_var_grid_desc_m_kblock,
                         num_k_block_tile_iteration,
                         p_x_global,
                         p_welford_mean,
                         p_welford_variance,
                         p_welford_count);
};

template <typename GridwiseWelfordNormalization,
          typename WorkspaceMeanVarDataType,
          typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType,
          typename ComputeDataType,
          typename YElementwiseOperation,
          typename MeanVarGridDesc_M_KBlock,
          typename CountGridDesc_M_KBlock,
          typename XYGammaBetaGridDesc_M_K,
          typename SaveMeanInvStdGridDesc_M>
__global__ void
kernel_normalizationSplitK2nd(const MeanVarGridDesc_M_KBlock mean_var_grid_desc_m_kblock,
                              const CountGridDesc_M_KBlock count_grid_desc_m_kblock,
                              const XYGammaBetaGridDesc_M_K x_grid_desc_m_k,
                              const XYGammaBetaGridDesc_M_K gamma_grid_desc_m_k,
                              const XYGammaBetaGridDesc_M_K beta_grid_desc_m_k,
                              const XYGammaBetaGridDesc_M_K y_grid_desc_m_k,
                              const SaveMeanInvStdGridDesc_M save_mean_grid_desc_m,
                              const SaveMeanInvStdGridDesc_M save_inv_std_grid_desc_m,
                              index_t num_k_mean_var_count_iteration,
                              index_t num_k_block_tile_iteration,
                              index_t k_grid_size,
                              ComputeDataType epsilon,
                              const WorkspaceMeanVarDataType* const p_mean_global,
                              const WorkspaceMeanVarDataType* const p_variance_global,
                              const int32_t* const p_welford_count_global,
                              const XDataType* const __restrict__ p_x_global,
                              const GammaDataType* const __restrict__ p_gamma_global,
                              const BetaDataType* const __restrict__ p_beta_global,
                              YDataType* const __restrict__ p_y_global,
                              SaveMeanInvStdDataType* const __restrict__ p_save_mean_global,
                              SaveMeanInvStdDataType* const __restrict__ p_save_inv_std_global,
                              const YElementwiseOperation y_elementwise_op)
{
    GridwiseWelfordNormalization::Run(mean_var_grid_desc_m_kblock,
                                      count_grid_desc_m_kblock,
                                      x_grid_desc_m_k,
                                      gamma_grid_desc_m_k,
                                      beta_grid_desc_m_k,
                                      y_grid_desc_m_k,
                                      save_mean_grid_desc_m,
                                      save_inv_std_grid_desc_m,
                                      num_k_mean_var_count_iteration,
                                      num_k_block_tile_iteration,
                                      k_grid_size,
                                      epsilon,
                                      p_mean_global,
                                      p_variance_global,
                                      p_welford_count_global,
                                      p_x_global,
                                      p_gamma_global,
                                      p_beta_global,
                                      p_y_global,
                                      p_save_mean_global,
                                      p_save_inv_std_global,
                                      y_elementwise_op);
};
} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

// Y = Normalization(X, Beta, Gamma)
// M: Invariant length
// K: Reduce length (Calculate mean and variance along K dimension)
// eg. Length = [N, C, H, W], reduce dim = [C, H, W]
// Then, M = N, K = C * H * W
template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename ComputeDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType,
          typename YElementwiseOperation,
          index_t Rank,
          index_t NumReduceDim,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XYVectorDim,
          index_t XSrcVectorSize,
          index_t GammaSrcVectorDim,
          index_t GammaSrcVectorSize,
          index_t BetaSrcVectorDim,
          index_t BetaSrcVectorSize,
          index_t YDstVectorSize,
          index_t SaveMeanInvStdDstVectorSize>
struct DeviceNormalizationFwdSplitKImpl : public DeviceNormalizationFwd<XDataType,
                                                                        GammaDataType,
                                                                        BetaDataType,
                                                                        YDataType,
                                                                        SaveMeanInvStdDataType,
                                                                        YElementwiseOperation,
                                                                        Rank,
                                                                        NumReduceDim>
{
    using WorkspaceMeanVarDataType = SaveMeanInvStdDataType;

    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize);
    static_assert(
        ((GammaSrcVectorDim == 0 && MThreadSliceSize % GammaSrcVectorSize == 0) ||
         (GammaSrcVectorDim == 1 && KThreadSliceSize % GammaSrcVectorSize == 0)),
        "Invalid thread slice sizes and/or gamma vector sizes configuration, please check!");

    static_assert(
        ((BetaSrcVectorDim == 0 && MThreadSliceSize % BetaSrcVectorSize == 0) ||
         (BetaSrcVectorDim == 1 && KThreadSliceSize % BetaSrcVectorSize == 0)),
        "Invalid thread slice sizes and/or beta vector sizes configuration, please check!");

    static_assert(MThreadSliceSize % SaveMeanInvStdDstVectorSize == 0,
                  "Invalid thread slice sizes and/or save mean and inverse std vector sizes "
                  "configuration, please check!");

    using PassThrough = tensor_operation::element_wise::PassThrough;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr index_t NumInvariantDim = Rank - NumReduceDim;
    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static constexpr bool reduceAllDim = (NumInvariantDim == 0);
    static_assert(!reduceAllDim); // TODO

    static auto MakeSrc2dDescriptor(const std::vector<index_t>& inLengths,
                                    const std::vector<index_t>& inStrides,
                                    int kBlockSize,
                                    int numBlockTileIteration)
    {
        static constexpr index_t numSrcDim = Rank;

        const auto tupleSrcLengths = make_tuple_from_array(inLengths, Number<numSrcDim>{});
        const auto tupleSrcStrides = make_tuple_from_array(inStrides, Number<numSrcDim>{});

        const auto inDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto in_grid_desc_m_k = [&]() {
            if constexpr(reduceAllDim)
            {
                const auto one_dim_inDesc = transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(tupleSrcLengths)),
                    make_tuple(typename arithmetic_sequence_gen<0, numSrcDim, 1>::type{}),
                    make_tuple(Sequence<0>{}));

                return transform_tensor_descriptor(one_dim_inDesc,
                                                   make_tuple(make_unmerge_transform(make_tuple(
                                                       1, one_dim_inDesc.GetLength(Number<0>{})))),
                                                   make_tuple(Sequence<0>{}),
                                                   make_tuple(Sequence<0, 1>{}));
            }
            else
            {
                using InvariantDims = typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type;
                using ReduceDims = typename arithmetic_sequence_gen<NumInvariantDim, Rank, 1>::type;

                const auto reduceDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, ReduceDims{});
                const auto invariantDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, InvariantDims{});

                return transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(invariantDimLengths),
                               make_merge_transform(reduceDimLengths)),
                    make_tuple(InvariantDims{}, ReduceDims{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        const auto invariantLength = in_grid_desc_m_k.GetLength(Number<0>{});
        const auto reduceLength    = in_grid_desc_m_k.GetLength(Number<1>{});

        const int reduceSizePerBlock = K_BlockTileSize * numBlockTileIteration;
        const auto inPad_M =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;
        const auto inPad_K = reduceSizePerBlock * kBlockSize - reduceLength;

        auto in_grid_desc_m_k_padded = transform_tensor_descriptor(
            in_grid_desc_m_k,
            make_tuple(make_right_pad_transform(invariantLength, inPad_M),
                       make_right_pad_transform(reduceLength, inPad_K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (in_grid_desc_m_k_padded);
    };

    template <typename DoPads, index_t MPerTile, index_t KPerTile>
    static auto MakeWorkspaceMeanVarDescriptor_M_K(index_t M, index_t K)
    {
        const auto grid_desc_m_k =
            make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(K, I1));
        return PadTensorDescriptor(grid_desc_m_k, make_tuple(MPerTile, KPerTile), DoPads{});
    }

    template <typename DoPads, index_t MPerTile, index_t KPerTile>
    static auto MakeWorkspaceCountDescriptor_M_K(index_t M, index_t K)
    {
        const auto grid_desc_m_k =
            make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I0, I1));
        return PadTensorDescriptor(grid_desc_m_k, make_tuple(MPerTile, KPerTile), DoPads{});
    }

    static auto MakeSaveMeanInvStdDescriptor_M(const std::vector<index_t>& lengths,
                                               const std::vector<index_t>& strides)
    {
        using InvariantDims = typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type;

        const auto tupleSrcLengths = make_tuple_from_array_and_index_seq(lengths, InvariantDims{});
        const auto tupleSrcStrides = make_tuple_from_array_and_index_seq(strides, InvariantDims{});

        const auto desc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto grid_desc_m =
            transform_tensor_descriptor(desc,
                                        make_tuple(make_merge_transform(tupleSrcLengths)),
                                        make_tuple(InvariantDims{}),
                                        make_tuple(Sequence<0>{}));

        const auto invariantLength = grid_desc_m.GetLength(Number<0>{});
        const auto pad_M =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;

        auto grid_desc_m_padded = transform_tensor_descriptor(
            grid_desc_m,
            make_tuple(make_right_pad_transform(invariantLength, pad_M)),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0>{}));

        return grid_desc_m_padded;
    }

    using SrcGridDesc_M_K = decltype(MakeSrc2dDescriptor({1}, {1}, 1, 1));
    using Kernel1MeanVarGridDesc_M_KBlock =
        decltype(MakeWorkspaceMeanVarDescriptor_M_K<Sequence<true, false>, 1, 1>(1, 1));

    using Kernel2MeanVarGridDesc_M_KBlock =
        decltype(MakeWorkspaceMeanVarDescriptor_M_K<Sequence<true, true>, 1, 1>(1, 1));

    using Kernel2CountGridDesc_M_KBlock =
        decltype(MakeWorkspaceCountDescriptor_M_K<Sequence<true, true>, 1, 1>(1, 1));

    using SaveMeanInvStdGridDesc_M = decltype(MakeSaveMeanInvStdDescriptor_M({1}, {1}));

    using GridwiseWelford = GridwiseNormalizationSplitK1st<XDataType,
                                                           ComputeDataType,
                                                           WorkspaceMeanVarDataType,
                                                           SrcGridDesc_M_K,
                                                           Kernel1MeanVarGridDesc_M_KBlock,
                                                           BlockSize,
                                                           MThreadClusterSize,
                                                           KThreadClusterSize,
                                                           MThreadSliceSize,
                                                           KThreadSliceSize,
                                                           XYVectorDim,
                                                           XSrcVectorSize>;

    using GridwiseWelfordNormalization =
        GridwiseNormalizationSplitK2nd<WorkspaceMeanVarDataType,
                                       XDataType,
                                       GammaDataType,
                                       BetaDataType,
                                       YDataType,
                                       SaveMeanInvStdDataType,
                                       ComputeDataType,
                                       YElementwiseOperation,
                                       Kernel2MeanVarGridDesc_M_KBlock,
                                       Kernel2CountGridDesc_M_KBlock,
                                       SrcGridDesc_M_K,
                                       SaveMeanInvStdGridDesc_M,
                                       BlockSize,
                                       MThreadClusterSize,
                                       KThreadClusterSize,
                                       MThreadSliceSize,
                                       KThreadSliceSize,
                                       XYVectorDim,
                                       XSrcVectorSize,
                                       GammaSrcVectorDim,
                                       GammaSrcVectorSize,
                                       BetaSrcVectorDim,
                                       BetaSrcVectorSize,
                                       XYVectorDim,
                                       YDstVectorSize,
                                       SaveMeanInvStdDstVectorSize>;

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<index_t> lengths,
                 const std::vector<index_t> xStrides,
                 const std::vector<index_t> gammaStrides,
                 const std::vector<index_t> betaStrides,
                 const std::vector<index_t> yStrides,
                 const std::vector<index_t> saveMeanStrides,
                 const std::vector<index_t> saveInvStdStrides,
                 const std::vector<index_t> reduceDims,
                 YElementwiseOperation y_elementwise_op,
                 double epsilon,
                 const XDataType* p_x,
                 const GammaDataType* p_gamma,
                 const BetaDataType* p_beta,
                 YDataType* p_y,
                 SaveMeanInvStdDataType* p_saveMean,
                 SaveMeanInvStdDataType* p_saveInvStd)
            : p_x_(p_x),
              p_gamma_(p_gamma),
              p_beta_(p_beta),
              p_y_(p_y),
              p_saveMean_(p_saveMean),
              p_saveInvStd_(p_saveInvStd),
              p_workspace_mean_{nullptr},
              p_workspace_var_{nullptr},
              p_workspace_count_{nullptr},
              y_elementwise_op_(y_elementwise_op)
        {
            epsilon_ = static_cast<ComputeDataType>(epsilon);

            Lengths_      = shuffle_tensor_dimensions<Rank, NumReduceDim>(lengths, reduceDims);
            xStrides_     = shuffle_tensor_dimensions<Rank, NumReduceDim>(xStrides, reduceDims);
            yStrides_     = shuffle_tensor_dimensions<Rank, NumReduceDim>(yStrides, reduceDims);
            gammaStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(gammaStrides, reduceDims);
            betaStrides_  = shuffle_tensor_dimensions<Rank, NumReduceDim>(betaStrides, reduceDims);
            saveMeanStrides_   = saveMeanStrides;
            saveInvStdStrides_ = saveInvStdStrides;

            std::tie(MRaw_, KRaw_) = get_2d_lengths<Rank, NumReduceDim>(Lengths_);

            numBlockTileIteration_ = 1;
            while(true)
            {
                int testKGridSize =
                    math::integer_divide_ceil(KRaw_, K_BlockTileSize * numBlockTileIteration_);

                // we want the kGridSize_ be not more than 128
                if(testKGridSize <= 128)
                    break;

                ++numBlockTileIteration_;
            };

            kGridSize_ = math::integer_divide_ceil(KRaw_, K_BlockTileSize * numBlockTileIteration_);
            gridSize_  = math::integer_divide_ceil(MRaw_, M_BlockTileSize) * kGridSize_;

            // We do not use vector load for mean, var and count
            static constexpr index_t K_MeanVarCountBlockTileSize = KThreadClusterSize;

            numMeanVarCountIteration_ =
                math::integer_divide_ceil(kGridSize_, K_MeanVarCountBlockTileSize);

            x_grid_desc_m_k_ =
                MakeSrc2dDescriptor(Lengths_, xStrides_, kGridSize_, numBlockTileIteration_);
            gamma_grid_desc_m_k_ =
                MakeSrc2dDescriptor(Lengths_, gammaStrides_, kGridSize_, numBlockTileIteration_);
            beta_grid_desc_m_k_ =
                MakeSrc2dDescriptor(Lengths_, betaStrides_, kGridSize_, numBlockTileIteration_);
            y_grid_desc_m_k_ =
                MakeSrc2dDescriptor(Lengths_, yStrides_, kGridSize_, numBlockTileIteration_);

            save_mean_grid_desc_m_    = MakeSaveMeanInvStdDescriptor_M(Lengths_, saveMeanStrides);
            save_inv_std_grid_desc_m_ = MakeSaveMeanInvStdDescriptor_M(Lengths_, saveInvStdStrides);

            // We don't need to pad in K dimension for Welford1. Set KPerTile 1.
            kernel1_mean_var_grid_desc_m_kblock_ =
                MakeWorkspaceMeanVarDescriptor_M_K<Sequence<true, false>, M_BlockTileSize, 1>(
                    MRaw_, kGridSize_);

            kernel2_mean_var_grid_desc_m_kblock_ =
                MakeWorkspaceMeanVarDescriptor_M_K<Sequence<true, true>,
                                                   M_BlockTileSize,
                                                   K_MeanVarCountBlockTileSize>(MRaw_, kGridSize_);

            kernel2_count_grid_desc_m_kblock_ =
                MakeWorkspaceCountDescriptor_M_K<Sequence<true, true>,
                                                 M_BlockTileSize,
                                                 K_MeanVarCountBlockTileSize>(MRaw_, kGridSize_);

            if constexpr(NumInvariantDim == 0)
                invariant_lowest_length_ = 1;
            else
                invariant_lowest_length_ = Lengths_[NumInvariantDim - 1];
        }

        ComputeDataType epsilon_;

        const XDataType* p_x_;
        const GammaDataType* p_gamma_;
        const BetaDataType* p_beta_;
        YDataType* p_y_;
        SaveMeanInvStdDataType* p_saveMean_;
        SaveMeanInvStdDataType* p_saveInvStd_;
        void* p_workspace_mean_;
        void* p_workspace_var_;
        void* p_workspace_count_;

        std::vector<index_t> Lengths_;
        std::vector<index_t> xStrides_;
        std::vector<index_t> gammaStrides_;
        std::vector<index_t> betaStrides_;
        std::vector<index_t> yStrides_;
        std::vector<index_t> saveMeanStrides_;
        std::vector<index_t> saveInvStdStrides_;

        YElementwiseOperation y_elementwise_op_;

        int kGridSize_;
        int numMeanVarCountIteration_;
        int numBlockTileIteration_;
        size_t gridSize_;

        SrcGridDesc_M_K x_grid_desc_m_k_;
        SrcGridDesc_M_K gamma_grid_desc_m_k_;
        SrcGridDesc_M_K beta_grid_desc_m_k_;
        SrcGridDesc_M_K y_grid_desc_m_k_;
        SaveMeanInvStdGridDesc_M save_mean_grid_desc_m_;
        SaveMeanInvStdGridDesc_M save_inv_std_grid_desc_m_;

        Kernel1MeanVarGridDesc_M_KBlock kernel1_mean_var_grid_desc_m_kblock_;
        Kernel2MeanVarGridDesc_M_KBlock kernel2_mean_var_grid_desc_m_kblock_;
        Kernel2CountGridDesc_M_KBlock kernel2_count_grid_desc_m_kblock_;

        index_t MRaw_; // Invariant length
        index_t KRaw_; // reduce length

        index_t invariant_lowest_length_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(arg.p_workspace_mean_ == nullptr || arg.p_workspace_var_ == nullptr ||
               arg.p_workspace_count_ == nullptr)
                throw std::runtime_error("wrong! WorkSpace pointer has not been set");

            auto kernel1 = kernel_normalizationSplitK1st<GridwiseWelford,
                                                         XDataType,
                                                         WorkspaceMeanVarDataType,
                                                         ComputeDataType,
                                                         SrcGridDesc_M_K,
                                                         Kernel1MeanVarGridDesc_M_KBlock>;

            auto kernel2 = kernel_normalizationSplitK2nd<GridwiseWelfordNormalization,
                                                         WorkspaceMeanVarDataType,
                                                         XDataType,
                                                         GammaDataType,
                                                         BetaDataType,
                                                         YDataType,
                                                         SaveMeanInvStdDataType,
                                                         ComputeDataType,
                                                         YElementwiseOperation,
                                                         Kernel2MeanVarGridDesc_M_KBlock,
                                                         Kernel2CountGridDesc_M_KBlock,
                                                         SrcGridDesc_M_K,
                                                         SaveMeanInvStdGridDesc_M>;

            float avg_time = 0;
            avg_time += launch_and_time_kernel(
                stream_config,
                kernel1,
                dim3(arg.gridSize_),
                dim3(BlockSize),
                0,
                arg.x_grid_desc_m_k_,
                arg.kernel1_mean_var_grid_desc_m_kblock_,
                arg.numBlockTileIteration_,
                arg.p_x_,
                static_cast<WorkspaceMeanVarDataType*>(arg.p_workspace_mean_),
                static_cast<WorkspaceMeanVarDataType*>(arg.p_workspace_var_),
                static_cast<int32_t*>(arg.p_workspace_count_));

            avg_time += launch_and_time_kernel(
                stream_config,
                kernel2,
                dim3(arg.gridSize_),
                dim3(BlockSize),
                0,
                arg.kernel2_mean_var_grid_desc_m_kblock_,
                arg.kernel2_count_grid_desc_m_kblock_,
                arg.x_grid_desc_m_k_,
                arg.gamma_grid_desc_m_k_,
                arg.beta_grid_desc_m_k_,
                arg.y_grid_desc_m_k_,
                arg.save_mean_grid_desc_m_,
                arg.save_inv_std_grid_desc_m_,
                arg.numMeanVarCountIteration_,
                arg.numBlockTileIteration_,
                arg.kGridSize_,
                arg.epsilon_,
                static_cast<const WorkspaceMeanVarDataType*>(arg.p_workspace_mean_),
                static_cast<const WorkspaceMeanVarDataType*>(arg.p_workspace_var_),
                static_cast<const int32_t*>(arg.p_workspace_count_),
                arg.p_x_,
                arg.p_gamma_,
                arg.p_beta_,
                arg.p_y_,
                arg.p_saveMean_,
                arg.p_saveInvStd_,
                arg.y_elementwise_op_);

            return avg_time;
        };

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        };
    };

    size_t GetWorkSpaceSize(const BaseArgument* pArg) const override
    {
        const Argument* pArg_ = dynamic_cast<const Argument*>(pArg);

        size_t workspace_size = 0;

        int welford_size = pArg_->MRaw_ * pArg_->kGridSize_;

        // workspace for welford intermediate mean
        workspace_size += welford_size * sizeof(WorkspaceMeanVarDataType) + 64;

        // workspace for welford intermediate variance
        workspace_size += welford_size * sizeof(WorkspaceMeanVarDataType) + 64;

        // workspace for welford intermediate count
        workspace_size += pArg_->kGridSize_ * sizeof(int32_t) + 64;

        return (workspace_size);
    };

    void SetWorkSpacePointer(BaseArgument* pArg,
                             void* p_workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        Argument* pArg_ = dynamic_cast<Argument*>(pArg);

        pArg_->p_workspace_ = p_workspace;

        int welford_size = pArg_->MRaw_ * pArg_->kGridSize_;

        // setup buffer used for intermediate welford mean
        pArg_->p_workspace_mean_ = static_cast<char*>(pArg_->p_workspace_);

        index_t mean_space_sz = welford_size * sizeof(WorkspaceMeanVarDataType);
        mean_space_sz         = math::integer_least_multiple(mean_space_sz, 64);

        // setup buffer used for intermediate welford varirance
        pArg_->p_workspace_var_ = reinterpret_cast<char*>(pArg_->p_workspace_mean_) + mean_space_sz;

        index_t variance_space_sz = welford_size * sizeof(WorkspaceMeanVarDataType);
        variance_space_sz         = math::integer_least_multiple(variance_space_sz, 64);

        // setup buffer used for intermediate welford count
        pArg_->p_workspace_count_ =
            reinterpret_cast<char*>(pArg_->p_workspace_var_) + variance_space_sz;
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* p_arg_ = dynamic_cast<const Argument*>(p_arg);

        if constexpr(XYVectorDim == 0)
        {
            if constexpr(NumInvariantDim == 0)
            {
                return false;
            }
            else
            {
                if(p_arg_->xStrides_[NumInvariantDim - 1] != 1)
                    return false;

                if(p_arg_->invariant_lowest_length_ % XSrcVectorSize != 0)
                    return false;

                if(p_arg_->invariant_lowest_length_ % YDstVectorSize != 0)
                    return false;
            };
        }
        else
        {
            if(p_arg_->xStrides_[Rank - 1] != 1)
                return false;

            if(p_arg_->Lengths_[Rank - 1] % XSrcVectorSize != 0)
                return false;

            if(p_arg_->Lengths_[Rank - 1] % YDstVectorSize != 0)
                return false;
        };

        // if fastest dim is not reduced
        if constexpr(GammaSrcVectorDim == 0)
        {
            if(p_arg_->gammaStrides_[NumInvariantDim - 1] != 1)
                return false;

            if(p_arg_->Lengths_[Rank - 1] % GammaSrcVectorSize != 0)
                return false;
        }
        else // if fastest dim is reduced
        {
            if(p_arg_->gammaStrides_[Rank - 1] != 1)
                return false;

            if(p_arg_->Lengths_[Rank - 1] % GammaSrcVectorSize != 0)
                return false;
        }

        // if fastest dim is not reduced
        if constexpr(BetaSrcVectorDim == 0)
        {
            if(p_arg_->betaStrides_[NumInvariantDim - 1] != 1)
                return false;

            if(p_arg_->invariant_lowest_length_ % BetaSrcVectorSize != 0)
                return false;
        }
        else // if fastest dim is reduced
        {
            if(p_arg_->betaStrides_[Rank - 1] != 1)
                return false;

            if(p_arg_->Lengths_[Rank - 1] % BetaSrcVectorSize != 0)
                return false;
        }

        if(p_arg_->kGridSize_ <= 1)
            return false;

        if(p_arg_->invariant_lowest_length_ % SaveMeanInvStdDstVectorSize != 0)
            return false;

        return true;
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> lengths,
                        const std::vector<index_t> xStrides,
                        const std::vector<index_t> gammaStrides,
                        const std::vector<index_t> betaStrides,
                        const std::vector<index_t> yStrides,
                        const std::vector<index_t> saveMeanStrides,
                        const std::vector<index_t> saveInvStdStrides,
                        const std::vector<index_t> reduceDims,
                        double epsilon,
                        const void* p_x,
                        const void* p_gamma,
                        const void* p_beta,
                        void* p_y,
                        void* p_saveMean,
                        void* p_saveInvStd,
                        YElementwiseOperation y_elementwise_op) override
    {
        if(lengths.size() != Rank || xStrides.size() != Rank || gammaStrides.size() != Rank ||
           betaStrides.size() != Rank || yStrides.size() != Rank ||
           saveMeanStrides.size() != NumInvariantDim || saveInvStdStrides.size() != NumInvariantDim)
            throw std::runtime_error("dimension is incorrect");

        return std::make_unique<Argument>(lengths,
                                          xStrides,
                                          gammaStrides,
                                          betaStrides,
                                          yStrides,
                                          saveMeanStrides,
                                          saveInvStdStrides,
                                          reduceDims,
                                          y_elementwise_op,
                                          epsilon,
                                          static_cast<const XDataType*>(p_x),
                                          static_cast<const GammaDataType*>(p_gamma),
                                          static_cast<const BetaDataType*>(p_beta),
                                          static_cast<YDataType*>(p_y),
                                          static_cast<SaveMeanInvStdDataType*>(p_saveMean),
                                          static_cast<SaveMeanInvStdDataType*>(p_saveInvStd));
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceNormalizationFwdSplitKImpl<" << BlockSize << ",";
        str << "Cluster_MK_" << MThreadClusterSize << "_" << KThreadClusterSize << ",";
        str << "Slice_MK_" << MThreadSliceSize << "_" << KThreadSliceSize << ",";
        str << "XYSrcVectorDim_" << XYVectorDim  << ",";
        str << "VectorSize_X" << XSrcVectorSize << "_Gamma" << GammaSrcVectorSize << "_Beta" << BetaSrcVectorSize << "_Y" << YDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
