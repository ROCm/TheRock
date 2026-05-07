// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization_fwd.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/normalization/gridwise_normalization_selector.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

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
          index_t XYSrcVectorDim,
          index_t XSrcVectorSize,
          index_t GammaSrcVectorDim,
          index_t GammaSrcVectorSize,
          index_t BetaSrcVectorDim,
          index_t BetaSrcVectorSize,
          index_t YDstVectorSize,
          index_t SaveMeanInvStdDstVectorSize,
          bool UseWelford = true>
struct DeviceNormalizationFwdImpl : public DeviceNormalizationFwd<XDataType,
                                                                  GammaDataType,
                                                                  BetaDataType,
                                                                  YDataType,
                                                                  SaveMeanInvStdDataType,
                                                                  YElementwiseOperation,
                                                                  Rank,
                                                                  NumReduceDim>
{
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

    static constexpr index_t NumInvariantDim = Rank - NumReduceDim;
    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static constexpr bool reduceAllDim = (NumInvariantDim == 0);
    static_assert(!reduceAllDim); // TODO

    static auto MakeSrc2dDescriptor(const std::vector<index_t>& inLengths,
                                    const std::vector<index_t>& inStrides,
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

        const auto inPad_M =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;
        const auto inPad_K = K_BlockTileSize * numBlockTileIteration - reduceLength;

        auto in_grid_desc_m_k_padded = transform_tensor_descriptor(
            in_grid_desc_m_k,
            make_tuple(make_right_pad_transform(invariantLength, inPad_M),
                       make_right_pad_transform(reduceLength, inPad_K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (in_grid_desc_m_k_padded);
    };

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

    using GridDesc_M_K = decltype(MakeSrc2dDescriptor({1}, {1}, 1));
    using GridDesc_M   = decltype(MakeSaveMeanInvStdDescriptor_M({1}, {1}));

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

            numBlockTileIteration_ = math::integer_divide_ceil(KRaw_, K_BlockTileSize);

            gridSize_ = math::integer_divide_ceil(MRaw_, M_BlockTileSize);

            x_grid_desc_m_k_ = MakeSrc2dDescriptor(Lengths_, xStrides_, numBlockTileIteration_);
            gamma_grid_desc_m_k_ =
                MakeSrc2dDescriptor(Lengths_, gammaStrides_, numBlockTileIteration_);
            beta_grid_desc_m_k_ =
                MakeSrc2dDescriptor(Lengths_, betaStrides_, numBlockTileIteration_);
            y_grid_desc_m_k_ = MakeSrc2dDescriptor(Lengths_, yStrides_, numBlockTileIteration_);
            save_mean_grid_desc_m_    = MakeSaveMeanInvStdDescriptor_M(Lengths_, saveMeanStrides);
            save_inv_std_grid_desc_m_ = MakeSaveMeanInvStdDescriptor_M(Lengths_, saveInvStdStrides);

            isSweeponce_ =
                x_grid_desc_m_k_.GetLength(Number<1>{}) <= KThreadClusterSize * KThreadSliceSize;

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

        std::vector<index_t> Lengths_;
        std::vector<index_t> xStrides_;
        std::vector<index_t> gammaStrides_;
        std::vector<index_t> betaStrides_;
        std::vector<index_t> yStrides_;
        std::vector<index_t> saveMeanStrides_;
        std::vector<index_t> saveInvStdStrides_;

        YElementwiseOperation y_elementwise_op_;

        int numBlockTileIteration_;
        size_t gridSize_;

        GridDesc_M_K x_grid_desc_m_k_;
        GridDesc_M_K gamma_grid_desc_m_k_;
        GridDesc_M_K beta_grid_desc_m_k_;
        GridDesc_M_K y_grid_desc_m_k_;
        GridDesc_M save_mean_grid_desc_m_;
        GridDesc_M save_inv_std_grid_desc_m_;
        bool isSweeponce_;

        index_t MRaw_; // Invariant length
        index_t KRaw_; // reduce length

        index_t invariant_lowest_length_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            auto kernel_main = NormalizationKernelSelector<XDataType,
                                                           GammaDataType,
                                                           BetaDataType,
                                                           YDataType,
                                                           SaveMeanInvStdDataType,
                                                           ComputeDataType,
                                                           YElementwiseOperation,
                                                           GridDesc_M_K,
                                                           GridDesc_M,
                                                           BlockSize,
                                                           MThreadClusterSize,
                                                           KThreadClusterSize,
                                                           MThreadSliceSize,
                                                           KThreadSliceSize,
                                                           XYSrcVectorDim,
                                                           XSrcVectorSize,
                                                           GammaSrcVectorDim,
                                                           GammaSrcVectorSize,
                                                           BetaSrcVectorDim,
                                                           BetaSrcVectorSize,
                                                           XYSrcVectorDim,
                                                           YDstVectorSize,
                                                           SaveMeanInvStdDstVectorSize,
                                                           UseWelford>(arg.isSweeponce_);

            float avg_time = 0;
            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_main,
                                               dim3(arg.gridSize_),
                                               dim3(BlockSize),
                                               0,
                                               arg.x_grid_desc_m_k_,
                                               arg.gamma_grid_desc_m_k_,
                                               arg.beta_grid_desc_m_k_,
                                               arg.y_grid_desc_m_k_,
                                               arg.save_mean_grid_desc_m_,
                                               arg.save_inv_std_grid_desc_m_,
                                               arg.numBlockTileIteration_,
                                               arg.epsilon_,
                                               arg.p_x_,
                                               arg.p_gamma_,
                                               arg.p_beta_,
                                               arg.p_y_,
                                               arg.p_saveMean_,
                                               arg.p_saveInvStd_,
                                               arg.y_elementwise_op_);

            return (avg_time);
        };

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        };
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* p_arg_ = dynamic_cast<const Argument*>(p_arg);

        if constexpr(XYSrcVectorDim == 0)
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
            {
                return false;
            }
        };

        // if fastest dim is not reduced
        if constexpr(GammaSrcVectorDim == 0)
        {
            if(p_arg_->gammaStrides_[NumInvariantDim - 1] != 1)
                return (false);

            if(p_arg_->Lengths_[Rank - 1] % GammaSrcVectorSize != 0)
                return (false);
        }
        else // if fastest dim is reduced
        {
            if(p_arg_->gammaStrides_[Rank - 1] != 1)
                return (false);

            if(p_arg_->Lengths_[Rank - 1] % GammaSrcVectorSize != 0)
                return (false);
        }

        // if fastest dim is not reduced
        if constexpr(BetaSrcVectorDim == 0)
        {
            if(p_arg_->betaStrides_[NumInvariantDim - 1] != 1)
                return (false);

            if(p_arg_->invariant_lowest_length_ % BetaSrcVectorSize != 0)
                return (false);
        }
        else // if fastest dim is reduced
        {
            if(p_arg_->betaStrides_[Rank - 1] != 1)
                return (false);

            if(p_arg_->Lengths_[Rank - 1] % BetaSrcVectorSize != 0)
                return (false);
        }

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
        str << "DeviceNormalizationFwdImpl<" << BlockSize << ",";
        str << "Cluster_MK_" << MThreadClusterSize << "_" << KThreadClusterSize << ",";
        str << "Slice_MK_" << MThreadSliceSize << "_" << KThreadSliceSize << ",";
        str << "XYSrcVectorDim_" << XYSrcVectorDim  << ",";
        str << "VectorSize_X" << XSrcVectorSize << "_Gamma" << GammaSrcVectorSize << "_Beta" << BetaSrcVectorSize << "_Y" << YDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
