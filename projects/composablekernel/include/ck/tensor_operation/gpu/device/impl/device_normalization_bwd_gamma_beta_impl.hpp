// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <vector>

#include "ck/tensor_operation/gpu/device/device_normalization_bwd_gamma_beta.hpp"
#include "ck/tensor_operation/gpu/grid/normalization/gridwise_normalization_bwd_gamma_beta.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_common.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

// M is Invariant dimension, K is reduced dimension
namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseReduction,
          typename DYDataType,
          typename XDataType,
          typename MeanInvStdDataType,
          typename DGammaDataType,
          typename DBetaDataType,
          typename GridDesc_M_K,
          typename GridDesc_M>
__global__ void
kernel_normalization_bwd_gamma_beta(const GridDesc_M_K dy_grid_desc_m_k,
                                    const GridDesc_M_K x_grid_desc_m_k,
                                    const GridDesc_M_K mean_grid_desc_m_k,
                                    const GridDesc_M_K inv_std_grid_desc_m_k,
                                    const GridDesc_M dgamma_grid_desc_m,
                                    const GridDesc_M dbeta_grid_desc_m,
                                    index_t num_k_block_tile_iteration,
                                    const DYDataType* const __restrict__ p_dy_global,
                                    const XDataType* const __restrict__ p_x_global,
                                    const MeanInvStdDataType* const __restrict__ p_mean_global,
                                    const MeanInvStdDataType* const __restrict__ p_inv_std_global,
                                    DGammaDataType* const __restrict__ p_dgamma_global,
                                    DBetaDataType* const __restrict__ p_dbeta_global)
{
    GridwiseReduction::Run(dy_grid_desc_m_k,
                           x_grid_desc_m_k,
                           mean_grid_desc_m_k,
                           inv_std_grid_desc_m_k,
                           dgamma_grid_desc_m,
                           dbeta_grid_desc_m,
                           num_k_block_tile_iteration,
                           p_dy_global,
                           p_x_global,
                           p_mean_global,
                           p_inv_std_global,
                           p_dgamma_global,
                           p_dbeta_global);
};

template <typename DYDataType,
          typename XDataType,
          typename MeanInvStdDataType,
          typename ComputeDataType,
          typename DGammaDataType,
          typename DBetaDataType,
          index_t Rank,
          index_t NumReduceDim,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          bool IsDYFastestDimReduced,
          index_t DYSrcVectorSize,
          bool IsXFastestDimReduced,
          index_t XSrcVectorSize,
          bool IsMeanInvStdFastestDimReduced,
          index_t MeanInvStdSrcVectorSize,
          index_t DGammaDstVectorSize,
          index_t DBetaDstVectorSize>
struct DeviceNormalizationBwdGammaBetaImpl
    : public DeviceNormalizationBwdGammaBeta<DYDataType,
                                             XDataType,
                                             MeanInvStdDataType,
                                             DGammaDataType,
                                             DBetaDataType,
                                             Rank,
                                             NumReduceDim>
{
    static constexpr index_t DYSrcVectorDim         = IsDYFastestDimReduced ? 1 : 0;
    static constexpr index_t XSrcVectorDim          = IsXFastestDimReduced ? 1 : 0;
    static constexpr index_t MeanInvStdSrcVectorDim = IsMeanInvStdFastestDimReduced ? 1 : 0;

    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize);

    static_assert(((DYSrcVectorDim == 0 && MThreadSliceSize % DYSrcVectorSize == 0) ||
                   (DYSrcVectorDim == 1 && KThreadSliceSize % DYSrcVectorSize == 0)),
                  "Invalid thread slice sizes and/or dy vector sizes configuration, please check!");

    static_assert(((XSrcVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0) ||
                   (XSrcVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0)),
                  "Invalid thread slice sizes and/or x vector sizes configuration, please check!");

    static_assert(
        (MeanInvStdSrcVectorDim == 0 && MThreadSliceSize % MeanInvStdSrcVectorSize == 0) ||
            (MeanInvStdSrcVectorDim == 1 && KThreadSliceSize % MeanInvStdSrcVectorSize == 0),
        "Invalid thread slice sizes and/or mean and inverse std vector sizes configuration, please "
        "check!");

    static_assert(
        ((MThreadSliceSize % DGammaDstVectorSize == 0) ||
         (MThreadSliceSize % DBetaDstVectorSize == 0)),
        "Invalid thread slice sizes and/or Gamma and beta vector sizes configuration, please "
        "check!");

    static constexpr index_t NumInvariantDim = Rank - NumReduceDim;
    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static constexpr bool reduceAllDim = (NumInvariantDim == 0);
    static_assert(!reduceAllDim);

    static auto MakeSrc2dDescriptor(const std::vector<index_t>& inLengths,
                                    const std::vector<index_t>& inStrides,
                                    int numBlockTileIteration)
    {
        const auto tupleSrcLengths = make_tuple_from_array(inLengths, Number<Rank>{});
        const auto tupleSrcStrides = make_tuple_from_array(inStrides, Number<Rank>{});

        const auto inDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto in_grid_desc_m_k = [&]() {
            using InvariantDims = typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type;
            using ReduceDims    = typename arithmetic_sequence_gen<NumInvariantDim, Rank, 1>::type;

            const auto reduceDimLengths =
                make_tuple_from_array_and_index_seq(inLengths, ReduceDims{});
            const auto invariantDimLengths =
                make_tuple_from_array_and_index_seq(inLengths, InvariantDims{});

            return transform_tensor_descriptor(inDesc,
                                               make_tuple(make_merge_transform(invariantDimLengths),
                                                          make_merge_transform(reduceDimLengths)),
                                               make_tuple(InvariantDims{}, ReduceDims{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
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

        return in_grid_desc_m_k_padded;
    }

    static auto MakeDst1dDescriptor(const std::vector<index_t>& outLengths,
                                    const std::vector<index_t>& outStrides)
    {
        const auto tupleDstLengths =
            generate_tuple([&](auto I) { return outLengths[I]; }, Number<NumInvariantDim>{});
        const auto tupleDstStrides =
            generate_tuple([&](auto I) { return outStrides[I]; }, Number<NumInvariantDim>{});

        auto outDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

        auto out_grid_desc_m = transform_tensor_descriptor(
            outDesc,
            make_tuple(make_merge_transform(tupleDstLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto invariantLength = out_grid_desc_m.GetLength(Number<0>{});

        const auto outPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;

        auto out_grid_desc_m_padded = transform_tensor_descriptor(
            out_grid_desc_m,
            make_tuple(make_right_pad_transform(invariantLength, outPad)),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0>{}));
        return (out_grid_desc_m_padded);
    };

    using GridDesc_M_K = decltype(MakeSrc2dDescriptor({1}, {1}, 1));
    using GridDesc_M   = decltype(MakeDst1dDescriptor({1}, {1}));

    using GridwiseNormalizationBwdGammaBeta =
        GridwiseNormalizationBwdGammaBeta_mk_to_k<DYDataType,
                                                  XDataType,
                                                  MeanInvStdDataType,
                                                  ComputeDataType,
                                                  DGammaDataType,
                                                  DBetaDataType,
                                                  GridDesc_M_K,
                                                  GridDesc_M,
                                                  BlockSize,
                                                  MThreadClusterSize,
                                                  KThreadClusterSize,
                                                  MThreadSliceSize,
                                                  KThreadSliceSize,
                                                  DYSrcVectorDim,
                                                  DYSrcVectorSize,
                                                  XSrcVectorDim,
                                                  XSrcVectorSize,
                                                  MeanInvStdSrcVectorDim,
                                                  MeanInvStdSrcVectorSize,
                                                  DGammaDstVectorSize,
                                                  DBetaDstVectorSize>;

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<index_t> inLengths,
                 const std::vector<index_t> dyStrides,
                 const std::vector<index_t> xStrides,
                 const std::vector<index_t> meanStrides,
                 const std::vector<index_t> invStdStrides,
                 const std::vector<index_t> outLengths,
                 const std::vector<index_t> dgammaStrides,
                 const std::vector<index_t> dbetaStrides,
                 const std::vector<index_t> reduceDims,
                 const DYDataType* p_dy,
                 const XDataType* p_x,
                 const MeanInvStdDataType* p_mean,
                 const MeanInvStdDataType* p_invStd,
                 DGammaDataType* p_dgamma,
                 DBetaDataType* p_dbeta)
            : p_dy_(p_dy),
              p_x_(p_x),
              p_mean_(p_mean),
              p_invStd_(p_invStd),
              p_dgamma_(p_dgamma),
              p_dbeta_(p_dbeta),
              outLengths_{outLengths},
              dgammaStrides_{dgammaStrides},
              dbetaStrides_{dbetaStrides}
        {
            inLengths_   = shuffle_tensor_dimensions<Rank, NumReduceDim>(inLengths, reduceDims);
            dyStrides_   = shuffle_tensor_dimensions<Rank, NumReduceDim>(dyStrides, reduceDims);
            xStrides_    = shuffle_tensor_dimensions<Rank, NumReduceDim>(xStrides, reduceDims);
            meanStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(meanStrides, reduceDims);
            invStdStrides_ =
                shuffle_tensor_dimensions<Rank, NumReduceDim>(invStdStrides, reduceDims);

            std::tie(MRaw_, KRaw_) = get_2d_lengths<Rank, NumReduceDim>(inLengths_);

            numBlockTileIteration_ = math::integer_divide_ceil(KRaw_, K_BlockTileSize);

            gridSize_ = math::integer_divide_ceil(MRaw_, M_BlockTileSize);

            dy_grid_desc_m_k_ = MakeSrc2dDescriptor(inLengths_, dyStrides_, numBlockTileIteration_);
            x_grid_desc_m_k_  = MakeSrc2dDescriptor(inLengths_, xStrides_, numBlockTileIteration_);
            mean_grid_desc_m_k_ =
                MakeSrc2dDescriptor(inLengths_, meanStrides_, numBlockTileIteration_);
            inv_std_grid_desc_m_k_ =
                MakeSrc2dDescriptor(inLengths_, invStdStrides_, numBlockTileIteration_);

            dgamma_grid_desc_m_ = MakeDst1dDescriptor(outLengths_, dgammaStrides_);
            dbeta_grid_desc_m_  = MakeDst1dDescriptor(outLengths_, dbetaStrides_);
        }

        const DYDataType* p_dy_;
        const XDataType* p_x_;
        const MeanInvStdDataType* p_mean_;
        const MeanInvStdDataType* p_invStd_;
        DGammaDataType* p_dgamma_;
        DBetaDataType* p_dbeta_;

        std::vector<index_t> inLengths_;
        std::vector<index_t> dyStrides_;
        std::vector<index_t> xStrides_;
        std::vector<index_t> meanStrides_;
        std::vector<index_t> invStdStrides_;
        std::vector<index_t> outLengths_;
        std::vector<index_t> dgammaStrides_;
        std::vector<index_t> dbetaStrides_;

        int numBlockTileIteration_;
        size_t gridSize_;

        // Source descriptor
        GridDesc_M_K dy_grid_desc_m_k_;
        GridDesc_M_K x_grid_desc_m_k_;
        GridDesc_M_K mean_grid_desc_m_k_;
        GridDesc_M_K inv_std_grid_desc_m_k_;

        // Destination descriptor
        GridDesc_M dgamma_grid_desc_m_;
        GridDesc_M dbeta_grid_desc_m_;

        index_t MRaw_; // Invariant length
        index_t KRaw_; // reduce length
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto kernel_main =
                kernel_normalization_bwd_gamma_beta<GridwiseNormalizationBwdGammaBeta,
                                                    DYDataType,
                                                    XDataType,
                                                    MeanInvStdDataType,
                                                    DGammaDataType,
                                                    DBetaDataType,
                                                    GridDesc_M_K,
                                                    GridDesc_M>;

            return launch_and_time_kernel(stream_config,
                                          kernel_main,
                                          dim3(arg.gridSize_),
                                          dim3(BlockSize),
                                          0,
                                          arg.dy_grid_desc_m_k_,
                                          arg.x_grid_desc_m_k_,
                                          arg.mean_grid_desc_m_k_,
                                          arg.inv_std_grid_desc_m_k_,
                                          arg.dgamma_grid_desc_m_,
                                          arg.dbeta_grid_desc_m_,
                                          arg.numBlockTileIteration_,
                                          arg.p_dy_,
                                          arg.p_x_,
                                          arg.p_mean_,
                                          arg.p_invStd_,
                                          arg.p_dgamma_,
                                          arg.p_dbeta_);
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    template <index_t SrcVectorDim, index_t SrcVectorSize>
    bool IsSrcVectorDimSizeValid(const std::vector<index_t>& lengths,
                                 const std::vector<index_t>& strides)
    {
        if constexpr(SrcVectorSize == 1)
            return true;

        // Fastest dimension is not reduced
        if constexpr(SrcVectorDim == 0)
        {
            if constexpr(NumInvariantDim == 0)
                return false;

            if(strides[NumInvariantDim - 1] != 1)
                return false;

            if(lengths[NumInvariantDim - 1] % SrcVectorSize != 0)
                return false;
        }
        else // Fastest dimension is reduced
        {
            if(strides[Rank - 1] != 1)
                return false;

            if(lengths[Rank - 1] % SrcVectorSize != 0)
                return false;
        };

        return true;
    }

    template <index_t DstVectorSize>
    bool IsDstVectorSizeValid(const std::vector<index_t>& lengths,
                              const std::vector<index_t>& strides)
    {
        if constexpr(DstVectorSize == 1)
            return true;

        if(strides[NumInvariantDim - 1] != 1)
            return false;

        if(lengths[NumInvariantDim - 1] % DstVectorSize != 0)
            return false;

        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* p_arg_ = dynamic_cast<const Argument*>(p_arg);

        bool pass = true;
        pass &= IsSrcVectorDimSizeValid<DYSrcVectorDim, DYSrcVectorSize>(p_arg_->inLengths_,
                                                                         p_arg_->dyStrides_);
        pass &= IsSrcVectorDimSizeValid<XSrcVectorDim, XSrcVectorSize>(p_arg_->inLengths_,
                                                                       p_arg_->xStrides_);
        pass &= IsSrcVectorDimSizeValid<MeanInvStdSrcVectorDim, MeanInvStdSrcVectorSize>(
            p_arg_->inLengths_, p_arg_->meanStrides_);
        pass &= IsSrcVectorDimSizeValid<MeanInvStdSrcVectorDim, MeanInvStdSrcVectorSize>(
            p_arg_->inLengths_, p_arg_->invStdStrides_);

        pass &=
            IsDstVectorSizeValid<DGammaDstVectorSize>(p_arg_->outLengths_, p_arg_->dgammaStrides_);
        pass &=
            IsDstVectorSizeValid<DBetaDstVectorSize>(p_arg_->outLengths_, p_arg_->dbetaStrides_);

        return pass;
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const std::vector<index_t> inLengths,
                                                      const std::vector<index_t> dyStrides,
                                                      const std::vector<index_t> xStrides,
                                                      const std::vector<index_t> meanStrides,
                                                      const std::vector<index_t> invStdStrides,
                                                      const std::vector<index_t> outLengths,
                                                      const std::vector<index_t> dgammaStrides,
                                                      const std::vector<index_t> dbetaStrides,
                                                      const std::vector<index_t> reduceDims,
                                                      const void* p_dy,
                                                      const void* p_x,
                                                      const void* p_mean,
                                                      const void* p_invStd,
                                                      void* p_dgamma,
                                                      void* p_dbeta) override
    {
        if(inLengths.size() != Rank || dyStrides.size() != Rank || xStrides.size() != Rank ||
           meanStrides.size() != Rank || invStdStrides.size() != Rank)
            throw std::runtime_error("dimension is incorrect");

        if(outLengths.size() != NumInvariantDim || dgammaStrides.size() != NumInvariantDim ||
           dbetaStrides.size() != NumInvariantDim)
            throw std::runtime_error("dimension is incorrect");

        return std::make_unique<Argument>(inLengths,
                                          dyStrides,
                                          xStrides,
                                          meanStrides,
                                          invStdStrides,
                                          outLengths,
                                          dgammaStrides,
                                          dbetaStrides,
                                          reduceDims,
                                          static_cast<const DYDataType*>(p_dy),
                                          static_cast<const XDataType*>(p_x),
                                          static_cast<const MeanInvStdDataType*>(p_mean),
                                          static_cast<const MeanInvStdDataType*>(p_invStd),
                                          static_cast<DGammaDataType*>(p_dgamma),
                                          static_cast<DBetaDataType*>(p_dbeta));
    }

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceNormalizationBwdGammaBetaImpl<" << BlockSize << ",";
        str << "Cluster_MK_" << MThreadClusterSize << "_" << KThreadClusterSize << ",";
        str << "Slice_MK_" << MThreadSliceSize << "_" << KThreadSliceSize << ",";
        str << "VectorSize_DY" << DYSrcVectorSize << "_X" << XSrcVectorSize ;
        str << "_DGamma" << DGammaDstVectorSize << "_DBeta" << DBetaDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
