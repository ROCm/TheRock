// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <vector>

#include "ck/tensor_operation/gpu/device/device_normalization_bwd_data.hpp"
#include "ck/tensor_operation/gpu/grid/normalization/gridwise_normalization_bwd_data.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_common.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

// M is Invariant dimension, K is reduced dimension
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseNormalizationBwd,
          typename DYDataType,
          typename XDataType,
          typename GammaDataType,
          typename MeanInvStdDataType,
          typename DXDataType,
          typename GridDesc_M_K>
__global__ void
kernel_normalization_bwd_data(const GridDesc_M_K dy_grid_desc_m_k,
                              const GridDesc_M_K x_grid_desc_m_k,
                              const GridDesc_M_K gamma_grid_desc_m_k,
                              const GridDesc_M_K mean_grid_desc_m_k,
                              const GridDesc_M_K inv_std_grid_desc_m_k,
                              const GridDesc_M_K dx_grid_desc_m_k,
                              index_t num_k_block_tile_iteration,
                              const DYDataType* const __restrict__ p_dy_global,
                              const XDataType* const __restrict__ p_x_global,
                              const GammaDataType* const __restrict__ p_gamma_global,
                              const MeanInvStdDataType* const __restrict__ p_mean_global,
                              const MeanInvStdDataType* const __restrict__ p_inv_std_global,
                              DXDataType* const __restrict__ p_dx_global)
{
    GridwiseNormalizationBwd::Run(dy_grid_desc_m_k,
                                  x_grid_desc_m_k,
                                  gamma_grid_desc_m_k,
                                  mean_grid_desc_m_k,
                                  inv_std_grid_desc_m_k,
                                  dx_grid_desc_m_k,
                                  num_k_block_tile_iteration,
                                  p_dy_global,
                                  p_x_global,
                                  p_gamma_global,
                                  p_mean_global,
                                  p_inv_std_global,
                                  p_dx_global);
};

template <typename DYDataType,
          typename XDataType,
          typename GammaDataType,
          typename MeanInvStdDataType,
          typename ComputeDataType,
          typename DXDataType,
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
          bool IsGammaFastestDimReduced,
          index_t GammaSrcVectorSize,
          bool IsMeanInvStdFastestDimReduced,
          index_t MeanInvStdSrcVectorSize,
          bool IsDxFastestDimReduced,
          index_t DXDstVectorSize>
struct DeviceNormalizationBwdDataImpl : public DeviceNormalizationBwdData<DYDataType,
                                                                          XDataType,
                                                                          GammaDataType,
                                                                          MeanInvStdDataType,
                                                                          DXDataType,
                                                                          Rank,
                                                                          NumReduceDim>
{
    static constexpr index_t DYSrcVectorDim         = IsDYFastestDimReduced ? 1 : 0;
    static constexpr index_t XSrcVectorDim          = IsXFastestDimReduced ? 1 : 0;
    static constexpr index_t GammaSrcVectorDim      = IsGammaFastestDimReduced ? 1 : 0;
    static constexpr index_t MeanInvStdSrcVectorDim = IsMeanInvStdFastestDimReduced ? 1 : 0;
    static constexpr index_t DXDstVectorDim         = IsDxFastestDimReduced ? 1 : 0;

    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize);

    static_assert(((DYSrcVectorDim == 0 && MThreadSliceSize % DYSrcVectorSize == 0) ||
                   (DYSrcVectorDim == 1 && KThreadSliceSize % DYSrcVectorSize == 0)),
                  "Invalid thread slice sizes and/or dy vector sizes configuration, please check!");

    static_assert(((XSrcVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0) ||
                   (XSrcVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0)),
                  "Invalid thread slice sizes and/or x vector sizes configuration, please check!");

    static_assert(
        ((GammaSrcVectorDim == 0 && MThreadSliceSize % GammaSrcVectorSize == 0) ||
         (GammaSrcVectorDim == 1 && KThreadSliceSize % GammaSrcVectorSize == 0)),
        "Invalid thread slice sizes and/or gamma vector sizes configuration, please check!");

    static_assert(
        (MeanInvStdSrcVectorDim == 0 && MThreadSliceSize % MeanInvStdSrcVectorSize == 0) ||
            (MeanInvStdSrcVectorDim == 1 && KThreadSliceSize % MeanInvStdSrcVectorSize == 0),
        "Invalid thread slice sizes and/or mean and inverse std vector sizes configuration, please "
        "check!");

    static_assert(((DXDstVectorDim == 0 && MThreadSliceSize % DXDstVectorSize == 0) ||
                   (DXDstVectorDim == 1 && KThreadSliceSize % DXDstVectorSize == 0)),
                  "Invalid thread slice sizes and/or dx vector sizes configuration, please check!");

    static constexpr index_t NumInvariantDim = Rank - NumReduceDim;
    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static constexpr bool reduceAllDim = (NumInvariantDim == 0);
    static_assert(!reduceAllDim);

    static auto Make2dDescriptor(const std::vector<index_t>& lengths,
                                 const std::vector<index_t>& strides,
                                 int numBlockTileIteration)
    {
        const auto tupleLengths = make_tuple_from_array(lengths, Number<Rank>{});
        const auto tupleStrides = make_tuple_from_array(strides, Number<Rank>{});

        const auto desc = make_naive_tensor_descriptor(tupleLengths, tupleStrides);

        const auto grid_desc_m_k = [&]() {
            using InvariantDims = typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type;
            using ReduceDims    = typename arithmetic_sequence_gen<NumInvariantDim, Rank, 1>::type;

            const auto reduceDimLengths =
                make_tuple_from_array_and_index_seq(lengths, ReduceDims{});
            const auto invariantDimLengths =
                make_tuple_from_array_and_index_seq(lengths, InvariantDims{});

            return transform_tensor_descriptor(desc,
                                               make_tuple(make_merge_transform(invariantDimLengths),
                                                          make_merge_transform(reduceDimLengths)),
                                               make_tuple(InvariantDims{}, ReduceDims{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }();

        const auto invariantLength = grid_desc_m_k.GetLength(Number<0>{});
        const auto reduceLength    = grid_desc_m_k.GetLength(Number<1>{});

        const auto pad_M =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;
        const auto pad_K = K_BlockTileSize * numBlockTileIteration - reduceLength;

        auto grid_desc_m_k_padded =
            transform_tensor_descriptor(grid_desc_m_k,
                                        make_tuple(make_right_pad_transform(invariantLength, pad_M),
                                                   make_right_pad_transform(reduceLength, pad_K)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return grid_desc_m_k_padded;
    }

    using GridDesc_M_K = decltype(Make2dDescriptor({1}, {1}, 1));

    using GridwiseNormalizationBwdDataGeneric =
        GridwiseNormalizationBwdData_mk_to_mk<DYDataType,
                                              XDataType,
                                              GammaDataType,
                                              MeanInvStdDataType,
                                              ComputeDataType,
                                              DXDataType,
                                              GridDesc_M_K,
                                              BlockSize,
                                              MThreadClusterSize,
                                              KThreadClusterSize,
                                              MThreadSliceSize,
                                              KThreadSliceSize,
                                              DYSrcVectorDim,
                                              DYSrcVectorSize,
                                              XSrcVectorDim,
                                              XSrcVectorSize,
                                              GammaSrcVectorDim,
                                              GammaSrcVectorSize,
                                              MeanInvStdSrcVectorDim,
                                              MeanInvStdSrcVectorSize,
                                              DXDstVectorDim,
                                              DXDstVectorSize,
                                              false>;

    using GridwiseNormalizationBwdDataSweepOnce =
        GridwiseNormalizationBwdData_mk_to_mk<DYDataType,
                                              XDataType,
                                              GammaDataType,
                                              MeanInvStdDataType,
                                              ComputeDataType,
                                              DXDataType,
                                              GridDesc_M_K,
                                              BlockSize,
                                              MThreadClusterSize,
                                              KThreadClusterSize,
                                              MThreadSliceSize,
                                              KThreadSliceSize,
                                              DYSrcVectorDim,
                                              DYSrcVectorSize,
                                              XSrcVectorDim,
                                              XSrcVectorSize,
                                              GammaSrcVectorDim,
                                              GammaSrcVectorSize,
                                              MeanInvStdSrcVectorDim,
                                              MeanInvStdSrcVectorSize,
                                              DXDstVectorDim,
                                              DXDstVectorSize,
                                              true>;

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<index_t> lengths,
                 const std::vector<index_t> dyStrides,
                 const std::vector<index_t> xStrides,
                 const std::vector<index_t> gammaStrides,
                 const std::vector<index_t> meanStrides,
                 const std::vector<index_t> invStdStrides,
                 const std::vector<index_t> dxStrides,
                 const std::vector<index_t> reduceDims,
                 const DYDataType* p_dy,
                 const XDataType* p_x,
                 const GammaDataType* p_gamma,
                 const MeanInvStdDataType* p_mean,
                 const MeanInvStdDataType* p_invStd,
                 DXDataType* p_dx)
            : p_dy_(p_dy),
              p_x_(p_x),
              p_gamma_(p_gamma),
              p_mean_(p_mean),
              p_invStd_(p_invStd),
              p_dx_(p_dx)
        {
            lengths_      = shuffle_tensor_dimensions<Rank, NumReduceDim>(lengths, reduceDims);
            dyStrides_    = shuffle_tensor_dimensions<Rank, NumReduceDim>(dyStrides, reduceDims);
            xStrides_     = shuffle_tensor_dimensions<Rank, NumReduceDim>(xStrides, reduceDims);
            gammaStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(gammaStrides, reduceDims);
            meanStrides_  = shuffle_tensor_dimensions<Rank, NumReduceDim>(meanStrides, reduceDims);
            invStdStrides_ =
                shuffle_tensor_dimensions<Rank, NumReduceDim>(invStdStrides, reduceDims);
            dxStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(dxStrides, reduceDims);

            std::tie(MRaw_, KRaw_) = get_2d_lengths<Rank, NumReduceDim>(lengths_);

            numBlockTileIteration_ = math::integer_divide_ceil(KRaw_, K_BlockTileSize);

            gridSize_ = math::integer_divide_ceil(MRaw_, M_BlockTileSize);

            dy_grid_desc_m_k_ = Make2dDescriptor(lengths_, dyStrides_, numBlockTileIteration_);
            x_grid_desc_m_k_  = Make2dDescriptor(lengths_, xStrides_, numBlockTileIteration_);
            gamma_grid_desc_m_k_ =
                Make2dDescriptor(lengths_, gammaStrides_, numBlockTileIteration_);
            mean_grid_desc_m_k_ = Make2dDescriptor(lengths_, meanStrides_, numBlockTileIteration_);
            inv_std_grid_desc_m_k_ =
                Make2dDescriptor(lengths_, invStdStrides_, numBlockTileIteration_);
            dx_grid_desc_m_k_ = Make2dDescriptor(lengths_, dxStrides_, numBlockTileIteration_);

            isSweeponce_ = dy_grid_desc_m_k_.GetLength(Number<1>{}) <= K_BlockTileSize;
        }

        const DYDataType* p_dy_;
        const XDataType* p_x_;
        const GammaDataType* p_gamma_;
        const MeanInvStdDataType* p_mean_;
        const MeanInvStdDataType* p_invStd_;
        DXDataType* p_dx_;

        std::vector<index_t> lengths_;
        std::vector<index_t> dyStrides_;
        std::vector<index_t> xStrides_;
        std::vector<index_t> gammaStrides_;
        std::vector<index_t> meanStrides_;
        std::vector<index_t> invStdStrides_;
        std::vector<index_t> dxStrides_;

        int numBlockTileIteration_;
        size_t gridSize_;

        // tensor descriptor
        GridDesc_M_K dy_grid_desc_m_k_;
        GridDesc_M_K x_grid_desc_m_k_;
        GridDesc_M_K gamma_grid_desc_m_k_;
        GridDesc_M_K mean_grid_desc_m_k_;
        GridDesc_M_K inv_std_grid_desc_m_k_;
        GridDesc_M_K dx_grid_desc_m_k_;

        bool isSweeponce_;
        index_t MRaw_; // Invariant length
        index_t KRaw_; // reduce length
    };

    struct Invoker : public BaseInvoker
    {
        auto KernelSelector(bool isSweepOnce)
        {
            return isSweepOnce
                       ? kernel_normalization_bwd_data<GridwiseNormalizationBwdDataSweepOnce,
                                                       DYDataType,
                                                       XDataType,
                                                       GammaDataType,
                                                       MeanInvStdDataType,
                                                       DXDataType,
                                                       GridDesc_M_K>
                       : kernel_normalization_bwd_data<GridwiseNormalizationBwdDataGeneric,
                                                       DYDataType,
                                                       XDataType,
                                                       GammaDataType,
                                                       MeanInvStdDataType,
                                                       DXDataType,
                                                       GridDesc_M_K>;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto kernel_main = KernelSelector(arg.isSweeponce_);

            return launch_and_time_kernel(stream_config,
                                          kernel_main,
                                          dim3(arg.gridSize_),
                                          dim3(BlockSize),
                                          0,
                                          arg.dy_grid_desc_m_k_,
                                          arg.x_grid_desc_m_k_,
                                          arg.gamma_grid_desc_m_k_,
                                          arg.mean_grid_desc_m_k_,
                                          arg.inv_std_grid_desc_m_k_,
                                          arg.dx_grid_desc_m_k_,
                                          arg.numBlockTileIteration_,
                                          arg.p_dy_,
                                          arg.p_x_,
                                          arg.p_gamma_,
                                          arg.p_mean_,
                                          arg.p_invStd_,
                                          arg.p_dx_);
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    template <index_t SrcVectorDim, index_t SrcVectorSize>
    bool IsVectorDimSizeValid(const std::vector<index_t>& lengths,
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

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* p_arg_ = dynamic_cast<const Argument*>(p_arg);

        bool pass = true;
        pass &= IsVectorDimSizeValid<DYSrcVectorDim, DYSrcVectorSize>(p_arg_->lengths_,
                                                                      p_arg_->dyStrides_);
        pass &= IsVectorDimSizeValid<XSrcVectorDim, XSrcVectorSize>(p_arg_->lengths_,
                                                                    p_arg_->xStrides_);
        pass &= IsVectorDimSizeValid<GammaSrcVectorDim, GammaSrcVectorSize>(p_arg_->lengths_,
                                                                            p_arg_->gammaStrides_);
        pass &= IsVectorDimSizeValid<MeanInvStdSrcVectorDim, MeanInvStdSrcVectorSize>(
            p_arg_->lengths_, p_arg_->meanStrides_);
        pass &= IsVectorDimSizeValid<MeanInvStdSrcVectorDim, MeanInvStdSrcVectorSize>(
            p_arg_->lengths_, p_arg_->invStdStrides_);

        pass &= IsVectorDimSizeValid<DXDstVectorDim, DXDstVectorSize>(p_arg_->lengths_,
                                                                      p_arg_->dxStrides_);
        return pass;
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const std::vector<index_t> lengths,
                                                      const std::vector<index_t> dyStrides,
                                                      const std::vector<index_t> xStrides,
                                                      const std::vector<index_t> gammaStrides,
                                                      const std::vector<index_t> meanStrides,
                                                      const std::vector<index_t> invStdStrides,
                                                      const std::vector<index_t> dxStrides,
                                                      const std::vector<index_t> reduceDims,
                                                      const void* p_dy,
                                                      const void* p_x,
                                                      const void* p_gamma,
                                                      const void* p_mean,
                                                      const void* p_invStd,
                                                      void* p_dx) override
    {
        if(lengths.size() != Rank || dyStrides.size() != Rank || xStrides.size() != Rank ||
           gammaStrides.size() != Rank || meanStrides.size() != Rank ||
           invStdStrides.size() != Rank || dxStrides.size() != Rank)
            throw std::runtime_error("dimension is incorrect");

        return std::make_unique<Argument>(lengths,
                                          dyStrides,
                                          xStrides,
                                          gammaStrides,
                                          meanStrides,
                                          invStdStrides,
                                          dxStrides,
                                          reduceDims,
                                          static_cast<const DYDataType*>(p_dy),
                                          static_cast<const XDataType*>(p_x),
                                          static_cast<const GammaDataType*>(p_gamma),
                                          static_cast<const MeanInvStdDataType*>(p_mean),
                                          static_cast<const MeanInvStdDataType*>(p_invStd),
                                          static_cast<DXDataType*>(p_dx));
    }

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceNormalizationBwdDataImpl<"  << BlockSize << ",";
        str << "Cluster_MK_" << MThreadClusterSize << "_" << KThreadClusterSize << ",";
        str << "Slice_MK_" << MThreadSliceSize << "_" << KThreadSliceSize << ",";
        str << "DYSrcVectorSize" << DYSrcVectorSize << "_X" << XSrcVectorSize << "_Gamma" << GammaSrcVectorSize << "_MeanRstd" << MeanInvStdSrcVectorSize  << "_Dx" << DXDstVectorSize;
        str << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
