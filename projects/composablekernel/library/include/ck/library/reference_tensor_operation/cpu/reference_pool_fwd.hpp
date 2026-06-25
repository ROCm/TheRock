// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <index_t InOutRank,
          index_t WindowRank,
          typename InDataType,
          typename OutDataType,
          typename ComputeDataType,
          typename IndexDataType,
          ck::ReduceTensorOp ReduceOpId,
          bool PropagateNan,
          bool OutputIndex>
struct ReferencePoolingFwd : public device::BaseOperator
{
    using ReduceOperation = typename ck::reduce_binary_operator<ReduceOpId>::opType;

    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<InDataType>& in,
                 Tensor<OutDataType>& out,
                 Tensor<IndexDataType>& out_indices,
                 const std::vector<ck::index_t>& window_spatial_lengths,
                 const std::vector<ck::index_t>& window_strides,
                 const std::vector<ck::index_t>& window_dilations,
                 const std::vector<ck::index_t>& in_left_pads,
                 const std::vector<ck::index_t>& /*in_right_pads*/)
            : in_(in),
              out_(out),
              out_indices_(out_indices),
              window_spatial_lengths_(window_spatial_lengths),
              window_strides_(window_strides),
              window_dilations_(window_dilations),
              in_left_pads_(in_left_pads),
              reduceLength_(1)
        {
            static_for<0, WindowRank, 1>{}(
                [&](auto I) { reduceLength_ *= window_spatial_lengths[I]; });
        }

        const Tensor<InDataType>& in_;
        Tensor<OutDataType>& out_;
        Tensor<IndexDataType>& out_indices_;
        const std::vector<ck::index_t>& window_spatial_lengths_;
        const std::vector<ck::index_t>& window_strides_;
        const std::vector<ck::index_t>& window_dilations_;
        const std::vector<ck::index_t>& in_left_pads_;
        int reduceLength_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float RunPooling3dFwd(const Argument& arg)
        {

            auto elementwise_ops =
                ck::reduce_unary_operator<ReduceOpId, true, true>::GetElementwiseOperator(
                    arg.reduceLength_);

            auto in_elementwise_op  = std::get<0>(elementwise_ops);
            auto acc_elementwise_op = std::get<1>(elementwise_ops);

            if constexpr(!OutputIndex)
            {
                using Accumulation = ck::detail::
                    AccumulateWithNanCheck<PropagateNan, ReduceOperation, ComputeDataType>;

                auto f_ncdhw = [&](auto n, auto c, auto do_, auto ho, auto wo) {
                    auto accuVal = ReduceOperation::template GetIdentityValue<ComputeDataType>();

                    for(ck::index_t z = 0; z < arg.window_spatial_lengths_[0]; ++z)
                    {
                        ck::index_t di = do_ * arg.window_strides_[0] +
                                         z * arg.window_dilations_[0] - arg.in_left_pads_[0];
                        for(ck::index_t y = 0; y < arg.window_spatial_lengths_[1]; ++y)
                        {
                            ck::index_t hi = ho * arg.window_strides_[1] +
                                             y * arg.window_dilations_[1] - arg.in_left_pads_[1];
                            for(ck::index_t x = 0; x < arg.window_spatial_lengths_[2]; ++x)
                            {
                                ck::index_t wi = wo * arg.window_strides_[2] +
                                                 x * arg.window_dilations_[2] -
                                                 arg.in_left_pads_[2];
                                if(di >= 0 &&
                                   di < static_cast<ck::index_t>(arg.in_.mDesc.GetLengths()[2]) &&
                                   hi >= 0 &&
                                   hi < static_cast<ck::index_t>(arg.in_.mDesc.GetLengths()[3]) &&
                                   wi >= 0 &&
                                   wi < static_cast<ck::index_t>(arg.in_.mDesc.GetLengths()[4]))
                                {
                                    ComputeDataType currVal = ck::type_convert<ComputeDataType>(
                                        arg.in_(n, c, di, hi, wi));

                                    in_elementwise_op(currVal, currVal);

                                    Accumulation::Calculate(accuVal, currVal);
                                }
                            }
                        }
                    }
                    acc_elementwise_op(accuVal, accuVal);

                    arg.out_(n, c, do_, ho, wo) = ck::type_convert<OutDataType>(accuVal);
                };

                make_ParallelTensorFunctor(f_ncdhw,
                                           arg.out_.mDesc.GetLengths()[0],
                                           arg.out_.mDesc.GetLengths()[1],
                                           arg.out_.mDesc.GetLengths()[2],
                                           arg.out_.mDesc.GetLengths()[3],
                                           arg.out_.mDesc.GetLengths()[4])(
                    std::thread::hardware_concurrency());
            }
            else
            {
                using Accumulation = ck::detail::AccumulateWithIndexAndNanCheck<PropagateNan,
                                                                                ReduceOperation,
                                                                                ComputeDataType,
                                                                                IndexDataType>;

                auto f_ncdhw = [&](auto n, auto c, auto do_, auto ho, auto wo) {
                    auto accuVal = ReduceOperation::template GetIdentityValue<ComputeDataType>();
                    IndexDataType accuIndex = 0;

                    for(ck::index_t z = 0; z < arg.window_spatial_lengths_[0]; ++z)
                    {
                        ck::index_t di = do_ * arg.window_strides_[0] +
                                         z * arg.window_dilations_[0] - arg.in_left_pads_[0];
                        for(ck::index_t y = 0; y < arg.window_spatial_lengths_[1]; ++y)
                        {
                            ck::index_t hi = ho * arg.window_strides_[1] +
                                             y * arg.window_dilations_[1] - arg.in_left_pads_[1];
                            for(ck::index_t x = 0; x < arg.window_spatial_lengths_[2]; ++x)
                            {
                                ck::index_t wi = wo * arg.window_strides_[2] +
                                                 x * arg.window_dilations_[2] -
                                                 arg.in_left_pads_[2];
                                if(di >= 0 &&
                                   di < static_cast<ck::index_t>(arg.in_.mDesc.GetLengths()[2]) &&
                                   hi >= 0 &&
                                   hi < static_cast<ck::index_t>(arg.in_.mDesc.GetLengths()[3]) &&
                                   wi >= 0 &&
                                   wi < static_cast<ck::index_t>(arg.in_.mDesc.GetLengths()[4]))
                                {
                                    ComputeDataType currVal = ck::type_convert<ComputeDataType>(
                                        arg.in_(n, c, di, hi, wi));
                                    IndexDataType currIndex =
                                        arg.in_.GetOffsetFromMultiIndex(n, c, di, hi, wi);

                                    in_elementwise_op(currVal, currVal);

                                    Accumulation::Calculate(accuVal, currVal, accuIndex, currIndex);
                                }
                            }
                        }
                    }

                    acc_elementwise_op(accuVal, accuVal);

                    arg.out_(n, c, do_, ho, wo)         = ck::type_convert<OutDataType>(accuVal);
                    arg.out_indices_(n, c, do_, ho, wo) = accuIndex;
                };

                make_ParallelTensorFunctor(f_ncdhw,
                                           arg.out_.mDesc.GetLengths()[0],
                                           arg.out_.mDesc.GetLengths()[1],
                                           arg.out_.mDesc.GetLengths()[2],
                                           arg.out_.mDesc.GetLengths()[3],
                                           arg.out_.mDesc.GetLengths()[4])(
                    std::thread::hardware_concurrency());
            };

            return 0;
        }

        float RunPooling2dFwd(const Argument& arg)
        {

            auto elementwise_ops =
                ck::reduce_unary_operator<ReduceOpId, true, true>::GetElementwiseOperator(
                    arg.reduceLength_);

            auto in_elementwise_op  = std::get<0>(elementwise_ops);
            auto acc_elementwise_op = std::get<1>(elementwise_ops);

            if constexpr(!OutputIndex)
            {
                using Accumulation = ck::detail::
                    AccumulateWithNanCheck<PropagateNan, ReduceOperation, ComputeDataType>;

                auto f_nchw = [&](auto n, auto c, auto ho, auto wo) {
                    auto accuVal = ReduceOperation::template GetIdentityValue<ComputeDataType>();

                    for(ck::index_t y = 0; y < arg.window_spatial_lengths_[0]; ++y)
                    {
                        ck::index_t hi = ho * arg.window_strides_[0] +
                                         y * arg.window_dilations_[0] - arg.in_left_pads_[0];
                        for(ck::index_t x = 0; x < arg.window_spatial_lengths_[1]; ++x)
                        {
                            ck::index_t wi = wo * arg.window_strides_[1] +
                                             x * arg.window_dilations_[1] - arg.in_left_pads_[1];
                            if(hi >= 0 &&
                               hi < static_cast<ck::index_t>(arg.in_.mDesc.GetLengths()[2]) &&
                               wi >= 0 &&
                               wi < static_cast<ck::index_t>(arg.in_.mDesc.GetLengths()[3]))
                            {
                                ComputeDataType currVal =
                                    ck::type_convert<ComputeDataType>(arg.in_(n, c, hi, wi));

                                in_elementwise_op(currVal, currVal);

                                Accumulation::Calculate(accuVal, currVal);
                            }
                        }
                    }

                    acc_elementwise_op(accuVal, accuVal);
                    arg.out_(n, c, ho, wo) = ck::type_convert<OutDataType>(accuVal);
                };

                make_ParallelTensorFunctor(f_nchw,
                                           arg.out_.mDesc.GetLengths()[0],
                                           arg.out_.mDesc.GetLengths()[1],
                                           arg.out_.mDesc.GetLengths()[2],
                                           arg.out_.mDesc.GetLengths()[3])(
                    std::thread::hardware_concurrency());
            }
            else
            {
                using Accumulation = ck::detail::AccumulateWithIndexAndNanCheck<PropagateNan,
                                                                                ReduceOperation,
                                                                                ComputeDataType,
                                                                                IndexDataType>;

                auto f_nchw = [&](auto n, auto c, auto ho, auto wo) {
                    auto accuVal = ReduceOperation::template GetIdentityValue<ComputeDataType>();
                    IndexDataType accuIndex = 0;

                    for(ck::index_t y = 0; y < arg.window_spatial_lengths_[0]; ++y)
                    {
                        ck::index_t hi = ho * arg.window_strides_[0] +
                                         y * arg.window_dilations_[0] - arg.in_left_pads_[0];
                        for(ck::index_t x = 0; x < arg.window_spatial_lengths_[1]; ++x)
                        {
                            ck::index_t wi = wo * arg.window_strides_[1] +
                                             x * arg.window_dilations_[1] - arg.in_left_pads_[1];
                            if(hi >= 0 &&
                               hi < static_cast<ck::index_t>(arg.in_.mDesc.GetLengths()[2]) &&
                               wi >= 0 &&
                               wi < static_cast<ck::index_t>(arg.in_.mDesc.GetLengths()[3]))
                            {
                                ComputeDataType currVal =
                                    ck::type_convert<ComputeDataType>(arg.in_(n, c, hi, wi));

                                IndexDataType currIndex =
                                    arg.in_.GetOffsetFromMultiIndex(n, c, hi, wi);

                                in_elementwise_op(currVal, currVal);

                                Accumulation::Calculate(accuVal, currVal, accuIndex, currIndex);
                            }
                        }
                    }

                    acc_elementwise_op(accuVal, accuVal);
                    arg.out_(n, c, ho, wo)         = ck::type_convert<OutDataType>(accuVal);
                    arg.out_indices_(n, c, ho, wo) = accuIndex;
                };

                make_ParallelTensorFunctor(f_nchw,
                                           arg.out_.mDesc.GetLengths()[0],
                                           arg.out_.mDesc.GetLengths()[1],
                                           arg.out_.mDesc.GetLengths()[2],
                                           arg.out_.mDesc.GetLengths()[3])(
                    std::thread::hardware_concurrency());
            };

            return 0;
        }

        float Run(const Argument& arg)
        {
            // TODO - support generic pooling
            if constexpr(InOutRank == 5 && WindowRank == 3)
                return RunPooling3dFwd(arg);
            else if constexpr(InOutRank == 4 && WindowRank == 2)
                return RunPooling2dFwd(arg);
            else
                throw std::runtime_error("Only support pooling3d or pooling2d so far");
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /* stream_config */ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(const Tensor<InDataType>& in,
                             Tensor<OutDataType>& out,
                             Tensor<IndexDataType>& out_indices,
                             const std::vector<ck::index_t>& window_spatial_lengths,
                             const std::vector<ck::index_t>& window_strides,
                             const std::vector<ck::index_t>& window_dilations,
                             const std::vector<ck::index_t>& in_left_pads,
                             const std::vector<ck::index_t>& in_right_pads)
    {
        return Argument{in,
                        out,
                        out_indices,
                        window_spatial_lengths,
                        window_strides,
                        window_dilations,
                        in_left_pads,
                        in_right_pads};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "ReferencePoolingFwd"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
