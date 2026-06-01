// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <thread>

#include "ck/ck.hpp"
#include "ck/utility/ignore.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          index_t Rank,
          index_t NumReduceDim,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename AccElementwiseOperation,
          bool PropagateNan,
          bool OutputIndex>
struct ReferenceReduce : public device::DeviceReduce<InDataType,
                                                     AccDataType,
                                                     OutDataType,
                                                     Rank,
                                                     NumReduceDim,
                                                     ReduceOperation,
                                                     InElementwiseOperation,
                                                     AccElementwiseOperation,
                                                     PropagateNan,
                                                     OutputIndex>
{
    using IndexDataType = int32_t;

    static constexpr int NumInvariantDim = Rank - NumReduceDim;

    static constexpr index_t NumSrcDim = Rank;
    static constexpr index_t NumDstDim = (NumInvariantDim == 0) ? 1 : NumInvariantDim;
    static constexpr bool reduceAllDim = (NumInvariantDim == 0);

    struct Argument : public device::BaseArgument
    {
        Argument(const std::array<index_t, Rank> inLengths,
                 const std::array<index_t, Rank> inStrides,
                 const std::array<index_t, NumDstDim> outLengths,
                 const std::array<index_t, NumDstDim> outStrides,
                 const std::array<int, NumReduceDim> reduceDims,
                 double alpha,
                 double beta,
                 const InDataType* in_host,
                 OutDataType* out_host,
                 IndexDataType* out_index_host,
                 const InElementwiseOperation in_elementwise_op,
                 const AccElementwiseOperation acc_elementwise_op)
            : reduceDims_(reduceDims),
              outLengths_(outLengths),
              outStrides_(outStrides),
              in_host_(in_host),
              out_host_(out_host),
              out_index_host_(out_index_host),
              in_elementwise_op_(in_elementwise_op),
              acc_elementwise_op_(acc_elementwise_op)
        {
            using ck::host_common::get_index_set;

            if(std::any_of(
                   reduceDims.begin(), reduceDims.end(), [](int d) { return d < 0 || d >= Rank; }))
                throw std::runtime_error("Invalid reduce dimensions!");

            if constexpr(NumInvariantDim > 0)
            {
                // get invariant_dims[] and invariant_lengths[]
                for(int dim = 0, i = 0; dim < Rank; dim++)
                    if(std::none_of(
                           reduceDims.begin(), reduceDims.end(), [&](int d) { return d == dim; }))
                    {
                        invariantDims_[i]     = dim;
                        invariant_lengths_[i] = inLengths[dim];
                        i++;
                    };
            };

            // get reduce_lengths_[]
            for(int j = 0, i = 0; j < NumReduceDim; j++)
            {
                int dim              = reduceDims[j];
                reduce_lengths_[i++] = inLengths[dim];
            };

            if constexpr(NumInvariantDim > 0)
            {
                // check invariant_lengths_ and outLengths
                for(int i = 0; i < NumInvariantDim; i++)
                    if(invariant_lengths_[i] != outLengths_[i])
                        throw std::runtime_error("Invalid lengths parameters!");
            }

            if constexpr(NumInvariantDim > 0)
            {
                for(int j = 0, i = 0; j < NumInvariantDim; j++)
                {
                    int dim                  = invariantDims_[j];
                    in_invariant_strides_[i] = inStrides[dim];
                    i++;
                };
            };

            for(int j = 0, i = 0; j < NumReduceDim; j++)
            {
                int dim               = reduceDims_[j];
                in_reduce_strides_[i] = inStrides[dim];
                i++;
            };

            if constexpr(NumInvariantDim > 0)
                invariant_index_set_ = get_index_set<NumInvariantDim>(invariant_lengths_);

            reduce_index_set_ = get_index_set<NumReduceDim>(reduce_lengths_);

            alpha_ = type_convert<AccDataType>(alpha);
            beta_  = type_convert<AccDataType>(beta);
        };

        const std::array<int, NumReduceDim> reduceDims_;
        std::array<int, NumInvariantDim> invariantDims_;
        std::array<index_t, NumInvariantDim> invariant_lengths_;
        std::array<index_t, NumReduceDim> reduce_lengths_;

        const std::array<index_t, NumDstDim> outLengths_;
        const std::array<index_t, NumDstDim> outStrides_;

        std::array<index_t, NumInvariantDim> in_invariant_strides_;
        std::array<index_t, NumReduceDim> in_reduce_strides_;

        const InDataType* in_host_;
        OutDataType* out_host_;
        IndexDataType* out_index_host_;
        const InElementwiseOperation in_elementwise_op_;
        const AccElementwiseOperation acc_elementwise_op_;

        AccDataType alpha_;
        AccDataType beta_;

        std::vector<std::array<index_t, NumInvariantDim>> invariant_index_set_;
        std::vector<std::array<index_t, NumReduceDim>> reduce_index_set_;
    };

    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            ignore = stream_config;

            using ck::float_equal_one;
            using ck::float_equal_zero;
            using ck::type_convert;
            using ck::host_common::get_index_set;
            using ck::host_common::get_offset_from_index;

            if constexpr(OutputIndex)
            {
                using Accumulation = ck::detail::AccumulateWithIndexAndNanCheck<PropagateNan,
                                                                                ReduceOperation,
                                                                                AccDataType,
                                                                                IndexDataType>;

                if constexpr(NumInvariantDim == 0)
                {
                    AccDataType accuVal = ReduceOperation::template GetIdentityValue<AccDataType>();
                    IndexDataType accuIndex = 0;

                    for(std::size_t i = 0; i < arg.reduce_index_set_.size(); i++)
                    {
                        auto in_offset = get_offset_from_index<NumReduceDim>(
                            arg.in_reduce_strides_, arg.reduce_index_set_[i]);

                        auto currVal = type_convert<AccDataType>(arg.in_host_[in_offset]);

                        arg.in_elementwise_op_(currVal, currVal);

                        auto currIndex = static_cast<IndexDataType>(i);

                        Accumulation::Calculate(accuVal, currVal, accuIndex, currIndex);
                    };

                    arg.acc_elementwise_op_(accuVal, accuVal);

                    if(!float_equal_one{}(arg.alpha_))
                        accuVal *= type_convert<AccDataType>(arg.alpha_);

                    if(!float_equal_zero{}(arg.beta_))
                        accuVal += type_convert<AccDataType>(arg.out_host_[0]) *
                                   type_convert<AccDataType>(arg.beta_);

                    arg.out_host_[0]       = type_convert<OutDataType>(accuVal);
                    arg.out_index_host_[0] = accuIndex;
                }
                else
                {
                    auto thread_reduce_func = [&](auto invariant_index) {
                        AccDataType accuVal =
                            ReduceOperation::template GetIdentityValue<AccDataType>();
                        IndexDataType accuIndex = 0;

                        auto in_invariant_offset = get_offset_from_index<NumInvariantDim>(
                            arg.in_invariant_strides_, invariant_index);

                        for(std::size_t i = 0; i < arg.reduce_index_set_.size(); i++)
                        {
                            auto in_reduce_offset = get_offset_from_index<NumReduceDim>(
                                arg.in_reduce_strides_, arg.reduce_index_set_[i]);

                            auto currVal = type_convert<AccDataType>(
                                arg.in_host_[in_invariant_offset + in_reduce_offset]);

                            arg.in_elementwise_op_(currVal, currVal);

                            auto currIndex = static_cast<IndexDataType>(i);

                            Accumulation::Calculate(accuVal, currVal, accuIndex, currIndex);
                        };

                        arg.acc_elementwise_op_(accuVal, accuVal);

                        if(!float_equal_one{}(arg.alpha_))
                            accuVal *= type_convert<AccDataType>(arg.alpha_);

                        auto dst_offset = get_offset_from_index<NumInvariantDim>(arg.outStrides_,
                                                                                 invariant_index);

                        if(!float_equal_zero{}(arg.beta_))
                            accuVal += type_convert<AccDataType>(arg.out_host_[dst_offset]) *
                                       type_convert<AccDataType>(arg.beta_);

                        arg.out_host_[dst_offset]       = type_convert<OutDataType>(accuVal);
                        arg.out_index_host_[dst_offset] = accuIndex;
                    };

                    std::size_t num_thread = std::thread::hardware_concurrency();

                    std::size_t work_per_thread =
                        (arg.invariant_index_set_.size() + num_thread - 1) / num_thread;

                    std::vector<joinable_thread> threads(num_thread);

                    for(std::size_t it = 0; it < num_thread; ++it)
                    {
                        std::size_t i_begin = it * work_per_thread;
                        std::size_t i_end =
                            std::min((it + 1) * work_per_thread, arg.invariant_index_set_.size());

                        auto f = [=] {
                            for(std::size_t i = i_begin; i < i_end; i++)
                            {
                                thread_reduce_func(arg.invariant_index_set_[i]);
                            }
                        };

                        threads[it] = joinable_thread(f);
                    }
                };
            }
            else
            {
                using Accumulation =
                    ck::detail::AccumulateWithNanCheck<PropagateNan, ReduceOperation, AccDataType>;

                if constexpr(NumInvariantDim == 0)
                {
                    AccDataType accuVal = ReduceOperation::template GetIdentityValue<AccDataType>();

                    for(const auto& reduce_index : arg.reduce_index_set_)
                    {
                        auto in_offset = get_offset_from_index<NumReduceDim>(arg.in_reduce_strides_,
                                                                             reduce_index);

                        auto currVal = type_convert<AccDataType>(arg.in_host_[in_offset]);

                        arg.in_elementwise_op_(currVal, currVal);

                        Accumulation::Calculate(accuVal, currVal);
                    };

                    arg.acc_elementwise_op_(accuVal, accuVal);

                    if(!float_equal_one{}(arg.alpha_))
                        accuVal *= type_convert<AccDataType>(arg.alpha_);

                    if(!float_equal_zero{}(arg.beta_))
                        accuVal += type_convert<AccDataType>(arg.out_host_[0]) *
                                   type_convert<AccDataType>(arg.beta_);

                    arg.out_host_[0] = type_convert<OutDataType>(accuVal);
                }
                else
                {
                    auto thread_reduce_func = [&](auto invariant_index) {
                        AccDataType accuVal =
                            ReduceOperation::template GetIdentityValue<AccDataType>();

                        auto in_invariant_offset = get_offset_from_index<NumInvariantDim>(
                            arg.in_invariant_strides_, invariant_index);

                        for(const auto& reduce_index : arg.reduce_index_set_)
                        {
                            auto in_reduce_offset = get_offset_from_index<NumReduceDim>(
                                arg.in_reduce_strides_, reduce_index);

                            auto currVal = type_convert<AccDataType>(
                                arg.in_host_[in_invariant_offset + in_reduce_offset]);

                            arg.in_elementwise_op_(currVal, currVal);

                            Accumulation::Calculate(accuVal, currVal);
                        };

                        arg.acc_elementwise_op_(accuVal, accuVal);

                        if(!float_equal_one{}(arg.alpha_))
                            accuVal *= type_convert<AccDataType>(arg.alpha_);

                        auto dst_offset = get_offset_from_index<NumInvariantDim>(arg.outStrides_,
                                                                                 invariant_index);

                        if(!float_equal_zero{}(arg.beta_))
                            accuVal += type_convert<AccDataType>(arg.out_host_[dst_offset]) *
                                       type_convert<AccDataType>(arg.beta_);

                        arg.out_host_[dst_offset] = type_convert<OutDataType>(accuVal);
                    };

                    std::size_t num_thread = std::thread::hardware_concurrency();

                    std::size_t work_per_thread =
                        (arg.invariant_index_set_.size() + num_thread - 1) / num_thread;

                    std::vector<joinable_thread> threads(num_thread);

                    for(std::size_t it = 0; it < num_thread; ++it)
                    {
                        std::size_t i_begin = it * work_per_thread;
                        std::size_t i_end =
                            std::min((it + 1) * work_per_thread, arg.invariant_index_set_.size());

                        auto f = [=] {
                            for(std::size_t i = i_begin; i < i_end; i++)
                            {
                                thread_reduce_func(arg.invariant_index_set_[i]);
                            }
                        };

                        threads[it] = joinable_thread(f);
                    }
                };
            };

            return (0.0f);
        };

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        };
    };

    bool IsSupportedArgument(const device::BaseArgument* p_arg) override
    {
        ignore = p_arg;

        return true;
    };

    std::unique_ptr<device::BaseArgument>
    MakeArgumentPointer(const std::array<index_t, Rank> inLengths,
                        const std::array<index_t, Rank> inStrides,
                        const std::array<index_t, NumDstDim> outLengths,
                        const std::array<index_t, NumDstDim> outStrides,
                        const std::array<int, NumReduceDim> reduceDims,
                        double alpha,
                        double beta,
                        const void* in_host,
                        const void* in_index_host,
                        void* out_host,
                        void* out_index_host,
                        const InElementwiseOperation in_elementwise_op,
                        const AccElementwiseOperation acc_elementwise_op) override
    {
        ignore = in_index_host;

        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          outLengths,
                                          outStrides,
                                          reduceDims,
                                          alpha,
                                          beta,
                                          static_cast<const InDataType*>(in_host),
                                          static_cast<OutDataType*>(out_host),
                                          static_cast<IndexDataType*>(out_index_host),
                                          in_elementwise_op,
                                          acc_elementwise_op);
    };

    std::unique_ptr<device::BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "Reference_Reduce<" << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
