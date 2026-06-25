// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename InDataType, typename OutDataType, typename AccDataType>
struct ReferenceSoftmax : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<InDataType>& in,
                 Tensor<OutDataType>& out,
                 double alpha,
                 double beta,
                 const std::vector<index_t> sm_reduce_dims)
            : in_(in), out_(out), sm_reduce_dims_(sm_reduce_dims)
        {
            alpha_ = static_cast<AccDataType>(alpha);
            beta_  = static_cast<AccDataType>(beta);

            // std::cout << "debug: scalar dims: ";
            for(size_t i = 0; i < in.mDesc.GetNumOfDimension(); i++)
            {
                if(std::find(sm_reduce_dims.begin(), sm_reduce_dims.end(), i) ==
                   sm_reduce_dims.end())
                {
                    sm_scalar_dims_.push_back(i);
                    // std::cout << i << ", ";
                }
            }
            // std::cout << std::endl;
        }

        const Tensor<InDataType>& in_;
        Tensor<OutDataType>& out_;
        AccDataType alpha_;
        AccDataType beta_;
        std::vector<index_t> sm_reduce_dims_;
        std::vector<index_t> sm_scalar_dims_; // dim after internal max/sum reduction
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            std::vector<size_t> scalar_lengths;
            for(index_t dim : arg.sm_scalar_dims_)
            {
                scalar_lengths.push_back(arg.in_.mDesc.GetLengths()[dim]);
            }
            // max and sum reduction with final reduced values of dim=0 is a scalar so give it
            // appropriate lengths of {1}
            if(arg.sm_scalar_dims_.size() == 0)
            {
                scalar_lengths.push_back(1);
            }

            Tensor<AccDataType> reduce_max(scalar_lengths);
            reduce_max.GenerateTensorValue(
                GeneratorTensor_1<AccDataType>{std::numeric_limits<AccDataType>::lowest()});
            Tensor<AccDataType> reduce_sum(scalar_lengths);
            reduce_sum.GenerateTensorValue(GeneratorTensor_1<AccDataType>{0});

            // when final reduced values is of dim=0, the index will be transformed into empty
            // std::vector which is actually a valid input for Tensor::operator(std::vector) and
            // internally accesses 0'th element
            auto to_sm_scalar_idx = [&](auto idx) {
                std::vector<size_t> sm_scalar_idx;
                for(index_t dim : arg.sm_scalar_dims_)
                {
                    sm_scalar_idx.push_back(idx[dim]);
                }
                return sm_scalar_idx;
            };

            arg.in_.ForEach([&](auto& self, auto idx) {
                reduce_max(to_sm_scalar_idx(idx)) = std::max(
                    reduce_max(to_sm_scalar_idx(idx)), ck::type_convert<AccDataType>(self(idx)));
            });

            // LogRangeAsType<float>(std::cout << "reduce_max: ", reduce_max.mData, ",") <<
            // std::endl;

            Tensor<AccDataType> in_stable(arg.in_.mDesc);
            in_stable.ForEach([&](auto& self, auto idx) {
                // numerator = exp(x - max(x))
                self(idx) = std::exp(ck::type_convert<AccDataType>(arg.in_(idx)) -
                                     reduce_max(to_sm_scalar_idx(idx)));
            });

            // LogRangeAsType<float>(std::cout << "in_stable: ", in_stable.mData, ",") << std::endl;

            in_stable.ForEach([&](auto& self, auto idx) {
                // denominator = sum(exp(x - max(x)))
                reduce_sum(to_sm_scalar_idx(idx)) += self(idx);
            });

            // LogRangeAsType<float>(std::cout << "reduce_sum: ", reduce_sum.mData, ",") <<
            // std::endl;

            arg.out_.ForEach([&](auto& self, auto idx) {
                AccDataType temp_result =
                    arg.alpha_ * in_stable(idx) / reduce_sum(to_sm_scalar_idx(idx)) +
                    arg.beta_ * self(idx);
                self(idx) = ck::type_convert<OutDataType>(temp_result);
            });

            // LogRangeAsType<float>(std::cout << "out: ", arg.out_.mData, ",") << std::endl;
            // reduction along reduce dims
            // LogRangeAsType<float>(std::cout << "reduce_max: ", reduce_max.mData, ",") <<
            // std::endl; LogRangeAsType<float>(std::cout << "reduce_sum: ", reduce_sum.mData, ",")
            // << std::endl;

            return 0;
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /* stream_config */ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(const Tensor<InDataType>& in,
                             Tensor<OutDataType>& out,
                             double alpha,
                             double beta,
                             const std::vector<index_t> sm_reduce_dims)
    {
        return Argument{in, out, alpha, beta, sm_reduce_dims};
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
        str << "ReferenceSoftmax"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
