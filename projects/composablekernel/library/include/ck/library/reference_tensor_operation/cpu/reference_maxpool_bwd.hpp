// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

namespace ck {
namespace tensor_operation {
namespace host {
using namespace std;

template <typename DOutDataType,
          typename IndexDataType,
          typename ConputeDataType,
          typename DInDataType,
          typename ElementwiseOperation>
struct ReferenceMaxPoolBwd : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<DOutDataType>& dout,
                 const Tensor<IndexDataType>& indices,
                 Tensor<DInDataType>& din,
                 ElementwiseOperation elementwise_op)
            : dout_(dout), indices_(indices), din_(din), elementwise_op_(elementwise_op)
        {
        }

        const Tensor<DOutDataType>& dout_;
        const Tensor<IndexDataType>& indices_;
        Tensor<DInDataType>& din_;
        ElementwiseOperation elementwise_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            int din_length  = arg.din_.GetElementSpaceSize();
            int dout_length = arg.dout_.GetElementSpaceSize();
            std::vector<ConputeDataType> buf(din_length, 0);

            for(int i = 0; i < dout_length; ++i)
            {
                int index = arg.indices_.mData[i];
                if(index >= 0 && index < din_length)
                {
                    if constexpr(is_same_v<ConputeDataType, bhalf_t>)
                    {
                        float buf_val = ck::type_convert<float>(buf[index]);
                        buf_val += ck::type_convert<float>(arg.dout_.mData[i]);
                        buf[index] = ck::type_convert<ConputeDataType>(buf_val);
                    }
                    else
                        buf[index] += ck::type_convert<ConputeDataType>(arg.dout_.mData[i]);
                }
            }

            for(int i = 0; i < din_length; ++i)
                arg.din_.mData[i] = ck::type_convert<DInDataType>(buf[i]);
            return 0;
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /* stream_config */ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(const Tensor<DOutDataType>& dout,
                             const Tensor<IndexDataType>& indices,
                             Tensor<DInDataType>& din,
                             ElementwiseOperation elementwise_op)
    {
        return Argument{dout, indices, din, elementwise_op};
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
        str << "ReferenceMaxPoolBwd"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
