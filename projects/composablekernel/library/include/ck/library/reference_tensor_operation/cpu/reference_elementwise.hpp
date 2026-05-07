// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <index_t NumATensors, typename ADataType, typename BDataType, typename ElementOp>
struct ReferenceElementwise : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const std::array<Tensor<ADataType>, NumATensors>& a_tensors,
                 Tensor<BDataType>& b_tensor,
                 ElementOp element_op)
            : a_tensors_{a_tensors}, b_tensor_{b_tensor}, element_op_{element_op}
        {
        }

        const std::array<Tensor<ADataType>, NumATensors>& a_tensors_;
        Tensor<BDataType>& b_tensor_;
        ElementOp element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceElementwise::Argument;

        float Run(const Argument& arg)
        {
            if constexpr(NumATensors == 1)
            {
                arg.b_tensor_.ForEach([&](auto& self, auto idx) {
                    arg.element_op_(self(idx), arg.a_tensors_[0](idx));
                });
            }
            else if constexpr(NumATensors == 2)
            {
                arg.b_tensor_.ForEach([&](auto& self, auto idx) {
                    arg.element_op_(self(idx), arg.a_tensors_[0](idx), arg.a_tensors_[1](idx));
                });
            }
            else if constexpr(NumATensors == 3)
            {
                arg.b_tensor_.ForEach([&](auto& self, auto idx) {
                    arg.element_op_(self(idx),
                                    arg.a_tensors_[0](idx),
                                    arg.a_tensors_[1](idx),
                                    arg.a_tensors_[2](idx));
                });
            }
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

    static auto MakeArgument(const std::array<Tensor<ADataType>, NumATensors>& a_tensors,
                             Tensor<BDataType>& b_tensor,
                             ElementOp element_op)
    {
        return Argument{a_tensors, b_tensor, element_op};
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
        str << "ReferenceElementwise"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
