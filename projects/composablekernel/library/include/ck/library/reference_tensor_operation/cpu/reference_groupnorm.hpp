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

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType,
          typename ComputeDataType,
          typename YElementwiseOperation>
struct ReferenceGroupnorm : public device::BaseOperator
{
    // x = [N, H, W, G, C]
    // y = [N, H, W, G, C]
    // reduce dim [H, W, C], mean, var = [N, G]
    // gamma, beta = [G, C]
    // beta: [G, C]
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<XDataType>& x,
                 const Tensor<GammaDataType>& gamma,
                 const Tensor<BetaDataType>& beta,
                 Tensor<YDataType>& y,
                 Tensor<SaveMeanInvStdDataType>& save_mean,
                 Tensor<SaveMeanInvStdDataType>& save_inv_std,
                 YElementwiseOperation y_elementwise_op,
                 const std::vector<index_t> lengths,
                 ComputeDataType epsilon)
            : x_(x),
              gamma_(gamma),
              beta_(beta),
              y_(y),
              save_mean_(save_mean),
              save_inv_std_(save_inv_std),
              y_elementwise_op_(y_elementwise_op),
              lengths_(lengths),
              epsilon_(epsilon)
        {
        }

        const Tensor<XDataType> x_;
        const Tensor<XDataType> gamma_;
        const Tensor<XDataType> beta_;
        Tensor<YDataType>& y_;
        Tensor<SaveMeanInvStdDataType>& save_mean_;
        Tensor<SaveMeanInvStdDataType>& save_inv_std_;
        YElementwiseOperation y_elementwise_op_;
        std::vector<index_t> lengths_;
        ComputeDataType epsilon_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            int N = arg.lengths_[0];
            int H = arg.lengths_[1];
            int W = arg.lengths_[2];
            int G = arg.lengths_[3];
            int C = arg.lengths_[4];

            Tensor<ComputeDataType> mean({N, G});
            Tensor<ComputeDataType> var({N, G});

            // Compute mean & var in [H, W, C] by Welford Algorithm
            // TODO - parallel for each HWC
            // TODO - address calculation
            for(int n = 0; n < N; ++n)
            {
                for(int g = 0; g < G; ++g)
                {
                    ComputeDataType mean_val = type_convert<ComputeDataType>(0.0f);
                    ComputeDataType var_val  = type_convert<ComputeDataType>(0.0f);
                    int32_t curr_count       = 0;

                    for(int h = 0; h < H; ++h)
                    {
                        for(int w = 0; w < W; ++w)
                        {
                            for(int c = 0; c < C; ++c)
                            {
                                curr_count++;
                                ComputeDataType x =
                                    type_convert<ComputeDataType>(arg.x_(n, h, w, g, c));
                                ComputeDataType delta = x - mean_val;
                                mean_val += delta / curr_count;
                                ComputeDataType delta2 = x - mean_val;
                                var_val += delta * delta2;
                            }
                        }
                    }

                    mean(n, g) = mean_val;
                    var(n, g)  = var_val / curr_count;

                    arg.save_mean_(n, g) = ck::type_convert<SaveMeanInvStdDataType>(mean(n, g));

                    ComputeDataType divisor =
                        static_cast<ComputeDataType>(1) / ck::math::sqrt(var(n, g) + arg.epsilon_);
                    arg.save_inv_std_(n, g) = ck::type_convert<SaveMeanInvStdDataType>(divisor);
                }
            }

            // Normalization
            for(int n = 0; n < N; ++n)
            {
                for(int h = 0; h < H; ++h)
                {
                    for(int w = 0; w < W; ++w)
                    {
                        for(int g = 0; g < G; ++g)
                        {
                            for(int c = 0; c < C; ++c)
                            {
                                ComputeDataType x =
                                    type_convert<ComputeDataType>(arg.x_(n, h, w, g, c));
                                ComputeDataType gamma =
                                    type_convert<ComputeDataType>(arg.gamma_(g, c));
                                ComputeDataType beta =
                                    type_convert<ComputeDataType>(arg.beta_(g, c));
                                ComputeDataType mean_val =
                                    type_convert<ComputeDataType>(mean(n, g));
                                ComputeDataType var_val = type_convert<ComputeDataType>(var(n, g));
                                ComputeDataType y       = gamma * (x - mean_val) /
                                                        ck::math::sqrt(arg.epsilon_ + var_val) +
                                                    beta;
                                arg.y_elementwise_op_(y, y);
                                arg.y_(n, h, w, g, c) = type_convert<YDataType>(y);
                            }
                        }
                    }
                }
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

    bool IsSupportedArgument(const device::BaseArgument* p_arg) override
    {
        const Argument* p_arg_ = dynamic_cast<const Argument*>(p_arg);
        if(p_arg_->lengths_.size() != 5)
            return false;

        return true;
    }

    static auto MakeArgument(const Tensor<XDataType>& x,
                             const Tensor<GammaDataType>& gamma,
                             const Tensor<BetaDataType>& beta,
                             Tensor<YDataType>& y,
                             Tensor<SaveMeanInvStdDataType>& save_mean,
                             Tensor<SaveMeanInvStdDataType>& save_inv_std,
                             YElementwiseOperation y_elementwise_op,
                             const std::vector<index_t> lengths,
                             ComputeDataType epsilon)
    {
        return Argument{
            x, gamma, beta, y, save_mean, save_inv_std, y_elementwise_op, lengths, epsilon};
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
        str << "ReferenceLayernorm"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
