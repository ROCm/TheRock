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
          typename YElementwiseOperation,
          index_t Rank,
          index_t NumReduceDim>
struct ReferenceLayernorm : public device::BaseOperator
{
    // TODO - support generic layernorm
    static_assert((Rank == 2 && NumReduceDim == 1) || (Rank == 4 && NumReduceDim == 3),
                  "Only support 2D & 4D version so far");

    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<XDataType>& x_m_n,
                 const Tensor<GammaDataType>& gamma_n,
                 const Tensor<BetaDataType>& beta_n,
                 Tensor<YDataType>& y_m_n,
                 Tensor<SaveMeanInvStdDataType>& save_mean_m,
                 Tensor<SaveMeanInvStdDataType>& save_inv_std_m,
                 YElementwiseOperation y_elementwise_op,
                 const std::vector<index_t> lengths,
                 const std::vector<index_t> reduceDims,
                 ComputeDataType epsilon)
            : x_m_n_(x_m_n),
              gamma_n_(gamma_n),
              beta_n_(beta_n),
              y_m_n_(y_m_n),
              save_mean_m_(save_mean_m),
              save_inv_std_m_(save_inv_std_m),
              y_elementwise_op_(y_elementwise_op),
              lengths_(lengths),
              reduceDims_(reduceDims),
              epsilon_(epsilon)
        {
        }

        const Tensor<XDataType> x_m_n_;
        const Tensor<XDataType> gamma_n_;
        const Tensor<XDataType> beta_n_;
        Tensor<YDataType>& y_m_n_;
        Tensor<SaveMeanInvStdDataType>& save_mean_m_;
        Tensor<SaveMeanInvStdDataType>& save_inv_std_m_;
        YElementwiseOperation y_elementwise_op_;
        std::vector<index_t> lengths_;
        std::vector<index_t> reduceDims_;
        ComputeDataType epsilon_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float Run2D(const Argument& arg)
        {
            int M = arg.lengths_[0];
            int N = arg.lengths_[1];

            Tensor<ComputeDataType> mean({M});
            Tensor<ComputeDataType> var({M});

            for(int m = 0; m < M; ++m)
            {
                mean(m) = 0;
                var(m)  = 0;

                for(int n = 0; n < N; ++n)
                {
                    auto x_val = ck::type_convert<ComputeDataType>(arg.x_m_n_(m, n));
                    mean(m) += x_val;
                    var(m) += x_val * x_val;
                }

                mean(m) = mean(m) / N;
                var(m)  = (var(m) / N) - (mean(m) * mean(m));
            }

            for(int m = 0; m < M; ++m)
            {
                ComputeDataType divisor =
                    static_cast<ComputeDataType>(1) / ck::math::sqrt(var(m) + arg.epsilon_);

                for(int n = 0; n < N; ++n)
                {
                    auto x_val     = ck::type_convert<ComputeDataType>(arg.x_m_n_(m, n));
                    auto gamma_val = ck::type_convert<ComputeDataType>(arg.gamma_n_(n));
                    auto beta_val  = ck::type_convert<ComputeDataType>(arg.beta_n_(n));
                    auto y_val     = (x_val - mean(m)) * divisor;
                    y_val          = (y_val * gamma_val) + beta_val;
                    arg.y_elementwise_op_(y_val, y_val);
                    arg.y_m_n_(m, n) = ck::type_convert<YDataType>(y_val);
                }
                arg.save_mean_m_(m)    = ck::type_convert<SaveMeanInvStdDataType>(mean(m));
                arg.save_inv_std_m_(m) = ck::type_convert<SaveMeanInvStdDataType>(divisor);
            }

            return 0;
        }

        float Run4D(const Argument& arg)
        {
            int N = arg.lengths_[0];
            int H = arg.lengths_[1];
            int W = arg.lengths_[2];
            int C = arg.lengths_[3];

            Tensor<ComputeDataType> mean({N});
            Tensor<ComputeDataType> var({N});

            int reduce_length = H * W * C;

            for(int n = 0; n < N; ++n)
            {
                mean(n) = 0;
                var(n)  = 0;

                for(int h = 0; h < H; ++h)
                    for(int w = 0; w < W; ++w)
                        for(int c = 0; c < C; ++c)
                        {
                            auto x_val = ck::type_convert<ComputeDataType>(arg.x_m_n_(n, h, w, c));
                            mean(n) += x_val;
                            var(n) += x_val * x_val;
                        }

                mean(n) = mean(n) / reduce_length;
                var(n)  = (var(n) / reduce_length) - (mean(n) * mean(n));
            }

            for(int n = 0; n < N; ++n)
            {
                ComputeDataType divisor =
                    static_cast<ComputeDataType>(1) / ck::math::sqrt(var(n) + arg.epsilon_);

                for(int h = 0; h < H; ++h)
                    for(int w = 0; w < W; ++w)
                        for(int c = 0; c < C; ++c)
                        {
                            auto x_val = ck::type_convert<ComputeDataType>(arg.x_m_n_(n, h, w, c));
                            auto gamma_val =
                                ck::type_convert<ComputeDataType>(arg.gamma_n_(h, w, c));
                            auto beta_val = ck::type_convert<ComputeDataType>(arg.beta_n_(h, w, c));
                            auto y_val    = (x_val - mean(n)) * divisor;
                            y_val         = (y_val * gamma_val) + beta_val;
                            arg.y_elementwise_op_(y_val, y_val);
                            arg.y_m_n_(n, h, w, c) = ck::type_convert<YDataType>(y_val);
                        }
                arg.save_mean_m_(n)    = ck::type_convert<SaveMeanInvStdDataType>(mean(n));
                arg.save_inv_std_m_(n) = ck::type_convert<SaveMeanInvStdDataType>(divisor);
            }

            return 0;
        }

        float Run(const Argument& arg)
        {
            if(arg.lengths_.size() == 2)
                return Run2D(arg);
            else if(arg.lengths_.size() == 4)
                return Run4D(arg);

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

        if(p_arg_->lengths_.size() == 2 && p_arg_->reduceDims_.size() == 1 &&
           p_arg_->reduceDims_[0] == 1)
            return true;

        else if(p_arg_->lengths_.size() == 4 && p_arg_->reduceDims_.size() == 3 &&
                p_arg_->reduceDims_[0] == 1 && p_arg_->reduceDims_[1] == 2 &&
                p_arg_->reduceDims_[2] == 3)
            return true;

        return false;
    }

    static auto MakeArgument(const Tensor<XDataType>& x_m_n,
                             const Tensor<GammaDataType>& gamma_n,
                             const Tensor<BetaDataType>& beta_n,
                             Tensor<YDataType>& y_m_n,
                             Tensor<SaveMeanInvStdDataType>& save_mean_m,
                             Tensor<SaveMeanInvStdDataType>& save_inv_std_m,
                             YElementwiseOperation y_elementwise_op,
                             const std::vector<index_t> lengths,
                             const std::vector<index_t> reduceDims,
                             ComputeDataType epsilon)
    {
        return Argument{x_m_n,
                        gamma_n,
                        beta_n,
                        y_m_n,
                        save_mean_m,
                        save_inv_std_m,
                        y_elementwise_op,
                        lengths,
                        reduceDims,
                        epsilon};
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
