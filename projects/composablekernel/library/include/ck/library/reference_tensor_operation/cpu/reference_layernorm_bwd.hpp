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

// def normalization_backward_x(dy, x, gamma, x_mean, rstd, reduce_axis, reduce_size):
//     ds = np.sum(dy * gamma * x, axis=reduce_axis, keepdims=True)
//     db = np.sum(dy * gamma, axis=reduce_axis, keepdims=True)
//     b = (db * x_mean - ds) * rstd ** (3) / reduce_size
//     c = -b * x_mean - db * rstd / reduce_size
//     dx = rstd * dy * gamma + b * x + c
//     return dx

// def normalization_beta_backward_gamma_beta(dy, x, x_mean, rstd, reduce_axis):
//     # Assume shape of gamma and beta are the same
//     dgamma = np.sum(dy * (x - x_mean) * rstd, axis=reduce_axis, keepdims=True)
//     dbeta = np.sum(dy, axis=reduce_axis, keepdims=True)
//     return dgamma, dbeta

// def layernorm_backward(dy, x, gamma, x_mean, rstd):
//     # dy, x = [M, K], gamma = [1, K], x_mean, rstd = [M, 1]
//     # dx = [M, K], dgamma, dbeta = [1, K]
//     M, K = x.shape
//     dx = normalization_input_backward(dy, x, gamma, x_mean, rstd, 1, K)
//     dgamma, dbeta = normalization_gamma_beta_backward(dy, x, x_mean, rstd, 0)
//     return dx, dgamma, dbeta

// Reference (Layernorm and groupnorm):
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/layer_norm_kernel.cpp#L196
template <typename DYDataType,
          typename XDataType,
          typename GammaDataType,
          typename MeanInvStdDataType,
          typename DGammaDataType,
          typename DBetaDataType,
          typename DXDataType,
          typename ComputeDataType>
struct ReferenceLayernormBwd : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<DYDataType>& dy_m_n,
                 const Tensor<XDataType>& x_m_n,
                 const Tensor<GammaDataType>& gamma_n,
                 const Tensor<MeanInvStdDataType>& mean_m,
                 const Tensor<MeanInvStdDataType>& inv_std_m,
                 Tensor<DGammaDataType>& dgamma_n,
                 Tensor<DBetaDataType>& dbeta_n,
                 Tensor<DXDataType>& dx_m_n,
                 const std::vector<index_t> lengths)
            : dy_m_n_(dy_m_n),
              x_m_n_(x_m_n),
              gamma_n_(gamma_n),
              mean_m_(mean_m),
              inv_std_m_(inv_std_m),
              dgamma_n_(dgamma_n),
              dbeta_n_(dbeta_n),
              dx_m_n_(dx_m_n),
              lengths_(lengths)
        {
        }

        const Tensor<DYDataType>& dy_m_n_;
        const Tensor<XDataType>& x_m_n_;
        const Tensor<GammaDataType>& gamma_n_;
        const Tensor<MeanInvStdDataType>& mean_m_;
        const Tensor<MeanInvStdDataType>& inv_std_m_;
        Tensor<DGammaDataType>& dgamma_n_;
        Tensor<DBetaDataType>& dbeta_n_;
        Tensor<DXDataType>& dx_m_n_;
        std::vector<index_t> lengths_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            int M = arg.lengths_[0];
            int N = arg.lengths_[1];

            // Calculate dgamma and dbeta
            for(int n = 0; n < N; ++n)
            {
                ComputeDataType dgamma = 0;
                ComputeDataType dbeta  = 0;

                for(int m = 0; m < M; ++m)
                {
                    ComputeDataType dy   = ck::type_convert<ComputeDataType>(arg.dy_m_n_(m, n));
                    ComputeDataType x    = ck::type_convert<ComputeDataType>(arg.x_m_n_(m, n));
                    ComputeDataType mean = ck::type_convert<ComputeDataType>(arg.mean_m_(m));
                    ComputeDataType rstd = ck::type_convert<ComputeDataType>(arg.inv_std_m_(m));
                    dgamma += dy * rstd * (x - mean);
                    dbeta += dy;
                }
                arg.dgamma_n_(n) = ck::type_convert<DGammaDataType>(dgamma);
                arg.dbeta_n_(n)  = ck::type_convert<DBetaDataType>(dbeta);
            }

            // Calculate dx
            for(int m = 0; m < M; ++m)
            {
                ComputeDataType ds = 0;
                ComputeDataType db = 0;

                ComputeDataType mean = ck::type_convert<ComputeDataType>(arg.mean_m_(m));
                ComputeDataType rstd = ck::type_convert<ComputeDataType>(arg.inv_std_m_(m));

                for(int n = 0; n < N; ++n)
                {
                    ComputeDataType dy    = ck::type_convert<ComputeDataType>(arg.dy_m_n_(m, n));
                    ComputeDataType x     = ck::type_convert<ComputeDataType>(arg.x_m_n_(m, n));
                    ComputeDataType gamma = ck::type_convert<ComputeDataType>(arg.gamma_n_(n));

                    ds += dy * gamma * x;
                    db += dy * gamma;
                }

                for(int n = 0; n < N; ++n)
                {
                    ComputeDataType dy    = ck::type_convert<ComputeDataType>(arg.dy_m_n_(m, n));
                    ComputeDataType x     = ck::type_convert<ComputeDataType>(arg.x_m_n_(m, n));
                    ComputeDataType gamma = ck::type_convert<ComputeDataType>(arg.gamma_n_(n));

                    ComputeDataType b = (db * mean - ds) * rstd * rstd * rstd / N;
                    ComputeDataType c = -b * mean - db * rstd / N;

                    arg.dx_m_n_(m, n) = ck::type_convert<DXDataType>(dy * gamma * rstd + b * x + c);
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

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(const Tensor<DYDataType>& dy_m_n,
                             const Tensor<XDataType>& x_m_n,
                             const Tensor<GammaDataType>& gamma_n,
                             const Tensor<MeanInvStdDataType>& mean_m,
                             const Tensor<MeanInvStdDataType>& inv_std_m,
                             Tensor<DGammaDataType>& dgamma_n,
                             Tensor<DBetaDataType>& dbeta_n,
                             Tensor<DXDataType>& dx_m_n,
                             const std::vector<index_t> lengths)
    {
        return Argument{
            dy_m_n, x_m_n, gamma_n, mean_m, inv_std_m, dgamma_n, dbeta_n, dx_m_n, lengths};
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
        str << "ReferenceLayernormBwd"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
