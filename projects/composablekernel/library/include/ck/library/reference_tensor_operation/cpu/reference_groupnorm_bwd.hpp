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

// def normalization_backward_gamma_beta(dy, x, x_mean, rstd, reduce_axis):
//     # Assume shape of gamma and beta are the same
//     dgamma = np.sum(dy * (x - x_mean) * rstd, axis=reduce_axis, keepdims=True)
//     dbeta = np.sum(dy, axis=reduce_axis, keepdims=True)
//     return dgamma, dbeta

// def groupnorm_backward(dy, x, gamma, x_mean, rstd):
//     # dy, x = [N, H, W, G, C], gamma = [1, 1, 1, G, C], x_mean, rstd = [N, 1, 1, G, 1]
//     N, H, W, G, C = x.shape
//     dx = normalization_input_backward(
//         dy, x, gamma, x_mean, rstd, (1, 2, 4), H * W * C)
//     dgamma, dbeta = normalization_gamma_beta_backward(
//         dy, x, x_mean, rstd, (0, 1, 2))
//     return dx, dgamma, dbeta

// Reference (Layernorm and groupnorm):
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/group_norm_kernel.cpp#L655
template <typename DYDataType,
          typename XDataType,
          typename GammaDataType,
          typename MeanInvStdDataType,
          typename DGammaDataType,
          typename DBetaDataType,
          typename DXDataType,
          typename ComputeDataType>
struct ReferenceGroupnormBwd : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<DYDataType>& dy_nhwgc,
                 const Tensor<XDataType>& x_nhwgc,
                 const Tensor<GammaDataType>& gamma_gc,
                 const Tensor<MeanInvStdDataType>& mean_ng,
                 const Tensor<MeanInvStdDataType>& inv_std_ng,
                 Tensor<DGammaDataType>& dgamma_gc,
                 Tensor<DBetaDataType>& dbeta_gc,
                 Tensor<DXDataType>& dx_nhwgc,
                 const std::vector<index_t> lengths)
            : dy_nhwgc_(dy_nhwgc),
              x_nhwgc_(x_nhwgc),
              gamma_gc_(gamma_gc),
              mean_ng_(mean_ng),
              inv_std_ng_(inv_std_ng),
              dgamma_gc_(dgamma_gc),
              dbeta_gc_(dbeta_gc),
              dx_nhwgc_(dx_nhwgc),
              lengths_(lengths)
        {
        }

        const Tensor<DYDataType>& dy_nhwgc_;
        const Tensor<XDataType>& x_nhwgc_;
        const Tensor<GammaDataType>& gamma_gc_;
        const Tensor<MeanInvStdDataType>& mean_ng_;
        const Tensor<MeanInvStdDataType>& inv_std_ng_;
        Tensor<DGammaDataType>& dgamma_gc_;
        Tensor<DBetaDataType>& dbeta_gc_;
        Tensor<DXDataType>& dx_nhwgc_;
        std::vector<index_t> lengths_;
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

            // Calculate dgamma and dbeta
            for(int g = 0; g < G; ++g)
                for(int c = 0; c < C; ++c)
                {
                    ComputeDataType dgamma = 0;
                    ComputeDataType dbeta  = 0;

                    for(int n = 0; n < N; ++n)
                        for(int h = 0; h < H; ++h)
                            for(int w = 0; w < W; ++w)
                            {
                                ComputeDataType dy =
                                    ck::type_convert<ComputeDataType>(arg.dy_nhwgc_(n, h, w, g, c));
                                ComputeDataType x =
                                    ck::type_convert<ComputeDataType>(arg.x_nhwgc_(n, h, w, g, c));
                                ComputeDataType mean =
                                    ck::type_convert<ComputeDataType>(arg.mean_ng_(n, g));
                                ComputeDataType rstd =
                                    ck::type_convert<ComputeDataType>(arg.inv_std_ng_(n, g));
                                dgamma += dy * rstd * (x - mean);
                                dbeta += dy;
                            }
                    arg.dgamma_gc_(g, c) = ck::type_convert<DGammaDataType>(dgamma);
                    arg.dbeta_gc_(g, c)  = ck::type_convert<DBetaDataType>(dbeta);
                }

            // Calculate dx
            int reduce_size = H * W * C;
            for(int n = 0; n < N; ++n)
                for(int g = 0; g < G; ++g)
                {
                    ComputeDataType ds = 0;
                    ComputeDataType db = 0;

                    ComputeDataType mean = ck::type_convert<ComputeDataType>(arg.mean_ng_(n, g));
                    ComputeDataType rstd = ck::type_convert<ComputeDataType>(arg.inv_std_ng_(n, g));

                    for(int h = 0; h < H; ++h)
                        for(int w = 0; w < W; ++w)
                            for(int c = 0; c < C; ++c)
                            {
                                ComputeDataType dy =
                                    ck::type_convert<ComputeDataType>(arg.dy_nhwgc_(n, h, w, g, c));
                                ComputeDataType x =
                                    ck::type_convert<ComputeDataType>(arg.x_nhwgc_(n, h, w, g, c));
                                ComputeDataType gamma =
                                    ck::type_convert<ComputeDataType>(arg.gamma_gc_(g, c));

                                ds += dy * gamma * x;
                                db += dy * gamma;
                            }

                    for(int h = 0; h < H; ++h)
                        for(int w = 0; w < W; ++w)
                            for(int c = 0; c < C; ++c)
                            {
                                ComputeDataType dy =
                                    ck::type_convert<ComputeDataType>(arg.dy_nhwgc_(n, h, w, g, c));
                                ComputeDataType x =
                                    ck::type_convert<ComputeDataType>(arg.x_nhwgc_(n, h, w, g, c));
                                ComputeDataType gamma =
                                    ck::type_convert<ComputeDataType>(arg.gamma_gc_(g, c));

                                ComputeDataType b =
                                    (db * mean - ds) * rstd * rstd * rstd / reduce_size;
                                ComputeDataType c1 = -b * mean - db * rstd / reduce_size;
                                arg.dx_nhwgc_(n, h, w, g, c) =
                                    ck::type_convert<DXDataType>(dy * gamma * rstd + b * x + c1);
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

    static auto MakeArgument(const Tensor<DYDataType>& dy_nhwgc,
                             const Tensor<XDataType>& x_nhwgc,
                             const Tensor<GammaDataType>& gamma_gc,
                             const Tensor<MeanInvStdDataType>& mean_ng,
                             const Tensor<MeanInvStdDataType>& inv_std_ng,
                             Tensor<DGammaDataType>& dgamma_gc,
                             Tensor<DBetaDataType>& dbeta_gc,
                             Tensor<DXDataType>& dx_nhwgc,
                             const std::vector<index_t> lengths)
    {
        return Argument{dy_nhwgc,
                        x_nhwgc,
                        gamma_gc,
                        mean_ng,
                        inv_std_ng,
                        dgamma_gc,
                        dbeta_gc,
                        dx_nhwgc,
                        lengths};
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
        str << "ReferenceGroupnormBwd"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
