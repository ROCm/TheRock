// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

// dinput descriptor in [N, C, Do, Ho, Wo] order
// doutput descriptor in [N, C, Di, Hi, Wi] order
// phyiscal layout is irrelavent
template <ck::index_t NDimSpatial,
          typename DInDataType,
          typename DOutDataType,
          typename std::enable_if<NDimSpatial >= 1 && NDimSpatial <= 3, bool>::type = false>
struct ReferenceAvgPoolBwd : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(Tensor<DInDataType>& dinput,
                 const Tensor<DOutDataType>& doutput,
                 std::vector<ck::index_t> window_spatial_lengths,
                 std::vector<ck::index_t> window_strides,
                 std::vector<ck::index_t> window_dilations,
                 std::vector<ck::index_t> dinput_left_pads,
                 std::vector<ck::index_t> dinput_right_pads)
            : dinput_{dinput},
              doutput_{doutput},
              window_spatial_lengths_{window_spatial_lengths},
              window_strides_{window_strides},
              window_dilations_{window_dilations},
              in_left_pads_{dinput_left_pads},
              in_right_pads_{dinput_right_pads}
        {
        }

        Tensor<DInDataType>& dinput_;
        const Tensor<DOutDataType>& doutput_;

        std::vector<ck::index_t> window_spatial_lengths_;
        std::vector<index_t> window_strides_;
        std::vector<index_t> window_dilations_;
        std::vector<index_t> in_left_pads_;
        std::vector<index_t> in_right_pads_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceAvgPoolBwd::Argument;

        template <ck::index_t NDimSpatial_,
                  typename std::enable_if<NDimSpatial_ == 1, bool>::type = false>
        float RunAvgPoolBwd(const Argument& arg)
        {
            // Let input = x, outpu = y
            // shape of x = [10], y = [6]
            // window_size = 5, pad = 0, stride = 1, dilation = 1
            // Forward:
            // y0 = 1/5 * (x0 + x1 + x2 + x3 + x4)
            // y1 = 1/5 * (x1 + x2 + x3 + x4 + x5)
            // ...
            // y5 = 1/5 * (x5 + x6 + x7 + x8 + x9)
            // y6 = 1/5 * (x6 + x7 + x8 + x9)
            // ...
            // y9 = 1/5 * (x9)

            // Backward:
            // shape of dy = [6], dx = [10]
            // dx0 = 1/5 * dy0
            // dx1 = 1/5 * (dy0 + dy1)
            // dx2 = 1/5 * (dy0 + dy1 + dy2)
            // ...
            // dx4 = 1/5 * (dy0 + dy1 + dy2 + dy3 + dy4)
            // dx5 = 1/5 * (dy1 + dy2 + dy3 + dy4 + dy5)
            // ...
            // dx9 = 1/5 * (dy5 + dy6 + dy7 + dy8 + dy9)

            auto f_ncw = [&](auto n, auto c, auto wi) {
                std::size_t X  = arg.window_spatial_lengths_[0];
                std::size_t Wo = arg.doutput_.GetLengths()[2];

                float v_acc = 0;

                for(std::size_t x = 0; x < X; ++x)
                {
                    // Out_Position = (In_Position + pad - x * dilation) / stride
                    auto w_tmp = static_cast<ck::long_index_t>(wi) +
                                 static_cast<ck::long_index_t>(arg.in_left_pads_[0]) -
                                 static_cast<ck::long_index_t>(x * arg.window_dilations_[0]);

                    // Check the input pixel validity (in perspective of being affected by some
                    // doutput pixel)
                    if(w_tmp % arg.window_strides_[0] == 0)
                    {
                        auto wo = static_cast<ck::long_index_t>(w_tmp) /
                                  static_cast<ck::long_index_t>(arg.window_strides_[0]);

                        // Get the doutput pixel in valid range to accumulate the gradients for this
                        // input pixel
                        if(wo >= 0 && ck::type_convert<std::size_t>(wo) < Wo)
                        {
                            v_acc += ck::type_convert<float>(arg.doutput_(n, c, wo));
                        }
                    }
                }

                v_acc /= ck::type_convert<float>(X);
                arg.dinput_(n, c, wi) = ck::type_convert<DInDataType>(v_acc);
            };

            make_ParallelTensorFunctor(f_ncw,
                                       arg.dinput_.GetLengths()[0],
                                       arg.dinput_.GetLengths()[1],
                                       arg.dinput_.GetLengths()[2])(
                std::thread::hardware_concurrency());

            return 0;
        }

        template <ck::index_t NDimSpatial_,
                  typename std::enable_if<NDimSpatial_ == 2, bool>::type = false>
        float RunAvgPoolBwd(const Argument& arg)
        {
            auto f_nchw = [&](auto n, auto c, auto hi, auto wi) {
                std::size_t Y = arg.window_spatial_lengths_[0];
                std::size_t X = arg.window_spatial_lengths_[1];

                std::size_t Ho = arg.doutput_.GetLengths()[2];
                std::size_t Wo = arg.doutput_.GetLengths()[3];

                float v_acc = 0;

                for(std::size_t y = 0; y < Y; ++y)
                {
                    // Out_Position = (In_Position + pad - x * dilation) / stride
                    auto h_tmp = static_cast<ck::long_index_t>(hi) +
                                 static_cast<ck::long_index_t>(arg.in_left_pads_[0]) -
                                 static_cast<ck::long_index_t>(y * arg.window_dilations_[0]);

                    // Check the input pixel validity (in perspective of being affected by some
                    // doutput pixel)
                    if(h_tmp % arg.window_strides_[0] == 0)
                    {
                        auto ho = static_cast<ck::long_index_t>(h_tmp) /
                                  static_cast<ck::long_index_t>(arg.window_strides_[0]);

                        // Get the doutput pixel in valid range to accumulate the gradients for this
                        // input pixel
                        if(ho >= 0 && ck::type_convert<std::size_t>(ho) < Ho)
                        {
                            for(std::size_t x = 0; x < X; ++x)
                            {
                                auto w_tmp =
                                    static_cast<ck::long_index_t>(wi) +
                                    static_cast<ck::long_index_t>(arg.in_left_pads_[1]) -
                                    static_cast<ck::long_index_t>(x * arg.window_dilations_[1]);
                                if(w_tmp % arg.window_strides_[1] == 0)
                                {
                                    auto wo = static_cast<ck::long_index_t>(w_tmp) /
                                              static_cast<ck::long_index_t>(arg.window_strides_[1]);
                                    if(wo >= 0 && ck::type_convert<std::size_t>(wo) < Wo)
                                    {
                                        v_acc +=
                                            ck::type_convert<float>(arg.doutput_(n, c, ho, wo));
                                    }
                                }
                            }
                        }
                    }
                }

                v_acc /= ck::type_convert<float>(Y * X);
                arg.dinput_(n, c, hi, wi) = ck::type_convert<DInDataType>(v_acc);
            };

            make_ParallelTensorFunctor(f_nchw,
                                       arg.dinput_.GetLengths()[0],
                                       arg.dinput_.GetLengths()[1],
                                       arg.dinput_.GetLengths()[2],
                                       arg.dinput_.GetLengths()[3])(
                std::thread::hardware_concurrency());

            return 0;
        }

        template <ck::index_t NDimSpatial_,
                  typename std::enable_if<NDimSpatial_ == 3, bool>::type = false>
        float RunAvgPoolBwd(const Argument& arg)
        {
            auto f_ncdhw = [&](auto n, auto c, auto di, auto hi, auto wi) {
                std::size_t Z = arg.window_spatial_lengths_[0];
                std::size_t Y = arg.window_spatial_lengths_[1];
                std::size_t X = arg.window_spatial_lengths_[2];

                std::size_t Do = arg.doutput_.GetLengths()[2];
                std::size_t Ho = arg.doutput_.GetLengths()[3];
                std::size_t Wo = arg.doutput_.GetLengths()[4];

                float v_acc = 0;

                for(std::size_t z = 0; z < Z; ++z)
                {
                    // Out_Position = (In_Position + pad - x * dilation) / stride
                    auto d_tmp = static_cast<ck::long_index_t>(di) +
                                 static_cast<ck::long_index_t>(arg.in_left_pads_[0]) -
                                 static_cast<ck::long_index_t>(z * arg.window_dilations_[0]);

                    // Check the input pixel validity (in perspective of being affected by some
                    // doutput pixel)
                    if(d_tmp % arg.window_strides_[0] == 0)
                    {
                        auto do_ = static_cast<ck::long_index_t>(d_tmp) /
                                   static_cast<ck::long_index_t>(arg.window_strides_[0]);

                        // Get the doutput pixel in valid range to accumulate the gradients for this
                        // input pixel
                        if(do_ >= 0 && ck::type_convert<std::size_t>(do_) < Do)
                        {
                            for(std::size_t y = 0; y < Y; ++y)
                            {
                                auto h_tmp =
                                    static_cast<ck::long_index_t>(hi) +
                                    static_cast<ck::long_index_t>(arg.in_left_pads_[1]) -
                                    static_cast<ck::long_index_t>(y * arg.window_dilations_[1]);
                                if(h_tmp % arg.window_strides_[1] == 0)
                                {
                                    auto ho = static_cast<ck::long_index_t>(h_tmp) /
                                              static_cast<ck::long_index_t>(arg.window_strides_[1]);
                                    if(ho >= 0 && ck::type_convert<std::size_t>(ho) < Ho)
                                    {
                                        for(std::size_t x = 0; x < X; ++x)
                                        {
                                            auto w_tmp = static_cast<ck::long_index_t>(wi) +
                                                         static_cast<ck::long_index_t>(
                                                             arg.in_left_pads_[2]) -
                                                         static_cast<ck::long_index_t>(
                                                             x * arg.window_dilations_[2]);

                                            if(w_tmp % arg.window_strides_[2] == 0)
                                            {
                                                auto wo = static_cast<ck::long_index_t>(w_tmp) /
                                                          static_cast<ck::long_index_t>(
                                                              arg.window_strides_[2]);
                                                if(wo >= 0 &&
                                                   ck::type_convert<std::size_t>(wo) < Wo)
                                                {
                                                    v_acc += ck::type_convert<float>(
                                                        arg.doutput_(n, c, do_, ho, wo));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                v_acc /= ck::type_convert<float>(Z * Y * X);
                arg.dinput_(n, c, di, hi, wi) = ck::type_convert<DInDataType>(v_acc);
            };

            make_ParallelTensorFunctor(f_ncdhw,
                                       arg.dinput_.GetLengths()[0],
                                       arg.dinput_.GetLengths()[1],
                                       arg.dinput_.GetLengths()[2],
                                       arg.dinput_.GetLengths()[3],
                                       arg.dinput_.GetLengths()[4])(
                std::thread::hardware_concurrency());

            return 0;
        }

        float Run(const Argument& arg)
        {
            if(!(arg.dinput_.GetNumOfDimension() == NDimSpatial + 2 &&
                 arg.doutput_.GetNumOfDimension() == NDimSpatial + 2))
            {
                throw std::runtime_error("wrong! inconsistent dimension");
            }

            return RunAvgPoolBwd<NDimSpatial>(arg);
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

    static auto MakeArgument(Tensor<DInDataType>& dinput,
                             const Tensor<DOutDataType>& doutput,
                             std::vector<ck::index_t> window_spatial_lengths,
                             std::vector<ck::index_t> window_strides,
                             std::vector<ck::index_t> window_dilations,
                             std::vector<ck::index_t> dinput_left_pads,
                             std::vector<ck::index_t> dinput_right_pads)
    {
        if(window_spatial_lengths.size() != NDimSpatial || window_strides.size() != NDimSpatial ||
           window_dilations.size() != NDimSpatial || dinput_left_pads.size() != NDimSpatial ||
           dinput_right_pads.size() != NDimSpatial)
            throw std::runtime_error("dimension is incorrect");

        return Argument{dinput,
                        doutput,
                        window_spatial_lengths,
                        window_strides,
                        window_dilations,
                        dinput_left_pads,
                        dinput_right_pads};
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
        str << "ReferenceAvgPoolBwd"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
