// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <type_traits>
#include <sstream>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/numeric.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

/**
 * \brief Reference implementation for image to column.
 *
 * Input tensor descriptor has [G, N, C, Di, Hi, Wi] data layout.
 * Output tensor descriptor has [G * N * Do * Ho * Wo, Z * Y * X * C] data layout.
 *
 * \tparam NDimSpatial Number of spatial dimensions.
 * \tparam ImageLayout Image Layout.
 * \tparam InDataType Input Data Type.
 * \tparam OutDataType Output Data Type.
 */
template <ck::index_t NDimSpatial,
          typename ImageLayout,
          typename InDataType,
          typename OutDataType,
          typename std::enable_if<NDimSpatial >= 1 && NDimSpatial <= 3, bool>::type = false>
struct ReferenceImageToColumn : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        public:
        Argument(const Tensor<InDataType>& input,
                 Tensor<OutDataType>& output,
                 std::vector<ck::long_index_t> filter_spatial_lengths,
                 std::vector<ck::long_index_t> conv_filter_strides,
                 std::vector<ck::long_index_t> conv_filter_dilations,
                 std::vector<ck::long_index_t> input_left_pads,
                 std::vector<ck::long_index_t> input_right_pads)
            : input_{input},
              output_{output},
              conv_strides_{conv_filter_strides},
              conv_dilations_{conv_filter_dilations},
              in_left_pads_{input_left_pads},
              in_right_pads_{input_right_pads},
              filter_spatial_lengths_{filter_spatial_lengths}
        {
            initOutputSpatialLengths();
        }

        const Tensor<InDataType>& input_;
        Tensor<OutDataType>& output_;

        std::vector<long_index_t> conv_strides_;
        std::vector<long_index_t> conv_dilations_;
        std::vector<long_index_t> in_left_pads_;
        std::vector<long_index_t> in_right_pads_;

        std::vector<long_index_t> filter_spatial_lengths_;
        std::vector<long_index_t> output_spatial_lengths_;

        private:
        void initOutputSpatialLengths()
        {
            constexpr auto input_offset_to_spatial = 3;

            for(ck::index_t i = 0; i < NDimSpatial; ++i)
            {
                // XEff = (X - 1) * conv_dilation_w + 1;
                // Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
                const ck::long_index_t x_eff =
                    (filter_spatial_lengths_[i] - 1) * conv_dilations_[i] + 1;

                output_spatial_lengths_.push_back(
                    (input_.GetLengths()[i + input_offset_to_spatial] + in_left_pads_[i] +
                     in_right_pads_[i] - x_eff) /
                        conv_strides_[i] +
                    1);
            }
        }
    };

    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceImageToColumn::Argument;

        float Run(const Argument& arg)
        {
            if(!(arg.input_.GetNumOfDimension() == NDimSpatial + 3 &&
                 arg.output_.GetNumOfDimension() == 3))
            {
                throw std::runtime_error("wrong! inconsistent dimension");
            }

            const long_index_t G = arg.input_.GetLengths()[0];
            const long_index_t N = arg.input_.GetLengths()[1];
            const long_index_t C = arg.input_.GetLengths()[2];

            if constexpr(NDimSpatial == 1)
            {
                const long_index_t Wo = arg.output_spatial_lengths_[0];
                auto func             = [&](auto g, auto n, auto wo) {
                    long_index_t row    = n * Wo + wo;
                    long_index_t column = 0;

                    for(long_index_t x = 0; x < arg.filter_spatial_lengths_[0]; ++x)
                    {
                        auto wi = static_cast<ck::long_index_t>(wo * arg.conv_strides_[0]) +
                                  static_cast<ck::long_index_t>(x * arg.conv_dilations_[0]) -
                                  static_cast<ck::long_index_t>(arg.in_left_pads_[0]);

                        for(long_index_t c = 0; c < C; ++c)
                        {
                            if(wi >= 0 &&
                               ck::type_convert<std::size_t>(wi) < arg.input_.GetLengths()[3])
                            {
                                InDataType v_in             = arg.input_(g, n, c, wi);
                                arg.output_(g, row, column) = ck::type_convert<OutDataType>(v_in);
                            }
                            column++;
                        }
                    }
                };

                make_ParallelTensorFunctor(func, G, N, Wo)(std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NDimSpatial == 2)
            {
                const long_index_t Ho = arg.output_spatial_lengths_[0];
                const long_index_t Wo = arg.output_spatial_lengths_[1];

                auto func = [&](auto g, auto n, auto ho, auto wo) {
                    long_index_t row    = n * Ho * Wo + ho * Wo + wo;
                    long_index_t column = 0;

                    for(long_index_t y = 0; y < arg.filter_spatial_lengths_[0]; ++y)
                    {
                        auto hi = static_cast<ck::long_index_t>(ho * arg.conv_strides_[0]) +
                                  static_cast<ck::long_index_t>(y * arg.conv_dilations_[0]) -
                                  static_cast<ck::long_index_t>(arg.in_left_pads_[0]);

                        for(long_index_t x = 0; x < arg.filter_spatial_lengths_[1]; ++x)
                        {
                            auto wi = static_cast<ck::long_index_t>(wo * arg.conv_strides_[1]) +
                                      static_cast<ck::long_index_t>(x * arg.conv_dilations_[1]) -
                                      static_cast<ck::long_index_t>(arg.in_left_pads_[1]);

                            for(long_index_t c = 0; c < C; ++c)
                            {

                                if(hi >= 0 &&
                                   ck::type_convert<std::size_t>(hi) < arg.input_.GetLengths()[3] &&
                                   wi >= 0 &&
                                   ck::type_convert<std::size_t>(wi) < arg.input_.GetLengths()[4])
                                {
                                    InDataType v_in = arg.input_(g, n, c, hi, wi);
                                    arg.output_(g, row, column) =
                                        ck::type_convert<OutDataType>(v_in);
                                }
                                column++;
                            }
                        }
                    }
                };

                make_ParallelTensorFunctor(func, G, N, Ho, Wo)(std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NDimSpatial == 3)
            {
                const long_index_t Do = arg.output_spatial_lengths_[0];
                const long_index_t Ho = arg.output_spatial_lengths_[1];
                const long_index_t Wo = arg.output_spatial_lengths_[2];

                auto func = [&](auto g, auto n, auto d_o, auto ho, auto wo) {
                    long_index_t row    = n * Do * Ho * Wo + d_o * Ho * Wo + ho * Wo + wo;
                    long_index_t column = 0;

                    for(long_index_t z = 0; z < arg.filter_spatial_lengths_[0]; ++z)
                    {
                        auto di = static_cast<ck::long_index_t>(d_o * arg.conv_strides_[0]) +
                                  static_cast<ck::long_index_t>(z * arg.conv_dilations_[0]) -
                                  static_cast<ck::long_index_t>(arg.in_left_pads_[0]);
                        for(long_index_t y = 0; y < arg.filter_spatial_lengths_[1]; ++y)
                        {
                            auto hi = static_cast<ck::long_index_t>(ho * arg.conv_strides_[1]) +
                                      static_cast<ck::long_index_t>(y * arg.conv_dilations_[1]) -
                                      static_cast<ck::long_index_t>(arg.in_left_pads_[1]);
                            for(long_index_t x = 0; x < arg.filter_spatial_lengths_[2]; ++x)
                            {
                                auto wi =
                                    static_cast<ck::long_index_t>(wo * arg.conv_strides_[2]) +
                                    static_cast<ck::long_index_t>(x * arg.conv_dilations_[2]) -
                                    static_cast<ck::long_index_t>(arg.in_left_pads_[2]);
                                for(long_index_t c = 0; c < C; ++c)
                                {
                                    if(di >= 0 &&
                                       ck::type_convert<std::size_t>(di) <
                                           arg.input_.GetLengths()[3] &&
                                       hi >= 0 &&
                                       ck::type_convert<std::size_t>(hi) <
                                           arg.input_.GetLengths()[4] &&
                                       wi >= 0 &&
                                       ck::type_convert<std::size_t>(wi) <
                                           arg.input_.GetLengths()[5])
                                    {
                                        InDataType v_in = arg.input_(g, n, c, di, hi, wi);
                                        arg.output_(g, row, column) =
                                            ck::type_convert<OutDataType>(v_in);
                                    }
                                    column++;
                                }
                            }
                        }
                    }
                };

                make_ParallelTensorFunctor(func, G, N, Do, Ho, Wo)(
                    std::thread::hardware_concurrency());

                return 0;
            }
            throw std::runtime_error("Img2Col: number of dimensions should be between 1 and 3.");
            return 1;
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /*stream_config*/ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        using namespace tensor_layout::convolution;

        if constexpr(!(std::is_same_v<ImageLayout, GNWC> || std::is_same_v<ImageLayout, GNHWC> ||
                       std::is_same_v<ImageLayout, GNDHWC>))
        {
            return false;
        }
        if constexpr(!(NDimSpatial >= 1 && NDimSpatial <= 3))
        {
            return false;
        }
        return true;
    }

    bool IsSupportedArgument(const Argument& arg)
    {
        const ck::long_index_t G = arg.input_.GetLengths()[0];
        const ck::long_index_t N = arg.input_.GetLengths()[1];
        const ck::long_index_t C = arg.input_.GetLengths()[2];

        const long_index_t NDoHoWo =
            N * ck::accumulate_n<long_index_t>(
                    arg.output_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());
        const long_index_t CZYX =
            C * ck::accumulate_n<long_index_t>(
                    arg.filter_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());

        if(!(arg.output_.GetLengths()[0] == static_cast<std::size_t>(G) &&
             arg.output_.GetLengths()[1] == static_cast<std::size_t>(NDoHoWo) &&
             arg.output_.GetLengths()[2] == static_cast<std::size_t>(CZYX)))
        {
            return false;
        }

        if(G != 1)
        {
            return false;
        }
        return true;
    }

    bool IsSupportedArgument(const device::BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const Tensor<InDataType>& input,
                             Tensor<OutDataType>& output,
                             std::vector<ck::long_index_t> filter_spatial_lengths,
                             std::vector<ck::long_index_t> conv_filter_strides,
                             std::vector<ck::long_index_t> conv_filter_dilations,
                             std::vector<ck::long_index_t> input_left_pads,
                             std::vector<ck::long_index_t> input_right_pads)
    {
        return Argument{input,
                        output,
                        filter_spatial_lengths,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads};
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
        str << "ReferenceImageToColumn"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
