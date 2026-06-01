// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef DEVICE_CONV3D_FWD_NAIVE_HPP
#define DEVICE_CONV3D_FWD_NAIVE_HPP

#include <iostream>
#include <memory>
#include <sstream>
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/stream_config.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_fwd_gpu.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// specialization for #D conv: in[n, di, hi, wi, c] * wei[k, z, y, x, c] = out[n, do, ho, wo, k]
template <typename InDataType,
          typename WeiDataType, // WeiDataType must be the same as InDataType
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
struct DeviceConv3dFwdNaive_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K
    : public DeviceConvFwd<3,
                           ck::tensor_layout::convolution::NDHWC,
                           ck::tensor_layout::convolution::KZYXC,
                           ck::tensor_layout::convolution::NDHWK,
                           InDataType,
                           WeiDataType,
                           OutDataType,
                           InElementwiseOperation,
                           WeiElementwiseOperation,
                           OutElementwiseOperation>

{
    using DeviceOp = DeviceConv3dFwdNaive_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K;

    using ADataType = InDataType;
    using BDataType = WeiDataType;
    using CDataType = OutDataType;
    // TODO make A/B datatype different
    using ABDataType = InDataType;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in,
                 const WeiDataType* p_wei,
                 OutDataType* p_out,
                 const index_t N,
                 const index_t K,
                 const index_t C,
                 std::vector<ck::index_t> input_spatial_lengths,
                 std::vector<ck::index_t> filter_spatial_lengths,
                 std::vector<ck::index_t> output_spatial_lengths,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : params_{3,
                      1, // G (group count, always 1 for non-grouped)
                      N,
                      K,
                      C,
                      filter_spatial_lengths,
                      input_spatial_lengths,
                      conv_filter_strides,
                      conv_filter_dilations,
                      input_left_pads,
                      input_right_pads},
              out_spatial_lengths_{output_spatial_lengths},
              p_in_{p_in},
              p_wei_{p_wei},
              p_out_{p_out},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}

        {
        }

        //  private:
        utils::conv::ConvParam params_;
        std::vector<index_t> out_spatial_lengths_;

        const InDataType* p_in_;
        const WeiDataType* p_wei_;
        OutDataType* p_out_;

        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            using InLayout  = ck::tensor_layout::convolution::GNCDHW;
            using WeiLayout = ck::tensor_layout::convolution::GKCZYX;
            using OutLayout = ck::tensor_layout::convolution::GNKDHW;

            // Use simplified ConvParam-based API
            ref::naive_conv_fwd<InLayout,
                                WeiLayout,
                                OutLayout,
                                InDataType,
                                WeiDataType,
                                OutDataType,
                                InElementwiseOperation,
                                WeiElementwiseOperation,
                                OutElementwiseOperation>(arg.p_in_,
                                                         arg.p_wei_,
                                                         arg.p_out_,
                                                         arg.params_,
                                                         arg.in_element_op_,
                                                         arg.wei_element_op_,
                                                         arg.out_element_op_,
                                                         stream_config.stream_id_);
            return 0; // No timing for naive implementation
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        auto out_spatial_lengths_long = arg.params_.GetOutputSpatialLengths();
        std::vector<index_t> out_spatial_lengths(out_spatial_lengths_long.begin(),
                                                 out_spatial_lengths_long.end());

        bool out_lengths_are_consistent = out_spatial_lengths[0] == arg.out_spatial_lengths_[0] &&
                                          out_spatial_lengths[1] == arg.out_spatial_lengths_[1] &&
                                          out_spatial_lengths[2] == arg.out_spatial_lengths_[2];
        return out_lengths_are_consistent;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const InDataType* p_in,
                             const WeiDataType* p_wei,
                             OutDataType* p_out,
                             const index_t N,
                             const index_t K,
                             const index_t C,
                             std::vector<ck::index_t> input_spatial_lengths,
                             std::vector<ck::index_t> filter_spatial_lengths,
                             std::vector<ck::index_t> output_spatial_lengths,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{p_in,
                        p_wei,
                        p_out,
                        N,
                        K,
                        C,
                        input_spatial_lengths,
                        filter_spatial_lengths,
                        output_spatial_lengths,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        const void* p_wei,
                        void* p_out,
                        const index_t N,
                        const index_t K,
                        const index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) override

    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in),
                                          static_cast<const WeiDataType*>(p_wei),
                                          static_cast<OutDataType*>(p_out),
                                          N,
                                          K,
                                          C,
                                          input_spatial_lengths,
                                          filter_spatial_lengths,
                                          output_spatial_lengths,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceConv3dFwdNaive_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K<>";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
