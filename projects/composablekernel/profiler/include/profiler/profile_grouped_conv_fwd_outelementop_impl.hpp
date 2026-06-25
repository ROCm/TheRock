// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convscale.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convinvscale.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "profiler/common.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

namespace ck {
namespace profiler {

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename OutElementOp,
          typename AComputeType = InDataType,
          typename BComputeType = AComputeType>
bool profile_grouped_conv_fwd_outelementop_impl(int do_verification,
                                                int init_method,
                                                bool do_log,
                                                bool time_kernel,
                                                const ck::utils::conv::ConvParam& conv_param)
{
    auto pass = true; // return status

    using CShuffleDataType = float;

    using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
    using InElementOp  = PassThrough;
    using WeiElementOp = PassThrough;

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    Tensor<InDataType> input(in_g_n_c_wis_desc);
    Tensor<WeiDataType> weight(wei_g_k_c_xs_desc);
    Tensor<CShuffleDataType> c(out_g_n_k_wos_desc);
    Tensor<OutDataType> host_output(out_g_n_k_wos_desc);
    Tensor<OutDataType> device_output(out_g_n_k_wos_desc);

    std::cout << "input: " << input.mDesc << std::endl;
    std::cout << "weight: " << weight.mDesc << std::endl;
    std::cout << "output: " << host_output.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        input.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        weight.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-1, 1});
        break;
    default:
        input.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0});
        weight.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-1.0, 1.0});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weight.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * device_output.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(input.mData.data());
    wei_device_buf.ToDevice(weight.mData.data());

    // random scale values
    auto scale_in = type_convert<float>(
        type_convert<f8_t>(2.0f * float(RAND_MAX / 2 - std::rand()) / float(RAND_MAX)));
    auto scale_wei = type_convert<float>(
        type_convert<f8_t>(2.0f * float(RAND_MAX / 2 - std::rand()) / float(RAND_MAX)));
    auto scale_out = type_convert<float>(
        type_convert<f8_t>(2.0f * float(RAND_MAX / 2 - std::rand()) / float(RAND_MAX)));

    OutElementOp out_element_op;
    if constexpr(std::is_same_v<OutElementOp, ck::tensor_operation::element_wise::ScaleScalePass>)
    {
        using Scale    = ck::tensor_operation::element_wise::Scale;
        out_element_op = OutElementOp{Scale{scale_in}, Scale{scale_wei}, PassThrough{}};
    }
    else if constexpr(std::is_same_v<OutElementOp,
                                     ck::tensor_operation::element_wise::ScaleScaleRelu>)
    {
        using Scale    = ck::tensor_operation::element_wise::Scale;
        using Relu     = ck::tensor_operation::element_wise::Relu;
        out_element_op = OutElementOp{Scale{scale_in}, Scale{scale_wei}, Relu{}};
    }
    else if constexpr(std::is_same_v<OutElementOp, ck::tensor_operation::element_wise::Scale>)
    {
        out_element_op = OutElementOp{scale_out};
    }
    else
    {
        out_element_op = OutElementOp{scale_in, scale_wei, scale_out};
    }

    std::cout << "scale_in: " << scale_in << std::endl;
    std::cout << "scale_wei: " << scale_wei << std::endl;
    std::cout << "scale_out: " << scale_out << std::endl;

    // run reference op
    if(do_verification)
    {

        std::cout << "\nVerifying algorithm against reference convolution..." << std::endl;
        std::cout << "\tUsing (rel_tol,abs_tol) = (" << std::setprecision(7)
                  << get_rtol<OutDataType>() << ", " << get_atol<OutDataType>() << ")" << std::endl;

        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     CShuffleDataType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     PassThrough>{};

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(input,
                                                  weight,
                                                  c,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  PassThrough{});

        c.SetZero();
        ref_invoker.Run(ref_argument);

        host_output.ForEach([&](auto&, auto idx) {
            if constexpr(std::is_same_v<OutElementOp, ck::tensor_operation::element_wise::Scale>)
            {
                const auto conv_shuffle = ck::type_convert<CShuffleDataType>(c(idx));
                if constexpr(std::is_same_v<OutDataType, int8_t>)
                {
                    const auto conv_val = ck::type_convert<OutDataType>(conv_shuffle);
                    out_element_op(host_output(idx), conv_val);
                }
                else
                {
                    out_element_op(host_output(idx), conv_shuffle);
                }
            }
            else
            {
                out_element_op(host_output(idx), c(idx));
            }
        });
    }

    std::string best_op_name;
    float best_avg_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    int valids            = 0;

    auto run_impl = [&](auto& op_ptr, auto& argument_ptr) {
        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            // re-init output to zero before profiling next kernel
            out_device_buf.SetZero();

            std::string op_name = op_ptr->GetTypeString();
            valids++;

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            float avg_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop      = conv_param.GetFlops();
            std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

            float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

            float gb_per_sec = num_btype / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                out_device_buf.FromDevice(device_output.mData.data());

                pass = pass & ck::utils::check_err(device_output,
                                                   host_output,
                                                   "Error: Device and Host results do not match!",
                                                   get_rtol<OutDataType>(),
                                                   get_atol<OutDataType>());

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "input : ", input.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "weight: ", weight.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "host_output  : ", host_output.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "device_output: ", device_output.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    };

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                                   InLayout,
                                                                                   WeiLayout,
                                                                                   ck::Tuple<>,
                                                                                   OutLayout,
                                                                                   InDataType,
                                                                                   WeiDataType,
                                                                                   ck::Tuple<>,
                                                                                   OutDataType,
                                                                                   InElementOp,
                                                                                   WeiElementOp,
                                                                                   OutElementOp,
                                                                                   AComputeType,
                                                                                   BComputeType>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "ckProfiler found " << op_ptrs.size() << " instances" << std::endl;

    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(in_device_buf.GetDeviceBuffer(),
                                                        wei_device_buf.GetDeviceBuffer(),
                                                        {},
                                                        out_device_buf.GetDeviceBuffer(),
                                                        a_g_n_c_wis_lengths,
                                                        a_g_n_c_wis_strides,
                                                        b_g_k_c_xs_lengths,
                                                        b_g_k_c_xs_strides,
                                                        {},
                                                        {},
                                                        e_g_n_k_wos_lengths,
                                                        e_g_n_k_wos_strides,
                                                        conv_filter_strides,
                                                        conv_filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        in_element_op,
                                                        wei_element_op,
                                                        out_element_op);

        run_impl(op_ptr, argument_ptr);
    }

    printf("\033[36mvalids: %d\033[0m\n", valids);

    std::cout << "Best configuration parameters:" << "\nname: " << best_op_name
              << "\navg_time: " << best_avg_time << "\ntflops: " << best_tflops
              << "\nGB/s: " << best_gb_per_sec << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
