// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_bwd_data_gpu.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data.hpp"

namespace ck {
namespace profiler {

template <ck::index_t NDimSpatial,
          typename OutLayout,
          typename WeiLayout,
          typename InLayout,
          typename OutDataType,
          typename WeiDataType,
          typename InDataType,
          typename ComputeDataType = InDataType>
bool profile_grouped_conv_bwd_data_impl(int do_verification,
                                        int init_method,
                                        bool do_log,
                                        bool time_kernel,
                                        const ck::utils::conv::ConvParam& conv_param,
                                        ck::index_t split_k    = 1,
                                        index_t instance_index = -1)
{
    using OutElementOp = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;

    const auto out_element_op = OutElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto in_element_op  = InElementOp{};

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    Tensor<OutDataType> out(out_g_n_k_wos_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<InDataType> in_host(in_g_n_c_wis_desc);
    Tensor<InDataType> in_device(in_g_n_c_wis_desc);

    std::cout << "out: " << out.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "in: " << in_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    case 2:
        out.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        break;
    default:
        out.GenerateTensorValue(GeneratorTensor_1<OutDataType>{1});
        wei.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{1});
    }

    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem in_device_buf(sizeof(InDataType) * in_device.mDesc.GetElementSpaceSize());

    out_device_buf.ToDevice(out.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    float max_accumulated_value = 0;
    if(do_verification == 2)
    {
        // Use GPU reference for verification
        std::cout << "Using GPU reference for verification" << std::endl;

        // Allocate GPU reference output buffer
        DeviceMem gpu_ref_in_buf(sizeof(InDataType) * in_host.mDesc.GetElementSpaceSize());

        // Call GPU reference with ConvParam directly
        ref::naive_conv_bwd_data<InLayout,
                                 WeiLayout,
                                 OutLayout,
                                 InDataType,
                                 WeiDataType,
                                 OutDataType,
                                 InElementOp,
                                 WeiElementOp,
                                 OutElementOp>(
            reinterpret_cast<InDataType*>(gpu_ref_in_buf.GetDeviceBuffer()),
            reinterpret_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
            reinterpret_cast<const OutDataType*>(out_device_buf.GetDeviceBuffer()),
            conv_param,
            in_element_op,
            wei_element_op,
            out_element_op);

        // Copy GPU reference result to host for comparison
        gpu_ref_in_buf.FromDevice(in_host.mData.data());
        max_accumulated_value = *std::max_element(in_host.mData.begin(), in_host.mData.end());
    }
    else if(do_verification == 1)
    {
        // Use CPU reference for verification (default)
        auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                                         InDataType,
                                                                         WeiDataType,
                                                                         OutDataType,
                                                                         InElementOp,
                                                                         WeiElementOp,
                                                                         OutElementOp,
                                                                         0,
                                                                         0,
                                                                         0,
                                                                         ComputeDataType>();

        auto ref_invoker = ref_conv.MakeInvoker();

        in_host.SetZero();

        auto ref_argument = ref_conv.MakeArgument(in_host,
                                                  wei,
                                                  out,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  out_element_op,
                                                  wei_element_op,
                                                  in_element_op);

        ref_invoker.Run(ref_argument);
        max_accumulated_value = *std::max_element(in_host.mData.begin(), in_host.mData.end());
    }

    std::string best_op_name;
    float best_avg_time      = 0;
    float best_tflops        = 0;
    float best_gb_per_sec    = 0;
    ck::index_t best_split_k = 1;

    // profile device op instances
    bool pass          = true;
    index_t num_kernel = 0;
    auto run_impl      = [&](auto& op_ptr, auto& argument_ptr, const index_t& split_k_for_run) {
        // workspace_sz will be equal to 0 for other layout than NGCHW
        const std::size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
        DeviceMem workspace_dev(workspace_sz);
        op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            num_kernel++;
            if((instance_index != -1) && (instance_index + 1 != num_kernel))
            {
                // skip test if instance_index is specified
                return;
            }
            std::string op_name = op_ptr->GetTypeString();

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            float avg_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop      = conv_param.GetFlops();
            std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

            float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

            float gb_per_sec = num_btype / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << ", SplitK " << split_k_for_run
                      << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
                best_split_k    = split_k_for_run;
            }

            if(do_verification)
            {
                in_device_buf.FromDevice(in_device.mData.data());

                using ComputeType_ = std::conditional_t<sizeof(OutDataType) < sizeof(WeiDataType),
                                                             OutDataType,
                                                             WeiDataType>;
                using ComputeType =
                    std::conditional_t<sizeof(ComputeType_) < sizeof(ComputeDataType),
                                            ComputeType_,
                                            ComputeDataType>;
                using AccDataType =
                    std::conditional_t<std::is_same_v<ComputeType, int8_t>, int32_t, float>;
                const index_t num_accums = conv_param.K_;
                // Calculate thresholds
                auto rtol = ck::utils::get_relative_threshold<ComputeType, InDataType, AccDataType>(
                    num_accums / split_k_for_run);
                auto atol = ck::utils::get_absolute_threshold<ComputeType, InDataType, AccDataType>(
                    max_accumulated_value / split_k_for_run, num_accums / split_k_for_run);
                // Calculate error due to split_k accumulation
                auto rtol_split_k =
                    ck::utils::get_relative_threshold<InDataType, InDataType, InDataType>(
                        split_k_for_run);
                auto atol_split_k =
                    ck::utils::get_absolute_threshold<InDataType, InDataType, InDataType>(
                        max_accumulated_value, split_k_for_run);
                // Use higher threshold
                rtol = std::max(rtol, rtol_split_k);
                atol = std::max(atol, atol_split_k);
                if(split_k_for_run > 1)
                {
                    pass &= ck::utils::check_err(
                        in_device, in_host, "Error: Incorrect results!", rtol, atol);
                    std::cout << "Relative error threshold: " << rtol
                              << " Absolute error threshold: " << atol << std::endl;
                }
                else
                {
                    pass &= ck::utils::check_err(in_device, in_host, "Error: Incorrect results!");
                }

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "output : ", out.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "weight: ", wei.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "in_host  : ", in_host.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "in_device: ", in_device.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    };

    // do GEMM
    using DeviceOp =
        ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<NDimSpatial,
                                                                        OutLayout,
                                                                        WeiLayout,
                                                                        ck::Tuple<>,
                                                                        InLayout,
                                                                        OutDataType,
                                                                        WeiDataType,
                                                                        ck::Tuple<>,
                                                                        InDataType,
                                                                        OutElementOp,
                                                                        WeiElementOp,
                                                                        InElementOp,
                                                                        ComputeDataType,
                                                                        ComputeDataType>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::array<ck::index_t, NDimSpatial + 3> out_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> out_strides{};
    std::array<ck::index_t, NDimSpatial + 3> wei_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> wei_strides{};
    std::array<ck::index_t, NDimSpatial + 3> in_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> in_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(out_g_n_k_wos_desc.GetLengths(), out_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), out_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), wei_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), wei_strides);
    copy(in_g_n_c_wis_desc.GetLengths(), in_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), in_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    std::vector<ck::index_t> split_k_list = {1, 2, 4, 8, 16, 32, 64, 128};

    if(split_k > 0)
    {
        split_k_list = {split_k};
    }

    for(auto& op_ptr : op_ptrs)
    {
        for(std::size_t split_k_id = 0; split_k_id < split_k_list.size(); split_k_id++)
        {
            auto argument_ptr = op_ptr->MakeArgumentPointer(
                static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                {},
                static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                out_lengths,
                out_strides,
                wei_lengths,
                wei_strides,
                {},
                {},
                in_lengths,
                in_strides,
                conv_filter_strides,
                conv_filter_dilations,
                input_left_pads,
                input_right_pads,
                out_element_op,
                wei_element_op,
                in_element_op,
                split_k_list[split_k_id]);

            run_impl(op_ptr, argument_ptr, split_k_list[split_k_id]);
        }
    }

    std::cout << "Best configuration parameters:" << "\nname: " << best_op_name
              << "\navg_time: " << best_avg_time << "\ntflops: " << best_tflops
              << "\nGB/s: " << best_gb_per_sec << ", SplitK " << best_split_k << std::endl;

    if(instance_index != -1)
    {
        std::cout << "grouped_conv_bwd_data_instance (" << instance_index << "/" << num_kernel
                  << "): Passed" << std::endl;
    }
    return pass;
}

} // namespace profiler
} // namespace ck
