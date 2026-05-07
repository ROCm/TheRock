// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>
#include <limits>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_tensor_rearrange.hpp"
#include "ck/tensor_operation/gpu/device/conv_tensor_rearrange_op.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_image_to_column_impl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_column_to_image_impl.hpp"
#include "ck/library/tensor_operation_instance/gpu/conv_tensor_rearrange.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_image_to_column.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_column_to_image.hpp"

namespace ck {
namespace profiler {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using namespace conv_tensor_rearrange_op;

template <typename InputDataType, typename ConvTensorRearrangeOp>
Tensor<InputDataType> create_input(const HostTensorDescriptor& image_desc,
                                   const HostTensorDescriptor& gemm_desc)
{
    if constexpr(std::is_same_v<ConvTensorRearrangeOp, ImageToColumn>)
    {
        Tensor<InputDataType> input(image_desc);
        return input;
    }
    else if constexpr(std::is_same_v<ConvTensorRearrangeOp, ColumnToImage>)
    {
        Tensor<InputDataType> input(gemm_desc);
        return input;
    }
    else
    {
        throw std::runtime_error("Unsupported op!");
    }
}

template <typename OutputDataType, typename ConvTensorRearrangeOp>
Tensor<OutputDataType> create_output(const HostTensorDescriptor& image_desc,
                                     const HostTensorDescriptor& gemm_desc)
{
    if constexpr(std::is_same_v<ConvTensorRearrangeOp, ImageToColumn>)
    {
        Tensor<OutputDataType> output(gemm_desc);
        return output;
    }
    else if constexpr(std::is_same_v<ConvTensorRearrangeOp, ColumnToImage>)
    {
        Tensor<OutputDataType> output(image_desc);
        return output;
    }
    else
    {
        throw std::runtime_error("Unsupported op!");
    }
}

template <index_t NDimSpatial,
          typename InputLayout,
          typename InputDataType,
          typename OutputDataType,
          typename ConvTensorRearrangeOp>
static auto make_ref_op()
{
    if constexpr(std::is_same_v<ConvTensorRearrangeOp, ImageToColumn>)
    {
        return ck::tensor_operation::host::
            ReferenceImageToColumn<NDimSpatial, InputLayout, InputDataType, OutputDataType>{};
    }
    else if constexpr(std::is_same_v<ConvTensorRearrangeOp, ColumnToImage>)
    {
        return ck::tensor_operation::host::
            ReferenceColumnToImage<NDimSpatial, InputLayout, InputDataType, OutputDataType>{};
    }
    else
    {
        throw std::runtime_error("Unsupported op!");
    }
}

template <typename InputLayout>
static auto create_gemm_desc(const ck::index_t G, const ck::index_t NDoHoWo, const ck::index_t CZYX)
{
    using namespace ck::tensor_layout::convolution;
    if constexpr(std::is_same_v<InputLayout, GNWC> || std::is_same_v<InputLayout, GNHWC> ||
                 std::is_same_v<InputLayout, GNDHWC>)
    {
        return HostTensorDescriptor({G, NDoHoWo, CZYX}, InputLayout{});
    }
    else if constexpr(std::is_same_v<InputLayout, NWGC> || std::is_same_v<InputLayout, NHWGC> ||
                      std::is_same_v<InputLayout, NDHWGC>)
    {
        return HostTensorDescriptor({G, NDoHoWo, CZYX}, {CZYX, CZYX * G, 1}, InputLayout{});
    }
    else
    {
        throw std::runtime_error("Unsupported layout!");
    }
}

template <index_t NDimSpatial,
          typename InputLayout,
          typename InputDataType,
          typename OutputDataType,
          typename ConvTensorRearrangeOp>
bool profile_conv_tensor_rearrange_impl(int do_verification,
                                        int init_method,
                                        bool do_log,
                                        bool time_kernel,
                                        const ck::utils::conv::ConvParam& conv_param,
                                        index_t instance_index = -1)
{
    const ck::index_t NDoHoWo =
        conv_param.N_ *
        ck::accumulate_n<ck::index_t>(
            conv_param.output_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());
    const ck::index_t CZYX =
        conv_param.C_ *
        ck::accumulate_n<ck::index_t>(
            conv_param.filter_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());

    const auto image_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InputLayout>(
            conv_param);
    const auto gemm_desc = create_gemm_desc<InputLayout>(conv_param.G_, NDoHoWo, CZYX);

    std::array<ck::index_t, NDimSpatial> input_spatial_lengths{};
    std::array<ck::index_t, NDimSpatial> filter_spatial_lengths{};
    std::array<ck::index_t, NDimSpatial> output_spatial_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> image_g_n_c_wis_strides{};
    std::array<ck::index_t, 3> gemm_g_m_k_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { std::copy(x.begin(), x.end(), y.begin()); };

    copy(conv_param.input_spatial_lengths_, input_spatial_lengths);
    copy(conv_param.filter_spatial_lengths_, filter_spatial_lengths);
    copy(conv_param.output_spatial_lengths_, output_spatial_lengths);
    copy(image_desc.GetStrides(), image_g_n_c_wis_strides);
    copy(gemm_desc.GetStrides(), gemm_g_m_k_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    Tensor<InputDataType> input =
        create_input<InputDataType, ConvTensorRearrangeOp>(image_desc, gemm_desc);
    Tensor<OutputDataType> device_output =
        create_output<OutputDataType, ConvTensorRearrangeOp>(image_desc, gemm_desc);
    Tensor<OutputDataType> host_output =
        create_output<OutputDataType, ConvTensorRearrangeOp>(image_desc, gemm_desc);

    std::cout << "input: " << input.mDesc << std::endl;
    std::cout << "output: " << host_output.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1: input.GenerateTensorValue(GeneratorTensor_2<InputDataType>{-5, 5}); break;
    default: input.GenerateTensorValue(GeneratorTensor_3<InputDataType>{0.0, 1.0});
    }

    DeviceMem in_device_buf(sizeof(InputDataType) * input.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutputDataType) * device_output.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(input.mData.data());

    // run reference op
    if(do_verification)
    {
        auto ref_conv_tensor_rearrange = make_ref_op<NDimSpatial,
                                                     InputLayout,
                                                     InputDataType,
                                                     OutputDataType,
                                                     ConvTensorRearrangeOp>();

        auto ref_invoker = ref_conv_tensor_rearrange.MakeInvoker();
        auto ref_argument =
            ref_conv_tensor_rearrange.MakeArgument(input,
                                                   host_output,
                                                   conv_param.filter_spatial_lengths_,
                                                   conv_param.conv_filter_strides_,
                                                   conv_param.conv_filter_dilations_,
                                                   conv_param.input_left_pads_,
                                                   conv_param.input_right_pads_);

        // init host output to zero
        host_output.SetZero();

        ref_invoker.Run(ref_argument);
    }

    using DeviceOp = ck::tensor_operation::device::DeviceConvTensorRearrange<NDimSpatial,
                                                                             InputLayout,
                                                                             InputDataType,
                                                                             OutputDataType,
                                                                             ConvTensorRearrangeOp>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    // profile device op instances
    bool pass                   = true;
    bool is_supporting_instance = false;
    index_t num_kernel          = 0;
    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            static_cast<InputDataType*>(in_device_buf.GetDeviceBuffer()),
            static_cast<OutputDataType*>(out_device_buf.GetDeviceBuffer()),
            conv_param.G_,
            conv_param.N_,
            conv_param.C_,
            input_spatial_lengths,
            filter_spatial_lengths,
            output_spatial_lengths,
            image_g_n_c_wis_strides,
            gemm_g_m_k_strides,
            conv_filter_strides,
            conv_filter_dilations,
            input_left_pads,
            input_right_pads);

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            num_kernel++;
            if((instance_index != -1) && (instance_index + 1 != num_kernel))
            {
                // skip test if instance_index is specified
                continue;
            }
            is_supporting_instance = true;
            // re-init output to zero before profiling next kernel
            out_device_buf.SetZero();
            std::string op_name = op_ptr->GetTypeString();
            auto invoker_ptr    = op_ptr->MakeInvokerPointer();
            float avg_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});
            std::size_t num_btype =
                conv_param.G_ * NDoHoWo * CZYX * (sizeof(OutputDataType) + sizeof(InputDataType));
            float gb_per_sec = num_btype / 1.E6 / avg_time;
            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << gb_per_sec << " GB/s, "
                      << op_name << std::endl;

            if(avg_time < best_avg_time)
            {
                best_op_name    = op_name;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                out_device_buf.FromDevice(device_output.mData.data());
                pass = pass & ck::utils::check_err(device_output, host_output);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "input : ", input.mData, ",") << std::endl;
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
    }

    std::cout << "Best configuration parameters:" << "\nname: " << best_op_name
              << "\navg_time: " << best_avg_time << "\nGB/s: " << best_gb_per_sec << std::endl;

    if(instance_index != -1)
    {
        std::cout << "conv_tensor_rearrange_instance (" << instance_index << "/" << num_kernel
                  << "): Passed" << std::endl;
    }
    return is_supporting_instance && pass;
}

} // namespace profiler
} // namespace ck
