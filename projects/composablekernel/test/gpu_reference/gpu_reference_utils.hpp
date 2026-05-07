// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

// CPU references
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"

// GPU references
#include "ck/library/reference_tensor_operation/gpu/naive_conv_fwd_gpu.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_bwd_data_gpu.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_bwd_weight_gpu.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "common_test_params.hpp"

namespace ck {
namespace test {

enum class ConvKernelType
{
    Forward,
    BackwardData,
    BackwardWeight
};

// Helper function to initialize and copy a tensor to device
template <typename DataType>
void initialize_and_copy_tensor(Tensor<DataType>& host_tensor, DeviceMem& device_mem)
{
    host_tensor.GenerateTensorValue(GeneratorTensor_2<DataType>{-5, 5});
    device_mem.ToDevice(host_tensor.mData.data());
}

// Helper to get default layout types based on NDimSpatial
template <index_t NDimSpatial>
struct DefaultConvLayouts
{
    using InLayout  = std::conditional_t<NDimSpatial == 3,
                                         tensor_layout::convolution::GNCDHW,
                                         std::conditional_t<NDimSpatial == 2,
                                                            tensor_layout::convolution::GNCHW,
                                                            tensor_layout::convolution::GNCW>>;
    using WeiLayout = std::conditional_t<NDimSpatial == 3,
                                         tensor_layout::convolution::GKCZYX,
                                         std::conditional_t<NDimSpatial == 2,
                                                            tensor_layout::convolution::GKCYX,
                                                            tensor_layout::convolution::GKCX>>;
    using OutLayout = std::conditional_t<NDimSpatial == 3,
                                         tensor_layout::convolution::GNKDHW,
                                         std::conditional_t<NDimSpatial == 2,
                                                            tensor_layout::convolution::GNKHW,
                                                            tensor_layout::convolution::GNKW>>;
};

// Forward convolution implementation
template <index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
bool test_conv_fwd_impl(const ck::utils::conv::ConvParam& params,
                        const Tensor<InDataType>& input_cpu,
                        const Tensor<WeiDataType>& weight_cpu,
                        DeviceMem& input_dev,
                        DeviceMem& weight_dev,
                        DeviceMem& output_dev)
{
    using InElementOp  = tensor_operation::element_wise::PassThrough;
    using WeiElementOp = tensor_operation::element_wise::PassThrough;
    using OutElementOp = tensor_operation::element_wise::PassThrough;

    // Call GPU reference with ConvParam directly
    ref::naive_conv_fwd<InLayout,
                        WeiLayout,
                        OutLayout,
                        InDataType,
                        WeiDataType,
                        OutDataType,
                        InElementOp,
                        WeiElementOp,
                        OutElementOp>(
        reinterpret_cast<const InDataType*>(input_dev.GetDeviceBuffer()),
        reinterpret_cast<const WeiDataType*>(weight_dev.GetDeviceBuffer()),
        reinterpret_cast<OutDataType*>(output_dev.GetDeviceBuffer()),
        params);

    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Run CPU reference
    std::vector<long_index_t> strides_long(params.conv_filter_strides_.begin(),
                                           params.conv_filter_strides_.end());
    std::vector<long_index_t> dilations_long(params.conv_filter_dilations_.begin(),
                                             params.conv_filter_dilations_.end());
    std::vector<long_index_t> pads_long(params.input_left_pads_.begin(),
                                        params.input_left_pads_.end());

    Tensor<InDataType> input_ref   = input_cpu;
    Tensor<WeiDataType> weight_ref = weight_cpu;
    Tensor<OutDataType> output_ref(
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(params));

    auto ref_conv    = tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                InDataType,
                                                                WeiDataType,
                                                                OutDataType,
                                                                InElementOp,
                                                                WeiElementOp,
                                                                OutElementOp>();
    auto ref_invoker = ref_conv.MakeInvoker();
    auto ref_arg     = ref_conv.MakeArgument(input_ref,
                                         weight_ref,
                                         output_ref,
                                         strides_long,
                                         dilations_long,
                                         pads_long,
                                         pads_long,
                                         InElementOp{},
                                         WeiElementOp{},
                                         OutElementOp{});
    ref_invoker.Run(ref_arg);

    // Copy result from device and compare
    Tensor<OutDataType> output_gpu(output_ref.mDesc);
    output_dev.FromDevice(output_gpu.mData.data());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Compare results
    return ck::utils::check_err(output_gpu, output_ref);
}

// Backward data convolution implementation
template <index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
bool test_conv_bwd_data_impl(const ck::utils::conv::ConvParam& params,
                             const Tensor<WeiDataType>& weight_cpu,
                             const Tensor<OutDataType>& output_cpu,
                             DeviceMem& weight_dev,
                             DeviceMem& output_dev,
                             DeviceMem& input_dev)
{
    using InElementOp  = tensor_operation::element_wise::PassThrough;
    using WeiElementOp = tensor_operation::element_wise::PassThrough;
    using OutElementOp = tensor_operation::element_wise::PassThrough;

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
        reinterpret_cast<InDataType*>(input_dev.GetDeviceBuffer()),
        reinterpret_cast<const WeiDataType*>(weight_dev.GetDeviceBuffer()),
        reinterpret_cast<const OutDataType*>(output_dev.GetDeviceBuffer()),
        params);

    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Run CPU reference
    std::vector<long_index_t> strides_long(params.conv_filter_strides_.begin(),
                                           params.conv_filter_strides_.end());
    std::vector<long_index_t> dilations_long(params.conv_filter_dilations_.begin(),
                                             params.conv_filter_dilations_.end());
    std::vector<long_index_t> pads_long(params.input_left_pads_.begin(),
                                        params.input_left_pads_.end());

    Tensor<InDataType> input_ref(
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(params));
    Tensor<WeiDataType> weight_ref = weight_cpu;
    Tensor<OutDataType> output_ref = output_cpu;

    auto ref_conv    = tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                                    InDataType,
                                                                    WeiDataType,
                                                                    OutDataType,
                                                                    InElementOp,
                                                                    WeiElementOp,
                                                                    OutElementOp>();
    auto ref_invoker = ref_conv.MakeInvoker();
    auto ref_arg     = ref_conv.MakeArgument(input_ref,
                                         weight_ref,
                                         output_ref,
                                         strides_long,
                                         dilations_long,
                                         pads_long,
                                         pads_long,
                                         InElementOp{},
                                         WeiElementOp{},
                                         OutElementOp{});
    ref_invoker.Run(ref_arg);

    // Copy result from device and compare
    Tensor<InDataType> input_gpu(input_ref.mDesc);
    input_dev.FromDevice(input_gpu.mData.data());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Compare results
    return ck::utils::check_err(input_gpu, input_ref);
}

// Backward weight convolution implementation
template <index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
bool test_conv_bwd_weight_impl(const ck::utils::conv::ConvParam& params,
                               const Tensor<InDataType>& input_cpu,
                               const Tensor<OutDataType>& output_cpu,
                               DeviceMem& input_dev,
                               DeviceMem& output_dev,
                               DeviceMem& weight_dev)
{
    using InElementOp  = tensor_operation::element_wise::PassThrough;
    using WeiElementOp = tensor_operation::element_wise::PassThrough;
    using OutElementOp = tensor_operation::element_wise::PassThrough;

    // Call GPU reference with ConvParam directly
    ref::naive_conv_bwd_weight<InLayout,
                               WeiLayout,
                               OutLayout,
                               InDataType,
                               WeiDataType,
                               OutDataType,
                               InElementOp,
                               WeiElementOp,
                               OutElementOp>(
        reinterpret_cast<const InDataType*>(input_dev.GetDeviceBuffer()),
        reinterpret_cast<WeiDataType*>(weight_dev.GetDeviceBuffer()),
        reinterpret_cast<const OutDataType*>(output_dev.GetDeviceBuffer()),
        params);

    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Run CPU reference
    std::vector<long_index_t> strides_long(params.conv_filter_strides_.begin(),
                                           params.conv_filter_strides_.end());
    std::vector<long_index_t> dilations_long(params.conv_filter_dilations_.begin(),
                                             params.conv_filter_dilations_.end());
    std::vector<long_index_t> pads_long(params.input_left_pads_.begin(),
                                        params.input_left_pads_.end());

    Tensor<InDataType> input_ref = input_cpu;
    Tensor<WeiDataType> weight_ref(
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(params));
    Tensor<OutDataType> output_ref = output_cpu;

    auto ref_conv    = tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                      InDataType,
                                                                      WeiDataType,
                                                                      OutDataType,
                                                                      InElementOp,
                                                                      WeiElementOp,
                                                                      OutElementOp>();
    auto ref_invoker = ref_conv.MakeInvoker();
    auto ref_arg     = ref_conv.MakeArgument(input_ref,
                                         weight_ref,
                                         output_ref,
                                         strides_long,
                                         dilations_long,
                                         pads_long,
                                         pads_long,
                                         InElementOp{},
                                         WeiElementOp{},
                                         OutElementOp{});
    ref_invoker.Run(ref_arg);

    // Copy result from device and compare
    Tensor<WeiDataType> weight_gpu(weight_ref.mDesc);
    weight_dev.FromDevice(weight_gpu.mData.data());
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    // Compare results
    return ck::utils::check_err(weight_gpu, weight_ref);
}

// Main test function - dispatches to specific implementations
template <index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout  = typename DefaultConvLayouts<NDimSpatial>::InLayout,
          typename WeiLayout = typename DefaultConvLayouts<NDimSpatial>::WeiLayout,
          typename OutLayout = typename DefaultConvLayouts<NDimSpatial>::OutLayout>
bool test_conv_gpu_ref(const ck::utils::conv::ConvParam& params, ConvKernelType kernel_type)
{
    // Create tensor descriptors using the specified layouts
    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(params);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(params);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(params);

    // Create tensors using tensor descriptors (supports multiple layouts)
    Tensor<InDataType> input(in_g_n_c_wis_desc);
    Tensor<WeiDataType> weight(wei_g_k_c_xs_desc);
    Tensor<OutDataType> output(out_g_n_k_wos_desc);

    // Allocate device memory
    DeviceMem input_dev(input.mData.size() * sizeof(InDataType));
    DeviceMem weight_dev(weight.mData.size() * sizeof(WeiDataType));
    DeviceMem output_dev(output.mData.size() * sizeof(OutDataType));

    // Initialize and copy tensors based on kernel type
    if(kernel_type == ConvKernelType::Forward)
    {
        initialize_and_copy_tensor(input, input_dev);
        initialize_and_copy_tensor(weight, weight_dev);
    }
    else if(kernel_type == ConvKernelType::BackwardData)
    {
        initialize_and_copy_tensor(weight, weight_dev);
        initialize_and_copy_tensor(output, output_dev);
    }
    else // BackwardWeight
    {
        initialize_and_copy_tensor(input, input_dev);
        initialize_and_copy_tensor(output, output_dev);
    }

    // Dispatch to appropriate implementation with layout types
    if(kernel_type == ConvKernelType::Forward)
    {
        return test_conv_fwd_impl<NDimSpatial,
                                  InDataType,
                                  WeiDataType,
                                  OutDataType,
                                  InLayout,
                                  WeiLayout,
                                  OutLayout>(
            params, input, weight, input_dev, weight_dev, output_dev);
    }
    else if(kernel_type == ConvKernelType::BackwardData)
    {
        return test_conv_bwd_data_impl<NDimSpatial,
                                       InDataType,
                                       WeiDataType,
                                       OutDataType,
                                       InLayout,
                                       WeiLayout,
                                       OutLayout>(
            params, weight, output, weight_dev, output_dev, input_dev);
    }
    else // BackwardWeight
    {
        return test_conv_bwd_weight_impl<NDimSpatial,
                                         InDataType,
                                         WeiDataType,
                                         OutDataType,
                                         InLayout,
                                         WeiLayout,
                                         OutLayout>(
            params, input, output, input_dev, output_dev, weight_dev);
    }
}

} // namespace test
} // namespace ck
