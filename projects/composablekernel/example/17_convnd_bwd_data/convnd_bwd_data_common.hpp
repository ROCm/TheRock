// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_bwd_data_gpu.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/host_utility/hip_check_error.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <typename DataType, typename GemmType = DataType>
inline __host__ __device__ constexpr double get_rtol()
{
    if constexpr(std::is_same_v<DataType, float> && std::is_same_v<GemmType, ck::tf32_t>)
        return 5e-3;
    else if constexpr(std::is_same_v<DataType, float>)
        return 1e-3;
    else if constexpr(std::is_same_v<DataType, double>)
        return 1e-6;
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
        return 1e-3;
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
        return 5e-2;
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
        return 1e-1;
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
        return 1.5e-1;
    else
        return 1e-3;
}

template <typename DataType, typename GemmType = DataType>
inline __host__ __device__ constexpr double get_atol()
{
    if constexpr(std::is_same_v<DataType, float> && std::is_same_v<GemmType, ck::tf32_t>)
        return 1e-3;
    else if constexpr(std::is_same_v<DataType, float>)
        return 1e-3;
    else if constexpr(std::is_same_v<DataType, double>)
        return 1e-6;
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
        return 1e-3;
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
        return 5e-2;
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
        return 16.1;
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
        return 16.1;
    else
        return 1e-3;
}

void print_helper_msg()
{
    std::cout << "arg1: verification (0=no, 1=CPU, 2=GPU)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp,
          typename DeviceConvNdBwdDataInstance,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
int run_conv_bwd_data(int do_verification,
                      int init_method,
                      bool time_kernel,
                      const ck::utils::conv::ConvParam& conv_param,
                      const HostTensorDescriptor& in_g_n_c_wis_desc,
                      const HostTensorDescriptor& wei_g_k_c_xs_desc,
                      const HostTensorDescriptor& out_g_n_k_wos_desc,
                      const InElementOp& in_element_op,
                      const WeiElementOp& wei_element_op,
                      const OutElementOp& out_element_op)
{
    Tensor<InDataType> in_host(in_g_n_c_wis_desc);
    Tensor<InDataType> in_device(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<OutDataType> out(out_g_n_k_wos_desc);

    std::cout << "in: " << in_host.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out.mDesc << std::endl;

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

    DeviceMem in_device_buf(sizeof(InDataType) * in_device.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());

    out_device_buf.ToDevice(out.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    // reset input to zero
    in_device_buf.SetZero();

    std::vector<ck::index_t> input_spatial_lengths_i32(NDimSpatial);
    std::vector<ck::index_t> filter_spatial_lengths_i32(NDimSpatial);
    std::vector<ck::index_t> output_spatial_lengths_i32(NDimSpatial);
    std::vector<ck::index_t> conv_filter_strides_i32(NDimSpatial);
    std::vector<ck::index_t> conv_filter_dilations_i32(NDimSpatial);
    std::vector<ck::index_t> input_left_pads_i32(NDimSpatial);
    std::vector<ck::index_t> input_right_pads_i32(NDimSpatial);

    for(ck::index_t d = 0; d < NDimSpatial; d++)
    {
        input_spatial_lengths_i32[d] =
            static_cast<ck::index_t>(conv_param.input_spatial_lengths_[d]);
        filter_spatial_lengths_i32[d] =
            static_cast<ck::index_t>(conv_param.filter_spatial_lengths_[d]);
        output_spatial_lengths_i32[d] =
            static_cast<ck::index_t>(conv_param.GetOutputSpatialLengths()[d]);
        conv_filter_strides_i32[d] = static_cast<ck::index_t>(conv_param.conv_filter_strides_[d]);
        conv_filter_dilations_i32[d] =
            static_cast<ck::index_t>(conv_param.conv_filter_dilations_[d]);
        input_left_pads_i32[d]  = static_cast<ck::index_t>(conv_param.input_left_pads_[d]);
        input_right_pads_i32[d] = static_cast<ck::index_t>(conv_param.input_right_pads_[d]);
    }

    // do GEMM
    auto conv    = DeviceConvNdBwdDataInstance{};
    auto invoker = conv.MakeInvoker();
    auto argument =
        conv.MakeArgumentPointer(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                 static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                 static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                 static_cast<ck::index_t>(conv_param.N_),
                                 static_cast<ck::index_t>(conv_param.K_),
                                 static_cast<ck::index_t>(conv_param.C_),
                                 input_spatial_lengths_i32,
                                 filter_spatial_lengths_i32,
                                 output_spatial_lengths_i32,
                                 conv_filter_strides_i32,
                                 conv_filter_dilations_i32,
                                 input_left_pads_i32,
                                 input_right_pads_i32,
                                 in_element_op,
                                 wei_element_op,
                                 out_element_op);

    // Check if optimized kernel supports these parameters
    if(!conv.IsSupportedArgument(argument.get()))
    {
        std::cout << "Not support,please check parameters or device";
        return 0;
    }

    // Run optimized kernel
    float ave_time = invoker.Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::size_t flop      = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    std::cout << "do_verification = " << do_verification << std::endl;

    if(do_verification == 1)
    {
        // CPU verification
        auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                                         InDataType,
                                                                         WeiDataType,
                                                                         OutDataType,
                                                                         InElementOp,
                                                                         WeiElementOp,
                                                                         OutElementOp>();

        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(in_host,
                                                  wei,
                                                  out,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);

        ref_invoker.Run(ref_argument);

        in_device_buf.FromDevice(in_device.mData.data());

        return ck::utils::check_err(in_device, in_host) ? 0 : 1;
    }
    else if(do_verification == 2)
    {
        // GPU verification using naive GPU reference
        std::cout << "Running GPU verification..." << std::endl;

        // Allocate and ZERO GPU memory for reference input
        DeviceMem in_device_ref_buf(sizeof(InDataType) * in_device.mDesc.GetElementSpaceSize());
        in_device_ref_buf.SetZero();

        // Call GPU reference with ConvParam directly, using the correct layout types
        ck::ref::naive_conv_bwd_data<InLayout,
                                     WeiLayout,
                                     OutLayout,
                                     InDataType,
                                     WeiDataType,
                                     OutDataType,
                                     InElementOp,
                                     WeiElementOp,
                                     OutElementOp>(
            reinterpret_cast<InDataType*>(in_device_ref_buf.GetDeviceBuffer()),
            reinterpret_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
            reinterpret_cast<const OutDataType*>(out_device_buf.GetDeviceBuffer()),
            conv_param,
            in_element_op,
            wei_element_op,
            out_element_op);

        HIP_CHECK_ERROR(hipDeviceSynchronize());

        std::cout << "GPU reference function completed successfully, copying results..."
                  << std::endl;

        // Copy GPU reference result to host
        Tensor<InDataType> in_gpu_ref(in_host.mDesc);
        in_device_ref_buf.FromDevice(in_gpu_ref.mData.data());

        // Copy GPU kernel result to host
        in_device_buf.FromDevice(in_device.mData.data());

        std::cout << "Comparing GPU kernel output vs GPU reference..." << std::endl;

        // Compare: Optimized kernel result vs GPU reference result
        bool pass = ck::utils::check_err(in_device,
                                         in_gpu_ref,
                                         "Error: Incorrect results!",
                                         get_rtol<InDataType, float>(),
                                         get_atol<InDataType, float>());

        std::cout << "GPU verification result is:" << (pass ? "correct" : "fail") << std::endl;

        return pass ? 0 : 1;
    }

    return 0;
}
