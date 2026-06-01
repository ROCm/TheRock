// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>

#include "ck/ck.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using PassThrough   = ck::tensor_operation::element_wise::PassThrough;
using ConvScaleRelu = ck::tensor_operation::element_wise::ConvScaleRelu;

void print_helper_msg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

template <typename DataType>
inline __host__ __device__ constexpr double get_rtol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 1e-1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 1.5e-1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <typename DataType>
inline __host__ __device__ constexpr double get_atol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 16.1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 8192.1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <ck::index_t NumDimSpatial, ck::index_t NumNonSpatialDim = 3>
std::size_t
GetFlops(const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& output_lengths,
         const std::array<ck::index_t, NumDimSpatial + NumNonSpatialDim>& weights_lengths,
         const std::size_t& ds_size)
{
    // G * N * C * <output spatial lengths product> * (2 * K * <filter spatial lengths product> +
    // <number of scale factors>)
    ck::index_t G = weights_lengths[0];
    ck::index_t N = output_lengths[1];
    ck::index_t K = weights_lengths[1];
    ck::index_t C = weights_lengths[2];

    return G * N * C *
           std::accumulate(std::next(std::begin(output_lengths), NumNonSpatialDim),
                           std::end(output_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<>()) *
           (static_cast<std::size_t>(2) * K *
                std::accumulate(std::next(std::begin(weights_lengths), NumNonSpatialDim),
                                std::end(weights_lengths),
                                static_cast<std::size_t>(1),
                                std::multiplies<>()) +
            ds_size);
}

template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp,
          typename DeviceConvNDFwdInstance>
bool run_grouped_conv_fwd(bool do_verification,
                          int init_method,
                          bool time_kernel,
                          const ck::utils::conv::ConvParam& conv_param,
                          const HostTensorDescriptor& in_g_n_c_wis_desc,
                          const HostTensorDescriptor& wei_g_k_c_xs_desc,
                          const HostTensorDescriptor& out_g_n_k_wos_desc,
                          const InElementOp& in_element_op,
                          const WeiElementOp& wei_element_op)
{
    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<CShuffleDataType> c(out_g_n_k_wos_desc);
    Tensor<OutDataType> out_host(out_g_n_k_wos_desc);
    Tensor<OutDataType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-1.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

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

    // random scale values
    float scale_in  = float(std::rand()) / float(RAND_MAX);
    float scale_wei = float(std::rand()) / float(RAND_MAX);
    float scale_out = float(std::rand()) / float(RAND_MAX);

    std::cout << std::endl;
    std::cout << "scale_in: " << scale_in << std::endl;
    std::cout << "scale_wei: " << scale_wei << std::endl;
    std::cout << "scale_out: " << scale_out << std::endl;

    // initialize out_element_op for each iteration
    const auto out_element_op = OutElementOp{scale_in, scale_wei, scale_out};

    // do Conv
    auto conv     = DeviceConvNDFwdInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      std::array<const void*, 0>{},
                                      out_device_buf.GetDeviceBuffer(),
                                      a_g_n_c_wis_lengths,
                                      a_g_n_c_wis_strides,
                                      b_g_k_c_xs_lengths,
                                      b_g_k_c_xs_strides,
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                      e_g_n_k_wos_lengths,
                                      e_g_n_k_wos_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      in_element_op,
                                      wei_element_op,
                                      out_element_op);

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t ds_size   = 3 + 1; // 3 element-wise scale multipliers + 1 element-wise relu
    std::size_t flop      = GetFlops<NDimSpatial>(e_g_n_k_wos_lengths, b_g_k_c_xs_lengths, ds_size);
    std::size_t num_btype = conv_param.GetInputByte<InDataType>() +
                            conv_param.GetWeightByte<WeiDataType>() + sizeof(float) +
                            sizeof(float) + sizeof(float) + conv_param.GetOutputByte<OutDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    if(do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     CShuffleDataType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     PassThrough>();

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei,
                                                  c,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);

        out_host.ForEach([&](auto&, auto idx) { out_element_op(out_host(idx), c(idx)); });

        out_device_buf.FromDevice(out_device.mData.data());

        return ck::utils::check_err(out_device,
                                    out_host,
                                    "Error: incorrect results!",
                                    get_rtol<OutDataType>(),
                                    get_atol<OutDataType>());
    }

    return true;
}
