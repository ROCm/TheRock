// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

constexpr ck::index_t NDimSpatial = 3;
using InDataType                  = ck::half_t;
using WeiDataType                 = ck::half_t;
using AccDataType                 = float;
using CShuffleDataType            = ck::half_t;
using OutDataType                 = ck::half_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InLayout  = ck::tensor_layout::convolution::GNDHWC;
using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
using OutLayout = ck::tensor_layout::convolution::GNDHWK;

using OutElementOp = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using InElementOp  = ck::tensor_operation::element_wise::Bilinear;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default;

template <typename OutElementOp>
using DeviceGroupedConvNDBwdDataInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<
        NDimSpatial,
        OutLayout,
        WeiLayout,
        ck::Tuple<InLayout>,
        InLayout,
        OutDataType,
        WeiDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<InDataType>,
        InDataType,
        OutElementOp,
        WeiElementOp,
        InElementOp,
        ConvSpec, // ConvForwardSpecialization
        true,
        true,
        1,           //
        256,         // BlockSize
        128,         // MPerBlock
        256,         // NPerBlock
        32,          // KPerBlock
        8,           // AK1
        2,           // BK1
        16,          // MPerXdl
        16,          // NPerXdl
        4,           // MXdlPerWave
        8,           // NXdlPerWave
        S<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,           // ABlockTransferSrcVectorDim
        8,           // ABlockTransferSrcScalarPerVector
        8,           // ABlockTransferDstScalarPerVector_AK1
        1,           // ABlockLdsExtraM
        S<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<0, 2, 1>,  // BBlockTransferThreadClusterArrangeOrder
        S<0, 2, 1>,  // BBlockTransferSrcAccessOrder
        1,           // BBlockTransferSrcVectorDim
        4,           // BBlockTransferSrcScalarPerVector
        2,           // BBlockTransferDstScalarPerVector_BK1
        0,           // BBlockLdsExtraN
        1,
        1,
        S<1, 32, 1, 8>,
        4>;

using DeviceGroupedConvNDActivInstance = DeviceGroupedConvNDBwdDataInstance<OutElementOp>;

namespace {
// Use custom implementation to pass two more tensors for post op
template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp,
          typename DeviceConvNDInstance>
bool run_grouped_conv(bool do_verification,
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
    constexpr ck::index_t NumDs = 1;
    Tensor<OutDataType> out(out_g_n_k_wos_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<InDataType> in_host(in_g_n_c_wis_desc);

    std::cout << "out: " << out.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "in: " << in_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        in_host.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        break;
    default:
        out.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        in_host.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
    }

    // Initialize based on out_host
    Tensor<InDataType> in_device(in_host);

    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem in_device_buf(sizeof(InDataType) * in_device.mDesc.GetElementSpaceSize());

    out_device_buf.ToDevice(out.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    in_device_buf.ToDevice(in_device.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(out_g_n_k_wos_desc.GetLengths(), a_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), a_g_n_k_wos_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(in_g_n_c_wis_desc.GetLengths(), e_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), e_g_n_c_wis_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    // Use output as D
    const std::array<const void*, NumDs> ds = {in_device_buf.GetDeviceBuffer()};

    auto conv     = DeviceConvNDInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(
        out_device_buf.GetDeviceBuffer(),
        wei_device_buf.GetDeviceBuffer(),
        ds,
        in_device_buf.GetDeviceBuffer(),
        a_g_n_k_wos_lengths,
        a_g_n_k_wos_strides,
        b_g_k_c_xs_lengths,
        b_g_k_c_xs_strides,
        std::array<std::array<ck::index_t, NDimSpatial + 3>, NumDs>{e_g_n_c_wis_lengths},
        std::array<std::array<ck::index_t, NDimSpatial + 3>, NumDs>{e_g_n_c_wis_strides},
        e_g_n_c_wis_lengths,
        e_g_n_c_wis_strides,
        conv_filter_strides,
        conv_filter_dilations,
        input_left_pads,
        input_right_pads,
        out_element_op,
        wei_element_op,
        in_element_op);

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error("The device op with the specified compilation parameters does "
                                 "not support this convolution problem.");
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop =
        conv_param.GetFlops() + 3 * conv_param.GetInputByte<InDataType>() / sizeof(InDataType);
    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>() +
                            conv_param.GetOutputByte<InDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    if(do_verification)
    {
        std::array<Tensor<OutDataType>, NumDs> d_tensors = {in_host};
        auto ref_conv =
            ck::tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                             InDataType,
                                                             WeiDataType,
                                                             OutDataType,
                                                             InElementOp,
                                                             WeiElementOp,
                                                             OutElementOp,
                                                             0, /*Num A Elementwise Tensors*/
                                                             0, /*Num B Elementwise Tensors*/
                                                             NumDs>();

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
                                                  out_element_op,
                                                  {},
                                                  {},
                                                  d_tensors);

        ref_invoker.Run(ref_argument);

        in_device_buf.FromDevice(in_device.mData.data());

        return ck::utils::check_err(in_device.mData, in_host.mData);
    }

    return true;
}

} // namespace

#include "../run_convnd_activ_example.inc"

int main(int argc, char* argv[]) { return !run_convnd_example(argc, argv); }
