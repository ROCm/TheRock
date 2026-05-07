// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_multiple_d_xdl_cshuffle.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

constexpr ck::index_t NDimSpatial = 3;
using InDataType                  = ck::half_t;
using WeiDataType                 = ck::half_t;
using AccDataType                 = float;
using OutDataType                 = ck::half_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InLayout  = ck::tensor_layout::convolution::GNDHWC;
using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
using OutLayout = ck::tensor_layout::convolution::GNDHWK;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::Bilinear;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

template <typename WeiElementOp>
using DeviceGroupedConvNDBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle<
        NDimSpatial,
        InLayout,               // InLayout
        WeiLayout,              // WeiLayout
        OutLayout,              // OutLayout
        ck::Tuple<WeiLayout>,   // DsLayout
        InDataType,             // InDataType
        WeiDataType,            // WeiDataType
        OutDataType,            // OutDataType
        AccDataType,            // AccDataType
        ck::Tuple<WeiDataType>, // DsLayout
        InElementOp,            // InElementwiseOperation
        WeiElementOp,           // WeiElementwiseOperation
        OutElementOp,           // OutElementwiseOperation
        ConvBwdWeightDefault,   // ConvolutionBackwardWeightSpecialization
        256,                    // BlockSize
        128,                    // MPerBlock
        128,                    // NPerBlock
        4,                      // K0PerBlock
        8,                      // K1
        16,                     // MPerXdl
        16,                     // NPerXdl
        4,                      // MXdlPerWave
        4,                      // NXdlPerWave
        S<1, 4, 16, 4>,         // ABlockTransferThreadClusterLengths_K0_M_K1
        S<0, 3, 1, 2>,          // ABlockTransferThreadClusterArrangeOrder
        S<0, 2, 1, 3>,          // ABlockTransferSrcAccessOrder
        2,                      // ABlockTransferSrcVectorDim
        8,                      // ABlockTransferSrcScalarPerVector
        2,                      // ABlockTransferDstScalarPerVector_K1
        true,                   // ABlockLdsAddExtraM
        S<1, 4, 16, 4>,         // BBlockTransferThreadClusterLengths_K0_N_K1
        S<0, 3, 1, 2>,          // BBlockTransferThreadClusterArrangeOrder
        S<0, 2, 1, 3>,          // BBlockTransferSrcAccessOrder
        2,                      // BBlockTransferSrcVectorDim
        8,                      // BBlockTransferSrcScalarPerVector
        2,                      // BBlockTransferDstScalarPerVector_K1
        true,                   // BBlockLdsAddExtraN
        1,                      // CShuffleMXdlPerWavePerShuffle
        1,                      // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 4>,         // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        64 / (sizeof(WeiDataType) * CHAR_BIT)>; // CBlockTransferScalarPerVector_NWaveNPerXdl
using DeviceGroupedConvNDActivInstance = DeviceGroupedConvNDBwdWeightInstance<WeiElementOp>;

namespace {
// Use custom implementation to pass two more tensors for post op
template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp,
          typename DeviceConvNDFwdInstance>
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
    constexpr ck::index_t split_k = 1;
    constexpr ck::index_t NumDs   = 1;
    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei_host(wei_g_k_c_xs_desc);
    Tensor<OutDataType> out(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei_host.mDesc << std::endl;
    std::cout << "out: " << out.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        wei_host.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        out.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
        wei_host.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
    }

    // Initialize based on wei_host
    Tensor<WeiDataType> wei_device(wei_host);

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_device.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei_device.mData.data());
    out_device_buf.ToDevice(out.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> b_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), b_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), b_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), e_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), e_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), a_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), a_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    // Use weight as D
    const std::array<const void*, NumDs> ds = {wei_device_buf.GetDeviceBuffer()};

    auto conv     = DeviceConvNDFwdInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(
        static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
        static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
        static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
        ds,
        b_g_n_c_wis_lengths,
        b_g_n_c_wis_strides,
        e_g_k_c_xs_lengths,
        e_g_k_c_xs_strides,
        a_g_n_k_wos_lengths,
        a_g_n_k_wos_strides,
        std::array<std::array<ck::index_t, NDimSpatial + 3>, NumDs>{e_g_k_c_xs_lengths},
        std::array<std::array<ck::index_t, NDimSpatial + 3>, NumDs>{e_g_k_c_xs_strides},
        conv_filter_strides,
        conv_filter_dilations,
        input_left_pads,
        input_right_pads,
        in_element_op,
        wei_element_op,
        out_element_op,
        split_k);

    DeviceMem workspace_buf(argument.GetWorkspaceSizeBytes());
    conv.SetWorkSpacePointer(&argument, workspace_buf.GetDeviceBuffer());

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error("The device op with the specified compilation parameters does "
                                 "not support this convolution problem.");
    }
    float avg_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop =
        conv_param.GetFlops() + 3 * conv_param.GetOutputByte<WeiDataType>() / sizeof(WeiDataType);
    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>() +
                            conv_param.GetOutputByte<WeiDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    if(do_verification)
    {
        std::array<Tensor<OutDataType>, NumDs> d_tensors = {wei_host};
        auto ref_conv =
            ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                               InDataType,
                                                               WeiDataType,
                                                               OutDataType,
                                                               InElementOp,
                                                               WeiElementOp,
                                                               OutElementOp,
                                                               0, /*Num A Elementwise Tensors*/
                                                               0, /*Num B Elementwise Tensors*/
                                                               NumDs>{};

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei_host,
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
        wei_device_buf.FromDevice(wei_device.mData.data());

        return ck::utils::check_err(wei_device, wei_host, "Error: incorrect results!");
    }

    return true;
}

} // namespace

#include "../run_convnd_activ_example.inc"

int main(int argc, char* argv[])
{
    // temp disable test on gfx11
    if(ck::is_gfx11_supported())
    {
        return 0;
    }
    return !run_convnd_example(argc, argv);
}
