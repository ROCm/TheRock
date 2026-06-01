// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using InDataType  = FP32; // ck::bhalf_t;//FP32;
using OutDataType = FP32; // ck::bhalf_t;//FP32;

using ImLayout        = ck::tensor_layout::convolution::GNHWC;
using ColumnToImageOp = ck::conv_tensor_rearrange_op::ColumnToImage;

// clang-format off
using DeviceColToImgInstance = ck::tensor_operation::device::DeviceColumnToImageImpl
        //#####################|        Num| ImLayout| InDataType| OutDataType| Block|  MPer|  KPer|    Thread| Scalar|
        //#####################|        Dim|         |           |            |  Size| Block| Block|   Cluster|    Per|
        //#####################|    Spatial|         |           |            |      |      |      |   Lengths| Vector|
        //#####################|           |         |           |            |      |      |      |          |       |
                              < NDimSpatial, ImLayout, InDataType, OutDataType,   256,   128,   128, S<16, 16>,     1>;
// clang-format on

bool RunColumnToImage(const ExecutionConfig& config, const ck::utils::conv::ConvParam& conv_params)
{
    const auto G = conv_params.G_;
    const auto N = conv_params.N_;
    const auto C = conv_params.C_;

    const ck::index_t NDoHoWo =
        N * ck::accumulate_n<ck::index_t>(
                conv_params.output_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());
    const ck::index_t CZYX =
        C * ck::accumulate_n<ck::index_t>(
                conv_params.filter_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());

    const auto in_desc = HostTensorDescriptor({G, NDoHoWo, CZYX});
    const auto out_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<ImLayout>(conv_params);

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

    copy(conv_params.input_spatial_lengths_, input_spatial_lengths);
    copy(conv_params.filter_spatial_lengths_, filter_spatial_lengths);
    copy(conv_params.output_spatial_lengths_, output_spatial_lengths);
    copy(in_desc.GetStrides(), gemm_g_m_k_strides);
    copy(out_desc.GetStrides(), image_g_n_c_wis_strides);
    copy(conv_params.conv_filter_strides_, conv_filter_strides);
    copy(conv_params.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_params.input_left_pads_, input_left_pads);
    copy(conv_params.input_right_pads_, input_right_pads);

    Tensor<InDataType> in(in_desc);
    Tensor<OutDataType> out_device(out_desc);
    Tensor<OutDataType> out_host(out_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "out: " << out_device.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1: in.GenerateTensorValue(GeneratorTensor_2<InDataType>{1, 2}); break;
    default: in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());

    // reset input to zero
    out_device_buf.SetZero();

    static_assert(std::is_default_constructible_v<DeviceColToImgInstance>);

    // do conv
    auto col2img  = DeviceColToImgInstance{};
    auto invoker  = col2img.MakeInvoker();
    auto argument = col2img.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                         out_device_buf.GetDeviceBuffer(),
                                         G,
                                         N,
                                         C,
                                         input_spatial_lengths,
                                         filter_spatial_lengths,
                                         output_spatial_lengths,
                                         image_g_n_c_wis_strides,
                                         gemm_g_m_k_strides,
                                         conv_filter_strides,
                                         conv_filter_dilations,
                                         input_left_pads,
                                         input_right_pads);

    if(!col2img.IsSupportedArgument(argument))
    {
        std::cerr << "wrong! device_col2img with the specified compilation parameters does "
                     "not support this col2img problem"
                  << std::endl;

        return false;
    }

    float ave_time        = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
    std::size_t num_btype = G * NDoHoWo * CZYX * (sizeof(OutDataType) + sizeof(InDataType));
    float gb_per_sec      = num_btype / 1.E6 / ave_time;
    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    if(config.do_verification)
    {
        auto ref_column_to_image = ck::tensor_operation::host::
            ReferenceColumnToImage<NDimSpatial, ImLayout, InDataType, OutDataType>();

        auto ref_invoker = ref_column_to_image.MakeInvoker();

        auto ref_argument = ref_column_to_image.MakeArgument(in,
                                                             out_host,
                                                             conv_params.filter_spatial_lengths_,
                                                             conv_params.conv_filter_strides_,
                                                             conv_params.conv_filter_dilations_,
                                                             conv_params.input_left_pads_,
                                                             conv_params.input_right_pads_);

        if(!ref_column_to_image.IsSupportedArgument(&ref_argument))
        {
            std::cerr << "wrong! ref_col2img with the specified compilation parameters does "
                         "not support this col2img problem"
                      << std::endl;
            return false;
        }

        ref_invoker.Run(ref_argument);
        out_device_buf.FromDevice(out_device.mData.data());
        return ck::utils::check_err(out_device.mData, out_host.mData);
    }

    return true;
}

int RunColumnToImageExample(int argc, char* argv[])
{
    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_params = DefaultConvParams;

    if(!parse_cmd_args(argc, argv, config, conv_params))
    {
        return EXIT_FAILURE;
    }

    if(conv_params.num_dim_spatial_ != NDimSpatial)
    {
        std::cerr << "unsupported # of spatial dimensions" << std::endl;
        return EXIT_FAILURE;
    }

    return !RunColumnToImage(config, conv_params);
}

int main(int argc, char* argv[]) { return RunColumnToImageExample(argc, argv); }
