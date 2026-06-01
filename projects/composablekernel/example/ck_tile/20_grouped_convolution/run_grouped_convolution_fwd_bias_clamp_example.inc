// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

using BiasAndClamp = ck_tile::element_wise::
    Compose<ck_tile::element_wise::MultiDAdd, ck_tile::element_wise::Clamp, true>;

template <ck_tile::index_t NDimSpatial,
          typename GemmWarpConfig,
          typename Invoker,
          typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
float invoke_grouped_conv_fwd_bias_clamp(const ck_tile::GroupedConvFwdHostArgs<BiasAndClamp>& args,
                                         int n_warmup,
                                         int n_repeat)
{
    float ave_time = Invoker::template grouped_conv_fwd<NDimSpatial,
                                                        GemmWarpConfig,
                                                        InDataType,
                                                        WeiDataType,
                                                        AccDataType,
                                                        OutDataType,
                                                        InLayout,
                                                        WeiLayout,
                                                        OutLayout,
                                                        ck_tile::tuple<OutDataType>,
                                                        ck_tile::tuple<OutLayout>,
                                                        BiasAndClamp>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::size_t flop     = args.GetFlops();
    std::size_t num_byte = args.GetByte<InDataType, WeiDataType, OutDataType>();
    float tflops         = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec     = num_byte / 1.E6 / ave_time;

    std::cout << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << std::endl;

    return ave_time;
}

template <ck_tile::index_t NDimSpatial,
          typename GemmWarpConfig,
          typename Invoker,
          typename InDataType,
          typename WeiDataType = InDataType,
          typename OutDataType = InDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
int run_grouped_conv_fwd_bias_clamp_example_with_layouts(
    int argc, char* argv[], const InLayout, const WeiLayout, const OutLayout)
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using AccDataType = float;

    std::vector<ck_tile::index_t> filter_spatial_lengths;
    std::vector<ck_tile::index_t> image_spatial_lengths;
    std::vector<ck_tile::index_t> strides;
    std::vector<ck_tile::index_t> dilations;
    std::vector<ck_tile::index_t> lpads;
    std::vector<ck_tile::index_t> rpads;

    const ck_tile::index_t num_dim_sp = fill_spatial_dimensions(filter_spatial_lengths,
                                                                image_spatial_lengths,
                                                                strides,
                                                                dilations,
                                                                lpads,
                                                                rpads,
                                                                arg_parser);

    ck_tile::conv::ConvParam conv_param{num_dim_sp,
                                        arg_parser.get_int("g"),
                                        arg_parser.get_int("n"),
                                        arg_parser.get_int("k"),
                                        arg_parser.get_int("c"),
                                        filter_spatial_lengths,
                                        image_spatial_lengths,
                                        strides,
                                        dilations,
                                        lpads,
                                        rpads};

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");

    const float floor = -100.f;
    const float ceil  = 100.f;

    const ck_tile::element_wise::MultiDAdd bias_op{};
    const ck_tile::element_wise::Clamp clamp_op{floor, ceil};
    const BiasAndClamp bias_clamp_op{bias_op, clamp_op};

    const auto in_g_n_c_wis_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    const auto wei_g_k_c_xs_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    const auto out_g_n_k_wos_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    ck_tile::HostTensor<InDataType> input(in_g_n_c_wis_desc);
    ck_tile::HostTensor<WeiDataType> weight(wei_g_k_c_xs_desc);
    ck_tile::HostTensor<OutDataType> output(out_g_n_k_wos_desc);
    ck_tile::HostTensor<OutDataType> bias(out_g_n_k_wos_desc);

    std::string bias_str = "";
    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(input);
        ck_tile::FillUniformDistribution<WeiDataType>{-5.f, 5.f}(weight);
        ck_tile::FillUniformDistribution<OutDataType>{-5.f, 5.f}(bias);
        bias_str = " (Uniform(-5,5))";
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<InDataType>{}(input);
        ck_tile::FillMonotonicSeq<WeiDataType>{}(weight);
        ck_tile::FillMonotonicSeq<OutDataType>{}(bias);
        bias_str = " (Monotonic)";
    }
    else if(init_method == 2)
    {
        ck_tile::FillUniformDistribution<InDataType>{1.f, 1.f}(input);
        ck_tile::FillUniformDistribution<WeiDataType>{1.f, 1.f}(weight);
        ck_tile::FillUniformDistribution<OutDataType>{1.f, 1.f}(bias);
        bias_str = " (Constant 1)";
    }
    else
    {
        input.SetZero();
        weight.SetZero();
        bias.SetZero();
    }

    ck_tile::DeviceMem input_dev_buf(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev_buf(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_dev_buf(output.get_element_space_size_in_bytes());
    ck_tile::DeviceMem bias_dev_buf(bias.get_element_space_size_in_bytes());

    input_dev_buf.ToDevice(input.data());
    weight_dev_buf.ToDevice(weight.data());
    output_dev_buf.SetZero();
    bias_dev_buf.ToDevice(bias.data());

    ck_tile::GroupedConvFwdHostArgs<BiasAndClamp> args(conv_param,
                                                       input_dev_buf.GetDeviceBuffer(),
                                                       weight_dev_buf.GetDeviceBuffer(),
                                                       {bias_dev_buf.GetDeviceBuffer()},
                                                       output_dev_buf.GetDeviceBuffer(),
                                                       kbatch,
                                                       bias_clamp_op);

    std::cout << "Run Grouped Conv Fwd kernel with bias" << bias_str << " and clamp (" << floor
              << ", " << ceil << ")." << std::endl;
    std::cout << "input: " << input.mDesc << std::endl;
    std::cout << "weight: " << weight.mDesc << std::endl;
    std::cout << "output: " << output.mDesc << std::endl;

    invoke_grouped_conv_fwd_bias_clamp<NDimSpatial,
                                       GemmWarpConfig,
                                       Invoker,
                                       InDataType,
                                       WeiDataType,
                                       AccDataType,
                                       OutDataType,
                                       InLayout,
                                       WeiLayout,
                                       OutLayout>(args, n_warmup, n_repeat);

    output_dev_buf.FromDevice(output.data());
    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<OutDataType> output_host_ref(out_g_n_k_wos_desc);
        output_host_ref.SetZero();

        auto bias_clamp_host = [floor,
                                ceil](float& y, const float& x, const OutDataType& element_bias) {
            float x_float = ck_tile::type_convert<float>(x);
            x_float += ck_tile::type_convert<float>(element_bias);
            if(x_float < floor)
                x_float = floor;
            else if(x_float > ceil)
                x_float = ceil;
            y = x_float;
        };
        auto bias_tuple = ck_tile::make_tuple(bias);
        ck_tile::reference_grouped_conv_fwd<NDimSpatial,
                                            InDataType,
                                            WeiDataType,
                                            OutDataType,
                                            decltype(bias_clamp_host)>(
            input,
            weight,
            output_host_ref,
            conv_param.conv_filter_strides_,
            conv_param.conv_filter_dilations_,
            conv_param.input_left_pads_,
            conv_param.input_right_pads_,
            bias_clamp_host,
            bias_tuple);
        const ck_tile::index_t GemmK = weight.get_element_size() / (conv_param.G_ * conv_param.K_);
        const float max_accumulated_value =
            *std::max_element(output_host_ref.mData.begin(), output_host_ref.mData.end());
        const auto rtol_atol =
            calculate_rtol_atol<InDataType, WeiDataType, AccDataType, OutDataType>(
                GemmK, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(output,
                                  output_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        // GPU verification for fused operation (Conv + Bias + Clamp) is complex
        // For now, we only support GPU verification for basic convolution operations
        // The bias+clamp fused variant can use CPU verification (-v=1) or no verification (-v=0)
        throw std::runtime_error("GPU verification not yet supported for fused operations! Use "
                                 "-v=1 for CPU verification.");
    }

    return pass;
}

template <typename Invoker,
          typename GemmWarpConfig,
          typename InPrecType,
          typename WeiPrecType = InPrecType,
          typename OutPrecType = InPrecType>
int run_grouped_conv_fwd_bias_clamp_example_prec_type(
    std::string in_layout, std::string wei_layout, std::string out_layout, int argc, char* argv[])
{
    using NWGC   = ck_tile::tensor_layout::convolution::NWGC;
    using NHWGC  = ck_tile::tensor_layout::convolution::NHWGC;
    using NDHWGC = ck_tile::tensor_layout::convolution::NDHWGC;

    using GKXC   = ck_tile::tensor_layout::convolution::GKXC;
    using GKYXC  = ck_tile::tensor_layout::convolution::GKYXC;
    using GKZYXC = ck_tile::tensor_layout::convolution::GKZYXC;

    using NWGK   = ck_tile::tensor_layout::convolution::NWGK;
    using NHWGK  = ck_tile::tensor_layout::convolution::NHWGK;
    using NDHWGK = ck_tile::tensor_layout::convolution::NDHWGK;

    if(in_layout == "NWGC" && wei_layout == "GKXC" && out_layout == "NWGK")
    {
        return run_grouped_conv_fwd_bias_clamp_example_with_layouts<ck_tile::number<1>{},
                                                                    GemmWarpConfig,
                                                                    Invoker,
                                                                    InPrecType,
                                                                    WeiPrecType,
                                                                    OutPrecType>(
            argc, argv, NWGC{}, GKXC{}, NWGK{});
    }
    else if(in_layout == "NHWGC" && wei_layout == "GKYXC" && out_layout == "NHWGK")
    {
        return run_grouped_conv_fwd_bias_clamp_example_with_layouts<ck_tile::number<2>{},
                                                                    GemmWarpConfig,
                                                                    Invoker,
                                                                    InPrecType,
                                                                    WeiPrecType,
                                                                    OutPrecType>(
            argc, argv, NHWGC{}, GKYXC{}, NHWGK{});
    }
    else if(in_layout == "NDHWGC" && wei_layout == "GKZYXC" && out_layout == "NDHWGK")
    {
        return run_grouped_conv_fwd_bias_clamp_example_with_layouts<ck_tile::number<3>{},
                                                                    GemmWarpConfig,
                                                                    Invoker,
                                                                    InPrecType,
                                                                    WeiPrecType,
                                                                    OutPrecType>(
            argc, argv, NDHWGC{}, GKZYXC{}, NDHWGK{});
    }
    else
    {
        throw std::runtime_error("Unsupported memory layout!");
    }
}
