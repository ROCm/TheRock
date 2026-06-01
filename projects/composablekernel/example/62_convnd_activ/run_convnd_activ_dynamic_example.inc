// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

void print_helper_msg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

template <typename OutElementOp>
bool run_convnd_example(int argc, char* argv[], const OutElementOp& out_element_op)
{
    print_helper_msg();

    bool do_verification = true;
    // Use floats for SoftRelu by default to avoid overflow after e^x.
    int init_method =
        std::is_same_v<OutElementOp, ck::tensor_operation::element_wise::SoftRelu> ? 2 : 1;
    bool time_kernel = false;

    // Following shapes are selected to avoid overflow. Expect inf in case of
    // size increase for some elementwise ops.
    ck::utils::conv::ConvParam conv_param{
        3, 2, 16, 128, 8, {3, 3, 3}, {17, 17, 17}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};

    if(argc == 1)
    {
        // use default
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        do_verification                   = std::stoi(argv[1]);
        init_method                       = std::stoi(argv[2]);
        time_kernel                       = std::stoi(argv[3]);
        const ck::index_t num_dim_spatial = std::stoi(argv[4]);

        conv_param = ck::utils::conv::parse_conv_param(num_dim_spatial, 5, argv);
    }

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};

    const auto run = [&]() {
        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        return run_grouped_conv<NDimSpatial,
                                InDataType,
                                WeiDataType,
                                OutDataType,
                                InElementOp,
                                WeiElementOp,
                                OutElementOp,
                                DeviceGroupedConvNDActivInstance>(do_verification,
                                                                  init_method,
                                                                  time_kernel,
                                                                  conv_param,
                                                                  in_g_n_c_wis_desc,
                                                                  wei_g_k_c_xs_desc,
                                                                  out_g_n_k_wos_desc,
                                                                  in_element_op,
                                                                  wei_element_op,
                                                                  out_element_op);
    };

    if(conv_param.num_dim_spatial_ == 3)
    {
        return run();
    }

    return false;
}
