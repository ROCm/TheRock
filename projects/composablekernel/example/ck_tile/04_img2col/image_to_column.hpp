// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/image_to_column.hpp"
#include <string>

#define DefaultConvParams                                                    \
    ck_tile::conv::ConvParam                                                 \
    {                                                                        \
        2, 2, 32, 32, 32, {4, 4}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, { 0, 0 } \
    }

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

inline void print_help_msg()
{
    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck_tile::conv::get_conv_param_parser_helper_msg() << std::endl;
}

inline bool parse_cmd_args(int argc,
                           char* argv[],
                           ExecutionConfig& config,
                           ck_tile::conv::ConvParam& conv_params)
{
    constexpr int num_execution_config_args =
        3; // arguments for do_verification, init_method, time_kernel
    constexpr int num_conv_param_leading_args = 5; // arguments for num_dim_spatial_, G_, N_, K_, C_

    constexpr int threshold_to_catch_partial_args = 1 + num_execution_config_args;
    constexpr int threshold_to_catch_all_args =
        threshold_to_catch_partial_args + num_conv_param_leading_args;

    if(argc == 1)
    {
        // use default
        config = ExecutionConfig{};
    }
    // catch only ExecutionConfig arguments
    else if(argc == threshold_to_catch_partial_args)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    // catch both ExecutionConfig & ConvParam arguments
    else if(threshold_to_catch_all_args < argc && ((argc - threshold_to_catch_all_args) % 3 == 0))
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        const ck_tile::index_t num_dim_spatial = std::stoi(argv[4]);
        conv_params =
            ck_tile::conv::parse_conv_param(num_dim_spatial, threshold_to_catch_partial_args, argv);
    }
    else
    {
        print_help_msg();
        return false;
    }

    return true;
}

struct image_to_column_traits
{
    std::string data_type;
};

template <ck_tile::index_t NDimSpatial>
struct image_to_column_args
{
    const void* p_in;
    void* p_out;
    const ck_tile::long_index_t G;
    const ck_tile::long_index_t N;
    const ck_tile::long_index_t C;
    const ck_tile::array<ck_tile::long_index_t, NDimSpatial> input_spatial_lengths;
    const ck_tile::array<ck_tile::long_index_t, NDimSpatial> filter_spatial_lengths;
    const ck_tile::array<ck_tile::long_index_t, NDimSpatial> output_spatial_lengths;
    const ck_tile::array<ck_tile::long_index_t, NDimSpatial + 3> image_g_n_c_wis_strides;
    const ck_tile::array<ck_tile::long_index_t, 3> gemm_g_m_k_strides;
    const ck_tile::array<ck_tile::long_index_t, NDimSpatial> conv_filter_strides;
    const ck_tile::array<ck_tile::long_index_t, NDimSpatial> conv_filter_dilations;
    const ck_tile::array<ck_tile::long_index_t, NDimSpatial> input_left_pads;
    const ck_tile::array<ck_tile::long_index_t, NDimSpatial> input_right_pads;
};

// host API
template <ck_tile::index_t NDimSpatial>
float image_to_column(const image_to_column_traits&,
                      const image_to_column_args<NDimSpatial>&,
                      const ck_tile::stream_config&);
