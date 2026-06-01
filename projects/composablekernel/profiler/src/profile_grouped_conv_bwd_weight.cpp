// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "profiler/profile_grouped_conv_bwd_weight_impl.hpp"
#include "profiler_operation_registry.hpp"

namespace {

enum struct ConvLayout
{
    GNCHW_GKCYX_GNKHW, // 0
    GNHWC_GKYXC_GNHWK, // 1
    NHWGC_GKYXC_NHWGK, // 2
    NGCHW_GKYXC_NGKHW, // 3
    NGCHW_GKCYX_NGKHW, // 4
};

enum struct ConvDataType
{
    F32_F32_F32,        // 0
    F16_F16_F16,        // 1
    BF16_F32_BF16,      // 2
    F16_F16_F16_BF8_F8, // 3
    I8_I8_I8,           // 4
    BF16_BF16_BF16,     // 5
    F32_F32_F32_TF32,   // 6
};

#define OP_NAME "grouped_conv_bwd_weight"
#define OP_DESC "Grouped Convolution Backward Weight"

static void print_helper_msg()
{
    std::cout << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
              << "arg2: data type (0: Input fp32, Weight fp32, Output fp32\n"
              << "                 1: Input fp16, Weight fp16, Output fp16\n"
              << "                 2: Input bf16, Weight fp32, Output bf16\n"
              << "                 3: Input fp16, Weight fp16, Output fp16, Gemm bf8@fp8\n"
              << "                 4: Input int8, Weight int8, Output int8\n"
              << "                 5: Input bf16, Weight bf16, Output bf16\n"
              << "                 6: Input fp32, Weight fp32, Output fp32, Compute tf32)\n"
              << "arg3: tensor layout (0: Input[G, N, C, Hi, Wi], Weight[G, K, C, Y, X], Output[G, "
                 "N, K, Ho, Wo]\n"
              << "                     1: Input[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Output[G, "
                 "N, Ho, Wo, K]\n"
              << "                     2: Input[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Output[N, "
                 "Ho, Wo, G, K]\n"
              << "                     3: Input[N, G, C, Hi, Wi], Weight[G, K, Y, X, C], Output[N, "
                 "G, K, Ho, Wo]\n"
              << "                     4: Input[N, G, C, Hi, Wi], Weight[G, K, C, Y, X], Output[N, "
                 "G, K, Ho, Wo]\n"
              << "arg4: verification (0: no, 1: yes)\n"
              << "arg5: initialization (0: no init, 1: integer value, 2: decimal value)\n"
              << "arg6: print tensor value (0: no; 1: yes)\n"
              << "arg7: time kernel (0: no, 1: yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg()
              << " SplitK (-1 for internally computed split-K value, positive value to set k "
                 "batches explicitly, or 'all' to test all internal split-K values)\n"
              << std::endl;
}

} // namespace

int profile_grouped_conv_bwd_weight(int argc, char* argv[])
{
    // 8 for control, 1 for num_dim_spatial
    if(argc < 9)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<ConvDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<ConvLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);
    const int num_dim_spatial  = std::stoi(argv[8]);

    // 8 for control, 1 for num_dim_spatial, 4 for G/N/K/C, and 6 * num_dim_spatial, 1 for split-K
    if(argc != 8 + 1 + 4 + 6 * num_dim_spatial + 1)
    {
        print_helper_msg();
        return 1;
    }

    const auto params = ck::utils::conv::parse_conv_param(num_dim_spatial, 9, argv);

    const auto& split_k = std::string(argv[8 + 1 + 4 + 6 * num_dim_spatial]);

    using F32  = float;
    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using F8   = ck::f8_t;
    using BF8  = ck::bf8_t;
    using TF32 = ck::tf32_t;

    using namespace ck::tensor_layout::convolution;

    constexpr auto I1 = ck::Number<1>{};
    constexpr auto I2 = ck::Number<2>{};
    constexpr auto I3 = ck::Number<3>{};

    auto profile = [&](auto num_dim_spatial_tmp,
                       auto in_layout,
                       auto wei_layout,
                       auto out_layout,
                       auto in_type,
                       auto wei_type,
                       auto out_type,
                       auto compute_type_a,
                       auto compute_type_b) {
        constexpr ck::index_t NDimSpatial = num_dim_spatial_tmp.value;

        using InLayout  = decltype(in_layout);
        using WeiLayout = decltype(wei_layout);
        using OutLayout = decltype(out_layout);

        using InDataType  = decltype(in_type);
        using WeiDataType = decltype(wei_type);
        using OutDataType = decltype(out_type);

        using ComputeTypeA = decltype(compute_type_a);
        using ComputeTypeB = decltype(compute_type_b);

        bool pass = ck::profiler::profile_grouped_conv_bwd_weight_impl<NDimSpatial,
                                                                       InLayout,
                                                                       WeiLayout,
                                                                       OutLayout,
                                                                       InDataType,
                                                                       WeiDataType,
                                                                       OutDataType,
                                                                       ComputeTypeA,
                                                                       ComputeTypeB>(
            do_verification, init_method, do_log, time_kernel, params, split_k);

        return pass ? 0 : 1;
    };

    if(num_dim_spatial == 1 && layout == ConvLayout::GNHWC_GKYXC_GNHWK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        if(data_type == ConvDataType::BF16_F32_BF16)
        {
            // fp32 atomic add is used for weight tensor in bf16 kernel
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, BF16{}, F32{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    if(num_dim_spatial == 2 && layout == ConvLayout::GNHWC_GKYXC_GNHWK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        if(data_type == ConvDataType::BF16_F32_BF16)
        {
            // fp32 atomic add is used for weight tensor in bf16 kernel
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, BF16{}, F32{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    if(num_dim_spatial == 2 && layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        if(data_type == ConvDataType::BF16_F32_BF16)
        {
            // fp32 atomic add is used for weight tensor in bf16 kernel
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, BF16{}, F32{}, BF16{}, BF16{}, BF16{});
        }
        if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    else if(num_dim_spatial == 2 && layout == ConvLayout::NGCHW_GKYXC_NGKHW)
    {
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, NGCHW{}, GKYXC{}, NGKHW{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            // fp32 atomic add is used for weight tensor in bf16 kernel
            return profile(I2, NGCHW{}, GKYXC{}, NGKHW{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
    }
    else if(num_dim_spatial == 2 && layout == ConvLayout::NGCHW_GKCYX_NGKHW)
    {
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, NGCHW{}, GKCYX{}, NGKHW{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I2, NGCHW{}, GKCYX{}, NGKHW{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I2, NGCHW{}, GKCYX{}, NGKHW{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I2, NGCHW{}, GKCYX{}, NGKHW{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    if(num_dim_spatial == 3 && layout == ConvLayout::GNHWC_GKYXC_GNHWK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        if(data_type == ConvDataType::BF16_F32_BF16)
        {
            // fp32 atomic add is used for weight tensor in bf16 kernel
            return profile(I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, BF16{}, F32{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::I8_I8_I8)
        {
            return profile(
                I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, int8_t{}, int8_t{}, int8_t{}, int8_t{}, int8_t{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    if(num_dim_spatial == 3 && layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        if(data_type == ConvDataType::BF16_F32_BF16)
        {
            // fp32 atomic add is used for weight tensor in bf16 kernel
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, BF16{}, F32{}, BF16{}, BF16{}, BF16{});
        }
        if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(
                I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        if(data_type == ConvDataType::F16_F16_F16_BF8_F8)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F16{}, F16{}, F16{}, BF8{}, F8{});
        }
        else if(data_type == ConvDataType::I8_I8_I8)
        {
            return profile(
                I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, int8_t{}, int8_t{}, int8_t{}, int8_t{}, int8_t{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    else if(num_dim_spatial == 3 && layout == ConvLayout::NGCHW_GKYXC_NGKHW)
    {
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I3, NGCDHW{}, GKZYXC{}, NGKDHW{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(
                I3, NGCDHW{}, GKZYXC{}, NGKDHW{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
    }
    else if(num_dim_spatial == 3 && layout == ConvLayout::NGCHW_GKCYX_NGKHW)
    {
        if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I3, NGCDHW{}, GKCZYX{}, NGKDHW{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(
                I3, NGCDHW{}, GKCZYX{}, NGKDHW{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I3, NGCDHW{}, GKCZYX{}, NGKDHW{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I3, NGCDHW{}, GKCZYX{}, NGKDHW{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_conv_bwd_weight);
