// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_grouped_conv_fwd_impl.hpp"
#include "profiler_operation_registry.hpp"

namespace {

enum struct ConvLayout
{
    GNHWC_GKYXC_GNHWK, // 0
    NHWGC_GKYXC_NHWGK, // 1
    NGCHW_GKYXC_NGKHW, // 2
    NGCHW_GKCYX_NGKHW, // 3
};

enum struct ConvDataType
{
    F32_F32_F32,      // 0
    F16_F16_F16,      // 1
    BF16_BF16_BF16,   // 2
    INT8_INT8_INT8,   // 3
    F8_F8_F8,         // 4
    BF8_BF8_F8,       // 5
    F8_BF8_F8,        // 6
    BF8_F8_F8,        // 7
    F32_F32_F32_TF32, // 8
};

enum struct IndexType
{
    INDEX_T,      // 0
    LONG_INDEX_T, // 1
};

#define OP_NAME "grouped_conv_fwd"
#define OP_DESC "Grouped Convolution Forward"

static void print_helper_msg()
{
    std::cout
        // clang-format off
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: data type (0: Input fp32, Weight fp32, Output fp32\n"
        << "                 1: Input fp16, Weight fp16, Output fp16\n"
        << "                 2: Input bf16, Weight bf16, Output bf16\n"
        << "                 3: Input int8, Weight int8, Output int8\n"
        << "                 4: Input fp8, Weight fp8, Output fp8\n"
        << "                 5: Input bf8, Weight bf8, Output fp8\n"
        << "                 6: Input fp8, Weight bf8, Output fp8\n"
        << "                 7: Input bf8, Weight fp8, Output fp8\n"
        << "                 8: Input fp32, Weight fp32, Output fp32, Compute tf32)\n"
        << "arg3: tensor layout (0: Input[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Output[G, N, Ho, Wo, K]\n"
        << "                     1: Input[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Output[N, Ho, Wo, G, K]\n"
        << "                     2: Input[N, G, C, Hi, Wi], Weight[G, K, Y, X, C], Output[N, "
        "G, K, Ho, Wo]\n"
        << "                     3: Input[N, G, C, Hi, Wi], Weight[G, K, C, Y, X], Output[N, "
        "G, K, Ho, Wo])\n"
        << "arg4: indexing data type (0: 32-bit, 1: 64-bit)\n"
        << "arg5: verification (0: no, 1: yes)\n"
        << "arg6: initialization (0: no init, 1: integer value, 2: decimal value)\n"
        << "arg7: print tensor value (0: no; 1: yes)\n"
        << "arg8: time kernel (0: no, 1: yes)\n"
        << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
    // clang-format on
}

} // namespace

int profile_grouped_conv_fwd(int argc, char* argv[])
{
    // 8 for control, 1 for num_dim_spatial
    if(argc < 10)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<ConvDataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<ConvLayout>(std::stoi(argv[3]));
    const auto index_type      = static_cast<IndexType>(std::stoi(argv[4]));
    const bool do_verification = std::stoi(argv[5]);
    const int init_method      = std::stoi(argv[6]);
    const bool do_log          = std::stoi(argv[7]);
    const bool time_kernel     = std::stoi(argv[8]);
    const int num_dim_spatial  = std::stoi(argv[9]);

    // 9 for control, 1 for num_dim_spatial, 4 for G/N/K/C, and 6 * num_dim_spatial
    if(argc != 9 + 1 + 4 + 6 * num_dim_spatial)
    {
        print_helper_msg();
        return 1;
    }

    const auto params = ck::utils::conv::parse_conv_param(num_dim_spatial, 10, argv);

    using F32  = float;
    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using INT8 = int8_t;
    using F8   = ck::f8_t;
    using BF8  = ck::bf8_t;
    using TF32 = ck::tf32_t;

    //
    using GNWC   = ck::tensor_layout::convolution::GNWC;
    using GNHWC  = ck::tensor_layout::convolution::GNHWC;
    using GNDHWC = ck::tensor_layout::convolution::GNDHWC;

    using GKXC   = ck::tensor_layout::convolution::GKXC;
    using GKYXC  = ck::tensor_layout::convolution::GKYXC;
    using GKZYXC = ck::tensor_layout::convolution::GKZYXC;

    // using GKCX   = ck::tensor_layout::convolution::GKXC;
    using GKCYX  = ck::tensor_layout::convolution::GKCYX;
    using GKCZYX = ck::tensor_layout::convolution::GKCZYX;

    using GNWK   = ck::tensor_layout::convolution::GNWK;
    using GNHWK  = ck::tensor_layout::convolution::GNHWK;
    using GNDHWK = ck::tensor_layout::convolution::GNDHWK;

    //
    using NGCHW  = ck::tensor_layout::convolution::NGCHW;
    using NGCDHW = ck::tensor_layout::convolution::NGCDHW;

    using NGKHW  = ck::tensor_layout::convolution::NGKHW;
    using NGKDHW = ck::tensor_layout::convolution::NGKDHW;

    //
    using NWGC   = ck::tensor_layout::convolution::NWGC;
    using NHWGC  = ck::tensor_layout::convolution::NHWGC;
    using NDHWGC = ck::tensor_layout::convolution::NDHWGC;

    using NWGK   = ck::tensor_layout::convolution::NWGK;
    using NHWGK  = ck::tensor_layout::convolution::NHWGK;
    using NDHWGK = ck::tensor_layout::convolution::NDHWGK;

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
                       auto a_compute_type,
                       auto b_compute_type) {
        constexpr ck::index_t NDimSpatial = num_dim_spatial_tmp.value;

        using InLayout  = decltype(in_layout);
        using WeiLayout = decltype(wei_layout);
        using OutLayout = decltype(out_layout);

        using InDataType  = decltype(in_type);
        using WeiDataType = decltype(wei_type);
        using OutDataType = decltype(out_type);

        using AComputeType = decltype(a_compute_type);
        using BComputeType = decltype(b_compute_type);

        if(index_type == IndexType::INDEX_T)
        {
            bool pass = ck::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                                    InLayout,
                                                                    WeiLayout,
                                                                    OutLayout,
                                                                    InDataType,
                                                                    WeiDataType,
                                                                    OutDataType,
                                                                    AComputeType,
                                                                    BComputeType,
                                                                    ck::index_t>(
                do_verification, init_method, do_log, time_kernel, params);

            return pass ? 0 : 1;
        }
        else if(index_type == IndexType::LONG_INDEX_T)
        {
            bool pass = ck::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                                    InLayout,
                                                                    WeiLayout,
                                                                    OutLayout,
                                                                    InDataType,
                                                                    WeiDataType,
                                                                    OutDataType,
                                                                    AComputeType,
                                                                    BComputeType,
                                                                    ck::long_index_t>(
                do_verification, init_method, do_log, time_kernel, params);

            return pass ? 0 : 1;
        }
        else
        {
            std::cout << "this indexing data type is not implemented" << std::endl;
            return 1;
        }
    };

    // GNHWC_GKYXC_GNHWK
    if(num_dim_spatial == 1 && layout == ConvLayout::GNHWC_GKYXC_GNHWK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::INT8_INT8_INT8)
        {
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, INT8{}, INT8{}, INT8{}, INT8{}, INT8{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I1, GNWC{}, GKXC{}, GNWK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    else if(num_dim_spatial == 2 && layout == ConvLayout::GNHWC_GKYXC_GNHWK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::INT8_INT8_INT8)
        {
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, INT8{}, INT8{}, INT8{}, INT8{}, INT8{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I2, GNHWC{}, GKYXC{}, GNHWK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    else if(num_dim_spatial == 3 && layout == ConvLayout::GNHWC_GKYXC_GNHWK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(
                I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::INT8_INT8_INT8)
        {
            return profile(
                I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, INT8{}, INT8{}, INT8{}, INT8{}, INT8{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I3, GNDHWC{}, GKZYXC{}, GNDHWK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    // NHWGC_GKYXC_NHWGK
    else if(num_dim_spatial == 1 && layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I1, NWGC{}, GKXC{}, NWGK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I1, NWGC{}, GKXC{}, NWGK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I1, NWGC{}, GKXC{}, NWGK{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::INT8_INT8_INT8)
        {
            return profile(I1, NWGC{}, GKXC{}, NWGK{}, INT8{}, INT8{}, INT8{}, INT8{}, INT8{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I1, NWGC{}, GKXC{}, NWGK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    else if(num_dim_spatial == 2 && layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::INT8_INT8_INT8)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, INT8{}, INT8{}, INT8{}, INT8{}, INT8{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I2, NHWGC{}, GKYXC{}, NHWGK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    else if(num_dim_spatial == 2 && layout == ConvLayout::NGCHW_GKYXC_NGKHW)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I2, NGCHW{}, GKYXC{}, NGKHW{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, NGCHW{}, GKYXC{}, NGKHW{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I2, NGCHW{}, GKYXC{}, NGKHW{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I2, NGCHW{}, GKYXC{}, NGKHW{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    else if(num_dim_spatial == 2 && layout == ConvLayout::NGCHW_GKCYX_NGKHW)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I2, NGCHW{}, GKCYX{}, NGKHW{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I2, NGCHW{}, GKCYX{}, NGKHW{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(I2, NGCHW{}, GKCYX{}, NGKHW{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I2, NGCHW{}, GKCYX{}, NGKHW{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    else if(num_dim_spatial == 3 && layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(
                I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::INT8_INT8_INT8)
        {
            return profile(
                I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, INT8{}, INT8{}, INT8{}, INT8{}, INT8{});
        }
        else if(data_type == ConvDataType::F8_F8_F8)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F8{}, F8{}, F8{}, F8{}, F8{});
        }
        else if(data_type == ConvDataType::BF8_BF8_F8)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, BF8{}, BF8{}, F8{}, BF8{}, BF8{});
        }
        else if(data_type == ConvDataType::F8_BF8_F8)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F8{}, BF8{}, F8{}, F8{}, BF8{});
        }
        else if(data_type == ConvDataType::BF8_F8_F8)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, BF8{}, F8{}, F8{}, BF8{}, F8{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }
    // NGCDHW_GKCZYX_NGKDHW
    else if(num_dim_spatial == 3 && layout == ConvLayout::NGCHW_GKCYX_NGKHW)
    {
        if(data_type == ConvDataType::F32_F32_F32)
        {
            return profile(I3, NGCDHW{}, GKCZYX{}, NGKDHW{}, F32{}, F32{}, F32{}, F32{}, F32{});
        }
        else if(data_type == ConvDataType::F16_F16_F16)
        {
            return profile(I3, NGCDHW{}, GKCZYX{}, NGKDHW{}, F16{}, F16{}, F16{}, F16{}, F16{});
        }
        else if(data_type == ConvDataType::BF16_BF16_BF16)
        {
            return profile(
                I3, NGCDHW{}, GKCZYX{}, NGKDHW{}, BF16{}, BF16{}, BF16{}, BF16{}, BF16{});
        }
        else if(data_type == ConvDataType::F32_F32_F32_TF32)
        {
            return profile(I3, NGCDHW{}, GKCZYX{}, NGKDHW{}, F32{}, F32{}, F32{}, TF32{}, TF32{});
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_conv_fwd);
