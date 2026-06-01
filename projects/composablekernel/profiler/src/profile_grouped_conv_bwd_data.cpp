// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_grouped_conv_bwd_data_impl.hpp"
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
    F32_F32_F32_TF32, // 3
};

#define OP_NAME "grouped_conv_bwd_data"
#define OP_DESC "Grouped Convolution Backward Data"

static void print_helper_msg()
{
    std::cout
        // clang-format off
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: data type (0: Output fp32, Weight fp32, Input fp32\n"
        << "                 1: Output fp16, Weight fp16, Input fp16\n"
        << "                 2: Output bf16, Weight bf16, Input bf16\n"
        << "                 3: Output fp32, Weight fp32, Input fp32, Compute tf32)\n"
        << "arg3: tensor layout (0: Output[G, N, Ho, Wo, C], Weight[G, K, Y, X, C], Input[G, N, Hi, Wi, K]\n"
        << "                     1: Output[N, Ho, Wo, G, C], Weight[G, K, Y, X, C], Input[N, Hi, Wi, G, K])\n"
        << "                     2: Output[N, G, C, Ho, Wo], Weight[G, K, Y, X, C], Input[N, G, K, Hi, Wi])\n"
        << "                     3: Output[N, G, C, Ho, Wo], Weight[G, K, C, Y, X], Input[N, G, K, Hi, Wi])\n"
        << "arg4: verification (0: no, 1: yes)\n"
        << "arg5: initialization (0: no init, 1: integer value, 2: decimal value)\n"
        << "arg6: print tensor value (0: no; 1: yes)\n"
        << "arg7: time kernel (0: no, 1: yes)\n"
        << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl
        << "arg8: split-K (0: internally computed split-K value; 1, 2, 4, 8, 16, 32, 64, 128: set k batches explicitly)\n";
    // clang-format on
}

} // namespace

int profile_grouped_conv_bwd_data(int argc, char* argv[])
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

    ck::index_t split_k = std::stoi(argv[8 + 1 + 4 + 6 * num_dim_spatial]);

    using F32  = float;
    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using TF32 = ck::tf32_t;

    using namespace ck::tensor_layout::convolution;

    constexpr auto I2 = ck::Number<2>{};
    constexpr auto I3 = ck::Number<3>{};

    auto profile = [&](auto num_dim_spatial_tmp,
                       auto out_layout,
                       auto wei_layout,
                       auto in_layout,
                       auto wei_type,
                       auto out_type,
                       auto in_type,
                       auto compute_type) {
        constexpr ck::index_t NDimSpatial = num_dim_spatial_tmp.value;

        using OutLayout = decltype(out_layout);
        using WeiLayout = decltype(wei_layout);
        using InLayout  = decltype(in_layout);

        using OutDataType     = decltype(out_type);
        using WeiDataType     = decltype(wei_type);
        using InDataType      = decltype(in_type);
        using ComputeDataType = decltype(compute_type);

        bool pass = ck::profiler::profile_grouped_conv_bwd_data_impl<NDimSpatial,
                                                                     OutLayout,
                                                                     WeiLayout,
                                                                     InLayout,
                                                                     OutDataType,
                                                                     WeiDataType,
                                                                     InDataType,
                                                                     ComputeDataType>(
            do_verification, init_method, do_log, time_kernel, params, split_k);

        return pass ? 0 : 1;
    };

    if(num_dim_spatial == 2)
    {
        if(layout == ConvLayout::GNHWC_GKYXC_GNHWK)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                return profile(I2, GNHWK{}, GKYXC{}, GNHWC{}, F32{}, F32{}, F32{}, F32{});
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                return profile(I2, GNHWK{}, GKYXC{}, GNHWC{}, F16{}, F16{}, F16{}, F16{});
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                return profile(I2, GNHWK{}, GKYXC{}, GNHWC{}, BF16{}, BF16{}, BF16{}, BF16{});
            }
            else if(data_type == ConvDataType::F32_F32_F32_TF32)
            {
                return profile(I2, GNHWK{}, GKYXC{}, GNHWC{}, F32{}, F32{}, F32{}, TF32{});
            }
        }
        else if(layout == ConvLayout::NHWGC_GKYXC_NHWGK)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                return profile(I2, NHWGK{}, GKYXC{}, NHWGC{}, F32{}, F32{}, F32{}, F32{});
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                return profile(I2, NHWGK{}, GKYXC{}, NHWGC{}, F16{}, F16{}, F16{}, F16{});
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                return profile(I2, NHWGK{}, GKYXC{}, NHWGC{}, BF16{}, BF16{}, BF16{}, BF16{});
            }
            else if(data_type == ConvDataType::F32_F32_F32_TF32)
            {
                return profile(I2, NHWGK{}, GKYXC{}, NHWGC{}, F32{}, F32{}, F32{}, TF32{});
            }
        }
        else if(layout == ConvLayout::NGCHW_GKYXC_NGKHW)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                return profile(I2, NGKHW{}, GKYXC{}, NGCHW{}, F32{}, F32{}, F32{}, F32{});
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                return profile(I2, NGKHW{}, GKYXC{}, NGCHW{}, F16{}, F16{}, F16{}, F16{});
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                return profile(I2, NGKHW{}, GKYXC{}, NGCHW{}, BF16{}, BF16{}, BF16{}, BF16{});
            }
            else if(data_type == ConvDataType::F32_F32_F32_TF32)
            {
                return profile(I2, NGKHW{}, GKYXC{}, NGCHW{}, F32{}, F32{}, F32{}, TF32{});
            }
        }
        else if(layout == ConvLayout::NGCHW_GKCYX_NGKHW)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                return profile(I2, NGKHW{}, GKCYX{}, NGCHW{}, F32{}, F32{}, F32{}, F32{});
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                return profile(I2, NGKHW{}, GKCYX{}, NGCHW{}, F16{}, F16{}, F16{}, F16{});
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                return profile(I2, NGKHW{}, GKCYX{}, NGCHW{}, BF16{}, BF16{}, BF16{}, BF16{});
            }
            else if(data_type == ConvDataType::F32_F32_F32_TF32)
            {
                return profile(I2, NGKHW{}, GKCYX{}, NGCHW{}, F32{}, F32{}, F32{}, TF32{});
            }
        }
    }
    else if(num_dim_spatial == 3)
    {
        if(layout == ConvLayout::GNHWC_GKYXC_GNHWK)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                return profile(I3, GNDHWK{}, GKZYXC{}, GNDHWC{}, F32{}, F32{}, F32{}, F32{});
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                return profile(I3, GNDHWK{}, GKZYXC{}, GNDHWC{}, F16{}, F16{}, F16{}, F16{});
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                return profile(I3, GNDHWK{}, GKZYXC{}, GNDHWC{}, BF16{}, BF16{}, BF16{}, BF16{});
            }
            else if(data_type == ConvDataType::F32_F32_F32_TF32)
            {
                return profile(I3, GNDHWK{}, GKZYXC{}, GNDHWC{}, F32{}, F32{}, F32{}, TF32{});
            }
        }
        else if(layout == ConvLayout::NHWGC_GKYXC_NHWGK)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                return profile(I3, NDHWGK{}, GKZYXC{}, NDHWGC{}, F32{}, F32{}, F32{}, F32{});
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                return profile(I3, NDHWGK{}, GKZYXC{}, NDHWGC{}, F16{}, F16{}, F16{}, F16{});
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                return profile(I3, NDHWGK{}, GKZYXC{}, NDHWGC{}, BF16{}, BF16{}, BF16{}, BF16{});
            }
            else if(data_type == ConvDataType::F32_F32_F32_TF32)
            {
                return profile(I3, NDHWGK{}, GKZYXC{}, NDHWGC{}, F32{}, F32{}, F32{}, TF32{});
            }
        }
        else if(layout == ConvLayout::NGCHW_GKYXC_NGKHW)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                return profile(I3, NGKDHW{}, GKZYXC{}, NGCDHW{}, F32{}, F32{}, F32{}, F32{});
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                return profile(I3, NGKDHW{}, GKZYXC{}, NGCDHW{}, F16{}, F16{}, F16{}, F16{});
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                return profile(I3, NGKDHW{}, GKZYXC{}, NGCDHW{}, BF16{}, BF16{}, BF16{}, BF16{});
            }
            else if(data_type == ConvDataType::F32_F32_F32_TF32)
            {
                return profile(I3, NGKDHW{}, GKZYXC{}, NGCDHW{}, F32{}, F32{}, F32{}, TF32{});
            }
        }
        else if(layout == ConvLayout::NGCHW_GKYXC_NGKHW)
        {
            if(data_type == ConvDataType::F32_F32_F32)
            {
                return profile(I3, NGKDHW{}, GKCZYX{}, NGCDHW{}, F32{}, F32{}, F32{}, F32{});
            }
            else if(data_type == ConvDataType::F16_F16_F16)
            {
                return profile(I3, NGKDHW{}, GKCZYX{}, NGCDHW{}, F16{}, F16{}, F16{}, F16{});
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                return profile(I3, NGKDHW{}, GKCZYX{}, NGCDHW{}, BF16{}, BF16{}, BF16{}, BF16{});
            }
            else if(data_type == ConvDataType::F32_F32_F32_TF32)
            {
                return profile(I3, NGKDHW{}, GKCZYX{}, NGCDHW{}, F32{}, F32{}, F32{}, TF32{});
            }
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_grouped_conv_bwd_data);
