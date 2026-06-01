// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "profiler/profile_grouped_conv_fwd_outelementop_impl.hpp"

#include "ck/utility/data_type.hpp"
#include "profiler_operation_registry.hpp"

#include <iostream>

enum struct ConvLayout
{
    GNHWC_GKYXC_GNHWK = 0,
    NHWGC_GKYXC_NHWGK = 1
};

enum struct OutElementOp
{
    ConvScale    = 0,
    ConvInvScale = 1,
    Scale        = 2
};

enum struct ConvDataType
{
    F8_F8_F8       = 0,
    BF8_BF8_F8     = 1,
    F8_BF8_F8      = 2,
    BF8_F8_F8      = 3,
    F16_F16_F16    = 4,
    BF16_BF16_BF16 = 5,
    I8_I8_I8       = 6
};

#define OP_NAME "grouped_conv_fwd_outelementop"
#define OP_DESC "Grouped Convolution Forward+Elementwise Operation"

static void print_helper_msg()
{
    // clang-format off
    std::cout
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: data type (0: Input fp8, Weight fp8, Output fp8\n"
        << "                 1: Input bf8, Weight bf8, Output fp8\n"
        << "                 2: Input fp8, Weight bf8, Output fp8\n"
        << "                 3: Input bf8, Weight fp8, Output fp8)\n"
        << "                 4: Input f16, Weight f16, Output f16)\n"
        << "                 5: Input bf16, Weight bf16, Output bf16)\n"
        << "                 6: Input i8, Weight i8, Output i8)\n"
        << "arg3: element-wise operation (0: ConvScale\n"
        << "                              1: ConvInvScale\n"
        << "                              2: Scale\n"
        << "arg4: tensor layout (0: Input[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Output[G, N, Ho, Wo, K]\n"
        << "                     1: Input[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Output[N, Ho, Wo, G, K])\n"
        << "arg5: verification (0: no, 1: yes)\n"
        << "arg6: initialization (0: no init, 1: integer value, 2: decimal value)\n"
        << "arg7: print tensor value (0: no; 1: yes)\n"
        << "arg8: time kernel (0: no, 1: yes)\n"
        << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
    // clang-format on
}

int grouped_conv_fwd_outelementop(int argc, char* argv[])
{

    // 9 total, 1 for num_dim_spatial
    if(argc < 10)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<ConvDataType>(std::stoi(argv[2]));
    const auto op              = static_cast<OutElementOp>(std::stoi(argv[3]));
    const auto layout          = static_cast<ConvLayout>(std::stoi(argv[4]));
    const bool do_verification = std::stoi(argv[5]);
    const int init_method      = std::stoi(argv[6]);
    const bool do_log          = std::stoi(argv[7]);
    const bool time_kernel     = std::stoi(argv[8]);
    const int num_dim_spatial  = std::stoi(argv[9]);

    // 8 for control, 1 for num_dim_spatial, 4 for G/N/K/C, and 6 * num_dim_spatial + 1 for argv[0]
    if(argc != 8 + 1 + 4 + 6 * num_dim_spatial + 1)
    {
        print_helper_msg();
        return 1;
    }

    const auto params = ck::utils::conv::parse_conv_param(num_dim_spatial, 10, argv);

    using F8   = ck::f8_t;
    using F16  = ck::half_t;
    using BF8  = ck::bf8_t;
    using BF16 = ck::bhalf_t;
    using I8   = int8_t;

    using GKZYXC = ck::tensor_layout::convolution::GKZYXC;
    using NDHWGC = ck::tensor_layout::convolution::NDHWGC;
    using NDHWGK = ck::tensor_layout::convolution::NDHWGK;

    using ConvScale    = ck::tensor_operation::element_wise::ConvScale;
    using ConvInvScale = ck::tensor_operation::element_wise::ConvInvscale;
    using Scale        = ck::tensor_operation::element_wise::Scale;

    constexpr auto I3 = ck::Number<3>{};

    auto profile = [&](auto num_dim_spatial_tmp,
                       auto in_layout,
                       auto wei_layout,
                       auto out_layout,
                       auto in_type,
                       auto wei_type,
                       auto out_type,
                       auto out_element_op,
                       auto a_compute_type,
                       auto b_compute_type) {
        constexpr ck::index_t NDimSpatial = num_dim_spatial_tmp.value;

        using InLayout  = decltype(in_layout);
        using WeiLayout = decltype(wei_layout);
        using OutLayout = decltype(out_layout);

        using InDataType  = decltype(in_type);
        using WeiDataType = decltype(wei_type);
        using OutDataType = decltype(out_type);

        using OutElementOp = decltype(out_element_op);

        using AComputeType = decltype(a_compute_type);
        using BComputeType = decltype(b_compute_type);

        bool pass = ck::profiler::profile_grouped_conv_fwd_outelementop_impl<NDimSpatial,
                                                                             InLayout,
                                                                             WeiLayout,
                                                                             OutLayout,
                                                                             InDataType,
                                                                             WeiDataType,
                                                                             OutDataType,
                                                                             OutElementOp,
                                                                             AComputeType,
                                                                             BComputeType>(
            do_verification, init_method, do_log, time_kernel, params);

        return pass ? 0 : 1;
    };

    if(num_dim_spatial == 3 && layout == ConvLayout::NHWGC_GKYXC_NHWGK)
    {
        if(op == OutElementOp::ConvScale)
        {
            if(data_type == ConvDataType::F8_F8_F8)
            {
                return profile(
                    I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F8{}, F8{}, F8{}, ConvScale{}, F8{}, F8{});
            }
            else if(data_type == ConvDataType::BF8_BF8_F8)
            {
                return profile(I3,
                               NDHWGC{},
                               GKZYXC{},
                               NDHWGK{},
                               BF8{},
                               BF8{},
                               F8{},
                               ConvScale{},
                               BF8{},
                               BF8{});
            }
            else if(data_type == ConvDataType::F8_BF8_F8)
            {
                return profile(
                    I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F8{}, BF8{}, F8{}, ConvScale{}, F8{}, BF8{});
            }
            else if(data_type == ConvDataType::BF8_F8_F8)
            {
                return profile(
                    I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, BF8{}, F8{}, F8{}, ConvScale{}, BF8{}, F8{});
            }
        }
        else if(op == OutElementOp::ConvInvScale)
        {
            if(data_type == ConvDataType::F8_F8_F8)
            {
                return profile(
                    I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F8{}, F8{}, F8{}, ConvInvScale{}, F8{}, F8{});
            }
            else if(data_type == ConvDataType::BF8_BF8_F8)
            {
                return profile(I3,
                               NDHWGC{},
                               GKZYXC{},
                               NDHWGK{},
                               BF8{},
                               BF8{},
                               F8{},
                               ConvInvScale{},
                               BF8{},
                               BF8{});
            }
            else if(data_type == ConvDataType::F8_BF8_F8)
            {
                return profile(I3,
                               NDHWGC{},
                               GKZYXC{},
                               NDHWGK{},
                               F8{},
                               BF8{},
                               F8{},
                               ConvInvScale{},
                               F8{},
                               BF8{});
            }
            else if(data_type == ConvDataType::BF8_F8_F8)
            {
                return profile(I3,
                               NDHWGC{},
                               GKZYXC{},
                               NDHWGK{},
                               BF8{},
                               F8{},
                               F8{},
                               ConvInvScale{},
                               BF8{},
                               F8{});
            }
        }
        else if(op == OutElementOp::Scale)
        {
            if(data_type == ConvDataType::F16_F16_F16)
            {
                return profile(
                    I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, F16{}, F16{}, F16{}, Scale{}, F16{}, F16{});
            }
            else if(data_type == ConvDataType::BF16_BF16_BF16)
            {
                return profile(I3,
                               NDHWGC{},
                               GKZYXC{},
                               NDHWGK{},
                               BF16{},
                               BF16{},
                               BF16{},
                               Scale{},
                               BF16{},
                               BF16{});
            }
            else if(data_type == ConvDataType::I8_I8_I8)
            {
                return profile(
                    I3, NDHWGC{}, GKZYXC{}, NDHWGK{}, I8{}, I8{}, I8{}, Scale{}, I8{}, I8{});
            }
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;

    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, grouped_conv_fwd_outelementop);
