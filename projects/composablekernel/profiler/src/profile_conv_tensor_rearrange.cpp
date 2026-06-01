// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_conv_tensor_rearrange_impl.hpp"
#include "profiler_operation_registry.hpp"

namespace {

enum struct RearrangeOp
{
    ImageToColumn, // 0
    ColumnToImage, // 1
};

enum struct ConvLayout
{
    GNHWC, // 0
    NHWGC, // 1
};

enum struct DataType
{
    F32_F32,   // 0
    F16_F16,   // 1
    BF16_BF16, // 2
    INT8_INT8, // 3
};

#define OP_NAME "conv_tensor_rearrange"
#define OP_DESC "Conv Tensor Rearrange"

static void print_helper_msg()
{
    std::cout
        // clang-format off
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: data type (0: Input fp32, Weight fp32, Output fp32\n"
        << "                 1: Input fp16, Weight fp16, Output fp16\n"
        << "                 2: Input bf16, Weight bf16, Output bf16\n"
        << "                 3: Input int8, Weight int8, Output int8)\n"
        << "arg3: tensor layout (0: Input[G, N, Hi, Wi, C], Output[G * N * Ho * Wo, Y * X * C],\n"
        << "                     1: Input[N, Hi, Wi, G, C], Output[N * Ho * Wo * G, Y * X * C])\n"
        << "arg4: verification (0: no, 1: yes)\n"
        << "arg5: initialization (0: no init, 1: integer value, 2: decimal value)\n"
        << "arg6: print tensor value (0: no; 1: yes)\n"
        << "arg7: time kernel (0: no, 1: yes)\n"
        << "arg8: operation type (0: ImageToColumn, 1: ColumnToImage)\n"
        << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
    // clang-format on
}

} // namespace

int profile_conv_tensor_rearrange(int argc, char* argv[])
{
    // 9 for control, 1 for num_dim_spatial
    if(argc < 10)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<DataType>(std::stoi(argv[2]));
    const auto layout          = static_cast<ConvLayout>(std::stoi(argv[3]));
    const bool do_verification = std::stoi(argv[4]);
    const int init_method      = std::stoi(argv[5]);
    const bool do_log          = std::stoi(argv[6]);
    const bool time_kernel     = std::stoi(argv[7]);
    const auto rearrange_op    = static_cast<RearrangeOp>(std::stoi(argv[8]));
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

    using namespace ck::tensor_layout::convolution;
    using namespace ck::conv_tensor_rearrange_op;

    constexpr auto I1 = ck::Number<1>{};
    constexpr auto I2 = ck::Number<2>{};
    constexpr auto I3 = ck::Number<3>{};

    auto profile = [&](auto num_dim_spatial_tmp,
                       auto in_layout,
                       auto in_type,
                       auto out_type,
                       auto rearrange_op_type) {
        constexpr ck::index_t NDimSpatial = num_dim_spatial_tmp.value;

        using InLayout = decltype(in_layout);

        using InDataType  = decltype(in_type);
        using OutDataType = decltype(out_type);

        using Op = decltype(rearrange_op_type);

        bool pass = ck::profiler::
            profile_conv_tensor_rearrange_impl<NDimSpatial, InLayout, InDataType, OutDataType, Op>(
                do_verification, init_method, do_log, time_kernel, params);

        return pass ? 0 : 1;
    };

    if(rearrange_op == RearrangeOp::ImageToColumn)
    {
        if(layout == ConvLayout::GNHWC)
        {
            if(num_dim_spatial == 1)
            {
                if(data_type == DataType::F32_F32)
                {
                    return profile(I1, GNWC{}, F32{}, F32{}, ImageToColumn{});
                }
                else if(data_type == DataType::F16_F16)
                {
                    return profile(I1, GNWC{}, F16{}, F16{}, ImageToColumn{});
                }
                else if(data_type == DataType::BF16_BF16)
                {
                    return profile(I1, GNWC{}, BF16{}, BF16{}, ImageToColumn{});
                }
                else if(data_type == DataType::INT8_INT8)
                {
                    return profile(I1, GNWC{}, INT8{}, INT8{}, ImageToColumn{});
                }
            }
            else if(num_dim_spatial == 2)
            {
                if(data_type == DataType::F32_F32)
                {
                    return profile(I2, GNHWC{}, F32{}, F32{}, ImageToColumn{});
                }
                else if(data_type == DataType::F16_F16)
                {
                    return profile(I2, GNHWC{}, F16{}, F16{}, ImageToColumn{});
                }
                else if(data_type == DataType::BF16_BF16)
                {
                    return profile(I2, GNHWC{}, BF16{}, BF16{}, ImageToColumn{});
                }
                else if(data_type == DataType::INT8_INT8)
                {
                    return profile(I2, GNHWC{}, INT8{}, INT8{}, ImageToColumn{});
                }
            }
            else if(num_dim_spatial == 3)
            {
                if(data_type == DataType::F32_F32)
                {
                    return profile(I3, GNDHWC{}, F32{}, F32{}, ImageToColumn{});
                }
                else if(data_type == DataType::F16_F16)
                {
                    return profile(I3, GNDHWC{}, F16{}, F16{}, ImageToColumn{});
                }
                else if(data_type == DataType::BF16_BF16)
                {
                    return profile(I3, GNDHWC{}, BF16{}, BF16{}, ImageToColumn{});
                }
                else if(data_type == DataType::INT8_INT8)
                {
                    return profile(I3, GNDHWC{}, INT8{}, INT8{}, ImageToColumn{});
                }
            }
        }
        else if(layout == ConvLayout::NHWGC)
        {
            if(num_dim_spatial == 1)
            {
                if(data_type == DataType::F32_F32)
                {
                    return profile(I1, NWGC{}, F32{}, F32{}, ImageToColumn{});
                }
                else if(data_type == DataType::F16_F16)
                {
                    return profile(I1, NWGC{}, F16{}, F16{}, ImageToColumn{});
                }
                else if(data_type == DataType::BF16_BF16)
                {
                    return profile(I1, NWGC{}, BF16{}, BF16{}, ImageToColumn{});
                }
                else if(data_type == DataType::INT8_INT8)
                {
                    return profile(I1, NWGC{}, INT8{}, INT8{}, ImageToColumn{});
                }
            }
            else if(num_dim_spatial == 2)
            {
                if(data_type == DataType::F32_F32)
                {
                    return profile(I2, NHWGC{}, F32{}, F32{}, ImageToColumn{});
                }
                else if(data_type == DataType::F16_F16)
                {
                    return profile(I2, NHWGC{}, F16{}, F16{}, ImageToColumn{});
                }
                else if(data_type == DataType::BF16_BF16)
                {
                    return profile(I2, NHWGC{}, BF16{}, BF16{}, ImageToColumn{});
                }
                else if(data_type == DataType::INT8_INT8)
                {
                    return profile(I2, NHWGC{}, INT8{}, INT8{}, ImageToColumn{});
                }
            }
            else if(num_dim_spatial == 3)
            {
                if(data_type == DataType::F32_F32)
                {
                    return profile(I3, NDHWGC{}, F32{}, F32{}, ImageToColumn{});
                }
                else if(data_type == DataType::F16_F16)
                {
                    return profile(I3, NDHWGC{}, F16{}, F16{}, ImageToColumn{});
                }
                else if(data_type == DataType::BF16_BF16)
                {
                    return profile(I3, NDHWGC{}, BF16{}, BF16{}, ImageToColumn{});
                }
                else if(data_type == DataType::INT8_INT8)
                {
                    return profile(I3, NDHWGC{}, INT8{}, INT8{}, ImageToColumn{});
                }
            }
        }
    }
    else if(rearrange_op == RearrangeOp::ColumnToImage)
    {
        if(layout == ConvLayout::GNHWC)
        {
            if(num_dim_spatial == 1)
            {
                if(data_type == DataType::F32_F32)
                {
                    return profile(I1, GNWC{}, F32{}, F32{}, ColumnToImage{});
                }
                else if(data_type == DataType::F16_F16)
                {
                    return profile(I1, GNWC{}, F16{}, F16{}, ColumnToImage{});
                }
                else if(data_type == DataType::BF16_BF16)
                {
                    return profile(I1, GNWC{}, BF16{}, BF16{}, ColumnToImage{});
                }
                else if(data_type == DataType::INT8_INT8)
                {
                    return profile(I1, GNWC{}, INT8{}, INT8{}, ColumnToImage{});
                }
            }
            else if(num_dim_spatial == 2)
            {
                if(data_type == DataType::F32_F32)
                {
                    return profile(I2, GNHWC{}, F32{}, F32{}, ColumnToImage{});
                }
                else if(data_type == DataType::F16_F16)
                {
                    return profile(I2, GNHWC{}, F16{}, F16{}, ColumnToImage{});
                }
                else if(data_type == DataType::BF16_BF16)
                {
                    return profile(I2, GNHWC{}, BF16{}, BF16{}, ColumnToImage{});
                }
                else if(data_type == DataType::INT8_INT8)
                {
                    return profile(I2, GNHWC{}, INT8{}, INT8{}, ColumnToImage{});
                }
            }
            else if(num_dim_spatial == 3)
            {
                if(data_type == DataType::F32_F32)
                {
                    return profile(I3, GNDHWC{}, F32{}, F32{}, ColumnToImage{});
                }
                else if(data_type == DataType::F16_F16)
                {
                    return profile(I3, GNDHWC{}, F16{}, F16{}, ColumnToImage{});
                }
                else if(data_type == DataType::BF16_BF16)
                {
                    return profile(I3, GNDHWC{}, BF16{}, BF16{}, ColumnToImage{});
                }
                else if(data_type == DataType::INT8_INT8)
                {
                    return profile(I3, GNDHWC{}, INT8{}, INT8{}, ColumnToImage{});
                }
            }
        }
        else if(layout == ConvLayout::NHWGC)
        {
            if(num_dim_spatial == 1)
            {
                if(data_type == DataType::F32_F32)
                {
                    return profile(I1, NWGC{}, F32{}, F32{}, ColumnToImage{});
                }
                else if(data_type == DataType::F16_F16)
                {
                    return profile(I1, NWGC{}, F16{}, F16{}, ColumnToImage{});
                }
                else if(data_type == DataType::BF16_BF16)
                {
                    return profile(I1, NWGC{}, BF16{}, BF16{}, ColumnToImage{});
                }
                else if(data_type == DataType::INT8_INT8)
                {
                    return profile(I1, NWGC{}, INT8{}, INT8{}, ColumnToImage{});
                }
            }
            else if(num_dim_spatial == 2)
            {
                if(data_type == DataType::F32_F32)
                {
                    return profile(I2, NHWGC{}, F32{}, F32{}, ColumnToImage{});
                }
                else if(data_type == DataType::F16_F16)
                {
                    return profile(I2, NHWGC{}, F16{}, F16{}, ColumnToImage{});
                }
                else if(data_type == DataType::BF16_BF16)
                {
                    return profile(I2, NHWGC{}, BF16{}, BF16{}, ColumnToImage{});
                }
                else if(data_type == DataType::INT8_INT8)
                {
                    return profile(I2, NHWGC{}, INT8{}, INT8{}, ColumnToImage{});
                }
            }
            else if(num_dim_spatial == 3)
            {
                if(data_type == DataType::F32_F32)
                {
                    return profile(I3, NDHWGC{}, F32{}, F32{}, ColumnToImage{});
                }
                else if(data_type == DataType::F16_F16)
                {
                    return profile(I3, NDHWGC{}, F16{}, F16{}, ColumnToImage{});
                }
                else if(data_type == DataType::BF16_BF16)
                {
                    return profile(I3, NDHWGC{}, BF16{}, BF16{}, ColumnToImage{});
                }
                else if(data_type == DataType::INT8_INT8)
                {
                    return profile(I3, NDHWGC{}, INT8{}, INT8{}, ColumnToImage{});
                }
            }
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;
    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_conv_tensor_rearrange);
