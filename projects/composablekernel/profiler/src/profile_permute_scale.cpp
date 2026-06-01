// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "profiler/profile_permute_scale_impl.hpp"
#include "profiler_operation_registry.hpp"

namespace {

enum struct DataType
{
    F32_F32, // 0
    F16_F16  // 1
};

#define OP_NAME "permute_scale"
#define OP_DESC "Permute Scale"

static void print_helper_msg()
{
    std::cout
        // clang-format off
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: data type (0: Input fp32, Output fp32\n"
        << "                 1: Input fp16, Output fp16\n"
        << "arg4: verification (0: no, 1: yes)\n"
        << "arg5: initialization (0: no init, 1: integer value, 2: decimal value)\n"
        << "arg6: print tensor value (0: no; 1: yes)\n"
        << "arg7: time kernel (0: no, 1: yes)\n"
        << "from arg8: tensor lengths\n"
        << "           input strides\n"
        << "           output strides\n" << std::endl;
    // clang-format on
}

void init_strides(const std::vector<ck::index_t>& lengths,
                  const std::vector<ck::index_t>& dims_order,
                  std::vector<ck::index_t>& strides)
{

    ck::index_t stride = 1;
    for(ck::index_t d = lengths.size() - 1; d >= 0; d--)
    {
        ck::index_t dim = dims_order[d];
        strides[dim]    = stride;
        stride *= lengths[dim];
    }
}

} // namespace

int profile_permute_scale(int argc, char* argv[])
{
    constexpr int control_argc = 7;
    const int dims_argc        = argc - control_argc;
    // Number of lenghs, input strides and outputs strides must be equal
    if(argc < control_argc && dims_argc % 3 != 0)
    {
        print_helper_msg();
        return 1;
    }

    const auto data_type       = static_cast<DataType>(std::stoi(argv[2]));
    const bool do_verification = std::stoi(argv[3]);
    const int init_method      = std::stoi(argv[4]);
    const bool do_log          = std::stoi(argv[5]);
    const bool time_kernel     = std::stoi(argv[6]);
    const int num_dims         = dims_argc / 3;

    std::vector<ck::index_t> lengths(num_dims);
    std::vector<ck::index_t> input_dims_order(num_dims);
    std::vector<ck::index_t> output_dims_order(num_dims);

    for(int i = 0; i < num_dims; i++)
    {
        lengths[i]           = std::stoi(argv[control_argc + i]);
        input_dims_order[i]  = std::stoi(argv[control_argc + num_dims + i]);
        output_dims_order[i] = std::stoi(argv[control_argc + 2 * num_dims + i]);
    }

    std::vector<ck::index_t> input_strides(num_dims);
    std::vector<ck::index_t> output_strides(num_dims);
    init_strides(lengths, input_dims_order, input_strides);
    init_strides(lengths, output_dims_order, output_strides);

    using F32 = float;
    using F16 = ck::half_t;

    constexpr auto I1 = ck::Number<1>{};
    constexpr auto I2 = ck::Number<2>{};
    constexpr auto I3 = ck::Number<3>{};
    constexpr auto I4 = ck::Number<4>{};
    constexpr auto I5 = ck::Number<5>{};
    constexpr auto I6 = ck::Number<6>{};

    auto profile = [&](auto num_dim_tmp, auto in_type, auto out_type) {
        constexpr ck::index_t NDim = num_dim_tmp.value;

        using InDataType  = decltype(in_type);
        using OutDataType = decltype(out_type);

        bool pass =
            ck::profiler::profile_permute_scale_impl<InDataType, OutDataType, NDim>(do_verification,
                                                                                    init_method,
                                                                                    do_log,
                                                                                    time_kernel,
                                                                                    lengths,
                                                                                    input_strides,
                                                                                    output_strides);

        return pass ? 0 : 1;
    };

    if(num_dims == 1)
    {
        if(data_type == DataType::F32_F32)
        {
            return profile(I1, F32{}, F32{});
        }
        else if(data_type == DataType::F16_F16)
        {
            return profile(I1, F16{}, F16{});
        }
    }
    else if(num_dims == 2)
    {
        if(data_type == DataType::F32_F32)
        {
            return profile(I2, F32{}, F32{});
        }
        else if(data_type == DataType::F16_F16)
        {
            return profile(I2, F16{}, F16{});
        }
    }
    else if(num_dims == 3)
    {
        if(data_type == DataType::F32_F32)
        {
            return profile(I3, F32{}, F32{});
        }
        else if(data_type == DataType::F16_F16)
        {
            return profile(I3, F16{}, F16{});
        }
    }
    else if(num_dims == 4)
    {
        if(data_type == DataType::F32_F32)
        {
            return profile(I4, F32{}, F32{});
        }
        else if(data_type == DataType::F16_F16)
        {
            return profile(I4, F16{}, F16{});
        }
    }
    else if(num_dims == 5)
    {
        if(data_type == DataType::F32_F32)
        {
            return profile(I5, F32{}, F32{});
        }
        else if(data_type == DataType::F16_F16)
        {
            return profile(I5, F16{}, F16{});
        }
    }
    else if(num_dims == 6)
    {
        if(data_type == DataType::F32_F32)
        {
            return profile(I6, F32{}, F32{});
        }
        else if(data_type == DataType::F16_F16)
        {
            return profile(I6, F16{}, F16{});
        }
    }

    std::cout << "this data_type & layout is not implemented" << std::endl;
    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, profile_permute_scale);
