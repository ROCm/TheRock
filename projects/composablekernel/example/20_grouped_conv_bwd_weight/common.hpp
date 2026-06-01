// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_bwd_weight_gpu.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;
using F8   = ck::f8_t;
using BF8  = ck::bf8_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

template <typename DataType, typename GemmType = DataType>
inline __host__ __device__ constexpr double get_rtol()
{
    if constexpr(std::is_same_v<DataType, float> && std::is_same_v<GemmType, ck::tf32_t>)
        return 5e-3;
    else if constexpr(std::is_same_v<DataType, float>)
        return 1e-3;
    else if constexpr(std::is_same_v<DataType, double>)
        return 1e-6;
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
        return 1e-3;
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
        return 5e-2;
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
        return 1e-1;
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
        return 1.5e-1;
    else
        return 1e-3;
}

template <typename DataType, typename GemmType = DataType>
inline __host__ __device__ constexpr double get_atol()
{
    if constexpr(std::is_same_v<DataType, float> && std::is_same_v<GemmType, ck::tf32_t>)
        return 1e-3;
    else if constexpr(std::is_same_v<DataType, float>)
        return 1e-3;
    else if constexpr(std::is_same_v<DataType, double>)
        return 1e-6;
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
        return 1e-3;
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
        return 5e-2;
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
        return 16.1;
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
        return 16.1;
    else
        return 1e-3;
}

template <typename InputLay, typename WeightLay, typename OutputLay>
struct CommonLayoutSetting
{
    using InputLayout  = InputLay;
    using WeightLayout = WeightLay;
    using OutputLayout = OutputLay;
};

namespace ctl = ck::tensor_layout::convolution;
template <ck::index_t NDimSpatial>
struct CommonLayoutSettingSelector
    : CommonLayoutSetting<ck::tuple_element_t<NDimSpatial - 1,
                                              ck::Tuple<ck::tensor_layout::convolution::GNWC,
                                                        ck::tensor_layout::convolution::GNHWC,
                                                        ck::tensor_layout::convolution::GNDHWC>>,
                          ck::tuple_element_t<NDimSpatial - 1,
                                              ck::Tuple<ck::tensor_layout::convolution::GKXC,
                                                        ck::tensor_layout::convolution::GKYXC,
                                                        ck::tensor_layout::convolution::GKZYXC>>,
                          ck::tuple_element_t<NDimSpatial - 1,
                                              ck::Tuple<ck::tensor_layout::convolution::GNWK,
                                                        ck::tensor_layout::convolution::GNHWK,
                                                        ck::tensor_layout::convolution::GNDHWK>>>
{
};

template <ck::index_t NDimSpatial>
using InputLayout = typename CommonLayoutSettingSelector<NDimSpatial>::InputLayout;

template <ck::index_t NDimSpatial>
using WeightLayout = typename CommonLayoutSettingSelector<NDimSpatial>::WeightLayout;

template <ck::index_t NDimSpatial>
using OutputLayout = typename CommonLayoutSettingSelector<NDimSpatial>::OutputLayout;

struct ExecutionConfig final
{
    int do_verification = 1; // 0=no, 1=CPU, 2=GPU
    int init_method     = 1;
    bool time_kernel    = false;
};

#define DefaultConvParam                                                                         \
    ck::utils::conv::ConvParam                                                                   \
    {                                                                                            \
        3, 4, 1, 128, 256, {3, 3, 3}, {14, 14, 14}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, { 1, 1, 1 } \
    }

inline void print_help_msg()
{
    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

inline bool parse_cmd_args(int argc,
                           char* argv[],
                           ExecutionConfig& config,
                           ck::utils::conv::ConvParam& conv_param)
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

        const ck::index_t num_dim_spatial = std::stoi(argv[4]);
        conv_param                        = ck::utils::conv::parse_conv_param(
            num_dim_spatial,
            threshold_to_catch_partial_args + 1, // +1 because we already parsed num_dim_spatial
            argv);
    }
    else
    {
        print_help_msg();
        return false;
    }

    return true;
}
