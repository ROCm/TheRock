// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>
#include <initializer_list>
#include <gtest/gtest.h>

#include "ck/utility/data_type.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scaleadd_ab.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

using I8   = int8_t;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

// This is pretty much a fully functional profiler function, but I only implemented it here to add a
// proper gtest test for the scaleadd_ab flavor. At some point we may want to move this and add it
// to the ckProfiler.
template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename IndexType = ck::index_t>
bool profile_grouped_conv_fwd_scaleadd_ab_impl(int do_verification,
                                               int init_method,
                                               bool do_log,
                                               [[maybe_unused]] bool time_kernel,
                                               const ck::utils::conv::ConvParam& conv_param)
{
    constexpr ck::index_t NumAs = 2;
    constexpr ck::index_t NumBs = 2;
    using InElementOp           = ck::tensor_operation::element_wise::ScaleAdd;
    using WeiElementOp          = ck::tensor_operation::element_wise::ScaleAdd;
    using OutElementOp          = ck::tensor_operation::element_wise::PassThrough;

    constexpr float scale = 1.5f;

    const auto in_element_op  = InElementOp{scale};
    const auto wei_element_op = WeiElementOp{scale};
    const auto out_element_op = OutElementOp{};

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    std::array<IndexType, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<IndexType, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<IndexType, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<IndexType, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<IndexType, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<IndexType, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<IndexType, NDimSpatial> conv_filter_strides{};
    std::array<IndexType, NDimSpatial> conv_filter_dilations{};
    std::array<IndexType, NDimSpatial> input_left_pads{};
    std::array<IndexType, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    ck::Tensor<InDataType> input(in_g_n_c_wis_desc);
    ck::Tensor<InDataType> input_bias(in_g_n_c_wis_desc);
    ck::Tensor<WeiDataType> weight(wei_g_k_c_xs_desc);
    ck::Tensor<WeiDataType> weight_bias(wei_g_k_c_xs_desc);
    ck::Tensor<OutDataType> host_output(out_g_n_k_wos_desc);
    ck::Tensor<OutDataType> device_output(out_g_n_k_wos_desc);

    std::cout << "input: " << input.mDesc << std::endl;
    std::cout << "weight: " << weight.mDesc << std::endl;
    std::cout << "output: " << host_output.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        input.GenerateTensorValue(GeneratorTensor_2<InDataType>{-2, 2});
        input_bias.GenerateTensorValue(GeneratorTensor_2<InDataType>{-2, 2});
        weight.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-2, 2});
        weight_bias.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-2, 2});
        break;
    default:
        input.GenerateTensorValue(GeneratorTensor_3<InDataType>{-1.0, 1.0});
        input_bias.GenerateTensorValue(GeneratorTensor_3<InDataType>{-1.0, 1.0});
        weight.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.05, 0.05});
        weight_bias.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-1.0, 1.0});
    }

    ck::DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpaceSize());
    ck::DeviceMem in_bias_device_buf(sizeof(InDataType) * input_bias.mDesc.GetElementSpaceSize());
    ck::DeviceMem wei_device_buf(sizeof(WeiDataType) * weight.mDesc.GetElementSpaceSize());
    ck::DeviceMem wei_bias_device_buf(sizeof(WeiDataType) *
                                      weight_bias.mDesc.GetElementSpaceSize());
    ck::DeviceMem out_device_buf(sizeof(OutDataType) * device_output.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(input.mData.data());
    in_bias_device_buf.ToDevice(input_bias.mData.data());
    wei_device_buf.ToDevice(weight.mData.data());
    wei_bias_device_buf.ToDevice(weight_bias.mData.data());

    // Run reference op
    if(do_verification)
    {
        const std::array<ck::Tensor<InDataType>, NumAs - 1> elementwise_a_tensors  = {input_bias};
        const std::array<ck::Tensor<WeiDataType>, NumBs - 1> elementwise_b_tensors = {weight_bias};
        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     OutDataType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     OutElementOp,
                                                                     NumAs - 1,
                                                                     NumBs - 1>();

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(input,
                                                  weight,
                                                  host_output,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op,
                                                  elementwise_a_tensors,
                                                  elementwise_b_tensors);

        // init host output to zero
        host_output.SetZero();

        ref_invoker.Run(ref_argument);
    }

    std::string best_op_name;
    float best_avg_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    int valids            = 0;

    // profile device op instances
    bool pass = true;

    auto run_impl = [&](auto& op_ptr, auto& argument_ptr) {
        // workspace_sz will be equal to 0 for other layout than NGCHW
        // TODO: Is workspace even necessary?
        const std::size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
        ck::DeviceMem workspace_dev(workspace_sz);
        op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            // re-init output to zero before profiling next kernel
            out_device_buf.SetZero();

            valids++;

            std::string op_name = op_ptr->GetTypeString();

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            float avg_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop = conv_param.GetFlops() +
                               2 * conv_param.GetOutputByte<InDataType>() / sizeof(InDataType) +
                               2 * conv_param.GetOutputByte<WeiDataType>() / sizeof(WeiDataType);
            std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>() +
                                    conv_param.GetInputByte<InDataType>() +
                                    conv_param.GetWeightByte<WeiDataType>();

            float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

            float gb_per_sec = num_btype / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                out_device_buf.FromDevice(device_output.mData.data());

                pass = pass & ck::utils::check_err(device_output, host_output);

                if(do_log)
                {
                    printf("log\n");
                    //     LogRangeAsType<float>(std::cout << "input : ", input.mData, ",") <<
                    //     std::endl; LogRangeAsType<float>(std::cout << "input_bias: ",
                    //     input_bias.mData, ",")
                    //         << std::endl;
                    //     LogRangeAsType<float>(std::cout << "weight: ", weight.mData, ",") <<
                    //     std::endl; LogRangeAsType<float>(std::cout << "weight_bias: ",
                    //     weight_bias.mData, ",")
                    //         << std::endl;
                    //     LogRangeAsType<float>(std::cout << "host_output  : ", host_output.mData,
                    //     ",")
                    //         << std::endl;
                    //     LogRangeAsType<float>(std::cout << "device_output: ",
                    //     device_output.mData, ",")
                    //         << std::endl;
                }
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    };

    // InDataType and WeiDataType must be tuple, inLayout and weiLayout are single.
    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
        NDimSpatial,
        InLayout,
        WeiLayout,
        ck::Tuple<>,
        OutLayout,
        ck::Tuple<InDataType, InDataType>,
        ck::Tuple<WeiDataType, WeiDataType>,
        ck::Tuple<>,
        OutDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "ckProfiler found " << op_ptrs.size() << " instances" << std::endl;

    std::array<const void*, NumAs> as{in_device_buf.GetDeviceBuffer(),
                                      in_bias_device_buf.GetDeviceBuffer()};
    std::array<const void*, NumBs> bs{wei_device_buf.GetDeviceBuffer(),
                                      wei_bias_device_buf.GetDeviceBuffer()};
    std::array<const void*, 0> ds{};

    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(as,
                                                        bs,
                                                        ds,
                                                        out_device_buf.GetDeviceBuffer(),
                                                        a_g_n_c_wis_lengths,
                                                        a_g_n_c_wis_strides,
                                                        b_g_k_c_xs_lengths,
                                                        b_g_k_c_xs_strides,
                                                        {},
                                                        {},
                                                        e_g_n_k_wos_lengths,
                                                        e_g_n_k_wos_strides,
                                                        conv_filter_strides,
                                                        conv_filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        in_element_op,
                                                        wei_element_op,
                                                        out_element_op);

        run_impl(op_ptr, argument_ptr);
    }

    printf("\033[36mvalids: %d\n\033[0m", valids);

    std::cout << "Best configuration parameters:" << "\nname: " << best_op_name
              << "\navg_time: " << best_avg_time << "\ntflops: " << best_tflops
              << "\nGB/s: " << best_gb_per_sec << std::endl;

    return pass;
}

template <typename Tuple>
class TestGroupedConvndFwdScaleaddAB : public ::testing::Test
{
    protected:
    using InDataType  = std::tuple_element_t<0, Tuple>;
    using WeiDataType = std::tuple_element_t<1, Tuple>;
    using OutDataType = std::tuple_element_t<2, Tuple>;
    using InLayout    = std::tuple_element_t<3, Tuple>;
    using WeiLayout   = std::tuple_element_t<4, Tuple>;
    using OutLayout   = std::tuple_element_t<5, Tuple>;

    std::vector<ck::utils::conv::ConvParam> conv_params;

    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;
        for(auto& param : conv_params)
        {
            pass = pass && profile_grouped_conv_fwd_scaleadd_ab_impl<NDimSpatial,
                                                                     InLayout,
                                                                     WeiLayout,
                                                                     OutLayout,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     OutDataType>(
                               true,  // do_verification
                               1,     // init_method: integer value
                               false, // do_log
                               false, // time_kernel
                               param);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

// TODO: Not all possible layouts exist in the instance factory, (GNDHWC, GKZYXC, GNDHWK) only
// exists in example 62.
using KernelTypes3d = ::testing::Types<std::tuple<F16, F16, F16, NDHWGC, GKZYXC, NDHWGK>,
                                       std::tuple<BF16, BF16, BF16, NDHWGC, GKZYXC, NDHWGK>,
                                       std::tuple<F32, F32, F32, NDHWGC, GKZYXC, NDHWGK>,
                                       std::tuple<I8, I8, I8, NDHWGC, GKZYXC, NDHWGK>>;

template <typename Tuple>
class TestGroupedConvndFwdScaleaddAB3d : public TestGroupedConvndFwdScaleaddAB<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndFwdScaleaddAB3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndFwdScaleaddAB3d, Test3D)
{
    this->conv_params.clear();

    // Generic problems, same set as for vanilla, clamp, and (gk) bias clamp tests.
    this->conv_params.push_back(
        {3, 3, 5, 96, 200, {1, 1, 1}, {37, 37, 16}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {1, 1, 1}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {2, 2, 2}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {5, 5, 5}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {9, 9, 9}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {1, 1, 1}, {16, 16, 16}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    this->conv_params.push_back(
        {3, 1, 1, 1, 32, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 64, 3, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 1, 1, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    this->conv_params.push_back(
        {3, 96, 1, 1, 1, {1, 1, 1}, {120, 40, 20}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 96, 1, 1, 1, {3, 3, 3}, {120, 40, 20}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->template Run<3>();
}
