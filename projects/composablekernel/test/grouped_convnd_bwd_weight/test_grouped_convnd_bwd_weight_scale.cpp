// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight_scale.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"

template <typename Tuple>
class TestGroupedConvndBwdWeight : public ::testing::Test
{
    protected:
    using InDataType   = std::tuple_element_t<0, Tuple>;
    using WeiDataType  = std::tuple_element_t<1, Tuple>;
    using OutDataType  = std::tuple_element_t<2, Tuple>;
    using InLayout     = ck::tensor_layout::convolution::NDHWGC;
    using WeiLayout    = ck::tensor_layout::convolution::GKZYXC;
    using OutLayout    = ck::tensor_layout::convolution::NDHWGK;
    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::Scale;
    using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

    static constexpr ck::index_t NDimSpatial = std::tuple_element_t<3, Tuple>{};
    static constexpr float alpha             = 2.f;

    std::vector<ck::utils::conv::ConvParam> conv_params;
    std::vector<ck::index_t> split_ks{1, 2};

    void RunReference(ck::utils::conv::ConvParam& conv_param,
                      ck::Tensor<InDataType>& in,
                      ck::Tensor<WeiDataType>& wei_host,
                      ck::Tensor<OutDataType>& out)
    {
        auto ref_conv =
            ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                               InDataType,
                                                               WeiDataType,
                                                               OutDataType,
                                                               InElementOp,
                                                               WeiElementOp,
                                                               OutElementOp,
                                                               0, /*Num A Elementwise Tensors*/
                                                               0, /*Num B Elementwise Tensors*/
                                                               0> /*Num D Elementwise Tensors*/
            {};

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei_host,
                                                  out,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  InElementOp{},
                                                  WeiElementOp{alpha},
                                                  OutElementOp{},
                                                  {},
                                                  {},
                                                  {});

        ref_invoker.Run(ref_argument);
    }

    bool PerformConvWeightScale(ck::utils::conv::ConvParam& conv_param, const ck::index_t split_k)
    {
        bool passed = true;

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);
        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);
        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        ck::Tensor<InDataType> in(in_g_n_c_wis_desc);
        ck::Tensor<OutDataType> out(out_g_n_k_wos_desc);
        ck::Tensor<WeiDataType> wei_host(wei_g_k_c_xs_desc);
        ck::Tensor<WeiDataType> wei_device(wei_g_k_c_xs_desc);

        std::cout << "in: " << in.mDesc << std::endl;
        std::cout << "wei: " << wei_host.mDesc << std::endl;
        std::cout << "out: " << out.mDesc << std::endl;

        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});

        ck::DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
        ck::DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());
        ck::DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_device.mDesc.GetElementSpaceSize());
        in_device_buf.ToDevice(in.mData.data());
        wei_device_buf.ToDevice(wei_device.mData.data());
        out_device_buf.ToDevice(out.mData.data());

        std::array<ck::index_t, NDimSpatial + 3> b_g_n_c_wis_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> b_g_n_c_wis_strides{};
        std::array<ck::index_t, NDimSpatial + 3> e_g_k_c_xs_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> e_g_k_c_xs_strides{};
        std::array<ck::index_t, NDimSpatial + 3> a_g_n_k_wos_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> a_g_n_k_wos_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
        std::array<ck::index_t, NDimSpatial> input_left_pads{};
        std::array<ck::index_t, NDimSpatial> input_right_pads{};

        auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

        copy(in_g_n_c_wis_desc.GetLengths(), b_g_n_c_wis_lengths);
        copy(in_g_n_c_wis_desc.GetStrides(), b_g_n_c_wis_strides);
        copy(wei_g_k_c_xs_desc.GetLengths(), e_g_k_c_xs_lengths);
        copy(wei_g_k_c_xs_desc.GetStrides(), e_g_k_c_xs_strides);
        copy(out_g_n_k_wos_desc.GetLengths(), a_g_n_k_wos_lengths);
        copy(out_g_n_k_wos_desc.GetStrides(), a_g_n_k_wos_strides);
        copy(conv_param.conv_filter_strides_, conv_filter_strides);
        copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
        copy(conv_param.input_left_pads_, input_left_pads);
        copy(conv_param.input_right_pads_, input_right_pads);

        RunReference(conv_param, in, wei_host, out);

        using DeviceOp =
            ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<NDimSpatial,
                                                                              InLayout,
                                                                              WeiLayout,
                                                                              OutLayout,
                                                                              ck::Tuple<>,
                                                                              InDataType,
                                                                              WeiDataType,
                                                                              OutDataType,
                                                                              ck::Tuple<>,
                                                                              InElementOp,
                                                                              WeiElementOp,
                                                                              OutElementOp>;

        // get device op instances
        const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();

        int num_kernel = 0;

        for(std::size_t i = 0; i < op_ptrs.size(); ++i)
        {
            auto& op_ptr      = op_ptrs[i];
            auto argument_ptr = op_ptr->MakeArgumentPointer(
                static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                std::array<const void*, 0>{},
                b_g_n_c_wis_lengths,
                b_g_n_c_wis_strides,
                e_g_k_c_xs_lengths,
                e_g_k_c_xs_strides,
                a_g_n_k_wos_lengths,
                a_g_n_k_wos_strides,
                std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                conv_filter_strides,
                conv_filter_dilations,
                input_left_pads,
                input_right_pads,
                InElementOp{},
                WeiElementOp{alpha},
                OutElementOp{},
                split_k);

            ck::DeviceMem workspace_buf(op_ptr->GetWorkSpaceSize(argument_ptr.get()));
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_buf.GetDeviceBuffer());

            auto invoker_ptr    = op_ptr->MakeInvokerPointer();
            std::string op_name = op_ptr->GetTypeString();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                num_kernel++;
                float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr});
                wei_device_buf.FromDevice(wei_device.mData.data());

                using AccDataType = float;
                float max_accumulated_value =
                    *std::max_element(wei_host.mData.begin(), wei_host.mData.end());

                const ck::index_t num_accums         = out.GetElementSize() / conv_param.K_;
                const ck::index_t num_accums_split_k = split_k;
                double rtol =
                    ck::utils::get_relative_threshold<InDataType, WeiDataType, AccDataType>(
                        num_accums / num_accums_split_k);
                double atol =
                    ck::utils::get_absolute_threshold<InDataType, WeiDataType, AccDataType>(
                        max_accumulated_value / num_accums_split_k,
                        num_accums / num_accums_split_k);

                // Calculate error due to split_k accumulation
                auto rtol_split_k =
                    ck::utils::get_relative_threshold<WeiDataType, WeiDataType, WeiDataType>(
                        num_accums_split_k);
                auto atol_split_k =
                    ck::utils::get_absolute_threshold<WeiDataType, WeiDataType, WeiDataType>(
                        max_accumulated_value, num_accums_split_k);
                // Use higher threshold
                rtol = std::max(rtol, rtol_split_k);
                atol = std::max(atol, atol_split_k);

                passed &= ck::utils::check_err(
                    wei_device, wei_host, "Error: incorrect results!", rtol, atol);

                std::size_t flop =
                    conv_param.GetFlops() +
                    3 * conv_param.GetOutputByte<WeiDataType>() / sizeof(WeiDataType);
                std::size_t num_bytes = conv_param.GetByte<InDataType, WeiDataType, OutDataType>() +
                                        conv_param.GetOutputByte<WeiDataType>();

                float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
                float gb_per_sec = num_bytes / 1.E6 / avg_time;

                std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops
                          << " TFlops, " << gb_per_sec << " GB/s, " << op_name << std::endl;
            }
            else
            {
                std::cerr << op_name << " does not support this problem" << std::endl;
            }
        }

        printf("\033[36mvalids: %d\033[0m\n", num_kernel);
        return passed;
    }

    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;

        for(auto split_k : split_ks)
        {
            for(auto& param : conv_params)
            {
                pass = pass && PerformConvWeightScale(param, split_k);
            }
        }
        EXPECT_TRUE(pass);
    }
};

template <typename Tuple>
class TestGroupedConvndBwdWeight3d : public TestGroupedConvndBwdWeight<Tuple>
{
};

using KernelTypes3d =
    ::testing::Types<std::tuple<float, float, float, ck::Number<3>>,
                     std::tuple<ck::half_t, ck::half_t, ck::half_t, ck::Number<3>>,
                     std::tuple<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t, ck::Number<3>>,
                     std::tuple<ck::bhalf_t, float, ck::bhalf_t, ck::Number<3>>>;

TYPED_TEST_SUITE(TestGroupedConvndBwdWeight3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndBwdWeight3d, Test3D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {3, 2, 16, 128, 128, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 2, 128, 128, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 2, 32, 128, 128, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 1, 32, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 64, 3, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 1, 1, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 4, 4, {3, 3, 3}, {14, 28, 28}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->Run();
}
