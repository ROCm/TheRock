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

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data_bilinear.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;

template <typename Tuple>
class TestGroupedConvndBwdData : public ::testing::Test
{
    protected:
    using InDataType  = std::tuple_element_t<0, Tuple>;
    using WeiDataType = std::tuple_element_t<0, Tuple>;
    using OutDataType = std::tuple_element_t<0, Tuple>;

    using ComputeDataType = InDataType;
    using InLayout        = std::tuple_element_t<3, Tuple>;
    using WeiLayout       = std::tuple_element_t<2, Tuple>;
    using OutLayout       = std::tuple_element_t<1, Tuple>;

    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
    using InElementOp  = ck::tensor_operation::element_wise::Bilinear;
    using OutElementOp = ck::tensor_operation::element_wise::PassThrough;
    using PassThrough  = ck::tensor_operation::element_wise::PassThrough;

    using Bilinear                           = ck::tensor_operation::element_wise::Bilinear;
    static constexpr ck::index_t NDimSpatial = 3;
    static constexpr float alpha             = 2.f;
    static constexpr float beta              = 2.f;
    static constexpr ck::index_t NumDs       = 1;

    std::vector<ck::utils::conv::ConvParam> conv_params;
    std::vector<ck::index_t> split_ks{1};

    void RunReference(ck::utils::conv::ConvParam& conv_param,
                      Tensor<InDataType>& in_host,
                      Tensor<WeiDataType>& wei,
                      Tensor<OutDataType>& out,
                      Tensor<InDataType>& d)
    {

        std::array<Tensor<InDataType>, NumDs> d_tensors = {d};
        auto ref_conv =
            ck::tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                             InDataType,
                                                             WeiDataType,
                                                             OutDataType,
                                                             InElementOp,
                                                             WeiElementOp,
                                                             OutElementOp,
                                                             0, /*Num A Elementwise Tensors*/
                                                             0, /*Num B Elementwise Tensors*/
                                                             NumDs>();

        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(in_host,
                                                  wei,
                                                  out,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  Bilinear{alpha, beta},
                                                  WeiElementOp{},
                                                  OutElementOp{},
                                                  {},
                                                  {},
                                                  d_tensors);

        ref_invoker.Run(ref_argument);
    }

    bool PerformConvDataBilinear(ck::utils::conv::ConvParam& conv_param,
                                 const ck::index_t split_k,
                                 ck::index_t instance_index_ = -1)
    {
        bool passed = true;

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
        Tensor<OutDataType> out(out_g_n_k_wos_desc);
        Tensor<InDataType> in_host(in_g_n_c_wis_desc);
        Tensor<InDataType> in_device(in_g_n_c_wis_desc);
        Tensor<InDataType> d(in_g_n_c_wis_desc);

        std::cout << "in: " << in_host.mDesc << std::endl;
        std::cout << "wei: " << wei.mDesc << std::endl;
        std::cout << "out: " << out.mDesc << std::endl;

        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        d.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});

        DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
        DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());
        DeviceMem in_device_buf(sizeof(InDataType) * in_device.mDesc.GetElementSpaceSize());
        DeviceMem d_device_buf(sizeof(InDataType) * d.mDesc.GetElementSpaceSize());

        in_device_buf.ToDevice(in_device.mData.data());
        out_device_buf.ToDevice(out.mData.data());
        wei_device_buf.ToDevice(wei.mData.data());
        d_device_buf.ToDevice(d.mData.data());

        std::array<ck::index_t, NDimSpatial + 3> out_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> out_strides{};
        std::array<ck::index_t, NDimSpatial + 3> wei_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> wei_strides{};
        std::array<ck::index_t, NDimSpatial + 3> in_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> in_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
        std::array<ck::index_t, NDimSpatial> input_left_pads{};
        std::array<ck::index_t, NDimSpatial> input_right_pads{};

        auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

        copy(out_g_n_k_wos_desc.GetLengths(), out_lengths);
        copy(out_g_n_k_wos_desc.GetStrides(), out_strides);
        copy(wei_g_k_c_xs_desc.GetLengths(), wei_lengths);
        copy(wei_g_k_c_xs_desc.GetStrides(), wei_strides);
        copy(in_g_n_c_wis_desc.GetLengths(), in_lengths);
        copy(in_g_n_c_wis_desc.GetStrides(), in_strides);
        copy(conv_param.conv_filter_strides_, conv_filter_strides);
        copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
        copy(conv_param.input_left_pads_, input_left_pads);
        copy(conv_param.input_right_pads_, input_right_pads);

        RunReference(conv_param, in_host, wei, out, d);

        using DeviceOp =
            ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<NDimSpatial,
                                                                            OutLayout,
                                                                            WeiLayout,
                                                                            ck::Tuple<InLayout>,
                                                                            InLayout,
                                                                            OutDataType,
                                                                            WeiDataType,
                                                                            ck::Tuple<InDataType>,
                                                                            InDataType,
                                                                            PassThrough,
                                                                            PassThrough,
                                                                            Bilinear>;

        // get device op instances
        const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();

        int num_kernel = 0;

        for(std::size_t i = 0; i < op_ptrs.size(); ++i)
        {
            auto& op_ptr      = op_ptrs[i];
            auto argument_ptr = op_ptr->MakeArgumentPointer(
                out_device_buf.GetDeviceBuffer(),
                wei_device_buf.GetDeviceBuffer(),
                {d_device_buf.GetDeviceBuffer()},
                in_device_buf.GetDeviceBuffer(),
                out_lengths,
                out_strides,
                wei_lengths,
                wei_strides,
                std::array<std::array<ck::index_t, NDimSpatial + 3>, NumDs>{in_lengths},
                std::array<std::array<ck::index_t, NDimSpatial + 3>, NumDs>{in_strides},
                in_lengths,
                in_strides,
                conv_filter_strides,
                conv_filter_dilations,
                input_left_pads,
                input_right_pads,
                PassThrough{},
                PassThrough{},
                Bilinear{alpha, beta},
                split_k);

            DeviceMem workspace_buf(op_ptr->GetWorkSpaceSize(argument_ptr.get()));
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_buf.GetDeviceBuffer());

            auto invoker_ptr    = op_ptr->MakeInvokerPointer();
            std::string op_name = op_ptr->GetTypeString();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                ++num_kernel;
                if((instance_index_ != -1) && (instance_index_ + 1 != num_kernel))
                {
                    // skip test if instance_index is specified
                    continue;
                }

                float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});
                in_device_buf.FromDevice(in_device.mData.data());

                passed &= ck::utils::check_err(in_device, in_host);

                std::size_t flop = conv_param.GetFlops() +
                                   3 * conv_param.GetOutputByte<InDataType>() / sizeof(InDataType);
                std::size_t num_bytes = conv_param.GetByte<InDataType, WeiDataType, OutDataType>() +
                                        conv_param.GetOutputByte<InDataType>();

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
        if(instance_index != -1)
        {
            std::cout << "grouped_conv_bwd_data_instance (" << instance_index << "/" << num_kernel
                      << "): Passed" << std::endl;
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
                pass = pass && PerformConvDataBilinear(param, split_k, instance_index);
            }
        }
        EXPECT_TRUE(pass);
    }
};

template <typename Tuple>
class TestGroupedConvndBwdData3d : public TestGroupedConvndBwdData<Tuple>
{
};

using NDHWGC = ck::tensor_layout::convolution::NDHWGC;
using GKZYXC = ck::tensor_layout::convolution::GKZYXC;
using NDHWGK = ck::tensor_layout::convolution::NDHWGK;

using KernelTypes3d = ::testing::Types<std::tuple<float, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::half_t, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::bhalf_t, NDHWGK, GKZYXC, NDHWGC>>;

TYPED_TEST_SUITE(TestGroupedConvndBwdData3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndBwdData3d, Test3D)
{
    // TODO: To fix the impl to pass with stride greater than 1.
    // this->conv_params.push_back(
    //  {3, 2, 16, 128, 128, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 16, 128, 128, {1, 1, 1}, {7, 7, 7}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
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

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 3)
    {
        param_mask     = strtol(argv[1], nullptr, 0);
        instance_index = atoi(argv[2]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1,2: param_mask instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
