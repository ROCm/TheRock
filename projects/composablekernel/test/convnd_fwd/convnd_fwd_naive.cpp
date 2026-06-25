// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_conv3d_fwd_naive_ndhwc_kzyxc_ndhwk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

using InDataType  = float;
using WeiDataType = float;
using OutDataType = float;
using AccDataType = float;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

using DeviceConvNaive = ck::tensor_operation::device::
    DeviceConv3dFwdNaive_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K<InDataType,
                                                                                 WeiDataType,
                                                                                 OutDataType,
                                                                                 AccDataType,
                                                                                 InElementOp,
                                                                                 WeiElementOp,
                                                                                 OutElementOp>;

template <ck::index_t NDimSpatial>
bool run_conv3d_naive_test(const ck::utils::conv::ConvParam& conv_param)
{
    using namespace ck;
    using namespace ck::tensor_operation::host;

    using InLayout  = ck::tensor_layout::convolution::GNCDHW;
    using WeiLayout = ck::tensor_layout::convolution::GKCZYX;
    using OutLayout = ck::tensor_layout::convolution::GNKDHW;

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<OutDataType> out_host(out_g_n_k_wos_desc);
    Tensor<OutDataType> out_device(out_g_n_k_wos_desc);

    // Initialize tensors
    in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
    wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    // Run device kernel - convert long_index_t vectors to index_t
    std::vector<ck::index_t> input_spatial_lengths(conv_param.input_spatial_lengths_.begin(),
                                                   conv_param.input_spatial_lengths_.end());
    std::vector<ck::index_t> filter_spatial_lengths(conv_param.filter_spatial_lengths_.begin(),
                                                    conv_param.filter_spatial_lengths_.end());
    auto output_spatial_lengths_long = conv_param.GetOutputSpatialLengths();
    std::vector<ck::index_t> output_spatial_lengths(output_spatial_lengths_long.begin(),
                                                    output_spatial_lengths_long.end());
    std::vector<ck::index_t> conv_filter_strides(conv_param.conv_filter_strides_.begin(),
                                                 conv_param.conv_filter_strides_.end());
    std::vector<ck::index_t> conv_filter_dilations(conv_param.conv_filter_dilations_.begin(),
                                                   conv_param.conv_filter_dilations_.end());
    std::vector<ck::index_t> input_left_pads(conv_param.input_left_pads_.begin(),
                                             conv_param.input_left_pads_.end());
    std::vector<ck::index_t> input_right_pads(conv_param.input_right_pads_.begin(),
                                              conv_param.input_right_pads_.end());

    auto conv    = DeviceConvNaive{};
    auto invoker = conv.MakeInvoker();
    auto argument =
        conv.MakeArgument(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                          static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                          static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                          conv_param.N_,
                          conv_param.K_,
                          conv_param.C_,
                          input_spatial_lengths,
                          filter_spatial_lengths,
                          output_spatial_lengths,
                          conv_filter_strides,
                          conv_filter_dilations,
                          input_left_pads,
                          input_right_pads,
                          InElementOp{},
                          WeiElementOp{},
                          OutElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        std::cout << "Unsupported argument for naive conv3d kernel" << std::endl;
        return false;
    }

    invoker.Run(argument, StreamConfig{nullptr, false});

    // Run CPU reference
    auto ref_conv = ReferenceConvFwd<NDimSpatial,
                                     InDataType,
                                     WeiDataType,
                                     OutDataType,
                                     InElementOp,
                                     WeiElementOp,
                                     OutElementOp,
                                     0,
                                     0,
                                     0,
                                     AccDataType>();

    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(in,
                                              wei,
                                              out_host,
                                              conv_param.conv_filter_strides_,
                                              conv_param.conv_filter_dilations_,
                                              conv_param.input_left_pads_,
                                              conv_param.input_right_pads_,
                                              InElementOp{},
                                              WeiElementOp{},
                                              OutElementOp{});

    ref_invoker.Run(ref_argument);

    // Compare results
    out_device_buf.FromDevice(out_device.mData.data());

    return ck::utils::check_err(out_device, out_host, "Error: incorrect results!", 1e-3, 1e-3);
}

TEST(TestConv3dNaive, Conv3dNaive_Small)
{
    // Small 3D convolution test
    ck::utils::conv::ConvParam param{
        3,         // spatial_dim
        1,         // G
        2,         // N
        16,        // K
        16,        // C
        {3, 3, 3}, // filter
        {7, 7, 7}, // input spatial
        {2, 2, 2}, // strides
        {1, 1, 1}, // dilations
        {1, 1, 1}, // left pads
        {1, 1, 1}  // right pads
    };

    bool pass = run_conv3d_naive_test<3>(param);
    EXPECT_TRUE(pass);
}

TEST(TestConv3dNaive, Conv3dNaive_Medium)
{
    // Medium size 3D convolution test
    ck::utils::conv::ConvParam param{
        3,            // spatial_dim
        1,            // G
        4,            // N
        32,           // K
        32,           // C
        {3, 3, 3},    // filter
        {14, 14, 14}, // input spatial
        {1, 1, 1},    // strides
        {1, 1, 1},    // dilations
        {1, 1, 1},    // left pads
        {1, 1, 1}     // right pads
    };

    bool pass = run_conv3d_naive_test<3>(param);
    EXPECT_TRUE(pass);
}

TEST(TestConv3dNaive, Conv3dNaive_UnitFilter)
{
    // 1x1x1 filter (no padding)
    ck::utils::conv::ConvParam param{
        3,         // spatial_dim
        1,         // G
        2,         // N
        24,        // K
        24,        // C
        {1, 1, 1}, // filter
        {8, 8, 8}, // input spatial
        {1, 1, 1}, // strides
        {1, 1, 1}, // dilations
        {0, 0, 0}, // left pads
        {0, 0, 0}  // right pads
    };

    bool pass = run_conv3d_naive_test<3>(param);
    EXPECT_TRUE(pass);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
