// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "profiler/common.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

namespace ck {
namespace profiler {

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename DLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename DDataType,
          typename OutDataType,
          typename AComputeType = InDataType,
          typename BComputeType = AComputeType,
          typename IndexType    = ck::index_t>
bool profile_grouped_conv_fwd_bilinear_impl(
    int do_verification,
    int init_method,
    bool do_log,
    bool time_kernel,
    const ck::utils::conv::ConvParam& conv_param,
    const ck::tensor_operation::element_wise::Bilinear& bilinear_op =
        ck::tensor_operation::element_wise::Bilinear{})
{
    using InElementOp      = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp     = ck::tensor_operation::element_wise::PassThrough;
    using OutElementOp     = ck::tensor_operation::element_wise::Bilinear;
    using CShuffleDataType = float;

    bool pass = true;

    auto f_host_tensor_descriptor =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    auto f_host_tensor_descriptor_packed =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    auto e_host_tensor_descriptor =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    auto d_host_tensor_descriptor =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<DLayout>(conv_param);

    std::array<IndexType, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<IndexType, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<IndexType, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<IndexType, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<IndexType, NDimSpatial + 3> d_g_n_k_wos_lengths{};
    std::array<IndexType, NDimSpatial + 3> d_g_n_k_wos_strides{};
    std::array<IndexType, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<IndexType, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<IndexType, NDimSpatial> conv_filter_strides{};
    std::array<IndexType, NDimSpatial> conv_filter_dilations{};
    std::array<IndexType, NDimSpatial> input_left_pads{};
    std::array<IndexType, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(f_host_tensor_descriptor.GetLengths(), a_g_n_c_wis_lengths);
    copy(f_host_tensor_descriptor.GetStrides(), a_g_n_c_wis_strides);
    copy(f_host_tensor_descriptor_packed.GetLengths(), b_g_k_c_xs_lengths);
    copy(f_host_tensor_descriptor_packed.GetStrides(), b_g_k_c_xs_strides);
    copy(d_host_tensor_descriptor.GetLengths(), d_g_n_k_wos_lengths);
    copy(d_host_tensor_descriptor.GetStrides(), d_g_n_k_wos_strides);
    copy(e_host_tensor_descriptor.GetLengths(), e_g_n_k_wos_lengths);
    copy(e_host_tensor_descriptor.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    Tensor<InDataType> input(f_host_tensor_descriptor);
    Tensor<WeiDataType> weight(f_host_tensor_descriptor_packed);
    Tensor<DDataType> d_tensor(d_host_tensor_descriptor);
    Tensor<CShuffleDataType> c(e_host_tensor_descriptor);
    Tensor<OutDataType> host_output(e_host_tensor_descriptor);
    Tensor<OutDataType> device_output(e_host_tensor_descriptor);

    std::cout << "input: " << input.mDesc << std::endl;
    std::cout << "weight: " << weight.mDesc << std::endl;
    std::cout << "d_tensor: " << d_tensor.mDesc << std::endl;
    std::cout << "output: " << host_output.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        input.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        weight.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        d_tensor.GenerateTensorValue(GeneratorTensor_2<DDataType>{-5, 5});
        break;
    default:
        input.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        weight.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        d_tensor.GenerateTensorValue(GeneratorTensor_3<DDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weight.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(DDataType) * d_tensor.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * device_output.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(input.mData.data());
    wei_device_buf.ToDevice(weight.mData.data());
    d_device_buf.ToDevice(d_tensor.mData.data());

    if(do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<
            NDimSpatial,
            InDataType,
            WeiDataType,
            CShuffleDataType,
            InElementOp,
            WeiElementOp,
            ck::tensor_operation::element_wise::PassThrough>{};

        auto ref_invoker = ref_conv.MakeInvoker();
        auto ref_argument =
            ref_conv.MakeArgument(input,
                                  weight,
                                  c,
                                  conv_param.conv_filter_strides_,
                                  conv_param.conv_filter_dilations_,
                                  conv_param.input_left_pads_,
                                  conv_param.input_right_pads_,
                                  InElementOp{},
                                  WeiElementOp{},
                                  ck::tensor_operation::element_wise::PassThrough{});

        c.SetZero();
        ref_invoker.Run(ref_argument);

        host_output.ForEach([&](auto&, auto idx) {
            const auto conv_shuffle = ck::type_convert<CShuffleDataType>(c(idx));

            const auto conv_val = conv_shuffle;
            const auto d_val    = ck::type_convert<CShuffleDataType>(d_tensor(idx));

            CShuffleDataType out_val{};
            bilinear_op(out_val, conv_val, d_val);
            host_output(idx) = ck::type_convert<OutDataType>(out_val);
        });
    }

    std::string best_op_name;
    float best_avg_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    int valids            = 0;

    using DeviceOp =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                      InLayout,
                                                                      WeiLayout,
                                                                      ck::Tuple<DLayout>,
                                                                      OutLayout,
                                                                      InDataType,
                                                                      WeiDataType,
                                                                      ck::Tuple<DDataType>,
                                                                      OutDataType,
                                                                      InElementOp,
                                                                      WeiElementOp,
                                                                      OutElementOp,
                                                                      AComputeType,
                                                                      BComputeType>;

    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    for(std::size_t i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
            static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
            std::array<const void*, 1>{static_cast<DDataType*>(d_device_buf.GetDeviceBuffer())},
            static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
            a_g_n_c_wis_lengths,
            a_g_n_c_wis_strides,
            b_g_k_c_xs_lengths,
            b_g_k_c_xs_strides,
            std::array<std::array<IndexType, NDimSpatial + 3>, 1>{d_g_n_k_wos_lengths},
            std::array<std::array<IndexType, NDimSpatial + 3>, 1>{d_g_n_k_wos_strides},
            e_g_n_k_wos_lengths,
            e_g_n_k_wos_strides,
            conv_filter_strides,
            conv_filter_dilations,
            input_left_pads,
            input_right_pads,
            InElementOp{},
            WeiElementOp{},
            bilinear_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            ++valids;

            std::string op_name = op_ptr->GetTypeString();

            if(do_log)
            {
                std::cout << "Evaluating [" << i << "] " << op_name << std::endl;
            }

            out_device_buf.SetZero();
            auto ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            auto flop      = conv_param.GetFlops();
            auto num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>() +
                             sizeof(DDataType) * (conv_param.G_ * conv_param.N_ * conv_param.K_);

            for(std::size_t j = 0; j < conv_param.filter_spatial_lengths_.size(); ++j)
            {
                num_btype += sizeof(DDataType) * conv_param.output_spatial_lengths_[j];
            }

            float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
            float gb_per_sec = num_btype / 1.E6 / ave_time;

            if(do_log)
            {
                std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops
                          << " TFlops, " << gb_per_sec << " GB/s, " << op_name << std::endl;
            }

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_avg_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                out_device_buf.FromDevice(device_output.mData.data());

                bool is_valid = ck::utils::check_err(device_output,
                                                     host_output,
                                                     "Error: Device and Host results do not match!",
                                                     get_rtol<OutDataType>(),
                                                     get_atol<OutDataType>());

                if(!is_valid)
                {
                    pass = false;
                }

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "input : ", input.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "weight: ", weight.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "d_tensor: ", d_tensor.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "host_output  : ", host_output.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "device_output: ", device_output.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            if(do_log)
            {
                std::cout << op_ptr->GetTypeString() << " does not support this problem"
                          << std::endl;
            }
        }
    }

    printf("\033[36mvalids: %d\033[0m\n", valids);

    if(valids > 0)
    {
        std::cout << "Best Perf: " << std::setw(10) << best_avg_time << " ms, " << best_tflops
                  << " TFlops, " << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;
    }

    return pass;
}

} // namespace profiler
} // namespace ck
