// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_pool2d_fwd_nhwc_nhwc.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_pool_fwd.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <typename InDataType,
          typename OutDataType,
          typename ComputeDataType,
          typename IndexDataType,
          typename InLayout,
          typename OutLayout,
          ck::ReduceTensorOp ReduceOpId,
          bool PropagateNan,
          bool OutputIndex>
bool pool_test(bool do_verification,
               int init_method,
               bool time_kernel,
               ck::index_t N,
               ck::index_t C,
               ck::index_t Y,
               ck::index_t X,
               ck::index_t Hi,
               ck::index_t Wi,
               ck::index_t window_stride_h,
               ck::index_t window_stride_w,
               ck::index_t window_dilation_h,
               ck::index_t window_dilation_w,
               ck::index_t in_left_pad_h,
               ck::index_t in_left_pad_w,
               ck::index_t in_right_pad_h,
               ck::index_t in_right_pad_w)
{
    using DevicePoolFwdInstance =
        ck::tensor_operation::device::DevicePool2dFwd_NHWC_NHWC<InDataType,
                                                                OutDataType,
                                                                IndexDataType,
                                                                ComputeDataType,
                                                                ReduceOpId,
                                                                OutputIndex,
                                                                64, // BlockSize
                                                                64, // ReduceMThreadClusterSize
                                                                1,  // ReduceKThreadClusterSize
                                                                4,  // ReduceMThreadSliceSize
                                                                1,  // ReduceKThreadSliceSize
                                                                1>; // InSrcOutDstVectorSize

    const ck::index_t Ys = (Y - 1) * window_dilation_h + 1;
    const ck::index_t Xs = (X - 1) * window_dilation_w + 1;
    const ck::index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - Ys) / window_stride_h + 1;
    const ck::index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - Xs) / window_stride_w + 1;

    const std::vector<ck::index_t> window_spatial_lengths{Y, X};
    const std::vector<ck::index_t> window_strides{window_stride_h, window_stride_w};
    const std::vector<ck::index_t> window_dilations{window_dilation_h, window_dilation_w};
    const std::vector<ck::index_t> input_left_pads{in_left_pad_h, in_left_pad_w};
    const std::vector<ck::index_t> input_right_pads{in_right_pad_h, in_right_pad_w};

    // tensor layout
    auto f_host_tensor_descriptor =
        [](std::size_t N_, std::size_t C_, std::size_t H, std::size_t W, auto layout) {
            using namespace ck::literals;

            if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::convolution::NCHW>::value)
            {
                return HostTensorDescriptor({N_, C_, H, W}, {C_ * H * W, H * W, W, 1_uz}, layout);
            }
            else if constexpr(ck::is_same<decltype(layout),
                                          ck::tensor_layout::convolution::NHWC>::value)
            {
                return HostTensorDescriptor({N_, C_, H, W}, {C_ * H * W, 1_uz, W * C_, C_}, layout);
            }
        };

    Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi, InLayout{}));
    Tensor<OutDataType> out_n_c_ho_wo_host(f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));
    Tensor<IndexDataType> out_indices_n_c_ho_wo_host(
        f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));
    Tensor<OutDataType> out_n_c_ho_wo_device(f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));
    Tensor<IndexDataType> out_indices_n_c_ho_wo_device(
        f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
    std::cout << "out_n_c_ho_wo: " << out_n_c_ho_wo_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}); break;
    case 2: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}); break;
    default: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) *
                             out_n_c_ho_wo_device.mDesc.GetElementSpaceSize());
    DeviceMem out_indices_device_buf(sizeof(IndexDataType) *
                                     out_indices_n_c_ho_wo_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());

    auto pool         = DevicePoolFwdInstance{};
    auto invoker_ptr  = pool.MakeInvokerPointer();
    auto argument_ptr = pool.MakeArgumentPointer(
        static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
        static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
        static_cast<IndexDataType*>(out_indices_device_buf.GetDeviceBuffer()),
        {N, C, Hi, Wi},
        {Y, X},
        {N, C, Ho, Wo},
        {C * Hi * Wi, 1, Wi * C, C},
        {C * Ho * Wo, 1, Wo * C, C},
        {C * Ho * Wo, 1, Wo * C, C},
        window_strides,
        window_dilations,
        input_left_pads,
        input_right_pads,
        {2, 3});

    if(!pool.IsSupportedArgument(argument_ptr.get()))
    {
        throw std::runtime_error("wrong! device_op with the specified compilation parameters does "
                                 "not support this problem");
    }

    float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    std::size_t flop = std::size_t(2) * N * C * Ho * Wo * Y * X;

    std::size_t num_btype =
        sizeof(InDataType) * (N * C * Hi * Wi) + sizeof(OutDataType) * (N * C * Ho * Wo);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
              << " GB / s " << std::endl;

    bool pass = true;

    if(do_verification)
    {
        using ReferencePoolingFwdInstance =
            ck::tensor_operation::host::ReferencePoolingFwd<4,
                                                            2,
                                                            InDataType,
                                                            OutDataType,
                                                            ComputeDataType,
                                                            IndexDataType,
                                                            ReduceOpId,
                                                            PropagateNan,
                                                            OutputIndex>;

        auto ref_pooling          = ReferencePoolingFwdInstance{};
        auto ref_pooling_invoker  = ref_pooling.MakeInvoker();
        auto ref_pooling_argument = ref_pooling.MakeArgument(in_n_c_hi_wi,
                                                             out_n_c_ho_wo_host,
                                                             out_indices_n_c_ho_wo_host,
                                                             window_spatial_lengths,
                                                             window_strides,
                                                             window_dilations,
                                                             input_left_pads,
                                                             input_right_pads);

        ref_pooling_invoker.Run(ref_pooling_argument);

        out_device_buf.FromDevice(out_n_c_ho_wo_device.mData.data());

        pass = pass && ck::utils::check_err(out_n_c_ho_wo_device, out_n_c_ho_wo_host);

        if constexpr(OutputIndex)
        {
            out_indices_device_buf.FromDevice(out_indices_n_c_ho_wo_device.mData.data());

            pass = pass &&
                   ck::utils::check_err(out_indices_n_c_ho_wo_device, out_indices_n_c_ho_wo_host);
        };
    }

    return (pass);
};
