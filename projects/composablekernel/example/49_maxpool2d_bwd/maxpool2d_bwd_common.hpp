// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_pool2d_fwd_nhwc_nhwc.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_max_pool_bwd_impl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_pool_fwd.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_maxpool_bwd.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <typename InDataType,
          typename OutDataType,
          typename IndexDataType,
          typename ComputeDataType,
          typename DInDataType,
          typename DOutDataType,
          bool PropagateNan>
bool maxpool_bwd_test(bool do_verification,
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
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using DevicePoolFwdInstance =
        ck::tensor_operation::device::DevicePool2dFwd_NHWC_NHWC<InDataType,      // InDataType
                                                                OutDataType,     // OutDataType
                                                                IndexDataType,   // IndexDataType
                                                                ComputeDataType, // ComputeDataType
                                                                ck::ReduceTensorOp::MAX,
                                                                true,
                                                                64, // BlockSize
                                                                64, // ReduceMThreadClusterSize
                                                                1,  // ReduceKThreadClusterSize
                                                                4,  // ReduceMThreadSliceSize
                                                                1,  // ReduceKThreadSliceSize
                                                                1>; // InSrcOutDstVectorSize

    using DeviceMaxPoolBwdInstance = ck::tensor_operation::device::
        DeviceMaxPoolBwdImpl<DOutDataType, IndexDataType, DInDataType, 4>;

    const ck::index_t Ys = (Y - 1) * window_dilation_h + 1;
    const ck::index_t Xs = (X - 1) * window_dilation_w + 1;
    const ck::index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - Ys) / window_stride_h + 1;
    const ck::index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - Xs) / window_stride_w + 1;

    const std::vector<ck::index_t> window_spatial_lengths{Y, X};
    const std::vector<ck::index_t> window_strides{window_stride_h, window_stride_w};
    const std::vector<ck::index_t> window_dilations{window_dilation_h, window_dilation_w};
    const std::vector<ck::index_t> input_left_pads{in_left_pad_h, in_left_pad_w};
    const std::vector<ck::index_t> input_right_pads{in_right_pad_h, in_right_pad_w};

    auto f_host_tensor_descriptor =
        [](std::size_t N_, std::size_t C_, std::size_t H, std::size_t W) {
            using namespace ck::literals;
            // reference need Tensor with NCHW order
            return HostTensorDescriptor({N_, C_, H, W},
                                        {C_ * H * W, 1_uz, W * C_, C_},
                                        ck::tensor_layout::convolution::NCHW{});
        };

    // in
    Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi));

    // out
    Tensor<OutDataType> out_n_c_ho_wo_host(f_host_tensor_descriptor(N, C, Ho, Wo));
    Tensor<OutDataType> out_n_c_ho_wo_device(f_host_tensor_descriptor(N, C, Ho, Wo));

    // indices
    Tensor<IndexDataType> indices_n_c_ho_wo_device(f_host_tensor_descriptor(N, C, Ho, Wo));
    Tensor<IndexDataType> indices_n_c_ho_wo_host(f_host_tensor_descriptor(N, C, Ho, Wo));

    // dout
    Tensor<DOutDataType> dout_n_c_ho_wo(f_host_tensor_descriptor(N, C, Ho, Wo));

    // din
    Tensor<DInDataType> din_n_c_hi_wi_host(f_host_tensor_descriptor(N, C, Hi, Wi));
    Tensor<DInDataType> din_n_c_hi_wi_device(f_host_tensor_descriptor(N, C, Hi, Wi));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
    std::cout << "out_n_c_ho_wo: " << out_n_c_ho_wo_host.mDesc << std::endl;
    std::cout << "indices_n_c_ho_wo: " << indices_n_c_ho_wo_host.mDesc << std::endl;
    std::cout << "dout_n_c_ho_wo: " << dout_n_c_ho_wo.mDesc << std::endl;
    std::cout << "din_n_c_hi_wi: " << din_n_c_hi_wi_host.mDesc << std::endl;

    in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{-1.0, 1.0});
    dout_n_c_ho_wo.GenerateTensorValue(GeneratorTensor_3<DOutDataType>{-1.0, 1.0});

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) *
                             out_n_c_ho_wo_device.mDesc.GetElementSpaceSize());
    DeviceMem indices_device_buf(sizeof(IndexDataType) *
                                 indices_n_c_ho_wo_device.mDesc.GetElementSpaceSize());
    DeviceMem dout_device_buf(sizeof(DOutDataType) * dout_n_c_ho_wo.mDesc.GetElementSpaceSize());
    DeviceMem din_device_buf(sizeof(DInDataType) *
                             din_n_c_hi_wi_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
    dout_device_buf.ToDevice(dout_n_c_ho_wo.mData.data());

    auto pool_fwd              = DevicePoolFwdInstance{};
    auto pool_fwd_invoker_ptr  = pool_fwd.MakeInvokerPointer();
    auto pool_fwd_argument_ptr = pool_fwd.MakeArgumentPointer(
        static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
        static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
        static_cast<IndexDataType*>(indices_device_buf.GetDeviceBuffer()),
        {N, C, Hi, Wi},
        window_spatial_lengths,
        {N, C, Ho, Wo},
        {C * Hi * Wi, 1, Wi * C, C},
        {C * Ho * Wo, 1, Wo * C, C},
        {C * Ho * Wo, 1, Wo * C, C},
        window_strides,
        window_dilations,
        input_left_pads,
        input_right_pads,
        {2, 3});

    if(!pool_fwd.IsSupportedArgument(pool_fwd_argument_ptr.get()))
    {
        throw std::runtime_error("wrong! pool_fwd with the specified compilation parameters does "
                                 "not support this problem");
    }

    float ave_time_fwd =
        pool_fwd_invoker_ptr->Run(pool_fwd_argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    auto pool_bwd              = DeviceMaxPoolBwdInstance{};
    auto pool_bwd_invoker_ptr  = pool_bwd.MakeInvokerPointer();
    auto pool_bwd_argument_ptr = pool_bwd.MakeArgumentPointer(
        static_cast<DOutDataType*>(dout_device_buf.GetDeviceBuffer()),
        static_cast<IndexDataType*>(indices_device_buf.GetDeviceBuffer()),
        static_cast<DInDataType*>(din_device_buf.GetDeviceBuffer()),
        dout_n_c_ho_wo.mDesc.GetElementSpaceSize(),
        din_n_c_hi_wi_device.mDesc.GetElementSpaceSize(),
        window_spatial_lengths,
        window_strides,
        window_dilations);

    if(!pool_bwd.IsSupportedArgument(pool_bwd_argument_ptr.get()))
    {
        throw std::runtime_error("wrong! pool_bwd with the specified compilation parameters does "
                                 "not support this problem");
    }

    size_t pool_bwd_workspace_sz = pool_bwd.GetWorkSpaceSize(pool_bwd_argument_ptr.get());
    DeviceMem pool_bwd_workspace_device_buf(pool_bwd_workspace_sz);
    pool_bwd.SetWorkSpacePointer(pool_bwd_argument_ptr.get(),
                                 pool_bwd_workspace_device_buf.GetDeviceBuffer());

    float ave_time_bwd =
        pool_bwd_invoker_ptr->Run(pool_bwd_argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    std::cout << "Pool fwd perf: " << ave_time_fwd << " ms" << std::endl;
    std::cout << "Pool bwd perf: " << ave_time_bwd << " ms" << std::endl;

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
                                                            ck::ReduceTensorOp::MAX,
                                                            PropagateNan,
                                                            true>;

        auto ref_pooling_fwd          = ReferencePoolingFwdInstance{};
        auto ref_pooling_fwd_invoker  = ref_pooling_fwd.MakeInvoker();
        auto ref_pooling_fwd_argument = ref_pooling_fwd.MakeArgument(in_n_c_hi_wi,
                                                                     out_n_c_ho_wo_host,
                                                                     indices_n_c_ho_wo_host,
                                                                     window_spatial_lengths,
                                                                     window_strides,
                                                                     window_dilations,
                                                                     input_left_pads,
                                                                     input_right_pads);
        ref_pooling_fwd_invoker.Run(ref_pooling_fwd_argument);

        using ReferencePoolingBwdInstance =
            ck::tensor_operation::host::ReferenceMaxPoolBwd<DOutDataType,
                                                            IndexDataType,
                                                            ComputeDataType,
                                                            DInDataType,
                                                            PassThrough>;

        auto ref_pooling_bwd          = ReferencePoolingBwdInstance{};
        auto ref_pooling_bwd_invoker  = ref_pooling_bwd.MakeInvoker();
        auto ref_pooling_bwd_argument = ref_pooling_bwd.MakeArgument(
            dout_n_c_ho_wo, indices_n_c_ho_wo_host, din_n_c_hi_wi_host, PassThrough{});

        ref_pooling_bwd_invoker.Run(ref_pooling_bwd_argument);

        out_device_buf.FromDevice(out_n_c_ho_wo_device.mData.data());
        indices_device_buf.FromDevice(indices_n_c_ho_wo_device.mData.data());
        din_device_buf.FromDevice(din_n_c_hi_wi_device.mData.data());

        pass = pass && ck::utils::check_err(out_n_c_ho_wo_device, out_n_c_ho_wo_host);
        pass = pass && ck::utils::check_err(indices_n_c_ho_wo_device, indices_n_c_ho_wo_host);
        pass = pass && ck::utils::check_err(din_n_c_hi_wi_device, din_n_c_hi_wi_host);
    }

    return (pass);
};
