// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iomanip>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_pool_fwd.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/pool3d_fwd.hpp"

using InDataType    = ck::half_t;
using OutDataType   = ck::half_t;
using IndexDataType = int32_t;

// We use pool3d to implement pool2d in this example
using InLayout  = ck::tensor_layout::convolution::NDHWC;
using OutLayout = ck::tensor_layout::convolution::NDHWC;

constexpr ck::index_t InOutRank  = 5;
constexpr ck::index_t WindowRank = 3;
#if 1
constexpr auto ReduceOpId  = ck::ReduceTensorOp::MAX;
constexpr bool OutputIndex = true;
#else
constexpr auto ReduceOpId  = ck::ReduceTensorOp::AVG;
constexpr bool OutputIndex = false;
#endif

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

void TransformPool2dparamToPool3d(std::vector<ck::index_t>& input_lengths,
                                  std::vector<ck::index_t>& window_lengths,
                                  std::vector<ck::index_t>& output_lengths,
                                  std::vector<ck::index_t>& input_stride,
                                  std::vector<ck::index_t>& output_stride,
                                  std::vector<ck::index_t>& indices_stride,
                                  std::vector<ck::index_t>& window_strides,
                                  std::vector<ck::index_t>& window_dilations,
                                  std::vector<ck::index_t>& input_left_pads,
                                  std::vector<ck::index_t>& input_right_pads,
                                  std::vector<ck::index_t>& pooling_dims)
{
    // NCHW to NCDHW
    input_lengths.insert(input_lengths.begin() + 2, 1);
    output_lengths.insert(output_lengths.begin() + 2, 1);
    input_stride.insert(input_stride.begin() + 2, 0);
    output_stride.insert(output_stride.begin() + 2, 0);
    indices_stride.insert(indices_stride.begin() + 2, 0);

    // YX to ZYX
    window_lengths.insert(window_lengths.begin(), 1);
    window_strides.insert(window_strides.begin(), 0);
    window_dilations.insert(window_dilations.begin(), 0);
    input_left_pads.insert(input_left_pads.begin(), 0);
    input_right_pads.insert(input_right_pads.begin(), 0);

    pooling_dims = {2, 3, 4};
}

int main(int argc, char* argv[])
{
    ck::index_t N                 = 2;
    ck::index_t C                 = 32;
    ck::index_t Y                 = 2;
    ck::index_t X                 = 2;
    ck::index_t Hi                = 30;
    ck::index_t Wi                = 30;
    ck::index_t window_stride_h   = 2;
    ck::index_t window_stride_w   = 2;
    ck::index_t window_dilation_h = 1;
    ck::index_t window_dilation_w = 1;
    ck::index_t in_left_pad_h     = 1;
    ck::index_t in_left_pad_w     = 1;
    ck::index_t in_right_pad_h    = 1;
    ck::index_t in_right_pad_w    = 1;

    const ck::index_t Ys = (Y - 1) * window_dilation_h + 1;
    const ck::index_t Xs = (X - 1) * window_dilation_w + 1;
    ck::index_t Ho       = (Hi + in_left_pad_h + in_right_pad_h - Ys) / window_stride_h + 1;
    ck::index_t Wo       = (Wi + in_left_pad_w + in_right_pad_w - Xs) / window_stride_w + 1;

    // Pool API only support the order of NCHW
    std::vector<ck::index_t> in_length              = {N, C, Hi, Wi};
    std::vector<ck::index_t> out_length             = {N, C, Ho, Wo};
    std::vector<ck::index_t> window_spatial_lengths = {Y, X};
    std::vector<ck::index_t> window_strides         = {window_stride_h, window_stride_w};
    std::vector<ck::index_t> window_dilations       = {window_dilation_h, window_dilation_w};
    std::vector<ck::index_t> input_left_pads        = {in_left_pad_h, in_left_pad_w};
    std::vector<ck::index_t> input_right_pads       = {in_right_pad_h, in_right_pad_w};
    std::vector<ck::index_t> pooling_dims           = {2, 3};

    std::size_t in_tensor_size  = N * C * Hi * Wi;
    std::size_t out_tensor_size = N * C * Ho * Wo;

    // tensor layout = NHWC
    std::vector<ck::index_t> in_tensor_stride  = {C * Hi * Wi, 1, Wi * C, C};
    std::vector<ck::index_t> out_tensor_stride = {C * Ho * Wo, 1, Wo * C, C};

    TransformPool2dparamToPool3d(in_length,
                                 window_spatial_lengths,
                                 out_length,
                                 in_tensor_stride,
                                 out_tensor_stride,
                                 out_tensor_stride,
                                 window_strides,
                                 window_dilations,
                                 input_left_pads,
                                 input_right_pads,
                                 pooling_dims);

    SimpleDeviceMem in_device_buf(sizeof(InDataType) * in_tensor_size);
    SimpleDeviceMem out_device_buf(sizeof(OutDataType) * out_tensor_size);
    SimpleDeviceMem out_indices_device_buf(sizeof(IndexDataType) * out_tensor_size);

    using DeviceOp = ck::tensor_operation::device::DevicePoolFwd<InOutRank,
                                                                 WindowRank,
                                                                 InDataType,
                                                                 OutDataType,
                                                                 IndexDataType,
                                                                 InLayout,
                                                                 OutLayout,
                                                                 ReduceOpId,
                                                                 OutputIndex>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    bool found            = false;
    int best_op_id        = -1;
    float best_ave_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr      = op_ptrs[i];
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
            static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
            static_cast<IndexDataType*>(out_indices_device_buf.GetDeviceBuffer()),
            in_length,
            window_spatial_lengths,
            out_length,
            in_tensor_stride,
            out_tensor_stride,
            out_tensor_stride,
            window_strides,
            window_dilations,
            input_left_pads,
            input_right_pads,
            pooling_dims);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t num_bytes =
                in_tensor_size * sizeof(InDataType) + out_tensor_size * sizeof(OutDataType);

            if constexpr(OutputIndex)
                num_bytes += out_tensor_size * sizeof(IndexDataType);

            float gb_per_sec = num_bytes / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << gb_per_sec << " GB/s, "
                      << op_name << std::endl;

            if(ave_time < best_ave_time)
            {
                found           = true;
                best_op_id      = i;
                best_op_name    = op_name;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    // run the best intance
    if(found)
    {
        std::cout << "Best Perf: " << best_ave_time << " ms, " << best_gb_per_sec << " GB/s, "
                  << best_op_name << std::endl;

        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
            static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
            static_cast<IndexDataType*>(out_indices_device_buf.GetDeviceBuffer()),
            in_length,
            window_spatial_lengths,
            out_length,
            in_tensor_stride,
            out_tensor_stride,
            out_tensor_stride,
            window_strides,
            window_dilations,
            input_left_pads,
            input_right_pads,
            pooling_dims);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
