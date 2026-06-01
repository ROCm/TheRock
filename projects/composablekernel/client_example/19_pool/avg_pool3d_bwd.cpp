// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iomanip>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/avg_pool3d_bwd.hpp"

using DOutDataType = ck::half_t;
using DInDataType  = ck::half_t;

using DOutLayout = ck::tensor_layout::convolution::NDHWC;
using DInLayout  = ck::tensor_layout::convolution::NDHWC;

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}, mMemSize_(mem_size)
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    void SetZero() const { (void)hipMemset(p_mem_, 0, mMemSize_); }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
    std::size_t mMemSize_;
};

int main(int argc, char* argv[])
{
    ck::index_t N                 = 2;
    ck::index_t C                 = 32;
    ck::index_t Z                 = 2;
    ck::index_t Y                 = 2;
    ck::index_t X                 = 2;
    ck::index_t Di                = 30;
    ck::index_t Hi                = 30;
    ck::index_t Wi                = 30;
    ck::index_t window_stride_d   = 2;
    ck::index_t window_stride_h   = 2;
    ck::index_t window_stride_w   = 2;
    ck::index_t window_dilation_d = 1;
    ck::index_t window_dilation_h = 1;
    ck::index_t window_dilation_w = 1;
    ck::index_t in_left_pad_d     = 1;
    ck::index_t in_left_pad_h     = 1;
    ck::index_t in_left_pad_w     = 1;
    ck::index_t in_right_pad_d    = 1;
    ck::index_t in_right_pad_h    = 1;
    ck::index_t in_right_pad_w    = 1;

    const ck::index_t Zs = (Z - 1) * window_dilation_d + 1;
    const ck::index_t Ys = (Y - 1) * window_dilation_h + 1;
    const ck::index_t Xs = (X - 1) * window_dilation_w + 1;
    ck::index_t Do       = (Di + in_left_pad_d + in_right_pad_d - Zs) / window_stride_d + 1;
    ck::index_t Ho       = (Hi + in_left_pad_h + in_right_pad_h - Ys) / window_stride_h + 1;
    ck::index_t Wo       = (Wi + in_left_pad_w + in_right_pad_w - Xs) / window_stride_w + 1;

    // Pool API only support the order of NCDHW
    std::vector<ck::index_t> in_length              = {N, C, Di, Hi, Wi};
    std::vector<ck::index_t> out_length             = {N, C, Do, Ho, Wo};
    std::vector<ck::index_t> window_spatial_lengths = {Z, Y, X};
    std::vector<ck::index_t> window_strides = {window_stride_d, window_stride_h, window_stride_w};
    std::vector<ck::index_t> window_dilations{
        window_dilation_d, window_dilation_h, window_dilation_w};
    std::vector<ck::index_t> input_left_pads  = {in_left_pad_d, in_left_pad_h, in_left_pad_w};
    std::vector<ck::index_t> input_right_pads = {in_right_pad_d, in_right_pad_h, in_right_pad_w};

    std::size_t in_tensor_size  = N * C * Di * Hi * Wi;
    std::size_t out_tensor_size = N * C * Do * Ho * Wo;

    // tensor layout = NDHWC
    std::vector<ck::index_t> in_tensor_stride  = {Di * C * Hi * Wi, 1, C * Hi * Wi, Wi * C, C};
    std::vector<ck::index_t> out_tensor_stride = {Do * C * Ho * Wo, 1, C * Ho * Wo, Wo * C, C};

    SimpleDeviceMem dout_device_buf(sizeof(DOutDataType) * out_tensor_size);
    SimpleDeviceMem din_device_buf(sizeof(DInDataType) * in_tensor_size);

    using DeviceOp = ck::tensor_operation::device::
        DeviceAvgPoolBwd<3, DOutDataType, DInDataType, DOutLayout, DInLayout>;

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
            static_cast<DOutDataType*>(dout_device_buf.GetDeviceBuffer()),
            static_cast<DInDataType*>(din_device_buf.GetDeviceBuffer()),
            out_length,
            in_length,
            out_tensor_stride,
            in_tensor_stride,
            window_spatial_lengths,
            window_strides,
            window_dilations,
            input_left_pads,
            input_right_pads);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            din_device_buf.SetZero();

            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t num_bytes =
                in_tensor_size * sizeof(DInDataType) + out_tensor_size * sizeof(DOutDataType);

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
            static_cast<DOutDataType*>(dout_device_buf.GetDeviceBuffer()),
            static_cast<DInDataType*>(din_device_buf.GetDeviceBuffer()),
            out_length,
            in_length,
            out_tensor_stride,
            in_tensor_stride,
            window_spatial_lengths,
            window_strides,
            window_dilations,
            input_left_pads,
            input_right_pads);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            din_device_buf.SetZero();
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
