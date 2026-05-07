// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/conv_tensor_rearrange.hpp"
#include "ck/tensor_operation/gpu/device/conv_tensor_rearrange_op.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InDataType  = ck::half_t;
using OutDataType = ck::half_t;

using ImageLayout = ck::tensor_layout::convolution::NHWGC;

static constexpr ck::index_t NumDimSpatial = 2;
static constexpr ck::index_t G             = 2;
static constexpr ck::index_t N             = 32; // batch size
static constexpr ck::index_t C             = 32; // input channel (per group)
static constexpr ck::index_t Y             = 3;  // filter H
static constexpr ck::index_t X             = 3;  // filter W
static constexpr ck::index_t Hi            = 28; // input H
static constexpr ck::index_t Wi            = 28; // input W
static constexpr ck::index_t Ho            = 28; // output H
static constexpr ck::index_t Wo            = 28; // output W

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

int main()
{

    std::array<ck::index_t, 2> in_spatial_lengths{Hi, Wi};
    std::array<ck::index_t, 2> wei_spatial_lengths{Y, X};
    std::array<ck::index_t, 2> out_spatial_lengths{Ho, Wo};

    // We have NHWGC in memory space
    // However, CK's API only accepts lengths and strides with order of GNCHW.
    // Hence, we need to adjust the order of strides.
    std::array<ck::index_t, 5> image_strides{C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
    std::array<ck::index_t, 3> gemm_strides{Y * X * C, G * Y * X * C, 1};

    std::array<ck::index_t, NumDimSpatial> filter_strides{1, 1};
    std::array<ck::index_t, NumDimSpatial> filter_dilations{1, 1};
    std::array<ck::index_t, NumDimSpatial> input_left_pads{1, 1};
    std::array<ck::index_t, NumDimSpatial> input_right_pads{1, 1};

    SimpleDeviceMem in(sizeof(InDataType) * N * Hi * Wi * G * C);
    SimpleDeviceMem out(sizeof(OutDataType) * G * N * Ho * Wo * Y * X * C);

    using namespace ck::conv_tensor_rearrange_op;

    using DeviceOp = ck::tensor_operation::device::DeviceConvTensorRearrange<NumDimSpatial,
                                                                             ImageLayout,
                                                                             InDataType,
                                                                             OutDataType,
                                                                             ImageToColumn>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    int best_op_id        = -1;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr        = op_ptrs[i];
        auto argument_ptr   = op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                                        out.GetDeviceBuffer(),
                                                        G,
                                                        N,
                                                        C,
                                                        in_spatial_lengths,
                                                        out_spatial_lengths,
                                                        wei_spatial_lengths,
                                                        image_strides,
                                                        gemm_strides,
                                                        filter_strides,
                                                        filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads);
        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t num_bytes = sizeof(InDataType) * N * Hi * Wi * G * C +
                                    sizeof(OutDataType) * G * N * Ho * Wo * Y * X * C;

            float gb_per_sec = num_bytes / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << gb_per_sec << " GB/s, "
                      << op_name << std::endl;

            if(avg_time < best_avg_time)
            {
                best_op_id      = i;
                best_op_name    = op_name;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cerr << op_name << " does not support this problem" << std::endl;
        }
    }

    if(best_op_id < 0)
    {
        std::cerr << "no suitable instance" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Best Perf: " << std::setw(10) << best_avg_time << " ms, " << best_gb_per_sec
              << " GB/s, " << best_op_name << std::endl;

    // run the best intance
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                                        out.GetDeviceBuffer(),
                                                        G,
                                                        N,
                                                        C,
                                                        in_spatial_lengths,
                                                        out_spatial_lengths,
                                                        wei_spatial_lengths,
                                                        image_strides,
                                                        gemm_strides,
                                                        filter_strides,
                                                        filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }
}
