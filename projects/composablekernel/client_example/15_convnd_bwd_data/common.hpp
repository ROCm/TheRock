// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/convolution_backward_data.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_bwd_data.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

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

std::size_t GetFlops(ck::index_t N,
                     ck::index_t K,
                     ck::index_t C,
                     const std::vector<ck::index_t>& output_spatial_lengths,
                     const std::vector<ck::index_t>& weights_spatial_lengths)
{
    // 2 * N * K * C * <output spatial lengths product> * <filter spatial lengths product>

    return static_cast<std::size_t>(2) * N * K * C *
           std::accumulate(std::begin(output_spatial_lengths),
                           std::end(output_spatial_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<>()) *
           std::accumulate(std::begin(weights_spatial_lengths),
                           std::end(weights_spatial_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<>());
}

template <typename InDataType>
std::size_t
GetInputByte(ck::index_t N, ck::index_t C, const std::vector<ck::index_t>& input_spatial_lengths)
{
    // sizeof(InDataType) * (N * C * <input spatial lengths product>) +
    return sizeof(InDataType) * N * C *
           std::accumulate(std::begin(input_spatial_lengths),
                           std::end(input_spatial_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<>());
}

template <typename WeiDataType>
std::size_t
GetWeightByte(ck::index_t K, ck::index_t C, const std::vector<ck::index_t>& weights_spatial_lengths)
{
    // sizeof(WeiDataType) * (K * C * <filter spatial lengths product>) +
    return sizeof(WeiDataType) * K * C *
           std::accumulate(std::begin(weights_spatial_lengths),
                           std::end(weights_spatial_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<>());
}

template <typename OutDataType>
std::size_t
GetOutputByte(ck::index_t N, ck::index_t K, const std::vector<ck::index_t>& output_spatial_lengths)
{
    // sizeof(OutDataType) * (N * K * <output spatial lengths product>);
    return sizeof(OutDataType) * N * K *
           std::accumulate(std::begin(output_spatial_lengths),
                           std::end(output_spatial_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<std::size_t>());
}

template <ck::index_t NumDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
bool run_conv_bwd_data(ck::index_t N,
                       ck::index_t K,
                       ck::index_t C,
                       const std::vector<ck::index_t>& in_spatial_lengths,
                       const std::vector<ck::index_t>& wei_spatial_lengths,
                       const std::vector<ck::index_t>& out_spatial_lengths)
{
    std::size_t in_mem_size  = GetInputByte<InDataType>(N, C, in_spatial_lengths);
    std::size_t wei_mem_size = GetWeightByte<WeiDataType>(K, C, wei_spatial_lengths);
    std::size_t out_mem_size = GetOutputByte<OutDataType>(N, K, out_spatial_lengths);

    SimpleDeviceMem in(in_mem_size);
    SimpleDeviceMem wei(wei_mem_size);
    SimpleDeviceMem out(out_mem_size);

    std::vector<ck::index_t> filter_strides(NumDimSpatial, 1);
    std::vector<ck::index_t> filter_dilations(NumDimSpatial, 1);
    std::vector<ck::index_t> input_left_pads(NumDimSpatial, 1);
    std::vector<ck::index_t> input_right_pads(NumDimSpatial, 1);

    using DeviceOp = ck::tensor_operation::device::DeviceConvBwdData<NumDimSpatial,
                                                                     InLayout,
                                                                     WeiLayout,
                                                                     OutLayout,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     OutDataType,
                                                                     PassThrough,
                                                                     PassThrough,
                                                                     PassThrough>;
    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    int best_op_id        = -1;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;
    float best_tflops     = 0;

    std::size_t flop      = GetFlops(N, K, C, out_spatial_lengths, wei_spatial_lengths);
    std::size_t num_bytes = in_mem_size + wei_mem_size + out_mem_size;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr        = op_ptrs[i];
        auto argument_ptr   = op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                                        wei.GetDeviceBuffer(),
                                                        out.GetDeviceBuffer(),
                                                        N,
                                                        K,
                                                        C,
                                                        in_spatial_lengths,
                                                        wei_spatial_lengths,
                                                        out_spatial_lengths,
                                                        filter_strides,
                                                        filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        PassThrough{});
        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = num_bytes / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_id      = i;
                best_op_name    = op_name;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
                best_tflops     = tflops;
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
        return false;
    }

    std::cout << "Best Perf: " << std::setw(10) << best_avg_time << " ms, " << best_tflops
              << " TFlops, " << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    // run the best intance
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                                        wei.GetDeviceBuffer(),
                                                        out.GetDeviceBuffer(),
                                                        N,
                                                        K,
                                                        C,
                                                        in_spatial_lengths,
                                                        wei_spatial_lengths,
                                                        out_spatial_lengths,
                                                        filter_strides,
                                                        filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        PassThrough{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }
    return true;
}
