// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd.hpp"
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

template <ck::index_t NumDimSpatial>
std::size_t GetFlops(const std::array<ck::index_t, NumDimSpatial>& output_lengths,
                     const std::array<ck::index_t, NumDimSpatial>& filter_lengths)
{
    constexpr ck::index_t spatial_offset = 3;
    const auto C                         = filter_lengths[2];
    // 2 * G * N * K * C * <output spatial lengths product> * <filter spatial lengths product>
    return static_cast<std::size_t>(2) * C *
           std::accumulate(std::begin(output_lengths),
                           std::end(output_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<>()) *
           std::accumulate(std::begin(filter_lengths) + spatial_offset,
                           std::end(filter_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<>());
}

template <typename InDataType, ck::index_t NumDimSpatial>
std::size_t GetInputByte(const std::array<ck::index_t, NumDimSpatial>& input_lengths)
{
    // sizeof(InDataType) * (G * N * C * <input spatial lengths product>) +
    return sizeof(InDataType) * (std::accumulate(std::begin(input_lengths),
                                                 std::end(input_lengths),
                                                 static_cast<std::size_t>(1),
                                                 std::multiplies<>()));
}

template <typename WeiDataType, ck::index_t NumDimSpatial>
std::size_t GetWeightByte(const std::array<ck::index_t, NumDimSpatial>& filter_lengths)
{
    // sizeof(WeiDataType) * (G * K * C * <filter spatial lengths product>) +
    return sizeof(WeiDataType) * (std::accumulate(std::begin(filter_lengths),
                                                  std::end(filter_lengths),
                                                  static_cast<std::size_t>(1),
                                                  std::multiplies<>()));
}

template <typename OutDataType, ck::index_t NumDimSpatial>
std::size_t GetOutputByte(const std::array<ck::index_t, NumDimSpatial>& output_lengths)
{
    // sizeof(OutDataType) * (G * N * K * <output spatial lengths product>);
    return sizeof(OutDataType) * (std::accumulate(std::begin(output_lengths),
                                                  std::end(output_lengths),
                                                  static_cast<std::size_t>(1),
                                                  std::multiplies<std::size_t>()));
}

template <ck::index_t NumDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename AComputeType = InDataType,
          typename BComputeType = AComputeType>
bool run_grouped_conv_bwd_weight(
    const std::array<ck::index_t, NumDimSpatial + 3>& input_lengths,
    const std::array<ck::index_t, NumDimSpatial + 3>& input_strides,
    const std::array<ck::index_t, NumDimSpatial + 3>& filter_lengths,
    const std::array<ck::index_t, NumDimSpatial + 3>& weights_strides,
    const std::array<ck::index_t, NumDimSpatial + 3>& output_lengths,
    const std::array<ck::index_t, NumDimSpatial + 3>& output_strides,
    const std::array<ck::index_t, NumDimSpatial>& conv_filter_strides,
    const std::array<ck::index_t, NumDimSpatial>& conv_filter_dilations,
    const std::array<ck::index_t, NumDimSpatial>& input_left_pads,
    const std::array<ck::index_t, NumDimSpatial>& input_right_pads)
{

    ck::index_t split_k = 2;
    SimpleDeviceMem in(GetInputByte<InDataType, NumDimSpatial + 3>(input_lengths));
    SimpleDeviceMem wei(GetWeightByte<WeiDataType, NumDimSpatial + 3>(filter_lengths));
    SimpleDeviceMem out(GetOutputByte<OutDataType, NumDimSpatial + 3>(output_lengths));

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<NumDimSpatial,
                                                                              InLayout,
                                                                              WeiLayout,
                                                                              OutLayout,
                                                                              InDataType,
                                                                              WeiDataType,
                                                                              OutDataType,
                                                                              PassThrough,
                                                                              PassThrough,
                                                                              PassThrough,
                                                                              AComputeType,
                                                                              BComputeType>;
    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    int best_op_id        = -1;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;
    float best_tflops     = 0;

    std::array<ck::index_t, NumDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NumDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NumDimSpatial + 3> b_g_k_c_xs_lengths{};

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr        = op_ptrs[i];
        auto argument_ptr   = op_ptr->MakeArgumentPointer(in.GetDeviceBuffer(),
                                                        wei.GetDeviceBuffer(),
                                                        out.GetDeviceBuffer(),
                                                        input_lengths,
                                                        input_strides,
                                                        filter_lengths,
                                                        weights_strides,
                                                        output_lengths,
                                                        output_strides,
                                                        conv_filter_strides,
                                                        conv_filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        PassThrough{},
                                                        split_k);
        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        const std::size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
        SimpleDeviceMem workspace_dev(workspace_sz);
        op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop      = GetFlops<NumDimSpatial + 3>(output_lengths, filter_lengths);
            std::size_t num_bytes = GetInputByte<InDataType, NumDimSpatial + 3>(input_lengths) +
                                    GetWeightByte<WeiDataType, NumDimSpatial + 3>(filter_lengths) +
                                    GetOutputByte<OutDataType, NumDimSpatial + 3>(output_lengths);

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
                                                        input_lengths,
                                                        input_strides,
                                                        filter_lengths,
                                                        weights_strides,
                                                        output_lengths,
                                                        output_strides,
                                                        conv_filter_strides,
                                                        conv_filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        PassThrough{},
                                                        PassThrough{},
                                                        PassThrough{},
                                                        split_k);
        auto invoker_ptr  = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return true;
}
