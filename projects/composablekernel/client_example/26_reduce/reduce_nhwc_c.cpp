// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <functional>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/reduce/reduce.hpp"

using InDataType  = float;
using OutDataType = float;
using AccDataType = float;
using ReduceAdd   = ck::reduce::Add;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using UnaryDivide = ck::tensor_operation::element_wise::UnaryDivide;

constexpr bool PropagateNan = false;
constexpr bool OutputIndex  = false;

constexpr int Rank         = 4;
constexpr int NumReduceDim = 3;

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

int main(int argc, char* argv[])
{
    std::array<ck::index_t, Rank> in_lengths{16, 8, 128, 256};
    std::array<ck::index_t, Rank> in_strides{8 * 128 * 256, 128 * 256, 256, 1};
    std::array<ck::index_t, Rank - NumReduceDim> out_lengths{256};
    std::array<ck::index_t, Rank - NumReduceDim> out_strides{1};
    std::array<int, NumReduceDim> reduce_dims{0, 1, 2};

    ck::index_t num_in_elements =
        std::accumulate(in_lengths.begin(), in_lengths.end(), 1, std::multiplies<ck::index_t>());

    ck::index_t num_out_elements =
        std::accumulate(out_lengths.begin(), out_lengths.end(), 1, std::multiplies<ck::index_t>());

    ck::index_t reduce_length = 1;

    for(auto dim : reduce_dims)
        reduce_length *= in_lengths[dim];

    double alpha{1.0};
    double beta{0.0};

    SimpleDeviceMem in(sizeof(InDataType) * num_in_elements);
    SimpleDeviceMem out(sizeof(OutDataType) * num_out_elements);

    using DeviceOp     = ck::tensor_operation::device::DeviceReduce<InDataType,
                                                                    AccDataType,
                                                                    OutDataType,
                                                                    Rank,
                                                                    NumReduceDim,
                                                                    ReduceAdd,
                                                                    PassThrough,
                                                                    UnaryDivide,
                                                                    PropagateNan,
                                                                    OutputIndex>;
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
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr   = op_ptr->MakeArgumentPointer(in_lengths,
                                                        in_strides,
                                                        out_lengths,
                                                        out_strides,
                                                        reduce_dims,
                                                        alpha,
                                                        beta,
                                                        in.GetDeviceBuffer(),
                                                        nullptr,
                                                        out.GetDeviceBuffer(),
                                                        nullptr,
                                                        PassThrough{},
                                                        UnaryDivide{reduce_length});
        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t num_bytes = num_in_elements * sizeof(InDataType) +
                                    (beta == 0.0f ? 1 : 2) * num_out_elements * sizeof(OutDataType);

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

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_gb_per_sec << " GB/s, "
              << best_op_name << std::endl;

    // run the best intance
    if(found)
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(in_lengths,
                                                        in_strides,
                                                        out_lengths,
                                                        out_strides,
                                                        reduce_dims,
                                                        alpha,
                                                        beta,
                                                        in.GetDeviceBuffer(),
                                                        nullptr,
                                                        out.GetDeviceBuffer(),
                                                        nullptr,
                                                        PassThrough{},
                                                        UnaryDivide{reduce_length});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
