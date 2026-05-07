// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iomanip>
#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/quantization/gemm_quantization.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using ActivationOp = PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::Activation_Mul_Clamp<ActivationOp>;

using ADataType = int8_t;
using BDataType = int8_t;
using EDataType = int8_t;

using ALayout = Row;
using BLayout = Col;
using ELayout = Row;

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
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA = 1024;
    ck::index_t StrideB = 1024;
    ck::index_t StrideE = 1024;

    float requant_scale = 0.03;

    auto f_matrix_space_size =
        [](std::size_t nRow, std::size_t nCol, std::size_t stride, auto layout) {
            using Layout = decltype(layout);

            if constexpr(std::is_same<Layout, ck::tensor_layout::gemm::RowMajor>::value)
            {
                return (nRow - 1) * stride + nCol;
            }
            else
            {
                return (nCol - 1) * stride + nRow;
            }
        };

    SimpleDeviceMem a_device_buf(sizeof(ADataType) * f_matrix_space_size(M, K, StrideA, ALayout{}));
    SimpleDeviceMem b_device_buf(sizeof(BDataType) * f_matrix_space_size(K, N, StrideB, BLayout{}));
    SimpleDeviceMem e_device_buf(sizeof(EDataType) * f_matrix_space_size(M, N, StrideE, ELayout{}));

    using DeviceOp = ck::tensor_operation::device::DeviceGemmMultipleD<ALayout,
                                                                       BLayout,
                                                                       ck::Tuple<>,
                                                                       ELayout,
                                                                       ADataType,
                                                                       BDataType,
                                                                       ck::Tuple<>,
                                                                       EDataType,
                                                                       AElementOp,
                                                                       BElementOp,
                                                                       CDEElementOp>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    const auto a_element_op   = AElementOp{};
    const auto b_element_op   = BElementOp{};
    const auto cde_element_op = CDEElementOp{requant_scale, ActivationOp{}};

    std::string best_op_name;
    int best_op_id        = -1;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;
    float best_tflops     = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr      = op_ptrs[i];
        auto argument_ptr = op_ptr->MakeArgumentPointer(a_device_buf.GetDeviceBuffer(),
                                                        b_device_buf.GetDeviceBuffer(),
                                                        {},
                                                        e_device_buf.GetDeviceBuffer(),
                                                        M,
                                                        N,
                                                        K,
                                                        StrideA,
                                                        StrideB,
                                                        {},
                                                        StrideE,
                                                        a_element_op,
                                                        b_element_op,
                                                        cde_element_op);

        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop = std::size_t(2) * M * N * K;
            std::size_t num_bytes =
                sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(EDataType) * M * N;

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
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    if(best_op_id != -1)
    {
        std::cout << "Best Perf: " << std::setw(10) << best_avg_time << " ms, " << best_tflops
                  << " TFlops, " << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(a_device_buf.GetDeviceBuffer(),
                                                        b_device_buf.GetDeviceBuffer(),
                                                        {},
                                                        e_device_buf.GetDeviceBuffer(),
                                                        M,
                                                        N,
                                                        K,
                                                        StrideA,
                                                        StrideB,
                                                        {},
                                                        StrideE,
                                                        a_element_op,
                                                        b_element_op,
                                                        cde_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}