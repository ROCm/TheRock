// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iomanip>
#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_layernorm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using F16 = ck::half_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddReluAdd  = ck::tensor_operation::element_wise::AddReluAdd;

// DataType
using ADataType     = F16;
using BDataType     = F16;
using D0DataType    = F16;
using D1DataType    = F16;
using GammaDataType = F16;
using BetaDataType  = F16;
using HDataType     = F16;

// Layout
using ALayout  = Row;
using BLayout  = Col;
using D0Layout = Row;
using D1Layout = Row;
using HLayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = AddReluAdd;
using HElementOp   = PassThrough;

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
    // GEMM shape
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA  = K;
    ck::index_t StrideB  = K;
    ck::index_t StrideD0 = 0;
    ck::index_t StrideD1 = N;
    ck::index_t StrideH  = N;

    float epsilon = 1e-5;

    auto f_matrix_space_size =
        [](std::size_t nRow, std::size_t nCol, std::size_t stride, auto layout) {
            using Layout = decltype(layout);

            if constexpr(std::is_same<Layout, Row>::value)
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
    SimpleDeviceMem d0_device_buf(sizeof(D0DataType) *
                                  f_matrix_space_size(M, N, StrideD0, D0Layout{}));
    SimpleDeviceMem d1_device_buf(sizeof(D1DataType) *
                                  f_matrix_space_size(M, N, StrideD1, D1Layout{}));
    SimpleDeviceMem gamma_device_buf(sizeof(GammaDataType) * N);
    SimpleDeviceMem beta_device_buf(sizeof(BetaDataType) * N);
    SimpleDeviceMem h_device_buf(sizeof(HDataType) * f_matrix_space_size(M, N, StrideH, HLayout{}));

    using DeviceOp = ck::tensor_operation::device::DeviceGemmMultipleDLayernorm<
        ALayout,
        BLayout,
        ck::Tuple<D0Layout, D1Layout>,
        HLayout,
        ADataType,
        BDataType,
        ck::Tuple<D0DataType, D1DataType>,
        GammaDataType,
        BetaDataType,
        HDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::AddReluAdd,
        ck::tensor_operation::element_wise::PassThrough>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    const auto a_element_op   = AElementOp{};
    const auto b_element_op   = BElementOp{};
    const auto cde_element_op = CDEElementOp{};
    const auto h_element_op   = HElementOp{};

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

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            a_device_buf.GetDeviceBuffer(),
            b_device_buf.GetDeviceBuffer(),
            {d0_device_buf.GetDeviceBuffer(), d1_device_buf.GetDeviceBuffer()},
            gamma_device_buf.GetDeviceBuffer(),
            beta_device_buf.GetDeviceBuffer(),
            h_device_buf.GetDeviceBuffer(),
            M,
            N,
            K,
            StrideA,
            StrideB,
            {StrideD0, StrideD1},
            StrideH,
            epsilon,
            a_element_op,
            b_element_op,
            cde_element_op,
            h_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
            SimpleDeviceMem workspace_dev(workspace_sz);
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());
            h_device_buf.SetZero();

            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t num_byte =
                sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                (sizeof(D0DataType) + sizeof(D1DataType) + sizeof(HDataType)) * M * N +
                (sizeof(GammaDataType) + sizeof(BetaDataType)) * N;

            float gb_per_sec = num_byte / 1.E6 / ave_time;

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
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            a_device_buf.GetDeviceBuffer(),
            b_device_buf.GetDeviceBuffer(),
            {d0_device_buf.GetDeviceBuffer(), d1_device_buf.GetDeviceBuffer()},
            gamma_device_buf.GetDeviceBuffer(),
            beta_device_buf.GetDeviceBuffer(),
            h_device_buf.GetDeviceBuffer(),
            M,
            N,
            K,
            StrideA,
            StrideB,
            {StrideD0, StrideD1},
            StrideH,
            epsilon,
            a_element_op,
            b_element_op,
            cde_element_op,
            h_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
            SimpleDeviceMem workspace_dev(workspace_sz);
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());
            h_device_buf.SetZero();

            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}