// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <iomanip>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_multi_abd.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using I8   = int8_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = BF16;
using AsDataType       = ck::Tuple<A0DataType>;
using B0DataType       = I8;
using B1DataType       = BF16;
using BsDataType       = ck::Tuple<B0DataType, B1DataType>;
using AccDataType      = F32;
using CShuffleDataType = BF16;
using DsDataType       = ck::Tuple<>;
using EDataType        = BF16;

using A0Layout = Row;
using AsLayout = ck::Tuple<A0Layout>;
using B0Layout = Row;
using B1Layout = B0Layout;
using BsLayout = ck::Tuple<B0Layout, B1Layout>;
using D0Layout = Row;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using Multiply    = ck::tensor_operation::element_wise::Multiply;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;

using AElementOp   = PassThrough;
using BElementOp   = Multiply;
using CDEElementOp = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

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

// clang-format on
int main(int argc, char* argv[])
{
    // GEMM shape
    ck::index_t M = 4096;
    ck::index_t N = 768;
    ck::index_t K = 6144;

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideE = N;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 7)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);

        StrideA = std::stoi(argv[4]);
        StrideB = std::stoi(argv[5]);
        StrideE = std::stoi(argv[6]);
    }
    else
    {
        printf("arg1 to 7: M, N, K, StrideA, StrideB, StrideE\n");
        exit(0);
    }

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

    SimpleDeviceMem a0_device_buf(sizeof(A0DataType) *
                                  f_matrix_space_size(M, K, StrideA, A0Layout{}));
    SimpleDeviceMem b0_device_buf(sizeof(B0DataType) *
                                  f_matrix_space_size(K, N, StrideB, B0Layout{}));
    SimpleDeviceMem b1_device_buf(sizeof(B1DataType) * f_matrix_space_size(K, N, 0, B1Layout{}));
    SimpleDeviceMem e_device_buf(sizeof(EDataType) * f_matrix_space_size(M, N, StrideE, ELayout{}));

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumATensor = 1;
    constexpr ck::index_t NumBTensor = 2;
    constexpr ck::index_t NumDTensor = 0;

    using DeviceOp = ck::tensor_operation::device::DeviceGemmMultipleABD<AsLayout,
                                                                         BsLayout,
                                                                         DsLayout,
                                                                         Row,
                                                                         AsDataType,
                                                                         BsDataType,
                                                                         DsDataType,
                                                                         BF16,
                                                                         AElementOp,
                                                                         BElementOp,
                                                                         CDEElementOp>;

    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    bool found            = false;
    int best_op_id        = -1;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            std::array<const void*, NumATensor>{a0_device_buf.GetDeviceBuffer()},
            std::array<const void*, NumBTensor>{b0_device_buf.GetDeviceBuffer(),
                                                b1_device_buf.GetDeviceBuffer()},
            std::array<const void*, NumDTensor>{},
            e_device_buf.GetDeviceBuffer(),
            M,
            N,
            K,
            std::array<ck::index_t, NumATensor>{StrideA},
            std::array<ck::index_t, NumBTensor>{StrideB, 0},
            std::array<ck::index_t, NumDTensor>{},
            StrideE,
            a_element_op,
            b_element_op,
            cde_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop = std::size_t(2) * M * N * K;

            std::size_t num_btype =
                sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N + sizeof(EDataType) * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                found           = true;
                best_op_id      = i;
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    // run the best intance
    if(found)
    {
        auto& op_ptr = op_ptrs[best_op_id];

        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            std::array<const void*, NumATensor>{a0_device_buf.GetDeviceBuffer()},
            std::array<const void*, NumBTensor>{b0_device_buf.GetDeviceBuffer(),
                                                b1_device_buf.GetDeviceBuffer()},
            std::array<const void*, NumDTensor>{},
            e_device_buf.GetDeviceBuffer(),
            M,
            N,
            K,
            std::array<ck::index_t, NumATensor>{StrideA},
            std::array<ck::index_t, NumBTensor>{StrideB, 0},
            std::array<ck::index_t, NumDTensor>{},
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
