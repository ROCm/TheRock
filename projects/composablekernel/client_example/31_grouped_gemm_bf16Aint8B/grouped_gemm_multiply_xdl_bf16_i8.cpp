// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <iomanip>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_tile_loop_multiply.hpp"

#include "ck/host_utility/hip_check_error.hpp"

using ::ck::hip_check_error;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using I8   = int8_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = BF16;
using B0DataType       = I8;
using B1DataType       = BF16;
using AccDataType      = F32;
using CShuffleDataType = BF16;
using D0DataType       = BF16;
using DsDataType       = ck::Tuple<B1DataType>;
using EDataType        = BF16;

using A0Layout = Row;
using B0Layout = Row;
using B1Layout = B0Layout;
using D0Layout = Row;
using DsLayout = ck::Tuple<B1Layout>;
using ELayout  = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Multiply    = ck::tensor_operation::element_wise::Multiply;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = Multiply;

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

struct ProblemSize final
{
    std::vector<ck::index_t> Ms;
    std::vector<ck::index_t> Ns;
    std::vector<ck::index_t> Ks;

    std::vector<ck::index_t> stride_As;
    std::vector<ck::index_t> stride_Bs;
    std::vector<ck::index_t> stride_Cs;

    ck::index_t group_count;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    int k_batch          = 1;
};

bool run_grouped_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    auto group_count = problem_size.group_count;

    // GEMM shape
    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;

    gemm_descs.reserve(group_count);

    int sum_of_m = 0;

    using DeviceMemPtr = std::unique_ptr<SimpleDeviceMem>;

    std::vector<DeviceMemPtr> a0_tensors_device, b0_tensors_device, b1_tensors_device,
        c_tensors_device;

    a0_tensors_device.reserve(group_count);
    b0_tensors_device.reserve(group_count);
    b1_tensors_device.reserve(group_count);
    c_tensors_device.reserve(group_count);

    std::size_t flop = 0, num_btype = 0;

    for(int i = 0; i < group_count; i++)
    {
        sum_of_m += problem_size.Ms[i];
    }

    constexpr ck::index_t NumDTensor = 1;

    using GroupedGemmKernelArgument =
        ck::tensor_operation::device::GroupedGemmKernelArgument<NumDTensor>;

    std::vector<GroupedGemmKernelArgument> grouped_gemm_kernel_args_;
    grouped_gemm_kernel_args_.reserve(group_count);

    for(int i = 0; i < group_count; i++)
    {
        a0_tensors_device.emplace_back(std::make_unique<SimpleDeviceMem>(
            sizeof(A0DataType) * problem_size.Ms[i] * problem_size.Ks[i]));

        b0_tensors_device.emplace_back(std::make_unique<SimpleDeviceMem>(
            sizeof(B0DataType) * problem_size.Ns[i] * problem_size.Ks[i]));

        b1_tensors_device.emplace_back(
            std::make_unique<SimpleDeviceMem>(sizeof(B1DataType) * problem_size.Ns[i]));

        c_tensors_device.emplace_back(std::make_unique<SimpleDeviceMem>(
            sizeof(EDataType) * problem_size.Ms[i] * problem_size.Ns[i]));

        gemm_descs.push_back({problem_size.Ms[i],
                              problem_size.Ns[i],
                              problem_size.Ks[i],
                              problem_size.stride_As[i],
                              problem_size.stride_Bs[i],
                              problem_size.stride_Cs[i],
                              {0}});

        grouped_gemm_kernel_args_.push_back({a0_tensors_device[i]->GetDeviceBuffer(),
                                             b0_tensors_device[i]->GetDeviceBuffer(),
                                             {b1_tensors_device[i]->GetDeviceBuffer()},
                                             c_tensors_device[i]->GetDeviceBuffer(),
                                             problem_size.Ms[i],
                                             problem_size.Ns[i],
                                             problem_size.Ks[i],
                                             problem_size.stride_As[i],
                                             problem_size.stride_Bs[i],
                                             {0},
                                             problem_size.stride_Cs[i]});
    }

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedGemmTileLoop<A0Layout,
                                                                             B0Layout,
                                                                             DsLayout,
                                                                             ELayout,
                                                                             A0DataType,
                                                                             B0DataType,
                                                                             DsDataType,
                                                                             EDataType,
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

        std::vector<const void*> p_As                         = {};
        std::vector<const void*> p_Bs                         = {};
        std::vector<std::array<const void*, NumDTensor>> p_Ds = {};
        std::vector<void*> p_Cs                               = {};

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            p_As, p_Bs, p_Ds, p_Cs, gemm_descs, a_element_op, b_element_op, cde_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            SimpleDeviceMem gemm_kernel_args_dev(
                op_ptr->GetDeviceKernelArgSize(argument_ptr.get()));
            hip_check_error(hipMemcpy(gemm_kernel_args_dev.GetDeviceBuffer(),
                                      grouped_gemm_kernel_args_.data(),
                                      op_ptr->GetDeviceKernelArgSize(argument_ptr.get()),
                                      hipMemcpyHostToDevice));

            op_ptr->SetDeviceKernelArgs(argument_ptr.get(), gemm_kernel_args_dev.GetDeviceBuffer());

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true, 0, 20, 50});

            std::size_t flop = std::size_t(2) * sum_of_m * problem_size.Ns[0] * problem_size.Ks[0];

            std::size_t num_btype = sizeof(A0DataType) * sum_of_m * problem_size.Ks[0] +
                                    sizeof(B0DataType) * problem_size.Ks[0] * problem_size.Ns[0] +
                                    sizeof(EDataType) * sum_of_m * problem_size.Ns[0];

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

    return true;
}

int main(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    problem_size.group_count = 16;

    for(int i = 0; i < problem_size.group_count; i++)
    {
        problem_size.Ms.push_back(1 + rand() % 1024);
        problem_size.Ns.push_back(4096);
        problem_size.Ks.push_back(4096);

        problem_size.stride_As.push_back(problem_size.Ks[i]);
        problem_size.stride_Bs.push_back(problem_size.Ns[i]);
        problem_size.stride_Cs.push_back(problem_size.Ns[i]);

        std::cout << " M = " << problem_size.Ms[i] << " N = " << problem_size.Ns[i] << " K "
                  << problem_size.Ks[i] << std::endl;
    }

    return !run_grouped_gemm(problem_size, config);
}
