// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iomanip>
#include <iostream>
#include <vector>
#include <random>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_fixed_nk.hpp"

using F8  = ck::f8_t;
using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType  = F16;
using BDataType  = F8;
using DsDataType = ck::Tuple<>;
using EDataType  = F16;

using ALayout  = Row;
using BLayout  = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

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
    std::vector<int> Ms, Ns, Ks, StrideAs, StrideBs, StrideEs;

    int sum_of_m = 0;

    const int group_count = 16;

    for(int i = 0; i < group_count; ++i)
    {
        Ms.push_back(256 + 256 * i);
        Ns.push_back(128 + 128 * i);
        Ks.push_back(128 + 64 * i);

        StrideAs.push_back(std::is_same<Row, ALayout>::value ? Ks[i] : Ms[i]);
        StrideBs.push_back(std::is_same<Row, BLayout>::value ? Ns[i] : Ks[i]);
        StrideEs.push_back(std::is_same<Row, ELayout>::value ? Ns[i] : Ms[i]);

        sum_of_m += Ms[i];
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

    std::vector<SimpleDeviceMem> a_dev_bufs, b_dev_bufs, e_dev_bufs;

    a_dev_bufs.reserve(group_count);
    b_dev_bufs.reserve(group_count);
    e_dev_bufs.reserve(group_count);

    std::vector<void*> p_e;

    p_e.reserve(group_count);

    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;

    gemm_descs.reserve(group_count);

    std::vector<ck::tensor_operation::device::GroupedGemmKernelArgument<1>>
        grouped_gemm_kernel_args_;
    grouped_gemm_kernel_args_.reserve(group_count);

    for(int i = 0; i < group_count; ++i)
    {
        a_dev_bufs.emplace_back(sizeof(ADataType) *
                                f_matrix_space_size(Ms[i], Ks[i], StrideAs[i], ALayout{}));
        b_dev_bufs.emplace_back(sizeof(BDataType) *
                                f_matrix_space_size(Ks[i], Ns[i], StrideBs[i], BLayout{}));
        e_dev_bufs.emplace_back(sizeof(EDataType) *
                                f_matrix_space_size(Ms[i], Ns[i], StrideEs[i], ELayout{}));

        gemm_descs.push_back({sum_of_m, Ns[i], Ks[i], 1, StrideBs[i], 1, {0}});

        p_e.push_back(e_dev_bufs[i].GetDeviceBuffer());

        grouped_gemm_kernel_args_.push_back({a_dev_bufs[i].GetDeviceBuffer(),
                                             b_dev_bufs[i].GetDeviceBuffer(),
                                             {},
                                             e_dev_bufs[i].GetDeviceBuffer(),
                                             Ms[i],
                                             Ns[i],
                                             Ks[i],
                                             StrideAs[i],
                                             StrideBs[i],
                                             {},
                                             StrideEs[i]});
    }

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedGemmFixedNK<ALayout,
                                                                            BLayout,
                                                                            DsLayout,
                                                                            ELayout,
                                                                            ADataType,
                                                                            BDataType,
                                                                            DsDataType,
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
    const auto cde_element_op = CDEElementOp{};

    std::string best_op_name;
    bool found            = false;
    int best_op_id        = -1;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    std::vector<const void*> p_a = {}, p_b = {};
    std::vector<std::array<const void*, 0>> p_ds = {};

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            p_a, p_b, p_ds, p_e, gemm_descs, a_element_op, b_element_op, cde_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        SimpleDeviceMem grouped_gemm_kernel_args_dev(
            op_ptr->GetDeviceKernelArgSize(argument_ptr.get()));

        SimpleDeviceMem grouped_gemm_workspace_dev(op_ptr->GetWorkSpaceSize(argument_ptr.get()));

        std::string op_name = op_ptr->GetTypeString();

        hipGetErrorString(hipMemcpy(grouped_gemm_kernel_args_dev.GetDeviceBuffer(),
                                    grouped_gemm_kernel_args_.data(),
                                    op_ptr->GetDeviceKernelArgSize(argument_ptr.get()),
                                    hipMemcpyHostToDevice));

        op_ptr->SetWorkSpacePointer(argument_ptr.get(),
                                    grouped_gemm_workspace_dev.GetDeviceBuffer());

        op_ptr->SetDeviceKernelArgs(argument_ptr.get(),
                                    grouped_gemm_kernel_args_dev.GetDeviceBuffer());

        op_ptr->SetKBatch(argument_ptr.get(), 16);

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop = 0, num_btype = 0;
            for(std::size_t j = 0; j < gemm_descs.size(); ++j)
            {
                flop += std::size_t(2) * Ms[j] * Ns[j] * Ks[j];

                num_btype += sizeof(ADataType) * Ms[j] * Ks[j] + sizeof(BDataType) * Ks[j] * Ns[j] +
                             sizeof(EDataType) * Ms[j] * Ns[j];
            }

            float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
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

    return 0;
}
