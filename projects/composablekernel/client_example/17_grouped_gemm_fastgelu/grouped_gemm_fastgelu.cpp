// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iomanip>
#include <iostream>
#include <vector>
#include <random>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_fastgelu.hpp"

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using FastGelu    = ck::tensor_operation::element_wise::FastGelu;

using ADataType  = F16;
using BDataType  = F16;
using DsDataType = ck::Tuple<>;
using EDataType  = F16;

using ALayout  = Row;
using BLayout  = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = FastGelu;

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
    std::mt19937 gen(19391);
    std::uniform_int_distribution<> distrib(1, 10);
    int group_count = distrib(gen);

    std::vector<int> Ms, Ns, Ks, StrideAs, StrideBs, StrideEs;

    for(int i = 0; i < group_count; ++i)
    {
        Ms.push_back(256 + 256 * distrib(gen));
        Ns.push_back(256 + 256 * distrib(gen));
        Ks.push_back(128 + 128 * distrib(gen));

        StrideAs.push_back(std::is_same<Row, ALayout>::value ? Ks[i] : Ms[i]);
        StrideBs.push_back(std::is_same<Row, BLayout>::value ? Ns[i] : Ks[i]);
        StrideEs.push_back(std::is_same<Row, ELayout>::value ? Ns[i] : Ms[i]);
    }

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

    std::vector<SimpleDeviceMem> a_dev_bufs, b_dev_bufs, e_dev_bufs;

    a_dev_bufs.reserve(group_count);
    b_dev_bufs.reserve(group_count);
    e_dev_bufs.reserve(group_count);

    std::vector<const void*> p_a, p_b;
    std::vector<void*> p_e;

    p_a.reserve(group_count);
    p_b.reserve(group_count);
    p_e.reserve(group_count);

    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;

    gemm_descs.reserve(group_count);

    for(int i = 0; i < group_count; ++i)
    {
        a_dev_bufs.emplace_back(sizeof(ADataType) *
                                f_matrix_space_size(Ms[i], Ks[i], StrideAs[i], ALayout{}));
        b_dev_bufs.emplace_back(sizeof(BDataType) *
                                f_matrix_space_size(Ks[i], Ns[i], StrideBs[i], BLayout{}));
        e_dev_bufs.emplace_back(sizeof(EDataType) *
                                f_matrix_space_size(Ms[i], Ns[i], StrideEs[i], ELayout{}));

        gemm_descs.push_back({Ms[i], Ns[i], Ks[i], StrideAs[i], StrideBs[i], StrideEs[i], {}});

        p_a.push_back(a_dev_bufs[i].GetDeviceBuffer());
        p_b.push_back(b_dev_bufs[i].GetDeviceBuffer());
        p_e.push_back(e_dev_bufs[i].GetDeviceBuffer());
    }

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedGemm<ALayout,
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

    auto p_ds = std::vector<std::array<const void*, 0>>{};

    // profile device operation instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            p_a, p_b, p_ds, p_e, gemm_descs, a_element_op, b_element_op, cde_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        SimpleDeviceMem gemm_desc_workspace(op_ptr->GetWorkSpaceSize(argument_ptr.get()));
        op_ptr->SetWorkSpacePointer(argument_ptr.get(), gemm_desc_workspace.GetDeviceBuffer());
        std::string op_name = op_ptr->GetTypeString();

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

    // run the best intance
    if(found)
    {
        auto& op_ptr = op_ptrs[best_op_id];

        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;

        auto argument_ptr = op_ptr->MakeArgumentPointer(
            p_a, p_b, p_ds, p_e, gemm_descs, a_element_op, b_element_op, cde_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        SimpleDeviceMem gemm_desc_workspace(op_ptr->GetWorkSpaceSize(argument_ptr.get()));
        op_ptr->SetWorkSpacePointer(argument_ptr.get(), gemm_desc_workspace.GetDeviceBuffer());

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
