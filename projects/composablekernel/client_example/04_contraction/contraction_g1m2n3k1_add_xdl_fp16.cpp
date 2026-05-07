// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iomanip>
#include <numeric>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/batched_gemm_bias_permute.hpp"
#include "ck/library/utility/numeric.hpp"

using F16 = ck::half_t;
using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = Add;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F16;
using DDataType        = F16;
using DsDataType       = ck::Tuple<DDataType>;
using EDataType        = F16;

static constexpr ck::index_t NumDimG = 1;
static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 3;
static constexpr ck::index_t NumDimK = 1;

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
    ck::index_t G0 = 1;

    ck::index_t M0 = 64;
    ck::index_t M1 = 256;

    ck::index_t N0 = 3;
    ck::index_t N1 = 12;
    ck::index_t N2 = 64;

    ck::index_t K0 = 768;

    // A[M0, M1, M2, K0]
    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, M0, M1, K0};
    std::vector<ck::index_t> a_gs_ms_ks_strides{M0 * M1 * K0, M1 * K0, K0, 1};
    // B[N0, N1, N2, K0]
    std::vector<ck::index_t> b_gs_ns_ks_lengths{G0, N0, N1, N2, K0};
    std::vector<ck::index_t> b_gs_ns_ks_strides{N0 * N1 * N2 * K0, N1 * N2 * K0, N2 * K0, K0, 1};

    // D[N0, M0, N1, M1, N2]
    std::vector<ck::index_t> d_gs_ms_ns_lengths{G0, M0, M1, N0, N1, N2};
    std::vector<ck::index_t> d_gs_ms_ns_strides{N0 * N1 * N2, 0, 0, N1 * N2, N2, 1};
    // E[N0 M0 N1 N2 M1]
    std::vector<ck::index_t> e_gs_ms_ns_lengths{G0, M0, M1, N0, N1, N2};
    std::vector<ck::index_t> e_gs_ms_ns_strides{
        M0 * M1 * N0 * N1 * N2, N1 * N2 * M1, 1, M0 * N1 * N2 * M1, M1 * N2, M1};

    auto f_tensor_space_size = [](auto lengths, auto strides) {
        std::size_t space_size = 1;
        for(std::size_t i = 0; i < lengths.size(); ++i)
        {
            space_size += (lengths[i] - 1) * strides[i];
        }
        return space_size;
    };

    SimpleDeviceMem a_device_buf(sizeof(ADataType) *
                                 f_tensor_space_size(a_gs_ms_ks_lengths, a_gs_ms_ks_strides));
    SimpleDeviceMem b_device_buf(sizeof(BDataType) *
                                 f_tensor_space_size(b_gs_ns_ks_lengths, b_gs_ns_ks_strides));
    SimpleDeviceMem d_device_buf(sizeof(DDataType) *
                                 f_tensor_space_size(d_gs_ms_ns_lengths, d_gs_ms_ns_strides));
    SimpleDeviceMem e_device_buf(sizeof(EDataType) *
                                 f_tensor_space_size(e_gs_ms_ns_lengths, e_gs_ms_ns_strides));

    using DeviceOp = ck::tensor_operation::device::DeviceBatchedContractionMultipleD<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        ADataType,
        BDataType,
        DsDataType,
        EDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::Add>;

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

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr =
            op_ptr->MakeArgumentPointer(a_device_buf.GetDeviceBuffer(),
                                        b_device_buf.GetDeviceBuffer(),
                                        std::array<const void*, 1>{d_device_buf.GetDeviceBuffer()},
                                        e_device_buf.GetDeviceBuffer(),
                                        a_gs_ms_ks_lengths,
                                        a_gs_ms_ks_strides,
                                        b_gs_ns_ks_lengths,
                                        b_gs_ns_ks_strides,
                                        std::array<std::vector<ck::index_t>, 1>{d_gs_ms_ns_lengths},
                                        std::array<std::vector<ck::index_t>, 1>{d_gs_ms_ns_strides},
                                        e_gs_ms_ns_lengths,
                                        e_gs_ms_ns_strides,
                                        a_element_op,
                                        b_element_op,
                                        cde_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            ck::index_t M = ck::accumulate_n<ck::index_t>(
                e_gs_ms_ns_lengths.begin() + NumDimG, NumDimM, 1, std::multiplies<>{});

            ck::index_t N = ck::accumulate_n<ck::index_t>(
                e_gs_ms_ns_lengths.begin() + NumDimG + NumDimM, NumDimN, 1, std::multiplies<>{});

            ck::index_t K = ck::accumulate_n<ck::index_t>(
                a_gs_ms_ks_lengths.begin() + NumDimG + NumDimM, NumDimK, 1, std::multiplies<>{});

            std::size_t flop      = std::size_t(2) * M * N * K;
            std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                    sizeof(DDataType) * M * N + sizeof(EDataType) * M * N;

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

    return 0;
}
