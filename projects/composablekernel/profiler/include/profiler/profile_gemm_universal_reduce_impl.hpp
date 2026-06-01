// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3r1.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3r1.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_universal_reduce.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout>
bool profile_gemm_universal_reduce_impl(int do_verification,
                                        int init_method,
                                        bool do_log,
                                        bool time_kernel,
                                        int M,
                                        int N,
                                        int K,
                                        int StrideA,
                                        int StrideB,
                                        int StrideC,
                                        int KBatch,
                                        int n_warmup,
                                        int n_iter,
                                        uint64_t rotating = 0)
{
    bool pass = true;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz}, layout);
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride}, layout);
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    int total_gemm_needed = a_m_k.GetElementSpaceSizeInBytes() + b_k_n.GetElementSpaceSizeInBytes();
    int rotating_count    = std::max(
        1,
        std::min(n_iter,
                 static_cast<int>(std::ceil(static_cast<double>(rotating) / total_gemm_needed))));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_device_result.mDesc << std::endl;
    std::cout << "rotating count: " << rotating_count << std::endl;

    switch(init_method)
    {
    case 0:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        break;
    case 3:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    }

    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

    using DeviceOp = ck::tensor_operation::device::DeviceGemmV2R1<ALayout,
                                                                  BLayout,
                                                                  DsLayout,
                                                                  CLayout,
                                                                  ADataType,
                                                                  BDataType,
                                                                  DsDataType,
                                                                  CDataType,
                                                                  AElementOp,
                                                                  BElementOp,
                                                                  CElementOp>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    // Run reference GEMM
    if(do_verification)
    {
        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                CElementOp>;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);
    }

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    float best_kbatch     = 0;

    // profile device GEMM instances
    for(auto& op_ptr : op_ptrs)
    {
        std::vector<int> kbatch_list = {1, 2, 4, 8, 12, 16, 19, 20, 32, 38};

        if(KBatch > 0)
        {
            kbatch_list = {KBatch};
        }

        for(std::size_t i = 0; i < kbatch_list.size(); i++)
        {
            auto kbatch_curr = kbatch_list[i];

            auto argument_ptr =
                op_ptr->MakeArgumentPointer(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                            static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                            {},
                                            static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                            M,
                                            N,
                                            K,
                                            StrideA,
                                            StrideB,
                                            {},
                                            StrideC,
                                            kbatch_curr,
                                            a_element_op,
                                            b_element_op,
                                            c_element_op);

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {

                DeviceMem gemm_workspace_dev(op_ptr->GetWorkSpaceSize(argument_ptr.get()));
                op_ptr->SetWorkSpacePointer(
                    argument_ptr.get(), gemm_workspace_dev.GetDeviceBuffer(), StreamConfig{});

                // re-init C to zero before profiling next kernel
                c_device_buf.SetZero();

                invoker_ptr->Run(argument_ptr.get(),
                                 StreamConfig{nullptr, false, 0, n_warmup, n_iter});

                if(do_verification)
                {
                    c_device_buf.FromDevice(c_m_n_device_result.mData.data());

                    pass = pass & ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);

                    if(do_log)
                    {
                        LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",") << std::endl;
                        LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",") << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "c_host  : ", c_m_n_host_result.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "c_device: ", c_m_n_device_result.mData, ",")
                            << std::endl;
                    }
                }

                std::string op_name = op_ptr->GetTypeString();

                float ave_time = invoker_ptr->Run(argument_ptr.get(),
                                                  StreamConfig{nullptr,
                                                               time_kernel,
                                                               0,
                                                               n_warmup,
                                                               n_iter,
                                                               rotating_count > 1,
                                                               rotating_count});

                std::size_t flop = std::size_t(2) * M * N * K;

                std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                        sizeof(CDataType) * M * N;

                float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

                float gb_per_sec = num_btype / 1.E6 / ave_time;

                std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops
                          << " TFlops, " << gb_per_sec << " GB/s, " << op_name << ", KBatch "
                          << kbatch_curr << std::endl;

#if defined CK_ENABLE_FP8
                // set softer tolerances for fp8
                if constexpr(is_same_v<ADataType, f8_t> || is_same_v<BDataType, f8_t> ||
                             is_same_v<CDataType, f8_t>)
                {
                    std::string msg = "Error: Incorrect results!";
                    double rtol     = 1e-1;
                    double atol     = 1e-1;
                    pass            = pass & ck::utils::check_err(
                                      c_m_n_device_result, c_m_n_host_result, msg, rtol, atol);
                }
                else
                {
#endif
                    pass = pass & ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);
#if defined CK_ENABLE_FP8
                }
#endif

                if(tflops > best_tflops)
                {
                    best_op_name    = op_name;
                    best_tflops     = tflops;
                    best_ave_time   = ave_time;
                    best_gb_per_sec = gb_per_sec;
                    best_kbatch     = kbatch_curr;
                }
            }
            else
            {
                std::cout << op_ptr->GetTypeString() << " does not support this problem"
                          << std::endl;
            }
        }
    }

    if constexpr(is_same<CDataType, float>::value)
    {
        std::cout << "Best Perf for datatype = f32";
    }
    else if constexpr(is_same<CDataType, half_t>::value)
    {
        std::cout << "Best Perf for datatype = f16";
    }
    else if constexpr(is_same<CDataType, bhalf_t>::value)
    {
        std::cout << "Best Perf for datatype = bf16";
    }
    else if constexpr(is_same<CDataType, int8_t>::value)
    {
        std::cout << "Best Perf for datatype = int8";
    }

    if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value)
    {
        std::cout << " ALayout =  RowMajor";
    }
    else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value)
    {
        std::cout << " ALayout =  ColumnMajor";
    }

    if constexpr(is_same<BLayout, tensor_layout::gemm::RowMajor>::value)
    {
        std::cout << " BLayout =  RowMajor";
    }
    else if constexpr(is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value)
    {
        std::cout << " BLayout =  ColumnMajor";
    }

    std::cout << " M = " << M << " N = " << N << " K = " << K << " StrideA = " << StrideA
              << " StrideB = " << StrideB << " StrideC = " << StrideC << " KBatch = " << best_kbatch
              << " : " << best_ave_time << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec
              << " GB/s, " << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
