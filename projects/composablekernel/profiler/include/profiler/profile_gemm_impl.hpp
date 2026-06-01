// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>
#if defined(__unix__)
#include <unistd.h>
#endif

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/fill.hpp"

namespace ck {
namespace profiler {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType>
int profile_gemm_impl(int do_verification,
                      int init_method,
                      bool do_log,
                      bool time_kernel,
                      int M,
                      int N,
                      int K,
                      int StrideA,
                      int StrideB,
                      int StrideC,
                      int n_warmup,
                      int n_iter)
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

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_device_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0:
        ck::utils::FillConstant<ADataType>{type_convert<ADataType>(1.f)}(a_m_k);
        ck::utils::FillConstant<BDataType>{type_convert<BDataType>(1.f)}(b_k_n);
        break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n);
        break;
    default:
        ck::utils::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
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

    using DeviceOp = ck::tensor_operation::device::DeviceGemm<ALayout,
                                                              BLayout,
                                                              CLayout,
                                                              ADataType,
                                                              BDataType,
                                                              CDataType,
                                                              AElementOp,
                                                              BElementOp,
                                                              CElementOp>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    // Run reference op
    if(do_verification)
    {
        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                CElementOp>;

        auto ref_op      = ReferenceGemmInstance{};
        auto ref_invoker = ref_op.MakeInvoker();

        auto ref_argument = ref_op.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);
    }

    float best_tflops    = 0;
    int best_instance_id = 0;

    int instance_id = 0;
    // profile device op instances
    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr =
            op_ptr->MakeArgumentPointer(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                        static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                        static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                        M,
                                        N,
                                        K,
                                        StrideA,
                                        StrideB,
                                        StrideC,
                                        a_element_op,
                                        b_element_op,
                                        c_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            // re-init C to zero before profiling next kernel
            c_device_buf.SetZero();

            std::string op_name = op_ptr->GetTypeString();

            float avg_time = invoker_ptr->Run(
                argument_ptr.get(), StreamConfig{nullptr, time_kernel, 0, n_warmup, n_iter});

            std::size_t flop = std::size_t(2) * M * N * K;

            std::size_t num_btype =
                sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

            float gb_per_sec = num_btype / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_instance_id = instance_id;
                best_tflops      = tflops;
            }

            if(do_verification)
            {
                c_device_buf.FromDevice(c_m_n_device_result.mData.data());

                pass = pass & ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host  : ", c_m_n_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "c_device: ", c_m_n_device_result.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }

        instance_id++;
    }

#if defined(__unix__)
    sleep(2);
#endif

    // Run the best instance again
    {
        auto& op_ptr = op_ptrs[best_instance_id];
        auto argument_ptr =
            op_ptr->MakeArgumentPointer(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                        static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                        static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                        M,
                                        N,
                                        K,
                                        StrideA,
                                        StrideB,
                                        StrideC,
                                        a_element_op,
                                        b_element_op,
                                        c_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::string op_name = op_ptr->GetTypeString();

            float avg_time = invoker_ptr->Run(argument_ptr.get(),
                                              StreamConfig{nullptr, time_kernel, 0, 50, 200});

            std::size_t flop = std::size_t(2) * M * N * K;

            std::size_t num_btype =
                sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

            float gb_per_sec = num_btype / 1.E6 / avg_time;

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
#if defined CK_ENABLE_FP8
            else if constexpr(is_same<CDataType, f8_t>::value)
            {
                std::cout << "Best Perf for datatype = fp8";
            }
#endif

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
                      << " StrideB = " << StrideB << " StrideC = " << StrideC << " : " << avg_time
                      << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, " << op_name
                      << std::endl;
        }
    }

    return pass;
}

} // namespace profiler
} // namespace ck
