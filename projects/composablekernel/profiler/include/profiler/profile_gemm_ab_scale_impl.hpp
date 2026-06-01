// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_ab_scale.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

namespace ck {
namespace profiler {

template <typename A0DataType,
          typename A1DataType,
          typename B0DataType,
          typename B1DataType,
          typename ComputeDataType,
          typename AccDataType,
          typename EDataType,
          index_t ScaleBlockM,
          index_t ScaleBlockN,
          index_t ScaleBlockK,
          typename ALayout,
          typename BLayout,
          typename ELayout>
bool profile_gemm_ab_scale_impl(int do_verification,
                                int init_method,
                                bool do_log,
                                bool time_kernel,
                                int M,
                                int N,
                                int K,
                                int StrideA,
                                int StrideB,
                                int StrideE,
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
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    ck::index_t Scale_Stride_AM = ck::is_same_v<ALayout, tensor_layout::gemm::RowMajor>
                                      ? ((K + ScaleBlockK - 1) / ScaleBlockK)
                                      : ((M + ScaleBlockM - 1) / ScaleBlockM);
    ck::index_t Scale_Stride_BN = ck::is_same_v<BLayout, ck::tensor_layout::gemm::ColumnMajor>
                                      ? ((K + ScaleBlockK - 1) / ScaleBlockK)
                                      : ((N + ScaleBlockN - 1) / ScaleBlockN);

    Tensor<A0DataType> a0_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<A1DataType> a1_m_k(f_host_tensor_descriptor((M + ScaleBlockM - 1) / ScaleBlockM,
                                                       (K + ScaleBlockK - 1) / ScaleBlockK,
                                                       Scale_Stride_AM,
                                                       ALayout{}));
    Tensor<B0DataType> b0_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<B1DataType> b1_k_n(f_host_tensor_descriptor((K + ScaleBlockK - 1) / ScaleBlockK,
                                                       (N + ScaleBlockN - 1) / ScaleBlockN,
                                                       Scale_Stride_BN,
                                                       BLayout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    int total_gemm_needed =
        a0_m_k.GetElementSpaceSizeInBytes() + b0_k_n.GetElementSpaceSizeInBytes() +
        a1_m_k.GetElementSpaceSizeInBytes() + b1_k_n.GetElementSpaceSizeInBytes();
    int rotating_count = std::max(
        1,
        std::min(n_iter,
                 static_cast<int>(std::ceil(static_cast<double>(rotating) / total_gemm_needed))));

    std::cout << "a0_m_k: " << a0_m_k.mDesc << std::endl;
    std::cout << "a1_m_k: " << a1_m_k.mDesc << std::endl;
    std::cout << "b0_k_n: " << b0_k_n.mDesc << std::endl;
    std::cout << "b1_k_n: " << b1_k_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_device_result.mDesc << std::endl;
    std::cout << "rotating count: " << rotating_count << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        a1_m_k.GenerateTensorValue(GeneratorTensor_2<A1DataType>{-1, 2});
        b1_k_n.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-1, 2});
        break;
    default:
        a0_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{-0.5, 0.5});
        b0_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        a1_m_k.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
    }

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    using CElementOp = PassThrough;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpaceSize());
    DeviceMem a1_device_buf(sizeof(A1DataType) * a1_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a0_device_buf.ToDevice(a0_m_k.mData.data());
    b0_device_buf.ToDevice(b0_k_n.mData.data());
    a1_device_buf.ToDevice(a1_m_k.mData.data());
    b1_device_buf.ToDevice(b1_k_n.mData.data());

    using DeviceOp = ck::tensor_operation::device::DeviceGemmMultipleD_ABScale<ALayout,
                                                                               BLayout,
                                                                               ck::Tuple<>,
                                                                               ELayout,
                                                                               A0DataType,
                                                                               A1DataType,
                                                                               B0DataType,
                                                                               B1DataType,
                                                                               ck::Tuple<>,
                                                                               EDataType,
                                                                               ScaleBlockM,
                                                                               ScaleBlockN,
                                                                               ScaleBlockK,
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
        Tensor<AccDataType> c_m_n({M, N});
        Tensor<float> a_m_k({M, K});
        Tensor<float> b_k_n({K, N});

        for(int m = 0; m < M; m++)
        {
            for(int k = 0; k < K; k++)
            {
                a_m_k(m, k) = ck::type_convert<float>(a0_m_k(m, k)) *
                              a1_m_k(m / ScaleBlockM, k / ScaleBlockK);
            }
        }

        for(int n = 0; n < N; n++)
        {
            for(int k = 0; k < K; k++)
            {
                b_k_n(k, n) = ck::type_convert<float>(b0_k_n(k, n)) *
                              b1_k_n(k / ScaleBlockK, n / ScaleBlockN);
            }
        }

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<float,
                                                                                float,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                PassThrough,
                                                                                float>;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument =
            ref_gemm.MakeArgument(a_m_k, b_k_n, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                e_m_n_host_result(m, n) = ck::type_convert<EDataType>(c_m_n(m, n));
            }
        }
    }

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device GEMM instances
    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr =
            op_ptr->MakeArgumentPointer(static_cast<A0DataType*>(a0_device_buf.GetDeviceBuffer()),
                                        static_cast<B0DataType*>(b0_device_buf.GetDeviceBuffer()),
                                        std::array<const void*, 0>{},
                                        static_cast<EDataType*>(c_device_buf.GetDeviceBuffer()),
                                        M,
                                        N,
                                        K,
                                        StrideA,
                                        StrideB,
                                        std::array<ck::index_t, 0>{},
                                        StrideE,
                                        a1_device_buf.GetDeviceBuffer(),
                                        b1_device_buf.GetDeviceBuffer(),
                                        a_element_op,
                                        b_element_op,
                                        c_element_op);
        op_ptr->SetKBatch(argument_ptr.get(), KBatch);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {

            // re-init C to zero before profiling next kernel
            c_device_buf.SetZero();

            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false, 0, n_warmup, n_iter});

            if(do_verification)
            {
                c_device_buf.FromDevice(e_m_n_device_result.mData.data());

#if defined CK_ENABLE_FP8
                // set softer tolerances for fp8
                if constexpr(is_same_v<A0DataType, f8_t> || is_same_v<B0DataType, f8_t> ||
                             is_same_v<EDataType, f8_t>)
                {
                    std::string msg = "Error: Incorrect results!";
                    double rtol     = 5e-2;
                    double atol     = 5e-2;
                    pass            = pass & ck::utils::check_err(
                                      e_m_n_device_result, e_m_n_host_result, msg, rtol, atol);
                }
                else
                {
#endif
                    pass = pass & ck::utils::check_err(e_m_n_device_result, e_m_n_host_result);
#if defined CK_ENABLE_FP8
                }
#endif

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a0_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b0_k_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host  : ", e_m_n_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "c_device: ", e_m_n_device_result.mData, ",")
                        << std::endl;
                }
            }

            std::string op_name = op_ptr->GetTypeString();

            float ave_time = invoker_ptr->Run(
                argument_ptr.get(),
                StreamConfig{
                    nullptr, time_kernel, 0, n_warmup, n_iter, rotating_count > 1, rotating_count});

            std::size_t flop = std::size_t(2) * M * N * K;

            std::size_t num_btype =
                sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N + sizeof(EDataType) * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << ", KBatch " << KBatch << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    }

    if constexpr(is_same<EDataType, float>::value)
    {
        std::cout << "Best Perf for datatype = f32";
    }
    else if constexpr(is_same<EDataType, half_t>::value)
    {
        std::cout << "Best Perf for datatype = f16";
    }
    else if constexpr(is_same<EDataType, bhalf_t>::value)
    {
        std::cout << "Best Perf for datatype = bf16";
    }
    else if constexpr(is_same<EDataType, int8_t>::value)
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
              << " StrideB = " << StrideB << " StrideE = " << StrideE << " : " << best_ave_time
              << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec << " GB/s, "
              << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
