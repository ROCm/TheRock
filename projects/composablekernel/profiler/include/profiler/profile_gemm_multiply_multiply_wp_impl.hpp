// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_b_preshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply_wp.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "profiler/common.hpp"

namespace ck {
namespace profiler {

template <typename InOutDataType>
void preShuffleBuffer(const InOutDataType* src, InOutDataType* dst, int N, int K, int NXdl)
{
    int KPack = 16;
    int NLane = NXdl;
    int KLane = ck::get_warp_size() / NLane;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K + k];
        }
    }
}

template <typename ADataType,
          typename BDataType,
          typename ComputeDataType,
          typename AccDataType,
          typename D0DataType,
          typename D1DataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename ELayout>
bool profile_gemm_multiply_multiply_weight_preshuffle_impl(int do_verification,
                                                           int init_method,
                                                           bool do_log,
                                                           bool time_kernel,
                                                           int M,
                                                           int N,
                                                           int K,
                                                           int StrideA,
                                                           int StrideB,
                                                           int StrideD0,
                                                           int StrideD1,
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

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<BDataType> b_preshuffled_mfma16(
        f_host_tensor_descriptor(K, N, StrideB, BLayout{})); // use layout only for size
    Tensor<BDataType> b_preshuffled_mfma32(
        f_host_tensor_descriptor(K, N, StrideB, BLayout{})); // use layout only for size
    Tensor<D0DataType> d0_m_n(f_host_tensor_descriptor(M, N, StrideD0, D0Layout{}));
    Tensor<D1DataType> d1_m_n(f_host_tensor_descriptor(M, N, StrideD1, D1Layout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    // Update strides based on tensor properties if they are <= 0
    auto get_stride = [](auto& tensor, auto layout, ck::index_t current_stride) -> ck::index_t {
        if(current_stride <= 0)
        {
            if constexpr(std::is_same_v<decltype(layout), tensor_layout::gemm::RowMajor>)
            {
                return tensor.GetStrides()[0];
            }
            else
            {
                return tensor.GetStrides()[1];
            }
        }
        return current_stride;
    };

    StrideA  = get_stride(a_m_k, ALayout{}, StrideA);
    StrideB  = get_stride(b_k_n, BLayout{}, StrideB);
    StrideD0 = get_stride(d0_m_n, D0Layout{}, StrideD0);
    StrideD1 = get_stride(d1_m_n, D1Layout{}, StrideD1);
    StrideE  = get_stride(e_m_n_host_result, ELayout{}, StrideE);

    int total_gemm_needed =
        a_m_k.GetElementSpaceSizeInBytes() + b_k_n.GetElementSpaceSizeInBytes() +
        d0_m_n.GetElementSpaceSizeInBytes() + d1_m_n.GetElementSpaceSizeInBytes();
    int rotating_count = std::max(
        1,
        std::min(n_iter,
                 static_cast<int>(std::ceil(static_cast<double>(rotating) / total_gemm_needed))));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "d0_m_n: " << d0_m_n.mDesc << std::endl;
    std::cout << "d1_m_n: " << d1_m_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_device_result.mDesc << std::endl;
    std::cout << "rotating count: " << rotating_count << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-1, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-1, 2});
        d0_m_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{-2, 2});
        d1_m_n.GenerateTensorValue(GeneratorTensor_2<D1DataType>{-2, 2});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d0_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{0.0, 1.0});
        d1_m_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{0.0, 1.0});
    }

    preShuffleBuffer(b_k_n.mData.data(), b_preshuffled_mfma16.mData.data(), N, K, 16);
    preShuffleBuffer(b_k_n.mData.data(), b_preshuffled_mfma32.mData.data(), N, K, 32);

    using PassThrough      = ck::tensor_operation::element_wise::PassThrough;
    using MultiplyMultiply = ck::tensor_operation::element_wise::MultiplyMultiply;

    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    using CElementOp = MultiplyMultiply;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf_mfma16(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf_mfma32(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_m_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_m_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf_mfma16.ToDevice(b_preshuffled_mfma16.mData.data());
    b_device_buf_mfma32.ToDevice(b_preshuffled_mfma32.mData.data());

    d0_device_buf.ToDevice(d0_m_n.mData.data());
    d1_device_buf.ToDevice(d1_m_n.mData.data());

    using DeviceOp = ck::tensor_operation::device::DeviceGemmMultipleDSplitKBPreShuffle<
        ALayout,
        BLayout,
        ck::Tuple<D0Layout, D1Layout>,
        ELayout,
        ADataType,
        BDataType,
        ck::Tuple<D0DataType, D1DataType>,
        EDataType,
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

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                PassThrough,
                                                                                ComputeDataType>;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument =
            ref_gemm.MakeArgument(a_m_k, b_k_n, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                c_element_op(e_m_n_host_result(m, n), c_m_n(m, n), d0_m_n(m, n), d1_m_n(m, n));
            }
        }
    }

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    float best_kbatch     = 0;

    // profile device GEMM instances
    for(auto& op_ptr : op_ptrs)
    {
        int NPerXdl = op_ptr->GetPreShuffleParameters();

        std::vector<int> kbatch_list = {1, 2, 4, 8};

        if(KBatch > 0)
        {
            kbatch_list = {KBatch};
        }

        for(std::size_t i = 0; i < kbatch_list.size(); i++)
        {
            auto kbatch_curr = kbatch_list[i];

            auto argument_ptr = op_ptr->MakeArgumentPointer(
                static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                static_cast<BDataType*>(NPerXdl == 16 ? b_device_buf_mfma16.GetDeviceBuffer()
                                                      : b_device_buf_mfma32.GetDeviceBuffer()),
                std::array<const void*, 2>{d0_device_buf.GetDeviceBuffer(),
                                           d1_device_buf.GetDeviceBuffer()},
                static_cast<EDataType*>(c_device_buf.GetDeviceBuffer()),
                M,
                N,
                K,
                StrideA,
                StrideB,
                std::array<ck::index_t, 2>{StrideD0, StrideD1},
                StrideE,
                kbatch_curr,
                a_element_op,
                b_element_op,
                c_element_op);

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                c_device_buf.SetZero();

                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});

                if(do_verification)
                {
                    c_device_buf.FromDevice(e_m_n_device_result.mData.data());

#if defined CK_ENABLE_FP8 || defined CK_ENABLE_INT8
                    // set softer tolerances for fp8
                    if constexpr((is_same_v<ADataType, f8_t> || is_same_v<BDataType, f8_t> ||
                                  is_same_v<EDataType, f8_t>) ||
                                 (is_same_v<ADataType, int8_t> || is_same_v<BDataType, int8_t> ||
                                  is_same_v<EDataType, int8_t>))
                    {
                        std::string msg = "Error: Incorrect results!";
                        double rtol     = get_rtol<EDataType>();
                        double atol     = get_atol<EDataType>();
                        pass            = pass & ck::utils::check_err(
                                          e_m_n_device_result, e_m_n_host_result, msg, rtol, atol);
                    }
                    else
                    {
#endif
                        pass = pass & ck::utils::check_err(e_m_n_device_result, e_m_n_host_result);
#if defined CK_ENABLE_FP8 || defined CK_ENABLE_INT8
                    }
#endif

                    if(do_log)
                    {
                        LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",") << std::endl;
                        LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",") << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "c_host  : ", e_m_n_host_result.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "c_device: ", e_m_n_device_result.mData, ",")
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
                                        sizeof(D0DataType) * M * N + sizeof(D1DataType) * M * N +
                                        sizeof(EDataType) * M * N;

                float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

                float gb_per_sec = num_btype / 1.E6 / ave_time;

                std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops
                          << " TFlops, " << gb_per_sec << " GB/s, " << op_name << ", KBatch "
                          << kbatch_curr << std::endl;

                if(tflops > best_tflops && ave_time > 1e-10)
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
              << " StrideB = " << StrideB << " StrideE = " << StrideE << " KBatch = " << best_kbatch
              << " : " << best_ave_time << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec
              << " GB/s, " << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
