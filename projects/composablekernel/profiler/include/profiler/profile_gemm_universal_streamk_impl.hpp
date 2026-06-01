// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_streamk_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_universal_streamk.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/reference_tensor_operation/gpu/reference_gemm.hpp"

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename ComputeDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
bool profile_gemm_universal_streamk_impl(int do_verification,
                                         int init_method,
                                         bool do_log,
                                         bool time_kernel,
                                         int M,
                                         int N,
                                         int K,
                                         int StrideA,
                                         int StrideB,
                                         int StrideC,
                                         int Streamk_sel,
                                         int Grid_size,
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
    Tensor<CDataType> c_m_n_device_ref_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

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
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-1, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-1, 2});
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

    DeviceMem c_m_n_device_ref_buf(sizeof(CDataType) *
                                   c_m_n_device_ref_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

    using DeviceOp = ck::tensor_operation::device::DeviceGemm_Streamk_V2<ALayout,
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

    // Run reference GEMM
    if(do_verification)
    {
        // Use GPU validation
        using ReferenceGemmInstanceGPU =
            ck::tensor_operation::device::ReferenceGemm<ALayout,
                                                        BLayout,
                                                        CLayout,
                                                        ADataType,
                                                        BDataType,
                                                        CDataType,
                                                        AccDataType,
                                                        AElementOp,
                                                        BElementOp,
                                                        CElementOp,
                                                        ComputeDataType,
                                                        ComputeDataType>;

        auto ref_gemm_gpu     = ReferenceGemmInstanceGPU{};
        auto ref_invoker_gpu  = ref_gemm_gpu.MakeInvoker();
        auto ref_argument_gpu = ref_gemm_gpu.MakeArgument(
            static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_m_n_device_ref_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            a_element_op,
            b_element_op,
            c_element_op);

        if(ref_gemm_gpu.IsSupportedArgument(&ref_argument_gpu))
        {
            ref_invoker_gpu.Run(ref_argument_gpu, StreamConfig{nullptr, true});
            c_m_n_device_ref_buf.FromDevice(c_m_n_host_result.mData.data());
        }
        else
        {
            std::cerr << "GPU reference GEMM does not support this problem configuration so does "
                         "CPU validation."
                      << std::endl;

            // Use CPU validation

            using ReferenceGemmInstanceCPU =
                ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                          BDataType,
                                                          CDataType,
                                                          AccDataType,
                                                          AElementOp,
                                                          BElementOp,
                                                          CElementOp,
                                                          ComputeDataType>;
            auto ref_gemm_cpu     = ReferenceGemmInstanceCPU{};
            auto ref_invoker_cpu  = ref_gemm_cpu.MakeInvoker();
            auto ref_argument_cpu = ref_gemm_cpu.MakeArgument(
                a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);
            ref_invoker_cpu.Run(ref_argument_cpu);
        }
    }

    std::string best_op_name;
    float best_ave_time    = 0;
    float best_tflops      = 0;
    float best_gb_per_sec  = 0;
    float best_grid_size   = 0;
    float best_streamk_sel = 0;

    // Get number of SMs on the current GPU
    int device_id;
    hipError_t err = hipGetDevice(&device_id);
    if(err != hipSuccess)
    {
        std::cerr << "hipGetDevice failed: " << hipGetErrorString(err) << std::endl;
        return false;
    }

    hipDeviceProp_t props;
    err = hipGetDeviceProperties(&props, device_id);
    if(err != hipSuccess)
    {
        std::cerr << "hipGetDeviceProperties failed: " << hipGetErrorString(err) << std::endl;
        return false;
    }
    int num_sms = props.multiProcessorCount;

    // Generate grid sizes based on SM count with multipliers
    std::vector<float> multipliers = {0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f, 1.4f, 1.6f, 2.0f};
    std::vector<int> grid_size_list;

    for(float mult : multipliers)
    {
        int grid_size = static_cast<int>(num_sms * mult);
        if(grid_size > 0)
        {
            grid_size_list.push_back(grid_size);
        }
    }

    std::cout << "Number of SMs: " << num_sms << std::endl;
    std::cout << "Grid sizes to test: ";
    for(auto gs : grid_size_list)
    {
        std::cout << gs << " ";
    }
    std::cout << std::endl;

    // profile device GEMM instances
    for(auto& op_ptr : op_ptrs)
    {
        std::vector<int> streamk_sel_list = {
            0, 1, 2, 3, 4}; // 0: Data Parallel (DP) mode (Stream-K OFF), 1: 1-tile Stream-K+ DP,
                            // 2:2-tile Stream-K + DP

        if(Grid_size == -1)
        {
            grid_size_list = {Grid_size};
        }
        if(Streamk_sel != -1)
        {
            streamk_sel_list = {Streamk_sel};
        }
        for(std::size_t j = 0; j < streamk_sel_list.size(); j++)
        {
            for(std::size_t i = 0; i < grid_size_list.size(); i++)
            {
                auto grid_size_curr      = grid_size_list[i];
                index_t streamk_sel_curr = streamk_sel_list[j];
                printf("streamk_sel_curr=%0d\n", streamk_sel_curr);
                auto argument_ptr = op_ptr->MakeArgumentPointer(
                    static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                    static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                    static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                    M,
                    N,
                    K,
                    StrideA,
                    StrideB,
                    StrideC,
                    streamk_sel_curr,
                    grid_size_curr,
                    a_element_op,
                    b_element_op,
                    c_element_op);

                auto invoker_ptr = op_ptr->MakeInvokerPointer();

                if(op_ptr->IsSupportedArgument(argument_ptr.get()))
                {

                    // re-init C to zero before profiling next kernel
                    c_device_buf.SetZero();

                    invoker_ptr->Run(argument_ptr.get(),
                                     StreamConfig{nullptr, false, 0, n_warmup, n_iter});

                    if(do_verification)
                    {
                        c_device_buf.FromDevice(c_m_n_device_result.mData.data());

                        // Always compare against CPU reference results computed earlier
                        pass = pass & ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);

                        if(do_log)
                        {
                            LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",")
                                << std::endl;
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
                              << " TFlops, " << gb_per_sec << " GB/s, " << op_name << ", Grid_size "
                              << grid_size_curr << ", streamk selection strategy"
                              << streamk_sel_curr << std::endl;

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
                        best_op_name     = op_name;
                        best_tflops      = tflops;
                        best_ave_time    = ave_time;
                        best_gb_per_sec  = gb_per_sec;
                        best_grid_size   = grid_size_curr;
                        best_streamk_sel = streamk_sel_curr;
                    }
                }
                else
                {
                    std::cout << op_ptr->GetTypeString() << " does not support this problem"
                              << std::endl;
                }
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
              << " StrideB = " << StrideB << " StrideC = " << StrideC
              << " Grid_size = " << best_grid_size
              << " Stream-K selection strategy = " << best_streamk_sel << " : " << best_ave_time
              << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec << " GB/s, "
              << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
