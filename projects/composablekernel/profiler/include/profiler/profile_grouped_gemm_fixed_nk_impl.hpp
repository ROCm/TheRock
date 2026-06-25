// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_fixed_nk.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
bool profile_grouped_gemm_fixed_nk_impl(int do_verification,
                                        int init_method,
                                        bool do_log,
                                        bool time_kernel,
                                        const std::vector<int>& Ms,
                                        const std::vector<int>& Ns,
                                        const std::vector<int>& Ks,
                                        const std::vector<int>& StrideAs,
                                        const std::vector<int>& StrideBs,
                                        const std::vector<int>& StrideCs,
                                        int kbatch   = 1,
                                        int n_warmup = 1,
                                        int n_iter   = 10)
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

    std::size_t group_count = Ms.size();

    if(!(group_count == Ns.size() && group_count == Ks.size() && group_count == StrideAs.size() &&
         group_count == StrideBs.size() && group_count == StrideCs.size()))
    {
        throw std::runtime_error("wrong! inconsistent M/N/Ks, StrideA/B/Cs size\n");
    }

    std::vector<Tensor<ADataType>> a_m_k;
    std::vector<Tensor<BDataType>> b_k_n;
    std::vector<Tensor<CDataType>> c_m_n_host_results;
    std::vector<Tensor<CDataType>> c_m_n_device_results;
    int sum_of_m = 0;

    for(std::size_t i = 0; i < group_count; i++)
    {
        sum_of_m += Ms[i];
        a_m_k.push_back(
            Tensor<ADataType>(f_host_tensor_descriptor(Ms[i], Ks[i], StrideAs[i], ALayout{})));
        b_k_n.push_back(
            Tensor<BDataType>(f_host_tensor_descriptor(Ks[i], Ns[i], StrideBs[i], BLayout{})));

        c_m_n_device_results.push_back(
            Tensor<CDataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideCs[i], CLayout{})));

        c_m_n_host_results.push_back(
            Tensor<CDataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideCs[i], CLayout{})));
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            std::cout << "group: " << i << " a_m_k[" << i << "]:" << a_m_k[i].mDesc << ", b_k_n["
                      << i << "]:" << b_k_n[i].mDesc << ", c_m_n_device_results[" << i
                      << "]:" << c_m_n_device_results[i].mDesc << std::endl;
        }
        std::size_t num_thread = 1;
        switch(init_method)
        {
        case 0: break;
        case 1:
            a_m_k[i].GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5}, num_thread);
            b_k_n[i].GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5}, num_thread);
            break;
        default:
            a_m_k[i].GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0}, num_thread);
            b_k_n[i].GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5}, num_thread);
        }
    }

    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;
    std::vector<DeviceMemPtr> a_device_buf, b_device_buf, c_device_buf;

    a_device_buf.reserve(group_count);
    b_device_buf.reserve(group_count);
    c_device_buf.reserve(group_count);

    std::vector<const void*> p_a, p_b;
    std::vector<void*> p_c;

    p_a.reserve(group_count);
    p_b.reserve(group_count);
    p_c.reserve(group_count);

    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;
    gemm_descs.reserve(group_count);

    std::vector<ck::tensor_operation::device::GroupedGemmKernelArgument<1>>
        grouped_gemm_kernel_args_;
    grouped_gemm_kernel_args_.reserve(group_count);

    for(std::size_t i = 0; i < group_count; i++)
    {
        a_device_buf.emplace_back(
            std::make_unique<DeviceMem>(sizeof(ADataType) * a_m_k[i].mDesc.GetElementSpaceSize()));
        b_device_buf.emplace_back(
            std::make_unique<DeviceMem>(sizeof(BDataType) * b_k_n[i].mDesc.GetElementSpaceSize()));
        c_device_buf.emplace_back(std::make_unique<DeviceMem>(
            sizeof(CDataType) * c_m_n_device_results[i].mDesc.GetElementSpaceSize()));

        a_device_buf[i]->ToDevice(a_m_k[i].mData.data());
        b_device_buf[i]->ToDevice(b_k_n[i].mData.data());

        gemm_descs.push_back({sum_of_m, Ns[i], Ks[i], StrideAs[i], StrideBs[i], StrideCs[i], {}});

        p_a.push_back(a_device_buf[i]->GetDeviceBuffer());
        p_b.push_back(b_device_buf[i]->GetDeviceBuffer());
        p_c.push_back(c_device_buf[i]->GetDeviceBuffer());

        grouped_gemm_kernel_args_.push_back({a_device_buf[i]->GetDeviceBuffer(),
                                             b_device_buf[i]->GetDeviceBuffer(),
                                             {},
                                             c_device_buf[i]->GetDeviceBuffer(),
                                             Ms[i],
                                             Ns[i],
                                             Ks[i],
                                             StrideAs[i],
                                             StrideBs[i],
                                             {},
                                             StrideCs[i]});
    }

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedGemmFixedNK<ALayout,
                                                                            BLayout,
                                                                            ck::Tuple<>,
                                                                            CLayout,
                                                                            ADataType,
                                                                            BDataType,
                                                                            ck::Tuple<>,
                                                                            CDataType,
                                                                            AElementOp,
                                                                            BElementOp,
                                                                            CElementOp>;

    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    if(op_ptrs.size() <= 0)
    {
        throw std::runtime_error("wrong! no device GEMM instance found");
    }

    std::string best_gemm_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    float best_kbatch     = 0;

    auto p_ds = std::vector<std::array<const void*, 0>>{};

    if(do_verification)
    {
        for(std::size_t i = 0; i < gemm_descs.size(); i++)
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

            auto ref_argument = ref_gemm.MakeArgument(a_m_k[i],
                                                      b_k_n[i],
                                                      c_m_n_host_results[i],
                                                      a_element_op,
                                                      b_element_op,
                                                      c_element_op);

            ref_invoker.Run(ref_argument);
        }
    }

    // profile device GEMM instances
    for(auto& gemm_ptr : op_ptrs)
    {
        auto argument_ptr =
            gemm_ptr->MakeArgumentPointer(p_a,
                                          p_b,
                                          p_ds,
                                          p_c,
                                          gemm_descs,
                                          ck::tensor_operation::element_wise::PassThrough{},
                                          ck::tensor_operation::element_wise::PassThrough{},
                                          ck::tensor_operation::element_wise::PassThrough{});

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

        DeviceMem gemm_desc_workspace(gemm_ptr->GetWorkSpaceSize(argument_ptr.get()));

        DeviceMem grouped_gemm_kernel_args_dev(
            gemm_ptr->GetDeviceKernelArgSize(argument_ptr.get()));

        hipGetErrorString(hipMemcpy(grouped_gemm_kernel_args_dev.GetDeviceBuffer(),
                                    grouped_gemm_kernel_args_.data(),
                                    gemm_ptr->GetDeviceKernelArgSize(argument_ptr.get()),
                                    hipMemcpyHostToDevice));

        gemm_ptr->SetWorkSpacePointer(argument_ptr.get(), gemm_desc_workspace.GetDeviceBuffer());

        gemm_ptr->SetDeviceKernelArgs(argument_ptr.get(),
                                      grouped_gemm_kernel_args_dev.GetDeviceBuffer());

        std::string gemm_name = gemm_ptr->GetTypeString();

        std::vector<int> kbatch_list = {1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64};

        if(kbatch > 0)
        {
            kbatch_list = {kbatch};
        }

        for(std::size_t j = 0; j < kbatch_list.size(); j++)
        {

            auto kbatch_curr = kbatch_list[j];

            gemm_ptr->SetKBatch(argument_ptr.get(), kbatch_curr);

            if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                for(std::size_t i = 0; i < gemm_descs.size(); i++)
                    c_device_buf[i]->SetZero();

                invoker_ptr->Run(argument_ptr.get(),
                                 StreamConfig{nullptr, false, 0, n_warmup, n_iter});

                if(do_verification)
                {
                    bool instance_pass = true;
                    for(std::size_t i = 0; i < gemm_descs.size(); i++)
                    {

                        c_device_buf[i]->FromDevice(c_m_n_device_results[i].mData.data());

                        if(std::is_same_v<CDataType, ck::half_t> && kbatch_curr > 1)
                        {
                            instance_pass =
                                instance_pass && ck::utils::check_err(c_m_n_device_results[i],
                                                                      c_m_n_host_results[i],
                                                                      "Error: Incorrect results!",
                                                                      0.06);
                        }
                        else
                        {
                            instance_pass =
                                instance_pass && ck::utils::check_err(c_m_n_device_results[i],
                                                                      c_m_n_host_results[i]);
                        }

                        if(do_log)
                        {
                            LogRangeAsType<float>(std::cout << "a : ", a_m_k[i].mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(std::cout << "b: ", b_k_n[i].mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(
                                std::cout << "c_device: ", c_m_n_device_results[i].mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(
                                std::cout << "c_host  : ", c_m_n_host_results[i].mData, ",")
                                << std::endl;
                        }
                    }

                    std::cout << "Instance: " << gemm_name << " verification "
                              << (instance_pass ? "SUCCEED" : "FAILED") << std::endl;

                    pass = pass && instance_pass;
                }

                float ave_time = invoker_ptr->Run(
                    argument_ptr.get(), StreamConfig{nullptr, time_kernel, 0, n_warmup, n_iter});

                if(time_kernel)
                {
                    std::size_t flop = 0, num_btype = 0;
                    for(std::size_t i = 0; i < gemm_descs.size(); i++)
                    {
                        flop += std::size_t(2) * Ms[i] * Ns[i] * Ks[i];

                        num_btype += sizeof(ADataType) * Ms[i] * Ks[i] +
                                     sizeof(BDataType) * Ks[i] * Ns[i] +
                                     sizeof(CDataType) * Ms[i] * Ns[i];
                    }

                    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

                    float gb_per_sec = num_btype / 1.E6 / ave_time;
                    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops
                              << " TFlops, " << gb_per_sec << " GB/s, " << gemm_name << ", KBatch "
                              << kbatch_curr << std::endl;

                    if(tflops > best_tflops)
                    {
                        best_gemm_name  = gemm_name;
                        best_tflops     = tflops;
                        best_ave_time   = ave_time;
                        best_gb_per_sec = gb_per_sec;
                        best_kbatch     = kbatch_curr;
                    }
                }
            }
            else
            {
                std::cout << "Instance: " << gemm_name << ", does not support this GEMM problem"
                          << std::endl;
            }
        }
    }

    if(time_kernel)
    {
        std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
                  << best_gb_per_sec << " GB/s, " << best_gemm_name << ", KBatch = " << best_kbatch
                  << std::endl;
    }
    return pass;
}

} // namespace profiler
} // namespace ck
