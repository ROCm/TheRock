// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_splitk.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_gemm.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_fastgelu.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

namespace ck {
namespace profiler {

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementOp = PassThrough,
          typename BElementOp = PassThrough,
          typename CElementOp = PassThrough>
bool profile_grouped_gemm_impl(int do_verification,
                               int init_method,
                               bool do_log,
                               bool time_kernel,
                               const std::vector<int>& Ms,
                               const std::vector<int>& Ns,
                               const std::vector<int>& Ks,
                               const std::vector<int>& StrideAs,
                               const std::vector<int>& StrideBs,
                               const std::vector<int>& StrideCs,
                               const std::vector<int>& kbatches   = {},
                               int n_warmup                       = -1,
                               int n_iter                         = -1,
                               int instance_index                 = -1,
                               bool fail_if_no_supported_instance = false)
{
    bool pass = true;
    // TODO: Fixme - we do not pass compute data type here but need it
    // to compute error thresholds.
    using ComputeDataType = ADataType;

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

    double max_abs_in_val = 0.f;
    for(std::size_t i = 0; i < group_count; i++)
    {
        a_m_k.push_back(
            Tensor<ADataType>(f_host_tensor_descriptor(Ms[i], Ks[i], StrideAs[i], ALayout{})));
        b_k_n.push_back(
            Tensor<BDataType>(f_host_tensor_descriptor(Ks[i], Ns[i], StrideBs[i], BLayout{})));

        c_m_n_device_results.push_back(
            Tensor<CDataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideCs[i], CLayout{})));

        c_m_n_host_results.push_back(
            Tensor<CDataType>(f_host_tensor_descriptor(Ms[i], Ns[i], StrideCs[i], CLayout{})));
        if(do_log)
        {
            std::cout << "group: " << i << " a_m_k[" << i << "]:" << a_m_k[i].mDesc << ", b_k_n["
                      << i << "]:" << b_k_n[i].mDesc << ", c_m_n_device_results[" << i
                      << "]:" << c_m_n_device_results[i].mDesc << std::endl;
        }
        switch(init_method)
        {
        case 0: break;
        case 1:
            ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k[i]);
            ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n[i]);
            max_abs_in_val = 5.f;
            break;
        default:
            ck::utils::FillUniformDistribution<ADataType>{0.0f, 1.0f}(a_m_k[i]);
            ck::utils::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_k_n[i]);
            max_abs_in_val = 1.0f;
        }
    }

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

        gemm_descs.push_back({Ms[i], Ns[i], Ks[i], StrideAs[i], StrideBs[i], StrideCs[i], {}});

        p_a.push_back(a_device_buf[i]->GetDeviceBuffer());
        p_b.push_back(b_device_buf[i]->GetDeviceBuffer());
        p_c.push_back(c_device_buf[i]->GetDeviceBuffer());
    }

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedGemm<ALayout,
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

    // If kbatch would be bigger than 1, then we will use SplitK version.
    using DeviceOpSplitK = ck::tensor_operation::device::DeviceGroupedGemmSplitK<ALayout,
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

    auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
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
    int num_kernel        = 0;
    auto p_ds             = std::vector<std::array<const void*, 0>>{};

    StreamConfig stream_config{nullptr, time_kernel};
    if(n_warmup >= 0)
    {
        stream_config.cold_niters_ = n_warmup;
    }

    if(n_iter >= 0)
    {
        stream_config.nrepeat_ = n_iter;
    }

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

    // If the user will provide not empty kbatches list, then we test predefined set of kbatch
    // values.
    std::vector<int> kbatch_list = {1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64};
    if(!kbatches.empty())
    {
        kbatch_list = kbatches;
    }

    // Check if the operation requested any KBatch size > 1
    bool operation_requires_splitk_support = false;
    for(auto kbatch : kbatch_list)
    {
        if(kbatch > 1)
        {
            operation_requires_splitk_support = true;
            break;
        }
    }

    // profile device GEMM instances
    int instances_supported         = 0;
    int instances_supporting_splitk = 0;
    for(auto& gemm_ptr : op_ptrs)
    {
        auto argument_ptr = gemm_ptr->MakeArgumentPointer(
            p_a, p_b, p_ds, p_c, gemm_descs, a_element_op, b_element_op, c_element_op);

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

        std::size_t workspace_size = gemm_ptr->GetWorkSpaceSize(argument_ptr.get());
        std::size_t kargs_size     = gemm_ptr->GetDeviceKernelArgSize(argument_ptr.get());

        DeviceMem gemm_workspace, gemm_kargs;

        // The following is necessary since TwoStage kernel is using additional memory both
        // for Workspace and kernel arguments.
        if(kargs_size > 0)
        {
            gemm_kargs.Realloc(kargs_size);
            gemm_ptr->SetDeviceKernelArgs(argument_ptr.get(), gemm_kargs.GetDeviceBuffer());
        }
        if(workspace_size > 0 && workspace_size != kargs_size)
        {
            gemm_workspace.Realloc(workspace_size);
            gemm_ptr->SetWorkSpacePointer(argument_ptr.get(), gemm_workspace.GetDeviceBuffer());
        }

        std::string gemm_name = gemm_ptr->GetTypeString();

        // Keep track if we found any supported instance
        bool any_supported_instance          = false;
        bool any_supported_nontrivial_kbatch = false;
        for(std::size_t j = 0; j < kbatch_list.size(); j++)
        {
            auto kbatch_curr = kbatch_list[j];

            if(kbatch_curr > 1 && dynamic_cast<DeviceOpSplitK*>(gemm_ptr.get()) != nullptr)
            {
                dynamic_cast<DeviceOpSplitK*>(gemm_ptr.get())
                    ->SetKBatchSize(argument_ptr.get(), kbatch_curr);
            }

            if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                ++num_kernel;
                if((instance_index != -1) && (instance_index + 1 != num_kernel))
                {
                    // skip test if instance_index is specified
                    continue;
                }

                // Keep track of which supported instances we found
                any_supported_instance = true;
                if(kbatch_curr > 1)
                {
                    any_supported_nontrivial_kbatch = true;
                }

                for(std::size_t i = 0; i < gemm_descs.size(); i++)
                    c_device_buf[i]->SetZero();

                float ave_time = invoker_ptr->Run(argument_ptr.get(), stream_config);

                if(do_verification)
                {
                    bool instance_pass = true;
                    for(std::size_t i = 0; i < gemm_descs.size(); i++)
                    {
                        c_device_buf[i]->FromDevice(c_m_n_device_results[i].mData.data());
                        auto atol = ck::utils::get_absolute_threshold<ComputeDataType, CDataType>(
                            max_abs_in_val, gemm_descs[i].K_);
                        auto rtol = ck::utils::get_relative_threshold<ComputeDataType, CDataType>(
                            gemm_descs[i].K_);

                        instance_pass =
                            instance_pass && ck::utils::check_err(c_m_n_device_results[i],
                                                                  c_m_n_host_results[i],
                                                                  "Error: Incorrect results!",
                                                                  rtol,
                                                                  atol);

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

                    std::cout << "Instance: " << gemm_name << "; KBatch: " << kbatch_curr << " "
                              << (instance_pass ? "SUCCEED" : "FAILED") << std::endl;

                    pass = pass && instance_pass;
                }

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
                std::cout << "Instance: " << gemm_name
                          << ", does not support this GEMM problem (KBatch: " << kbatch_curr << ")"
                          << std::endl;
            }
        }

        // If any kbatch sizes > 1 were supported by this instance, the instance can be marked as
        // 'supported' for this problem
        if(any_supported_nontrivial_kbatch)
        {
            ++instances_supporting_splitk;
        }

        if(any_supported_instance)
        {
            ++instances_supported;
        }
    }

    // Warn if not a single instance was supported
    if(instances_supported == 0)
    {
        std::cout << "Warning! No supported instance found." << std::endl;

        if(fail_if_no_supported_instance)
        {
            return false;
        }
    }
    else if(operation_requires_splitk_support && instances_supporting_splitk == 0)
    {
        std::cout << "Warning! No instance found that supported any of the kbatch sizes."
                  << std::endl;

        if(fail_if_no_supported_instance)
        {
            return false;
        }
    }

    if(time_kernel)
    {
        std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
                  << best_gb_per_sec << " GB/s, " << best_gemm_name << ", KBatch = " << best_kbatch
                  << std::endl;
    }
    if(instance_index != -1)
    {
        std::cout << "grouped_gemm_instance (" << instance_index << "/" << num_kernel << "): Passed"
                  << std::endl;
    }

    return pass;
}

} // namespace profiler
} // namespace ck
