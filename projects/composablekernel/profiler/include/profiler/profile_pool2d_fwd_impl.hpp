// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/pool2d_fwd.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_pool_fwd.hpp"

namespace ck {
namespace profiler {

template <typename InDataType,
          typename OutDataType,
          typename ComputeDataType,
          typename IndexDataType,
          typename InLayout,
          typename OutLayout,
          ck::ReduceTensorOp ReduceOpId,
          bool PropagateNan,
          bool OutputIndex>
bool profile_pool2d_fwd_impl(int do_verification,
                             int init_method,
                             bool do_log,
                             bool time_kernel,
                             std::vector<index_t> in_length, // NCHW
                             std::vector<index_t> window_spatial_lengths,
                             std::vector<index_t> window_strides,
                             std::vector<index_t> window_dilations,
                             std::vector<index_t> input_left_pads,
                             std::vector<index_t> input_right_pads,
                             index_t instance_index = -1)
{
    constexpr index_t InOutRank  = 4;
    constexpr index_t WindowRank = 2;

    if(in_length.size() != InOutRank || window_spatial_lengths.size() != WindowRank ||
       window_strides.size() != WindowRank || window_dilations.size() != WindowRank ||
       input_left_pads.size() != WindowRank || input_right_pads.size() != WindowRank)
        return false;

    std::vector<index_t> out_length(InOutRank);

    int N = in_length[0];
    int C = in_length[1];

    out_length[0] = N;
    out_length[1] = C;

    // Calculate Ho, Wo
    for(int i = 2; i < InOutRank; ++i)
    {
        auto pad1             = input_left_pads[i - 2];
        auto pad2             = input_right_pads[i - 2];
        auto windows_size     = window_spatial_lengths[i - 2];
        auto windows_stride   = window_strides[i - 2];
        auto windows_dilation = window_dilations[i - 2];
        auto eff              = (windows_size - 1) * windows_dilation + 1;
        out_length[i]         = (in_length[i] + pad1 + pad2 - eff) / windows_stride + 1;
    }

    int Hi = in_length[2];
    int Wi = in_length[3];
    int Ho = out_length[2];
    int Wo = out_length[3];

    auto f_host_tensor_descriptor =
        [](std::size_t N_, std::size_t C_, std::size_t H, std::size_t W) {
            using namespace ck::literals;

            return HostTensorDescriptor({N_, C_, H, W},
                                        {C_ * H * W, 1_uz, W * C_, C_},
                                        ck::tensor_layout::convolution::NCHW{});
        };

    Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi));
    Tensor<OutDataType> out_n_c_ho_wo_host(f_host_tensor_descriptor(N, C, Ho, Wo));
    Tensor<IndexDataType> out_indices_n_c_ho_wo_host(f_host_tensor_descriptor(N, C, Ho, Wo));

    Tensor<OutDataType> out_n_c_ho_wo_device(f_host_tensor_descriptor(N, C, Ho, Wo));
    Tensor<IndexDataType> out_indices_n_c_ho_wo_device(f_host_tensor_descriptor(N, C, Ho, Wo));

    switch(init_method)
    {
    case 0: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_1<InDataType>{}); break;
    case 1: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}); break;
    default: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) *
                             out_n_c_ho_wo_device.mDesc.GetElementSpaceSize());
    DeviceMem out_indices_device_buf(sizeof(IndexDataType) *
                                     out_indices_n_c_ho_wo_device.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());

    // add device normalization instances
    using DeviceOp = ck::tensor_operation::device::DevicePoolFwd<InOutRank,
                                                                 WindowRank,
                                                                 InDataType,
                                                                 OutDataType,
                                                                 IndexDataType,
                                                                 InLayout,
                                                                 OutLayout,
                                                                 ReduceOpId,
                                                                 OutputIndex>;

    // get device op instances
    const auto instance_ptrs =
        ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();

    std::cout << "found " << instance_ptrs.size() << " instances" << std::endl;

    std::string best_instance_name;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    if(do_verification)
    {
        using ReferenceInstance = ck::tensor_operation::host::ReferencePoolingFwd<InOutRank,
                                                                                  WindowRank,
                                                                                  InDataType,
                                                                                  OutDataType,
                                                                                  ComputeDataType,
                                                                                  IndexDataType,
                                                                                  ReduceOpId,
                                                                                  PropagateNan,
                                                                                  OutputIndex>;

        ReferenceInstance ref;
        auto ref_argument = ref.MakeArgument(in_n_c_hi_wi,
                                             out_n_c_ho_wo_host,
                                             out_indices_n_c_ho_wo_host,
                                             window_spatial_lengths,
                                             window_strides,
                                             window_dilations,
                                             input_left_pads,
                                             input_right_pads);
        auto ref_invoker  = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);
    }

    int num_kernel = 0;

    for(auto& inst_ptr : instance_ptrs)
    {
        auto argument_ptr = inst_ptr->MakeArgumentPointer(
            static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
            static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
            static_cast<IndexDataType*>(out_indices_device_buf.GetDeviceBuffer()),
            in_length,
            window_spatial_lengths,
            out_length,
            {C * Hi * Wi, 1, Wi * C, C},
            {C * Ho * Wo, 1, Wo * C, C},
            {C * Ho * Wo, 1, Wo * C, C},
            window_strides,
            window_dilations,
            input_left_pads,
            input_right_pads,
            {2, 3});

        if(inst_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            ++num_kernel;
            if((instance_index != -1) && (instance_index + 1 != num_kernel))
            {
                // skip test if instance_index is specified
                continue;
            }
        }
        else
        {
            if(time_kernel)
            {
                std::cout << inst_ptr->GetTypeString() << " skipped due to unsupported argument: ";
                LogRange(std::cout << "input lengths = ", in_length, ", ") << std::endl;
            }

            continue;
        }

        auto invoker_ptr = inst_ptr->MakeInvokerPointer();

        float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        std::size_t num_bytes = in_n_c_hi_wi.mDesc.GetElementSize() * sizeof(InDataType) +
                                out_n_c_ho_wo_host.mDesc.GetElementSize() * sizeof(OutDataType);

        if constexpr(OutputIndex)
            num_bytes += out_indices_n_c_ho_wo_host.mDesc.GetElementSize() * sizeof(IndexDataType);

        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        if(time_kernel)
            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << gb_per_sec << " GB/s, "
                      << inst_ptr->GetTypeString() << std::endl;

        if(avg_time < best_avg_time)
        {
            best_instance_name = inst_ptr->GetTypeString();
            best_avg_time      = avg_time;
            best_gb_per_sec    = gb_per_sec;
        }

        if(do_verification)
        {
            out_device_buf.FromDevice(out_n_c_ho_wo_device.mData.data());

            bool pass = ck::utils::check_err(out_n_c_ho_wo_device.mData,
                                             out_n_c_ho_wo_host.mData,
                                             "Error: Incorrect results",
                                             1e-3,
                                             1e-3);

            if constexpr(OutputIndex)
            {
                out_indices_device_buf.FromDevice(out_indices_n_c_ho_wo_device.mData.data());

                pass = pass && ck::utils::check_err(out_indices_n_c_ho_wo_device,
                                                    out_indices_n_c_ho_wo_host);
            }

            if(do_log)
            {
                LogRangeAsType<float>(std::cout << "in_n_c_hi_wi  : ", in_n_c_hi_wi.mData, ",")
                    << std::endl;
                LogRangeAsType<float>(
                    std::cout << "out_n_c_ho_wo_host  : ", out_n_c_ho_wo_host.mData, ",")
                    << std::endl;
                LogRangeAsType<float>(
                    std::cout << "out_n_c_ho_wo_device  : ", out_n_c_ho_wo_device.mData, ",")
                    << std::endl;

                if constexpr(OutputIndex)
                    LogRangeAsType<float>(std::cout << "out_indices_n_c_ho_wo_device  : ",
                                          out_indices_n_c_ho_wo_device.mData,
                                          ",")
                        << std::endl;
            }

            if(!pass)
            {
                std::cout << inst_ptr->GetTypeString() << " failed verification: ";
                LogRange(std::cout << "lengths = [", in_length, ", ") << "]." << std::endl;
                return false;
            }
            else
            {
                if(time_kernel)
                    std::cout << "pass" << std::endl;
            }
        }
    }

    if(time_kernel)
    {
        LogRange(std::cout << "length = ", in_length, ",") << std::endl;
        std::cout << "best perf = " << best_avg_time << " ms, " << best_gb_per_sec << " GB/s, "
                  << best_instance_name << std::endl;
    }

    if(num_kernel == 0)
    {
        std::cout << "Error: No kernel is applicable" << std::endl;
        return false;
    }
    if(instance_index != -1)
    {
        std::cout << "max_pool2d_fwd_instance (" << instance_index << "/" << num_kernel
                  << "): Passed" << std::endl;
    }
    return true;
}

} // namespace profiler
} // namespace ck
