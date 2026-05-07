// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/pool3d_fwd.hpp"
#include "ck/library/tensor_operation_instance/gpu/avg_pool3d_bwd.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_avgpool_bwd.hpp"

namespace ck {
namespace profiler {

template <typename TensorLayout>
std::vector<ck::index_t> f_tensor_strides_ncdhw(ck::index_t N_,
                                                ck::index_t C_,
                                                ck::index_t D,
                                                ck::index_t H,
                                                ck::index_t W,
                                                TensorLayout layout)
{
    using namespace ck::literals;
    (void)N_;
    if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::convolution::NDHWC>::value)
        return {D * C_ * H * W, 1_uz, C_ * H * W, W * C_, C_};
    else
        throw std::runtime_error("not supported yet");
};

template <typename DOutDataType,
          typename DInDataType,
          typename ComputeDataType,
          typename DOutLayout,
          typename DInLayout>
bool profile_avg_pool3d_bwd_impl(int do_verification,
                                 int init_method,
                                 bool do_log,
                                 bool time_kernel,
                                 std::vector<index_t> in_length, // NCDHW
                                 std::vector<index_t> window_spatial_lengths,
                                 std::vector<index_t> window_strides,
                                 std::vector<index_t> window_dilations,
                                 std::vector<index_t> input_left_pads,
                                 std::vector<index_t> input_right_pads,
                                 index_t instance_index = -1)
{
    constexpr index_t InOutRank  = 5;
    constexpr index_t WindowRank = 3;

    if(in_length.size() != InOutRank || window_spatial_lengths.size() != WindowRank ||
       window_strides.size() != WindowRank || window_dilations.size() != WindowRank ||
       input_left_pads.size() != WindowRank || input_right_pads.size() != WindowRank)
    {
        std::cout << "Parameter is incorrect" << std::endl;
        return false;
    }

    std::vector<index_t> out_length(InOutRank);

    int N = in_length[0];
    int C = in_length[1];

    out_length[0] = N;
    out_length[1] = C;

    // Calculate Do, Ho, Wo
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

    int Di = in_length[2];
    int Hi = in_length[3];
    int Wi = in_length[4];
    int Do = out_length[2];
    int Ho = out_length[3];
    int Wo = out_length[4];

    auto f_host_tensor_descriptor =
        [](std::size_t N_, std::size_t C_, std::size_t D, std::size_t H, std::size_t W) {
            using namespace ck::literals;

            return HostTensorDescriptor({N_, C_, D, H, W},
                                        {D * C_ * H * W, 1_uz, C_ * H * W, W * C_, C_},
                                        ck::tensor_layout::convolution::NDHWC{});
        };

    Tensor<DOutDataType> dout_n_c_do_ho_wo(f_host_tensor_descriptor(N, C, Do, Ho, Wo));
    Tensor<DInDataType> din_n_c_di_hi_wi_device(f_host_tensor_descriptor(N, C, Di, Hi, Wi));
    Tensor<DInDataType> din_n_c_di_hi_wi_host(f_host_tensor_descriptor(N, C, Di, Hi, Wi));

    switch(init_method)
    {
    case 0: dout_n_c_do_ho_wo.GenerateTensorValue(GeneratorTensor_1<DOutDataType>{}); break;
    case 1: dout_n_c_do_ho_wo.GenerateTensorValue(GeneratorTensor_2<DOutDataType>{-5, 5}); break;
    default: dout_n_c_do_ho_wo.GenerateTensorValue(GeneratorTensor_3<DOutDataType>{-0.5, 0.5});
    }

    DeviceMem dout_device_buf(sizeof(DOutDataType) * dout_n_c_do_ho_wo.mDesc.GetElementSpaceSize());
    DeviceMem din_device_buf(sizeof(DInDataType) *
                             din_n_c_di_hi_wi_device.mDesc.GetElementSpaceSize());

    dout_device_buf.ToDevice(dout_n_c_do_ho_wo.mData.data());

    using DeviceOp = ck::tensor_operation::device::
        DeviceAvgPoolBwd<3, DOutDataType, DInDataType, DOutLayout, DInLayout>;

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
        using ReferencePoolingBwdInstance =
            ck::tensor_operation::host::ReferenceAvgPoolBwd<3, DInDataType, DOutDataType>;

        ReferencePoolingBwdInstance ref_pooling_bwd;
        auto ref_pooling_bwd_argument = ref_pooling_bwd.MakeArgument(din_n_c_di_hi_wi_host,
                                                                     dout_n_c_do_ho_wo,
                                                                     window_spatial_lengths,
                                                                     window_strides,
                                                                     window_dilations,
                                                                     input_left_pads,
                                                                     input_right_pads);

        auto ref_invoker = ref_pooling_bwd.MakeInvoker();
        ref_invoker.Run(ref_pooling_bwd_argument);
    }

    int num_kernel = 0;

    for(auto& inst_ptr : instance_ptrs)
    {
        auto argument_ptr = inst_ptr->MakeArgumentPointer(
            static_cast<DOutDataType*>(dout_device_buf.GetDeviceBuffer()),
            static_cast<DInDataType*>(din_device_buf.GetDeviceBuffer()),
            {N, C, Do, Ho, Wo},
            {N, C, Di, Hi, Wi},
            f_tensor_strides_ncdhw(N, C, Do, Ho, Wo, DOutLayout{}),
            f_tensor_strides_ncdhw(N, C, Di, Hi, Wi, DInLayout{}),
            window_spatial_lengths,
            window_strides,
            window_dilations,
            input_left_pads,
            input_right_pads);

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
                LogRange(std::cout << "doutput lengths = ", out_length, ", ") << std::endl;
            }

            continue;
        }

        din_device_buf.SetZero();

        auto invoker_ptr = inst_ptr->MakeInvokerPointer();
        float avg_time   = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        std::size_t num_bytes =
            dout_n_c_do_ho_wo.mDesc.GetElementSize() * sizeof(DOutDataType) +
            din_n_c_di_hi_wi_device.mDesc.GetElementSize() * sizeof(DInDataType);

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
            din_device_buf.FromDevice(din_n_c_di_hi_wi_device.mData.data());
            bool pass = ck::utils::check_err(din_n_c_di_hi_wi_device.mData,
                                             din_n_c_di_hi_wi_host.mData,
                                             "Error: Incorrect results",
                                             1e-3,
                                             1e-3);

            if(do_log)
            {
                LogRangeAsType<float>(
                    std::cout << "din_n_c_di_hi_wi_device: ", din_n_c_di_hi_wi_device.mData, ",")
                    << std::endl;

                LogRangeAsType<float>(
                    std::cout << "din_n_c_di_hi_wi_host: ", din_n_c_di_hi_wi_host.mData, ",")
                    << std::endl;
            }

            if(!pass)
            {
                std::cout << inst_ptr->GetTypeString() << " failed verification: ";
                LogRange(std::cout << "doutput lengths = [", out_length, ", ") << "]." << std::endl;
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
        LogRange(std::cout << "length = ", out_length, ",") << std::endl;
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
        std::cout << "avg_pool3d_bwd_instance (" << instance_index << "/" << num_kernel
                  << "): Passed" << std::endl;
    }
    return true;
}

} // namespace profiler
} // namespace ck
