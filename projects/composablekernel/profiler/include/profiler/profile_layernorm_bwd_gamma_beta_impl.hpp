// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/layernorm_bwd_gamma_beta.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_layernorm_bwd.hpp"

namespace ck {
namespace profiler {

template <typename DYDataType,
          typename XDataType,
          typename MeanInvStdDataType,
          typename ComputeDataType,
          typename DGammaDataType,
          typename DBetaDataType,
          index_t Rank>
bool profile_layernorm_bwd_gamma_beta_impl(int do_verification,
                                           int init_method,
                                           bool do_log,
                                           bool time_kernel,
                                           std::vector<index_t> length,
                                           index_t instance_index = -1)
{
    // we don't need GammaDataType and DXDataType here, just for reference class
    using GammaDataType = DYDataType;
    using DXDataType    = DYDataType;

    if(length.size() != Rank || Rank < 2)
        return false;

    // Assume normalize dimension for first dimension
    // Layernorm 2D, input = [M, K], reduce on M axis
    // Layernorm 4D, input = [N, H, W, C], redice on N axis
    constexpr int NumReduceDim = Rank - 1;

    std::vector<index_t> reduce_dim = {0};
    std::vector<index_t> invarient_length{length.begin() + 1, length.end()};

    Tensor<DYDataType> dy(length);
    Tensor<XDataType> x(length);
    Tensor<GammaDataType> gamma(invarient_length); // dummy tensor, for reference
    Tensor<MeanInvStdDataType> mean({length[0]});
    Tensor<MeanInvStdDataType> inv_std({length[0]});
    Tensor<DGammaDataType> dgamma(invarient_length);
    Tensor<DBetaDataType> dbeta(invarient_length);

    Tensor<DXDataType> host_dx(length); // dummy tensor, for reference
    Tensor<DGammaDataType> host_dgamma(invarient_length);
    Tensor<DBetaDataType> host_dbeta(invarient_length);

    std::vector<index_t> strideDy =
        std::vector<ck::index_t>{dy.mDesc.GetStrides().begin(), dy.mDesc.GetStrides().end()};
    std::vector<index_t> strideX = strideDy;

    std::vector<index_t> strideDGamma{dgamma.mDesc.GetStrides().begin(),
                                      dgamma.mDesc.GetStrides().end()};

    std::vector<index_t> strideDBeta{dbeta.mDesc.GetStrides().begin(),
                                     dbeta.mDesc.GetStrides().end()};

    std::vector<index_t> strideMeanInvStd{Rank, 0};
    strideMeanInvStd[0] = 1;

    switch(init_method)
    {
    case 0:
        dy.GenerateTensorValue(GeneratorTensor_1<DYDataType>{});
        x.GenerateTensorValue(GeneratorTensor_1<XDataType>{});
        mean.GenerateTensorValue(GeneratorTensor_1<MeanInvStdDataType>{});
        inv_std.GenerateTensorValue(GeneratorTensor_1<MeanInvStdDataType>{});
        dgamma.GenerateTensorValue(GeneratorTensor_1<DGammaDataType>{});
        dbeta.GenerateTensorValue(GeneratorTensor_1<DBetaDataType>{});
        break;
    case 1:
        dy.GenerateTensorValue(GeneratorTensor_2<DYDataType>{-5, 5});
        x.GenerateTensorValue(GeneratorTensor_2<XDataType>{-5, 5});
        mean.GenerateTensorValue(GeneratorTensor_2<MeanInvStdDataType>{-5, 5});
        inv_std.GenerateTensorValue(GeneratorTensor_2<MeanInvStdDataType>{0, 5});
        dgamma.GenerateTensorValue(GeneratorTensor_2<DGammaDataType>{-5, 5});
        dbeta.GenerateTensorValue(GeneratorTensor_2<DBetaDataType>{-5, 5});
        break;
    default:
        dy.GenerateTensorValue(GeneratorTensor_3<DYDataType>{0, 1});
        x.GenerateTensorValue(GeneratorTensor_3<XDataType>{0, 1});
        mean.GenerateTensorValue(GeneratorTensor_3<MeanInvStdDataType>{-0.5, 0.5});
        inv_std.GenerateTensorValue(GeneratorTensor_3<MeanInvStdDataType>{0, 0.5});
        dgamma.GenerateTensorValue(GeneratorTensor_3<DGammaDataType>{-0.5, 0.5});
        dbeta.GenerateTensorValue(GeneratorTensor_3<DBetaDataType>{-0.5, 0.5});
    }

    DeviceMem dy_dev(sizeof(DYDataType) * dy.mDesc.GetElementSpaceSize());
    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem mean_dev(sizeof(MeanInvStdDataType) * mean.mDesc.GetElementSpaceSize());
    DeviceMem inv_std_dev(sizeof(MeanInvStdDataType) * inv_std.mDesc.GetElementSpaceSize());
    DeviceMem dgamma_dev(sizeof(DGammaDataType) * dgamma.mDesc.GetElementSpaceSize());
    DeviceMem dbeta_dev(sizeof(DBetaDataType) * dbeta.mDesc.GetElementSpaceSize());

    dy_dev.ToDevice(dy.mData.data());
    x_dev.ToDevice(x.mData.data());
    mean_dev.ToDevice(mean.mData.data());
    inv_std_dev.ToDevice(inv_std.mData.data());

    // add device normalization instances
    using DeviceOp =
        ck::tensor_operation::device::DeviceNormalizationBwdGammaBeta<DYDataType,
                                                                      XDataType,
                                                                      MeanInvStdDataType,
                                                                      DGammaDataType,
                                                                      DBetaDataType,
                                                                      Rank,
                                                                      NumReduceDim>;

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
        using ReferenceInstance =
            ck::tensor_operation::host::ReferenceLayernormBwd<DYDataType,
                                                              XDataType,
                                                              GammaDataType,
                                                              MeanInvStdDataType,
                                                              DGammaDataType,
                                                              DBetaDataType,
                                                              DXDataType,
                                                              ComputeDataType>;

        ReferenceInstance ref;
        auto ref_argument =
            ref.MakeArgument(dy, x, gamma, mean, inv_std, host_dgamma, host_dbeta, host_dx, length);
        auto ref_invoker = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);
    }

    std::size_t num_bytes = dy.mDesc.GetElementSize() * sizeof(DYDataType) +
                            x.mDesc.GetElementSize() * sizeof(XDataType) +
                            mean.mDesc.GetElementSize() * sizeof(MeanInvStdDataType) +
                            inv_std.mDesc.GetElementSize() * sizeof(MeanInvStdDataType) +
                            dgamma.mDesc.GetElementSize() * sizeof(DGammaDataType) +
                            dbeta.mDesc.GetElementSize() * sizeof(DBetaDataType);

    int num_kernel = 0;

    for(auto& inst_ptr : instance_ptrs)
    {
        auto argument_ptr = inst_ptr->MakeArgumentPointer(length,
                                                          strideDy,
                                                          strideX,
                                                          strideMeanInvStd,
                                                          strideMeanInvStd,
                                                          invarient_length,
                                                          strideDGamma,
                                                          strideDBeta,
                                                          reduce_dim,
                                                          dy_dev.GetDeviceBuffer(),
                                                          x_dev.GetDeviceBuffer(),
                                                          mean_dev.GetDeviceBuffer(),
                                                          inv_std_dev.GetDeviceBuffer(),
                                                          dgamma_dev.GetDeviceBuffer(),
                                                          dbeta_dev.GetDeviceBuffer());

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
                LogRange(std::cout << "input lengths = ", length, ", ") << std::endl;
            }

            continue;
        }

        size_t workspace_sz = inst_ptr->GetWorkSpaceSize(argument_ptr.get());
        DeviceMem workspace_dev(workspace_sz);
        inst_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

        auto invoker_ptr = inst_ptr->MakeInvokerPointer();

        float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

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
            dgamma_dev.FromDevice(dgamma.mData.data());
            dbeta_dev.FromDevice(dbeta.mData.data());
            bool pass =
                ck::utils::check_err(dgamma, host_dgamma, "Error: Incorrect dgamma", 1e-3, 1e-3);

            pass &= ck::utils::check_err(dbeta, host_dbeta, "Error: Incorrect dbeta", 1e-3, 1e-3);

            if(do_log)
            {
                LogRangeAsType<float>(std::cout << "dy  : ", dy.mData, ",") << std::endl;
                LogRangeAsType<float>(std::cout << "host_dgamma  : ", host_dgamma.mData, ",")
                    << std::endl;
                LogRangeAsType<float>(std::cout << "dgamma  : ", dgamma.mData, ",") << std::endl;
            }

            if(!pass)
            {
                std::cout << inst_ptr->GetTypeString() << " failed verification: ";
                LogRange(std::cout << "lengths = [", length, ", ") << "]." << std::endl;
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
        LogRange(std::cout << "length = ", length, ",") << ", ";
        LogRange(std::cout << "reduce dims ", reduce_dim, ",") << std::endl;
        std::cout << "best perf = " << best_avg_time << " ms, " << best_gb_per_sec << " GB/s,"
                  << best_instance_name << std::endl;
    }

    if(num_kernel == 0)
    {
        std::cout << "Error: No kernel is applicable" << std::endl;
        return false;
    }
    if(instance_index != -1)
    {
        std::cout << "layernorm_bwd_gamma_instance (" << instance_index << "/" << num_kernel
                  << "): Passed" << std::endl;
    }
    return true;
}

} // namespace profiler
} // namespace ck
