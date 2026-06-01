// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <stdexcept>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/tensor_operation_instance/gpu/batchnorm_infer.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batchnorm_infer.hpp"

namespace ck {
namespace profiler {

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          index_t Rank,
          index_t NumBatchNormReduceDim>
bool profile_batchnorm_infer_impl(int do_verification,
                                  int init_method,
                                  bool do_dumpout,
                                  bool time_kernel,
                                  const std::vector<size_t> inOutLengths,
                                  const std::vector<int> reduceDims,
                                  double epsilon,
                                  index_t instance_index = -1)
{
    if(inOutLengths.size() != Rank || reduceDims.size() != NumBatchNormReduceDim)
    {
        throw std::runtime_error("Invalid tensor lengths or number of reduce dimensions!");
    };

    std::vector<size_t> scaleBiasMeanVarLengths;
    std::vector<int> invariantDims;

    // used for calculating the effective transferred bytes by each operation
    size_t total_length;
    size_t invariant_length = 1;

    total_length =
        std::accumulate(inOutLengths.begin(), inOutLengths.end(), 1, std::multiplies<size_t>{});

    if(std::any_of(reduceDims.begin(), reduceDims.end(), [](int d) { return d < 0 || d >= Rank; }))
        throw std::runtime_error("Invalid reduce dimensions!");

    for(int dim = 0; dim < Rank; dim++)
    {
        if(std::none_of(reduceDims.begin(), reduceDims.end(), [&](int d) { return dim == d; }))
        {
            invariantDims.push_back(dim);
            scaleBiasMeanVarLengths.push_back(inOutLengths[dim]);
            invariant_length *= inOutLengths[dim];
        };
    }

    // input data of the batchnorm infer algorithm
    Tensor<XDataType> x(inOutLengths);
    Tensor<ScaleDataType> scale(scaleBiasMeanVarLengths);
    Tensor<BiasDataType> bias(scaleBiasMeanVarLengths);
    Tensor<MeanVarDataType> estimatedMean(scaleBiasMeanVarLengths);
    Tensor<MeanVarDataType> estimatedVariance(scaleBiasMeanVarLengths);

    // output data of the batchnorm infer algorithm
    Tensor<YDataType> y_ref(inOutLengths);
    Tensor<YDataType> y(inOutLengths);

    auto inOutStrides            = x.mDesc.GetStrides();
    auto scaleBiasMeanVarStrides = scale.mDesc.GetStrides();

    std::size_t num_thread = std::thread::hardware_concurrency();

    const float x_mean       = 0.0f;
    const float x_stddev     = 1.0f;
    const float noise_stddev = 0.04f;

    // input data in normal distribution
    x.GenerateTensorValue(GeneratorTensor_4<XDataType>{x_mean, x_stddev}, num_thread);

    // initialize the estimatedMean to be values with tiny variation to the mean of the x
    // values
    estimatedMean.GenerateTensorValue(GeneratorTensor_4<MeanVarDataType>{x_mean, noise_stddev},
                                      num_thread);

    // initialize the estimatedVariance to be values with tiny variation to the variance of
    // the x values
    estimatedVariance.GenerateTensorValue(
        GeneratorTensor_4<MeanVarDataType>{x_stddev * x_stddev, noise_stddev}, num_thread);

    if(do_verification)
    {
        switch(init_method)
        {
        case 0:
            scale.GenerateTensorValue(GeneratorTensor_0<ScaleDataType>{}, num_thread);
            bias.GenerateTensorValue(GeneratorTensor_0<BiasDataType>{}, num_thread);
            break;
        case 1:
            scale.GenerateTensorValue(GeneratorTensor_1<ScaleDataType>{1}, num_thread);
            bias.GenerateTensorValue(GeneratorTensor_1<BiasDataType>{0}, num_thread);
            break;
        case 2:
            scale.GenerateTensorValue(GeneratorTensor_2<ScaleDataType>{-5, 5}, num_thread);
            bias.GenerateTensorValue(GeneratorTensor_2<BiasDataType>{-5, 5}, num_thread);
            break;
        default:
            scale.GenerateTensorValue(GeneratorTensor_3<ScaleDataType>{-1.0f, 1.0f}, num_thread);
            bias.GenerateTensorValue(GeneratorTensor_3<BiasDataType>{-1.0f, 1.0f}, num_thread);
        }
    };

    // these buffers are usually provided by the user application
    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem y_dev(sizeof(XDataType) * y.mDesc.GetElementSpaceSize());
    DeviceMem scale_dev(sizeof(ScaleDataType) * scale.mDesc.GetElementSpaceSize());
    DeviceMem bias_dev(sizeof(BiasDataType) * bias.mDesc.GetElementSpaceSize());

    // estimatedMean_dev
    DeviceMem estimatedMean_dev(sizeof(MeanVarDataType) *
                                estimatedMean.mDesc.GetElementSpaceSize());
    // estimatedVariance_dev
    DeviceMem estimatedVariance_dev(sizeof(MeanVarDataType) *
                                    estimatedVariance.mDesc.GetElementSpaceSize());

    x_dev.ToDevice(x.mData.data());
    scale_dev.ToDevice(scale.mData.data());
    bias_dev.ToDevice(bias.mData.data());
    estimatedMean_dev.ToDevice(estimatedMean.mData.data());
    estimatedVariance_dev.ToDevice(estimatedVariance.mData.data());

    std::array<index_t, Rank> arrInOutLengths;
    std::array<index_t, Rank> arrInOutStrides;
    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarLengths;
    std::array<index_t, Rank - NumBatchNormReduceDim> arrScaleBiasMeanVarStrides;
    std::array<int, NumBatchNormReduceDim> arrReduceDims;

    std::copy(inOutLengths.begin(), inOutLengths.end(), arrInOutLengths.begin());
    std::copy(inOutStrides.begin(), inOutStrides.end(), arrInOutStrides.begin());
    std::copy(scaleBiasMeanVarLengths.begin(),
              scaleBiasMeanVarLengths.end(),
              arrScaleBiasMeanVarLengths.begin());
    std::copy(scaleBiasMeanVarStrides.begin(),
              scaleBiasMeanVarStrides.end(),
              arrScaleBiasMeanVarStrides.begin());

    std::copy(reduceDims.begin(), reduceDims.end(), arrReduceDims.begin());

    std::array<index_t, Rank> aligned_scaleBiasMeanVarStrides{0};

    int i = 0;
    for(auto dim : invariantDims)
    {
        assert(inOutLengths[dim] == scaleBiasMeanVarLengths[i]);

        aligned_scaleBiasMeanVarStrides[dim] = scaleBiasMeanVarStrides[i];
        i++;
    };

    using Normalize = ck::tensor_operation::element_wise::NormalizeInInfer;

    // add device batchnorm-infer instances
    using DeviceOp = ck::tensor_operation::device::DeviceElementwise<
        ck::Tuple<XDataType, MeanVarDataType, MeanVarDataType, ScaleDataType, BiasDataType>,
        ck::Tuple<YDataType>,
        Normalize,
        Rank>;

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
        using PassThroughOp = ck::tensor_operation::element_wise::PassThrough;

        using ReferenceBatchNormInferInstance =
            ck::tensor_operation::host::ReferenceBatchNormInfer<XDataType,
                                                                YDataType,
                                                                AccDataType,
                                                                ScaleDataType,
                                                                BiasDataType,
                                                                MeanVarDataType,
                                                                PassThroughOp,
                                                                Rank,
                                                                NumBatchNormReduceDim>;
        auto batchNormInfer_ref = ReferenceBatchNormInferInstance{};

        auto argument_ptr_ref =
            batchNormInfer_ref.MakeArgumentPointer(arrInOutLengths,
                                                   arrInOutStrides,
                                                   arrInOutStrides,
                                                   arrReduceDims,
                                                   arrScaleBiasMeanVarLengths,
                                                   arrScaleBiasMeanVarStrides,
                                                   arrScaleBiasMeanVarStrides,
                                                   arrScaleBiasMeanVarStrides,
                                                   x.mData.data(),
                                                   scale.mData.data(),
                                                   bias.mData.data(),
                                                   epsilon,
                                                   PassThroughOp{},
                                                   estimatedMean.mData.data(),
                                                   estimatedVariance.mData.data(),
                                                   y_ref.mData.data());

        if(!batchNormInfer_ref.IsSupportedArgument(argument_ptr_ref.get()))
        {
            std::cout << "The runtime parameters not supported by the reference instance, exiting!"
                      << std::endl;
            return (false);
        };

        auto invoker_ptr_ref = batchNormInfer_ref.MakeInvokerPointer();

        (void)invoker_ptr_ref->Run(argument_ptr_ref.get());
    }

    int num_kernel = 0;
    bool pass      = true;

    for(auto& inst_ptr : instance_ptrs)
    {
        auto argument_ptr = inst_ptr->MakeArgumentPointer(arrInOutLengths,
                                                          {arrInOutStrides,
                                                           aligned_scaleBiasMeanVarStrides,
                                                           aligned_scaleBiasMeanVarStrides,
                                                           aligned_scaleBiasMeanVarStrides,
                                                           aligned_scaleBiasMeanVarStrides},
                                                          {arrInOutStrides},
                                                          {x_dev.GetDeviceBuffer(),
                                                           estimatedMean_dev.GetDeviceBuffer(),
                                                           estimatedVariance_dev.GetDeviceBuffer(),
                                                           scale_dev.GetDeviceBuffer(),
                                                           bias_dev.GetDeviceBuffer()},
                                                          {y_dev.GetDeviceBuffer()},
                                                          Normalize{epsilon});

        if(inst_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            num_kernel++;
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
                std::cout << inst_ptr->GetTypeString()
                          << " skipped due to unsupported argument: " << std::endl;
            }

            continue;
        };

        auto invoker_ptr = inst_ptr->MakeInvokerPointer();

        float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        size_t num_bytes = 0;

        // inputing of x, scale, bias, outputing of y
        num_bytes += total_length * (sizeof(XDataType) + sizeof(YDataType)) +
                     invariant_length *
                         (sizeof(ScaleDataType) + sizeof(BiasDataType) + sizeof(MeanVarDataType));

        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        if(time_kernel)
            std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, "
                      << inst_ptr->GetTypeString() << std::endl;

        if(avg_time < best_avg_time)
        {
            best_instance_name = inst_ptr->GetTypeString();
            best_avg_time      = avg_time;
            best_gb_per_sec    = gb_per_sec;
        }

        if(do_verification)
        {
            using ck::utils::check_err;
            bool single_pass;

            y_dev.FromDevice(y.mData.data());

            if constexpr(ck::is_same_v<YDataType, ck::bhalf_t>)
                single_pass = check_err(y.mData, y_ref.mData, "y results", 1e-2, 1e-2);
            else
                single_pass = check_err(y.mData, y_ref.mData, "y results", 4e-3, 4e-3);

            pass = pass && single_pass;
        };

        if(do_dumpout)
        {
            using ck::host_common::dumpBufferToFile;

            // clang-format off
            dumpBufferToFile("dump_x.bin", x.mData.data(), x.mDesc.GetElementSize());
            dumpBufferToFile("dump_y.bin", y.mData.data(), y.mDesc.GetElementSize());
            dumpBufferToFile("dump_y_ref.bin", y_ref.mData.data(), y_ref.mDesc.GetElementSize());
            // clang-format off
        };
    }

    if(time_kernel)
    {
        std::cout << "best perf = " << best_avg_time << " ms, " << best_gb_per_sec << " GB/s, "
                  << best_instance_name << std::endl;
    }

    if(num_kernel == 0)
    {
        std::cout << "Error: No kernel is applicable" << std::endl;
        return false;
    }
    if (instance_index != -1)
    {
        std::cout << "batchnorm_infer_instance (" << instance_index << "/" << num_kernel
            << "): Passed" << std::endl;
    }
    return pass;
}

} // namespace profiler
} // namespace ck
