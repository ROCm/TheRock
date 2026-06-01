// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>
#include <limits>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/contraction_bilinear.hpp"
#include "ck/library/tensor_operation_instance/gpu/contraction_scale.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_contraction.hpp"
#include "ck/library/utility/numeric.hpp"

#include "ck/host_utility/io.hpp"

namespace ck {
namespace profiler {

using Bilinear = ck::tensor_operation::element_wise::Bilinear;
using Scale    = ck::tensor_operation::element_wise::Scale;

using F32 = float;
using F64 = double;

template <index_t NumDimMNK,
          typename ALayout,
          typename BLayout,
          typename CDELayout,
          typename DataType,
          typename ComputeDataType,
          typename DTupleDataType,
          typename CDElementOp>
int profile_contraction_impl(ck::index_t do_verification,
                             ck::index_t init_method,
                             bool do_log,
                             bool time_kernel,
                             CDElementOp cde_element_op,
                             const std::vector<ck::index_t>& M,
                             const std::vector<ck::index_t>& N,
                             const std::vector<ck::index_t>& K,
                             const std::vector<ck::index_t>& StridesA, // [M0, M1, K0, K1]
                             const std::vector<ck::index_t>& StridesB, // [N0, N1, K0, K1]
                             const std::vector<ck::index_t>& StridesE, // [M0, M1, N0, N1]
                             const std::vector<ck::index_t>& StridesD, // [M0, M1, N0, N1]
                             int instance_index = -1)
{
    bool pass = true;

    auto f_host_tensor_descriptor = [](const std::vector<ck::index_t>& dims01,
                                       const std::vector<ck::index_t>& dims23,
                                       const std::vector<ck::index_t>& strides,
                                       auto layout) {
        std::vector<std::size_t> dims_szt(dims01.begin(), dims01.end());
        dims_szt.insert(dims_szt.end(), dims23.begin(), dims23.end());

        // For ColumnMajor with more than 2 dimensions, the strides are custom-defined, so skip
        // verification.
        if constexpr(ck::is_same_v<decltype(layout), ck::tensor_layout::gemm::ColumnMajor>)
        {
            if(strides.size() > 2)
            {
                return HostTensorDescriptor(
                    dims_szt, strides, ck::tensor_layout::BypassLayoutVerification{});
            }
        }
        return HostTensorDescriptor(dims_szt, strides, layout);
    };

    Tensor<DataType> a_m_k(f_host_tensor_descriptor(M, K, StridesA, ALayout{}));
    Tensor<DataType> b_n_k(f_host_tensor_descriptor(N, K, StridesB, BLayout{}));
    Tensor<DataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StridesE, CDELayout{}));
    Tensor<DataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StridesE, CDELayout{}));
    Tensor<DataType> d_m_n(f_host_tensor_descriptor(M, N, StridesD, CDELayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_n_k: " << b_n_k.mDesc << std::endl;
    std::cout << "d_m_n: " << d_m_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_device_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<DataType>{-5, 5});
        b_n_k.GenerateTensorValue(GeneratorTensor_2<DataType>{-5, 5});
        d_m_n.GenerateTensorValue(GeneratorTensor_2<DataType>{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<DataType>{0.0, 1.0});
        b_n_k.GenerateTensorValue(GeneratorTensor_3<DataType>{-0.5, 0.5});
        d_m_n.GenerateTensorValue(GeneratorTensor_3<DataType>{-0.5, 0.5});
    }

    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;

    DeviceMem a_device_buf(sizeof(DataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(DataType) * b_n_k.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(DataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(DataType) * d_m_n.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_n_k.mData.data());
    e_device_buf.SetZero();
    d_device_buf.ToDevice(d_m_n.mData.data());

    auto merge_dims = [](const std::vector<ck::index_t>& dims01,
                         const std::vector<ck::index_t>& dims23) {
        std::vector<ck::index_t> dims_szt(dims01.begin(), dims01.end());
        dims_szt.insert(dims_szt.end(), dims23.begin(), dims23.end());
        return dims_szt;
    };

    const std::vector<index_t> a_ms_ks_lengths = merge_dims(M, K);
    const std::vector<index_t> b_ns_ks_lengths = merge_dims(N, K);
    const std::vector<index_t> e_ms_ns_lengths = merge_dims(M, N);
    const std::vector<index_t> d_m_n_lengths   = merge_dims(M, N);

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};

    using DeviceOp = ck::tensor_operation::device::DeviceContractionMultipleD<NumDimMNK,
                                                                              NumDimMNK,
                                                                              NumDimMNK,
                                                                              DataType,
                                                                              DataType,
                                                                              DTupleDataType,
                                                                              DataType,
                                                                              AElementOp,
                                                                              BElementOp,
                                                                              CDElementOp,
                                                                              ComputeDataType>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    using AccDataType =
        typename std::conditional<std::is_same<ComputeDataType, F64>::value, F64, F32>::type;

    // Run reference op
    if(do_verification)
    {
        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceContraction_M2_N2_K2<NumDimMNK,
                                                                      NumDimMNK,
                                                                      NumDimMNK,
                                                                      DataType,
                                                                      DataType,
                                                                      DataType,
                                                                      AccDataType,
                                                                      ComputeDataType,
                                                                      AElementOp,
                                                                      BElementOp>;

        auto ref_op      = ReferenceGemmInstance{};
        auto ref_invoker = ref_op.MakeInvoker();

        Tensor<DataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StridesE, CDELayout{}));

        auto ref_argument =
            ref_op.MakeArgument(a_m_k, b_n_k, c_m_n_host_result, a_element_op, b_element_op);

        ref_invoker.Run(ref_argument);

        e_m_n_host_result.ForEach([&](auto& self, auto idx) {
            if constexpr(is_same<CDElementOp, Bilinear>::value)
            {
                cde_element_op(self(idx), c_m_n_host_result(idx), d_m_n(idx));
            }
            else if constexpr(is_same<CDElementOp, Scale>::value)
            {
                cde_element_op(self(idx), c_m_n_host_result(idx));
            }
            else
            {
                static_assert("Unsupported CDElementOp in contraction profiler.");
            }
        });
    }

    std::string best_op_name;
    float best_avg_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    int num_kernel        = 0;
    // profile device op instances
    for(auto& op_ptr : op_ptrs)
    {
        std::unique_ptr<tensor_operation::device::BaseArgument> argument_ptr;
        if constexpr(is_same<CDElementOp, Bilinear>::value)
        {
            argument_ptr = op_ptr->MakeArgumentPointer(
                static_cast<DataType*>(a_device_buf.GetDeviceBuffer()),
                static_cast<DataType*>(b_device_buf.GetDeviceBuffer()),
                std::array<const void*, 1>{d_device_buf.GetDeviceBuffer()},
                static_cast<DataType*>(e_device_buf.GetDeviceBuffer()),
                a_ms_ks_lengths,
                StridesA,
                b_ns_ks_lengths,
                StridesB,
                std::array<std::vector<ck::index_t>, 1>{d_m_n_lengths},
                std::array<std::vector<ck::index_t>, 1>{StridesD},
                e_ms_ns_lengths,
                StridesE,
                a_element_op,
                b_element_op,
                cde_element_op);
        }
        else if constexpr(is_same<CDElementOp, Scale>::value)
        {
            argument_ptr =
                op_ptr->MakeArgumentPointer(static_cast<DataType*>(a_device_buf.GetDeviceBuffer()),
                                            static_cast<DataType*>(b_device_buf.GetDeviceBuffer()),
                                            std::array<const void*, 0>{},
                                            static_cast<DataType*>(e_device_buf.GetDeviceBuffer()),
                                            a_ms_ks_lengths,
                                            StridesA,
                                            b_ns_ks_lengths,
                                            StridesB,
                                            std::array<std::vector<ck::index_t>, 0>{},
                                            std::array<std::vector<ck::index_t>, 0>{},
                                            e_ms_ns_lengths,
                                            StridesE,
                                            a_element_op,
                                            b_element_op,
                                            cde_element_op);
        }
        else
        {
            static_assert("Unsupported CDElementOp in contraction profiler.");
        }

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        auto nelems_m = ck::accumulate_n<ck::index_t>(
            a_ms_ks_lengths.begin(), NumDimMNK, 1, std::multiplies<>{});
        auto nelems_n = ck::accumulate_n<ck::index_t>(
            b_ns_ks_lengths.begin(), NumDimMNK, 1, std::multiplies<>{});
        auto nelems_k = ck::accumulate_n<ck::index_t>(
            a_ms_ks_lengths.begin() + NumDimMNK, NumDimMNK, 1, std::multiplies<>{});

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            ++num_kernel;
            if((instance_index != -1) && (instance_index + 1 != num_kernel))
            {
                // skip test if instance_index is specified
                continue;
            }
            // re-init C to zero before profiling next kernel
            e_device_buf.SetZero();

            std::string op_name = op_ptr->GetTypeString();

            float avg_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop = std::size_t(2) * nelems_m * nelems_n * nelems_k;

            std::size_t num_btype = sizeof(DataType) * nelems_m * nelems_k +
                                    sizeof(DataType) * nelems_k * nelems_n +
                                    sizeof(DataType) * nelems_m * nelems_n;

            float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

            float gb_per_sec = num_btype / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                e_device_buf.FromDevice(e_m_n_device_result.mData.data());

                // Both the kernel and the reference use `AccDataType`, so an absolute error of both
                // of them is bounded by `nelems_k * std::numeric_limits<AccDataType>::epsilon()`.
                // Comparing one to another can result in an absolute error as high as twice that
                // value.
                double threshold = 2 * nelems_k * std::numeric_limits<AccDataType>::epsilon();
                // Handle the possible casting error of either AccDataType -> DataType or
                // DataType -> ComputeDataType.
                // TODO: Add a generic solution for calculating thresholds in CK.
                if constexpr(ck::is_same_v<DataType, ck::bhalf_t> ||
                             ck::is_same_v<ComputeDataType, ck::bhalf_t>)
                {
                    const double epsilon = std::pow(2, -7);
                    // Maximum relative casting error when rounding to zero.
                    threshold += epsilon * 2;
                }
                else if constexpr(ck::is_same_v<DataType, ck::half_t> ||
                                  ck::is_same_v<ComputeDataType, ck::half_t>)
                {
                    const double epsilon = std::pow(2, -10);
                    // Maximum relative casting error when rounding to zero.
                    threshold += epsilon * 2;
                }

                pass = pass & ck::utils::check_err(e_m_n_device_result,
                                                   e_m_n_host_result,
                                                   "Error: incorrect results!",
                                                   threshold,
                                                   threshold);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_n_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host  : ", e_m_n_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "c_device: ", e_m_n_device_result.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    }

    if constexpr(is_same<DataType, float>::value)
    {
        std::cout << "Best Perf for datatype = f32";
    }
    else if constexpr(is_same<DataType, double>::value)
    {
        std::cout << "Best Perf for datatype = f64";
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

    if constexpr(is_same<CDELayout, tensor_layout::gemm::RowMajor>::value)
    {
        std::cout << " CDELayout =  RowMajor";
    }
    else if constexpr(is_same<CDELayout, tensor_layout::gemm::ColumnMajor>::value)
    {
        std::cout << " CDELayout =  ColumnMajor";
    }

    std::cout << " M = " << M << " N = " << N << " K = " << K << " StridesA = " << StridesA
              << " StridesB = " << StridesB << " StridesE = " << StridesE << " : " << best_avg_time
              << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec << " GB/s, "
              << best_op_name << std::endl;

    if(instance_index != -1)
    {
        std::cout << "contraction_instance (" << instance_index << "/" << num_kernel << "): Passed"
                  << std::endl;
    }
    return pass;
}

} // namespace profiler
} // namespace ck
