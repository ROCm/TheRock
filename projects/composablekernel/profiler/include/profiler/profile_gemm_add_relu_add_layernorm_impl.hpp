// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_layernorm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_layernorm.hpp"

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename D0DataType,
          typename D1DataType,
          typename EMeanVarDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename HDataType,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp,
          typename HElementOp>
void host_gemm_layernorm(Tensor<HDataType>& h_m_n,
                         const Tensor<ADataType>& a_m_k,
                         const Tensor<BDataType>& b_k_n,
                         const Tensor<D0DataType>& d0_m_n,
                         const Tensor<D1DataType>& d1_m_n,
                         const Tensor<GammaDataType>& gamma_n,
                         const Tensor<BetaDataType>& beta_n,
                         AElementOp a_element_op,
                         BElementOp b_element_op,
                         CDEElementOp cde_element_op,
                         HElementOp h_element_op,
                         int M,
                         int N,
                         AccDataType epsilon = 1e-5)
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using ReferenceGemm = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                    BDataType,
                                                                    AccDataType,
                                                                    AccDataType,
                                                                    AElementOp,
                                                                    BElementOp,
                                                                    PassThrough>;

    using ReferenceLayernorm = ck::tensor_operation::host::ReferenceLayernorm<EMeanVarDataType,
                                                                              GammaDataType,
                                                                              BetaDataType,
                                                                              HDataType,
                                                                              AccDataType,
                                                                              AccDataType,
                                                                              HElementOp,
                                                                              2,
                                                                              1>;

    Tensor<EMeanVarDataType> e_m_n(HostTensorDescriptor{M, N});
    Tensor<AccDataType> c_m_n(HostTensorDescriptor{M, N});
    Tensor<AccDataType> save_mean({M});
    Tensor<AccDataType> save_inv_std({M});

    auto ref_gemm         = ReferenceGemm{};
    auto ref_gemm_invoker = ref_gemm.MakeInvoker();

    auto ref_gemm_argument =
        ref_gemm.MakeArgument(a_m_k, b_k_n, c_m_n, a_element_op, b_element_op, PassThrough{});

    ref_gemm_invoker.Run(ref_gemm_argument);

    for(int n = 0; n < N; ++n)
    {
        for(int m = 0; m < M; ++m)
        {
            AccDataType e  = static_cast<AccDataType>(e_m_n(m, n));
            AccDataType d0 = static_cast<AccDataType>(d0_m_n(m, n));
            AccDataType d1 = static_cast<AccDataType>(d1_m_n(m, n));
            cde_element_op(e, c_m_n(m, n), d0, d1);
            e_m_n(m, n) = static_cast<EMeanVarDataType>(e);
        }
    }

    ReferenceLayernorm ref_layernorm;
    auto ref_layernorm_invoker = ref_layernorm.MakeInvoker();

    auto ref_layernorm_argument = ref_layernorm.MakeArgument(
        e_m_n, gamma_n, beta_n, h_m_n, save_mean, save_inv_std, h_element_op, {M, N}, {1}, epsilon);
    ref_layernorm_invoker.Run(ref_layernorm_argument);
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename D0DataType,
          typename D1DataType,
          typename EMeanVarDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename HDataType,
          typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename HLayout>
bool profile_gemm_add_relu_add_layernorm_impl(int do_verification,
                                              int init_method,
                                              bool /*do_log*/,
                                              bool time_kernel,
                                              int M,
                                              int N,
                                              int K,
                                              int StrideA,
                                              int StrideB,
                                              int StrideD0,
                                              int StrideD1,
                                              int StrideH,
                                              AccDataType epsilon = 1e-5)
{
    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor({len}, {stride});
    };

    auto f_host_tensor_descriptor2d = [](std::size_t row,
                                         std::size_t col,
                                         int& stride,
                                         auto layout) {
        using namespace ck::literals;

        if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
        {
            auto desc = HostTensorDescriptor({row, col}, {static_cast<std::size_t>(stride), 1_uz});
            if(stride <= 0)
                stride = desc.GetStrides()[0];
            return desc;
        }
        else
        {
            auto desc = HostTensorDescriptor({row, col}, {1_uz, static_cast<std::size_t>(stride)});
            if(stride <= 0)
                stride = desc.GetStrides()[1];
            return desc;
        }
    };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor2d(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor2d(K, N, StrideB, BLayout{}));
    Tensor<D1DataType> d0_m_n(f_host_tensor_descriptor2d(M, N, StrideD0, D0Layout{}));
    Tensor<D1DataType> d1_m_n(f_host_tensor_descriptor2d(M, N, StrideD1, D1Layout{}));
    Tensor<GammaDataType> gamma_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<BetaDataType> beta_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<HDataType> h_m_n(f_host_tensor_descriptor2d(M, N, StrideH, HLayout{}));
    Tensor<HDataType> h_m_n_host(f_host_tensor_descriptor2d(M, N, StrideH, HLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "d0_m_n: " << d0_m_n.mDesc << std::endl;
    std::cout << "d1_m_n: " << d1_m_n.mDesc << std::endl;
    std::cout << "h_m_n: " << h_m_n.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-1, 1});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-1, 1});
        d0_m_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{-1, 1});
        d1_m_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{-1, 1});
        gamma_n.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{-1, 1});
        beta_n.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{-1, 1});
        break;
    }

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using AddReluAdd  = ck::tensor_operation::element_wise::AddReluAdd;

    using AElementOp   = PassThrough;
    using BElementOp   = PassThrough;
    using CDEElementOp = AddReluAdd;
    using HElementOp   = PassThrough;

    const auto a_element_op   = AElementOp{};
    const auto b_element_op   = BElementOp{};
    const auto cde_element_op = CDEElementOp{};
    const auto h_element_op   = HElementOp{};

    using DeviceOp = ck::tensor_operation::device::DeviceGemmMultipleDLayernorm<
        ALayout,
        BLayout,
        ck::Tuple<D0Layout, D1Layout>,
        HLayout,
        ADataType,
        BDataType,
        ck::Tuple<D0DataType, D1DataType>,
        GammaDataType,
        BetaDataType,
        HDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::AddReluAdd,
        ck::tensor_operation::element_wise::PassThrough>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    // run reference
    if(do_verification)
    {
        host_gemm_layernorm<ADataType,
                            BDataType,
                            AccDataType,
                            D0DataType,
                            D1DataType,
                            EMeanVarDataType,
                            GammaDataType,
                            BetaDataType,
                            HDataType>(h_m_n_host,
                                       a_m_k,
                                       b_k_n,
                                       d0_m_n,
                                       d1_m_n,
                                       gamma_n,
                                       beta_n,
                                       a_element_op,
                                       b_element_op,
                                       cde_element_op,
                                       h_element_op,
                                       M,
                                       N,
                                       epsilon);
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d0_m_n_device_buf(sizeof(D0DataType) * d0_m_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_m_n_device_buf(sizeof(D1DataType) * d1_m_n.mDesc.GetElementSpaceSize());
    DeviceMem gamma_device_buf(sizeof(GammaDataType) * gamma_n.mDesc.GetElementSpaceSize());
    DeviceMem beta_device_buf(sizeof(BetaDataType) * beta_n.mDesc.GetElementSpaceSize());
    DeviceMem h_device_buf(sizeof(HDataType) * h_m_n.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    d0_m_n_device_buf.ToDevice(d0_m_n.mData.data());
    d1_m_n_device_buf.ToDevice(d1_m_n.mData.data());
    gamma_device_buf.ToDevice(gamma_n.mData.data());
    beta_device_buf.ToDevice(beta_n.mData.data());

    std::string best_op_name;
    float best_ave_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    bool pass      = true;
    int num_kernel = 0;

    // profile device operation instances
    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            a_device_buf.GetDeviceBuffer(),
            b_device_buf.GetDeviceBuffer(),
            {d0_m_n_device_buf.GetDeviceBuffer(), d1_m_n_device_buf.GetDeviceBuffer()},
            gamma_device_buf.GetDeviceBuffer(),
            beta_device_buf.GetDeviceBuffer(),
            h_device_buf.GetDeviceBuffer(),
            M,
            N,
            K,
            StrideA,
            StrideB,
            {StrideD0, StrideD1},
            StrideH,
            epsilon,
            a_element_op,
            b_element_op,
            cde_element_op,
            h_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            ++num_kernel;

            size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
            DeviceMem workspace_dev(workspace_sz);
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

            // re-init E to zero before profiling a kernel
            h_device_buf.SetZero();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t num_byte =
                sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                (sizeof(D0DataType) + sizeof(D1DataType) + sizeof(HDataType)) * M * N +
                (sizeof(GammaDataType) + sizeof(BetaDataType)) * N;

            float gb_per_sec = num_byte / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << gb_per_sec << " GB/s, "
                      << op_name << std::endl;

            if(ave_time < best_ave_time)
            {
                best_op_name    = op_name;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                h_device_buf.FromDevice(h_m_n.mData.data());

                pass = pass && ck::utils::check_err(
                                   h_m_n, h_m_n_host, "Error: Incorrect results h_m_n", 1e-2, 1e-2);
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    if(num_kernel == 0)
    {
        std::cout << "Error: No kernel is applicable" << std::endl;
        pass = false;
    }
    else
    {
        if(time_kernel)
            std::cout << "Best Perf: " << best_ave_time << " ms, " << best_gb_per_sec << " GB/s, "
                      << best_op_name << std::endl;
    }

    return pass;
}

} // namespace profiler
} // namespace ck
