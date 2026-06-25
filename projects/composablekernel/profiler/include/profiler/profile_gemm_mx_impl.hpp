// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>

#include "ck/ck.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_mx_gemm.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_mx.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

namespace ck {
namespace profiler {

#if 1
template <bool KLast>
void preShuffleScaleBuffer(ck::e8m0_bexp_t* src, ck::e8m0_bexp_t* dst, int MN, int K)
{
    int MNXdlPack = 2;
    int KXdlPack  = 2;

    int XdlMNThread = 16;
    int XdlKThread  = 64 / XdlMNThread;

    int K0 = K / KXdlPack / XdlKThread; // KRepeat

    // The 4 16x128 building blocks will be packed into 1 32x256 for F4
    // The 8 16x16x128 mfma will be packed into 1 32x32x256 for F4

    // unfold the MN32xK(256/32) scale buffer
    //    4            16             2           2
    // To XdlKThread-> XdlMNThread -> KXdlPack -> MNXdlPack
    // Then, MNRepeat->KRepeat

    for(int n = 0; n < MN; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0    = n / (XdlMNThread * MNXdlPack); // i MNRepeat
            int tempn = n % (XdlMNThread * MNXdlPack);
            int n1    = tempn % XdlMNThread; // i XdlMNThread
            int n2    = tempn / XdlMNThread; // i MNXdlPack

            int k0    = k / (XdlKThread * KXdlPack); // i KRepeat
            int tempk = k % (XdlKThread * KXdlPack);
            int k1    = tempk % XdlKThread; // i XdlKThread
            int k2    = tempk / XdlKThread; // i KXdlPack

            int outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                              k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                              k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
                              k2 * MNXdlPack + n2;
            // src[n * K + k] = ck::type_convert<ck::e8m0_bexp_t>(static_cast<float>(powf(2.0f, n2 +
            // k2 * MNXdlPack)));
            if constexpr(KLast)
                dst[outputIndex] = src[n * K + k];
            else
                dst[outputIndex] = src[k * MN + n];
        }
    }
}

void preShuffleBuffer(const ck::f4x2_pk_t* src, ck::f4x2_pk_t* dst, int N, int K, int NXdl)
{
    int KPack = 16;
    int NLane = NXdl;
    int KLane = 64 / NLane;
    int K_pk  = K / 2;
    int K0    = K_pk / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K_pk; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K_pk + k];
        }
    }
}
#endif

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          int ScaleBlockSize>
bool profile_gemm_mx_impl(int do_verification,
                          int init_method,
                          bool do_log,
                          bool time_kernel,
                          int M,
                          int N,
                          int K,
                          int StrideA,
                          int StrideB,
                          int StrideC,
                          int KBatch,
                          int n_warmup,
                          int n_iter,
                          uint64_t rotating = 0)
{
    using tensor_operation::device::instance::Col;
    using tensor_operation::device::instance::E8M0;
    using tensor_operation::device::instance::E8M0PK;
    using tensor_operation::device::instance::MFMA;
    using tensor_operation::device::instance::Row;

    constexpr bool BPreShuffle = is_same_v<BLayout, MFMA>;
    using BRefLayout           = conditional_t<BPreShuffle, Col, BLayout>;

    if(K % ScaleBlockSize != 0)
    {
        throw std::runtime_error("wrong! K must be multiple of ScaleBlockSize.");
    };

    using XDataType       = E8M0;
    using XPackedDataType = E8M0PK;
    using AScaleLayout    = Row;
    using BScaleLayout    = Col;

    auto f_host_tensor_descriptor =
        [](ck::index_t row, ck::index_t col, ck::index_t stride, auto layout) {
            using namespace ck::literals;

            if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
                return HostTensorDescriptor({row, col}, {stride, 1});
            else
                return HostTensorDescriptor({row, col}, {1, stride});
        };
    auto f_get_default_stride =
        [](ck::index_t row, ck::index_t col, ck::index_t stride, auto layout) {
            if(stride == -1)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                    return static_cast<ck::index_t>(col);
                else
                    return static_cast<ck::index_t>(row);
            }
            else
                return static_cast<ck::index_t>(stride);
        };

    auto Scale_Padded_M = (M + 32 - 1) / 32 * 32;
    auto Scale_Stride_AM =
        f_get_default_stride(Scale_Padded_M, K / ScaleBlockSize, -1, AScaleLayout{});
    auto Scale_Stride_BN = f_get_default_stride(K / ScaleBlockSize, N, -1, BScaleLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    auto b_k_n =
        std::make_shared<Tensor<BDataType>>(f_host_tensor_descriptor(K, N, StrideB, BRefLayout{}));
    auto b_input = b_k_n;
    if constexpr(BPreShuffle)
        b_input = std::make_shared<Tensor<BDataType>>(
            f_host_tensor_descriptor(K, N, StrideB, BRefLayout{})); // use layout only for size

    // scales for A and B
    Tensor<XDataType> a_m_k_scale(f_host_tensor_descriptor(
        Scale_Padded_M, K / ScaleBlockSize, Scale_Stride_AM, AScaleLayout{}));
    Tensor<XDataType> b_k_n_scale(
        f_host_tensor_descriptor(K / ScaleBlockSize, N, Scale_Stride_BN, BScaleLayout{}));

    // shuffled scales for A and B
    Tensor<XDataType> a_shuffled_scale(f_host_tensor_descriptor(
        Scale_Padded_M, K / ScaleBlockSize, Scale_Stride_AM, AScaleLayout{}));
    Tensor<XDataType> b_shuffled_scale(
        f_host_tensor_descriptor(K / ScaleBlockSize, N, Scale_Stride_BN, BScaleLayout{}));

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::size_t total_gemm_needed =
        a_m_k.GetElementSpaceSizeInBytes() + b_k_n->GetElementSpaceSizeInBytes() +
        a_m_k_scale.GetElementSpaceSizeInBytes() + b_k_n_scale.GetElementSpaceSizeInBytes() +
        a_shuffled_scale.GetElementSpaceSizeInBytes() +
        b_shuffled_scale.GetElementSpaceSizeInBytes();
    int rotating_count = std::max(
        1,
        std::min(n_iter,
                 static_cast<int>(std::ceil(static_cast<double>(rotating) / total_gemm_needed))));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "a_m_k_scale: " << a_m_k_scale.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n->mDesc << std::endl;
    std::cout << "b_k_n_scale: " << b_k_n_scale.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_device_result.mDesc << std::endl;
    std::cout << "rotating count: " << rotating_count << std::endl;

    auto a_data_element = [](float x) {
        if constexpr(ck::is_same_v<ADataType, ck::f4x2_pk_t>)
            return ck::type_convert<ADataType>(ck::float2_t(x));
        else if constexpr(ck::packed_size_v<ADataType> == 32)
            return ck::type_convert<ADataType>(ck::float32_t(x));
        else if constexpr(ck::packed_size_v<ADataType> == 16)
            return ck::type_convert<ADataType>(ck::float16_t(x));
        else
            return ck::type_convert<ADataType>(x);
    };
    auto b_data_element = [](float x) {
        if constexpr(ck::is_same_v<BDataType, ck::f4x2_pk_t>)
            return ck::type_convert<BDataType>(ck::float2_t(x));
        else if constexpr(ck::packed_size_v<BDataType> == 32)
            return ck::type_convert<BDataType>(ck::float32_t(x));
        else if constexpr(ck::packed_size_v<BDataType> == 16)
            return ck::type_convert<BDataType>(ck::float16_t(x));
        else
            return ck::type_convert<BDataType>(x);
    };

    using int_distr   = std::uniform_int_distribution<int>;
    using float_distr = std::uniform_real_distribution<float>;
    switch(init_method)
    {
    case 0: // Initializations for development and debugging
        ck::utils::FillConstant<ADataType>{a_data_element(1.0f)}(a_m_k);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(2.0f)}(a_m_k_scale);
        ck::utils::FillConstant<BDataType>{b_data_element(0.5f)}(*b_k_n);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(1.0f)}(b_k_n_scale);
        if(do_log)
        {
            std::cout << "Init A = {1}" << std::endl;
            std::cout << "Init A scale = {2.0}" << std::endl;
            std::cout << "Init B = {0.5}" << std::endl;
            std::cout << "Init B scale = {1.0}" << std::endl;
            std::cout << "Expect C = {K}" << std::endl;
        }
        break;

    case 1:

        a_m_k.GenerateTensorDistr(
            int_distr{-4, 4}, ck::identity{}, std::minstd_rand(time(nullptr))); // Z[-4,4]
        b_k_n->GenerateTensorDistr(int_distr{-4, 4});                           // Z[-4,4]

        a_m_k_scale.GenerateTensorDistr(int_distr{125, 128}); // scales: {0.25, 0.5, 1, 2}
        b_k_n_scale.GenerateTensorDistr(int_distr{125, 128}); // scales: {0.25, 0.5, 1, 2}
        break;

    default:
        a_m_k.GenerateTensorDistr(
            float_distr{-2.0, 2.0}, ck::identity{}, std::minstd_rand(time(nullptr)));
        a_m_k_scale.GenerateTensorDistr(float_distr{powf(2.0f, -125.0f), 1.0f});

        b_k_n->GenerateTensorDistr(float_distr{-2.0, 2.0});
        b_k_n_scale.GenerateTensorDistr(float_distr{powf(2.0f, -125.0f), 1.0f});
        break;
    }

#if 1
    preShuffleScaleBuffer<ck::is_same_v<ALayout, Row>>(a_m_k_scale.mData.data(),
                                                       a_shuffled_scale.mData.data(),
                                                       Scale_Padded_M,
                                                       K / ScaleBlockSize);
    preShuffleScaleBuffer<ck::is_same_v<BRefLayout, Col>>(
        b_k_n_scale.mData.data(), b_shuffled_scale.mData.data(), N, K / ScaleBlockSize);
    if constexpr(BPreShuffle)
    {
        int NPerXdl = 16; // Fixed 16
        preShuffleBuffer(b_k_n->mData.data(), b_input->mData.data(), N, K, NPerXdl);
    }
#endif

    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    if(do_log > 0)
        std::cout << "Device memory allocation..." << std::endl;
    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.GetElementSpaceSize());
    DeviceMem a_scale_device_buf(sizeof(XDataType) * a_m_k_scale.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n->GetElementSpaceSize());
    DeviceMem b_scale_device_buf(sizeof(XDataType) * b_k_n_scale.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.GetElementSpaceSize());

    if(do_log > 0)
        std::cout << "Upload data to device..." << std::endl;
    a_device_buf.ToDevice(a_m_k.mData.data());
    a_scale_device_buf.ToDevice(a_shuffled_scale.mData.data());
    b_device_buf.ToDevice(b_input->mData.data());
    b_scale_device_buf.ToDevice(b_shuffled_scale.mData.data());

    if(do_log > 0)
        std::cout << "Done." << std::endl;

    using DeviceOp = ck::tensor_operation::device::DeviceGemmMX<ALayout,
                                                                BLayout,
                                                                CLayout,
                                                                ADataType,
                                                                XPackedDataType,
                                                                BDataType,
                                                                XPackedDataType,
                                                                CDataType,
                                                                ScaleBlockSize,
                                                                AElementOp,
                                                                BElementOp,
                                                                CElementOp>;
    std::cout << "finding op instances..." << std::endl;
    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    // Run reference GEMM
    if(do_verification)
    {
        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceMXGemm< //
            ADataType,
            BDataType,
            CDataType,
            float, // AccDataType
            XDataType,
            AElementOp,
            BElementOp,
            CElementOp,
            float, // ComputeTypeA
            float  // ComputeTypeB
            >;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_m_k,
                                                  a_m_k_scale,
                                                  *b_k_n,
                                                  b_k_n_scale,
                                                  c_m_n_host_result,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op);

        ref_invoker.Run(ref_argument);
    }

    std::string best_op_name;
    std::optional<std::string> best_op_object_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    float best_kbatch     = 0;
    bool pass             = true;

    // profile device GEMM instances
    for(auto& op_ptr : op_ptrs)
    {
        std::vector<int> kbatch_list = {1, 2, 4, 8, 16, 19, 32, 38}; // use these when KBatch <= 0

        if(KBatch > 0)
        {
            kbatch_list = {KBatch};
        }

        for(std::size_t i = 0; i < kbatch_list.size(); i++)
        {
            auto kbatch_curr = kbatch_list[i];

            auto argument_ptr = op_ptr->MakeArgumentPointer(
                static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                static_cast<XPackedDataType*>(a_scale_device_buf.GetDeviceBuffer()),
                static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                static_cast<XPackedDataType*>(b_scale_device_buf.GetDeviceBuffer()),
                static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                M,
                N,
                K,
                StrideA,
                Scale_Stride_AM,
                StrideB,
                Scale_Stride_BN,
                StrideC,
                kbatch_curr,
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

                    if(do_log)
                    {

                        if(init_method == 0)
                        {
                            auto expected = static_cast<float>(K);
                            auto computed = type_convert<float>(c_m_n_device_result(0, 12));

                            pass = pass & (std::abs(expected - computed) <= 0.0f);
                            std::cout << "\nExpected vs Computed: " << expected << " vs "
                                      << computed << ((pass) ? " (PASSED!)" : " (FAILED!)")
                                      << std::endl
                                      << std::endl;
                        }
                        else
                        {
                            if constexpr(is_same_v<ADataType, ck::f8_t> ||
                                         is_same_v<ADataType, ck::bf8_t>)
                                LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",")
                                    << "\n";
                            else
                                std::cout << "A: WIP PRINT PACKED TYPE\n";
                            LogRangeAsType<float>(std::cout << "a_scale : ", a_m_k_scale.mData, ",")
                                << "\n";
                            if constexpr(is_same_v<BDataType, ck::f8_t> ||
                                         is_same_v<BDataType, ck::bf8_t>)
                                LogRangeAsType<float>(std::cout << "b : ", b_k_n->mData, ",")
                                    << "\n";
                            else
                                std::cout << "B: WIP PRINT PACKED TYPE\n";
                            LogRangeAsType<float>(std::cout << "b_scale: ", b_k_n_scale.mData, ",")
                                << "\n";
                            LogRangeAsType<float>(
                                std::cout << "c_host  : ", c_m_n_host_result.mData, ",")
                                << "\n";
                            LogRangeAsType<float>(
                                std::cout << "c_device: ", c_m_n_device_result.mData, ",")
                                << std::endl;
                        }
                    }

                    pass = pass & ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);
                }

                std::string op_name                    = op_ptr->GetTypeString();
                std::optional<std::string> op_obj_name = op_ptr->GetObjectName();

                float ave_time = invoker_ptr->Run(argument_ptr.get(),
                                                  StreamConfig{nullptr,
                                                               time_kernel,
                                                               0,
                                                               n_warmup,
                                                               n_iter,
                                                               rotating_count > 1,
                                                               rotating_count});

                // Output size(M*N) * [dot product(2K) + product of scales(K/ScaleBlockSize) +
                // scaling of partial sums(K/ScaleBlockSize)]
                // FLOPS = 2 * M * N * K + 2 * M * N * K / ScaleBlockSize
                std::size_t flop =
                    std::size_t(2) * M * N * K + std::size_t(2) * M * N * K / ScaleBlockSize;

                // TODO: fp6?
                std::size_t num_btype = sizeof(ADataType) * M * K / packed_size_v<ADataType> +
                                        sizeof(BDataType) * K * N / packed_size_v<BDataType> +
                                        sizeof(CDataType) * M * N +
                                        sizeof(XDataType) * (M * K + K * N) / ScaleBlockSize;

                float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

                float gb_per_sec = num_btype / 1.E6 / ave_time;

                std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops
                          << " TFlops, " << gb_per_sec << " GB/s, " << op_name << ", KBatch "
                          << kbatch_curr << std::endl;

                if(tflops > best_tflops && ave_time > 1e-10)
                {
                    best_op_name        = op_name;
                    best_op_object_name = op_obj_name;
                    best_tflops         = tflops;
                    best_ave_time       = ave_time;
                    best_gb_per_sec     = gb_per_sec;
                    best_kbatch         = kbatch_curr;
                }
            }
            else
            {
                std::cout << op_ptr->GetTypeString() << " does not support this problem"
                          << std::endl;
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
    std::cout << " ALayout = " << ALayout::name;
    std::cout << " BLayout = " << BLayout::name;
    std::cout << " CLayout = " << CLayout::name;

    std::cout << " M = " << M << " N = " << N << " K = " << K << " StrideA = " << StrideA
              << " StrideB = " << StrideB << " StrideC = " << StrideC << " KBatch = " << best_kbatch
              << " : " << best_ave_time << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec
              << " GB/s, " << best_op_name << std::endl;

    if(best_op_object_name)
        std::cout << best_op_object_name.value() << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
