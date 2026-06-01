// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using I8  = int8_t;
using I32 = int32_t;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using ActivationOp = PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::Activation_Mul_Clamp<ActivationOp>;

using ADataType        = I8;
using BDataType        = I8;
using AccDataType      = I32;
using CShuffleDataType = I32;
using DsDataType       = ck::Tuple<>;
using EDataType        = I8;

using ALayout  = Col;
using BLayout  = Row;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmMultipleD_Wmma_CShuffleV3<
    ALayout,
    BLayout,
    DsLayout,
    ELayout,
    ADataType,
    BDataType,
    DsDataType,
    EDataType,
    AccDataType,
    CShuffleDataType,
    ActivationOp,
    ActivationOp,
    CDEElementOp,
    GemmDefault,
    256,
    128,
    128,
    64,
    8,
    8,
    16,
    16,
    4,
    2,
    S<4, 64, 1>,
    S<0, 2, 1>,
    S<0, 2, 1>,
    1,
    1,
    8,
    true,
    S<4, 64, 1>,
    S<0, 2, 1>,
    S<0, 2, 1>,
    1,
    1,
    8,
    true,
    1,
    1,
    S<1, 32, 1, 8>,
    S<1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    I8,
    I8>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, EDataType, float, PassThrough, PassThrough, CDEElementOp>;

int main(int /* argc */, char* /* argv */[])
{
    bool do_verification = true;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA = K;
    ck::index_t StrideB = N;
    ck::index_t StrideE = N;

    float requant_scale = 0.03;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;

    a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
    b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

    auto a_element_op   = PassThrough{};
    auto b_element_op   = PassThrough{};
    auto cde_element_op = CDEElementOp{requant_scale, ActivationOp{}};

    // device GEMM
    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    auto argument = gemm.MakeArgument(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                      std::array<const void*, 0>{},
                                      static_cast<EDataType*>(e_device_buf.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      std::array<ck::index_t, 0>{},
                                      StrideE,
                                      1,
                                      a_element_op,
                                      b_element_op,
                                      cde_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    if(do_verification)
    {
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, e_m_n_host_result, a_element_op, b_element_op, cde_element_op);

        ref_invoker.Run(ref_argument);

        return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result) ? 0 : 1;
    }

    return 0;
}
