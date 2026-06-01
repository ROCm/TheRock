// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using I8   = int8_t;
using F32  = float;

using Row    = ck::tensor_layout::gemm::RowMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;

using A0DataType       = BF16;
using AsDataType       = ck::Tuple<A0DataType>;
using B0DataType       = I8;
using B1DataType       = BF16;
using BsDataType       = ck::Tuple<B0DataType, B1DataType>;
using AccDataType      = F32;
using CShuffleDataType = BF16;
using D0DataType       = BF16;
using DsDataType       = ck::Tuple<D0DataType>;
using EDataType        = BF16;

using A0Layout = Row;
using AsLayout = ck::Tuple<A0Layout>;
using B0Layout = Row;
using B1Layout = B0Layout;
using BsLayout = ck::Tuple<B0Layout, B1Layout>;
using D0Layout = Row;
using DsLayout = ck::Tuple<D0Layout>;
using ELayout  = Row;

using Multiply    = ck::tensor_operation::element_wise::Multiply;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddFastGelu = ck::tensor_operation::element_wise::AddFastGelu;

using AElementOp   = PassThrough;
using BElementOp   = Multiply;
using CDEElementOp = AddFastGelu;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultipleABD_Wmma_CShuffleV3<
    AsLayout,
    BsLayout,
    DsLayout,
    ELayout,
    AsDataType,
    BsDataType,
    AccDataType,
    CShuffleDataType,
    DsDataType,
    EDataType,
    AElementOp,
    BElementOp,
    CDEElementOp,
    GemmSpec,
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
    S<8, 32, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    0,
    S<8, 32, 1>,
    S<0, 2, 1>,
    S<0, 2, 1>,
    1,
    1,
    8,
    0,
    1,
    1,
    S<1, 32, 1, 8>,
    S<8, 8, 8>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v3>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 2;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 4096;
    ck::index_t N = 768;
    ck::index_t K = 6144;

    ck::index_t StrideA = K;
    ck::index_t StrideB = N;
    ck::index_t StrideD = N;
    ck::index_t StrideE = N;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 11)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideD = std::stoi(argv[9]);
        StrideE = std::stoi(argv[10]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 10: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD, StrideE\n");
        exit(0);
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz}, Bypass{});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride}, Bypass{});
            }
        };

    Tensor<A0DataType> a0_m_k(f_host_tensor_descriptor(M, K, StrideA, A0Layout{}));
    Tensor<B0DataType> b0_k_n(f_host_tensor_descriptor(K, N, StrideB, B0Layout{}));
    Tensor<B1DataType> b1_k_n(f_host_tensor_descriptor(K, N, StrideB, B1Layout{}));
    Tensor<D0DataType> d_m_n(f_host_tensor_descriptor(M, N, StrideD, D0Layout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    std::cout << "a0_m_k: " << a0_m_k.mDesc << std::endl;
    std::cout << "b0_k_n: " << b0_k_n.mDesc << std::endl;
    std::cout << "b1_k_n: " << b1_k_n.mDesc << std::endl;
    std::cout << "d_m_n: " << d_m_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-5, 5});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-5, 5});
        b1_k_n.GenerateTensorValue(GeneratorTensor_2<B1DataType>{0, 5});
        d_m_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{-5, 5});
        break;
    default:
        a0_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-5, 5});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 5});
        d_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{-0.5, 0.5});
    }

    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(D0DataType) * d_m_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a0_device_buf.ToDevice(a0_m_k.mData.data());
    b0_device_buf.ToDevice(b0_k_n.mData.data());
    b1_device_buf.ToDevice(b1_k_n.mData.data());
    d_device_buf.ToDevice(d_m_n.mData.data());
    e_device_buf.ToDevice(e_m_n_device_result.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumATensor = 1;
    constexpr ck::index_t NumBTensor = 2;
    constexpr ck::index_t NumDTensor = 1;

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(std::array<const void*, NumATensor>{a0_device_buf.GetDeviceBuffer()},
                               std::array<const void*, NumBTensor>{b0_device_buf.GetDeviceBuffer(),
                                                                   b1_device_buf.GetDeviceBuffer()},
                               std::array<const void*, NumDTensor>{d_device_buf.GetDeviceBuffer()},
                               e_device_buf.GetDeviceBuffer(),
                               M,
                               N,
                               K,
                               std::array<ck::index_t, NumATensor>{StrideA},
                               std::array<ck::index_t, NumBTensor>{StrideB, StrideB},
                               std::array<ck::index_t, NumDTensor>{StrideD},
                               StrideE,
                               1,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N + sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    if(do_verification)
    {
        Tensor<CShuffleDataType> c_m_n({M, N});

        Tensor<B1DataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, B0Layout{}));

        for(int n = 0; n < N; ++n)
        {
            for(int k = 0; k < K; ++k)
            {
                b_element_op(b_k_n(k, n), b0_k_n(k, n), b1_k_n(k, n));
            }
        }

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<A0DataType,
                                                                                B1DataType,
                                                                                CShuffleDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;
        auto ref_gemm               = ReferenceGemmInstance{};
        auto ref_invoker            = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a0_m_k, b_k_n, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                cde_element_op(e_m_n_host_result(m, n), c_m_n(m, n), d_m_n(m, n));
            }
        }

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result) ? 0 : 1;
    }

    return 0;
}
