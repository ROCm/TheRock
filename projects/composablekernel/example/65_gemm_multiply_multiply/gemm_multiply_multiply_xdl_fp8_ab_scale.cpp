// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

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
using FP8  = ck::f8_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = FP8;
using A1DataType       = F32;
using B0DataType       = FP8;
using B1DataType       = F32;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DsDataType       = ck::Tuple<>;
using EDataType        = BF16;

using A0Layout = Row;
using B0Layout = Col;
using D0Layout = Row;
using D1Layout = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr ck::index_t Scale_Block_M = 1;
static constexpr ck::index_t Scale_Block_N = 128;
static constexpr ck::index_t Scale_Block_K = 128;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3
    // clang-format off
         <Row, Col, DsLayout, ELayout,
          A0DataType, A1DataType, B0DataType, B1DataType, DsDataType, EDataType, AccDataType, CShuffleDataType, 
          AElementOp,  BElementOp, CDEElementOp, GemmSpec,
          256, Scale_Block_M, Scale_Block_N, Scale_Block_K,
          128, 128,
          128, 16, 16,
          16,   16,
          4,    4,
          S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
          S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
          1,    2,  S<1, 32, 1, 8>,  S<8>,
          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, FP8>;
// clang-format on

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
    bool flush_cache     = true;

    // GEMM shape
    ck::index_t M = 128;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideE = N;

    ck::index_t KBatch = 1;

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
    else if(argc == 8 || argc == 9)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        flush_cache = std::stoi(argv[7]);

        if(argc == 9)
        {
            KBatch = std::stoi(argv[8]);
        }

        StrideA = K;
        StrideB = K;
        StrideE = N;
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 6: M, N, K\n");
        printf("arg7: flush both I$ and L2$ (0=no, 1=yes)\n");
        printf("arg8: KBatch (default: 1)\n");
        exit(0);
    }

    ck::index_t Scale_Stride_AM = (K + Scale_Block_K - 1) / Scale_Block_K;
    ck::index_t Scale_Stride_BN = (K + Scale_Block_K - 1) / Scale_Block_K;

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

    Tensor<A0DataType> a0_m_k(f_host_tensor_descriptor(M, K, StrideA, A0Layout{}));
    Tensor<A1DataType> a1_m_k(f_host_tensor_descriptor((M + Scale_Block_M - 1) / Scale_Block_M,
                                                       (K + Scale_Block_K - 1) / Scale_Block_K,
                                                       Scale_Stride_AM,
                                                       A0Layout{}));
    Tensor<B0DataType> b0_k_n(f_host_tensor_descriptor(K, N, StrideB, B0Layout{}));
    Tensor<B1DataType> b1_k_n(f_host_tensor_descriptor((K + Scale_Block_K - 1) / Scale_Block_K,
                                                       (N + Scale_Block_N - 1) / Scale_Block_N,
                                                       Scale_Stride_BN,
                                                       B0Layout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    std::cout << "a0_m_k: " << a0_m_k.mDesc << std::endl;
    std::cout << "a1_m_k: " << a1_m_k.mDesc << std::endl;
    std::cout << "b0_k_n: " << b0_k_n.mDesc << std::endl;
    std::cout << "b1_k_n: " << b1_k_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;

#if 1
    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        a1_m_k.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
        break;
    case 2:
        a0_m_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b0_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        a1_m_k.GenerateTensorValue(GeneratorTensor_1<A1DataType>{});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 3:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        a1_m_k.GenerateTensorValue(GeneratorTensor_1<A1DataType>{});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 4:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        a1_m_k.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 5:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        a1_m_k.GenerateTensorValue(GeneratorTensor_1<A1DataType>{});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
        break;
    default:
        a0_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{-0.5, 0.5});
        b0_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        a1_m_k.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0, 1.0});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
    }
#endif
#if 0
    for(int im =0; im< (M + Scale_Block_M - 1) / Scale_Block_M; im++){
        float row_sum = .0;
        for(int ik =0; ik< (K + Scale_Block_K - 1) / Scale_Block_K; ik++){
            printf("%lf ",a1_m_k(im, ik));
            row_sum += a1_m_k(im, ik);
        }
        printf("sum: %lf\n", row_sum * 128);
    }
#endif

    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_m_k.mDesc.GetElementSpaceSize());
    DeviceMem a1_device_buf(sizeof(A1DataType) * a1_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_k_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a0_device_buf.ToDevice(a0_m_k.mData.data());
    a1_device_buf.ToDevice(a1_m_k.mData.data());
    b0_device_buf.ToDevice(b0_k_n.mData.data());
    b1_device_buf.ToDevice(b1_k_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    // do GEMM
    auto device_op  = DeviceOpInstance{};
    auto invoker    = device_op.MakeInvoker();
    auto argument   = device_op.MakeArgument(a0_device_buf.GetDeviceBuffer(),
                                           b0_device_buf.GetDeviceBuffer(),
                                           std::array<const void*, NumDTensor>{},
                                           e_device_buf.GetDeviceBuffer(),
                                           M,
                                           N,
                                           K,
                                           StrideA,
                                           StrideB,
                                           std::array<ck::index_t, NumDTensor>{},
                                           StrideE,
                                           a1_device_buf.GetDeviceBuffer(),
                                           b1_device_buf.GetDeviceBuffer(),
                                           a_element_op,
                                           b_element_op,
                                           cde_element_op);
    argument.KBatch = KBatch;

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N + sizeof(EDataType) * M * N;

    float ave_time = .0;

    if(flush_cache)
    {
        int rotating_buf = (512 * 1024 * 1024 + num_btype - 1) / num_btype;

        ave_time = invoker.Run(argument,
                               StreamConfig{nullptr, time_kernel, 0, 50, 100, true, rotating_buf});
    }
    else
    {
        ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel, 0, 50, 100});
    }

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        Tensor<AccDataType> c_m_n({M, N});
        Tensor<float> a_m_k({M, K});
        Tensor<float> b_k_n({K, N});

        for(int m = 0; m < M; m++)
        {
            for(int k = 0; k < K; k++)
            {
                a_m_k(m, k) = ck::type_convert<float>(a0_m_k(m, k)) *
                              a1_m_k(m / Scale_Block_M, k / Scale_Block_K);
            }
        }

        for(int n = 0; n < N; n++)
        {
            for(int k = 0; k < K; k++)
            {
                b_k_n(k, n) = ck::type_convert<float>(b0_k_n(k, n)) *
                              b1_k_n(k / Scale_Block_K, n / Scale_Block_N);
            }
        }

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<float,
                                                                                float,
                                                                                CShuffleDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;
        auto ref_gemm               = ReferenceGemmInstance{};
        auto ref_invoker            = ref_gemm.MakeInvoker();

        auto ref_argument =
            ref_gemm.MakeArgument(a_m_k, b_k_n, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

#if 1
        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                e_m_n_host_result(m, n) = ck::type_convert<EDataType>(c_m_n(m, n));
            }
        }
#endif

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        return ck::utils::check_err(
                   e_m_n_device_result, e_m_n_host_result, "Error: Incorrect results!", 5e-2, 5e-2)
                   ? 0
                   : 1;
    }

    return 0;
}
