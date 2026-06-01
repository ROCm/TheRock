// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
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

using F16 = ck::half_t;
using FP8 = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = F16;
using B0DataType       = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = F32;
using D1DataType       = F32;
using DsDataType       = ck::Tuple<D0DataType, D1DataType>;
using EDataType        = F16;

using A0Layout = Row;
using B0Layout = Col;
using D0Layout = Row;
using D1Layout = Row;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;
using ELayout  = Row;

struct AddAdd
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, float, float, float>(
        ck::half_t& e, const float& c, const float& d0, const float& d1) const
    {
        const float x0_f = c + d0 + d1;

        e = ck::type_convert<ck::half_t>(x0_f);
    }
};

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = AddAdd;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3
    // clang-format off
///######|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     DsData|     EData|     AccData|         CShuffle|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
///######|         |         |         |        |       Type|       Type|       Type|      Type|        Type|         DataType| Elementwise| Elementwise|  Elementwise| Spacialization|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
///######|         |         |         |        |           |           |           |          |            |                 |   Operation|   Operation|    Operation|               |      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
///######|         |         |         |        |           |           |           |          |            |                 |            |            |             |               |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |    S<C, D0, D1>|
///###### RCR
         <      Row,      Col, DsLayout, ELayout, A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   256,   128,   128,  16,  16,  16,   16,    8,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,               8,          0,     S<4, 64, 1>,    S<1, 0, 2>,     S<1, 0, 2>,             2,               8,             8,          0,          1,           1,               S<1, 32, 1, 8>,      S<4, 4, 4>,  ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, FP8>;
// clang-format on

int main(int argc, char* argv[])
{
    // fp8 are not supported on gfx11
    if(ck::is_gfx11_supported())
    {
        return 0;
    }

    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideD = K;
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
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideD, StrideE\n");
        exit(0);
    }

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
    Tensor<B0DataType> b0_k_n(f_host_tensor_descriptor(K, N, StrideB, B0Layout{}));
    Tensor<D0DataType> d0_m_n(f_host_tensor_descriptor(M, N, StrideD, D0Layout{}));
    Tensor<D1DataType> d1_m_n(f_host_tensor_descriptor(M, N, StrideD, D1Layout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    std::cout << "a0_m_k: " << a0_m_k.mDesc << std::endl;
    std::cout << "b0_k_n: " << b0_k_n.mDesc << std::endl;
    std::cout << "d1_m_n: " << d1_m_n.mDesc << std::endl;
    std::cout << "d0_m_n: " << d0_m_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{0, 2});
        d0_m_n.GenerateTensorValue(GeneratorTensor_2<D0DataType>{0, 2});
        d1_m_n.GenerateTensorValue(GeneratorTensor_2<D1DataType>{0, 2});
        break;
    default:
        a0_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        b0_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        d0_m_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{-0.5, 0.5});
        d1_m_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{-0.5, 0.5});
    }

    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_m_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_m_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    a0_device_buf.ToDevice(a0_m_k.mData.data());
    b0_device_buf.ToDevice(b0_k_n.mData.data());
    d0_device_buf.ToDevice(d0_m_n.mData.data());
    d1_device_buf.ToDevice(d1_m_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(a0_device_buf.GetDeviceBuffer(),
                               b0_device_buf.GetDeviceBuffer(),
                               std::array<const void*, NumDTensor>{d0_device_buf.GetDeviceBuffer(),
                                                                   d1_device_buf.GetDeviceBuffer()},
                               e_device_buf.GetDeviceBuffer(),
                               M,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               std::array<ck::index_t, NumDTensor>{StrideD, StrideD},
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

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel, 0, 20, 50});

    std::size_t flop      = std::size_t(2) * M * N * K;
    std::size_t num_btype = sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N +
                            sizeof(D0DataType) * M * N + sizeof(D1DataType) * M * N +
                            sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        Tensor<CShuffleDataType> c_m_n({M, N});

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<A0DataType,
                                                                                B0DataType,
                                                                                CShuffleDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;
        auto ref_gemm               = ReferenceGemmInstance{};
        auto ref_invoker            = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a0_m_k, b0_k_n, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                cde_element_op(e_m_n_host_result(m, n), c_m_n(m, n), d0_m_n(m, n), d1_m_n(m, n));
            }
        }

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        return ck::utils::check_err(e_m_n_device_result, e_m_n_host_result) ? 0 : 1;
    }

    return 0;
}
