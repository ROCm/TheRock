// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_contraction_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_contraction.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/numeric.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using Row    = ck::tensor_layout::gemm::RowMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F8  = ck::f8_t;
using F16 = ck::half_t;
using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using A0DataType       = F8;
using A1DataType       = F32;
using B0DataType       = F8;
using B1DataType       = F32;
using AccDataType      = F32;
using CShuffleDataType = F32;
using EDataType        = F16;
using ComputeDataType  = F8;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

struct Multiply
{
    __host__ __device__ constexpr void
    operator()(ck::f8_t& a, const ck::f8_t& a0, const float& a1) const
    {
        a = ck::type_convert<ck::half_t>(ck::type_convert<float>(a0) * a1);
    }
};

using AElementOp   = Multiply;
using BElementOp   = Multiply;
using CDEElementOp = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceOpInstance = ck::tensor_operation::device::DeviceContractionMultipleABD_Xdl_CShuffle<
    NumDimM,
    NumDimN,
    NumDimK,
    ck::Tuple<A0DataType, A1DataType>,
    ck::Tuple<B0DataType, B1DataType>,
    AccDataType,
    CShuffleDataType,
    ck::Tuple<>,
    EDataType,
    AElementOp,
    BElementOp,
    CDEElementOp,
    GemmSpec,
    1,
    256,
    256,
    128,
    32,
    8,
    8,
    32,
    32,
    4,
    2,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    1,
    8,
    1,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    1,
    1,
    1,
    S<1, 32, 1, 8>,
    8>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // A0[M0, M1, K0, K1]
    std::vector<ck::index_t> a0_ms_ks_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> a0_ms_ks_strides{128 * 32 * 64, 32 * 64, 64, 1};
    // A1[M1, K1] -> A1[M0, M1, K0, K1]
    std::vector<ck::index_t> a1_ms_ks_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> a1_ms_ks_strides{0, 64, 1, 0};
    // B0[N0, N1, K0, K1]
    std::vector<ck::index_t> b0_ns_ks_lengths{32, 64, 32, 64};
    std::vector<ck::index_t> b0_ns_ks_strides{64 * 32 * 64, 32 * 64, 64, 1};
    // B1[N0, N1, K0, K1]
    std::vector<ck::index_t> b1_ns_ks_lengths{32, 64, 32, 64};
    std::vector<ck::index_t> b1_ns_ks_strides{64 * 32 * 64, 32 * 64, 64, 1};
    // E[M0, M1, N0, N1]
    std::vector<ck::index_t> e_ms_ns_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> e_ms_ns_strides{128 * 32 * 64, 32 * 64, 64, 1};

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
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        exit(0);
    }

    Tensor<A0DataType> a0_ms_ks(a0_ms_ks_lengths, a0_ms_ks_strides, Row{});
    Tensor<A1DataType> a1_ms_ks(a1_ms_ks_lengths, a1_ms_ks_strides, Bypass{});
    Tensor<B0DataType> b0_ns_ks(b0_ns_ks_lengths, b0_ns_ks_strides, Row{});
    Tensor<B1DataType> b1_ns_ks(b1_ns_ks_lengths, b1_ns_ks_strides, Row{});
    Tensor<EDataType> e_ms_ns_host_result(e_ms_ns_lengths, e_ms_ns_strides, Row{});
    Tensor<EDataType> e_ms_ns_device_result(e_ms_ns_lengths, e_ms_ns_strides, Row{});

    std::cout << "a0_ms_ks: " << a0_ms_ks.mDesc << std::endl;
    std::cout << "a1_ms_ks: " << a1_ms_ks.mDesc << std::endl;

    std::cout << "b0_ns_ks: " << b0_ns_ks.mDesc << std::endl;
    std::cout << "b1_ns_ks: " << b1_ns_ks.mDesc << std::endl;

    std::cout << "e_ms_ns: " << e_ms_ns_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_ms_ks.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-5, 5});
        a1_ms_ks.GenerateTensorValue(GeneratorTensor_2<A1DataType>{-5, 5});
        b0_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-5, 5});
        b1_ns_ks.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-5, 5});
        break;
    default:
        a0_ms_ks.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        a1_ms_ks.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0.0, 1.0});
        b0_ns_ks.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        b1_ns_ks.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem a1_device_buf(sizeof(A1DataType) * a1_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_ms_ns_device_result.mDesc.GetElementSpaceSize());

    a0_device_buf.ToDevice(a0_ms_ks.mData.data());
    a1_device_buf.ToDevice(a1_ms_ks.mData.data());
    b0_device_buf.ToDevice(b0_ns_ks.mData.data());
    b1_device_buf.ToDevice(b1_ns_ks.mData.data());

    // set zero
    e_device_buf.SetZero();

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(
        std::array<const void*, 2>{a0_device_buf.GetDeviceBuffer(),
                                    a1_device_buf.GetDeviceBuffer()},
        std::array<const void*, 2>{b0_device_buf.GetDeviceBuffer(),
                                    b1_device_buf.GetDeviceBuffer()},
        std::array<const void*, 0>{},
        e_device_buf.GetDeviceBuffer(),
        std::array<std::vector<ck::index_t>, 2>{a0_ms_ks_lengths, a1_ms_ks_lengths},
        std::array<std::vector<ck::index_t>, 2>{a0_ms_ks_strides, a1_ms_ks_strides},
        std::array<std::vector<ck::index_t>, 2>{b0_ns_ks_lengths, b1_ns_ks_lengths},
        std::array<std::vector<ck::index_t>, 2>{b0_ns_ks_strides, b1_ns_ks_strides},
        std::array<std::vector<ck::index_t>, 0>{},
        std::array<std::vector<ck::index_t>, 0>{},
        e_ms_ns_lengths,
        e_ms_ns_strides,
        a_element_op,
        b_element_op,
        PassThrough{});

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_contraction with the specified compilation parameters does "
            "not support this problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    if(time_kernel)
    {
        ck::index_t M =
            ck::accumulate_n<ck::index_t>(e_ms_ns_lengths.begin(), NumDimM, 1, std::multiplies<>{});

        ck::index_t N = ck::accumulate_n<ck::index_t>(
            e_ms_ns_lengths.begin() + NumDimM, NumDimN, 1, std::multiplies<>{});

        ck::index_t K = ck::accumulate_n<ck::index_t>(
            a0_ms_ks_lengths.begin() + NumDimM, NumDimK, 1, std::multiplies<>{});

        std::size_t flop = std::size_t(2) * M * N * K;
        std::size_t num_btype =
            sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N + +sizeof(EDataType) * M * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s" << std::endl;
    }

    if(do_verification)
    {

        Tensor<CShuffleDataType> c_ms_ns_host_result(e_ms_ns_lengths, e_ms_ns_strides, Row{});

        Tensor<A0DataType> a_ms_ks(a0_ms_ks_lengths, a0_ms_ks_strides, Row{});

        for(size_t m0 = 0; m0 < a_ms_ks.mDesc.GetLengths()[0]; ++m0)
        {
            for(size_t m1 = 0; m1 < a_ms_ks.mDesc.GetLengths()[1]; ++m1)
            {
                for(size_t k0 = 0; k0 < a_ms_ks.mDesc.GetLengths()[2]; ++k0)
                {
                    for(size_t k1 = 0; k1 < a_ms_ks.mDesc.GetLengths()[3]; ++k1)
                    {
                        a_element_op(a_ms_ks(m0, m1, k0, k1),
                                     a0_ms_ks(m0, m1, k0, k1),
                                     a1_ms_ks(m0, m1, k0, k1));
                    }
                }
            }
        }

        Tensor<B0DataType> b_ns_ks(b0_ns_ks_lengths, b0_ns_ks_strides, Row{});

        for(size_t n0 = 0; n0 < b_ns_ks.mDesc.GetLengths()[0]; ++n0)
        {
            for(size_t n1 = 0; n1 < b_ns_ks.mDesc.GetLengths()[1]; ++n1)
            {
                for(size_t k0 = 0; k0 < b_ns_ks.mDesc.GetLengths()[2]; ++k0)
                {
                    for(size_t k1 = 0; k1 < b_ns_ks.mDesc.GetLengths()[3]; ++k1)
                    {
                        b_element_op(b_ns_ks(n0, n1, k0, k1),
                                     b0_ns_ks(n0, n1, k0, k1),
                                     b1_ns_ks(n0, n1, k0, k1));
                    }
                }
            }
        }

        using ReferenceOpInstance =
            ck::tensor_operation::host::ReferenceContraction_M2_N2_K2<NumDimM,
                                                                      NumDimN,
                                                                      NumDimK,
                                                                      A0DataType,
                                                                      B0DataType,
                                                                      CShuffleDataType,
                                                                      AccDataType,
                                                                      ComputeDataType,
                                                                      PassThrough,
                                                                      PassThrough>;

        auto ref_op      = ReferenceOpInstance{};
        auto ref_invoker = ref_op.MakeInvoker();

        Tensor<float> empty_tensor(std::vector<ck::index_t>{}, std::vector<ck::index_t>{});
        auto ref_argument = ref_op.MakeArgument(
            a_ms_ks, b_ns_ks, c_ms_ns_host_result, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

        e_device_buf.FromDevice(e_ms_ns_device_result.mData.data());

        return ck::utils::check_err(e_ms_ns_device_result, e_ms_ns_host_result) ? 0 : 1;
    }

    return 0;
}
