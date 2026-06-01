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

using F16 = ck::half_t;
using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using A0DataType       = F16;
using A1DataType       = F32;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DDataType        = F16;
using EDataType        = F16;
using ComputeDataType  = F16;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

struct AlphaBetaAdd
{
    AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename C, typename D>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, float, ck::half_t>(
        ck::half_t& e, const float& c, const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * c + beta_ * ck::type_convert<float>(d));
    };

    float alpha_;
    float beta_;
};

struct Multiply
{
    __host__ __device__ constexpr void
    operator()(ck::half_t& a, const ck::half_t& a0, const float& a1) const
    {
        a = ck::type_convert<ck::half_t>(ck::type_convert<float>(a0) * a1);
    }
};

using AElementOp   = Multiply;
using BElementOp   = PassThrough;
using CDEElementOp = AlphaBetaAdd;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceOpInstance = ck::tensor_operation::device::DeviceContractionMultipleABD_Xdl_CShuffle<
    NumDimM,
    NumDimN,
    NumDimK,
    ck::Tuple<A0DataType, A1DataType>,
    ck::Tuple<BDataType>,
    AccDataType,
    CShuffleDataType,
    ck::Tuple<DDataType>,
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
    16,
    16,
    8,
    4,
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
    4>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    float alpha = 1.0f;
    float beta  = 1.0f;

    // A0[M0, M1, K0, K1]
    std::vector<ck::index_t> a0_ms_ks_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> a0_ms_ks_strides{128 * 32 * 64, 32 * 64, 64, 1};
    // A1[M1, K1] -> A1[M0, M1, K0, K1]
    std::vector<ck::index_t> a1_ms_ks_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> a1_ms_ks_strides{0, 64, 1, 0};
    // B[N0, N1, K0, K1]
    std::vector<ck::index_t> b_ns_ks_lengths{32, 64, 32, 64};
    std::vector<ck::index_t> b_ns_ks_strides{64 * 32 * 64, 32 * 64, 64, 1};
    // D[M0, M1, N0, N1]
    std::vector<ck::index_t> d_ms_ns_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> d_ms_ns_strides{128 * 32 * 64, 32 * 64, 64, 1};
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
    Tensor<BDataType> b_ns_ks(b_ns_ks_lengths, b_ns_ks_strides, Row{});
    Tensor<EDataType> d_ms_ns(d_ms_ns_lengths, d_ms_ns_strides, Row{});
    Tensor<EDataType> e_ms_ns_host_result(e_ms_ns_lengths, e_ms_ns_strides, Row{});
    Tensor<EDataType> e_ms_ns_device_result(e_ms_ns_lengths, e_ms_ns_strides, Row{});

    std::cout << "a0_ms_ks: " << a0_ms_ks.mDesc << std::endl;
    std::cout << "a1_ms_ks: " << a1_ms_ks.mDesc << std::endl;
    std::cout << "b_ns_ks: " << b_ns_ks.mDesc << std::endl;
    std::cout << "d_ms_ns: " << d_ms_ns.mDesc << std::endl;
    std::cout << "e_ms_ns: " << e_ms_ns_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_ms_ks.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-5, 5});
        a1_ms_ks.GenerateTensorValue(GeneratorTensor_2<A1DataType>{-5, 5});
        b_ns_ks.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        d_ms_ns.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        a0_ms_ks.GenerateTensorValue(GeneratorTensor_3<A0DataType>{0.0, 1.0});
        a1_ms_ks.GenerateTensorValue(GeneratorTensor_3<A1DataType>{0.0, 1.0});
        b_ns_ks.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d_ms_ns.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem a1_device_buf(sizeof(A1DataType) * a1_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(DDataType) * d_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_ms_ns_device_result.mDesc.GetElementSpaceSize());

    a0_device_buf.ToDevice(a0_ms_ks.mData.data());
    a1_device_buf.ToDevice(a1_ms_ks.mData.data());
    b_device_buf.ToDevice(b_ns_ks.mData.data());
    d_device_buf.ToDevice(d_ms_ns.mData.data());

    // set zero
    e_device_buf.SetZero();

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{alpha, beta};

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(
        std::array<const void*, 2>{a0_device_buf.GetDeviceBuffer(),
                                    a1_device_buf.GetDeviceBuffer()},
        std::array<const void*, 1>{b_device_buf.GetDeviceBuffer()},
        std::array<const void*, 1>{d_device_buf.GetDeviceBuffer()},
        e_device_buf.GetDeviceBuffer(),
        std::array<std::vector<ck::index_t>, 2>{a0_ms_ks_lengths, a1_ms_ks_lengths},
        std::array<std::vector<ck::index_t>, 2>{a0_ms_ks_strides, a1_ms_ks_strides},
        std::array<std::vector<ck::index_t>, 1>{b_ns_ks_lengths},
        std::array<std::vector<ck::index_t>, 1>{b_ns_ks_strides},
        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
        std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},
        e_ms_ns_lengths,
        e_ms_ns_strides,
        a_element_op,
        b_element_op,
        cde_element_op);

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
            sizeof(A0DataType) * M * K + sizeof(BDataType) * K * N + +sizeof(EDataType) * M * N;

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

        using ReferenceOpInstance =
            ck::tensor_operation::host::ReferenceContraction_M2_N2_K2<NumDimM,
                                                                      NumDimN,
                                                                      NumDimK,
                                                                      A0DataType,
                                                                      BDataType,
                                                                      CShuffleDataType,
                                                                      AccDataType,
                                                                      ComputeDataType,
                                                                      PassThrough,
                                                                      BElementOp>;

        auto ref_op      = ReferenceOpInstance{};
        auto ref_invoker = ref_op.MakeInvoker();

        auto ref_argument =
            ref_op.MakeArgument(a_ms_ks, b_ns_ks, c_ms_ns_host_result, PassThrough{}, b_element_op);

        ref_invoker.Run(ref_argument);

        for(size_t m0 = 0; m0 < e_ms_ns_host_result.mDesc.GetLengths()[0]; ++m0)
        {
            for(size_t m1 = 0; m1 < e_ms_ns_host_result.mDesc.GetLengths()[1]; ++m1)
            {
                for(size_t n0 = 0; n0 < e_ms_ns_host_result.mDesc.GetLengths()[2]; ++n0)
                {
                    for(size_t n1 = 0; n1 < e_ms_ns_host_result.mDesc.GetLengths()[3]; ++n1)
                    {
                        cde_element_op(e_ms_ns_host_result(m0, m1, n0, n1),
                                       c_ms_ns_host_result(m0, m1, n0, n1),
                                       d_ms_ns(m0, m1, n0, n1));
                    }
                }
            }
        }

        e_device_buf.FromDevice(e_ms_ns_device_result.mData.data());

        return ck::utils::check_err(e_ms_ns_device_result, e_ms_ns_host_result) ? 0 : 1;
    }

    return 0;
}
