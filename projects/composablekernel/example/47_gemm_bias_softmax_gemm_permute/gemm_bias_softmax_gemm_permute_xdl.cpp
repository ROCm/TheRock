// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using Row    = ck::tensor_layout::gemm::RowMajor;
using Col    = ck::tensor_layout::gemm::ColumnMajor;
using Bypass = ck::tensor_layout::BypassLayoutVerification;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp    = ck::tensor_operation::element_wise::PassThrough;
using B0ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using C0DEElementOp = ck::tensor_operation::element_wise::ScaleAdd;
using Acc0ElementOp = ck::tensor_operation::element_wise::PassThrough;
using B1ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CElementOp    = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
constexpr static auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;
static constexpr auto TensorSpecA  = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecB0 = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecB1 = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecC  = ck::tensor_operation::device::TensorSpecialization::Default;

using F16              = ck::half_t;
using F32              = float;
using ADataType        = F16;
using B0DataType       = F16;
using B1DataType       = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using CDataType        = F16;
using D0DataType       = F16;
using Acc0BiasDataType = ck::Tuple<D0DataType>;
using Acc1BiasDataType = ck::Tuple<>;

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
static constexpr ck::index_t NumDimN = 1;
static constexpr ck::index_t NumDimK = 1;
static constexpr ck::index_t NumDimO = 1;

using DeviceOpInstance =
    ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        NumDimO,
        ADataType,
        B0DataType,
        B1DataType,
        CDataType,
        Acc0BiasDataType,
        Acc1BiasDataType,
        AccDataType,
        CShuffleDataType,
        AElementOp,
        B0ElementOp,
        C0DEElementOp,
        B1ElementOp,
        CElementOp,
        GemmSpec,
        TensorSpecA,
        TensorSpecB0,
        TensorSpecB1,
        TensorSpecC,
        1,
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        32,          // KPerBlock
        64,          // Gemm1NPerBlock
        32,          // Gemm1KPerBlock
        8,           // AK1
        8,           // BK1
        2,           // B1K1
        16,          // MPerXDL
        16,          // NPerXDL
        2,           // MXdlPerWave
        8,           // NXdlPerWave
        4,           // Gemm1NXdlPerWave
        S<4, 64, 1>, // ABlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<4, 64, 1>, // BBlockTransfer
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        true,
        S<16, 16, 1>, // B1BlockTransfer
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        2,              // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        4,              // CShuffleBlockTransferScalarPerVector_NPerBlock
        MaskingSpec,    // MaskingSpecialization
        1>;

// Ref Gemm0: fp16 in, fp32 out
using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B0DataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B0ElementOp,
                                                                                Acc0ElementOp>;

// Ref Softmax: fp32 in, fp16 out
using ReferenceSoftmaxInstance =
    ck::tensor_operation::host::ReferenceSoftmax<AccDataType, ADataType, AccDataType>;

// Ref Gemm1: fp16 in, fp16 out
using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B1DataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B1ElementOp,
                                                                                CElementOp>;
int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    int G0      = 3;
    int G1      = 2;
    int M       = 1024;
    int N       = 1024;
    int K       = 64;
    int O       = 64;
    float alpha = 1;

    // temp disable on gfx11, d0_gs_ms_ns isn't handled correctly when it is not a constant.
    if(ck::is_gfx11_supported())
    {
        return 0;
    }

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

        M  = std::stoi(argv[4]);
        N  = std::stoi(argv[5]);
        K  = std::stoi(argv[6]);
        O  = std::stoi(argv[7]);
        G0 = std::stoi(argv[8]);
        G1 = std::stoi(argv[9]);

        alpha = std::stof(argv[10]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 11: M, N, K, O, G0, G1\n");
        printf("arg10: scale (alpha)\n");
        exit(0);
    }

    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> a_gs_ms_ks_strides{
        M * G1 * K, K, G1 * K, 1}; // A layout [G0, M, G1, K]

    std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> b0_gs_ns_ks_strides{
        N * G1 * K, K, G1 * K, 1}; // B0 layout [G0, N, G1, K]

    std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> b1_gs_os_ns_strides{
        N * G1 * O, O, 1, G1 * O}; // B1 layout [G0, N, G1, O]

    std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> c_gs_ms_os_strides{
        M * G1 * O, O, G1 * O, 1}; // C layout [G0, M, G1, O]

    // D layout [G0, M, G1, N]
    std::vector<ck::index_t> d0_gs_ms_ns_lengths{G0, G1, M, N};
    std::vector<ck::index_t> d0_gs_ms_ns_strides{M * G1 * N, N, G1 * N, 1};

    Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides, Row{});
    Tensor<B0DataType> b0_gs_ns_ks(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides, Row{});
    Tensor<B1DataType> b1_gs_os_ns(b1_gs_os_ns_lengths, b1_gs_os_ns_strides, Col{});
    Tensor<D0DataType> d0_gs_ms_ns(d0_gs_ms_ns_lengths, d0_gs_ms_ns_strides, Row{});
    Tensor<CDataType> c_gs_ms_os_host_result(c_gs_ms_os_lengths, c_gs_ms_os_strides, Row{});
    Tensor<CDataType> c_gs_ms_os_device_result(c_gs_ms_os_lengths, c_gs_ms_os_strides, Row{});

    std::cout << "a_gs_ms_ks: " << a_gs_ms_ks.mDesc << std::endl;
    std::cout << "b0_gs_ns_ks: " << b0_gs_ns_ks.mDesc << std::endl;
    std::cout << "b1_gs_os_ns: " << b1_gs_os_ns.mDesc << std::endl;
    std::cout << "c_gs_ms_os: " << c_gs_ms_os_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
        d0_gs_ms_ns.GenerateTensorValue(GeneratorTensor_2<D0DataType>{-2, 2});
        break;
    case 2:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        d0_gs_ms_ns.GenerateTensorValue(GeneratorTensor_2<D0DataType>{-1, 1});
        break;
    case 3:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        d0_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<D0DataType>{1});
        break;
    default:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_Sequential<ADataType, 2>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        d0_gs_ms_ns.GenerateTensorValue(GeneratorTensor_1<D0DataType>{1});
    }

    DeviceMem a_device_buf(sizeof(ADataType) * G0 * G1 * M * K);
    DeviceMem b0_device_buf(sizeof(B0DataType) * G0 * G1 * N * K);
    DeviceMem d0_device_buf(sizeof(D0DataType) * G0 * G1 * M * N);
    DeviceMem b1_device_buf(sizeof(B1DataType) * G0 * G1 * O * N);
    DeviceMem c_device_buf(sizeof(CDataType) * G0 * G1 * M * O);

    a_device_buf.ToDevice(a_gs_ms_ks.mData.data());
    b0_device_buf.ToDevice(b0_gs_ns_ks.mData.data());
    b1_device_buf.ToDevice(b1_gs_os_ns.mData.data());
    d0_device_buf.ToDevice(d0_gs_ms_ns.mData.data());

    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto c0de_element_op = C0DEElementOp{alpha};
    auto acc0_element_op = Acc0ElementOp{};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    auto argument = device_op.MakeArgument(
        static_cast<const ADataType*>(a_device_buf.GetDeviceBuffer()),
        static_cast<const B0DataType*>(b0_device_buf.GetDeviceBuffer()),
        static_cast<const B1DataType*>(b1_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
        std::array<void*, 1>{d0_device_buf.GetDeviceBuffer()}, // p_acc0_biases
        {},                                                    // p_acc1_biases
        a_gs_ms_ks_lengths,
        a_gs_ms_ks_strides,
        b0_gs_ns_ks_lengths,
        b0_gs_ns_ks_strides,
        b1_gs_os_ns_lengths,
        b1_gs_os_ns_strides,
        c_gs_ms_os_lengths,
        c_gs_ms_os_strides,
        std::array<std::vector<ck::index_t>, 1>{
            d0_gs_ms_ns_lengths}, // acc0_biases_gs_ms_ns_lengths
        std::array<std::vector<ck::index_t>, 1>{
            d0_gs_ms_ns_strides}, // acc0_biases_gs_ms_ns_strides
        {},                       // acc1_biases_gs_ms_os_lengths
        {},                       // acc1_biases_gs_ms_os_strides
        a_element_op,
        b0_element_op,
        c0de_element_op,
        b1_element_op,
        c_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error("wrong! this device_op instance does not support this problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    ck::index_t BatchCount = G0 * G1;
    std::size_t flop       = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * BatchCount;
    std::size_t num_btype =
        (sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N + sizeof(B1DataType) * N * O +
         sizeof(CDataType) * M * O + sizeof(D0DataType) * M * N) *
        BatchCount;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << std::endl;

    if(do_verification)
    {
        c_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());

        Tensor<ADataType> a_g_m_k({BatchCount, M, K});
        Tensor<B0DataType> b0_g_k_n({BatchCount, K, N});
        Tensor<B1DataType> b1_g_n_o({BatchCount, N, O});
        Tensor<AccDataType> acc0_g_m_n({BatchCount, M, N});        // scratch object after gemm0
        Tensor<ADataType> a1_g_m_n({BatchCount, M, N});            // scratch object after softmax
        Tensor<CDataType> c_g_m_o_host_result({BatchCount, M, O}); // scratch object after gemm1
        Tensor<D0DataType> d0_g_m_n({BatchCount, M, N});

        // permute
        a_gs_ms_ks.ForEach([&](auto& self, auto idx) {
            a_g_m_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });
        b0_gs_ns_ks.ForEach([&](auto& self, auto idx) {
            b0_g_k_n(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
        });
        b1_gs_os_ns.ForEach([&](auto& self, auto idx) {
            b1_g_n_o(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
        });
        d0_gs_ms_ns.ForEach([&](auto& self, auto idx) {
            d0_g_m_n(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });

        // gemm 0
        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            a_g_m_k, b0_g_k_n, acc0_g_m_n, a_element_op, b0_element_op, acc0_element_op);

        ref_gemm0_invoker.Run(ref_gemm0_argument);

        acc0_g_m_n.ForEach([&](auto&, auto idx) {
            c0de_element_op(acc0_g_m_n(idx), acc0_g_m_n(idx), d0_g_m_n(idx));
        });
        // masking
        const auto mask = DeviceOpInstance::C0MatrixMask(N);
        acc0_g_m_n.ForEach([&](auto& self, auto idx) {
            if(mask.IsMaskedElement(idx[1], idx[2]))
                self(idx) = -ck::NumericLimits<float>::Infinity();
        });

        // softmax
        auto ref_softmax          = ReferenceSoftmaxInstance{};
        auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
        auto ref_softmax_argument = ref_softmax.MakeArgument(acc0_g_m_n, a1_g_m_n, 1, 0, {2});

        ref_softmax_invoker.Run(ref_softmax_argument);

        // gemm1
        auto ref_gemm1          = ReferenceGemm1Instance{};
        auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
        auto ref_gemm1_argument = ref_gemm1.MakeArgument(
            a1_g_m_n, b1_g_n_o, c_g_m_o_host_result, PassThrough{}, b1_element_op, c_element_op);

        ref_gemm1_invoker.Run(ref_gemm1_argument);

        // permute
        c_gs_ms_os_host_result.ForEach([&](auto& self, auto idx) {
            const size_t& g0 = idx[0];
            const size_t& g1 = idx[1];

            const size_t g = g0 * G1 + g1;

            self(idx) = c_g_m_o_host_result(g, idx[2], idx[3]);
        });

        // default absolute error and relative error is 0.001
        double rtol = 1e-3;
        double atol = 1e-3;

        return ck::utils::check_err(c_gs_ms_os_device_result.mData,
                                    c_gs_ms_os_host_result.mData,
                                    "Error: Incorrect results!",
                                    rtol,
                                    atol)
                   ? 0
                   : 1;
    }

    return 0;
}
