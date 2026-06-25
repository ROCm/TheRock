// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "test_batched_gemm_gemm_util.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_gemm_xdl_cshuffle.hpp"

template <GemmSpecialization GemmSpec>
struct DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using ALayout  = Row;
    using B0Layout = Col;
    using B1Layout = Row;
    using CLayout  = Row;

    using ADataType        = F16;
    using B0DataType       = F16;
    using B1DataType       = F16;
    using AccDataType      = float;
    using CShuffleDataType = float;
    using CDataType        = F16;

    using AElementOp    = PassThrough;
    using B0ElementOp   = PassThrough;
    using Acc0ElementOp = PassThrough;
    using B1ElementOp   = PassThrough;
    using CElementOp    = PassThrough;

    template <ck::index_t... Is>
    using S = ck::Sequence<Is...>;

    // static constexpr auto GemmSpec = std::tuple_element_t<0, Tuple>::value;

    using DeviceGemmGemmInstance = ck::tensor_operation::device::DeviceBatchedGemmGemm_Xdl_CShuffle<
        ALayout,
        B0Layout,
        B1Layout,
        CLayout,
        ADataType,
        B0DataType,
        B1DataType,
        CDataType,
        AccDataType,
        CShuffleDataType,
        AElementOp,
        B0ElementOp,
        Acc0ElementOp,
        B1ElementOp,
        CElementOp,
        GemmSpec,
        1,
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        32,          // KPerBlock
        128,         // Gemm1NPerBlock
        32,          // Gemm1KPerBlock
        8,           // AK1
        8,           // BK1
        2,           // B1K1
        16,          // MPerXDL
        16,          // NPerXDL
        2,           // MXdlPerWave
        8,           // NXdlPerWave
        8,           // Gemm1NXdlPerWave
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
        S<8, 32, 1>, // B1BlockTransfer
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        2,              // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        4>;             // CShuffleBlockTransferScalarPerVector_NPerBlock

    bool IsSupported(int M, int N, int K, int O)
    {
        auto gemm     = DeviceGemmGemmInstance{};
        auto invoker  = gemm.MakeInvoker();
        auto argument = gemm.MakeArgument(static_cast<ADataType*>(nullptr),
                                          static_cast<B0DataType*>(nullptr),
                                          static_cast<B1DataType*>(nullptr),
                                          static_cast<CDataType*>(nullptr),
                                          M,
                                          N,
                                          K,
                                          O,
                                          0,              // BatchCount
                                          0,              // StrideA
                                          0,              // StrideB0
                                          0,              // StrideB1
                                          0,              // StrideC
                                          0,              // BatchStrideA
                                          0,              // BatchStrideB0
                                          0,              // BatchStrideB1
                                          0,              // BatchStrideC
                                          PassThrough{},  // a_element_op
                                          PassThrough{},  // b0_element_op
                                          PassThrough{},  // acc0_element_op
                                          PassThrough{},  // b1_element_op
                                          PassThrough{}); // c_element_op

        return gemm.IsSupportedArgument(argument);
    }
};

template <typename Tuple>
class TestBatchedGemmGemmFP16 : public TestBatchedGemmGemm<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<F16, F16, F16, F16, Row, Col, Row, Row>,
    std::tuple<F16, F16, F16, F16, Row, Col, Col, Row>
    >;
// clang-format on

TYPED_TEST_SUITE(TestBatchedGemmGemmFP16, KernelTypes);

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16) { this->Run(); }

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadM)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {136, 128, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadN)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 136, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadK)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 40, 128, 1},
        {128, 128, 136, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadO)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 136, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddM)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {129, 128, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddN)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 129, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddK)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 33, 128, 1},
        {128, 128, 129, 128, 1},
    };
    this->Run();
}

// If kernel B1Layout is RowMajor, expect not to support odd O size
TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddO)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 129, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, DISABLED_Bench_FP16)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {256, 256, 64, 64, 768},
        {256, 256, 128, 128, 768},
        {512, 512, 64, 64, 768},
        {512, 512, 128, 128, 768},
        {1024, 1024, 64, 64, 768},
        {1024, 1024, 128, 128, 768},
        {2048, 2048, 64, 64, 768},
        {2048, 2048, 128, 128, 768},
        {4096, 4096, 64, 64, 768},
        {4096, 4096, 128, 128, 768},
    };
    this->bench_  = true;
    this->verify_ = false;
    this->Run();
}

using ck::tensor_operation::device::GemmSpecialization;

TEST(TestBatchedGemmGemmInterface, GemmSpecializationSizeMatch)
{
    int P = 120; // requires padding
    int Q = 128; // do not require padding

    // IsSupported(M, N, K, O)
    // clang-format off
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::Default>{}.IsSupported(Q, Q, Q, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MPadding>{}.IsSupported(P, Q, Q, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::NPadding>{}.IsSupported(Q, P, Q, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::KPadding>{}.IsSupported(Q, Q, P, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNPadding>{}.IsSupported(P, P, Q, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MKPadding>{}.IsSupported(P, Q, P, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::NKPadding>{}.IsSupported(Q, P, P, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKPadding>{}.IsSupported(P, P, P, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::OPadding>{}.IsSupported(Q, Q, Q, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MOPadding>{}.IsSupported(P, Q, Q, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::NOPadding>{}.IsSupported(Q, P, Q, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::KOPadding>{}.IsSupported(Q, Q, P, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNOPadding>{}.IsSupported(P, P, Q, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MKOPadding>{}.IsSupported(P, Q, P, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::NKOPadding>{}.IsSupported(Q, P, P, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKOPadding>{}.IsSupported(P, P, P, P));
    // clang-format on
}

TEST(TestBatchedGemmGemmInterface, GemmSpecializationSizeMismatch)
{
    // IsSupported(M, N, K, O)
    // clang-format off
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::Default>{}.IsSupported(128, 128, 120, 128));
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKPadding>{}.IsSupported(128, 128, 128, 120));
    // Kernel can't support odd K size because SrcVectorDim == KDim and must satisfy SizeKRaw % ABSrcScalarPerVector == 0
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKOPadding>{}.IsSupported(128, 128, 129, 128));
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKOPadding>{}.IsSupported(128, 128, 130, 128));
    // Kernel can't support odd O size because SrcVectorDim == ODim and must satisfy SizeORaw % B1SrcScalarPerVector == 0
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKOPadding>{}.IsSupported(128, 128, 128, 129));
    // clang-format on
}
