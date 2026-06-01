// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_contraction_multiple_d_xdl_cshuffle.hpp"

#include "ck/library/tensor_operation_instance/gpu/contraction_bilinear.hpp"

#include "ck/library/utility/device_memory.hpp"

using Pass     = ck::tensor_operation::element_wise::PassThrough;
using Bilinear = ck::tensor_operation::element_wise::Bilinear;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F32 = float;
using F64 = double;

template <ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t CDEBlockTransferScalarPerVector>
class ContractionInstanceWrapper
{
    public:
    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
    static constexpr ck::index_t NumDim = 2;
    // clang-format off
    using ContractionDeviceInstance = ck::tensor_operation::device::
        //#####################################| NumDimM| NumDimN| NumDimK| AData| BData| AccData| CShuffle|         DsData| EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer|             ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer|              BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|                  CBlockTransfer| Compute|
        //#####################################|        |        |        |  Type|  Type|    Type| DataType|           Type|  Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|               SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|               SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl|                 ScalarPerVector|    Data|
        //#####################################|        |        |        |      |      |        |         |               |      |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |                           |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |                           |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|                   _NWaveNPerXdl|    Type|
        //#####################################|        |        |        |      |      |        |         |               |      |            |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |                           |               |               |          |                |               |               |                           |               |               |          |            |            |                             |                                |        |
        DeviceContractionMultipleD_Xdl_CShuffle<  NumDim,  NumDim,  NumDim,   F32,   F32,     F32,      F32, ck::Tuple<F32>,   F32,        Pass,        Pass,     Bilinear,       GemmSpec,        1,   256,   256,   128,    16,   4,   4,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>, ABlockTransferSrcVectorDim,              4,              4,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>, BBlockTransferSrcVectorDim,              4,              4,         1,           1,           1,              S<1, 16, 1, 16>, CDEBlockTransferScalarPerVector,     F32>;
    // clang-format on

    bool isSupported(std::vector<ck::index_t>& ADims,
                     std::vector<ck::index_t>& BDims,
                     std::vector<ck::index_t>& DDims,
                     std::vector<ck::index_t>& EDims,
                     std::vector<ck::index_t>& AStrides,
                     std::vector<ck::index_t>& BStrides,
                     std::vector<ck::index_t>& DStrides,
                     std::vector<ck::index_t>& EStrides) const
    {
        auto contraction = ContractionDeviceInstance{};

        auto argument = contraction.MakeArgument(nullptr,
                                                 nullptr,
                                                 std::array<const void*, 1>{nullptr},
                                                 nullptr,
                                                 ADims,
                                                 AStrides,
                                                 BDims,
                                                 BStrides,
                                                 std::array<std::vector<ck::index_t>, 1>{DDims},
                                                 std::array<std::vector<ck::index_t>, 1>{DStrides},
                                                 EDims,
                                                 EStrides,
                                                 Pass{},
                                                 Pass{},
                                                 Bilinear{1.f, 1.f});
        return contraction.IsSupportedArgument(argument);
    }
};

template <typename DataTypeA,
          typename DataTypeB,
          typename DataTypeC,
          typename DataTypeD,
          ck::index_t NumDim>
class ContractionDeviceOpWrapper
{

    protected:
    using DeviceOp = ck::tensor_operation::device::DeviceContractionMultipleD<NumDim,
                                                                              NumDim,
                                                                              NumDim,
                                                                              DataTypeA,
                                                                              DataTypeB,
                                                                              ck::Tuple<DataTypeC>,
                                                                              DataTypeD,
                                                                              Pass,
                                                                              Pass,
                                                                              Bilinear>;

    public:
    bool IsSupportedInstance(std::vector<ck::index_t>& Dims,
                             std::vector<ck::index_t>& Strides) const
    {

        bool supported     = false;
        const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();

        for(auto& op_ptr : op_ptrs)
        {
            auto argument_ptr =
                op_ptr->MakeArgumentPointer(nullptr,
                                            nullptr,
                                            std::array<const void*, 1>{nullptr},
                                            nullptr,
                                            Dims,
                                            Strides,
                                            Dims,
                                            Strides,
                                            std::array<std::vector<ck::index_t>, 1>{Dims},
                                            std::array<std::vector<ck::index_t>, 1>{Strides},
                                            Dims,
                                            Strides,
                                            Pass{},
                                            Pass{},
                                            Bilinear{1.f, 1.f});

            supported = supported || op_ptr->IsSupportedArgument(argument_ptr.get());
        }
        return supported;
    }
};

TEST(TestContractionInterface, IncorrectDataTypes)
{
    std::vector<ck::index_t> Dims    = {4, 4, 4, 4};
    std::vector<ck::index_t> Strides = {64, 16, 4, 1};
    ContractionDeviceOpWrapper<F32, F32, F64, F64, 2> wrapper_1;
    ContractionDeviceOpWrapper<F64, F64, F32, F32, 2> wrapper_2;
    EXPECT_FALSE(wrapper_1.IsSupportedInstance(Dims, Strides));
    EXPECT_FALSE(wrapper_2.IsSupportedInstance(Dims, Strides));
}

TEST(TestContractionSupportedArgs, ABMemoryAccess)
{
    std::vector<ck::index_t> Dims           = {4, 4, 4, 4};
    std::vector<ck::index_t> Strides        = {64, 16, 4, 1};
    std::vector<ck::index_t> StridesM1      = {4, 1, 64, 16};
    std::vector<ck::index_t> StridesK1      = {64, 16, 4, 1};
    std::vector<ck::index_t> InvalidStrides = {4, 4, 4, 4};
    // Memory access to A
    ContractionInstanceWrapper<1, 2, 4> wrapperA1;
    ContractionInstanceWrapper<2, 2, 4> wrapperA2;
    EXPECT_FALSE(
        wrapperA1.isSupported(Dims, Dims, Dims, Dims, InvalidStrides, Strides, Strides, Strides));
    EXPECT_FALSE(
        wrapperA2.isSupported(Dims, Dims, Dims, Dims, InvalidStrides, Strides, Strides, Strides));
    EXPECT_TRUE(
        wrapperA1.isSupported(Dims, Dims, Dims, Dims, StridesM1, Strides, Strides, Strides));
    EXPECT_TRUE(
        wrapperA2.isSupported(Dims, Dims, Dims, Dims, StridesK1, Strides, Strides, Strides));
    // Memory access to B
    ContractionInstanceWrapper<2, 1, 4> wrapperB1;
    ContractionInstanceWrapper<2, 2, 4> wrapperB2;
    EXPECT_FALSE(
        wrapperB1.isSupported(Dims, Dims, Dims, Dims, Strides, InvalidStrides, Strides, Strides));
    EXPECT_FALSE(
        wrapperB2.isSupported(Dims, Dims, Dims, Dims, Strides, InvalidStrides, Strides, Strides));
    EXPECT_TRUE(
        wrapperB1.isSupported(Dims, Dims, Dims, Dims, Strides, StridesM1, Strides, Strides));
    EXPECT_TRUE(
        wrapperB2.isSupported(Dims, Dims, Dims, Dims, Strides, StridesK1, Strides, Strides));
}

TEST(TestContractionSupportedArgs, DEMemoryAccess)
{
    std::vector<ck::index_t> Dims           = {4, 4, 4, 4};
    std::vector<ck::index_t> Strides        = {64, 16, 4, 1};
    std::vector<ck::index_t> InvalidStrides = {64, 16, 1, 4};
    ContractionInstanceWrapper<2, 2, 4> wrapper;
    // Memory access to D
    EXPECT_FALSE(
        wrapper.isSupported(Dims, Dims, Dims, Dims, Strides, Strides, InvalidStrides, Strides));
    EXPECT_TRUE(wrapper.isSupported(Dims, Dims, Dims, Dims, Strides, Strides, Strides, Strides));
    // Memory access to E
    EXPECT_FALSE(
        wrapper.isSupported(Dims, Dims, Dims, Dims, Strides, Strides, Strides, InvalidStrides));
    EXPECT_TRUE(wrapper.isSupported(Dims, Dims, Dims, Dims, Strides, Strides, Strides, Strides));
}

int main(int argc, char** argv)
{
    if(ck::is_gfx11_supported() || ck::is_gfx12_supported())
    {
        std::cout << "FP32/64 are not supported on gfx11 and gfx12." << std::endl;
        return 0;
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
