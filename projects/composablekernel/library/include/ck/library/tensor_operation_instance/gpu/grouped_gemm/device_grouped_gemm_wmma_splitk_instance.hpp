// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_splitk_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/utility/loop_scheduler.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F8   = ck::f8_t;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Empty_Tuple = ck::Tuple<>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using FastGelu    = ck::tensor_operation::element_wise::FastGelu;

using AccDataType = F32;
using DsDataType  = Empty_Tuple;

using DsLayout = Empty_Tuple;
using ELayout  = Row;

static constexpr auto PipelineV1         = BlockGemmPipelineVersion::v1;
static constexpr auto PipelineV3         = BlockGemmPipelineVersion::v3;
static constexpr auto IntrawaveScheduler = BlockGemmPipelineScheduler::Intrawave;
static constexpr auto InterwaveScheduler = BlockGemmPipelineScheduler::Interwave;
static constexpr auto GemmMNKPadding     = device::GemmSpecialization::MNKPadding;
static constexpr auto GemmDefault        = device::GemmSpecialization::Default;

// Instances for 2 byte datatypes in CRR layout with ADataType = BDataType = EDataType
template <typename T,
          device::GemmSpecialization GemmSpec,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp,
          enable_if_t<sizeof(T) == 2, bool> = false>
using device_grouped_gemm_wmma_universal_km_kn_mn_instances =
    std::tuple<
        // clang-format off
        //##############################| ALayout| BLayout| DsLayout| ELayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //##############################|        |        |         |        |      Type|      Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|              _MBlock_MRepeat| ScalarPerVector|
        //##############################|        |        |         |        |          |          |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|              _NBlock_NRepeat|        _NRepeat|
        //##############################|        |        |         |        |          |          |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedGemm_Wmma_CShuffleV3<    Col,     Row, DsLayout, ELayout,         T,         T, AccDataType,                T, DsDataType,         T,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   8,   8,   16,   16,       2,       4,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>,
        DeviceGroupedGemm_Wmma_CShuffleV3<    Col,     Row, DsLayout, ELayout,         T,         T, AccDataType,                T, DsDataType,         T,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   2,   2,   16,   16,       2,       4,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              2,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              2,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>,
        DeviceGroupedGemm_Wmma_CShuffleV3<    Col,     Row, DsLayout, ELayout,         T,         T, AccDataType,                T, DsDataType,         T,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    32,   8,   8,   16,   16,       2,       4,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>
    // clang`-format on
    >;

// Instances for 2 byte datatypes in CCR layout with ADataType = BDataType = EDataType
template <typename T,
          device::GemmSpecialization GemmSpec,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp,
          enable_if_t<sizeof(T) == 2, bool> = false>
using device_grouped_gemm_wmma_universal_km_nk_mn_instances = std::tuple<
    // clang-format off
        //##############################| ALayout| BLayout| DsLayout| ELayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //##############################|        |        |         |        |      Type|      Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|              _MBlock_MRepeat| ScalarPerVector|
        //##############################|        |        |         |        |          |          |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|              _NBlock_NRepeat|        _NRepeat|
        //##############################|        |        |         |        |          |          |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedGemm_Wmma_CShuffleV3<    Col,     Col, DsLayout, ELayout,         T,         T, AccDataType,                T, DsDataType,         T,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   8,   8,   16,   16,       2,       4,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>,
        DeviceGroupedGemm_Wmma_CShuffleV3<    Col,     Col, DsLayout, ELayout,         T,         T, AccDataType,                T, DsDataType,         T,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   2,   2,   16,   16,       2,       4,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              2,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              2,              2,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>,
        DeviceGroupedGemm_Wmma_CShuffleV3<    Col,     Col, DsLayout, ELayout,         T,         T, AccDataType,                T, DsDataType,         T,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    32,   8,   8,   16,   16,       2,       4,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>
        // clang-format on
        >;

// Instances for 2 byte datatypes in RRR layout with ADataType = BDataType = EDataType
template <typename T,
          device::GemmSpecialization GemmSpec,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp,
          enable_if_t<sizeof(T) == 2, bool> = false>
using device_grouped_gemm_wmma_universal_mk_kn_mn_instances =
    std::tuple<
        // clang-format off
        //##############################| ALayout| BLayout| DsLayout| ELayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //##############################|        |        |         |        |      Type|      Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|              _MBlock_MRepeat| ScalarPerVector|
        //##############################|        |        |         |        |          |          |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|              _NBlock_NRepeat|        _NRepeat|
        //##############################|        |        |         |        |          |          |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedGemm_Wmma_CShuffleV3<    Row,     Row, DsLayout, ELayout,         T,         T, AccDataType,                T, DsDataType,         T,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   8,   8,   16,   16,       2,       4,     S<8, 32, 1>,     S<2, 0, 1>,     S<2, 0, 1>,              2,              8,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>,
        DeviceGroupedGemm_Wmma_CShuffleV3<    Row,     Row, DsLayout, ELayout,         T,         T, AccDataType,                T, DsDataType,         T,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   2,   2,   16,   16,       2,       4,     S<8, 32, 1>,     S<2, 0, 1>,     S<2, 0, 1>,              2,              2,              2,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              2,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>,
        DeviceGroupedGemm_Wmma_CShuffleV3<    Row,     Row, DsLayout, ELayout,         T,         T, AccDataType,                T, DsDataType,         T,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    32,   8,   8,   16,   16,       2,       4,     S<4, 32, 1>,     S<2, 0, 1>,     S<2, 0, 1>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>
        // clang-format on
        >;

// Instances for 2 byte datatypes in RCR layout with ADataType = BDataType = EDataType
template <typename T,
          device::GemmSpecialization GemmSpec,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp,
          enable_if_t<sizeof(T) == 2, bool> = false>
using device_grouped_gemm_wmma_universal_mk_nk_mn_instances =
    std::tuple<
        // clang-format off
        //##############################| ALayout| BLayout| DsLayout| ELayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //##############################|        |        |         |        |      Type|      Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|              _MBlock_MRepeat| ScalarPerVector|
        //##############################|        |        |         |        |          |          |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|              _NBlock_NRepeat|        _NRepeat|
        //##############################|        |        |         |        |          |          |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedGemm_Wmma_CShuffleV3<    Row,     Col, DsLayout, ELayout,         T,         T, AccDataType,                T, DsDataType,         T,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   8,   8,   16,   16,       2,       4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>,
        DeviceGroupedGemm_Wmma_CShuffleV3<    Row,     Col, DsLayout, ELayout,         T,         T, AccDataType,                T, DsDataType,         T,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   2,   2,   16,   16,       2,       4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              2,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              2,              2,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>,
        DeviceGroupedGemm_Wmma_CShuffleV3<    Row,     Col, DsLayout, ELayout,         T,         T, AccDataType,                T, DsDataType,         T,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    32,   8,   8,   16,   16,       2,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>
        // clang-format on
        >;

// List of instance variants to add (pipeline/scheduler/padding combinations)
// Some are disabled now, can be re-enabled if needed
using InstanceVariant =
    ck::Tuple<device::GemmSpecialization, BlockGemmPipelineScheduler, BlockGemmPipelineVersion>;
static constexpr InstanceVariant InstanceVariants[] = {

    make_tuple(GemmDefault, IntrawaveScheduler, PipelineV1),
    // make_tuple(GemmDefault, InterwaveScheduler, PipelineV1),
    make_tuple(GemmDefault, IntrawaveScheduler, PipelineV3),

    make_tuple(GemmMNKPadding, IntrawaveScheduler, PipelineV1),
    // make_tuple(GemmMNKPadding, InterwaveScheduler, PipelineV1),
    // make_tuple(GemmMNKPadding, IntrawaveScheduler, PipelineV3),
};

// Helper function to add a list of layout instances with specific A/B/E datatypes for all supported
// padding/scheduler/pipeline version combinations
template <typename ALayout,
          typename BLayout,
          template <device::GemmSpecialization GemmSpec,
                    BlockGemmPipelineScheduler BlkGemmPipeSched,
                    BlockGemmPipelineVersion BlkGemmPipelineVer,
                    typename AElementOp,
                    typename BElementOp,
                    typename CDEElementOp>
          typename LayoutInstances,
          typename ADataType, // NOTE: type parameters as last so that they can be inferred from the
          typename BDataType, // vector argument
          typename EDataType,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp>
void add_device_grouped_gemm_wmma_universal_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<ALayout,
                                                  BLayout,
                                                  DsLayout,
                                                  ELayout,
                                                  ADataType,
                                                  BDataType,
                                                  DsDataType,
                                                  EDataType,
                                                  AElementOp,
                                                  BElementOp,
                                                  CDEElementOp>>>& instances)
{
    // Add all instances from our instance list
    static_for<0, std::size(InstanceVariants), 1>{}([&](auto i) {
        constexpr auto instance = InstanceVariants[i];
        add_device_operation_instances(instances,
                                       LayoutInstances<instance.At(Number<0>{}),
                                                       instance.At(Number<1>{}),
                                                       instance.At(Number<2>{}),
                                                       AElementOp,
                                                       BElementOp,
                                                       CDEElementOp>{});
    });
}

// Helper function to add a list of layout instances for instances with matching A/B/E data types
// for all supported padding/scheduler/pipeline version combinations
template <typename T,
          typename ALayout,
          typename BLayout,
          template <typename T2,
                    device::GemmSpecialization GemmSpec,
                    BlockGemmPipelineScheduler BlkGemmPipeSched,
                    BlockGemmPipelineVersion BlkGemmPipelineVer,
                    typename AElementOp,
                    typename BElementOp,
                    typename CDEElementOp>
          typename LayoutInstances,
          typename AElementOp, // NOTE: element-wise op parameters as last so that they can be
          typename BElementOp, // inferred from the vector argument
          typename CDEElementOp>
void add_device_grouped_gemm_wmma_universal_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<ALayout,
                                                  BLayout,
                                                  DsLayout,
                                                  ELayout,
                                                  T,
                                                  T,
                                                  DsDataType,
                                                  T,
                                                  AElementOp,
                                                  BElementOp,
                                                  CDEElementOp>>>& instances)
{
    // Add all instances from our instance list
    static_for<0, std::size(InstanceVariants), 1>{}([&](auto i) {
        constexpr auto instance = InstanceVariants[i];
        add_device_operation_instances(instances,
                                       LayoutInstances<T,
                                                       instance.At(Number<0>{}),
                                                       instance.At(Number<1>{}),
                                                       instance.At(Number<2>{}),
                                                       AElementOp,
                                                       BElementOp,
                                                       CDEElementOp>{});
    });
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
