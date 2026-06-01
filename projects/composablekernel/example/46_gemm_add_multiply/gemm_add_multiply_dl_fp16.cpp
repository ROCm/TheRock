// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_dl.hpp"

using ADataType   = F16;
using BDataType   = F16;
using AccDataType = F32;
using D0DataType  = F16;
using D1DataType  = F16;
using DsDataType  = ck::Tuple<D0DataType, D1DataType>;
using EDataType   = F16;

using ALayout  = Row;
using BLayout  = Row;
using D0Layout = Row;
using D1Layout = Row;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;
using ELayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = AddMultiply;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNPadding;

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::
        //  ##################| ALayout| BLayout| DsLayout| ELayout|     AData|     BData|     AccData|     DsData|     EData|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|       ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|       BBlockTransfer|     CThreadTransfer|  CThreadTransfer|    CThreadTransfer|
        //  ##################|        |        |         |        |      Type|      Type|        Type|       Type|      Type| Elementwise| Elementwise|  Elementwise| Specialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|      DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|      DstVectorTensor|        SrcDstAccess|  SrcDstVectorDim| DstScalarPerVector|
        //  ##################|        |        |         |        |          |          |            |           |          |   Operation|   Operation|    Operation|               |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder|  Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder|  Lengths_K0_N0_N1_K1|               Order|                 |                   |
        //  ##################|        |        |         |        |          |          |            |           |          |            |            |             |               |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                     |                   |                     |               |               |                    |                   |                     |                    |                 |                   |
        DeviceGemmMultipleD_Dl< ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,    GemmDefault,   256,   128,   128,    16,  2,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 2>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,        S<1, 1, 1, 2>,      S<2, 1, 4, 2>,      S<8, 1,  32, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  4>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        AccDataType,
                                                                        AccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        PassThrough>;

#include "run_gemm_add_multiply_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_add_multiply_example(argc, argv); }
