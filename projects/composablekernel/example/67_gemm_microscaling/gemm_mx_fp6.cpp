// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_mx_common.hpp"

using ADataType = ck::f6x16_pk_t;
using BDataType = ck::f6x16_pk_t;

using XDataType = ck::e8m0_bexp_t;

using CDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = CDataType;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough; // elementwise transformation for A matrix
using BElementOp = PassThrough; // elementwise transformation for B matrix
using CElementOp = PassThrough; // elementwise transformation for C matrix

constexpr ck::index_t ScaleBlockSize = 32;                            // scaling block size
constexpr ck::index_t KPerBlock = 256 / ck::packed_size_v<ADataType>; // K dimension size per block

constexpr auto GemmSpec      = ck::tensor_operation::device::GemmSpecialization::Default;
constexpr auto BlkGemmPSched = ck::BlockGemmPipelineScheduler::Intrawave;
constexpr auto BlkGemmPVer   = ck::BlockGemmPipelineVersion::v1;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMX_Xdl_CShuffleV3<
    ALayout,          // ALayout
    BLayout,          // BLayout
    CLayout,          // CLayout
    ADataType,        // ADataType
    XDataType,        // AScaleDataType
    BDataType,        // BDataType
    XDataType,        // BScaleDataType
    CDataType,        // CDataType
    AccDataType,      // GemmAccDataType
    CShuffleDataType, // CShuffleDataType
    AElementOp,       // AElementwiseOperation
    BElementOp,       // BElementwiseOperation
    CElementOp,       // CElementwiseOperation
    GemmSpec,         // GemmSpec
    ScaleBlockSize,   // ScaleBlockSize: Scaling block size
    256,              // BlockSize: Number of threads per block
    128,              // MPerBlock
    128,              // NPerBlock
    KPerBlock,        // KPerBlock
    1,  // AK1 number of elements to read at a time when transferring from global memory to LDS
    1,  // BK1
    16, // MPerXDL
    16, // NPerXDL
    4,  // MXdlPerWave
    4,  // NXdlPerWave
    S<16, 16, 1>,   // ABlockTransferThreadClusterLengths_AK0_M_AK1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    1,              // ABlockTransferSrcScalarPerVector
    16,             // ABlockTransferDstScalarPerVector_AK1
    true,           // ABlockLdsExtraM
    S<16, 16, 1>,   // BBlockTransferThreadClusterLengths_BK0_N_BK1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    1,              // BBlockTransferSrcScalarPerVector
    16,             // BBlockTransferDstScalarPerVector_BK1
    true,           // BBlockLdsExtraN
    2,              // CShuffleMXdlPerWavePerShuffle
    2,              // CShuffleNXdlPerWavePerShuffle
    S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
    BlkGemmPSched,  // BlkGemmPipeSched
    BlkGemmPVer,    // BlkGemmPipelineVer
    ADataType,      // ComputeTypeA
    BDataType       // ComputeTypeB
    >;

int main(int argc, char* argv[])
{
    return run_mx_gemm_example<DeviceOpInstance,
                               ADataType,
                               BDataType,
                               XDataType,
                               XDataType,
                               CDataType,
                               ALayout,
                               BLayout,
                               CLayout,
                               AElementOp,
                               BElementOp,
                               CElementOp,
                               AccDataType,
                               CShuffleDataType,
                               ScaleBlockSize>(argc, argv)
               ? 0
               : -1;
}
