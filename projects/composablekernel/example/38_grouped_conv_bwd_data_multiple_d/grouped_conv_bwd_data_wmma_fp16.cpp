// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle.hpp"
#include "common.hpp"
#include "ck/host_utility/device_prop.hpp"

using OutDataType      = FP16;
using WeiDataType      = FP16;
using AccDataType      = FP32;
using CShuffleDataType = FP16;
using DsDataType       = ck::Tuple<>;
using InDataType       = FP16;

using OutLayout = ck::tensor_layout::convolution::GNHWK;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using DsLayout  = ck::Tuple<>;
using InLayout  = ck::tensor_layout::convolution::GNHWC;

using OutElementOp = PassThrough;
using WeiElementOp = PassThrough;
using InElementOp  = PassThrough;

// clang-format off
using DeviceConvInstance = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle
        //|    NumDim|        A|         B|       Ds|       E|        AData|        BData|    AccData|          CShuffle|     DsData|       EData|           A|           B|          CDE|       ConvForward| Block|  MPer|  NPer| K0Per| K1|  MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //|   Spatial|   Layout|    Layout|   Layout|  Layout|         Type|         Type|       Type|          DataType|       Type|        Type| Elementwise| Elementwise|  Elementwise|    Specialization|  Size| Block| Block| Block|   |  WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //|          |         |          |         |        |             |             |           |                  |           |            |   Operation|   Operation|    Operation|                  |      |      |      |      |   |      |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //|          |         |          |         |        |             |             |           |                  |           |            |            |            |             |                  |      |      |      |      |   |      |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
           <        2,OutLayout, WeiLayout, DsLayout, InLayout, OutDataType,  WeiDataType, AccDataType, CShuffleDataType, DsDataType,  InDataType, OutElementOp, WeiElementOp, InElementOp, ConvBwdDataDefault, 128,    64,    64,     4,  8,    16,   16,       2,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 32, 1, 4>,               1>;
// clang-format on

#include "run_grouped_conv_bwd_data_example.inc"

int main(int argc, char* argv[])
{
    bool is_supported = ck::is_gfx11_supported();
    if(!is_supported)
    {
        std::cout << "WARNING: wmma example not supported on the platform " << ck::get_device_name()
                  << std::endl;
        return 0;
    }
    return run_grouped_conv_bwd_data_example(argc, argv);
}
