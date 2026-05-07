// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_ab_scale.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_ab_scale.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_ab_scale.hpp"

namespace ck {

template <BlockGemmPipelineVersion BlkGemmPipelineVer,
          BlockGemmPipelineScheduler BlkGemmPipeSche,
          index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename ComputeDataType,
          typename AccDataType,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack>
constexpr auto BlockGemmABScalePipeline_Selector()
{
    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
    {
        return BlockwiseGemmXdlops_pipeline_v1_ab_scale<BlkGemmPipeSche,
                                                        BlockSize,
                                                        ADataType,
                                                        BDataType,
                                                        ComputeDataType,
                                                        AccDataType,
                                                        ATileDesc,
                                                        BTileDesc,
                                                        AMmaTileDesc,
                                                        BMmaTileDesc,
                                                        ABlockTransferSrcScalarPerVector,
                                                        BBlockTransferSrcScalarPerVector,
                                                        MPerBlock,
                                                        NPerBlock,
                                                        KPerBlock,
                                                        MPerXDL,
                                                        NPerXDL,
                                                        MRepeat,
                                                        NRepeat,
                                                        KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
    {
        return BlockwiseGemmXdlops_pipeline_v2_ab_scale<BlkGemmPipeSche,
                                                        BlockSize,
                                                        ADataType,
                                                        BDataType,
                                                        ComputeDataType,
                                                        AccDataType,
                                                        ATileDesc,
                                                        BTileDesc,
                                                        AMmaTileDesc,
                                                        BMmaTileDesc,
                                                        ABlockTransferSrcScalarPerVector,
                                                        BBlockTransferSrcScalarPerVector,
                                                        MPerBlock,
                                                        NPerBlock,
                                                        KPerBlock,
                                                        MPerXDL,
                                                        NPerXDL,
                                                        MRepeat,
                                                        NRepeat,
                                                        KPack>{};
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
    {
        return BlockwiseGemmXdlops_pipeline_v3_ab_scale<BlkGemmPipeSche,
                                                        BlockSize,
                                                        ADataType,
                                                        BDataType,
                                                        ComputeDataType,
                                                        AccDataType,
                                                        ATileDesc,
                                                        BTileDesc,
                                                        AMmaTileDesc,
                                                        BMmaTileDesc,
                                                        ABlockTransferSrcScalarPerVector,
                                                        BBlockTransferSrcScalarPerVector,
                                                        MPerBlock,
                                                        NPerBlock,
                                                        KPerBlock,
                                                        MPerXDL,
                                                        NPerXDL,
                                                        MRepeat,
                                                        NRepeat,
                                                        KPack>{};
    }
    else
    {
        std::cerr << "BlockGemmPipeline configuration is not available" << std::endl;
    }
}

} // namespace ck
