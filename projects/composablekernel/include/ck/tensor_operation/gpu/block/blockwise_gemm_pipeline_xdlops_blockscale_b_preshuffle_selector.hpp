// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_blockscale_b_preshuffle_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_blockscale_b_preshuffle_v3.hpp"
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
          index_t MScaleBlock,
          index_t NScaleBlock,
          index_t KScaleBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack>
constexpr auto BlockGemmBlockScaleBPreshufflePipeline_Selector()
{
    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
    {
        return BlockwiseGemmXdlops_pipeline_blockscale_bpreshuffle_v1<
            BlkGemmPipeSche,
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
            MScaleBlock,
            NScaleBlock,
            KScaleBlock,
            MPerXDL,
            NPerXDL,
            MRepeat,
            NRepeat,
            KPack>{};
    }
#if 0
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
    {
        return BlockwiseGemmXdlops_pipeline_blockscale_bpreshuffle_v2<
            BlkGemmPipeSche,
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
#endif
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
    {
        static_assert(MRepeat >= 4, "MRepeat should at least be 4 in BlockGemmPipelineVersion::v3");
        return BlockwiseGemmXdlops_pipeline_blockscale_bpreshuffle_v3<
            BlkGemmPipeSche,
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
            MScaleBlock,
            NScaleBlock,
            KScaleBlock,
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
