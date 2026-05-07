// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_v3.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_gufusion_v3.hpp"

namespace ck {
template <BlockGemmPipelineVersion BlkGemmPipelineVer,
          BlockGemmPipelineScheduler BlkGemmPipeSche,
          index_t ThreadBlockSize,
          index_t ScaleBlockSize,
          typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename ComputeDataType, // TODO: remove this as in this pipeline ADataType and BDataType
                                    // must be used for compute
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
          index_t KPack,
          bool GUFusion = false>
constexpr auto BlockGemmMXBPreshufflePipeline_Selector()
{

    // Hardware MX GEMM pipeline
    if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
    {
        if constexpr(GUFusion)
        {
            return nullptr;
        }
        else
        {
            return BlockwiseGemmXdlops_pipeline_bpreshuffle_mx_moe_v1<
                BlkGemmPipeSche,
                ThreadBlockSize,
                ScaleBlockSize,
                ADataType,
                AScaleDataType,
                BDataType,
                BScaleDataType,
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
    }
    else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
    {
        if constexpr(GUFusion)
        {
            return BlockwiseGemmXdlops_pipeline_bpreshuffle_mx_moe_gufusion_v3<
                BlkGemmPipeSche,
                ThreadBlockSize,
                ScaleBlockSize,
                ADataType,
                AScaleDataType,
                BDataType,
                BScaleDataType,
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
            return BlockwiseGemmXdlops_pipeline_bpreshuffle_mx_moe_v3<
                BlkGemmPipeSche,
                ThreadBlockSize,
                ScaleBlockSize,
                ADataType,
                AScaleDataType,
                BDataType,
                BScaleDataType,
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
    }
    else
    {
        std::cerr << "MX GEMM Pipeline configuration is not available" << std::endl;
    }
}

} // namespace ck
