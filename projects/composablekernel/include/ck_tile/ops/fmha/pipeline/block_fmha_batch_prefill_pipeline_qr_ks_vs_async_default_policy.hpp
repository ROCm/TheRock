// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
using BlockFmhaBatchPrefillPipelineQRKSVSAsyncDefaultPolicy =
    BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                        /* AsyncCopy = */ true,
                                        /* NumPrefetchK = */ 3,
                                        /* NumPrefetchV = */ 3>;

} // namespace ck_tile
