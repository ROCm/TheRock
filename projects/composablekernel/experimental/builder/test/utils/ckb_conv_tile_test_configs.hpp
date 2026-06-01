// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "impl/conv_algorithm_types.hpp"
#include "impl/conv_signature_types.hpp"
#include "ck_tile/builder/conv_builder.hpp"

namespace ck_tile::builder::test_utils {

using namespace ck_tile::builder;
using namespace test;

constexpr TileTransfer FwdTileTransfer_1x1x1{
    .a_scalar_per_vector = 1,
    .b_scalar_per_vector = 1,
    .c_scalar_per_vector = 1,
};

constexpr TileTransfer FwdTileTransfer_4x4x4{
    .a_scalar_per_vector = 4,
    .b_scalar_per_vector = 4,
    .c_scalar_per_vector = 4,
};

constexpr TileTransfer FwdTileTransfer_8x8x8{
    .a_scalar_per_vector = 8,
    .b_scalar_per_vector = 8,
    .c_scalar_per_vector = 8,
};

constexpr TileThreadBlock FwdTileThreadBlock_256x256x32{.tile_size = {.m = 256, .n = 256, .k = 32}};

constexpr TileThreadBlock FwdTileThreadBlock_256x128x32{.tile_size = {.m = 256, .n = 128, .k = 32}};

constexpr TileThreadBlock FwdTileThreadBlock_128x128x32{.tile_size = {.m = 128, .n = 128, .k = 32}};

constexpr TileThreadBlock FwdTileThreadBlock_128x128x16{.tile_size = {.m = 128, .n = 128, .k = 16}};

constexpr TileThreadBlock FwdTileThreadBlock_64x32x32{.tile_size = {.m = 64, .n = 32, .k = 32}};

constexpr TileThreadBlock FwdTileThreadBlock_64x64x64{.tile_size = {.m = 64, .n = 64, .k = 64}};

constexpr TileBlockGemm TileBlockGemmDesc_16x16_v1_intrawave = {
    .warps              = {.m = 2, .n = 2, .k = 1},
    .warp_tile          = {.m = 16, .n = 16, .k = 16},
    .double_smem_buffer = false,
    .num_wave_groups    = 1,
    .pipeline_version   = PipelineVersion::V1,
    .scheduler          = PipelineScheduler::INTRAWAVE};

constexpr TileBlockGemm TileBlockGemmDesc_16x16_v2_intrawave = {
    .warps              = {.m = 2, .n = 2, .k = 1},
    .warp_tile          = {.m = 16, .n = 16, .k = 16},
    .double_smem_buffer = false,
    .num_wave_groups    = 1,
    .pipeline_version   = PipelineVersion::V2,
    .scheduler          = PipelineScheduler::INTRAWAVE};

constexpr TileBlockGemm TileBlockGemmDesc_16x16_v3_intrawave = {
    .warps              = {.m = 2, .n = 2, .k = 1},
    .warp_tile          = {.m = 16, .n = 16, .k = 16},
    .double_smem_buffer = false,
    .num_wave_groups    = 1,
    .pipeline_version   = PipelineVersion::V3,
    .scheduler          = PipelineScheduler::INTRAWAVE};

constexpr TileBlockGemm TileBlockGemmDesc_16x16_v4_intrawave = {
    .warps              = {.m = 2, .n = 2, .k = 1},
    .warp_tile          = {.m = 16, .n = 16, .k = 16},
    .double_smem_buffer = false,
    .num_wave_groups    = 1,
    .pipeline_version   = PipelineVersion::V4,
    .scheduler          = PipelineScheduler::INTRAWAVE};

constexpr TileBlockGemm TileBlockGemmDesc_16x16_v5_intrawave = {
    .warps              = {.m = 2, .n = 2, .k = 1},
    .warp_tile          = {.m = 16, .n = 16, .k = 16},
    .double_smem_buffer = false,
    .num_wave_groups    = 1,
    .pipeline_version   = PipelineVersion::V5,
    .scheduler          = PipelineScheduler::INTRAWAVE};

} // namespace ck_tile::builder::test_utils
