// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "impl/conv_algorithm_types.hpp"
#include "impl/conv_signature_types.hpp"
#include "ck_tile/builder/conv_builder.hpp"

namespace ck_tile::builder::test_utils {

using namespace ck_tile::builder;
using namespace test;

constexpr DlThreadConfig DlThreadConfig_16x2x4x4x1{
    .k0_per_block = 16, .k1 = 2, .m1_per_thread = 4, .n1_per_thread = 4, .k_per_thread = 1};

constexpr DlThreadCluster DlThreadCluster_8x2{.m1_xs = {8, 2}, .n1_xs = {8, 2}};

constexpr DlBlockTransfer DlBlockTransferAB{.thread_slice_lengths         = {8, 1, 1, 2},
                                            .thread_cluster_lengths       = {2, 1, 128, 1},
                                            .thread_cluster_arrange_order = {1, 2, 0, 3},
                                            .src_access_order             = {1, 2, 0, 3},
                                            .src_vector_tensor_lengths    = {4, 1, 1, 2},
                                            .src_vector_tensor_contiguous_dim_order = {1, 2, 0, 3},
                                            .dst_vector_tensor_lengths              = {1, 1, 1, 2}};

constexpr DlTransferABC DlFwdTransfer{.a =
                                          {
                                              .block_transfer = DlBlockTransferAB,
                                          },
                                      .b =
                                          {
                                              .block_transfer = DlBlockTransferAB,
                                          },
                                      .c = {
                                          .epilogue = {.src_dst_access_order  = {0, 1, 2, 3, 4, 5},
                                                       .src_dst_vector_dim    = 5,
                                                       .dst_scalar_per_vector = 4},
                                      }};

constexpr TransferABC FwdTransfer_4x64x1{
    .a =
        {
            .block_transfer              = {.k0 = 4, .m_n = 64, .k1 = 1},
            .lds_transfer                = {.src_vector_dim            = 2,
                                            .src_scalar_per_vector     = 2,
                                            .lds_dst_scalar_per_vector = 8,
                                            .is_direct_load            = false,
                                            .lds_padding               = false},
            .block_transfer_access_order = {1, 0, 2},
            .src_access_order            = {1, 0, 2},
        },
    .b =
        {
            .block_transfer              = {.k0 = 4, .m_n = 64, .k1 = 1},
            .lds_transfer                = {.src_vector_dim            = 2,
                                            .src_scalar_per_vector     = 8,
                                            .lds_dst_scalar_per_vector = 8,
                                            .is_direct_load            = false,
                                            .lds_padding               = false},
            .block_transfer_access_order = {1, 0, 2},
            .src_access_order            = {1, 0, 2},
        },
    .c =
        {
            .thread_cluster_dims =
                {.m_block = 1, .m_wave_per_xdl = 32, .n_block = 1, .n_wave_per_xdl = 8},
            .epilogue = {.m_xdl_per_wave_per_shuffle = 1,
                         .n_per_wave_per_shuffle     = 1,
                         .scalar_per_vector          = 4},
        },
};

constexpr TransferABC FwdTransfer_4x64x1_fp8{
    .a =
        {
            .block_transfer              = {.k0 = 4, .m_n = 64, .k1 = 1},
            .lds_transfer                = {.src_vector_dim            = 2,
                                            .src_scalar_per_vector     = 8,
                                            .lds_dst_scalar_per_vector = 8,
                                            .is_direct_load            = false,
                                            .lds_padding               = true},
            .block_transfer_access_order = {1, 0, 2},
            .src_access_order            = {1, 0, 2},
        },
    .b =
        {
            .block_transfer              = {.k0 = 4, .m_n = 64, .k1 = 1},
            .lds_transfer                = {.src_vector_dim            = 2,
                                            .src_scalar_per_vector     = 8,
                                            .lds_dst_scalar_per_vector = 8,
                                            .is_direct_load            = false,
                                            .lds_padding               = true},
            .block_transfer_access_order = {1, 0, 2},
            .src_access_order            = {1, 0, 2},
        },
    .c =
        {
            .thread_cluster_dims =
                {.m_block = 1, .m_wave_per_xdl = 32, .n_block = 1, .n_wave_per_xdl = 8},
            .epilogue = {.m_xdl_per_wave_per_shuffle = 1,
                         .n_per_wave_per_shuffle     = 1,
                         .scalar_per_vector          = 8},
        },
};

constexpr TransferABC FwdTransfer_4x16x1{
    .a =
        {
            .block_transfer              = {.k0 = 4, .m_n = 16, .k1 = 1},
            .lds_transfer                = {.src_vector_dim            = 2,
                                            .src_scalar_per_vector     = 8,
                                            .lds_dst_scalar_per_vector = 8,
                                            .is_direct_load            = false,
                                            .lds_padding               = true},
            .block_transfer_access_order = {1, 0, 2},
            .src_access_order            = {1, 0, 2},
        },
    .b =
        {
            .block_transfer              = {.k0 = 4, .m_n = 16, .k1 = 1},
            .lds_transfer                = {.src_vector_dim            = 2,
                                            .src_scalar_per_vector     = 8,
                                            .lds_dst_scalar_per_vector = 8,
                                            .is_direct_load            = false,
                                            .lds_padding               = true},
            .block_transfer_access_order = {1, 0, 2},
            .src_access_order            = {1, 0, 2},
        },
    .c =
        {
            .thread_cluster_dims =
                {.m_block = 1, .m_wave_per_xdl = 16, .n_block = 1, .n_wave_per_xdl = 4},
            .epilogue = {.m_xdl_per_wave_per_shuffle = 1,
                         .n_per_wave_per_shuffle     = 1,
                         .scalar_per_vector          = 8},

        },
};

constexpr TransferABC FwdTransfer_4x32x1{
    .a =
        {
            .block_transfer              = {.k0 = 4, .m_n = 32, .k1 = 1},
            .lds_transfer                = {.src_vector_dim            = 2,
                                            .src_scalar_per_vector     = 16,
                                            .lds_dst_scalar_per_vector = 16,
                                            .is_direct_load            = false,
                                            .lds_padding               = true},
            .block_transfer_access_order = {1, 0, 2},
            .src_access_order            = {1, 0, 2},
        },
    .b =
        {
            .block_transfer              = {.k0 = 4, .m_n = 32, .k1 = 1},
            .lds_transfer                = {.src_vector_dim            = 2,
                                            .src_scalar_per_vector     = 16,
                                            .lds_dst_scalar_per_vector = 16,
                                            .is_direct_load            = false,
                                            .lds_padding               = true},
            .block_transfer_access_order = {1, 0, 2},
            .src_access_order            = {1, 0, 2},
        },
    .c =
        {
            .thread_cluster_dims =
                {.m_block = 1, .m_wave_per_xdl = 32, .n_block = 1, .n_wave_per_xdl = 4},
            .epilogue = {.m_xdl_per_wave_per_shuffle = 1,
                         .n_per_wave_per_shuffle     = 1,
                         .scalar_per_vector          = 8},
        },
};

constexpr GridwiseXdlGemm FwdGemmParams_Xdl_4x4_per_wave{
    .ak1 = 8, .bk1 = 8, .m_per_xdl = 32, .n_per_xdl = 32, .m_xdl_per_wave = 4, .n_xdl_per_wave = 4};

constexpr GridwiseXdlGemm FwdGemmParams_Xdl_4x2_per_wave{
    .ak1 = 8, .bk1 = 8, .m_per_xdl = 32, .n_per_xdl = 32, .m_xdl_per_wave = 4, .n_xdl_per_wave = 2};

constexpr GridwiseXdlGemm FwdGemmParams_Xdl_2x2_per_wave{
    .ak1 = 8, .bk1 = 8, .m_per_xdl = 32, .n_per_xdl = 32, .m_xdl_per_wave = 2, .n_xdl_per_wave = 2};

constexpr GridwiseXdlGemm FwdGemmParams_Xdl_2x1_per_wave{
    .ak1 = 8, .bk1 = 8, .m_per_xdl = 32, .n_per_xdl = 32, .m_xdl_per_wave = 2, .n_xdl_per_wave = 1};

constexpr GridwiseWmmaGemm FwdGemmParams_Wmma_2x1_per_wave{.k1               = 8,
                                                           .m_per_wmma       = 32,
                                                           .n_per_wmma       = 32,
                                                           .m_wmma_per_wave  = 2,
                                                           .n_wmma_per_wave  = 1,
                                                           .pipeline_version = PipelineVersion::V1};

constexpr ThreadBlock FwdThreadBlock_256_256x256x32{.block_size = 256,
                                                    .tile_size  = {.m = 256, .n = 256, .k = 32}};

constexpr ThreadBlock FwdThreadBlock_256_256x128x32{.block_size = 256,
                                                    .tile_size  = {.m = 256, .n = 128, .k = 32}};

constexpr ThreadBlock FwdThreadBlock_256_128x128x32{.block_size = 256,
                                                    .tile_size  = {.m = 128, .n = 128, .k = 32}};

constexpr ThreadBlock FwdThreadBlock_256_128x128x16{.block_size = 256,
                                                    .tile_size  = {.m = 128, .n = 128, .k = 16}};

constexpr ThreadBlock FwdThreadBlock_64_64x32x32{.block_size = 64,
                                                 .tile_size  = {.m = 64, .n = 32, .k = 32}};

constexpr ThreadBlock FwdThreadBlock_128_128x128x32{.block_size = 128,
                                                    .tile_size  = {.m = 128, .n = 128, .k = 32}};

constexpr ThreadBlock FwdThreadBlock_128_64x64x64{.block_size = 128,
                                                  .tile_size  = {.m = 64, .n = 64, .k = 64}};

constexpr BlockGemm BlockGemmDesc_v1_intrawave = {.pipeline_version = PipelineVersion::V1,
                                                  .scheduler        = PipelineScheduler::INTRAWAVE};

constexpr BlockGemm BlockGemmDesc_v2_intrawave = {.pipeline_version = PipelineVersion::V2,
                                                  .scheduler        = PipelineScheduler::INTRAWAVE};

constexpr BlockGemm BlockGemmDesc_v3_intrawave = {.pipeline_version = PipelineVersion::V3,
                                                  .scheduler        = PipelineScheduler::INTRAWAVE};

constexpr BlockGemm BlockGemmDesc_v4_intrawave = {.pipeline_version = PipelineVersion::V4,
                                                  .scheduler        = PipelineScheduler::INTRAWAVE};

constexpr BlockGemm BlockGemmDesc_v5_intrawave = {.pipeline_version = PipelineVersion::V5,
                                                  .scheduler        = PipelineScheduler::INTRAWAVE};

} // namespace ck_tile::builder::test_utils
