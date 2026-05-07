// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <concepts>
#include <array>

#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder {

/********************************************************************/
/* Descriptors for individual elements of the algorithm description */
/********************************************************************/

// Concept for thread block dimensions for a GEMM problem.
template <typename T>
concept ThreadBlockDescriptor = requires(T t) {
    { t.block_size } -> std::convertible_to<size_t>;
    { t.tile_size.m } -> std::convertible_to<size_t>;
    { t.tile_size.n } -> std::convertible_to<size_t>;
    { t.tile_size.k } -> std::convertible_to<size_t>;
};

// Concept for parameters that describe a gridwise XDL GEMM problem.
template <typename T>
concept GridwiseXdlGemmDescriptor = requires(T t) {
    { t.ak1 } -> std::convertible_to<size_t>;
    { t.bk1 } -> std::convertible_to<size_t>;
    { t.m_per_xdl } -> std::convertible_to<size_t>;
    { t.n_per_xdl } -> std::convertible_to<size_t>;
    { t.m_xdl_per_wave } -> std::convertible_to<size_t>;
    { t.n_xdl_per_wave } -> std::convertible_to<size_t>;
};

// Concept for parameter that describe block GEMM problem.
template <typename T>
concept BlockGemmDescriptor = requires(T t) {
    { t.pipeline_version } -> std::convertible_to<PipelineVersion>;
    { t.scheduler } -> std::convertible_to<PipelineScheduler>;
};

// Concept for parameters that describe a gridwise WMMA GEMM problem.
template <typename T>
concept GridwiseWmmaGemmDescriptor = requires(T t) {
    { t.k1 } -> std::convertible_to<size_t>;
    { t.m_per_wmma } -> std::convertible_to<size_t>;
    { t.n_per_wmma } -> std::convertible_to<size_t>;
    { t.m_wmma_per_wave } -> std::convertible_to<size_t>;
    { t.n_wmma_per_wave } -> std::convertible_to<size_t>;
    { t.pipeline_version } -> std::convertible_to<PipelineVersion>;
};

// Concept for vectorized data transfer for convolution input tensors.
template <typename T>
concept BlockTransferDescriptor = requires(T t) {
    { t.k0 } -> std::convertible_to<size_t>;
    { t.m_n } -> std::convertible_to<size_t>;
    { t.k1 } -> std::convertible_to<size_t>;
};

// Concept for thread cluster dimensions for GEMM output tensor.
template <typename T>
concept ThreadClusterDescriptor = requires(T t) {
    { t.m_block } -> std::convertible_to<size_t>;
    { t.m_wave_per_xdl } -> std::convertible_to<size_t>;
    { t.n_block } -> std::convertible_to<size_t>;
    { t.n_wave_per_xdl } -> std::convertible_to<size_t>;
};

// Concept for the LDS transfer for the convolution input tensors.
template <typename T>
concept LdsTransferDescriptor = requires(T t) {
    { t.src_vector_dim } -> std::convertible_to<size_t>;
    { t.src_scalar_per_vector } -> std::convertible_to<size_t>;
    { t.lds_dst_scalar_per_vector } -> std::convertible_to<size_t>;
    { t.is_direct_load } -> std::convertible_to<bool>;
    { t.lds_padding } -> std::convertible_to<bool>;
};

// Concept for the convolution output tensor epilogue (copy from registers to global memory via
// LDS).
template <typename T>
concept EpilogueDescriptor = requires(T t) {
    { t.m_xdl_per_wave_per_shuffle } -> std::convertible_to<size_t>;
    { t.n_per_wave_per_shuffle } -> std::convertible_to<size_t>;
    { t.scalar_per_vector } -> std::convertible_to<size_t>;
};

// Concept for the thread cluster access order
template <typename T>
concept AccessOrderDescriptor = requires(T t) {
    { t.order } -> std::convertible_to<std::array<size_t, 3>>;
};

// Concept for thread block dimensions for a GEMM problem for CK Tile (Block
// size is deduced from block gemm structure).
template <typename T>
concept TileThreadBlockDescriptor = requires(T t) {
    { t.tile_size.m } -> std::convertible_to<size_t>;
    { t.tile_size.n } -> std::convertible_to<size_t>;
    { t.tile_size.k } -> std::convertible_to<size_t>;
};

// Concept for thread block dimensions for a GEMM problem for CK Tile (Block
// size is deduced from block gemm structure).
template <typename T>
concept TileTransferDescriptor = requires(T t) {
    { t.a_scalar_per_vector } -> std::convertible_to<size_t>;
    { t.b_scalar_per_vector } -> std::convertible_to<size_t>;
    { t.c_scalar_per_vector } -> std::convertible_to<size_t>;
};

// Concept to check if struct specifies block GEMM (CK Tile).
template <typename T>
concept TileBlockGemmDescriptor = requires(T t) {
    { t.warps.m } -> std::convertible_to<int>;
    { t.warps.n } -> std::convertible_to<int>;
    { t.warps.k } -> std::convertible_to<int>;
    { t.warp_tile.m } -> std::convertible_to<int>;
    { t.warp_tile.n } -> std::convertible_to<int>;
    { t.warp_tile.k } -> std::convertible_to<int>;
    { t.double_smem_buffer } -> std::convertible_to<bool>;
    { t.num_wave_groups } -> std::convertible_to<int>;
    { t.pipeline_version } -> std::convertible_to<PipelineVersion>;
    { t.scheduler } -> std::convertible_to<PipelineScheduler>;
};

// Concept to check if struct specifies optimizations (CK Tile).
template <typename T>
concept TileOptimizationsDescriptor = requires(T t) {
    { t.num_groups_to_merge } -> std::convertible_to<int>;
    { t.split_image } -> std::convertible_to<bool>;
    { t.explicit_gemm } -> std::convertible_to<bool>;
};

// Base requirement for all ConvAlgorithm concepts, i.e., all conv algorithm concepts must meet this
// concept.
template <typename T>
concept ConvAlgorithmDescriptor = std::is_class_v<T>;

/******************************************** */
/* Requirements for the algorithm description */
/******************************************** */

// Concept to check if struct specifies thread block info.
template <typename T>
concept SpecifiesThreadBlock = requires {
    { T::thread_block } -> ThreadBlockDescriptor;
};

// Concept to check if struct specifies thread block info (CK Tile).
template <typename T>
concept SpecifiesTileThreadBlock = requires {
    { T::thread_block } -> TileThreadBlockDescriptor;
};

// Concept to check if a struct specifies gridwise XDL GEMM info.
template <typename T>
concept SpecifiesGridwiseXdlGemm = requires {
    { T::gridwise_gemm } -> GridwiseXdlGemmDescriptor;
};

// Concept to check if a struct specifies gridwise WMMA GEMM info.
template <typename T>
concept SpecifiesGridwiseWmmaGemm = requires {
    { T::gridwise_gemm } -> GridwiseWmmaGemmDescriptor;
};

// Concept to check if a struct specifies convolution input and output block transfer info.
template <typename T>
concept SpecifiesBlockTransfer = requires(T t) {
    { T::transfer.a.block_transfer } -> BlockTransferDescriptor;
    { T::transfer.b.block_transfer } -> BlockTransferDescriptor;
    { T::transfer.c.thread_cluster_dims } -> ThreadClusterDescriptor;
};

// Concept to check if a struct specifies convolution scalar per vector infor for A, B and C.
template <typename T>
concept SpecifiesTileTransfer = requires(T t) {
    { T::transfer.a_scalar_per_vector } -> std::convertible_to<size_t>;
    { T::transfer.b_scalar_per_vector } -> std::convertible_to<size_t>;
    { T::transfer.c_scalar_per_vector } -> std::convertible_to<size_t>;
};

// Concept to check if a struct specifies LDS transfer info for tensors A, B, and C.
template <typename T>
concept SpecifiesLdsTransfer = requires(T t) {
    { T::transfer.a.lds_transfer } -> LdsTransferDescriptor;
    { T::transfer.b.lds_transfer } -> LdsTransferDescriptor;
    { T::transfer.c.epilogue } -> EpilogueDescriptor;
};

// Concept to check if a struct specifies thread cluster access order info.
template <typename T>
concept SpecifiesThreadClusterAccessOrder = requires(T t) {
    { T::transfer.a.block_transfer_access_order } -> AccessOrderDescriptor;
    { T::transfer.b.block_transfer_access_order } -> AccessOrderDescriptor;
};

// Concept to check if a struct specifies source access order info.
template <typename T>
concept SpecifiesSourceAccessOrder = requires(T t) {
    { T::transfer.a.src_access_order } -> AccessOrderDescriptor;
    { T::transfer.b.src_access_order } -> AccessOrderDescriptor;
};

// Concept to check if struct specifies block GEMM.
template <typename T>
concept SpecifiesBlockGemm = requires {
    { T::block_gemm.pipeline_version } -> std::convertible_to<PipelineVersion>;
    { T::block_gemm.scheduler } -> std::convertible_to<PipelineScheduler>;
};

// Concept to check if struct specifies block GEMM (CK Tile).
template <typename T>
concept SpecifiesTileBlockGemm = requires {
    { T::block_gemm.warps.m } -> std::convertible_to<int>;
    { T::block_gemm.warps.n } -> std::convertible_to<int>;
    { T::block_gemm.warps.k } -> std::convertible_to<int>;
    { T::block_gemm.warp_tile.m } -> std::convertible_to<int>;
    { T::block_gemm.warp_tile.n } -> std::convertible_to<int>;
    { T::block_gemm.warp_tile.k } -> std::convertible_to<int>;
    { T::block_gemm.double_smem_buffer } -> std::convertible_to<bool>;
    { T::block_gemm.num_wave_groups } -> std::convertible_to<int>;
    { T::block_gemm.pipeline_version } -> std::convertible_to<PipelineVersion>;
    { T::block_gemm.scheduler } -> std::convertible_to<PipelineScheduler>;
};

// Concept to check if struct specifies block GEMM (CK Tile).
template <typename T>
concept SpecifiesTileOptimizations = requires {
    { T::optimizations.num_groups_to_merge } -> std::convertible_to<int>;
    { T::optimizations.split_image } -> std::convertible_to<bool>;
    { T::optimizations.explicit_gemm } -> std::convertible_to<bool>;
};

template <typename T>
concept SpecifiesTileConvSpecialization = requires {
    { T::specialization } -> std::convertible_to<TileConvSpecialization>;
};

template <typename T>
concept SpecifiesFwdConvSpecialization = requires {
    { T::fwd_specialization } -> std::convertible_to<ConvFwdSpecialization>;
};

template <typename T>
concept SpecifiesGemmSpecialization = requires {
    { T::gemm_specialization } -> std::convertible_to<GemmSpecialization>;
};

template <typename T>
concept SpecifiesNumPrefetchStages = requires {
    { T::num_gemm_k_prefetch_stages } -> std::convertible_to<size_t>;
};

template <typename T>
concept SpecifiesNumGroupsToMerge = requires {
    { T::num_groups_to_merge } -> std::convertible_to<size_t>;
};

template <typename T>
concept SpecifiesLoopScheduler = requires {
    { T::loop_scheduler } -> std::convertible_to<PipelineScheduler>;
};

template <typename T>
concept SpecifiesLargeTensorSupport = requires {
    { T::specialization } -> std::convertible_to<ConvAlgorithmSpecialization>;
    requires T::specialization == ConvAlgorithmSpecialization::LARGE_TENSOR;
};

/******************************************** */
/* DL-specific descriptors and requirements   */
/******************************************** */

// Concept for DL thread configuration
template <typename T>
concept DlThreadConfigDescriptor = requires(T t) {
    { t.k0_per_block } -> std::convertible_to<size_t>;
    { t.k1 } -> std::convertible_to<size_t>;
    { t.m1_per_thread } -> std::convertible_to<size_t>;
    { t.n1_per_thread } -> std::convertible_to<size_t>;
    { t.k_per_thread } -> std::convertible_to<size_t>;
};

// Concept for DL thread cluster
template <typename T>
concept DlThreadClusterDescriptor = requires(T t) {
    { t.m1_xs } -> std::convertible_to<std::array<size_t, 2>>;
    { t.n1_xs } -> std::convertible_to<std::array<size_t, 2>>;
};

// Concept for DL block transfer
template <typename T>
concept DlBlockTransferDescriptor = requires(T t) {
    { t.thread_slice_lengths } -> std::convertible_to<std::array<size_t, 4>>;
    { t.thread_cluster_lengths } -> std::convertible_to<std::array<size_t, 4>>;
    { t.thread_cluster_arrange_order } -> std::convertible_to<std::array<size_t, 4>>;
    { t.src_access_order } -> std::convertible_to<std::array<size_t, 4>>;
    { t.src_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, 4>>;
    { t.src_vector_tensor_contiguous_dim_order } -> std::convertible_to<std::array<size_t, 4>>;
    { t.dst_vector_tensor_lengths } -> std::convertible_to<std::array<size_t, 4>>;
};

// Concept for DL epilogue
template <typename T>
concept DlEpilogueDescriptor = requires(T t) {
    { t.src_dst_access_order } -> std::convertible_to<std::array<size_t, 6>>;
    { t.src_dst_vector_dim } -> std::convertible_to<size_t>;
    { t.dst_scalar_per_vector } -> std::convertible_to<size_t>;
};

// Concept to check if algorithm specifies DL thread config
template <typename T>
concept SpecifiesDlThreadConfig = requires {
    { T::thread_config } -> DlThreadConfigDescriptor;
};

// Concept to check if algorithm specifies DL thread cluster
template <typename T>
concept SpecifiesDlThreadCluster = requires {
    { T::thread_cluster } -> DlThreadClusterDescriptor;
};

// Concept to check if algorithm specifies DL block transfer
template <typename T>
concept SpecifiesDlBlockTransfer = requires {
    { T::transfer.a.block_transfer } -> DlBlockTransferDescriptor;
    { T::transfer.b.block_transfer } -> DlBlockTransferDescriptor;
};

// Concept to check if algorithm specifies DL C thread transfer
template <typename T>
concept SpecifiesDlEpilogue = requires {
    { T::transfer.c.epilogue } -> DlEpilogueDescriptor;
};

} // namespace ck_tile::builder
