// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::test {

namespace ckb = ck_tile::builder;

// Convenience struct for a tuple of m, n, and k values.
template <typename T>
struct MNK
{
    T m{};
    T n{};
    T k{};
};

// Specify thread block dimensions for a GEMM.
struct ThreadBlock
{
    // Thread block size.
    size_t block_size;
    // Size of the submatrix problem in a thread block.
    MNK<size_t> tile_size;
};
static_assert(ckb::ThreadBlockDescriptor<ThreadBlock>);

// Describe gridwise XDL GEMM parameters.
struct GridwiseXdlGemm
{
    // NOTE: ak1 and bk1 are difficult to verify in the kernel instantiation!!!
    size_t ak1            = 0;
    size_t bk1            = 0;
    size_t m_per_xdl      = 0;
    size_t n_per_xdl      = 0;
    size_t m_xdl_per_wave = 0;
    size_t n_xdl_per_wave = 0;
};
static_assert(ckb::GridwiseXdlGemmDescriptor<GridwiseXdlGemm>);

// Describe gridwise WMMA GEMM parameters.
struct GridwiseWmmaGemm
{
    size_t k1              = 0;
    size_t m_per_wmma      = 0;
    size_t n_per_wmma      = 0;
    size_t m_wmma_per_wave = 0;
    size_t n_wmma_per_wave = 0;
    PipelineVersion pipeline_version;
};
static_assert(ckb::GridwiseWmmaGemmDescriptor<GridwiseWmmaGemm>);

struct BlockGemm
{
    PipelineVersion pipeline_version;
    PipelineScheduler scheduler;
};
static_assert(ckb::BlockGemmDescriptor<BlockGemm>);

// Describe Aand B block transfer thread cluster lengths.
struct BlockTransfer
{
    size_t k0;
    size_t m_n;
    size_t k1;
};
static_assert(ckb::BlockTransferDescriptor<BlockTransfer>);

// Describe C block transfer thread cluster lengths.
struct ThreadCluster
{
    size_t m_block;
    size_t m_wave_per_xdl;
    size_t n_block;
    size_t n_wave_per_xdl;
};
static_assert(ThreadClusterDescriptor<ThreadCluster>);

struct LdsTransfer
{
    size_t src_vector_dim;
    size_t src_scalar_per_vector;
    size_t lds_dst_scalar_per_vector;
    bool is_direct_load;
    bool lds_padding;
};
static_assert(LdsTransferDescriptor<LdsTransfer>);

struct Epilogue
{
    size_t m_xdl_per_wave_per_shuffle;
    size_t n_per_wave_per_shuffle;
    size_t scalar_per_vector;
};
static_assert(EpilogueDescriptor<Epilogue>);

struct AccessOrder
{
    std::array<size_t, 3> order;
};
static_assert(AccessOrderDescriptor<AccessOrder>);

struct TransferAB
{
    BlockTransfer block_transfer;
    LdsTransfer lds_transfer;
    AccessOrder block_transfer_access_order;
    AccessOrder src_access_order;
};

struct TransferC
{
    ThreadCluster thread_cluster_dims;
    Epilogue epilogue;
};

struct TransferABC
{
    TransferAB a;
    TransferAB b;
    TransferC c;
};

// DL-specific descriptors
struct DlThreadConfig
{
    size_t k0_per_block;
    size_t k1;
    size_t m1_per_thread;
    size_t n1_per_thread;
    size_t k_per_thread;
};
static_assert(ckb::DlThreadConfigDescriptor<DlThreadConfig>);

struct DlThreadCluster
{
    std::array<size_t, 2> m1_xs;
    std::array<size_t, 2> n1_xs;
};
static_assert(ckb::DlThreadClusterDescriptor<DlThreadCluster>);

struct DlBlockTransfer
{
    std::array<size_t, 4> thread_slice_lengths;
    std::array<size_t, 4> thread_cluster_lengths;
    std::array<size_t, 4> thread_cluster_arrange_order;
    std::array<size_t, 4> src_access_order;
    std::array<size_t, 4> src_vector_tensor_lengths;
    std::array<size_t, 4> src_vector_tensor_contiguous_dim_order;
    std::array<size_t, 4> dst_vector_tensor_lengths;
};
static_assert(ckb::DlBlockTransferDescriptor<DlBlockTransfer>);

struct DlEpilogue
{
    std::array<size_t, 6> src_dst_access_order;
    size_t src_dst_vector_dim;
    size_t dst_scalar_per_vector;
};
static_assert(ckb::DlEpilogueDescriptor<DlEpilogue>);

// Factory types

struct ThreadBlock_
{
    ThreadBlock thread_block;
};

struct XdlGemm_
{
    GridwiseXdlGemm gridwise_gemm;
};

struct WmmaGemm_
{
    GridwiseWmmaGemm gridwise_gemm;
};

struct Transfer_
{
    TransferABC transfer;
};

struct ConvSpecialization_
{
    ConvFwdSpecialization fwd_specialization;
    GemmSpecialization gemm_specialization;
};

struct Prefetch_
{
    size_t num_gemm_k_prefetch_stages;
    size_t num_groups_to_merge;
    PipelineScheduler loop_scheduler;
};

struct BlockGemm_
{
    BlockGemm block_gemm;
};

struct DlThreadConfig_
{
    DlThreadConfig thread_config;
};

struct DlThreadCluster_
{
    DlThreadCluster thread_cluster;
};

struct DlBlockTransferAB
{
    DlBlockTransfer block_transfer;
};

struct DlBlockTransferC
{
    DlEpilogue epilogue;
};

struct DlTransferABC
{
    DlBlockTransferAB a;
    DlBlockTransferAB b;
    DlBlockTransferC c;
};

struct DlTransfer_
{
    DlTransferABC transfer;
};

// Specialization wrapper for large tensor support
template <typename BaseAlgorithm>
struct LargeTensorWrapper
{
    BaseAlgorithm base_algorithm;
    static constexpr ConvAlgorithmSpecialization specialization =
        ConvAlgorithmSpecialization::LARGE_TENSOR;
};

// Specify thread block dimensions for a GEMM (CK Tile).
struct TileThreadBlock
{
    // Size of the submatrix problem in a thread block.
    MNK<size_t> tile_size;
};
static_assert(ckb::TileThreadBlockDescriptor<TileThreadBlock>);

struct TileTransfer
{
    size_t a_scalar_per_vector;
    size_t b_scalar_per_vector;
    size_t c_scalar_per_vector;
};
static_assert(ckb::TileTransferDescriptor<TileTransfer>);

struct TileBlockGemm
{
    // Number of warps per each dimension.
    MNK<int> warps;
    // Number of data processed per each dimension for each XDL/WMMA instruction.
    MNK<int> warp_tile;
    // Double LDS buffer.
    bool double_smem_buffer;
    // Waves grouping (Ping-Pong scheduler).
    int num_wave_groups;
    PipelineVersion pipeline_version;
    PipelineScheduler scheduler;
};
static_assert(ckb::TileBlockGemmDescriptor<TileBlockGemm>);

struct TileOptimizations
{
    // Number of convolution groups processed per one workgroup
    int num_groups_to_merge;
    // Split image for large tensors
    bool split_image;
    // Explicit gemm for 1x1, stride=0, pad=0 cases
    bool explicit_gemm;
};
static_assert(ckb::TileOptimizationsDescriptor<TileOptimizations>);

struct TileConvSpecialization_
{
    TileConvSpecialization specialization;
};

struct TileThreadBlock_
{
    TileThreadBlock thread_block;
};

struct TileTransfer_
{
    TileTransfer transfer;
};

struct TileBlockGemm_
{
    TileBlockGemm block_gemm;
};

struct TileOptimizations_
{
    TileOptimizations optimizations;
};

// Factory

template <typename... Components>
struct ConvAlgorithmTemplate : Components...
{

    template <typename TB>
    constexpr auto with_thread_block(const TB& tb) const
    {
        static_assert(std::is_base_of_v<ThreadBlock_, ConvAlgorithmTemplate>);
        auto result         = *this;
        result.thread_block = tb;
        return result;
    }

    template <typename GemmConfig>
    constexpr auto with_gemm_config(const GemmConfig& gemm) const
    {
        auto result = *this;
        if constexpr(std::is_base_of_v<XdlGemm_, ConvAlgorithmTemplate>)
        {
            result.gridwise_gemm = gemm;
        }
        else if constexpr(std::is_base_of_v<WmmaGemm_, ConvAlgorithmTemplate>)
        {
            result.gridwise_gemm = gemm;
        }
        return result;
    }

    template <typename T>
    constexpr auto with_transfer(const T& t) const
    {
        static_assert(std::is_base_of_v<Transfer_, ConvAlgorithmTemplate>);
        auto result     = *this;
        result.transfer = t;
        return result;
    }

    constexpr auto with_specializations(ConvFwdSpecialization fwd_spec,
                                        GemmSpecialization gemm_spec) const
    {
        static_assert(std::is_base_of_v<ConvSpecialization_, ConvAlgorithmTemplate>);
        auto result                = *this;
        result.fwd_specialization  = fwd_spec;
        result.gemm_specialization = gemm_spec;
        return result;
    }

    constexpr auto with_prefetch_config(size_t k_prefetch_stages,
                                        size_t groups_to_merge,
                                        PipelineScheduler scheduler) const
    {
        static_assert(std::is_base_of_v<Prefetch_, ConvAlgorithmTemplate>);
        auto result                       = *this;
        result.num_gemm_k_prefetch_stages = k_prefetch_stages;
        result.num_groups_to_merge        = groups_to_merge;
        result.loop_scheduler             = scheduler;
        return result;
    }

    template <typename BG>
    constexpr auto with_block_gemm(const BG& bg) const
    {
        static_assert(std::is_base_of_v<BlockGemm_, ConvAlgorithmTemplate>);
        auto result       = *this;
        result.block_gemm = bg;
        return result;
    }

    template <typename TC>
    constexpr auto with_dl_thread_config(const TC& tc) const
    {
        static_assert(std::is_base_of_v<DlThreadConfig_, ConvAlgorithmTemplate>);
        auto result          = *this;
        result.thread_config = tc;
        return result;
    }

    template <typename TCl>
    constexpr auto with_dl_thread_cluster(const TCl& tcl) const
    {
        static_assert(std::is_base_of_v<DlThreadCluster_, ConvAlgorithmTemplate>);
        auto result           = *this;
        result.thread_cluster = tcl;
        return result;
    }

    template <typename T>
    constexpr auto with_dl_transfer(const T& t) const
    {
        static_assert(std::is_base_of_v<DlTransfer_, ConvAlgorithmTemplate>);
        auto result     = *this;
        result.transfer = t;
        return result;
    }

    template <typename S>
    constexpr auto with_tile_specializations(const S& s) const
    {
        static_assert(std::is_base_of_v<TileConvSpecialization_, ConvAlgorithmTemplate>);
        auto result           = *this;
        result.specialization = s;
        return result;
    }

    template <typename TB>
    constexpr auto with_tile_thread_block(const TB& tb) const
    {
        static_assert(std::is_base_of_v<TileThreadBlock_, ConvAlgorithmTemplate>);
        auto result         = *this;
        result.thread_block = tb;
        return result;
    }

    template <typename BG>
    constexpr auto with_tile_block_gemm(const BG& bg) const
    {
        static_assert(std::is_base_of_v<TileBlockGemm_, ConvAlgorithmTemplate>);
        auto result       = *this;
        result.block_gemm = bg;
        return result;
    }

    template <typename T>
    constexpr auto with_tile_transfer(const T& t) const
    {
        static_assert(std::is_base_of_v<TileTransfer_, ConvAlgorithmTemplate>);
        auto result     = *this;
        result.transfer = t;
        return result;
    }

    template <typename O>
    constexpr auto with_tile_optimizations(const O& o) const
    {
        static_assert(std::is_base_of_v<TileOptimizations_, ConvAlgorithmTemplate>);
        auto result          = *this;
        result.optimizations = o;
        return result;
    }
};

// Algorithm types

using ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_, XdlGemm_, Transfer_, ConvSpecialization_, Prefetch_>;

using ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 =
    ConvAlgorithmTemplate<ThreadBlock_, XdlGemm_, Transfer_, ConvSpecialization_, BlockGemm_>;

using ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle =
    ConvAlgorithmTemplate<ThreadBlock_, WmmaGemm_, Transfer_, ConvSpecialization_, Prefetch_>;
using ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK =
    ConvAlgorithmTemplate<ThreadBlock_,
                          ConvSpecialization_,
                          DlThreadConfig_,
                          DlThreadCluster_,
                          DlTransfer_>;

using ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor =
    LargeTensorWrapper<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>;

using ConvAlgorithm_Tile_GroupedConvolutionKernel = ConvAlgorithmTemplate<TileThreadBlock_,
                                                                          TileBlockGemm_,
                                                                          TileTransfer_,
                                                                          TileConvSpecialization_,
                                                                          TileOptimizations_>;

// Reference algorithm descriptor - for GPU reference validation
// This is a simple algorithm that requires no complex configuration,
// just a specialization marker to identify it as a reference implementation.
struct ConvAlgorithm_Reference
{
    static constexpr auto specialization = ckb::ConvAlgorithmSpecialization::REFERENCE;
    // GPU reference uses simple algorithm, no tile configuration needed
};

} // namespace ck_tile::builder::test
