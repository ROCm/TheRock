// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "gtest/gtest.h"
#include <array>

enum StreamKTilePartitionerBaseMethodId
{
    GET_LOCAL_ITER,
    GET_LOCAL_ITER_END,
    GET_TILE_BOUNDARIES,
    GET_TILE_INDEX,
    GET_ITER_BOUNDARIES,
    GET_OUTPUT_TILE_INDEX,
    GET_TILE_LOCAL_CTA_INDEX
};

// Base kernel wrapper class to facilitate testing class device functions.
template <typename T = ck_tile::index_t>
struct KernelWrapper
{
    static constexpr ck_tile::index_t kBlockSize = 1;

    struct KernelArgs
    {
        ck_tile::index_t arg1;
        ck_tile::index_t arg2;
        ck_tile::index_t arg3;
        void* result1;
        void* result2;
        T tile_partitioner;
    };

    CK_TILE_HOST static KernelArgs MakeKernelArgs(ck_tile::index_t arg1,
                                                  ck_tile::index_t arg2,
                                                  ck_tile::index_t arg3,
                                                  void* result1,
                                                  void* result2,
                                                  T tile_partitioner)
    {
        return KernelArgs{arg1, arg2, arg3, result1, result2, tile_partitioner};
    }
};

// Specialized derived class to support unique operator() functions. There is one template
// specialization per member in the StreamKTilePartitionerBaseMethodId enum.
template <typename TilePartitioner, StreamKTilePartitionerBaseMethodId Id>
struct KernelWrapperSpecialized;

template <typename TilePartitioner>
struct KernelWrapperSpecialized<TilePartitioner, StreamKTilePartitionerBaseMethodId::GET_LOCAL_ITER>
    : public KernelWrapper<>
{
    using Base = KernelWrapper<>;

    CK_TILE_DEVICE void operator()(Base::KernelArgs kargs)
    {
        *(static_cast<ck_tile::index_t*>(kargs.result1)) =
            TilePartitioner::get_local_iter(kargs.arg1, kargs.arg2);
    }
};

template <typename TilePartitioner>
struct KernelWrapperSpecialized<TilePartitioner,
                                StreamKTilePartitionerBaseMethodId::GET_TILE_BOUNDARIES>
    : public KernelWrapper<TilePartitioner>
{

    using Base = KernelWrapper<TilePartitioner>;

    CK_TILE_DEVICE void operator()(typename Base::KernelArgs kargs)
    {
        kargs.tile_partitioner.get_tile_boundaries(kargs.arg1, kargs.arg2, kargs.arg3);
        *(static_cast<ck_tile::index_t*>(kargs.result1)) = kargs.arg1;
        *(static_cast<ck_tile::index_t*>(kargs.result2)) = kargs.arg2;
    }
};

template <typename TilePartitioner>
struct KernelWrapperSpecialized<TilePartitioner,
                                StreamKTilePartitionerBaseMethodId::GET_ITER_BOUNDARIES>
    : public KernelWrapper<TilePartitioner>
{

    using Base = KernelWrapper<TilePartitioner>;

    CK_TILE_DEVICE void operator()(typename Base::KernelArgs kargs)
    {
        kargs.tile_partitioner.get_iter_boundaries(kargs.arg1, kargs.arg2, kargs.arg3);
        *(static_cast<ck_tile::index_t*>(kargs.result1)) = kargs.arg1;
        *(static_cast<ck_tile::index_t*>(kargs.result2)) = kargs.arg2;
    }
};

template <typename TilePartitioner>
struct KernelWrapperSpecialized<TilePartitioner,
                                StreamKTilePartitionerBaseMethodId::GET_LOCAL_ITER_END>
    : public KernelWrapper<>
{

    using Base = KernelWrapper<>;
    CK_TILE_DEVICE void operator()(Base::KernelArgs kargs)
    {
        *(static_cast<ck_tile::index_t*>(kargs.result1)) =
            TilePartitioner::get_local_iter_end(kargs.arg1, kargs.arg2, kargs.arg3);
    }
};

template <typename TilePartitioner>
struct KernelWrapperSpecialized<TilePartitioner, StreamKTilePartitionerBaseMethodId::GET_TILE_INDEX>
    : public KernelWrapper<TilePartitioner>
{

    using Base = KernelWrapper<TilePartitioner>;

    CK_TILE_DEVICE void operator()(typename Base::KernelArgs kargs)
    {
        *(static_cast<ck_tile::index_t*>(kargs.result1)) =
            kargs.tile_partitioner.get_tile_index(kargs.arg1);
    }
};

template <typename TilePartitioner>
struct KernelWrapperSpecialized<TilePartitioner,
                                StreamKTilePartitionerBaseMethodId::GET_OUTPUT_TILE_INDEX>
    : public KernelWrapper<TilePartitioner>
{

    using Base = KernelWrapper<TilePartitioner>;

    CK_TILE_DEVICE void operator()(typename Base::KernelArgs kargs)
    {
        auto [im, in] = kargs.tile_partitioner.get_output_tile_index(kargs.arg1);
        *(static_cast<ck_tile::index_t*>(kargs.result1)) = im;
        *(static_cast<ck_tile::index_t*>(kargs.result2)) = in;
    }
};

template <typename TilePartitioner>
struct KernelWrapperSpecialized<TilePartitioner,
                                StreamKTilePartitionerBaseMethodId::GET_TILE_LOCAL_CTA_INDEX>
    : public KernelWrapper<TilePartitioner>
{

    using Base = KernelWrapper<TilePartitioner>;

    CK_TILE_DEVICE void operator()(typename Base::KernelArgs kargs)
    {
        ck_tile::index_t tile_local_cta_index =
            kargs.tile_partitioner.get_tile_local_cta_index(kargs.arg1, kargs.arg2);
        *(static_cast<ck_tile::index_t*>(kargs.result1)) = tile_local_cta_index;
    }
};

struct StreamKTilePartitionerBaseExpected
{
    ck_tile::index_t sk_tiles_;
    ck_tile::index_t dp_tiles_;
    ck_tile::index_t sk_ctas_;
    ck_tile::index_t total_sk_iters_;
    ck_tile::index_t iters_per_sk_cta_;
    ck_tile::index_t iters_per_tile_;
    ck_tile::index_t extra_iters_;
    ck_tile::index_t total_dp_iters_;
    ck_tile::index_t num_tiles_;
    ck_tile::index_t grid_;
    ck_tile::index_t n_;
};

template <typename GemmShape>
void validate_streamk_base_constructor(
    StreamKTilePartitionerBaseExpected& expected_values,
    ck_tile::StreamKTilePartitionerBase<GemmShape>& tile_partitioner)
{
    EXPECT_EQ(tile_partitioner.get_sk_tiles(), expected_values.sk_tiles_);
    EXPECT_EQ(tile_partitioner.get_dp_tiles(), expected_values.dp_tiles_);
    EXPECT_EQ(tile_partitioner.get_sk_ctas(), expected_values.sk_ctas_);
    EXPECT_EQ(tile_partitioner.get_total_sk_iters(), expected_values.total_sk_iters_);
    EXPECT_EQ(tile_partitioner.get_iters_per_sk_cta(), expected_values.iters_per_sk_cta_);
    EXPECT_EQ(tile_partitioner.get_extra_iters(), expected_values.extra_iters_);
    EXPECT_EQ(tile_partitioner.get_iters_per_tile(), expected_values.iters_per_tile_);
    EXPECT_EQ(tile_partitioner.get_total_dp_iters(), expected_values.total_dp_iters_);
    EXPECT_EQ(tile_partitioner.get_num_tiles(), expected_values.num_tiles_);
    EXPECT_EQ(tile_partitioner.get_grid(), expected_values.grid_);
    EXPECT_EQ(tile_partitioner.get_n(), expected_values.n_);
}

struct StreamKTilePartitionerBaseConfig
{
    static constexpr ck_tile::index_t PLACEHOLDER = -1;
    static constexpr ck_tile::index_t UNUSED      = -1;
};

// Note: for the configs below, we only use BlockTiles in the TileGemmShape. We do not use
// BlockWarps or WarpTile.

struct StreamKTilePartitionerBaseConfigDP2TileSK : public StreamKTilePartitionerBaseConfig
{
    static constexpr ck_tile::index_t M    = 28;
    static constexpr ck_tile::index_t N    = 4;
    static constexpr ck_tile::index_t K    = 16;
    static constexpr ck_tile::index_t GRID = 3;

    static constexpr ck_tile::index_t M_TILE = 4;
    static constexpr ck_tile::index_t N_TILE = 4;
    static constexpr ck_tile::index_t K_TILE = 8;

    using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<M_TILE, N_TILE, K_TILE>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>>;
};

struct StreamKTilePartitionerBaseConfigSKOnlyWith2WgsPerSKTile
    : public StreamKTilePartitionerBaseConfig
{
    static constexpr ck_tile::index_t M    = 16;
    static constexpr ck_tile::index_t N    = 4;
    static constexpr ck_tile::index_t K    = 16;
    static constexpr ck_tile::index_t GRID = 8;

    static constexpr ck_tile::index_t M_TILE = 4;
    static constexpr ck_tile::index_t N_TILE = 4;
    static constexpr ck_tile::index_t K_TILE = 8;

    using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<M_TILE, N_TILE, K_TILE>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>>;
};

struct StreamKTilePartitionerBaseConfigDPOnly : public StreamKTilePartitionerBaseConfig
{
    static constexpr ck_tile::index_t M    = 12;
    static constexpr ck_tile::index_t N    = 4;
    static constexpr ck_tile::index_t K    = 16;
    static constexpr ck_tile::index_t GRID = 3;

    static constexpr ck_tile::index_t M_TILE = 4;
    static constexpr ck_tile::index_t N_TILE = 2;
    static constexpr ck_tile::index_t K_TILE = 8;

    using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<M_TILE, N_TILE, K_TILE>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>>;
};

struct StreamKTilePartitionerBaseConfigSKOnly : public StreamKTilePartitionerBaseConfig
{
    static constexpr ck_tile::index_t M    = 4;
    static constexpr ck_tile::index_t N    = 4;
    static constexpr ck_tile::index_t K    = 16;
    static constexpr ck_tile::index_t GRID = 3;

    static constexpr ck_tile::index_t M_TILE = 4;
    static constexpr ck_tile::index_t N_TILE = 2;
    static constexpr ck_tile::index_t K_TILE = 8;

    using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<M_TILE, N_TILE, K_TILE>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>>;
};

struct StreamKTilePartitionerBaseConfigSKOnlyLargeK : public StreamKTilePartitionerBaseConfig
{
    static constexpr ck_tile::index_t M    = 8;
    static constexpr ck_tile::index_t N    = 2;
    static constexpr ck_tile::index_t K    = 10;
    static constexpr ck_tile::index_t GRID = 5;

    static constexpr ck_tile::index_t M_TILE = 4;
    static constexpr ck_tile::index_t N_TILE = 2;
    static constexpr ck_tile::index_t K_TILE = 2;

    using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<M_TILE, N_TILE, K_TILE>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>>;
};

struct StreamKTilePartitionerBaseConfigEdgeCase : public StreamKTilePartitionerBaseConfig
{

    static constexpr ck_tile::index_t M    = 4;
    static constexpr ck_tile::index_t N    = 4;
    static constexpr ck_tile::index_t K    = 16;
    static constexpr ck_tile::index_t GRID = 4;

    static constexpr ck_tile::index_t M_TILE = 4;
    static constexpr ck_tile::index_t N_TILE = 4;
    static constexpr ck_tile::index_t K_TILE = 8;

    using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<M_TILE, N_TILE, K_TILE>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>>;
};

struct StreamKTilePartitionerBaseConfigLargerCTensor : public StreamKTilePartitionerBaseConfig
{
    // This config has 3 macro tiles in the M dimension and 4 macro tiles in the N dimension.
    // This facilitates testing the get_output_tile_index method.

    static constexpr ck_tile::index_t M    = 12;
    static constexpr ck_tile::index_t N    = 16;
    static constexpr ck_tile::index_t K    = 16;
    static constexpr ck_tile::index_t GRID = 4;

    static constexpr ck_tile::index_t M_TILE = 4;
    static constexpr ck_tile::index_t N_TILE = 4;
    static constexpr ck_tile::index_t K_TILE = 8;

    using GemmShape = ck_tile::TileGemmShape<ck_tile::sequence<M_TILE, N_TILE, K_TILE>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>,
                                             ck_tile::sequence<UNUSED, UNUSED, UNUSED>>;
};

void test_get_output_tile_index(ck_tile::index_t tile_idx,
                                ck_tile::tuple<ck_tile::index_t, ck_tile::index_t> expected_2d_idx)
{
    // Types
    using Config          = StreamKTilePartitionerBaseConfigLargerCTensor;
    using TilePartitioner = ck_tile::StreamKTilePartitionerBase<Config::GemmShape>;
    using Kernel =
        KernelWrapperSpecialized<TilePartitioner,
                                 StreamKTilePartitionerBaseMethodId::GET_OUTPUT_TILE_INDEX>;

    // Test parameters
    ck_tile::StreamKTilePartitionerBase<Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::GRID};
    ck_tile::DeviceMem im_dev(sizeof(ck_tile::index_t));
    ck_tile::DeviceMem in_dev(sizeof(ck_tile::index_t));

    // Launch kernel
    auto kargs = Kernel::MakeKernelArgs(tile_idx,
                                        Config::UNUSED,
                                        Config::UNUSED,
                                        im_dev.GetDeviceBuffer(),
                                        in_dev.GetDeviceBuffer(),
                                        tile_partitioner);
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, 1, 1, 0, kargs));

    // Validate results
    const auto [im_expected, in_expected] = expected_2d_idx;
    ck_tile::index_t im, in;
    im_dev.FromDevice(&im);
    in_dev.FromDevice(&in);
    EXPECT_EQ(im, im_expected);
    EXPECT_EQ(in, in_expected);
};

template <typename Config>
void test_get_tile_local_cta_idx(ck_tile::index_t tile_iter_start,
                                 ck_tile::index_t cta_idx,
                                 ck_tile::index_t expected_tile_local_cta_idx)
{
    // Types
    using TilePartitioner = ck_tile::StreamKTilePartitionerBase<typename Config::GemmShape>;
    using Kernel =
        KernelWrapperSpecialized<TilePartitioner,
                                 StreamKTilePartitionerBaseMethodId::GET_TILE_LOCAL_CTA_INDEX>;

    // Test parameters
    ck_tile::StreamKTilePartitionerBase<typename Config::GemmShape> tile_partitioner{
        Config::M, Config::N, Config::K, Config::GRID};
    ck_tile::DeviceMem tile_local_cta_idx_dev(sizeof(ck_tile::index_t));

    // Launch kernel
    auto kargs = Kernel::MakeKernelArgs(tile_iter_start,
                                        cta_idx,
                                        Config::UNUSED,
                                        tile_local_cta_idx_dev.GetDeviceBuffer(),
                                        nullptr,
                                        tile_partitioner);
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, 1, 1, 0, kargs));

    // Validate results
    ck_tile::index_t tile_local_cta_idx;
    tile_local_cta_idx_dev.FromDevice(&tile_local_cta_idx);
    EXPECT_EQ(tile_local_cta_idx, expected_tile_local_cta_idx);
}

// Configs for TilePartitioner Child structs
struct StreamKTilePartitionerV2PersistentExpected
{
    ck_tile::index_t dp_tiles_per_cta_;
    ck_tile::index_t extra_dp_tiles_;
    ck_tile::index_t grid_;
};

struct StreamKTilePartitionerV2NonPersistentExpected
{
    ck_tile::index_t dp_ctas_;
    ck_tile::index_t dp_start_block_idx_;
    ck_tile::index_t sk_start_block_idx_;
    ck_tile::index_t grid_;
};

// Persistent
template <typename GemmShape>
void validate_streamk_persistent(
    StreamKTilePartitionerV2PersistentExpected& expected_values,
    ck_tile::StreamKTilePartitioner<GemmShape, ck_tile::StreamKReductionStrategy::Atomic, true>&
        tile_partitioner)
{
    EXPECT_EQ(tile_partitioner.get_dp_tiles_per_cta(), expected_values.dp_tiles_per_cta_);
    EXPECT_EQ(tile_partitioner.get_extra_dp_tiles(), expected_values.extra_dp_tiles_);
    EXPECT_EQ(tile_partitioner.get_grid(), expected_values.grid_);
}

// Non-Persistent
template <typename GemmShape>
void validate_streamk_nonpersistent(
    StreamKTilePartitionerV2NonPersistentExpected& expected_values,
    ck_tile::StreamKTilePartitioner<GemmShape, ck_tile::StreamKReductionStrategy::Atomic, false>&
        tile_partitioner)
{
    EXPECT_EQ(tile_partitioner.get_dp_ctas(), expected_values.dp_ctas_);
    EXPECT_EQ(tile_partitioner.get_dp_start_block_idx(), expected_values.dp_start_block_idx_);
    EXPECT_EQ(tile_partitioner.get_sk_start_block_idx(), expected_values.sk_start_block_idx_);
    EXPECT_EQ(tile_partitioner.get_grid(), expected_values.grid_);
}
