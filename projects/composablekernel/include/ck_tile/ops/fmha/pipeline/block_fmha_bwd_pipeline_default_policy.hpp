// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"

namespace ck_tile {

struct BlockFmhaBwdPipelineDefaultPolicy
{
    template <index_t ndim>
    static constexpr auto swap_last2 = generate_sequence_v2(
        [](auto i) {
            return number < i == ndim - 2 ? ndim - 1 : i == ndim - 1 ? ndim - 2 : i > {};
        },
        number<ndim>{});

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::AccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kK0>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        using WarpGemm = WarpGemmDispatcher<
            typename Problem::QDataType,
            typename Problem::KDataType,
            typename Problem::AccDataType,
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
            false,
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}) == 16 ? false : true>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                typename Problem::KDataType,
                                                typename Problem::AccDataType,
                                                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                WarpGemm>;

        return BlockGemmARegBRegCRegV1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetPTOGradTBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::GemmDataType,
                             typename Problem::OGradDataType,
                             typename Problem::AccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kVHeaddim,
                                                    Problem::BlockFmhaShape::kK1>,
                                           typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        using WarpGemm =
            WarpGemmDispatcher<typename Problem::GemmDataType,
                               typename Problem::OGradDataType,
                               typename Problem::AccDataType,
                               Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                               Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                               Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                               true,
                               false, // SwizzleAccess
                               false, // UseStructuredSparsity
                               (Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}) == 32)
                                   ? WGAttrNumAccessEnum ::Double
                                   : WGAttrNumAccessEnum ::Single>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::GemmDataType,
                                                typename Problem::OGradDataType,
                                                typename Problem::AccDataType,
                                                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                WarpGemm>;

        return BlockGemmARegBRegCRegV1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetOGradVBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::OGradDataType,
                             typename Problem::VDataType,
                             typename Problem::AccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kK2>,
                                           typename Problem::BlockFmhaShape::Gemm2BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm2WarpTile>>;

        using WarpGemm = WarpGemmDispatcher<
            typename Problem::OGradDataType,
            typename Problem::VDataType,
            typename Problem::AccDataType,
            Problem::BlockFmhaShape::Gemm2WarpTile::at(number<0>{}),
            Problem::BlockFmhaShape::Gemm2WarpTile::at(number<1>{}),
            Problem::BlockFmhaShape::Gemm2WarpTile::at(number<2>{}),
            false,
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}) == 16 ? false : true>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::OGradDataType,
                                                typename Problem::VDataType,
                                                typename Problem::AccDataType,
                                                typename Problem::BlockFmhaShape::Gemm2BlockWarps,
                                                WarpGemm>;

        return BlockGemmARegBRegCRegV1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSGradTQTBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::GemmDataType,
                             typename Problem::QDataType,
                             typename Problem::AccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kQKHeaddim,
                                                    Problem::BlockFmhaShape::kK3>,
                                           typename Problem::BlockFmhaShape::Gemm3BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm3WarpTile>>;

        using WarpGemm =
            WarpGemmDispatcher<typename Problem::GemmDataType,
                               typename Problem::QDataType,
                               typename Problem::AccDataType,
                               Problem::BlockFmhaShape::Gemm3WarpTile::at(number<0>{}),
                               Problem::BlockFmhaShape::Gemm3WarpTile::at(number<1>{}),
                               Problem::BlockFmhaShape::Gemm3WarpTile::at(number<2>{}),
                               true,
                               false, // SwizzleAccess
                               false, // UseStructuredSparsity
                               (Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}) == 32)
                                   ? WGAttrNumAccessEnum ::Double
                                   : WGAttrNumAccessEnum ::Single>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::GemmDataType,
                                                typename Problem::QDataType,
                                                typename Problem::AccDataType,
                                                typename Problem::BlockFmhaShape::Gemm3BlockWarps,
                                                WarpGemm>;

        return BlockGemmARegBRegCRegV1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSGradKTBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::GemmDataType,
                             typename Problem::KDataType,
                             typename Problem::AccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kQKHeaddim,
                                                    Problem::BlockFmhaShape::kK4>,
                                           typename Problem::BlockFmhaShape::Gemm4BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm4WarpTile>>;

        using WarpGemm = WarpGemmDispatcher<typename Problem::GemmDataType,
                                            typename Problem::KDataType,
                                            typename Problem::AccDataType,
                                            Problem::BlockFmhaShape::Gemm4WarpTile::at(number<0>{}),
                                            Problem::BlockFmhaShape::Gemm4WarpTile::at(number<1>{}),
                                            Problem::BlockFmhaShape::Gemm4WarpTile::at(number<2>{}),
                                            false>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::GemmDataType,
                                                typename Problem::KDataType,
                                                typename Problem::AccDataType,
                                                typename Problem::BlockFmhaShape::Gemm4BlockWarps,
                                                WarpGemm>;

        return BlockGemmARegBRegCRegV1<GemmProblem, BlockGemmPolicy>{};
    }

    // these are for global load
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        using QDataType               = remove_cvref_t<typename Problem::QDataType>;
        constexpr index_t kBlockSize  = Problem::kBlockSize;
        constexpr index_t kMNPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock  = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kMaxVecLoad = 16 / sizeof(QDataType);
        constexpr index_t kMinVecLoad = 4 / sizeof(QDataType);

        constexpr index_t total_pixels = kMNPerBlock * kKPerBlock / kBlockSize;

        constexpr index_t kVecLoad = ((total_pixels / kMaxVecLoad) >= kMinVecLoad)
                                         ? kMaxVecLoad
                                         : (total_pixels / kMinVecLoad);

        return kVecLoad;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        using KDataType               = remove_cvref_t<typename Problem::KDataType>;
        constexpr index_t kBlockSize  = Problem::kBlockSize;
        constexpr index_t kMNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock  = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kMaxVecLoad = 16 / sizeof(KDataType);
        constexpr index_t kMinVecLoad = 4 / sizeof(KDataType);

        constexpr index_t total_pixels = kMNPerBlock * kKPerBlock / kBlockSize;

        constexpr index_t kVecLoad = ((total_pixels / kMaxVecLoad) >= kMinVecLoad)
                                         ? kMaxVecLoad
                                         : (total_pixels / kMinVecLoad);

        return kVecLoad;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        using VDataType                = remove_cvref_t<typename Problem::VDataType>;
        constexpr index_t kBlockSize   = Problem::kBlockSize;
        constexpr index_t kMNPerBlock  = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock   = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kMaxVecLoad  = 16 / sizeof(VDataType);
        constexpr index_t total_pixels = kMNPerBlock * kKPerBlock / kBlockSize;

        return total_pixels > kMaxVecLoad ? kMaxVecLoad : total_pixels;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        using ODataType = remove_cvref_t<typename Problem::ODataType>;
        return 16 / sizeof(ODataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOGrad()
    {
        using OGradDataType           = remove_cvref_t<typename Problem::OGradDataType>;
        constexpr index_t kBlockSize  = Problem::kBlockSize;
        constexpr index_t kMNPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock  = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kMaxVecLoad = 16 / sizeof(OGradDataType);
        constexpr index_t kMinVecLoad = 4 / sizeof(OGradDataType);

        constexpr index_t total_pixels = kMNPerBlock * kKPerBlock / kBlockSize;

        constexpr index_t kVecLoad = ((total_pixels / kMaxVecLoad) >= kMinVecLoad)
                                         ? kMaxVecLoad
                                         : (total_pixels / kMinVecLoad);

        return kVecLoad;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentBias()
    {
        using BiasDataType            = remove_cvref_t<typename Problem::BiasDataType>;
        constexpr index_t kBlockSize  = Problem::kBlockSize;
        constexpr index_t kMPerBlock  = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock  = Problem::BlockFmhaShape::kN0;
        constexpr index_t kMaxVecLoad = 16 / sizeof(BiasDataType);
        constexpr index_t kMinVecLoad = 4 / sizeof(BiasDataType);

        constexpr index_t total_pixels = kMPerBlock * kNPerBlock / kBlockSize;

        constexpr index_t kVecLoad = ((total_pixels / kMaxVecLoad) >= kMinVecLoad)
                                         ? kMaxVecLoad
                                         : (total_pixels / kMinVecLoad);

        return kVecLoad;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentKGrad()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        using CWarpDstr       = typename WG::CWarpDstr;
        constexpr auto vec =
            CWarpDstr{}.get_ys_to_d_descriptor().get_lengths().at(number<CWarpDstr::NDimY - 1>{});
        return vec;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentVGrad()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template at<0>())>;
        using CWarpDstr       = typename WG::CWarpDstr;
        constexpr auto vec =
            CWarpDstr{}.get_ys_to_d_descriptor().get_lengths().at(number<CWarpDstr::NDimY - 1>{});
        return vec;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTransposedAlignmentQ()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;

        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;

        return total_pixels / GetAlignmentQ<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTransposedAlignmentK()
    {
        constexpr index_t kBlockSize   = Problem::kBlockSize;
        constexpr index_t kNPerBlock   = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock   = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;

        return total_pixels / GetAlignmentK<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTransposedAlignmentOGrad()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;

        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;

        return total_pixels / GetAlignmentOGrad<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTransposedAlignmentBias()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t total_pixels = kMPerBlock * kNPerBlock / kBlockSize;

        return total_pixels / GetAlignmentBias<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentPostQGradAcc()
    {
        using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
        return 16 / sizeof(AccDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentPostQGrad()
    {
        return GetAlignmentPostQGradAcc<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;

        constexpr index_t K1 = GetAlignmentK<Problem>();
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N1 = get_warp_size() / K0;
        constexpr index_t N0 = kBlockSize / get_warp_size();
        constexpr index_t N2 = kNPerBlock / (N1 * N0);

        constexpr auto dstr = make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<2, 1>>{});

        if constexpr((kKPerBlock & (kKPerBlock - 1)) == 0) // kKPerBlock is power of 2
        {
            return dstr;
        }
        else
        {
            constexpr index_t kKPerIter = 32;
            static_assert(kKPerBlock % kKPerIter == 0);
            constexpr index_t K0_m = kKPerBlock / kKPerIter;
            constexpr index_t K2   = 2;
            constexpr index_t K1_m = kKPerIter / K2;
            constexpr index_t N1_m = get_warp_size() / K1_m;
            constexpr index_t N2_m = kNPerBlock / (N1_m * N0);
            constexpr auto dstr_m  = make_static_tile_distribution(
                tile_distribution_encoding<
                     sequence<>,
                     tuple<sequence<N0, N1_m, N2_m>, sequence<K0_m, K1_m, K2>>,
                     tuple<sequence<1>, sequence<1, 2>>, // N0, N1 K1
                     tuple<sequence<0>, sequence<1, 1>>,
                     sequence<2, 1, 2>, // K0 N2 K2
                     sequence<0, 2, 2>>{});
            static_assert(container_reduce(dstr_m.get_lengths(), std::multiplies<index_t>{}, 1) ==
                          kNPerBlock * kKPerBlock);
            return dstr_m;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;

        constexpr index_t K1 = GetAlignmentV<Problem>();
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        constexpr auto dstr = make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>, // N1, N2 K0
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>, // N0 K1
                                       sequence<0, 1>>{});
        if constexpr((kKPerBlock & (kKPerBlock - 1)) == 0) // kKPerBlock is power of 2
        {
            return dstr;
        }
        else
        {
            constexpr index_t kKPerIter = 32;
            static_assert(kKPerBlock % kKPerIter == 0);
            constexpr index_t K0_m = kKPerBlock / kKPerIter;
            constexpr index_t K2   = 2;
            constexpr index_t K1_m = kKPerIter / K2;
            constexpr index_t N2_m = get_warp_size() / K1_m;
            constexpr index_t N0_m = kNPerBlock / (N2_m * N1);
            constexpr auto dstr_m  = make_static_tile_distribution(
                tile_distribution_encoding<
                     sequence<>,
                     tuple<sequence<N0_m, N1, N2_m>, sequence<K0_m, K1_m, K2>>,
                     tuple<sequence<1>, sequence<1, 2>>, // N1, N2 K1
                     tuple<sequence<1>, sequence<2, 1>>,
                     sequence<2, 1, 2>, // K0 N0 K2
                     sequence<0, 0, 2>>{});
            static_assert(container_reduce(dstr_m.get_lengths(), std::multiplies<index_t>{}, 1) ==
                          kNPerBlock * kKPerBlock);
            return dstr_m;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;

        constexpr index_t K1 = GetAlignmentQ<Problem>();
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M1 = get_warp_size() / K0;
        constexpr index_t M0 = kBlockSize / get_warp_size();
        constexpr index_t M2 = kMPerBlock / (M1 * M0);

        constexpr auto dstr = make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<2, 1>>{});

        if constexpr((kKPerBlock & (kKPerBlock - 1)) == 0) // kKPerBlock is power of 2
        {
            return dstr;
        }
        else
        {
            // something not divisible, try a more flexible distribution
            constexpr index_t kKPerIter = 32;
            static_assert(kKPerBlock % kKPerIter == 0);
            constexpr index_t K0_m = kKPerBlock / kKPerIter;
            constexpr index_t K2   = 2;
            constexpr index_t K1_m = kKPerIter / K2;
            constexpr index_t M1_m = get_warp_size() / K1_m;
            constexpr index_t M2_m = kMPerBlock / (M1_m * M0);
            constexpr auto dstr_m  = make_static_tile_distribution(
                tile_distribution_encoding<
                     sequence<>,
                     tuple<sequence<M0, M1_m, M2_m>, sequence<K0_m, K1_m, K2>>,
                     tuple<sequence<1>, sequence<1, 2>>, // M0, M1 K1
                     tuple<sequence<0>, sequence<1, 1>>,
                     sequence<2, 1, 2>, // K0 M2 K2
                     sequence<0, 2, 2>>{});
            static_assert(container_reduce(dstr_m.get_lengths(), std::multiplies<index_t>{}, 1) ==
                          kMPerBlock * kKPerBlock);
            return dstr_m;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;

        constexpr index_t K1 = GetAlignmentOGrad<Problem>();
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M1 = get_warp_size() / K0;
        constexpr index_t M0 = kBlockSize / get_warp_size();
        constexpr index_t M2 = kMPerBlock / (M1 * M0);

        constexpr auto dstr = make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<2, 1>>{});

        if constexpr((kKPerBlock & (kKPerBlock - 1)) == 0) // kKPerBlock is power of 2
        {
            return dstr;
        }
        else
        {
            // something not divisible, try a more flexible distribution
            constexpr index_t kKPerIter = 32;
            static_assert(kKPerBlock % kKPerIter == 0);
            constexpr index_t K0_m = kKPerBlock / kKPerIter;
            constexpr index_t K2   = 2;
            constexpr index_t K1_m = kKPerIter / K2;
            constexpr index_t M1_m = get_warp_size() / K1_m;
            constexpr index_t M2_m = kMPerBlock / (M1_m * M0);
            constexpr auto dstr_m  = make_static_tile_distribution(
                tile_distribution_encoding<
                     sequence<>,
                     tuple<sequence<M0, M1_m, M2_m>, sequence<K0_m, K1_m, K2>>,
                     tuple<sequence<1>, sequence<1, 2>>, // M0, M1 K1
                     tuple<sequence<0>, sequence<1, 1>>,
                     sequence<2, 1, 2>, // K0 M2 K2
                     sequence<0, 2, 2>>{});
            static_assert(container_reduce(dstr_m.get_lengths(), std::multiplies<index_t>{}, 1) ==
                          kMPerBlock * kKPerBlock);
            return dstr_m;
        }
    }

    template <typename Problem, typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEDDramTileDistribution()
    {
        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;

        // Duplicate dimension
        constexpr index_t N0 = NWarp;
        constexpr index_t N1 =
            (get_warp_size() / kMPerBlock) > 1 ? (get_warp_size() / kMPerBlock) : 1;

        constexpr index_t M0 = MWarp;
        constexpr index_t M1 = (get_warp_size() / kMPerBlock) > 1 ? kMPerBlock : get_warp_size();
        constexpr index_t M2 =
            (get_warp_size() / kMPerBlock) > 1 ? 1 : (kMPerBlock / get_warp_size());

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<N0, N1>,
                                       tuple<sequence<M0, M1, M2>>,
                                       tuple<sequence<0, 1>, sequence<0, 1>>,
                                       tuple<sequence<0, 0>, sequence<1, 1>>,
                                       sequence<1>,
                                       sequence<2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t N1 = GetAlignmentBias<Problem>();
        constexpr index_t N0 = kNPerBlock / N1;
        constexpr index_t M1 = get_warp_size() / N0;
        constexpr index_t M0 = kBlockSize / get_warp_size();
        constexpr index_t M2 = kMPerBlock / (M1 * M0);

        constexpr auto dstr = make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2>, sequence<N0, N1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<2, 1>>{});
        static_assert(container_reduce(dstr.get_lengths(), std::multiplies<index_t>{}, 1) ==
                      kMPerBlock * kNPerBlock);
        return dstr;
    }

    template <typename DataType, index_t MPerBlock, index_t KPerBlock>
    CK_TILE_HOST_DEVICE static constexpr auto MakePreXDramTileDistribution()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);
        constexpr index_t K0 = KPerBlock / K1;
        constexpr index_t M2 = 1;
        constexpr index_t M1 = get_warp_size();
        constexpr index_t M0 = MPerBlock / M1;

        constexpr auto dstr = make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1>>,
                                       tuple<sequence<0>, sequence<1>>,
                                       sequence<1, 2, 2>,
                                       sequence<2, 0, 1>>{});
        static_assert(container_reduce(dstr.get_lengths(), std::multiplies<index_t>{}, 1) ==
                      MPerBlock * KPerBlock);
        return dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePreODramTileDistribution()
    {
        using ODataType = remove_cvref_t<typename Problem::ODataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kKPerBlock = Problem::kVHeaddim;

        return MakePreXDramTileDistribution<ODataType, kBlockSize, kKPerBlock>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePreOGradDramTileDistribution()
    {
        using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kKPerBlock = Problem::kVHeaddim;

        return MakePreXDramTileDistribution<OGradDataType, kBlockSize, kKPerBlock>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePostQGradAccDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::kM0;
        constexpr index_t kKPerBlock = Problem::kQKHeaddim;

        constexpr index_t K2 = GetAlignmentPostQGradAcc<Problem>();
        constexpr index_t K1 = min(kKPerBlock / K2, get_warp_size());
        constexpr index_t K0 = kKPerBlock / (K1 * K2);

        constexpr index_t M2 = get_warp_size() / K1;
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M1 * M2);

        constexpr auto dstr = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<1>, sequence<M0, M1, M2>, sequence<K0, K1, K2>>,
                tuple<sequence<2>, sequence<2, 3>>,
                tuple<sequence<1>, sequence<2, 1>>,
                sequence<1, 2, 3, 3>,
                sequence<0, 0, 0, 2>>{});
        static_assert(container_reduce(dstr.get_lengths(), std::multiplies<index_t>{}, 1) ==
                      kMPerBlock * kKPerBlock);
        return dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePostQGradDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::kM0;
        constexpr index_t kKPerBlock = Problem::kQKHeaddim;

        constexpr index_t K2 = GetAlignmentPostQGrad<Problem>();
        constexpr index_t K1 = min(kKPerBlock / K2, get_warp_size());
        constexpr index_t K0 = kKPerBlock / (K1 * K2);

        constexpr index_t M2 = get_warp_size() / K1;
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M1 * M2);

        constexpr auto dstr = make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 1>>,
                                       sequence<1, 2, 2>,
                                       sequence<0, 0, 2>>{});
        static_assert(container_reduce(dstr.get_lengths(), std::multiplies<index_t>{}, 1) ==
                      kMPerBlock * kKPerBlock);
        return dstr;
    }

    // these are for lds
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackQ()
    {
        return GetAlignmentQ<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackQT()
    {
        return GetTransposedAlignmentQ<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackK()
    {
        return GetAlignmentK<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackKT()
    {
        return GetTransposedAlignmentK<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        return GetAlignmentV<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackBias()
    {
        return GetAlignmentBias<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackBiasT()
    {
        return GetTransposedAlignmentBias<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackOGrad()
    {
        return GetAlignmentOGrad<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackOGradT()
    {
        return GetTransposedAlignmentOGrad<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackSGrad()
    {
        // TODO: this is for 3d layout
        using GemmDataType = remove_cvref_t<typename Problem::GemmDataType>;
        return 16 / sizeof(GemmDataType);
    }

    template <index_t KIter, index_t MNPerBlock, index_t KPerSubBlock, index_t KPack>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLdsBlockDescriptor()
    {
        constexpr auto DataTypeSize = 2; // sizeof(F16/BF16)
        constexpr auto MNLdsLayer =
            (32 * 4 / KPerSubBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / KPerSubBlock / DataTypeSize);

        constexpr auto x_lds_block_desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<KIter>{},
                                                    number<KPerSubBlock / KPack * MNLdsLayer>{},
                                                    number<MNPerBlock / MNLdsLayer>{},
                                                    number<KPack>{}),
                                         make_tuple(number<KPerSubBlock * MNPerBlock>{},
                                                    number<KPack>{},
                                                    number<KPerSubBlock * MNLdsLayer>{},
                                                    number<1>{}),
                                         number<KPack>{},
                                         number<1>{});

        constexpr auto x_lds_block_desc_permuted = transform_tensor_descriptor(
            x_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<KIter>{}),
                       make_xor_transform(make_tuple(number<MNPerBlock / MNLdsLayer>{},
                                                     number<KPerSubBlock / KPack * MNLdsLayer>{})),
                       make_pass_through_transform(number<KPack>{})),
            make_tuple(sequence<0>{}, sequence<2, 1>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<2, 1>{}, sequence<3>{}));

        constexpr auto x_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
            x_lds_block_desc_permuted,
            make_tuple(make_pass_through_transform(number<KIter>{}),
                       make_unmerge_transform(
                           make_tuple(number<KPerSubBlock / KPack>{}, number<MNLdsLayer>{})),
                       make_pass_through_transform(number<MNPerBlock / MNLdsLayer>{}),
                       make_pass_through_transform(number<KPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));

        constexpr auto x_lds_block_desc = transform_tensor_descriptor(
            x_lds_block_desc_xk0_mnldslayer_mn_xk1,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<MNPerBlock / MNLdsLayer>{}, number<MNLdsLayer>{})),
                       make_merge_transform_v3_division_mod(make_tuple(
                           number<KIter>{}, number<KPerSubBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<2, 3>{}, sequence<0, 1, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        static_assert(container_reduce(x_lds_block_desc.get_lengths(),
                                       std::multiplies<index_t>{},
                                       1) == KIter * MNPerBlock * KPerSubBlock);
        return x_lds_block_desc;
    }

    template <index_t MNPerBlock, index_t KPerBlock, index_t KPack>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLdsBlockDescriptor()
    {
        return MakeXLdsBlockDescriptor<1, MNPerBlock, KPerBlock, KPack>();
    }
    template <typename Problem,
              index_t MNPerBlock,
              index_t KPerBlock,
              index_t KPack,
              index_t KPackT>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXTLdsBlockDescriptor()
    {
        return MakeXTLdsBlockDescriptor<Problem, 1, MNPerBlock, KPerBlock, KPack, KPackT>();
    }
    template <typename Problem,
              index_t MNIter,
              index_t MNPerSubBlock,
              index_t KPerBlock,
              index_t KPack,
              index_t KPackT>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXTLdsBlockDescriptor()
    {
        // kfold and mpair dimension is not always required.
        // more dimension in merge_transform increase the difficulty of generating immarg offset
        // for compiler.
        constexpr auto MNPerXDL   = Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{});
        constexpr auto kBlockSize = Problem::kBlockSize;

        constexpr auto MN0 = MNPerSubBlock / KPack;
        constexpr auto MN1 = KPack;

        constexpr auto KThreadWrite     = kBlockSize / MN0;
        constexpr auto K0Number         = KPerBlock / KPackT;
        constexpr auto K0PerThreadWrite = K0Number / KThreadWrite;
        constexpr auto KThreadRead      = get_warp_size() / MNPerXDL; // assume 32x32x8 mfma
        constexpr auto K0PerThreadRead  = K0Number / KThreadRead;

        constexpr auto kfold = (KPackT * MN0 * 2 > 128) ? 1 : 128 / (KPackT * MN0 * 2);
        constexpr auto KThreadReadPerm =
            (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                : KThreadRead;

        // 1<=mnpair<=n0
        constexpr auto mnpair =
            (KPackT * MNPerXDL * 2 > 128)
                ? 1
                : ((128 / (KPackT * MNPerXDL * 2)) > MN0 ? MN0 : 128 / (KPackT * MNPerXDL * 2));

        constexpr auto xt_lds_block_desc_raw = make_naive_tensor_descriptor(
            make_tuple(number<MNIter>{},
                       number<KThreadWrite / kfold / KThreadReadPerm>{},
                       number<K0PerThreadWrite>{},
                       number<KThreadReadPerm * MN1>{},
                       number<kfold * MN0 / mnpair>{},
                       number<mnpair>{},
                       KPackT),
            make_tuple(number<KPackT * MN0 * KThreadWrite * MN1 * K0PerThreadWrite>{},
                       number<KPackT * kfold * MN0 * KThreadReadPerm * MN1 * K0PerThreadWrite>{},
                       number<KPackT * kfold * MN0 * KThreadReadPerm * MN1>{},
                       number<KPackT * kfold * MN0>{},
                       number<KPackT * mnpair>{},
                       number<KPackT>{},
                       number<1>{}),
            number<KPackT>{},
            number<1>{});

        constexpr auto xt_lds_block_desc_permuted = transform_tensor_descriptor(
            xt_lds_block_desc_raw,
            make_tuple(
                make_pass_through_transform(number<MNIter>{}),
                make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                make_pass_through_transform(number<K0PerThreadWrite>{}),
                make_xor_transform(
                    make_tuple(number<KThreadReadPerm * MN1>{}, number<kfold * MN0 / mnpair>{})),
                make_pass_through_transform(number<mnpair>{}),
                make_pass_through_transform(KPackT)),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3, 4>{},
                       sequence<5>{},
                       sequence<6>{}),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3, 4>{},
                       sequence<5>{},
                       sequence<6>{}));

        constexpr auto xt_lds_block_desc_unmerged = transform_tensor_descriptor(
            xt_lds_block_desc_permuted,
            make_tuple(
                make_pass_through_transform(number<MNIter>{}),
                make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                make_pass_through_transform(number<K0PerThreadWrite>{}),
                make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<MN1>{})),
                make_unmerge_transform(make_tuple(number<kfold>{}, number<MN0 / mnpair>{})),
                make_pass_through_transform(number<mnpair>{}),
                make_pass_through_transform(KPackT)),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4>{},
                       sequence<5>{},
                       sequence<6>{}),
            make_tuple(sequence<0>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<1, 4>{},
                       sequence<5, 6>{},
                       sequence<7>{},
                       sequence<8>{}));

        constexpr auto xt_lds_block_desc = transform_tensor_descriptor(
            xt_lds_block_desc_unmerged,
            make_tuple(
                make_merge_transform_v3_division_mod(
                    make_tuple(number<KThreadReadPerm>{},
                               number<KThreadWrite / kfold / KThreadReadPerm>{},
                               number<kfold>{},
                               number<K0PerThreadWrite>{},
                               number<KPackT>{})),
                make_merge_transform_v3_division_mod(make_tuple(
                    number<MNIter>{}, number<MN0 / mnpair>{}, number<mnpair>{}, number<MN1>{}))),
            make_tuple(sequence<1, 2, 5, 3, 8>{}, sequence<0, 6, 7, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        static_assert(container_reduce(xt_lds_block_desc.get_lengths(),
                                       std::multiplies<index_t>{},
                                       1) == MNPerSubBlock * MNIter * KPerBlock);
        return xt_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsWriteBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;

        using dram_encoding = typename decltype(MakeKDramTileDistribution<Problem>())::DstrEncode;
        constexpr index_t dram_y_ndim = typename dram_encoding::Ys2RHsMajor{}.size();
        if constexpr(dram_y_ndim == 2)
        {
            constexpr index_t kKPack = GetSmemKPackK<Problem>();
            return MakeXLdsBlockDescriptor<kNPerBlock, kKPerBlock, kKPack>();
        }
        else if constexpr(dram_y_ndim == 3)
        {
            constexpr index_t KIter  = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(0);
            constexpr index_t kKPack = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(2);
            return MakeXLdsBlockDescriptor<KIter, kNPerBlock, kKPerBlock / KIter, kKPack>();
        }
        else
        {
            static_assert(false, "Unexpected dram y dimension");
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKRegBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto k_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto k_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            k_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        constexpr auto k_block_dstr = make_static_tile_distribution(k_block_dstr_encode);
        static_assert(container_reduce(k_block_dstr.get_lengths(), std::multiplies<index_t>{}, 1) ==
                      kNPerBlock * kKPerBlock);
        return k_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;

        using dram_encoding = typename decltype(MakeVDramTileDistribution<Problem>())::DstrEncode;
        constexpr index_t dram_y_ndim = typename dram_encoding::Ys2RHsMajor{}.size();
        if constexpr(dram_y_ndim == 2)
        {
            constexpr index_t kVPack = GetSmemKPackV<Problem>();
            return MakeXLdsBlockDescriptor<kNPerBlock, kKPerBlock, kVPack>();
        }
        else if constexpr(dram_y_ndim == 3)
        {
            constexpr index_t KIter  = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(0);
            constexpr index_t kVPack = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(2);
            return MakeXLdsBlockDescriptor<KIter, kNPerBlock, kKPerBlock / KIter, kVPack>();
        }
        else
        {
            static_assert(false, "Unexpected dram y dimension");
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetOGradVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm2BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm2BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK2;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto v_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto v_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            v_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        constexpr auto v_block_dstr = make_static_tile_distribution(v_block_dstr_encode);
        static_assert(container_reduce(v_block_dstr.get_lengths(), std::multiplies<index_t>{}, 1) ==
                      kNPerBlock * kKPerBlock);
        return v_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledKRegWriteBlockDescriptor()
    {
        using dram_encoding = typename decltype(MakeKDramTileDistribution<Problem>())::DstrEncode;
        constexpr index_t y_ndim = typename dram_encoding::Ys2RHsMajor{}.size();
        static_assert(y_ndim >= 2);
        using shuffled_encoding_t =
            tile_distribution_encoding_shuffle_t<dram_encoding,
                                                 remove_cvref_t<decltype(swap_last2<y_ndim>)>>;
        return make_static_tile_distribution(shuffled_encoding_t{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledKLdsWriteBlockDescriptor()
    {
        // Hold all data
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        using dram_encoding = typename decltype(MakeKDramTileDistribution<Problem>())::DstrEncode;
        constexpr index_t dram_y_ndim = typename dram_encoding::Ys2RHsMajor{}.size();
        if constexpr(dram_y_ndim == 2)
        {
            constexpr index_t kKPack  = GetSmemKPackK<Problem>();
            constexpr index_t kKPackT = GetSmemKPackKT<Problem>();
            return MakeXTLdsBlockDescriptor<Problem, kNPerBlock, kKPerBlock, kKPack, kKPackT>();
        }
        else if constexpr(dram_y_ndim == 3)
        {
            constexpr index_t KIter   = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(0);
            constexpr index_t kKPack  = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(2);
            constexpr index_t kKPackT = typename dram_encoding::HsLengthss{}.at(number<0>{}).at(2);
            return MakeXTLdsBlockDescriptor<Problem,
                                            KIter,
                                            kNPerBlock / KIter,
                                            kKPerBlock,
                                            kKPack,
                                            kKPackT>();
        }
        else
        {
            static_assert(false, "Unexpected dram y dimension");
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKTLdsReadBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        auto shuffled_k_lds_block_desc = MakeShuffledKLdsWriteBlockDescriptor<Problem>();

        return transform_tensor_descriptor(
            shuffled_k_lds_block_desc,
            make_tuple(make_pass_through_transform(number<kNPerBlock>{}),
                       make_pass_through_transform(number<kKPerBlock>{})),
            make_tuple(sequence<1>{}, sequence<0>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKTRegBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto kt_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto kt_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            kt_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        constexpr auto kt_block_dstr = make_static_tile_distribution(kt_block_dstr_encode);
        static_assert(container_reduce(kt_block_dstr.get_lengths(),
                                       std::multiplies<index_t>{},
                                       1) == kNPerBlock * kKPerBlock);
        return kt_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kQKHeaddim;

        using dram_encoding = typename decltype(MakeQDramTileDistribution<Problem>())::DstrEncode;
        constexpr index_t dram_y_ndim = typename dram_encoding::Ys2RHsMajor{}.size();
        if constexpr(dram_y_ndim == 2)
        {
            constexpr index_t kKPack = GetSmemKPackQ<Problem>();
            return MakeXLdsBlockDescriptor<kMPerBlock, kKPerBlock, kKPack>();
        }
        else if constexpr(dram_y_ndim == 3)
        {
            constexpr index_t KIter  = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(0);
            constexpr index_t kKPack = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(2);
            return MakeXLdsBlockDescriptor<KIter, kMPerBlock, kKPerBlock / KIter, kKPack>();
        }
        else
        {
            static_assert(false, "Unexpected dram y dimension");
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegSliceBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto q_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto q_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            q_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        constexpr auto q_block_dstr = make_static_tile_distribution(q_block_dstr_encode);
        static_assert(container_reduce(q_block_dstr.get_lengths(), std::multiplies<index_t>{}, 1) ==
                      kMPerBlock * kKPerBlock);
        return q_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledQRegWriteBlockDescriptor()
    {
        using dram_encoding = typename decltype(MakeQDramTileDistribution<Problem>())::DstrEncode;
        constexpr index_t y_ndim = typename dram_encoding::Ys2RHsMajor{}.size();
        static_assert(y_ndim >= 2);
        using shuffled_encoding_t =
            tile_distribution_encoding_shuffle_t<dram_encoding,
                                                 remove_cvref_t<decltype(swap_last2<y_ndim>)>>;
        return make_static_tile_distribution(shuffled_encoding_t{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledQLdsWriteBlockDescriptor()
    {
        // Hold full block data
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kM0;

        using dram_encoding = typename decltype(MakeQDramTileDistribution<Problem>())::DstrEncode;
        constexpr index_t dram_y_ndim = typename dram_encoding::Ys2RHsMajor{}.size();
        if constexpr(dram_y_ndim == 2)
        {
            constexpr index_t kKPack  = GetSmemKPackQ<Problem>();
            constexpr index_t kKPackT = GetSmemKPackQT<Problem>();
            return MakeXTLdsBlockDescriptor<Problem, kNPerBlock, kKPerBlock, kKPack, kKPackT>();
        }
        else if constexpr(dram_y_ndim == 3)
        {
            constexpr index_t KIter   = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(0);
            constexpr index_t kKPack  = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(2);
            constexpr index_t kKPackT = typename dram_encoding::HsLengthss{}.at(number<0>{}).at(2);
            return MakeXTLdsBlockDescriptor<Problem,
                                            KIter,
                                            kNPerBlock / KIter,
                                            kKPerBlock,
                                            kKPack,
                                            kKPackT>();
        }
        else
        {
            static_assert(false, "Unexpected dram y dimension");
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQTLdsReadBlockDescriptor()
    {
        // Hold full block data
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kM0;

        auto shuffled_q_lds_block_desc = MakeShuffledQLdsWriteBlockDescriptor<Problem>();

        return transform_tensor_descriptor(
            shuffled_q_lds_block_desc,
            make_tuple(make_pass_through_transform(number<kNPerBlock>{}),
                       make_pass_through_transform(number<kKPerBlock>{})),
            make_tuple(sequence<1>{}, sequence<0>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQTRegSliceBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm3BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm3BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK3;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto qt_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto qt_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            qt_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        constexpr auto qt_block_dstr = make_static_tile_distribution(qt_block_dstr_encode);
        static_assert(container_reduce(qt_block_dstr.get_lengths(),
                                       std::multiplies<index_t>{},
                                       1) == kNPerBlock * kKPerBlock);

        return qt_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSGradTRegSliceBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm3BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm3BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK3;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto dst_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto dst_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            dst_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        constexpr auto dst_block_dstr = make_static_tile_distribution(dst_block_dstr_encode);
        static_assert(container_reduce(dst_block_dstr.get_lengths(),
                                       std::multiplies<index_t>{},
                                       1) == kMPerBlock * kKPerBlock);
        return dst_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEDLdsWriteBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        using LSEDType               = remove_cvref_t<typename Problem::DDataType>;
        constexpr index_t kMPack     = 16 / sizeof(LSEDType);

        constexpr auto lsed_lds_block_desc =
            make_naive_tensor_descriptor(make_tuple(number<kMPerBlock>{}),
                                         make_tuple(number<1>{}),
                                         number<kMPack>{},
                                         number<1>{});

        return lsed_lds_block_desc;
    }

    template <typename Problem, typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEDLdsReadBlockDescriptor()
    {
        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;

        constexpr index_t N1 = WG::WarpGemmAttribute::Impl::kCNLane;
        constexpr index_t N0 = NWarp;

        // M4 *2 and M2 /2 when swizzle mode enabled
        constexpr index_t SwizzleConfig = WG::kM == 16 ? 1 : 2;
        // constexpr index_t SwizzleConfig = 1;
        constexpr index_t M4 = WG::WarpGemmAttribute::Impl::kCM1PerLane * SwizzleConfig;
        constexpr index_t M3 = WG::WarpGemmAttribute::Impl::kCMLane;
        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kCM0PerLane / SwizzleConfig;
        constexpr index_t M1 = MWarp;
        constexpr index_t M0 = kMPerBlock / (M1 * WG::WarpGemmAttribute::Impl::kM);

        constexpr auto dstr = make_static_tile_distribution(
            tile_distribution_encoding<sequence<N0, N1>,
                                       tuple<sequence<M0, M1, M2, M3, M4>>,
                                       tuple<sequence<1, 0>, sequence<1, 0>>,
                                       tuple<sequence<1, 0>, sequence<3, 1>>,
                                       sequence<1, 1, 1>,
                                       sequence<0, 2, 4>>{});
        static_assert(container_reduce(dstr.get_lengths(), std::multiplies<index_t>{}, 1) ==
                      kMPerBlock);
        return dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradLdsBlockDescriptor()
    {
        // Hold full block data
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;

        using dram_encoding =
            typename decltype(MakeOGradDramTileDistribution<Problem>())::DstrEncode;
        constexpr index_t dram_y_ndim = typename dram_encoding::Ys2RHsMajor{}.size();
        if constexpr(dram_y_ndim == 2)
        {
            constexpr index_t kKPack = GetSmemKPackOGrad<Problem>();
            return MakeXLdsBlockDescriptor<kMPerBlock, kKPerBlock, kKPack>();
        }
        else if constexpr(dram_y_ndim == 3)
        {
            constexpr index_t KIter  = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(0);
            constexpr index_t kKPack = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(2);
            return MakeXLdsBlockDescriptor<KIter, kMPerBlock, kKPerBlock / KIter, kKPack>();
        }
        else
        {
            static_assert(false, "Unexpected dram y dimension");
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradRegSliceBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetOGradVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm2BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm2BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK2;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto do_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto do_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            do_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        constexpr auto do_block_dstr = make_static_tile_distribution(do_block_dstr_encode);
        static_assert(container_reduce(do_block_dstr.get_lengths(),
                                       std::multiplies<index_t>{},
                                       1) == kMPerBlock * kKPerBlock);
        return do_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledOGradRegWriteBlockDescriptor()
    {

        using dram_encoding =
            typename decltype(MakeOGradDramTileDistribution<Problem>())::DstrEncode;
        constexpr index_t y_ndim = typename dram_encoding::Ys2RHsMajor{}.size();
        static_assert(y_ndim >= 2);
        using shuffled_encoding_t =
            tile_distribution_encoding_shuffle_t<dram_encoding,
                                                 remove_cvref_t<decltype(swap_last2<y_ndim>)>>;
        return make_static_tile_distribution(shuffled_encoding_t{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledOGradLdsWriteBlockDescriptor()
    {
        // Hold all data
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kM0;

        using dram_encoding =
            typename decltype(MakeOGradDramTileDistribution<Problem>())::DstrEncode;
        constexpr index_t dram_y_ndim = typename dram_encoding::Ys2RHsMajor{}.size();
        if constexpr(dram_y_ndim == 2)
        {
            constexpr index_t kKPack  = GetSmemKPackOGrad<Problem>();
            constexpr index_t kKPackT = GetSmemKPackOGradT<Problem>();
            return MakeXTLdsBlockDescriptor<Problem, kNPerBlock, kKPerBlock, kKPack, kKPackT>();
        }
        else if constexpr(dram_y_ndim == 3)
        {
            constexpr index_t KIter   = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(0);
            constexpr index_t kKPack  = typename dram_encoding::HsLengthss{}.at(number<1>{}).at(2);
            constexpr index_t kKPackT = typename dram_encoding::HsLengthss{}.at(number<0>{}).at(2);
            return MakeXTLdsBlockDescriptor<Problem,
                                            KIter,
                                            kNPerBlock / KIter,
                                            kKPerBlock,
                                            kKPack,
                                            kKPackT>();
        }
        else
        {
            static_assert(false, "Unexpected dram y dimension");
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradTLdsReadBlockDescriptor()
    {
        // Hold all data
        constexpr index_t kNPerBlock    = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock    = Problem::BlockFmhaShape::kM0;
        auto shuffled_do_lds_block_desc = MakeShuffledOGradLdsWriteBlockDescriptor<Problem>();

        return transform_tensor_descriptor(
            shuffled_do_lds_block_desc,
            make_tuple(make_pass_through_transform(number<kNPerBlock>{}),
                       make_pass_through_transform(number<kKPerBlock>{})),
            make_tuple(sequence<1>{}, sequence<0>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradTRegSliceBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        // constexpr index_t kNPerBlock = 32;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto dot_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto dot_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            dot_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        constexpr auto dot_block_dstr = make_static_tile_distribution(dot_block_dstr_encode);
        static_assert(container_reduce(dot_block_dstr.get_lengths(),
                                       std::multiplies<index_t>{},
                                       1) == kNPerBlock * kKPerBlock);
        return dot_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePTRegSliceBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto pt_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto pt_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            pt_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        constexpr auto pt_block_dstr = make_static_tile_distribution(pt_block_dstr_encode);
        static_assert(container_reduce(pt_block_dstr.get_lengths(),
                                       std::multiplies<index_t>{},
                                       1) == kMPerBlock * kKPerBlock);
        return pt_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSGradLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPack     = GetSmemKPackSGrad<Problem>();

        return MakeXLdsBlockDescriptor<kMPerBlock, kKPerBlock, kKPack>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSGradRegSliceBlockDescriptor()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK4;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto ds_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto ds_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            ds_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        constexpr auto ds_block_dstr = make_static_tile_distribution(ds_block_dstr_encode);
        static_assert(container_reduce(ds_block_dstr.get_lengths(),
                                       std::multiplies<index_t>{},
                                       1) == kMPerBlock * kKPerBlock);
        return ds_block_dstr;
    }

    template <typename Problem, typename PTOutTensor, typename PInTensor>
    CK_TILE_DEVICE static constexpr void PTFromGemm0CToGemm1A(PTOutTensor& pt_out,
                                                              const PInTensor& p_in)
    {
        if constexpr(Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}) == 16)
        {
            using BlockGemm       = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<0>{});

            constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

            constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
            constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

            using AWarpDstr = typename WarpGemm::AWarpDstr;
            using CWarpDstr = typename WarpGemm::CWarpDstr;
            auto pt_warp_tensor =
                make_static_distributed_tensor<typename Problem::GemmDataType>(CWarpDstr{});

            constexpr auto a_warp_y_lengths =
                to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

            constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    pt_warp_tensor.get_thread_buffer() = p_in.get_y_sliced_thread_data(
                        merge_sequences(sequence<kIter, mIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    pt_out.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths),
                        pt_warp_tensor.get_thread_buffer());
                });
            });
        }
        else
        {
            pt_out.get_thread_buffer() = p_in.get_thread_buffer();
        }
    }

    template <typename Problem, typename SGradTOutTensor, typename SGradInTensor>
    CK_TILE_DEVICE static constexpr void SGradTFromGemm2CToGemm3A(SGradTOutTensor& dst_out,
                                                                  const SGradInTensor& ds_in)
    {
        if constexpr(Problem::BlockFmhaShape::Gemm3WarpTile::at(number<0>{}) == 16)
        {
            using BlockGemm       = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm3BlockWarps::at(number<0>{});

            constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kN0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK3;

            constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
            constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

            using AWarpDstr = typename WarpGemm::AWarpDstr;
            using CWarpDstr = typename WarpGemm::CWarpDstr;
            auto dst_warp_tensor =
                make_static_distributed_tensor<typename Problem::GemmDataType>(CWarpDstr{});

            constexpr auto a_warp_y_lengths =
                to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

            constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    dst_warp_tensor.get_thread_buffer() = ds_in.get_y_sliced_thread_data(
                        merge_sequences(sequence<kIter, mIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    dst_out.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths),
                        dst_warp_tensor.get_thread_buffer());
                });
            });
        }
        else
        {
            dst_out.get_thread_buffer() = ds_in.get_thread_buffer();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledBiasTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t N1 = GetAlignmentBias<Problem>();
        constexpr index_t N0 = kNPerBlock / N1;
        constexpr index_t M2 = GetTransposedAlignmentBias<Problem>();
        constexpr index_t M1 = get_warp_size() / N0;
        constexpr index_t M0 = kBlockSize / get_warp_size();

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2>, sequence<N0, N1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<2, 1>,
                                       sequence<1, 2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasLdsBlockDescriptor()
    {
        // Hold full block data
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;

        constexpr index_t kKPack  = GetSmemKPackBias<Problem>();
        constexpr index_t kKPackT = GetSmemKPackBiasT<Problem>();

        return MakeXTLdsBlockDescriptor<Problem, kNPerBlock, kMPerBlock, kKPack, kKPackT>();
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasSTileDistribution()
    {
        using c_block_tensor_type = decltype(BlockGemm{}.MakeCBlockTile());
        return c_block_tensor_type::get_tile_distribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQ()
    {
        constexpr index_t smem_size_q = sizeof(typename Problem::QDataType) *
                                        MakeQLdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_q;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQT()
    {
        constexpr index_t smem_size_qt =
            sizeof(typename Problem::QDataType) *
            MakeShuffledQLdsWriteBlockDescriptor<Problem>().get_element_space_size();

        return smem_size_qt;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeK()
    {
        constexpr index_t smem_size_k =
            sizeof(typename Problem::KDataType) *
            MakeKLdsWriteBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_k;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeKT()
    {
        constexpr index_t smem_size_kt =
            sizeof(typename Problem::KDataType) *
            MakeKTLdsReadBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_kt;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeLSE()
    {
        constexpr index_t smem_size_lse =
            sizeof(typename Problem::LSEDataType) *
            MakeLSEDLdsWriteBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_lse;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeD()
    {
        constexpr index_t smem_size_d =
            sizeof(typename Problem::DDataType) *
            MakeLSEDLdsWriteBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_d;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        constexpr index_t smem_size_v =
            sizeof(typename Problem::VDataType) *
            MakeVLdsWriteBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_v;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGrad()
    {
        constexpr index_t smem_size_do =
            sizeof(typename Problem::OGradDataType) *
            MakeOGradLdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_do;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGradT()
    {
        constexpr index_t smem_size_dot =
            sizeof(typename Problem::OGradDataType) *
            MakeShuffledOGradLdsWriteBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_dot;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeSGrad()
    {
        constexpr index_t smem_size_ds =
            sizeof(typename Problem::GemmDataType) *
            MakeSGradLdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_ds;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeBias()
    {
        constexpr index_t smem_size_bias = [&]() {
            if constexpr(Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                return sizeof(typename Problem::BiasDataType) *
                       MakeBiasLdsBlockDescriptor<Problem>().get_element_space_size();
            else
                return 0;
        }();
        return smem_size_bias;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_q    = GetSmemSizeQ<Problem>();
        constexpr index_t smem_size_qt   = GetSmemSizeQT<Problem>();
        constexpr index_t smem_size_lse  = GetSmemSizeLSE<Problem>();
        constexpr index_t smem_size_k    = GetSmemSizeK<Problem>();
        constexpr index_t smem_size_kt   = GetSmemSizeKT<Problem>();
        constexpr index_t smem_size_v    = GetSmemSizeV<Problem>();
        constexpr index_t smem_size_do   = GetSmemSizeOGrad<Problem>();
        constexpr index_t smem_size_dot  = GetSmemSizeOGradT<Problem>();
        constexpr index_t smem_size_d    = GetSmemSizeD<Problem>();
        constexpr index_t smem_size_ds   = GetSmemSizeSGrad<Problem>();
        constexpr index_t smem_size_bias = GetSmemSizeBias<Problem>();

        constexpr index_t smem_size_stage0_0 = smem_size_k + smem_size_kt;
        constexpr index_t smem_size_stage0_1 = smem_size_v;
        constexpr index_t smem_size_stage1   = smem_size_qt + smem_size_q + smem_size_dot +
                                             smem_size_do + smem_size_lse + smem_size_d +
                                             max(smem_size_bias, smem_size_ds);

        return max(smem_size_stage0_0, smem_size_stage0_1, smem_size_stage1);
    }

    template <typename Problem_>
    struct HotLoopScheduler
    {
        using Problem = Problem_;

        template <index_t GemmStage>
        CK_TILE_DEVICE static constexpr void GemmStagedScheduler()
        {
        }

        template <>
        CK_TILE_DEVICE constexpr void GemmStagedScheduler<0>()
        {
            // Mem: Q, LSE, OGrad, D global load, OGrad^T LDS load
            // Comp: Q x K
            constexpr index_t VMEM_READ_INST =
                Q_VMEM_READ + OGrad_VMEM_READ + LSE_VMEM_READ + D_VMEM_READ;
            constexpr index_t LDS_READ_INST = OGradT_LDS_READ;
            constexpr index_t MFMA_INST     = Gemm0MFMA;

            // Evenly distributed to relieve SQ->TA FIFO pressure
            constexpr index_t MFMA_PER_VMEM_READ = MFMA_INST / VMEM_READ_INST;
            constexpr index_t MFMA_Remainder     = MFMA_INST - MFMA_PER_VMEM_READ * VMEM_READ_INST;
            // To hide instruction issue latency
            constexpr index_t LDS_READ_PER_MFMA = LDS_READ_INST / MFMA_INST;

            static_for<0, VMEM_READ_INST, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                static_for<0, MFMA_PER_VMEM_READ, 1>{}([&](auto j) {
                    ignore = j;
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                 // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x100, LDS_READ_PER_MFMA, 0); // DS read
                });
            });
            static_for<0, MFMA_Remainder, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                 // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, LDS_READ_PER_MFMA, 0); // DS read
            });
        }

        template <>
        CK_TILE_DEVICE constexpr void GemmStagedScheduler<1>()
        {
            // Mem:  Q^T LDS load
            // Comp: OGrad x V
            constexpr index_t LDS_READ_INST = QT_LDS_READ;
            constexpr index_t MFMA_INST     = Gemm1MFMA;

            // To hide instruction issue latency
            constexpr index_t LDS_READ_PER_MFMA = LDS_READ_INST / MFMA_INST;

            static_for<0, MFMA_INST, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                 // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, LDS_READ_PER_MFMA, 0); // DS read
            });
        }

        template <>
        CK_TILE_DEVICE constexpr void GemmStagedScheduler<2>()
        {
            // Mem: Q, QT, LSE, OGrad, OGradT, D, LDS store
            // Comp: PT x OGrad
            constexpr index_t LDS_WRITE_INST = Q_LDS_WRITE + QT_LDS_WRITE + OGrad_LDS_WRITE +
                                               OGradT_LDS_WRITE + LSE_LDS_WRITE + D_LDS_WRITE;
            constexpr index_t MFMA_INST = Gemm2MFMA;

            // To hide instruction issue latency
            constexpr index_t LDS_WRITE_PER_MFMA = LDS_WRITE_INST / MFMA_INST;

            static_for<0, MFMA_INST, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                  // MFMA
                __builtin_amdgcn_sched_group_barrier(0x200, LDS_WRITE_PER_MFMA, 0); // DS write
            });
        }

        template <>
        CK_TILE_DEVICE constexpr void GemmStagedScheduler<3>()
        {
            // Mem: SGradT LDS store, SGrad, Q, LSE LDS load.
            // Comp: SGradT x QT
            constexpr index_t LDS_WRITE_INST = SGradT_LDS_WRITE;
            constexpr index_t LDS_READ_INST  = SGradT_LDS_READ_P1 + Q_LDS_READ + LSE_LDS_READ;
            constexpr index_t MFMA_INST      = Gemm3MFMA;

            // To hide instruction issue latency
            constexpr index_t LDS_WRITE_PER_MFMA =
                LDS_WRITE_INST / MFMA_INST >= 1 ? LDS_WRITE_INST / MFMA_INST : 1;
            constexpr index_t MFMA_INST_LDS_WRITE = LDS_WRITE_INST / LDS_WRITE_PER_MFMA;

            constexpr index_t LDS_READ_PER_MFMA =
                (MFMA_INST - MFMA_INST_LDS_WRITE) > 0
                    ? LDS_READ_INST / (MFMA_INST - MFMA_INST_LDS_WRITE) > 0
                          ? LDS_READ_INST / (MFMA_INST - MFMA_INST_LDS_WRITE)
                          : 1
                    : 0;

            static_for<0, MFMA_INST_LDS_WRITE, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                  // MFMA
                __builtin_amdgcn_sched_group_barrier(0x200, LDS_WRITE_PER_MFMA, 0); // DS Write
            });

            static_for<0, MFMA_INST - MFMA_INST_LDS_WRITE, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                 // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, LDS_READ_PER_MFMA, 0); // DS Read
            });
        }

        template <>
        CK_TILE_DEVICE constexpr void GemmStagedScheduler<4>()
        {
            // Mem: SGrad, OGrad, D LDS load.
            // Comp: SGrad x KT
            constexpr index_t LDS_READ_INST = SGradT_LDS_READ_P2 + OGrad_LDS_READ + D_LDS_READ;
            constexpr index_t MFMA_INST     = Gemm4MFMA;

            // To hide instruction issue latency
            constexpr index_t LDS_READ_PER_MFMA =
                LDS_READ_INST / MFMA_INST > 0 ? LDS_READ_INST / MFMA_INST : 1;

            static_for<0, MFMA_INST, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);                 // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, LDS_READ_PER_MFMA, 0); // DS Read
            });
        }

        private:
        static constexpr index_t kBlockSize = Problem::kBlockSize;
        static constexpr index_t kM0        = Problem::BlockFmhaShape::kM0;
        static constexpr index_t kN0        = Problem::BlockFmhaShape::kN0;
        static constexpr index_t kQKHeaddim = Problem::BlockFmhaShape::kQKHeaddim;
        static constexpr index_t kVHeaddim  = Problem::BlockFmhaShape::kVHeaddim;
        static constexpr index_t kK0        = Problem::BlockFmhaShape::kK0;
        static constexpr index_t kK2        = Problem::BlockFmhaShape::kK2;
        static constexpr index_t kK4        = Problem::BlockFmhaShape::kK4;

        static constexpr index_t WarpGemmM =
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{});
        static constexpr index_t WarpGemmN =
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{});
        static constexpr index_t WarpGemmK = WarpGemmM == 16 ? 16 : 8;
        static constexpr index_t Gemm4MWarp =
            Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<0>{});
        static constexpr index_t Gemm4NWarp =
            Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<1>{});

        // Compute
        static constexpr index_t Gemm0MFMA =
            kM0 * kN0 * kK0 / (kBlockSize / get_warp_size() * WarpGemmM * WarpGemmN * WarpGemmK);
        static constexpr index_t Gemm1MFMA =
            kN0 * kVHeaddim * kM0 /
            (kBlockSize / get_warp_size() * WarpGemmM * WarpGemmN * WarpGemmK);
        static constexpr index_t Gemm2MFMA =
            kM0 * kN0 * kK2 / (kBlockSize / get_warp_size() * WarpGemmM * WarpGemmN * WarpGemmK);
        static constexpr index_t Gemm3MFMA =
            kN0 * kQKHeaddim * kM0 /
            (kBlockSize / get_warp_size() * WarpGemmM * WarpGemmN * WarpGemmK);
        static constexpr index_t Gemm4MFMA =
            kM0 * kQKHeaddim * kN0 /
            (kBlockSize / get_warp_size() * WarpGemmM * WarpGemmN * WarpGemmK);

        // VMEM
        static constexpr index_t Q_VMEM_READ =
            kM0 * kQKHeaddim / kBlockSize / GetAlignmentQ<Problem>();
        static constexpr index_t OGrad_VMEM_READ =
            kM0 * kVHeaddim / kBlockSize / GetAlignmentOGrad<Problem>();
        static constexpr index_t LSE_VMEM_READ = 1;
        static constexpr index_t D_VMEM_READ   = 1;

        // LDS Read
        static constexpr index_t OGradT_LDS_READ =
            kM0 * kVHeaddim / get_warp_size() / GetTransposedAlignmentOGrad<Problem>();
        static constexpr index_t QT_LDS_READ =
            kM0 * kQKHeaddim / get_warp_size() / GetTransposedAlignmentQ<Problem>();
        static constexpr index_t SGradT_LDS_READ_P1 =
            kM0 * kK4 / (get_warp_size() * Gemm4MWarp) / GetSmemKPackSGrad<Problem>();
        static constexpr index_t Q_LDS_READ   = kM0 * kK0 / kBlockSize / GetAlignmentQ<Problem>();
        static constexpr index_t LSE_LDS_READ = WarpGemmM == 16 ? kM0 / (4 * 4) : kM0 / (2 * 4);
        static constexpr index_t SGradT_LDS_READ_P2 =
            kM0 * (kN0 - kK4) / (get_warp_size() * Gemm4MWarp) / GetSmemKPackSGrad<Problem>();
        static constexpr index_t OGrad_LDS_READ =
            kM0 * kK2 / kBlockSize / GetAlignmentOGrad<Problem>();
        static constexpr index_t D_LDS_READ = WarpGemmM == 16 ? kM0 / (4 * 4) : kM0 / (2 * 4);

        // LDS Write
        static constexpr index_t Q_LDS_WRITE =
            kM0 * kQKHeaddim / Problem::kBlockSize / GetAlignmentQ<Problem>();
        static constexpr index_t QT_LDS_WRITE =
            kM0 * kQKHeaddim / kBlockSize / GetTransposedAlignmentQ<Problem>();
        static constexpr index_t OGrad_LDS_WRITE =
            kM0 * kVHeaddim / kBlockSize / GetAlignmentOGrad<Problem>();
        static constexpr index_t OGradT_LDS_WRITE =
            kM0 * kVHeaddim / kBlockSize / GetTransposedAlignmentOGrad<Problem>();
        static constexpr index_t LSE_LDS_WRITE    = 1;
        static constexpr index_t D_LDS_WRITE      = 1;
        static constexpr index_t SGradT_LDS_WRITE = kM0 * kN0 / kBlockSize;
    };
};

} // namespace ck_tile
