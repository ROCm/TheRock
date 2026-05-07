// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"

#include "ck_tile/core/utility/debug.hpp"

namespace ck_tile {

struct BlockFmhaBwdPipelineTrLoadDefaultPolicy
{
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

        constexpr auto SwizzleA = false;
        using WarpGemm          = WarpGemmDispatcher< //
            typename Problem::QDataType,
            typename Problem::KDataType,
            typename Problem::AccDataType,
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
            false,
            SwizzleA>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::QDataType,
                                                typename Problem::KDataType,
                                                typename Problem::AccDataType,
                                                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                WarpGemm>;

        return BlockGemmARegBRegCRegV1<GemmProblem, BlockGemmPolicy, /* TransposeC */ true>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPTOGradTBlockGemm()
    {
        return BlockFmhaBwdPipelineDefaultPolicy::GetPTOGradTBlockGemm<Problem>();
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

        constexpr auto SwizzleA = false;
        using WarpGemm          = WarpGemmDispatcher< //
            typename Problem::OGradDataType,
            typename Problem::VDataType,
            typename Problem::AccDataType,
            Problem::BlockFmhaShape::Gemm2WarpTile::at(number<0>{}),
            Problem::BlockFmhaShape::Gemm2WarpTile::at(number<1>{}),
            Problem::BlockFmhaShape::Gemm2WarpTile::at(number<2>{}),
            false,
            SwizzleA>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::OGradDataType,
                                                typename Problem::VDataType,
                                                typename Problem::AccDataType,
                                                typename Problem::BlockFmhaShape::Gemm2BlockWarps,
                                                WarpGemm>;

        return BlockGemmARegBRegCRegV1<GemmProblem, BlockGemmPolicy, /* TransposeC */ true>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSGradTQTBlockGemm()
    {
        return BlockFmhaBwdPipelineDefaultPolicy::GetSGradTQTBlockGemm<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSGradKTBlockGemm()
    {
        using BlockFmhaShape = typename Problem::BlockFmhaShape;
        using GemmProblem    = BlockGemmProblem<
               typename Problem::GemmDataType,
               typename Problem::KDataType,
               typename Problem::AccDataType,
               Problem::kBlockSize,
               TileGemmShape<
                   sequence<BlockFmhaShape::kM0, BlockFmhaShape::kQKHeaddim, BlockFmhaShape::kK4>,
                   typename BlockFmhaShape::Gemm4BlockWarps,
                   typename BlockFmhaShape::Gemm4WarpTile>>;

        using WarpGemm = WarpGemmDispatcher< //
            typename Problem::GemmDataType,
            typename Problem::KDataType,
            typename Problem::AccDataType,
            BlockFmhaShape::Gemm4WarpTile::at(number<0>{}),
            BlockFmhaShape::Gemm4WarpTile::at(number<1>{}),
            BlockFmhaShape::Gemm4WarpTile::at(number<2>{}),
            false,
            false,
            false,
            (Problem::BlockFmhaShape::Gemm4WarpTile::at(number<2>{}) == 32)
                ? WGAttrNumAccessEnum ::Double
                : WGAttrNumAccessEnum ::Single>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::GemmDataType,
                                                typename Problem::KDataType,
                                                typename Problem::AccDataType,
                                                typename BlockFmhaShape::Gemm4BlockWarps,
                                                WarpGemm>;

        return BlockGemmARegBRegCRegV1<GemmProblem, BlockGemmPolicy>{};
    }

    // these are for global load
    template <typename Problem, typename T>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentX() noexcept
    {
        return 16 / sizeof(T);
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        return GetAlignmentX<Problem, typename Problem::QDataType>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        return GetAlignmentX<Problem, typename Problem::KDataType>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        return GetAlignmentX<Problem, typename Problem::VDataType>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        return GetAlignmentX<Problem, typename Problem::ODataType>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOGrad()
    {
        return GetAlignmentX<Problem, typename Problem::OGradDataType>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentBias()
    {
        return GetAlignmentX<Problem, typename Problem::BiasDataType>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentKGrad()
    {
        return GetAlignmentX<Problem, typename Problem::KGradDataType>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentVGrad()
    {
        return GetAlignmentX<Problem, typename Problem::VGradDataType>();
    }

    // these are for load_tr_b64
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr auto GetTransposedAlignmentX() noexcept
    {
        return 8 / sizeof(T);
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTransposedAlignmentQ() noexcept
    {
        return GetTransposedAlignmentX<typename Problem::QDataType>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetTransposedAlignmentOGrad()
    {
        return GetTransposedAlignmentX<typename Problem::OGradDataType>();
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

    // It is found that alignment of 8x dwordx4 can avoid bank conflicts for both transposed and
    // non-transposed load
    static constexpr index_t WarpAlignmentBytes = 128;

    // As load_lds requires contiguous LDS write, we need to transform the distribution of DRAM for
    // reading
    template <typename T, typename TensorView>
    CK_TILE_HOST_DEVICE static constexpr auto TransformXDramTensorView(const TensorView& naive_view)
    {
        if constexpr(std::is_same_v<TensorView, ck_tile::null_tensor_view>)
        {
            return naive_view;
        }
        else
        {
            const auto transformed_desc =
                TransformXDramDescriptor<T>(naive_view.get_tensor_descriptor());
            return tensor_view<typename TensorView::buffer_view,
                               remove_cvref_t<decltype(transformed_desc)>,
                               TensorView::DstInMemOp>{naive_view.buf_, transformed_desc};
        }
    }
    template <typename T, typename... TD_TS>
    CK_TILE_HOST_DEVICE static constexpr auto
    TransformXDramDescriptor(const tensor_descriptor<TD_TS...>& from_desc)
    {
        using from_desc_t = tensor_descriptor<TD_TS...>;

        constexpr auto ndims = from_desc_t::get_num_of_dimension();
        static_assert(ndims == 2, "XDram descriptor must have 2 dimensions");
        const auto Rows = from_desc.get_length(number<0>{});
        // constexpr auto Cols = 128;
        // assert(from_desc.get_length(number<1>{}) == 128);
        const auto Cols = from_desc.get_length(number<1>{});

        constexpr index_t Dwordx4Bytes = 16;
        constexpr index_t K2           = Dwordx4Bytes / sizeof(T);
        constexpr index_t K1           = WarpAlignmentBytes / Dwordx4Bytes;
        const index_t K0               = Cols / K1;
        const auto ColLens             = make_tuple(K0, number<K1>{}, number<K2>{});

        const auto desc_tmp1 = transform_tensor_descriptor(
            from_desc,
            make_tuple(make_pass_through_transform(Rows), make_unmerge_transform(ColLens)),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1, 2, 3>{}));

        const auto desc_tmp2 = transform_tensor_descriptor(
            desc_tmp1,
            make_tuple(make_xor_transform(make_tuple(Rows, number<K1>{})),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        return transform_tensor_descriptor(
            desc_tmp2,
            make_tuple(make_pass_through_transform(Rows),
                       make_merge_transform_v3_division_mod(ColLens)),
            make_tuple(sequence<0>{}, sequence<1, 2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    template <typename Problem, typename T, index_t RowsPerBlock, index_t ColsPerBlock>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kWarps     = kBlockSize / get_warp_size();

        constexpr index_t K3       = GetAlignmentK<Problem>();            // 8
        constexpr index_t K2       = WarpAlignmentBytes / sizeof(T) / K3; // 8
        constexpr index_t K_remain = ColsPerBlock / K2 / K3;
        constexpr index_t K1       = min(kWarps, K_remain);
        constexpr index_t K0       = K_remain / K1;
        static_assert((K0 * K1 * K2 * K3 == ColsPerBlock) &&
                          K2 * K3 * sizeof(T) == WarpAlignmentBytes,
                      "ColsPerBlock notdivisible");

        constexpr index_t N2 = get_warp_size() / K2; // 8
        constexpr index_t N1 = max(1, kWarps / K1);
        constexpr index_t N0 = RowsPerBlock / N1 / N2;
        static_assert((N0 * N1 * N2 == RowsPerBlock) && (K1 * N1 == kWarps) &&
                          (K2 * N2 == get_warp_size()),
                      "RowsPerBlock not divisible");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1, K2, K3>>,
                                       tuple<sequence<2, 1>, sequence<1, 2>>, // K1 N1, N2 K2
                                       tuple<sequence<1, 1>, sequence<2, 2>>,
                                       sequence<1, 2, 2>, // N0 K0 K3
                                       sequence<0, 0, 3>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        return MakeXDramTileDistribution<Problem,
                                         typename Problem::KDataType,
                                         Problem::BlockFmhaShape::kN0,
                                         Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        return MakeXDramTileDistribution<Problem,
                                         typename Problem::VDataType,
                                         Problem::BlockFmhaShape::kN0,
                                         Problem::BlockFmhaShape::kVHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        return MakeXDramTileDistribution<Problem,
                                         typename Problem::QDataType,
                                         Problem::BlockFmhaShape::kM0,
                                         Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradDramTileDistribution()
    {
        return MakeXDramTileDistribution<Problem,
                                         typename Problem::OGradDataType,
                                         Problem::BlockFmhaShape::kM0,
                                         Problem::BlockFmhaShape::kVHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEDDramTileDistribution()
    {
        using BlockGemm         = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;

        constexpr index_t N0 = MWarp * NWarp;

        constexpr index_t M1 = kMPerBlock;
        constexpr index_t M0 = get_warp_size() / M1;
        static_assert(M1 <= get_warp_size() && get_warp_size() % M1 == 0,
                      "M1 must be a factor of warp size");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<N0, M0>,
                                       tuple<sequence<M1, 1>>,
                                       tuple<sequence<0>, sequence<0, 1>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1>,
                                       sequence<1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasTileDistribution()
    {
        return BlockFmhaBwdPipelineDefaultPolicy::MakeBiasTileDistribution<Problem>();
    }

    template <typename DataType, index_t MPerBlock, index_t KPerBlock>
    CK_TILE_HOST_DEVICE static constexpr auto MakePreXDramTileDistribution()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);
        constexpr index_t K0 = KPerBlock / K1;
        constexpr index_t M2 = 1;
        constexpr index_t M1 = get_warp_size();
        constexpr index_t M0 = MPerBlock / M1;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1>>,
                                       tuple<sequence<0>, sequence<1>>,
                                       sequence<1, 2, 2>,
                                       sequence<2, 0, 1>>{});
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
        using AccDataType = remove_cvref_t<typename Problem::AccDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::kM0;
        constexpr index_t kKPerBlock = Problem::kQKHeaddim;

        constexpr index_t K1 = 16 / sizeof(AccDataType);
        constexpr index_t K0 = kKPerBlock / K1;

        constexpr index_t M2 = get_warp_size() / K0;
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M1 * M2);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<1>, sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<2>, sequence<2, 3>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2, 3>,
                                       sequence<0, 0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePostQGradDramTileDistribution()
    {
        using AccDataType = remove_cvref_t<typename Problem::AccDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::kM0;
        constexpr index_t kKPerBlock = Problem::kQKHeaddim;

        constexpr index_t K1 = 16 / sizeof(AccDataType);
        constexpr index_t K0 = kKPerBlock / K1;

        constexpr index_t M2 = get_warp_size() / K0;
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M1 * M2);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKRegBlockDescriptor()
    {
        return BlockFmhaBwdPipelineDefaultPolicy::MakeKRegBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegBlockDescriptor()
    {
        return BlockFmhaBwdPipelineDefaultPolicy::MakeVRegBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKTRegBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto kt_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<MWarp>,
            tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>, // 2 4, 4
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto kt_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            kt_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        auto output =
            make_static_tile_distribution(typename InputTileDistributionTraits<
                                          decltype(kt_block_dstr_encode),
                                          typename Problem::KDataType>::TransposedDstrEncode{});
        return output;
    }

    // lds write descriptor used together with block_sync_lds (transformed dram descriptor)
    template <typename T, index_t MNPerBlock, index_t KPerBlock>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLdsWriteBlockDescriptor()
    {
        constexpr index_t KPack = WarpAlignmentBytes / sizeof(T);

        constexpr auto desc_0 = make_naive_tensor_descriptor_packed(
            make_tuple(number<KPerBlock / KPack>{}, number<MNPerBlock>{}, number<KPack>{}));
        return transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<MNPerBlock>{}),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsWriteBlockDescriptor()
    {
        return MakeXLdsWriteBlockDescriptor<typename Problem::KDataType,
                                            Problem::BlockFmhaShape::kN0,
                                            Problem::BlockFmhaShape::kQKHeaddim>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
    {
        return MakeXLdsWriteBlockDescriptor<typename Problem::VDataType,
                                            Problem::BlockFmhaShape::kN0,
                                            Problem::BlockFmhaShape::kVHeaddim>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsWriteBlockDescriptor()
    {
        return MakeXLdsWriteBlockDescriptor<typename Problem::QDataType,
                                            Problem::BlockFmhaShape::kM0,
                                            Problem::BlockFmhaShape::kQKHeaddim>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradLdsWriteBlockDescriptor()
    {
        return MakeXLdsWriteBlockDescriptor<typename Problem::OGradDataType,
                                            Problem::BlockFmhaShape::kM0,
                                            Problem::BlockFmhaShape::kQKHeaddim>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasLdsBlockDescriptor()
    {
        return BlockFmhaBwdPipelineDefaultPolicy::MakeBiasLdsBlockDescriptor<Problem>();
    }

    template <typename Problem, bool Transposed = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSGradLdsBlockDescriptor()
    {
        // SGrad should be of the same distr as Gemm2 OGradV's output (i.e. PGrad)
        using BlockGemm = remove_cvref_t<decltype(GetOGradVBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t M2 = WarpGemm::WarpGemmAttribute::Impl::kCM1PerLane;
        constexpr index_t M1 = WarpGemm::WarpGemmAttribute::Impl::kCMLane;
        static_assert(WarpGemm::WarpGemmAttribute::Impl::kCM0PerLane == 1, "kCM0PerLane must be 1");
        constexpr index_t M0 = kMPerBlock / (M1 * M2);

        constexpr index_t N1 = WarpGemm::WarpGemmAttribute::Impl::kCNLane;
        constexpr index_t N0 = kNPerBlock / N1;

        constexpr auto desc_0 = make_naive_tensor_descriptor_packed(
            make_tuple(number<M0>{}, number<N0>{}, number<M1>{}, number<N1>{}, number<M2>{}));

        constexpr index_t M1_0 = 2, M1_1 = 2;
        constexpr index_t N1_0 = 2, N1_1 = 8;
        static_assert(M1_0 * M1_1 == M1, "M1_0 * M1_1 must equal M1");
        static_assert(N1_0 * N1_1 == N1, "N1_0 * N1_1 must equal N1");

        constexpr auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<M0>{}),
                       make_pass_through_transform(number<N0>{}),
                       make_unmerge_transform(make_tuple(number<M1_0>{}, number<M1_1>{})),
                       make_unmerge_transform(make_tuple(number<N1_0>{}, number<N1_1>{})),
                       make_pass_through_transform(number<M2>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
            make_tuple(
                sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4, 5>{}, sequence<6>{}));
        constexpr auto desc_2 = transform_tensor_descriptor(
            desc_1,
            make_tuple(make_pass_through_transform(number<M0>{}),
                       make_pass_through_transform(number<N0>{}),
                       make_xor_transform(make_tuple(number<M1_0>{}, number<N1_0>{})),
                       make_pass_through_transform(number<M1_1>{}),
                       make_pass_through_transform(number<N1_1>{}),
                       make_pass_through_transform(number<M2>{})),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2, 4>{},
                       sequence<3>{},
                       sequence<5>{},
                       sequence<6>{}),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2, 4>{},
                       sequence<3>{},
                       sequence<5>{},
                       sequence<6>{}));

        constexpr auto top_dims = []() {
            if constexpr(Transposed)
                return make_tuple(sequence<1>{}, sequence<0>{});
            else
                return make_tuple(sequence<0>{}, sequence<1>{});
        }();
        return transform_tensor_descriptor(
            desc_2,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<M0>{}, number<M1_0>{}, number<M1_1>{}, number<M2>{})),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<N0>{}, number<N1_0>{}, number<N1_1>{}))),
            make_tuple(sequence<0, 2, 3, 6>{}, sequence<1, 4, 5>{}),
            top_dims);
    }

    template <typename T, index_t MNPerBlock, index_t KPerBlock>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLdsReadBlockDescriptor()
    {
        const auto Dwordx4Bytes = 16;
        const auto K2           = Dwordx4Bytes / sizeof(T);
        const auto K1           = WarpAlignmentBytes / Dwordx4Bytes;
        const auto K0           = KPerBlock / (K1 * K2);

        constexpr auto desc_0 = make_naive_tensor_descriptor_packed(
            make_tuple(number<K0>{}, number<MNPerBlock>{}, number<K1>{}, number<K2>{}));
        constexpr auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<K0>{}),
                       make_xor_transform(make_tuple(number<MNPerBlock>{}, number<K1>{})),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));
        return transform_tensor_descriptor(
            desc_1,
            make_tuple(make_pass_through_transform(number<MNPerBlock>{}),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<K0>{}, number<K1>{}, number<K2>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsReadBlockDescriptor()
    {
        return MakeXLdsReadBlockDescriptor<typename Problem::KDataType,
                                           Problem::BlockFmhaShape::kN0,
                                           Problem::BlockFmhaShape::kQKHeaddim>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsReadBlockDescriptor()
    {
        return MakeXLdsReadBlockDescriptor<typename Problem::VDataType,
                                           Problem::BlockFmhaShape::kN0,
                                           Problem::BlockFmhaShape::kVHeaddim>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsReadBlockDescriptor()
    {
        return MakeXLdsReadBlockDescriptor<typename Problem::QDataType,
                                           Problem::BlockFmhaShape::kM0,
                                           Problem::BlockFmhaShape::kQKHeaddim>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradLdsReadBlockDescriptor()
    {
        return MakeXLdsReadBlockDescriptor<typename Problem::OGradDataType,
                                           Problem::BlockFmhaShape::kM0,
                                           Problem::BlockFmhaShape::kQKHeaddim>();
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

        return q_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQTRegSliceBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

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

        return make_static_tile_distribution(typename InputTileDistributionTraits<
                                             decltype(qt_block_dstr_encode),
                                             typename Problem::QDataType>::TransposedDstrEncode{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSGradTRegSliceBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

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

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEDLdsReadBlockDescriptor()
    {
        using BlockGemm         = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
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

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<N0, N1>,
                                       tuple<sequence<M0, M1, M2, M3, M4>>,
                                       tuple<sequence<1, 0>, sequence<1, 0>>,
                                       tuple<sequence<1, 0>, sequence<3, 1>>,
                                       sequence<1, 1, 1>,
                                       sequence<0, 2, 4>>{});
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

        return do_block_dstr;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOGradTRegSliceBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

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
        // CK_PRINT<typename WarpGemm::BWarpDstrEncoding>();
        // CK_PRINT<decltype(dot_block_dstr_encode)>();

        return make_static_tile_distribution(
            typename InputTileDistributionTraits<
                decltype(dot_block_dstr_encode),
                typename Problem::OGradDataType>::TransposedDstrEncode{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakePTRegSliceBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

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

        return pt_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSGradRegSliceBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

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

        return make_static_tile_distribution(
            typename InputTileDistributionTraits<
                decltype(ds_block_dstr_encode),
                typename Problem::GemmDataType>::TransposedDstrEncode{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledBiasTileDistribution()
    {
        return BlockFmhaBwdPipelineDefaultPolicy::MakeShuffledBiasTileDistribution<Problem>();
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
        return sizeof(typename Problem::QDataType) *
               MakeQLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeK()
    {
        return sizeof(typename Problem::KDataType) *
               MakeKLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeLSE()
    {
        return static_cast<index_t>(max( //
            sizeof(int) * get_warp_size(),
            sizeof(typename Problem::LSEDataType) *
                MakeLSEDLdsWriteBlockDescriptor<Problem>().get_element_space_size()));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeD()
    {
        return GetSmemSizeLSE<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        return sizeof(typename Problem::VDataType) *
               MakeVLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGrad()
    {
        return sizeof(typename Problem::OGradDataType) *
               MakeOGradLdsWriteBlockDescriptor<Problem>().get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeSGrad()
    {
        return sizeof(typename Problem::GemmDataType) *
               MakeSGradLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeBias()
    {
        if constexpr(Problem::BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            return sizeof(typename Problem::BiasDataType) *
                   MakeBiasLdsBlockDescriptor<Problem>().get_element_space_size();
        else
            return 0;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_q    = GetSmemSizeQ<Problem>();
        constexpr index_t smem_size_lse  = GetSmemSizeLSE<Problem>();
        constexpr index_t smem_size_k    = GetSmemSizeK<Problem>();
        constexpr index_t smem_size_v    = GetSmemSizeV<Problem>();
        constexpr index_t smem_size_do   = GetSmemSizeOGrad<Problem>();
        constexpr index_t smem_size_d    = GetSmemSizeD<Problem>();
        constexpr index_t smem_size_ds   = GetSmemSizeSGrad<Problem>();
        constexpr index_t smem_size_bias = GetSmemSizeBias<Problem>();

        constexpr index_t smem_size_stage0 = smem_size_k + smem_size_v;
        constexpr index_t smem_size_stage1 = smem_size_q * 2 + smem_size_do * 2 +
                                             smem_size_lse * 2 + smem_size_d * 2 +
                                             max(smem_size_bias, smem_size_ds);
        return max(smem_size_stage0, smem_size_stage1);
    }

    template <typename Problem>
    class HotLoopScheduler
    {
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
        static constexpr index_t WarpGemmK =
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{});
        static constexpr index_t Gemm4MWarp =
            Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<0>{});
        static constexpr index_t Gemm4NWarp =
            Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<1>{});

        static constexpr index_t blockWarps = kBlockSize / get_warp_size();
        using GemmDataType                  = typename Problem::GemmDataType;

        // Compute
        static constexpr index_t Gemm0MFMA =
            kM0 * kN0 * kK0 / (blockWarps * WarpGemmM * WarpGemmN * WarpGemmK);
        static constexpr index_t Gemm1MFMA =
            kN0 * kVHeaddim * kM0 / (blockWarps * WarpGemmM * WarpGemmN * WarpGemmK);
        static constexpr index_t Gemm2MFMA =
            kM0 * kN0 * kK2 / (blockWarps * WarpGemmM * WarpGemmN * WarpGemmK);
        static constexpr index_t Gemm3MFMA =
            kN0 * kQKHeaddim * kM0 / (blockWarps * WarpGemmM * WarpGemmN * WarpGemmK);
        static constexpr index_t Gemm4MFMA =
            kM0 * kQKHeaddim * kN0 / (blockWarps * WarpGemmM * WarpGemmN * WarpGemmK);

        // VMEM
        static constexpr index_t Q_VMEM_READ =
            kM0 * kQKHeaddim / kBlockSize / GetAlignmentQ<Problem>();
        static constexpr index_t OGrad_VMEM_READ =
            kM0 * kVHeaddim / kBlockSize / GetAlignmentOGrad<Problem>();
        static constexpr index_t LSE_VMEM_READ = 1;
        static constexpr index_t D_VMEM_READ   = 1;

        static constexpr index_t DQ_VMEM_WRITE = kM0 * kQKHeaddim / kBlockSize; // atomic add

        // LDS Read
        static constexpr index_t OGradT_LDS_READ =
            kM0 * kVHeaddim / get_warp_size() / GetTransposedAlignmentOGrad<Problem>();
        static constexpr index_t QT_LDS_READ =
            kM0 * kQKHeaddim / get_warp_size() / GetTransposedAlignmentQ<Problem>();
        static constexpr index_t SGradT_LDS_READ_P1 =
            kM0 * kK4 / (get_warp_size() * Gemm4MWarp) / GetTransposedAlignmentX<GemmDataType>();
        static constexpr index_t SGradT_LDS_READ_P2 =
            kM0 * kN0 / (get_warp_size() * Gemm4MWarp) / GetTransposedAlignmentX<GemmDataType>() -
            SGradT_LDS_READ_P1;
        static constexpr index_t Q_LDS_READ =
            kM0 * kK0 / get_warp_size() / GetAlignmentQ<Problem>();
        static constexpr index_t LSE_LDS_READ = kM0 / (4 * 4);
        static constexpr index_t D_LDS_READ   = LSE_LDS_READ;
        static constexpr index_t OGrad_LDS_READ =
            kM0 * kK2 / kBlockSize / GetAlignmentOGrad<Problem>();

        // LDS Write
        static constexpr index_t Q_LDS_WRITE =
            kM0 * kQKHeaddim / Problem::kBlockSize / GetAlignmentQ<Problem>();
        static constexpr index_t QT_LDS_WRITE =
            kM0 * kQKHeaddim / kBlockSize / GetTransposedAlignmentQ<Problem>();
        static constexpr index_t OGrad_LDS_WRITE =
            kM0 * kVHeaddim / kBlockSize / GetAlignmentOGrad<Problem>();
        static constexpr index_t OGradT_LDS_WRITE =
            kM0 * kVHeaddim / kBlockSize / GetTransposedAlignmentOGrad<Problem>();
        static constexpr index_t SGradT_LDS_WRITE = kM0 * kN0 / kBlockSize;

        public:
        static constexpr index_t TOTAL_VMEM_READ =
            Q_VMEM_READ + OGrad_VMEM_READ + LSE_VMEM_READ + D_VMEM_READ + DQ_VMEM_WRITE;

        CK_TILE_DEVICE static constexpr void SchedulerGemm0()
        {
            // Mem: Q, LSE, OGrad, D global load, OGrad^T LDS load
            // Comp: Q x K
            constexpr index_t VMEM_READ_INST =
                Q_VMEM_READ + OGrad_VMEM_READ + LSE_VMEM_READ + D_VMEM_READ;
            constexpr index_t MFMA_INST     = Gemm0MFMA;
            constexpr index_t LDS_READ_INST = OGradT_LDS_READ + LSE_LDS_READ + D_LDS_READ;

            constexpr index_t lcm_inst = lcm(VMEM_READ_INST, MFMA_INST, LDS_READ_INST);
            static_for<0, lcm_inst, 1>{}([&](auto i) {
                if constexpr(i % (lcm_inst / VMEM_READ_INST) == 0)
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                if constexpr(i % (lcm_inst / MFMA_INST) == 0)
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                if constexpr(i % (lcm_inst / LDS_READ_INST) == 0)
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            });
        }

        CK_TILE_DEVICE static constexpr void SchedulerGemm12()
        {
            // Mem:  Q^T LDS load
            // Comp: PT x OGrad
            constexpr index_t LDS_READ_INST = QT_LDS_READ;
            constexpr index_t MFMA_INST     = Gemm1MFMA + Gemm2MFMA;

            constexpr index_t lcm_inst = lcm(MFMA_INST, LDS_READ_INST);
            static_for<0, lcm_inst, 1>{}([&](auto i) {
                if constexpr(i % (lcm_inst / MFMA_INST) == 0)
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                if constexpr(i % (lcm_inst / LDS_READ_INST) == 0)
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // VMEM read
            });
        }

        CK_TILE_DEVICE static constexpr void SchedulerGemm3()
        {
            // Mem: LSE/D LDS store, SGradT LDS store, SGrad, Q, LSE LDS load.
            // Comp: SGradT x QT
            constexpr index_t LDS_WRITE_INST = SGradT_LDS_WRITE;
            constexpr index_t LDS_READ_INST  = SGradT_LDS_READ_P1 + Q_LDS_READ;
            constexpr index_t MFMA_INST      = Gemm3MFMA;

            constexpr index_t lds_rw_inst = LDS_WRITE_INST + LDS_READ_INST;
            constexpr index_t lcm_inst    = lcm(MFMA_INST, lds_rw_inst);

            static_for<0, lcm_inst, 1>{}([&](auto i) {
                if constexpr(i % (lcm_inst / MFMA_INST) == 0)
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                if constexpr(i % (lcm_inst / lds_rw_inst) == 0)
                {
                    if constexpr(i / (lcm_inst / lds_rw_inst) < LDS_WRITE_INST)
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS Write
                    else
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS Read
                }
            });
        }

        CK_TILE_DEVICE static constexpr void SchedulerGemm4()
        {
            // Mem: SGrad, OGrad, D LDS load.
            // Comp: SGrad x KT
            constexpr index_t LDS_READ_INST = SGradT_LDS_READ_P2 + OGrad_LDS_READ;
            constexpr index_t MFMA_INST     = Gemm4MFMA;

            constexpr index_t lcm_inst = lcm(MFMA_INST, LDS_READ_INST);
            static_for<0, lcm_inst, 1>{}([&](auto i) {
                if constexpr(i % (lcm_inst / MFMA_INST) == 0)
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                if constexpr(i % (lcm_inst / LDS_READ_INST) == 0)
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            });
        }
    };
};

} // namespace ck_tile
