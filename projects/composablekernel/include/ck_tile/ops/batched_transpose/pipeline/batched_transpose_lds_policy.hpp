// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "batched_transpose_common_policy.hpp"

namespace ck_tile {

struct BatchedTransposeLdsPolicy : public BatchedTransposeCommonPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr index_t GetSmemSize()
    {
        return integer_least_multiple(
            sizeof(typename Problem::DataType) *
                MakeLdsStoreBlockDescriptor<Problem>().get_element_space_size(),
            16);
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeOutputDistribution()
    {
        constexpr auto input_dstr = MakeLdsLoadTileDistribution<Problem>();

        using OutTileDstrEncode =
            typename OutputTileDistributionTraits<typename decltype(input_dstr)::DstrEncode,
                                                  typename Problem::DataType>::TransposedDstrEncode;
        constexpr auto block_dstr = make_static_tile_distribution(OutTileDstrEncode{});

        return block_dstr;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeLdsStoreBlockDescriptor()
    {
        constexpr index_t kLeadDimPerBlock   = Problem::kLeadSizePerBlock;
        constexpr index_t kSecondDimPerBlock = Problem::kSecondSizePerBlock;
        constexpr index_t kVectorSize        = Problem::LDSVectorSize;

        constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kSecondDimPerBlock>{},
                       number<kLeadDimPerBlock / kVectorSize>{},
                       number<kVectorSize>{}),
            make_tuple(number<kLeadDimPerBlock>{}, number<kVectorSize>{}, number<1>{}),
            number<kVectorSize>{},
            number<1>{});

        constexpr auto lds_block_desc = transform_tensor_descriptor(
            lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<kSecondDimPerBlock>{}),
                       make_merge_transform(make_tuple(number<kLeadDimPerBlock / kVectorSize>{},
                                                       number<kVectorSize>{}))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeLdsLoadBlockDescriptor()
    {
        constexpr index_t kLeadDimPerBlock   = Problem::kLeadSizePerBlock;
        constexpr index_t kSecondDimPerBlock = Problem::kSecondSizePerBlock;
        constexpr index_t kVectorSize        = Problem::LDSVectorSize;

        constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kSecondDimPerBlock>{},
                       number<kLeadDimPerBlock / kVectorSize>{},
                       number<kVectorSize>{}),
            make_tuple(number<kLeadDimPerBlock>{}, number<kVectorSize>{}, number<1>{}),
            number<kVectorSize>{},
            number<1>{});

        constexpr auto lds_block_desc = transform_tensor_descriptor(
            lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<kSecondDimPerBlock>{}),
                       make_merge_transform(make_tuple(number<kLeadDimPerBlock / kVectorSize>{},
                                                       number<kVectorSize>{}))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeLdsLoadTileDistribution()
    {
        using DataType = typename Problem::DataType;

        // Calculate block-level dimensions
        constexpr index_t kLeadIterPerWarp   = 1;
        constexpr index_t kSecondIterPerWarp = 1;
        constexpr index_t kLeadNumWarps      = Problem::kLeadNumWarps;
        constexpr index_t kSecondNumWarps    = Problem::kSecondNumWarps;

        // Calculate repetitions of base pattern
        constexpr index_t kLeadRepetitions     = Problem::kQuadNumPerLeadDim;
        constexpr index_t kSecondRepetitions   = Problem::kQuadNumPerSecondDim;
        constexpr index_t kSecondDimIterations = Problem::kIterationsInSecondDim;
        constexpr index_t kSecondDimStrSub     = kSecondRepetitions / kSecondDimIterations;

        constexpr index_t kLaneGroupSize      = 16;
        constexpr auto xdllevel_dstr_encoding = make_transposed_distr_encode<DataType,
                                                                             kLaneGroupSize,
                                                                             kSecondDimStrSub,
                                                                             kSecondDimIterations,
                                                                             kLeadRepetitions,
                                                                             1>();

        constexpr auto input_tile_encode =
            InputTileDistributionEncoding<decltype(xdllevel_dstr_encoding),
                                          kLeadIterPerWarp,
                                          kSecondIterPerWarp,
                                          kLeadNumWarps,
                                          kSecondNumWarps>();
        constexpr auto block_dstr = make_static_tile_distribution(input_tile_encode);
        return block_dstr;
    }
};

} // namespace ck_tile
