// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::factory::internal {

template <ConvSignatureDescriptor auto SIGNATURE,
          typename GroupedConvTraitsType,
          typename TilePartitioner,
          typename GemmPipeline,
          typename ConvEpilogue>
struct GroupedConvolutionTileKernel
{
    static_assert(false, "Unknown Direction");
};

template <ConvSignatureDescriptor auto SIGNATURE,
          typename GroupedConvTraitsType,
          typename TilePartitioner,
          typename GemmPipeline,
          typename ConvEpilogue>
    requires ConvDirectionIsForward<SIGNATURE>
struct GroupedConvolutionTileKernel<SIGNATURE,
                                    GroupedConvTraitsType,
                                    TilePartitioner,
                                    GemmPipeline,
                                    ConvEpilogue>
{
    using Instance = ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                              TilePartitioner,
                                                              GemmPipeline,
                                                              ConvEpilogue>;
};

template <ConvSignatureDescriptor auto SIGNATURE,
          typename GroupedConvTraitsType,
          typename TilePartitioner,
          typename GemmPipeline,
          typename ConvEpilogue>
    requires ConvDirectionIsBackwardData<SIGNATURE>
struct GroupedConvolutionTileKernel<SIGNATURE,
                                    GroupedConvTraitsType,
                                    TilePartitioner,
                                    GemmPipeline,
                                    ConvEpilogue>
{
    using Instance = ck_tile::GroupedConvolutionBackwardDataKernel<GroupedConvTraitsType,
                                                                   TilePartitioner,
                                                                   GemmPipeline,
                                                                   ConvEpilogue>;
};

template <ConvSignatureDescriptor auto SIGNATURE,
          typename GroupedConvTraitsType,
          typename TilePartitioner,
          typename GemmPipeline,
          typename ConvEpilogue>
    requires ConvDirectionIsBackwardWeight<SIGNATURE>
struct GroupedConvolutionTileKernel<SIGNATURE,
                                    GroupedConvTraitsType,
                                    TilePartitioner,
                                    GemmPipeline,
                                    ConvEpilogue>
{
    using Instance = ck_tile::GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType,
                                                                     TilePartitioner,
                                                                     GemmPipeline,
                                                                     ConvEpilogue>;
};

template <ConvSignatureDescriptor auto SIGNATURE>
consteval ck_tile::GroupedConvDirection SetTileConvDirection()
{
    constexpr auto direction = SIGNATURE.direction;
    using ck_tile_direction  = ck_tile::GroupedConvDirection;
    switch(direction)
    {
    case ConvDirection::FORWARD: return ck_tile_direction::FORWARD;
    case ConvDirection::BACKWARD_DATA: return ck_tile_direction::BACKWARD_DATA;
    case ConvDirection::BACKWARD_WEIGHT: return ck_tile_direction::BACKWARD_WEIGHT;
    default: throw "Unknown Direction";
    }
}

} // namespace ck_tile::builder::factory::internal
