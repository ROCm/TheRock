// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace ck_tile {

// This class is used for codegen pattern matching
enum class RotaryEmbeddingEnum
{
    NONE         = 0,
    INTERLEAVED  = 1, // combine dimensions 0 & 1, 2 & 3, etc
    HALF_ROTATED = 2, // combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1, etc
};

template <RotaryEmbeddingEnum>
struct RotaryEmbeddingEnumToStr;

template <>
struct RotaryEmbeddingEnumToStr<RotaryEmbeddingEnum::NONE>
{
    static constexpr const char* name = "";
};
template <>
struct RotaryEmbeddingEnumToStr<RotaryEmbeddingEnum::INTERLEAVED>
{
    static constexpr const char* name = "inter";
};
template <>
struct RotaryEmbeddingEnumToStr<RotaryEmbeddingEnum::HALF_ROTATED>
{
    static constexpr const char* name = "half";
};

template <RotaryEmbeddingEnum RotaryEnum, typename ComputeDataType = float>
struct BlockRotaryEmbedding
{
    template <typename DistributedTensor,
              typename OtherDramBlockWindow,
              typename RotaryCosDramBlockWindow,
              typename RotarySinDramBlockWindow>
    CK_TILE_HOST_DEVICE static void apply(DistributedTensor& tile,
                                          OtherDramBlockWindow other_window,
                                          RotaryCosDramBlockWindow rotary_cos_window,
                                          RotarySinDramBlockWindow rotary_sin_window,
                                          index_t rotary_dim,
                                          index_t thread_end)
    {
        using DataType = typename remove_cvref_t<DistributedTensor>::DataType;

        if constexpr(RotaryEnum == RotaryEmbeddingEnum::INTERLEAVED)
        {
            auto rotary_cos_tile = load_tile(rotary_cos_window);
            auto rotary_sin_tile = load_tile(rotary_sin_window);

            if(thread_end <= rotary_dim)
            {
                constexpr index_t thread_buffer_size = decltype(tile.thread_buf_)::size();
                static_for<0, thread_buffer_size, 2>{}([&](auto idx) {
                    const auto left  = type_convert<ComputeDataType>(tile.thread_buf_[idx]);
                    const auto right = type_convert<ComputeDataType>(tile.thread_buf_[idx + 1]);

                    const auto cos =
                        type_convert<ComputeDataType>(rotary_cos_tile.thread_buf_[idx / 2]);
                    const auto sin =
                        type_convert<ComputeDataType>(rotary_sin_tile.thread_buf_[idx / 2]);

                    tile.thread_buf_[idx]     = type_convert<DataType>(left * cos - right * sin);
                    tile.thread_buf_[idx + 1] = type_convert<DataType>(right * cos + left * sin);
                });
            }
        }
        else if constexpr(RotaryEnum == RotaryEmbeddingEnum::HALF_ROTATED)
        {
            if(thread_end <= rotary_dim)
            {
                const bool is_left = (thread_end <= (rotary_dim / 2));

                move_tile_window(other_window, {0, is_left ? rotary_dim / 2 : -(rotary_dim / 2)});
                auto other_tile = load_tile(other_window);

                move_tile_window(rotary_cos_window, {0, is_left ? 0 : -(rotary_dim / 2)});
                auto rotary_cos_tile = load_tile(rotary_cos_window);

                move_tile_window(rotary_sin_window, {0, is_left ? 0 : -(rotary_dim / 2)});
                auto rotary_sin_tile = load_tile(rotary_sin_window);

                constexpr index_t thread_buffer_size = decltype(tile.thread_buf_)::size();
                static_for<0, thread_buffer_size, 1>{}([&](auto idx) {
                    const auto curr  = type_convert<ComputeDataType>(tile.thread_buf_[idx]);
                    const auto other = type_convert<ComputeDataType>(other_tile.thread_buf_[idx]);

                    const auto cos =
                        type_convert<ComputeDataType>(rotary_cos_tile.thread_buf_[idx]);
                    const auto sin =
                        type_convert<ComputeDataType>(rotary_sin_tile.thread_buf_[idx]);

                    tile.thread_buf_[idx] =
                        type_convert<DataType>(curr * cos + other * (is_left ? -sin : sin));
                });
            }
        }
    }
};

} // namespace ck_tile
