// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"

namespace ck_tile {

// assume that we have only 1 page-block/tensor view
template <typename TensorView>
struct TrivialPageBlockNavigator
{
    using DataType     = typename TensorView::DataType;
    using WindowOrigin = multi_index<2>;

    CK_TILE_HOST_DEVICE constexpr TrivialPageBlockNavigator(const TensorView& tensor_view_)
        : tensor_view(tensor_view_)
    {
    }

    template <typename WindowLengths>
    CK_TILE_HOST_DEVICE constexpr auto make_tile_window(const WindowLengths& window_lengths,
                                                        const WindowOrigin& window_origin) const
    {
        return make_tuple(/*block_index=*/0,
                          ck_tile::make_tile_window(tensor_view, window_lengths, window_origin));
    }

    template <typename WindowLengths, typename TileDistribution>
    CK_TILE_HOST_DEVICE constexpr auto
    make_tile_window(const WindowLengths& window_lengths,
                     const WindowOrigin& window_origin,
                     const TileDistribution& tile_distribution) const
    {
        return make_tuple(
            /*block_index=*/0,
            ck_tile::make_tile_window(
                tensor_view, window_lengths, window_origin, tile_distribution));
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE static index_t
    move_tile_window(index_t /*block_index*/,
                     TileWindow& tile_window,
                     const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step)
    {
        ck_tile::move_tile_window(tile_window, step);

        return /*block_index=*/0;
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE index_t
    move_tile_window(index_t /*block_index*/,
                     TileWindow& tile_window,
                     const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step,
                     index_t /*id*/) const
    {

        ck_tile::move_tile_window(tile_window, step);
        return 0;
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE index_t
    prefetch_table_id(index_t /*block_index*/,
                      TileWindow /*tile_window*/,
                      const typename remove_cvref_t<TileWindow>::BottomTensorIndex& /*step*/) const
    {
        return -1;
    }

    CK_TILE_HOST_DEVICE static constexpr WindowOrigin
    to_local_window_origin(const WindowOrigin& global_window_origin)
    {
        return global_window_origin;
    }

    CK_TILE_HOST_DEVICE static constexpr WindowOrigin
    to_global_window_origin(index_t /*block_index*/, const WindowOrigin& local_window_origin)
    {
        return local_window_origin;
    }

    private:
    TensorView tensor_view;
};

// default page-block navigator, assume that tensor view size is same as page-block size or smaller
// if tile window on last page-block
template <typename DataType_, index_t VirtualDim, typename TensorView>
struct PageBlockNavigator
{
    using DataType = DataType_;
    static_assert(std::is_same_v<DataType, typename TensorView::DataType>);
    static_assert(VirtualDim == 0 || VirtualDim == 1, "only support 2d tile window");
    using WindowOrigin = multi_index<2>;

    CK_TILE_HOST_DEVICE constexpr PageBlockNavigator(copy_const_t<DataType, void>* physical_blocks_,
                                                     long_index_t block_stride_,
                                                     long_index_t fixed_offset_,
                                                     const int32_t* physical_block_indices_,
                                                     index_t num_blocks_,
                                                     index_t page_block_size_,
                                                     const TensorView& complete_view_,
                                                     const TensorView& last_view_)
        : physical_blocks(reinterpret_cast<DataType*>(physical_blocks_)),
          block_stride(block_stride_),
          fixed_offset(fixed_offset_),
          physical_block_indices(physical_block_indices_),
          num_blocks(num_blocks_),
          page_block_size(page_block_size_),
          complete_view(complete_view_),
          last_view(last_view_)
    {
    }

    template <typename WindowLengths>
    CK_TILE_HOST_DEVICE auto make_tile_window(const WindowLengths& window_lengths,
                                              const WindowOrigin& window_origin) const
    {
        const index_t block_index              = get_block_index(window_origin);
        const WindowOrigin local_window_origin = to_local_window_origin(window_origin);

        auto new_tile_window =
            ck_tile::make_tile_window(is_last_block(block_index) ? last_view : complete_view,
                                      window_lengths,
                                      local_window_origin);
        new_tile_window.set_bottom_tensor_view_data_ptr(get_block_ptr(block_index));

        return make_tuple(block_index, new_tile_window);
    }

    template <typename WindowLengths, typename TileDistribution>
    CK_TILE_HOST_DEVICE auto make_tile_window(const WindowLengths& window_lengths,
                                              const WindowOrigin& window_origin,
                                              const TileDistribution& tile_distribution) const
    {
        const index_t block_index              = get_block_index(window_origin);
        const WindowOrigin local_window_origin = to_local_window_origin(window_origin);

        auto new_tile_window =
            ck_tile::make_tile_window(is_last_block(block_index) ? last_view : complete_view,
                                      window_lengths,
                                      local_window_origin,
                                      tile_distribution);
        new_tile_window.set_bottom_tensor_view_data_ptr(get_block_ptr(block_index));

        return make_tuple(block_index, new_tile_window);
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE index_t
    move_tile_window(index_t block_index,
                     TileWindow& tile_window,
                     const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step) const
    {

        ck_tile::move_tile_window(tile_window, step);

        const WindowOrigin global_window_origin =
            to_global_window_origin(block_index, tile_window.get_window_origin());
        const WindowOrigin local_window_origin = to_local_window_origin(global_window_origin);

        const index_t new_block_index = get_block_index(global_window_origin);
        /// TODO: only update necessary attributes
        tile_window.bottom_tensor_view_.desc_ =
            (is_last_block(new_block_index) ? last_view : complete_view).get_tensor_descriptor();
        tile_window.set_window_origin(local_window_origin);
        tile_window.set_bottom_tensor_view_data_ptr(get_block_ptr(new_block_index));

        return new_block_index;
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE index_t
    move_tile_window(index_t block_index,
                     TileWindow& tile_window,
                     const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step,
                     index_t id) const
    {
        ck_tile::move_tile_window(tile_window, step);

        const WindowOrigin global_window_origin =
            to_global_window_origin(block_index, tile_window.get_window_origin());
        const WindowOrigin local_window_origin = to_local_window_origin(global_window_origin);

        const index_t new_block_index = get_block_index(global_window_origin);
        /// TODO: only update necessary attributes
        tile_window.bottom_tensor_view_.desc_ =
            (is_last_block(new_block_index) ? last_view : complete_view).get_tensor_descriptor();
        tile_window.set_window_origin(local_window_origin);
        if(id >= 0)
            tile_window.set_bottom_tensor_view_data_ptr(physical_blocks + id * block_stride +
                                                        fixed_offset);
        else
            tile_window.set_bottom_tensor_view_data_ptr(nullptr);

        return new_block_index;
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE index_t
    prefetch_table_id(index_t block_index,
                      TileWindow& tile_window,
                      const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step) const
    {
        auto local_tile_window = tile_window; // not affect origin window
        ck_tile::move_tile_window(local_tile_window, step);

        const WindowOrigin global_window_origin =
            to_global_window_origin(block_index, local_tile_window.get_window_origin());
        const index_t new_block_index = get_block_index(global_window_origin);

        if(new_block_index < num_blocks)
        {
            return physical_block_indices[new_block_index];
        }
        else
        {
            return -1;
        }
    }

    CK_TILE_HOST_DEVICE bool is_last_block(index_t block_index) const
    {
        return block_index == num_blocks - 1;
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE bool is_cross_block(index_t block_index,
                                            const TileWindow& tile_window) const
    {
        const index_t origin = tile_window.get_window_origin().at(number<VirtualDim>{});
        const index_t length = tile_window.get_window_lengths().at(number<VirtualDim>{});
        return (block_index < num_blocks - 1) && (page_block_size < origin + length);
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE void
    move_to_block(index_t block_index, TileWindow& tile_window, index_t new_block_index) const
    {
        const multi_index<2> step = [&]() {
            const index_t origin_diff = (block_index - new_block_index) * page_block_size;
            if constexpr(VirtualDim == 0)
            {
                return make_multi_index(origin_diff, 0);
            }
            else
            {
                return make_multi_index(0, origin_diff);
            }
        }();

        /// TODO: only update necessary attributes
        tile_window.bottom_tensor_view_.desc_ =
            (is_last_block(new_block_index) ? last_view : complete_view).get_tensor_descriptor();
        tile_window.set_window_origin(tile_window.get_window_origin() + step);
        tile_window.set_bottom_tensor_view_data_ptr(get_block_ptr(new_block_index));
    }

    CK_TILE_HOST_DEVICE WindowOrigin
    to_local_window_origin(const WindowOrigin& global_window_origin) const
    {
        if constexpr(VirtualDim == 0)
        {
            const index_t length              = global_window_origin.at(number<0>{});
            const index_t num_complete_blocks = integer_divide_floor(length, page_block_size);
            return make_multi_index(length - page_block_size * num_complete_blocks,
                                    global_window_origin.at(number<1>{}));
        }
        else
        {
            const index_t length              = global_window_origin.at(number<1>{});
            const index_t num_complete_blocks = integer_divide_floor(length, page_block_size);
            return make_multi_index(global_window_origin.at(number<0>{}),
                                    length - page_block_size * num_complete_blocks);
        }
    }

    CK_TILE_HOST_DEVICE WindowOrigin
    to_global_window_origin(index_t block_index, const WindowOrigin& local_window_origin) const
    {
        if constexpr(VirtualDim == 0)
        {
            return make_multi_index(block_index * page_block_size +
                                        local_window_origin.at(number<0>{}),
                                    local_window_origin.at(number<1>{}));
        }
        else
        {
            return make_multi_index(local_window_origin.at(number<0>{}),
                                    block_index * page_block_size +
                                        local_window_origin.at(number<1>{}));
        }
    }

    private:
    CK_TILE_HOST_DEVICE
    DataType* get_block_ptr(index_t block_index) const
    {
        if(block_index < num_blocks)
        {
            return physical_blocks + physical_block_indices[block_index] * block_stride +
                   fixed_offset;
        }
        else
        {
            return nullptr;
        }
    }

    CK_TILE_HOST_DEVICE int32_t get_block_index(const WindowOrigin& global_window_origin) const
    {
        return integer_divide_floor(global_window_origin.at(number<VirtualDim>{}), page_block_size);
    }

    DataType* physical_blocks;
    long_index_t block_stride;
    long_index_t fixed_offset;

    const int32_t* physical_block_indices;
    index_t num_blocks;
    index_t page_block_size;

    TensorView complete_view;
    TensorView last_view;
};

template <typename TensorView>
CK_TILE_HOST_DEVICE auto make_page_block_navigator(const TensorView& tensor_view)
{
    return TrivialPageBlockNavigator<TensorView>(tensor_view);
}

template <typename DataType, index_t VirtualDim, typename TensorView>
CK_TILE_HOST_DEVICE auto make_page_block_navigator(copy_const_t<DataType, void>* physical_blocks,
                                                   long_index_t block_stride,
                                                   long_index_t fixed_offset,
                                                   const int32_t* physical_block_indices,
                                                   index_t num_blocks,
                                                   index_t page_block_size,
                                                   const TensorView& complete_view,
                                                   const TensorView& last_view)
{
    return PageBlockNavigator<DataType, VirtualDim, TensorView>(physical_blocks,
                                                                block_stride,
                                                                fixed_offset,
                                                                physical_block_indices,
                                                                num_blocks,
                                                                page_block_size,
                                                                complete_view,
                                                                last_view);
}

} // namespace ck_tile
