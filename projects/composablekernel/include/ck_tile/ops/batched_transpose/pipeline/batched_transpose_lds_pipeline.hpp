// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

template <typename Problem_, typename Policy_>
struct BatchedTransposeLdsPipeline
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using DataType = remove_cvref_t<typename Problem::DataType>;

    static constexpr index_t kBlockSize          = Problem::kBlockSize;
    static constexpr index_t kLeadSizePerBlock   = Problem::kLeadSizePerBlock;
    static constexpr index_t kSecondSizePerBlock = Problem::kSecondSizePerBlock;

    static constexpr index_t GetVectorSize() { return Policy::template GetVectorSize<Problem>(); }

    CK_TILE_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename InputTileWindow, typename OutputTileWindow>
    CK_TILE_DEVICE void operator()(const InputTileWindow& input_window,
                                   OutputTileWindow& output_window)
    {
        __shared__ char smem[GetSmemSize()];
        auto input_tile_window =
            make_tile_window(input_window, Policy::template MakeInputDistribution<Problem>());
        auto output_tile_window =
            make_tile_window(output_window, Policy::template MakeOutputDistribution<Problem>());

        DataType* p_lds_ptr              = reinterpret_cast<DataType*>(smem);
        constexpr auto in_lds_block_desc = Policy::template MakeLdsStoreBlockDescriptor<Problem>();
        auto input_lds_block =
            make_tensor_view<address_space_enum::lds>(p_lds_ptr, in_lds_block_desc);

        constexpr auto out_lds_block_desc = Policy::template MakeLdsLoadBlockDescriptor<Problem>();
        auto output_lds_block =
            make_tensor_view<address_space_enum::lds>(p_lds_ptr, out_lds_block_desc);

        auto copy_to_lds_window =
            make_tile_window(input_lds_block,
                             make_tuple(number<kSecondSizePerBlock>{}, number<kLeadSizePerBlock>{}),
                             {0, 0});
        auto load_from_lds_window =
            make_tile_window(output_lds_block,
                             make_tuple(number<kSecondSizePerBlock>{}, number<kLeadSizePerBlock>{}),
                             {0, 0},
                             Policy::template MakeLdsLoadTileDistribution<Problem>());

        auto x = load_tile(input_tile_window);

        store_tile(copy_to_lds_window, x);
        block_sync_lds();

        auto y = load_tile_transpose(load_from_lds_window);

        store_tile(output_tile_window, y);
    }
};

} // namespace ck_tile
