// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
// #include "ck_tile/ops/permute/pipeline/generic_petmute_problem.hpp"

namespace ck_tile {

/* independent host side argument, no template
 */
struct GenericPermuteHostArgs
{
    static constexpr index_t kMaxRanks = 8; // TODO: hardcoded

    const void* p_src;
    void* p_dst;
    index_t rank;
    index_t shape[kMaxRanks]; // input shape
    index_t perm[kMaxRanks];  // permute index
};

/*
simulate torch.permute:
x_ = x_.view(x.shape[0],
                    x.shape[1]//16, 16,
                    x.shape[2]//32, 4, 8)
x_ = x_.permute(0,1,3,4,2,5)
x_ = x_.contiguous()
x_ = x_.view(x.shape[0], x.shape[1], x.shape[2]);//

this kernel is supposed not to be performant(just OK), with functional support up to kMaxRanks
dim of permutation, with a single kernel

*/
template <typename Problem_>
struct GenericPermute
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;

    using DataType                      = remove_cvref_t<typename Problem::DataType>;
    static constexpr index_t kBlockSize = Problem::kBlockSize;
    static constexpr index_t kMaxRanks  = Problem::kMaxRanks;
    static constexpr bool KeepLastDim   = Problem::KeepLastDim;

    struct __attribute__((packed)) Kargs
    {
        const void* p_src;
        void* p_dst;
        // index_t rank;
        index_t num_elements;
        index_t perm_length[kMaxRanks]; // tensor length after permutation
        index_t perm_stride[kMaxRanks]; // tensor stride after permutation
    };

    CK_TILE_HOST static constexpr index_t TotalElements(const GenericPermuteHostArgs& h)
    {
        index_t n = 1;
        for(auto i = 0; i < h.rank; i++)
        {
            n *= h.shape[i];
        }
        return n;
    }

    CK_TILE_HOST static constexpr Kargs MakeKargs(const GenericPermuteHostArgs& h)
    {
        Kargs a;
        a.p_src = h.p_src;
        a.p_dst = h.p_dst;

        // assert rank <= kMaxRanks
        index_t i = 0;

        index_t perm[kMaxRanks];
        index_t x_shape[kMaxRanks];
        index_t x_stride[kMaxRanks];
        // index_t perm_length[kMaxRanks];

        for(; i < h.rank; i++)
        {
            x_shape[i] = h.shape[i];
            perm[i]    = h.perm[i];
        }
        for(; i < kMaxRanks; i++)
        {
            x_shape[i] = 1;
            perm[i]    = i; // will index to len = 1
        }

        index_t stride = 1;
        for(index_t j = kMaxRanks - 1; j >= 0; j--)
        {
            x_stride[j] = stride;
            stride *= x_shape[j];
        }

        for(index_t j = 0; j < kMaxRanks; j++)
        {
            a.perm_length[j] = x_shape[perm[j]];
            a.perm_stride[j] = x_stride[perm[j]];
        }

        a.num_elements = TotalElements(h);
        return a;
    }

    CK_TILE_HOST static constexpr auto GridSize(GenericPermuteHostArgs h)
    {
        auto total = TotalElements(h);
        auto grids = dim3((total + BlockSize() - 1) / BlockSize());
        //  printf("### total:%d, grids:%dx%dx%d\n", total, );
        return grids;
    }

    CK_TILE_HOST_DEVICE static constexpr auto BlockSize() { return Problem::kBlockSize; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        index_t id = blockIdx.x * BlockSize() + threadIdx.x;

        if(id >= kargs.num_elements)
            return;

        const auto perm_length =
            generate_tuple([&](auto I) { return kargs.perm_length[I]; }, number<kMaxRanks>{});
        const auto perm_stride =
            generate_tuple([&](auto I) { return kargs.perm_stride[I]; }, number<kMaxRanks>{});

        const DataType* p_src = reinterpret_cast<const DataType*>(kargs.p_src);
        DataType* p_dst       = reinterpret_cast<DataType*>(kargs.p_dst);

        const auto src_view_0 = make_naive_tensor_view<address_space_enum::global>(
            p_src, perm_length, perm_stride, number<1>{}, number<1>{});

        const auto src_view = transform_tensor_view(
            src_view_0,
            make_tuple(make_merge_transform(perm_length)),
            make_tuple(typename arithmetic_sequence_gen<0, kMaxRanks, 1>::type{}),
            make_tuple(sequence<0>{}));

        auto dst_view_0 = make_naive_tensor_view_packed<address_space_enum::global>(
            p_dst, perm_length, number<1>{});

        auto dst_view = transform_tensor_view(
            dst_view_0,
            make_tuple(make_merge_transform(perm_length)),
            make_tuple(typename arithmetic_sequence_gen<0, kMaxRanks, 1>::type{}),
            make_tuple(sequence<0>{}));

        // TODO: hard code to vector 1
        using vector_t = thread_buffer<DataType, 1>;

        const auto src_coord =
            make_tensor_coordinate(src_view.get_tensor_descriptor(), array<index_t, 1>{id});
        const auto dst_coord =
            make_tensor_coordinate(dst_view.get_tensor_descriptor(), array<index_t, 1>{id});

        // printf("src id:%d, os:%d\n", id, src_coord.get_offset());
        // printf("dst id:%d, os:%d\n", id, dst_coord.get_offset());

        const vector_t x = src_view.template get_vectorized_elements<vector_t>(src_coord, 0);
        dst_view.template set_vectorized_elements<vector_t>(dst_coord, 0, x);
    }
};

} // namespace ck_tile
