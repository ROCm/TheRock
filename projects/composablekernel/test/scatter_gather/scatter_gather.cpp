// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <unordered_set>

#include "ck_tile/core.hpp"

#ifndef TEST_SCATTER_GATHER_VERBOSE
#define TEST_SCATTER_GATHER_VERBOSE 1
#endif

#define HIP_CALL(call)                                                              \
    do                                                                              \
    {                                                                               \
        hipError_t err = call;                                                      \
        if(err != hipSuccess)                                                       \
        {                                                                           \
            printf("[hiperror](%d) fail to call %s", static_cast<int>(err), #call); \
            exit(0);                                                                \
        }                                                                           \
    } while(0)

/*
TODO:
This is a simple design of scatter/gather through indexing transform, with limitations
We may design a scatter/gather adaptor layer directly inside tile window
*/
template <ck_tile::index_t ROW_TILE_SIZE = 8,
          ck_tile::index_t COL_TILE_SIZE = 32 * 8,
          ck_tile::index_t BLOCK_SIZE    = 256,
          ck_tile::index_t ALIGNMENT     = 8,
          typename INDEX_BUF_TYPE        = ck_tile::index_t,
          typename DATA_TYPE             = ck_tile::fp16_t>
__global__ void row_scatter_gather(const INDEX_BUF_TYPE* src_row_idx_ptr,
                                   const INDEX_BUF_TYPE* dst_row_idx_ptr,
                                   const DATA_TYPE* src_ptr,
                                   DATA_TYPE* dst_ptr,
                                   ck_tile::index_t n_row_total,
                                   ck_tile::index_t /*n_row_select*/,
                                   ck_tile::index_t n_cols)
{
    using namespace ck_tile;

    // some constexpr vars
    constexpr index_t vec = ALIGNMENT;
    static_assert(COL_TILE_SIZE % vec == 0);
    constexpr index_t col_lanes = COL_TILE_SIZE / vec;
    constexpr index_t warp_size = ck_tile::get_warp_size();
    static_assert(warp_size % col_lanes == 0);
    constexpr index_t row_lanes = warp_size / col_lanes;
    constexpr index_t num_warps = BLOCK_SIZE / warp_size;
    static_assert(ROW_TILE_SIZE % (num_warps * row_lanes) == 0);
    constexpr index_t row_repeat = ROW_TILE_SIZE / (num_warps * row_lanes);
    static_assert(
        row_repeat == 1,
        "currently indexing not support(and would be not performant) if row_repeat has more");

    // tile partitioner
    index_t tile_col_idx = 0;
    index_t tile_row_idx = blockIdx.x * ROW_TILE_SIZE;

    // create our tild distribution, which tell us the location of different threads
    constexpr auto src_dist = make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<1>,
            tuple<sequence<row_repeat, num_warps, row_lanes>, sequence<col_lanes, vec>>,
            tuple<sequence<1>, sequence<1, 2>>,
            tuple<sequence<1>, sequence<2, 0>>,
            sequence<1, 2>,
            sequence<0, 1>>{});
    const auto coord     = src_dist.calculate_index();
    const auto row_coord = coord[number<0>{}] + tile_row_idx;

    // load the current row index from the indexing buffer. we do not use ck_tile utility here
    INDEX_BUF_TYPE src_row_id = src_row_idx_ptr[row_coord];
    INDEX_BUF_TYPE dst_row_id = dst_row_idx_ptr[row_coord];

    // printf("-- tid:%d, src_row_id:%d, dst_row_id:%d\n", static_cast<int>(threadIdx.x),
    // static_cast<int>(src_row_id), static_cast<int>(dst_row_id));

    const auto src_view =
        make_naive_tensor_view<address_space_enum::global>(src_ptr,
                                                           make_tuple(n_row_total, n_cols),
                                                           make_tuple(n_cols, 1),
                                                           number<vec>{}, // alignement
                                                           number<1>{});

    const auto src_gather_view = transform_tensor_view(
        src_view,
        make_tuple(make_indexing_transform(
                       n_row_total,
                       src_row_id), // here we replace row_idx  which is loaded from another buffer
                   make_pass_through_transform(n_cols)),
        make_tuple(sequence<0>{}, sequence<1>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    auto src_tile = make_tile_window(src_gather_view,
                                     make_tuple(number<ROW_TILE_SIZE>{}, number<COL_TILE_SIZE>{}),
                                     {tile_row_idx, tile_col_idx},
                                     src_dist);

    const auto dst_view =
        make_naive_tensor_view<address_space_enum::global>(dst_ptr,
                                                           make_tuple(n_row_total, n_cols),
                                                           make_tuple(n_cols, 1),
                                                           number<vec>{},
                                                           number<1>{});

    const auto dst_scatter_view = transform_tensor_view(
        dst_view,
        make_tuple(make_indexing_transform(
                       n_row_total,
                       dst_row_id), // here we replace row_idx  which is loaded from another buffer
                   make_pass_through_transform(n_cols)),
        make_tuple(sequence<0>{}, sequence<1>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    auto dst_tile = make_tile_window(dst_scatter_view,
                                     make_tuple(number<ROW_TILE_SIZE>{}, number<COL_TILE_SIZE>{}),
                                     {tile_row_idx, tile_col_idx},
                                     src_dist /*reuse distribution*/);

    // we finished descriptor construction and index calculation, now start load/store
    for(auto i = 0; i < n_cols; i += COL_TILE_SIZE)
    {
        // note that scatter/gather are just the same API when doing load store as normal memory
        // operation
        auto data = load_tile(src_tile);
        store_tile(dst_tile, data);

        move_tile_window(src_tile, {number<0>{}, number<COL_TILE_SIZE>{}});
        move_tile_window(dst_tile, {number<0>{}, number<COL_TILE_SIZE>{}});
    }
}

union pixel
{
    struct __attribute__((packed))
    {
        ushort r : 6;
        ushort c : 10;
    };
    ushort data;
};

struct unique_linear_rand
{
    unique_linear_rand(int capacity_) : capacity(capacity_) {}
    std::unordered_set<int> set;
    int gen()
    {
        if(static_cast<int>(set.size()) >= capacity)
        {
            printf("overflow, but will give you an number as well\n");
            return std::rand() % capacity;
        }
        while(1)
        {
            int r = std::rand() % capacity;
            if(set.count(r) == 1)
            {
                continue;
            }
            set.insert(r);
            return r;
        }
    }

    int capacity;
};

int main()
{
    int row_total  = 64;
    int row_select = 8 * 2;
    int col        = 256 * 2;
    using fp16_t   = ck_tile::fp16_t;

    constexpr int row_tile = 8;
    constexpr int col_tile = 256;

    fp16_t* src = reinterpret_cast<fp16_t*>(malloc(row_total * col * sizeof(fp16_t)));
    for(int i_r = 0; i_r < row_total; i_r++)
    {
        for(int i_c = 0; i_c < col; i_c++)
        {
            int i = i_r * col + i_c;
            pixel p;
            p.r      = i_r;
            p.c      = i_c;
            ushort d = p.data;
            src[i]   = ck_tile::bit_cast<fp16_t>(d); // for simplicity, just cast
        }
    }

    fp16_t* dst  = reinterpret_cast<fp16_t*>(malloc(row_total * col * sizeof(fp16_t)));
    int* src_idx = reinterpret_cast<int*>(malloc(row_select * sizeof(int)));
    int* dst_idx = reinterpret_cast<int*>(malloc(row_select * sizeof(int)));
    // std::srand(std::time(std::nullptr));
    // std::srand(11935);
    std::srand(std::time(nullptr));
    auto src_gen = unique_linear_rand(row_total);
    auto dst_gen = unique_linear_rand(row_total); // dst index must be unique. src is fine
    for(int i_r = 0; i_r < row_select; i_r++)
    {
        src_idx[i_r] = src_gen.gen();
        dst_idx[i_r] = dst_gen.gen();
    }

    void* dev_src;
    void* dev_dst;
    void* dev_src_idx;
    void* dev_dst_idx;
    HIP_CALL(hipMalloc(&dev_src, row_total * col * sizeof(fp16_t)));
    HIP_CALL(hipMalloc(&dev_dst, row_total * col * sizeof(fp16_t)));
    HIP_CALL(hipMalloc(&dev_src_idx, row_select * sizeof(int)));
    HIP_CALL(hipMalloc(&dev_dst_idx, row_select * sizeof(int)));

    HIP_CALL(hipMemcpy(dev_src, src, row_total * col * sizeof(fp16_t), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_src_idx, src_idx, row_select * sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_dst_idx, dst_idx, row_select * sizeof(int), hipMemcpyHostToDevice));

    constexpr int bdim = 256;
    int gdim           = (row_select + row_tile - 1) / row_tile;
    row_scatter_gather<row_tile, col_tile><<<gdim, bdim>>>(reinterpret_cast<int*>(dev_src_idx),
                                                           reinterpret_cast<int*>(dev_dst_idx),
                                                           reinterpret_cast<fp16_t*>(dev_src),
                                                           reinterpret_cast<fp16_t*>(dev_dst),
                                                           row_total,
                                                           row_select,
                                                           col);

    HIP_CALL(hipMemcpy(dst, dev_dst, row_total * col * sizeof(fp16_t), hipMemcpyDeviceToHost));

#if TEST_SCATTER_GATHER_VERBOSE
    printf("select row:");
    for(int i_r = 0; i_r < row_select; i_r++)
    {
        printf("%d->%d->%d ", i_r, src_idx[i_r], dst_idx[i_r]);
    }
    printf("\n");
#endif

    int err_cnt = 0;
    for(int i_r = 0; i_r < row_select; i_r++)
    {
        for(int i_c = 0; i_c < col; i_c++)
        {
            int i      = dst_idx[i_r] * col + i_c;
            pixel p    = ck_tile::bit_cast<pixel>(dst[i]);
            bool is_ok = p.r == src_idx[i_r] && p.c == i_c;
            if(!is_ok)
            {
                if(i_c == 0)
                    printf("(%d)pixel: %dx%d -> %d\n", i_r, p.r, p.c, dst_idx[i_r]);
                err_cnt++;
            }
        }
    }
#if TEST_SCATTER_GATHER_VERBOSE
    printf("err:%d\n", err_cnt);
#endif

    free(src);
    free(dst);
    free(src_idx);
    free(dst_idx);
    return err_cnt == 0 ? 0 : -1;
}
