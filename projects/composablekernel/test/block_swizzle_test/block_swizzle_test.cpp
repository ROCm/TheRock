// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <string>
#include <algorithm>
#include <vector>
#include <limits>
#include "simple_args.h"

simple_args_t create_arg(int argc, char** argv)
{
    simple_args_t args;
    args.insert("m", "1024", "matrix m")
        .insert("n", "1024", "matrix n")
        .insert("k", "1024", "matrix k")
        .insert("m_per_block", "128", "m_per_block")
        .insert("n_per_block", "128", "n_per_block")
        .insert("k_per_block", "32", "k_per_block")
        .insert("num_cu", "104", "num cu")
        .insert("occupancy", "2", "occupancy")
        .parse(argc, argv);
    return args;
}

namespace impl {
template <typename T>
T integer_divide_ceil(T n, T d)
{
    return (n + d - 1) / d;
}

template <typename T>
T min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
T max(T a, T b)
{
    return a > b ? a : b;
}

} // namespace impl

struct block_dispatcher_t
{
    public:
    uint32_t m_per_block;
    uint32_t n_per_block;
    uint32_t k_per_block;
    uint32_t num_cu;
    uint32_t occupancy;
    uint32_t m;
    uint32_t n;
    uint32_t k;

    //--------------------------------------

    uint32_t sk_num_blocks;
    uint32_t sk_num_big_blocks;
    uint32_t sk_total_iters;

    // uint32_t sk_num_blocks_per_tile;    // how many

    uint32_t dp_start_block_idx;
    uint32_t dp_iters_per_block;
    uint32_t dp_num_blocks;

    uint32_t k_iters_per_tile;
    uint32_t k_iters_per_big_block;
    //--------------------------------------

    static constexpr uint32_t min_k_iters_per_sk_block = 1;

    void dump()
    {
        printf("%dx%dx%d(%dx%dx%d), cu:%d, occ:%d, grids:%d, sk_num_big_blocks:%d, "
               "sk_num_blocks:%d, sk_total_iters:%d, dp_start_block_idx:%d, dp_iters_per_block:%d, "
               "dp_num_blocks:%d, k_iters_per_tile:%d, k_iters_per_big_block:%d\n",
               m,
               n,
               k,
               m_per_block,
               n_per_block,
               k_per_block,
               num_cu,
               occupancy,
               get_grid_dims_x(),
               sk_num_big_blocks,
               sk_num_blocks,
               sk_total_iters,
               dp_start_block_idx,
               dp_iters_per_block,
               dp_num_blocks,
               k_iters_per_tile,
               k_iters_per_big_block);
    }

    block_dispatcher_t(uint32_t m_per_block_,
                       uint32_t n_per_block_,
                       uint32_t k_per_block_,
                       uint32_t num_cu_,
                       uint32_t occupancy_,
                       uint32_t m_,
                       uint32_t n_,
                       uint32_t k_)
        : m_per_block(m_per_block_),
          n_per_block(n_per_block_),
          k_per_block(k_per_block_),
          num_cu(num_cu_),
          occupancy(occupancy_),
          m(m_),
          n(n_),
          k(k_)
    {
        init();
    }

    uint32_t get_grid_dims_x() { return dp_start_block_idx + dp_num_blocks; }

    uint32_t get_block_idx(uint32_t bid)
    {
        // block id is linearily allocated along sk blocks (dp blocks are fine)
        // this function will compute blockIdx.x and the linear sk block mapping
        // uint32_t block_idx = 0;
        // if(bid < sk_num_big_blocks) {
        //     uint32_t current_k_iter = bid * k_iters_per_big_block;
        //     tile_idx = current_k_iter / k_iters_per_tile;
        // }
        return bid;
    }

    uint32_t get_current_itr(uint32_t block_idx)
    {
        uint32_t current_itr = 0;
        if(block_idx < sk_num_big_blocks)
        {
            current_itr = block_idx * k_iters_per_big_block;
        }
        else if(block_idx < sk_num_blocks)
        {
            current_itr = (sk_num_big_blocks * k_iters_per_big_block) +
                          (block_idx - sk_num_big_blocks) * (k_iters_per_big_block - 1);
        }
        else if(block_idx >= dp_start_block_idx)
        {
            current_itr = sk_total_iters + (block_idx - dp_start_block_idx) * dp_iters_per_block;
        }
        return current_itr;
    }

    void get_block_itr(uint32_t block_idx, uint32_t& iter_start, uint32_t& iter_end)
    {
        if(block_idx < sk_num_big_blocks)
        {
            iter_start = block_idx * k_iters_per_big_block;
            iter_end   = iter_start + k_iters_per_big_block;
        }
        else if(block_idx < sk_num_blocks)
        {
            iter_start = (sk_num_big_blocks * k_iters_per_big_block) +
                         (block_idx - sk_num_big_blocks) * (k_iters_per_big_block - 1);
            iter_end = iter_start + (k_iters_per_big_block - 1);
        }
        else if(block_idx >= dp_start_block_idx)
        {
            iter_start = sk_total_iters + (block_idx - dp_start_block_idx) * dp_iters_per_block;
            iter_end   = iter_start + dp_iters_per_block;
        }
    }

    private:
    void init()
    {
        uint32_t num_tiles =
            impl::integer_divide_ceil(m, m_per_block) * impl::integer_divide_ceil(n, n_per_block);
        k_iters_per_tile = impl::integer_divide_ceil(k, k_per_block);

        // one cu can hold one wg at one time, from the whole chip's point of view
        // if number of wg is same as num_cu, we call it 1 dispatch
        // if number of wg is 2x num_cu, we call it 2 dispatches.
        // one dispatch can deliever wg same as num_cu (full dispatch), or less than num_cu (partial
        // dispatch)
        //
        uint32_t full_dispatches         = num_tiles / num_cu;
        uint32_t full_dispatch_tiles     = full_dispatches * num_cu;
        uint32_t partial_dispatche_tiles = num_tiles - full_dispatch_tiles;

        uint32_t sk_occupancy = occupancy;
        uint32_t dp_tiles     = full_dispatch_tiles;
        uint32_t sk_tiles     = partial_dispatche_tiles;

        if(full_dispatches < occupancy)
        {
            // in this case, we allocate all blocks as sk blocks
            // sk_occupancy = occupancy - full_dispatches;
            sk_occupancy = 1; // TODO: single occ seems better
            dp_tiles     = full_dispatch_tiles;
            sk_tiles     = partial_dispatche_tiles;
        }
        else if((occupancy > 1) && (full_dispatches % occupancy == occupancy - 1))
        {
            // e.g. occupancy = 2, full_dispatches = 3, 5, 7 ...
            //      occupancy = 3, full_dispatches = 5, 8, 11 ...
            //      occupancy = 4, full_dispatches = 7, 11 ...
            sk_occupancy = 1; // left 1 slot for sk occupancy
            dp_tiles     = full_dispatch_tiles;
            sk_tiles     = partial_dispatche_tiles;
        }
        else
        {
            // others, we reduce 1 dispatch from dp, together with partial dispatch,
            // to construct sk dispatch
            sk_occupancy = occupancy - ((full_dispatches - 1) % occupancy);
            dp_tiles     = full_dispatch_tiles - num_cu;
            sk_tiles     = partial_dispatche_tiles + num_cu;
        }

        // dp_num_blocks = dp_tiles;
        // dp_start_block_idx = num_cu * sk_occupancy;
        dp_iters_per_block = k_iters_per_tile;

        sk_total_iters = k_iters_per_tile * sk_tiles;

        // printf("num_tiles:%d, full_dispatches:%d, full_dispatch_tiles:%d,
        // partial_dispatche_tiles:%d\n",
        //         num_tiles, full_dispatches, full_dispatch_tiles, partial_dispatche_tiles);

        {
            uint32_t min_sk_tiles = (sk_tiles >= num_cu) ? num_cu : (sk_tiles + 1);
            uint32_t max_sk_tiles =
                (sk_tiles >= num_cu) ? num_cu * sk_occupancy
                                     : impl::min(num_cu, sk_total_iters / min_k_iters_per_sk_block);

            // if use dp for sk-block, how many iters do we need
            uint32_t dp_for_sk_iters = k_iters_per_tile;

            uint32_t best_sk_score =
                std::numeric_limits<int>::max(); // we need to find the smallest sk iters
            for(uint32_t tentative_sk_blocks = min_sk_tiles; tentative_sk_blocks < max_sk_tiles;
                tentative_sk_blocks++)
            {
                uint32_t tentative_sk_iters_per_block =
                    (sk_total_iters + tentative_sk_blocks - 1) / tentative_sk_blocks;
                uint32_t tentative_sk_iters = tentative_sk_iters_per_block;
                uint32_t sk_blocks_per_tile = (tentative_sk_blocks + sk_tiles - 1) / sk_tiles;

                // TODO: carefully adjust this parameter
                //       the more sk_blocks_per_tile, the worse the overhead
                uint32_t cross_sk_blocks_overhead = sk_blocks_per_tile;
                if(tentative_sk_blocks % sk_tiles != 0)
                {
                    // penalty for uneven divide
                    cross_sk_blocks_overhead +=
                        sk_blocks_per_tile * tentative_sk_iters_per_block / 50;
                }

                uint32_t tentative_sk_score = tentative_sk_iters + cross_sk_blocks_overhead;

                if(tentative_sk_score < best_sk_score)
                {
                    best_sk_score = tentative_sk_score;
                    sk_num_blocks = tentative_sk_blocks;
                }
            }

            if(best_sk_score >= dp_for_sk_iters)
            {
                sk_num_blocks = 0;
            }

            if(sk_num_blocks == 0)
            {
                sk_num_big_blocks     = 0;
                k_iters_per_big_block = 0;

                dp_num_blocks      = num_tiles; // all tile to be dp block
                dp_start_block_idx = 0;
                sk_total_iters     = 0; // clear this tiles
            }
            else
            {
                uint32_t k_iters_per_sk_block = sk_total_iters / sk_num_blocks;
                sk_num_big_blocks     = sk_total_iters - k_iters_per_sk_block * sk_num_blocks;
                k_iters_per_big_block = k_iters_per_sk_block + 1;

                dp_num_blocks      = dp_tiles;
                dp_start_block_idx = (sk_num_blocks + num_cu - 1) / num_cu * num_cu;
            }
        }
    }
};

struct tile_work_t
{
    uint32_t tile_idx;
    uint32_t iter_begin;
    uint32_t k_begin;
    uint32_t k_end;
    uint32_t k_iters_remaining;
};

int main(int argc, char** argv)
{
    simple_args_t arg = create_arg(argc, argv);
    block_dispatcher_t block_dispatcher{arg.get_uint32("m_per_block"),
                                        arg.get_uint32("n_per_block"),
                                        arg.get_uint32("k_per_block"),
                                        arg.get_uint32("num_cu"),
                                        arg.get_uint32("occupancy"),
                                        arg.get_uint32("m"),
                                        arg.get_uint32("n"),
                                        arg.get_uint32("k")};
    block_dispatcher.dump();
    // simulate actual kernel launch
    uint32_t dim_x = block_dispatcher.get_grid_dims_x();
    uint32_t total_k_iters =
        impl::integer_divide_ceil(arg.get_uint32("k"), arg.get_uint32("k_per_block"));
    uint32_t num_tiles =
        impl::integer_divide_ceil(arg.get_uint32("m"), arg.get_uint32("m_per_block")) *
        impl::integer_divide_ceil(arg.get_uint32("n"), arg.get_uint32("n_per_block"));

    std::vector<int> valid_tile_record(num_tiles * total_k_iters);

    for(uint32_t bid = 0; bid < dim_x; bid++)
    {
        uint32_t block_idx = block_dispatcher.get_block_idx(bid);
        bool is_sk_block   = block_idx < (block_dispatcher.sk_num_blocks);
        bool is_dp_block   = block_idx >= block_dispatcher.dp_start_block_idx;
        uint32_t iter_start, iter_end;
        block_dispatcher.get_block_itr(block_idx, iter_start, iter_end);
        uint32_t total_iter_length = iter_end - iter_start;

        while(true)
        {
            uint32_t iter_length_mod = iter_end % block_dispatcher.k_iters_per_tile;
            uint32_t current_iter_length =
                impl::min(iter_length_mod == 0 ? (iter_end - iter_start) : iter_length_mod,
                          total_iter_length);
            uint32_t tile_idx = (iter_end - 1) / block_dispatcher.k_iters_per_tile;
            uint32_t tile_iter_start =
                ((iter_end - 1) % block_dispatcher.k_iters_per_tile) - current_iter_length + 1;

            if(is_sk_block)
            {
                printf("[sk_block] bid:%3d, block_idx:%3d, tile_idx:%3d, iter_start:%d(%d | %d), "
                       "iter_end:%d (len:%d)\n",
                       bid,
                       block_idx,
                       tile_idx,
                       iter_end - current_iter_length,
                       tile_iter_start,
                       iter_start,
                       iter_end,
                       current_iter_length);
            }
            else if(is_dp_block)
            {
                printf("[dp_block] bid:%3d, block_idx:%3d, tile_idx:%3d, iter_start:%d(%d | %d), "
                       "iter_end:%d (len:%d)\n",
                       bid,
                       block_idx,
                       tile_idx,
                       iter_end - current_iter_length,
                       tile_iter_start,
                       iter_start,
                       iter_end,
                       current_iter_length);
            }
            else
            {
                printf("[other   ] bid:%3d, block_idx:%3d\n", bid, block_idx);
            }

            // some validation check
            for(auto i = iter_end - current_iter_length; i < iter_end; i++)
            {
                if(i >= valid_tile_record.size())
                {
                    printf("unexpected, current iter:%d larger than max:%d\n",
                           i,
                           valid_tile_record.size());
                    return -1;
                }
                valid_tile_record[i] = 1;
            }

            iter_end -= current_iter_length;
            if(iter_end <= iter_start)
                break;
        }
    }

    int untouched = 0;
    for(auto i = 0; i < valid_tile_record.size(); i++)
    {
        if(valid_tile_record[i] != 1)
        {
            printf("untouched at %d (%d)\n", i, valid_tile_record.size());
            untouched++;
        }
    }
    printf("untouched %d/%d, %s\n",
           untouched,
           valid_tile_record.size(),
           untouched == 0 ? "valid" : "fail");
}
