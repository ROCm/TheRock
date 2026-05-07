// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <set>
#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <unordered_set>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "moe_sorting_api.hpp"

template <typename IndexType>
void topid_unique_gen(
    std::vector<IndexType>& host_tensor, int tokens, int topk, int num_expert, int seed)
{
    size_t total_size = topk * tokens;
    std::srand(seed);
    std::set<IndexType> unique_set;
    IndexType current_v;
    for(size_t i = 0; i < total_size; i++)
    {
        if(i % topk == 0)
        {
            unique_set.clear();
        }
        current_v = std::rand() % num_expert;
        while(unique_set.find(current_v) != unique_set.end())
        {
            current_v = std::rand() % num_expert;
        }
        unique_set.insert(current_v);
        host_tensor[i] = current_v;
    }
}

void print_vector(std::vector<int>& data)
{
    for(const auto& x : data)
    {
        std::cout << x << ",";
    }
    std::cout << " ";
}

template <typename Tuple>
class TestCkTileMoeSorting : public ::testing::Test
{

    protected:
    using WeightType = std::tuple_element_t<0, Tuple>;
    using IndexType  = std::tuple_element_t<1, Tuple>;

    void RunSingle(int tokens,
                   int local_tokens,
                   int num_experts,
                   int topk,
                   int unit_size,
                   std::vector<int>& local_eid,
#if MOE_SORTING_FMOE_2D_BUF
                   int moe_buf_interm_dim,
                   int moe_buf_elem_bytes)
#else
                   int64_t moe_buf_size)
#endif
    {
        std::string index_prec  = get_precision_string<IndexType>();
        std::string weight_prec = get_precision_string<WeightType>();

        bool clear_inside   = true;
        int dispatch_policy = 0;

        int max_output_ids = ck_tile::integer_least_multiple(
            topk * tokens + num_experts * unit_size - topk, unit_size);

        int seed = 42; // Fixed seed for testing reproducibility

        if(topk > num_experts)
        {
            printf("topk:%d value should be smaller than, or equal to number of num_experts:%d\n",
                   topk,
                   num_experts);
            EXPECT_TRUE(false);
        }

        // if local_tokens == tokens, not local_token, but better avoid this since no meaning for
        // such case
        bool is_local_token = local_tokens >= 0 && local_tokens < tokens;

        if(local_tokens > tokens)
        {
            printf("local_tokens:%d larger than tokens:%d, invalid\n", local_tokens, tokens);
            EXPECT_TRUE(false);
        }

        bool local_expert_masking      = !local_eid.empty();
        auto local_expert_masking_host = [&]() {
            if(local_expert_masking)
            {
                // auto local_eid = args.get_int_vec("local_eid");
                ck_tile::HostTensor<IndexType> v_{{num_experts}};
                v_.SetZero();
                for(auto eid : local_eid)
                {
                    if(eid >= num_experts)
                    {
                        throw std::runtime_error(
                            "local_eid larger than number of expert, please check");
                    }
                    v_.mData[eid] = 1;
                }
                return v_;
            }
            else
                return ck_tile::HostTensor<IndexType>{{1}};
        }();

        // tokens already considered batch size
        ck_tile::HostTensor<IndexType> topk_ids_host({tokens, topk}, {topk, 1});
        ck_tile::HostTensor<WeightType> weights_host({tokens, topk}, {topk, 1});
        ck_tile::HostTensor<IndexType> sorted_ids_host({max_output_ids}, {1});
        ck_tile::HostTensor<WeightType> sorted_weights_host({max_output_ids}, {1});
        ck_tile::HostTensor<IndexType> sorted_expert_ids_host({max_output_ids / unit_size}, {1});
        // for simplicity, below buffer allocate 2 dword
        ck_tile::HostTensor<IndexType> sorted_id_cnt_host({2}, {1});
#if MOE_SORTING_FMOE_2D_BUF
        ck_tile::HostTensor<int8_t> moe_buf_host(
            {static_cast<std::size_t>(is_local_token ? local_tokens : tokens) * moe_buf_interm_dim *
             moe_buf_elem_bytes});
        auto moe_buf_bytes = moe_buf_interm_dim == 0
                                 ? static_cast<std::size_t>(0)
                                 : moe_buf_host.get_element_space_size_in_bytes();
#else
        ck_tile::HostTensor<float> moe_buf_host({moe_buf_size});
        auto moe_buf_bytes = moe_buf_size == 0 ? static_cast<std::size_t>(0)
                                               : moe_buf_host.get_element_space_size_in_bytes();
#endif

        ck_tile::FillUniformDistribution<WeightType>{-.5f, .5f}(weights_host);
#if MOE_SORTING_FMOE_2D_BUF
        ck_tile::FillUniformDistribution<int8_t>{-.5f, .5f}(moe_buf_host);
#else
        ck_tile::FillUniformDistribution<WeightType>{-.5f, .5f}(moe_buf_host);
#endif
        topid_unique_gen<IndexType>(topk_ids_host.mData, tokens, topk, num_experts, seed);

        ck_tile::DeviceMem topk_ids_dev(topk_ids_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem weights_dev(weights_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem sorted_ids_dev(sorted_ids_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem sorted_weights_dev(
            sorted_weights_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem sorted_expert_ids_dev(
            sorted_expert_ids_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem sorted_id_cnt_dev(sorted_id_cnt_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem moe_buf_dev(moe_buf_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem local_expert_masking_dev(
            local_expert_masking_host.get_element_space_size_in_bytes());

        // used for simulating dynamic_tokens for EP case
        ck_tile::DeviceMem local_tokens_dev(sizeof(ck_tile::index_t));
        if(is_local_token)
        {
            local_tokens_dev.ToDevice(&local_tokens);
        }

        topk_ids_dev.ToDevice(topk_ids_host.data());
        weights_dev.ToDevice(weights_host.data());
        if(moe_buf_bytes > 0)
        {
            moe_buf_dev.ToDevice(moe_buf_host.data());
        }
        if(local_expert_masking)
            local_expert_masking_dev.ToDevice(local_expert_masking_host.data());

        // if return zero, means no need workspace, can set moe_sorting_args.p_ws to nullptr
        ck_tile::index_t workspace_size =
            moe_sorting_get_workspace_size(tokens, num_experts, topk, dispatch_policy);
        ck_tile::DeviceMem moe_sorting_ws(workspace_size != 0 ? workspace_size : 0);
        if(workspace_size != 0 && clear_inside == false)
            moe_sorting_ws.SetZero(); // note, clear here!!!!

        moe_sorting_trait trait{
            index_prec, weight_prec, local_expert_masking, clear_inside, dispatch_policy};

        moe_sorting_args karg{topk_ids_dev.GetDeviceBuffer(),
                              weights_dev.GetDeviceBuffer(),
                              local_expert_masking ? local_expert_masking_dev.GetDeviceBuffer()
                                                   : nullptr,
                              is_local_token ? local_tokens_dev.GetDeviceBuffer() : nullptr,
                              sorted_ids_dev.GetDeviceBuffer(),
                              sorted_weights_dev.GetDeviceBuffer(),
                              sorted_expert_ids_dev.GetDeviceBuffer(),
                              sorted_id_cnt_dev.GetDeviceBuffer(),
                              moe_buf_bytes > 0 ? moe_buf_dev.GetDeviceBuffer() : nullptr,
                              workspace_size != 0 ? moe_sorting_ws.GetDeviceBuffer() : nullptr,
                              tokens,
                              unit_size,
                              num_experts,
                              topk,
#if MOE_SORTING_FMOE_2D_BUF
                              moe_buf_interm_dim,
                              moe_buf_elem_bytes
#else
                              static_cast<ck_tile::long_index_t>(moe_buf_size * sizeof(float))
#endif
        };

        ck_tile::stream_config sc{nullptr, false};

        auto ret_val = moe_sorting(trait, karg, sc);

        printf("[%s|%s|%s|%d]tokens:%d",
               index_prec.c_str(),
               weight_prec.c_str(),
               workspace_size == 0 ? "cx" : (clear_inside ? "ci" : "co"),
               dispatch_policy,
               tokens);
        if(is_local_token)
        {
            printf("(%d)", local_tokens);
        }
        printf(
            ", num_experts:%d, topk:%d, mp:%d, ", num_experts, topk, workspace_size != 0 ? 1 : 0);

        if(local_expert_masking)
        {
            printf("local_eid:");
            print_vector(local_eid);
        }

        if(moe_buf_bytes > 0)
        {
#if MOE_SORTING_FMOE_2D_BUF
            printf("moe_buf:%" PRIu64 "(%d,%d), ",
                   static_cast<uint64_t>(moe_buf_bytes),
                   moe_buf_interm_dim,
                   moe_buf_elem_bytes);
#else

            printf("moe_buf:%" PRIu64 ", ", static_cast<uint64_t>(moe_buf_bytes));
#endif
        }

        if(ret_val < 0)
        {
            printf("not supported\n");
            fflush(stdout);
            EXPECT_TRUE(false);
        }

        sorted_ids_dev.FromDevice(sorted_ids_host.data());
        sorted_weights_dev.FromDevice(sorted_weights_host.data());
        sorted_expert_ids_dev.FromDevice(sorted_expert_ids_host.data());
        sorted_id_cnt_dev.FromDevice(sorted_id_cnt_host.data());
        if(moe_buf_bytes > 0)
        {
            moe_buf_dev.FromDevice(moe_buf_host.data());
        }

        bool rtn = true;
        ck_tile::HostTensor<IndexType> sorted_ids_ref({max_output_ids}, {1});
        ck_tile::HostTensor<WeightType> sorted_weights_ref({max_output_ids}, {1});
        ck_tile::HostTensor<IndexType> sorted_expert_ids_ref({max_output_ids / unit_size}, {1});

        int32_t ref_total_tokens_post_pad = 0;
        ck_tile::reference_moe_sorting<WeightType, IndexType>(topk_ids_host,
                                                              weights_host,
                                                              local_expert_masking_host,
                                                              sorted_ids_ref,
                                                              sorted_weights_ref,
                                                              sorted_expert_ids_ref,
                                                              ref_total_tokens_post_pad,
                                                              num_experts,
                                                              unit_size,
                                                              is_local_token ? local_tokens
                                                                             : tokens,
                                                              local_expert_masking);
        printf("total_tokens_post_pad:%d(%d), ",
               ref_total_tokens_post_pad,
               sorted_id_cnt_host.mData[0]);
        if(ref_total_tokens_post_pad == sorted_id_cnt_host.mData[0])
        {
            size_t slen = ref_total_tokens_post_pad;
            rtn &= ck_tile::check_err(sorted_ids_host.slice({0}, {slen}),
                                      sorted_ids_ref.slice({0}, {slen}),
                                      std::string("OUT Error: Incorrect ids!"),
                                      1e-6,
                                      1e-6);
            rtn &= ck_tile::check_err(sorted_weights_host.slice({0}, {slen}),
                                      sorted_weights_ref.slice({0}, {slen}),
                                      std::string("OUT Error: Incorrect w!"),
                                      1e-6,
                                      1e-6);
            rtn &= ck_tile::check_err(sorted_expert_ids_host.slice({0}, {slen / unit_size}),
                                      sorted_expert_ids_ref.slice({0}, {slen / unit_size}),
                                      std::string("OUT Error: Incorrect eid!"),
                                      1e-6,
                                      1e-6);

            auto t_ = is_local_token ? local_tokens : tokens;
            bool _f = t_ == sorted_id_cnt_host.mData[1];
            rtn &= _f;
            if(!_f)
            {
                printf("not equal token buffer pad %d(%d)\n", t_, sorted_id_cnt_host.mData[1]);
            }
        }
        else
        {
            printf("(token size not equal!!)");
            rtn = false;
        }

        if(moe_buf_bytes)
        {
#if MOE_SORTING_FMOE_2D_BUF
            ck_tile::HostTensor<int8_t> moe_buf_ref({moe_buf_bytes});
#else
            ck_tile::HostTensor<WeightType> moe_buf_ref({moe_buf_size});
#endif
            rtn &= ck_tile::check_err(
                moe_buf_host, moe_buf_ref, std::string("OUT Error: Incorrect zero buf!"), 0, 0);
        }

        printf("valid:%s", rtn ? "y" : "n");
        fflush(stdout);
        if(!rtn)
            printf(", (%d)", seed);
        printf("\n");
        fflush(stdout);

        EXPECT_TRUE(rtn);
    }

    template <typename PrecisionType>
    static std::string get_precision_string()
    {
        if constexpr(std::is_same_v<PrecisionType, float>)
        {
            return "fp32";
        }
        else if(std::is_same_v<PrecisionType, ck_tile::index_t>)
        {
            return "int32";
        }
        else
        {
            throw std::runtime_error("Invalid precision.");
        }
    }
};
