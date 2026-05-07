// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          int MoeGemmKind = 0, // 0: gemm1_gate_only, 1: gemm1_gate_up, 2: gemm2, 3:gemm1_split_k
          typename ActivationOp = identity>
__global__ void moe_gemm_kernel(const ck_tile::index_t* p_sorted_token_ids_,
                                const ck_tile::index_t* p_sorted_expert_ids_,
                                const ck_tile::index_t* p_max_token_id_,
                                const ADataType* A,
                                const BDataType* B,
                                CDataType* C,
                                const AccDataType* expert_weight_ptr,
                                ck_tile::index_t Num_tokens,
                                ck_tile::index_t TokensPerBlock,
                                ck_tile::index_t TopK,
                                ck_tile::index_t M,
                                ck_tile::index_t N,
                                ck_tile::index_t K,
                                ck_tile::index_t strideA,
                                ck_tile::index_t strideB,
                                ck_tile::index_t strideC,
                                index_t scale_granularity_m,
                                index_t scale_granularity_n,
                                index_t scale_granularity_k,
                                float* scale_A_ptr,
                                float* scale_B_ptr,
                                float* expert_bias_ptr)
{
    constexpr auto is_split_k = MoeGemmKind == 3;
    int idx                   = blockIdx.x * blockDim.x + threadIdx.x;
    int problem_N             = MoeGemmKind == 1 ? N / 2 : N;
    int row                   = idx / problem_N; // Compute row index
    int col                   = idx % problem_N; // Compute column index

    index_t gather_token_id  = 0;
    index_t scatter_token_id = 0;
    index_t expert_id        = 0;

    if(row < p_max_token_id_[0])
    {
        expert_id        = p_sorted_expert_ids_[row / TokensPerBlock];
        gather_token_id  = p_sorted_token_ids_[row] & 0xff'ffff;
        scatter_token_id = p_sorted_token_ids_[row] & 0xff'ffff;
        if(gather_token_id >= Num_tokens)
        {
            return;
        }
        if(MoeGemmKind == 2)
        {
            gather_token_id = gather_token_id * TopK + (p_sorted_token_ids_[row] >> 24);
        }
        else
        {
            scatter_token_id = scatter_token_id * TopK + (p_sorted_token_ids_[row] >> 24);
        }
    }
    else
    {
        return;
    }

    if(row < M)
    {
        AccDataType acc    = 0.0;
        AccDataType acc_up = 0.0;

        AccDataType acc_temp    = 0.0;
        AccDataType acc_up_temp = 0.0;

        float scale_A    = 0;
        float scale_B    = 0;
        float scale_B_up = 0;

        index_t scale_A_stride        = (M + scale_granularity_m - 1) / scale_granularity_m;
        index_t scale_B_stride        = (N + scale_granularity_n - 1) / scale_granularity_n;
        index_t scale_B_expert_stride = scale_B_stride * K / scale_granularity_k;

        for(int k = 0; k < K; ++k)
        {
            if(k % scale_granularity_k == 0)
            {
                // update acc
                acc += acc_temp * scale_A * scale_B;
                acc_up += acc_up_temp * scale_A * scale_B_up;
                // reset acc temp
                acc_temp    = 0.0;
                acc_up_temp = 0.0;
                // update scale factors
                scale_A = scale_A_ptr[(gather_token_id / scale_granularity_m) +
                                      (k / scale_granularity_k) * scale_A_stride];
                scale_B =
                    scale_B_ptr[expert_id * scale_B_expert_stride + col / scale_granularity_n +
                                (k / scale_granularity_k) * scale_B_stride];
                if constexpr(MoeGemmKind == 1)
                    scale_B_up = scale_B_ptr[expert_id * scale_B_expert_stride +
                                             (col + problem_N) / scale_granularity_n +
                                             (k / scale_granularity_k) * scale_B_stride];
            }

            constexpr index_t packed_size_a = ck_tile::numeric_traits<ADataType>::PackedSize;
            constexpr index_t packed_size_b = ck_tile::numeric_traits<BDataType>::PackedSize;
            // Adjust indexing based on matrix layout
            int a_index = (std::is_same_v<LayoutA, tensor_layout::gemm::RowMajor>)
                              ? gather_token_id * strideA + k
                              : k * strideA + gather_token_id;

            long b_index =
                long(expert_id) * N * K +
                ((std::is_same_v<LayoutB, tensor_layout::gemm::ColumnMajor>) ? col * strideB + k
                                                                             : k * strideB + col);
            long b_index_up;
            if constexpr(MoeGemmKind == 1)
                b_index_up = long(expert_id) * N * K +
                             ((std::is_same_v<LayoutB, tensor_layout::gemm::ColumnMajor>)
                                  ? (col + problem_N) * strideB + k
                                  : k * strideB + col + problem_N);

            AccDataType v_a;
            AccDataType v_b;
            AccDataType v_b_up;
            if constexpr(std::is_same_v<ADataType, pk_int4_t>)
            {
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(A[a_index / packed_size_a]);
                if(k % 2 == 1)
                    v_a = fp32_val.hi;
                else
                    v_a = fp32_val.lo;
            }
            else if constexpr(std::is_same_v<ADataType, pk_fp4_t>)
            {
                const fp32x2_t fp32_val = pk_fp4_to_fp32x2(A[a_index / packed_size_a]);
                if(k % 2 == 1)
                    v_a = fp32_val.hi;
                else
                    v_a = fp32_val.lo;
            }
            else
            {
                v_a = ck_tile::type_convert<AccDataType>(A[a_index]);
            }
            if constexpr(std::is_same_v<BDataType, pk_int4_t>)
            {
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(B[b_index / packed_size_b]);
                if(k % 2 == 1)
                    v_b = fp32_val.hi;
                else
                    v_b = fp32_val.lo;
                if constexpr(MoeGemmKind == 1)
                {
                    const fp32x2_t fp32_val_up =
                        pk_int4_t_to_fp32x2_t(B[b_index_up / packed_size_b]);
                    if(k % 2 == 1)
                        v_b_up = fp32_val_up.hi;
                    else
                        v_b_up = fp32_val_up.lo;
                }
            }
            else if constexpr(std::is_same_v<BDataType, pk_fp4_t>)
            {
                const fp32x2_t fp32_val = pk_fp4_to_fp32x2(B[b_index / packed_size_b], 1.0f);
                if(k % 2 == 1)
                    v_b = fp32_val.hi;
                else
                    v_b = fp32_val.lo;
                if constexpr(MoeGemmKind == 1)
                {
                    const fp32x2_t fp32_val_up =
                        pk_fp4_to_fp32x2(B[b_index_up / packed_size_b], 1.0f);
                    if(k % 2 == 1)
                        v_b_up = fp32_val_up.hi;
                    else
                        v_b_up = fp32_val_up.lo;
                }
            }
            else
            {
                v_b = ck_tile::type_convert<AccDataType>(B[b_index]);
                if constexpr(MoeGemmKind == 1)
                    v_b_up = ck_tile::type_convert<AccDataType>(B[b_index_up]);
            }
            acc_temp += v_a * v_b;
            if constexpr(MoeGemmKind == 1)
                acc_up_temp += v_a * v_b_up;
        }

        acc += acc_temp * scale_A * scale_B;
        acc_up += acc_up_temp * scale_A * scale_B_up;

        float bias = 0.f, bias_up = 0.f;
        if(expert_bias_ptr != nullptr && !is_split_k)
        {
            bias = expert_bias_ptr[expert_id * N + col];
            if constexpr(MoeGemmKind == 1)
                bias_up = expert_bias_ptr[expert_id * N + col + problem_N];
        }

        int c_index = (std::is_same_v<LayoutC, tensor_layout::gemm::RowMajor>)
                          ? scatter_token_id * strideC + col
                          : col * strideC + scatter_token_id;
        if constexpr(MoeGemmKind < 2)
        {
            C[c_index] = ck_tile::type_convert<CDataType>(
                ActivationOp{}(acc + bias, MoeGemmKind == 1 ? acc_up + bias_up : 1));
        }
        else
        {
            // moe gemm2 don't use activation.
            auto weight =
                is_split_k ? ck_tile::type_convert<AccDataType>(1.0f) : expert_weight_ptr[row];
            CDataType res = ck_tile::type_convert<CDataType>((acc + bias) * weight);

            thread_buffer<CDataType, 2> add_v = 0;
            if(c_index % 2)
            {
                // result is the second value of fp16 pair.
                add_v.template get_as<CDataType>()[1] = res;
            }
            else
            {
                // result is the first value of fp16 pair.
                add_v.template get_as<CDataType>()[0] = res;
            }
            // mask last bit to make sure atomicAdd pointer is aligned of DWORD.
            atomic_add_g<CDataType, 2>(reinterpret_cast<CDataType*>(C + (c_index & 0xffff'fffe)),
                                       add_v);
        }
    }
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          int MoeGemmKind = 0, // 0: gemm1_gate_only, 1: gemm1_gate_up, 2: gemm2, 3:gemm1_split_k
          typename ActivationOp = identity>
void reference_moe_gemm_gpu(const index_t* p_sorted_token_ids_,
                            const index_t* p_sorted_expert_ids_,
                            const index_t* p_max_token_id_,
                            const ADataType* a_ptr,
                            const BDataType* b_ptr,
                            CDataType* c_ptr,
                            const AccDataType* expert_weight_ptr,
                            index_t Num_tokens,
                            index_t TokensPerBlock,
                            index_t TopK,
                            index_t M,
                            index_t N,
                            index_t K,
                            index_t stride_a,
                            index_t stride_b,
                            index_t stride_c,
                            index_t scale_granularity_m,
                            index_t scale_granularity_n,
                            index_t scale_granularity_k,
                            float* scale_A_ptr,
                            float* scale_B_ptr,
                            float* exp_bias = nullptr)
{
    int problem_N          = MoeGemmKind == 1 ? N / 2 : N;
    int totalElements      = M * problem_N;
    int numThreadsPerBlock = 256; // Common choice for threads per block
    int numBlocks          = (totalElements + numThreadsPerBlock - 1) / numThreadsPerBlock;

    moe_gemm_kernel<ADataType,
                    BDataType,
                    AccDataType,
                    CDataType,
                    LayoutA,
                    LayoutB,
                    LayoutC,
                    MoeGemmKind,
                    ActivationOp><<<numBlocks, numThreadsPerBlock>>>(p_sorted_token_ids_,
                                                                     p_sorted_expert_ids_,
                                                                     p_max_token_id_,
                                                                     a_ptr,
                                                                     b_ptr,
                                                                     c_ptr,
                                                                     expert_weight_ptr,
                                                                     Num_tokens,
                                                                     TokensPerBlock,
                                                                     TopK,
                                                                     M,
                                                                     N,
                                                                     K,
                                                                     stride_a,
                                                                     stride_b,
                                                                     stride_c,
                                                                     scale_granularity_m,
                                                                     scale_granularity_n,
                                                                     scale_granularity_k,
                                                                     scale_A_ptr,
                                                                     scale_B_ptr,
                                                                     exp_bias);

    return;
}

} // namespace ck_tile
