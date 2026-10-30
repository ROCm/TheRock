# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Known release/2.14 transformer failures found by the AIPYTORCH-1609
# cross-architecture validation.
skip_tests = {
    "gfx110": {
        "transformers": [
            # Produces non-finite fp16 output on gfx1101.
            "test_multiheadattention_fastpath_fp16_head_dim_alignment_cuda",
        ],
    },
    "gfx120": {
        "transformers": [
            # Produces non-finite fp16 output on gfx1200.
            "test_multiheadattention_fastpath_fp16_head_dim_alignment_cuda",
            # Memory-efficient attention gradient mismatches on gfx1201.
            "test_mem_efficient_attention_attn_mask_vs_math_ref_grads_batch_size_8_seq_len_q_1024_seq_len_k_8_head_dim_8_is_causal_False_dropout_p_0_0_float32_scale0_cuda_float32",
            "test_mem_efficient_attention_vs_math_ref_grads_batch_size_1_seq_len_q_1024_seq_len_k_8_head_dim_32_is_causal_False_dropout_p_0_0_float32_scale0_cuda_float32",
        ],
    },
}
