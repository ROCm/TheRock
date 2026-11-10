# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# These skips are limited to failures reproduced with release/2.14 by the
# AIPYTORCH-1609 cross-architecture validation. Other refs are validated
# separately instead of inheriting version-specific exclusions.
skip_tests = {
    "windows": {
        "transformers": [
            # Fails fake-tensor execution on every validated Windows family.
            "test_transformer_encoder_layer_fwd_fake_cuda",
        ],
    },
    "linux-gfx1101": {
        "transformers": [
            # Produces non-finite fp16 output on gfx1101.
            "test_multiheadattention_fastpath_fp16_head_dim_alignment_cuda",
        ],
    },
    "linux-gfx1200": {
        "transformers": [
            # Produces non-finite fp16 output on gfx1200.
            "test_multiheadattention_fastpath_fp16_head_dim_alignment_cuda",
        ],
    },
    "linux-gfx1201": {
        "transformers": [
            # Exceeds the release/2.14 gradient error threshold on gfx1201.
            "test_mem_efficient_attention_attn_mask_vs_math_ref_grads_batch_size_8_seq_len_q_1024_seq_len_k_8_head_dim_8_is_causal_False_dropout_p_0_0_float32_scale0_cuda_float32",
            "test_mem_efficient_attention_vs_math_ref_grads_batch_size_1_seq_len_q_1024_seq_len_k_8_head_dim_32_is_causal_False_dropout_p_0_0_float32_scale0_cuda_float32",
        ],
    },
    "windows-gfx1201": {
        "transformers": [
            # Exceeds the release/2.14 gradient error threshold on Windows gfx1201.
            "test_mem_efficient_attention_vs_math_ref_grads_batch_size_1_seq_len_q_1024_seq_len_k_8_head_dim_16_is_causal_False_dropout_p_0_0_float32_scale_l1_cuda_float32",
            "test_mem_efficient_attention_vs_math_ref_grads_batch_size_1_seq_len_q_1024_seq_len_k_8_head_dim_32_is_causal_False_dropout_p_0_0_float32_scale_l1_cuda_float32",
        ],
    },
}
