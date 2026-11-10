# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Known nightly transformer failures found by the AIPYTORCH-1609
# cross-version validation with PyTorch 2.15 wheels.
skip_tests = {
    "windows": {
        "transformers": [
            # Fails fake-tensor execution on each validated Windows family.
            "test_transformer_encoder_layer_fwd_fake_cuda",
        ],
    },
    "linux-gfx1201": {
        "transformers": [
            # Exceeds the nightly gradient error threshold on gfx1201.
            "test_mem_efficient_attention_attn_mask_vs_math_ref_grads_batch_size_8_seq_len_q_1024_seq_len_k_8_head_dim_8_is_causal_False_dropout_p_0_0_float32_scale0_cuda_float32",
            "test_mem_efficient_attention_vs_math_ref_grads_batch_size_1_seq_len_q_1024_seq_len_k_8_head_dim_32_is_causal_False_dropout_p_0_0_float32_scale0_cuda_float32",
        ],
    },
    "windows-gfx1201": {
        "transformers": [
            # Exceeds the nightly gradient error threshold on Windows gfx1201.
            "test_mem_efficient_attention_vs_math_ref_grads_batch_size_1_seq_len_q_1024_seq_len_k_8_head_dim_32_is_causal_False_dropout_p_0_0_float32_scale_l1_cuda_float32",
            "test_mem_efficient_attention_vs_math_ref_grads_batch_size_1_seq_len_q_1024_seq_len_k_8_head_dim_16_is_causal_False_dropout_p_0_0_float32_scale_l1_cuda_float32",
        ],
    },
}
