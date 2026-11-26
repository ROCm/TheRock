# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Known failures on the PyTorch 2.13 wheels. These are already tracked in the
# other version skip files (pytorch_2.11.py - pytorch_2.12.py) and/or generic.py,
# but those exclusions are not picked up for the 2.13 version, so they are
# mirrored here. See
# https://github.com/ROCm/TheRock/issues/5596

skip_tests = {
    "windows": {
        "transformers": [
            # Fails fake-tensor execution on each validated Windows family.
            "test_transformer_encoder_layer_fwd_fake_cuda",
        ],
    },
    "gfx1201": {
        "transformers": [
            # Exceeds the release/2.13 gradient error threshold on gfx1201.
            "test_mem_efficient_attention_vs_math_ref_grads_batch_size_1_seq_len_q_1024_seq_len_k_8_head_dim_32_is_causal_False_dropout_p_0_0_float32_scale_l1_cuda_float32",
            "test_mem_efficient_attention_vs_math_ref_grads_batch_size_1_seq_len_q_1024_seq_len_k_8_head_dim_16_is_causal_False_dropout_p_0_0_float32_scale_l1_cuda_float32",
        ],
    },
}
