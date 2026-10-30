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
            # Multiple parameterizations produce gradient mismatches on gfx120x.
            "test_mem_efficient_attention_attn_mask_vs_math_ref_grads",
            "test_mem_efficient_attention_vs_math_ref_grads",
        ],
    },
}
