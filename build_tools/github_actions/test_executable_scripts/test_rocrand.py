# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

QUICK_TESTS = [
    "*basic_tests*",
    "*config_dispatch_tests.*",
    "*cpp_utils_tests.*",
    "*cpp_wrapper*",
    "*distributions/*",
    "*generate_host_test/*",
    "*generate_long_long_tests/*",
    "*generate_normal_tests/*",
    "*generate_uniform_tests/*",
    "*generator_type_tests.*",
    "*kernel_lfsr113*",
    "*kernel_lfsr113_poisson/*",
    "*kernel_mrg/*",
    "*kernel_mtgp32*",
    "*kernel_mtgp32_poisson/*",
    "*kernel_philox4x32_10*",
    "*kernel_philox4x32_10_poisson/*",
    "*kernel_scrambled_sobol32*",
    "*kernel_scrambled_sobol32_poisson/*",
    "*kernel_scrambled_sobol64*",
    "*kernel_scrambled_sobol64_poisson/*",
    "*kernel_sobol32*",
    "*kernel_sobol32_poisson/*",
    "*kernel_sobol64*",
    "*kernel_sobol64_poisson/*",
    "*kernel_threefry2x32_20*",
    "*kernel_threefry2x32_20_poisson/*",
    "*kernel_threefry2x64_20*",
    "*kernel_threefry2x64_20_poisson/*",
    "*kernel_threefry4x32_20*",
    "*kernel_threefry4x32_20_poisson/*",
    "*kernel_threefry4x64_20*",
    "*kernel_threefry4x64_20_poisson/*",
    "*kernel_xorwow*",
    "*kernel_xorwow_poisson/*",
    "*lfsr113_engine_api_tests.*",
    "*lfsr113_generator/*",
    "*lfsr113_generator_prng_tests/*",
    "*linkage_tests.*",
    "*log_normal_distribution_tests.*",
    "*log_normal_tests.*",
    "*mrg/*",
    "*mrg_generator_prng_tests.*",
    "*mrg_log_normal_distribution_tests/*",
    "*mrg_normal_distribution_tests/*",
    "*mrg_prng_engine_tests/*",
    "*mrg_uniform_distribution_tests/*",
    "*mtgp32_generator/*",
    "*normal_distribution_tests.*",
    "*philox4x32_10_generator/*",
    "*philox_prng_state_tests.*",
    "*poisson_distribution_tests/*",
    "*poisson_tests.*",
    "*rocrand_generate_tests.*",
    "*rocrand_hipgraph_generate_tests.*",
    "*sobol_log_normal_distribution_tests/*",
    "*sobol_normal_distribution_tests.*",
    "*sobol_qrng_tests/*",
    "*threefry2x32_20_generator/*",
    "*threefry2x64_20_generator/*",
    "*threefry4x32_20_generator/*",
    "*threefry4x64_20_generator/*",
    "*threefry_prng_state_tests.*",
    "*xorwow_engine_type_test.*",
    "*xorwow_generator/*",
    "-*basic_tests/rocrand_basic_tests.rocrand_create_destroy_generator_test/10*",
]

# Experimental FFM tests. Copied from emulation test suite
FFM_TESTS = [
    "*normal_distribution*"
    "*poisson_distribution*"
    "*basic*"
    "*cpp_basic*"
    "*rocrand_cpp_wrapper_distributions/1.*:rocrand_cpp_wrapper_distributions/2.*:rocrand_cpp_wrapper_distributions/3.*:rocrand_cpp_wrapper_distributions/4.*:rocrand_cpp_wrapper_distributions/5.*:rocrand_cpp_wrapper_distributions/6.*:rocrand_cpp_wrapper_distributions/7.*:rocrand_cpp_wrapper_distributions/8.*:rocrand_cpp_wrapper_distributions/9.*:rocrand_cpp_wrapper_distributions/10.*:rocrand_cpp_wrapper_distributions/11.*:rocrand_cpp_wrapper_distributions/12.*:rocrand_cpp_wrapper_distributions/13.*:rocrand_cpp_wrapper_distributions/14.*:rocrand_cpp_wrapper_distributions/15.*:rocrand_cpp_wrapper_distributions/16.*:rocrand_cpp_wrapper_distributions/17.*:rocrand_cpp_wrapper_distributions/18.*:rocrand_cpp_wrapper_distributions/19.*:rocrand_cpp_wrapper_distributions/20.*:rocrand_cpp_wrapper_distributions/21.*:rocrand_cpp_wrapper_distributions/22.*:rocrand_cpp_wrapper_distributions/23.*:rocrand_cpp_wrapper_distributions/24.*:rocrand_cpp_wrapper_distributions/25.*"
    "*generate*"
    "*generate_log_normal*"
    "*generate_normal*"
    "*generate_poisson*"
    "*generate_uniform*"
    "*generator_type*"
    "*kernel_lfsr113*"
    "*kernel_mrg*"
    "*kernel_mtgp32*"
    "*kernel_philox4x32_10*"
    "*kernel_scrambled_sobol32*"
    "*kernel_scrambled_sobol64*"
    "*kernel_sobol32*"
    "*kernel_sobol64*"
    "*kernel_threefry2x32_20*"
    "*kernel_threefry2x64_20*"
    "*kernel_threefry4x32_20*"
    "*kernel_threefry4x64_20*"
    "*kernel_xorwow*"
    "*lfsr113_prng*"
    "*linkage*"
    "*mrg_prng*"
    "*mt19937_prng*"
    "*mtgp32_prng*"
    "*philox_prng*"
    "*scrambled_sobol32_qrng*"
    "*scrambled_sobol64_qrng*"
    "*sobol32_qrng*"
    "*sobol64_qrng*"
    "*threefry2x32_20_prng*"
    "*threefry2x64_20_prng*"
    "*threefry4x32_20_prng*"
    "*threefry4x64_20_prng*"
    "*xorwow_prng*"
    "*uniform_distribution*"
    "-poisson_distribution_tests/poisson_distribution_tests.histogram_compare/2:poisson_distribution_tests/poisson_distribution_tests.histogram_compare/3:poisson_distribution_tests/poisson_distribution_tests.histogram_compare/4:poisson_distribution_tests/poisson_distribution_tests.histogram_compare/5:rocrand_basic_tests/rocrand_basic_tests.rocrand_create_destroy_generator_test/10:rocrand_basic_tests/rocrand_basic_tests.rocrand_initialize_generator_test/5:rocrand_cpp_basic_tests/0.move_construction:rocrand_cpp_basic_tests/0.move_assignment:rocrand_cpp_basic_tests/3.move_construction:rocrand_cpp_basic_tests/3.move_assignment:rocrand_cpp_wrapper.rocrand_rng_ctor:rocrand_generate_tests/rocrand_generate_tests.int_test/0:rocrand_generate_log_normal_tests/rocrand_generate_log_normal_tests.float_test/0:rocrand_generate_normal_tests/rocrand_generate_normal_tests.float_test/0:rocrand_generate_poisson_tests/rocrand_generate_poisson_tests.generate_test/0:rocrand_generate_uniform_tests/rocrand_generate_uniform_tests.float_test/0:rocrand_generator_type_tests.set_stream_test:rocrand_kernel_lfsr113.rocrand:rocrand_kernel_lfsr113.rocrand_uniform_range:rocrand_kernel_lfsr113.rocrand_uniform_double_range:rocrand_kernel_mrg/0.rocrand:rocrand_kernel_mrg/0.rocrand_uniform_range:rocrand_kernel_mrg/0.rocrand_uniform_double_range:rocrand_kernel_mrg/1.rocrand_uniform_range:rocrand_kernel_mrg/1.rocrand_uniform_double_range:rocrand_kernel_mtgp32.rocrand:rocrand_kernel_philox4x32_10.rocrand_init:rocrand_kernel_scrambled_sobol32.rocrand:rocrand_kernel_scrambled_sobol64.rocrand:rocrand_kernel_scrambled_sobol64_poisson/rocrand_kernel_scrambled_sobol64_poisson.rocrand_poisson/3:rocrand_kernel_sobol32.rocrand:rocrand_kernel_sobol64.rocrand:rocrand_kernel_threefry2x32_20.rocrand:rocrand_kernel_threefry2x32_20.rocrand_uniform_range:rocrand_kernel_threefry2x32_20.rocrand_uniform_double_range:rocrand_kernel_threefry2x64_20.rocrand:rocrand_kernel_threefry2x64_20.rocrand_uniform_range:rocrand_kernel_threefry2x64_20.rocrand_uniform_double_range:rocrand_kernel_threefry4x32_20.rocrand:rocrand_kernel_threefry4x32_20.rocrand_uniform_range:rocrand_kernel_threefry4x32_20.rocrand_uniform_double_range:rocrand_kernel_threefry4x64_20.rocrand:rocrand_kernel_threefry4x64_20.rocrand_uniform_range:rocrand_kernel_threefry4x64_20.rocrand_uniform_double_range:rocrand_kernel_xorwow.rocrand_init:rocrand_lfsr113_prng_tests.uniform_uint_test:rocrand_lfsr113_prng_tests.same_seed_test:rocrand_lfsr113_prng_tests.different_seed_test:rocrand_lfsr113_prng_tests.different_seed_uint4_test:rocrand_mrg_prng_tests.mad_u64_u32_test:rocrand_mrg_prng_tests/0.uniform_float_range_test:rocrand_mrg_prng_tests/0.uniform_double_range_test:rocrand_mrg_prng_tests/0.discard_test:rocrand_mrg_prng_tests/1.uniform_float_range_test:rocrand_mrg_prng_tests/1.uniform_double_range_test:rocrand_mrg_prng_tests/1.discard_test:rocrand_mt19937_prng_tests.uniform_uint_test:rocrand_mt19937_prng_tests.same_seed_test:rocrand_mt19937_prng_tests.different_seed_test:rocrand_mt19937_prng_tests.subsequence_test:rocrand_mt19937_prng_tests.jump_ahead_test:rocrand_mt19937_prng_tests.continuity_uniform_uint_test:rocrand_mt19937_prng_tests.continuity_uniform_char_test:rocrand_mt19937_prng_tests.continuity_uniform_short_test:rocrand_mt19937_prng_tests.continuity_uniform_float_test:rocrand_mt19937_prng_tests.continuity_uniform_half_test:rocrand_mt19937_prng_tests.continuity_uniform_double_test:rocrand_mt19937_prng_tests.continuity_normal_float_test:rocrand_mt19937_prng_tests.continuity_normal_double_test:rocrand_mt19937_prng_tests.continuity_log_normal_float_test:rocrand_mt19937_prng_tests.continuity_log_normal_double_test:rocrand_mt19937_prng_tests.continuity_poisson_test:rocrand_mt19937_prng_tests.head_and_tail_normal_float_test:rocrand_mt19937_prng_tests.head_and_tail_normal_double_test:rocrand_mt19937_prng_tests.head_and_tail_log_normal_float_test:rocrand_mt19937_prng_tests.head_and_tail_log_normal_double_test:rocrand_mt19937_prng_tests.change_distribution0_test:rocrand_mt19937_prng_tests.change_distribution1_test:rocrand_mt19937_prng_tests.change_distribution2_test:rocrand_mt19937_prng_tests.change_distribution3_test:rocrand_mtgp32_prng_tests.uniform_uint_test:rocrand_philox_prng_tests.uniform_uint_test:rocrand_scrambled_sobol32_float_tests/0.uniform_test:rocrand_scrambled_sobol32_qrng_tests.discard_test:rocrand_scrambled_sobol32_qrng_continuity/rocrand_scrambled_sobol32_qrng_continuity.continuity_test/2:rocrand_scrambled_sobol32_qrng_continuity/rocrand_scrambled_sobol32_qrng_continuity.continuity_test/3:rocrand_scrambled_sobol64_float_tests/0.uniform_test:rocrand_scrambled_sobol64_qrng_offset/rocrand_scrambled_sobol64_qrng_offset.offsets_test/15:rocrand_scrambled_sobol64_qrng_continuity/rocrand_scrambled_sobol64_qrng_continuity.continuity_test/2:rocrand_scrambled_sobol64_qrng_continuity/rocrand_scrambled_sobol64_qrng_continuity.continuity_test/3:rocrand_sobol32_qrng_tests.uniform_uint_test:rocrand_sobol32_qrng_tests.discard_test:rocrand_sobol32_qrng_continuity/rocrand_sobol32_qrng_continuity.continuity_test/2:rocrand_sobol32_qrng_continuity/rocrand_sobol32_qrng_continuity.continuity_test/3:rocrand_sobol64_qrng_tests.uniform_double_test:rocrand_sobol64_qrng_continuity/rocrand_sobol64_qrng_continuity.continuity_test/2:rocrand_sobol64_qrng_continuity/rocrand_sobol64_qrng_continuity.continuity_test/3:rocrand_threefry_prng_tests.uniform_uint_test:rocrand_threefry_prng_tests.uniform_ulonglong_test:rocrand_xorwow_prng_tests.init_test:rocrand_generate_*tests/rocrand_generate_*tests.*/5:rocrand_cpp_wrapper/0.rocrand_rng_stream:rocrand_hipgraph_generate_tests/rocrand_hipgraph_generate_tests.normal_float_test/0:rocrand_generate_host_test/rocrand_generate_host_test.int_test/0:rocrand_generate_host_test/rocrand_generate_host_test.int_test/1:rocrand_generate_host_test/rocrand_generate_host_test.int_test/9:rocrand_generate_host_test/rocrand_generate_host_test.char_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.char_parity_test/9:rocrand_generate_host_test/rocrand_generate_host_test.short_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.short_parity_test/9:rocrand_generate_host_test/rocrand_generate_host_test.int_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.int_parity_test/9:rocrand_generate_host_test/rocrand_generate_host_test.uniform_half_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.uniform_half_parity_test/9:rocrand_generate_host_test/rocrand_generate_host_test.uniform_float_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.uniform_float_parity_test/9:rocrand_generate_host_test/rocrand_generate_host_test.uniform_double_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.uniform_double_parity_test/9:rocrand_generate_host_test/rocrand_generate_host_test.normal_half_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.normal_half_parity_test/9:rocrand_generate_host_test/rocrand_generate_host_test.normal_float_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.normal_float_parity_test/9:rocrand_generate_host_test/rocrand_generate_host_test.normal_double_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.normal_double_parity_test/9:rocrand_generate_host_test/rocrand_generate_host_test.log_normal_half_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.log_normal_half_parity_test/2:rocrand_generate_host_test/rocrand_generate_host_test.log_normal_half_parity_test/3:rocrand_generate_host_test/rocrand_generate_host_test.log_normal_half_parity_test/9:rocrand_generate_host_test/rocrand_generate_host_test.log_normal_float_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.log_normal_float_parity_test/9:rocrand_generate_host_test/rocrand_generate_host_test.log_normal_double_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.log_normal_double_parity_test/9:rocrand_generate_host_test/rocrand_generate_host_test.poisson_parity_test/1:rocrand_generate_host_test/rocrand_generate_host_test.poisson_parity_test/9:rocrand_config_dispatch_tests.host_matches_device:lfsr113_generator/generator_prng_tests/0.init_test:rocrand_mrg/generator_prng_tests/0.init_test:mrg_generator_prng_tests/0.uniform_float_range_test:mrg_generator_prng_tests/0.uniform_double_range_test:mrg_generator_prng_tests/1.uniform_float_range_test:mrg_generator_prng_tests/1.uniform_double_range_test:mrg_generator_prng_tests/2.uniform_float_range_test:mrg_generator_prng_tests/2.uniform_double_range_test:mrg_generator_prng_tests/3.uniform_float_range_test:mrg_generator_prng_tests/3.uniform_double_range_test:mrg_prng_engine_tests/0.discard_test:mrg_prng_engine_tests/1.discard_test:mt19937_generator/generator_prng_tests/0.init_test:mt19937_generator/generator_prng_tests/0.same_seed_test:mt19937_generator/generator_prng_tests/0.different_seed_test:mt19937_generator/generator_prng_continuity_tests/0.continuity_uniform_uint_test:mt19937_generator/generator_prng_continuity_tests/0.continuity_uniform_char_test:mt19937_generator/generator_prng_continuity_tests/0.continuity_uniform_float_test:mt19937_generator/generator_prng_continuity_tests/0.continuity_uniform_double_test:mt19937_generator/generator_prng_continuity_tests/0.continuity_normal_float_test:mt19937_generator/generator_prng_continuity_tests/0.continuity_normal_double_test:mt19937_generator/generator_prng_continuity_tests/0.continuity_log_normal_float_test:mt19937_generator/generator_prng_continuity_tests/0.continuity_log_normal_double_test:mt19937_generator/generator_prng_continuity_tests/0.continuity_poisson_test:mt19937_generator_prng_tests/0.head_and_tail_normal_float_test:mt19937_generator_prng_tests/0.head_and_tail_normal_double_test:mt19937_generator_prng_tests/0.head_and_tail_log_normal_float_test:mt19937_generator_prng_tests/0.head_and_tail_log_normal_double_test:mt19937_generator_prng_tests/0.change_distribution0_test:mt19937_generator_prng_tests/0.change_distribution1_test:mt19937_generator_prng_tests/0.change_distribution2_test:mt19937_generator_prng_tests/0.change_distribution3_test:mt19937_generator_prng_continuity_tests/0.continuity_uniform_short_test:mt19937_generator_prng_continuity_tests/0.continuity_uniform_half_test:mt19937_generator_engine_tests/0.subsequence_test:mt19937_generator_engine_tests/0.jump_ahead_test:mtgp32_generator/generator_prng_tests/0.init_test:philox4x32_10_generator/generator_prng_tests/0.uniform_uint_test:sobol_qrng_tests/sobol_qrng_tests/0.init_test:threefry2x32_20_generator/generator_prng_tests/0.uniform_uint_test:threefry2x64_20_generator/generator_prng_tests/0.uniform_uint_test:threefry4x32_20_generator/generator_prng_tests/0.uniform_uint_test:threefry4x64_20_generator/generator_prng_tests/0.uniform_uint_test:xorwow_generator/generator_prng_tests/0.init_test"
]

# Allow external consumers (e.g. FFM runners with tighter resource budgets) to
# override ctest timeout and parallelism without modifying this script. Defaults
# preserve the existing hardcoded values so TheRock's own CI is unaffected.
ctest_timeout = int(os.getenv("CTEST_TIMEOUT_OVERRIDE", "900"))
ctest_parallel = int(os.getenv("CTEST_PARALLEL_OVERRIDE", "8"))

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/rocRAND",
    "--output-on-failure",
    "--parallel",
    str(ctest_parallel),
    "--timeout",
    str(ctest_timeout),
    "--repeat",
    "until-pass:3",
]

# If quick tests are enabled, we run quick tests only.
# Otherwise, we run the normal test suite
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "quick":
    environ_vars["GTEST_FILTER"] = ":".join(QUICK_TESTS)
elif test_type == "ffm":
    environ_vars["GTEST_FILTER"] = ":".join(FFM_TESTS)

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
