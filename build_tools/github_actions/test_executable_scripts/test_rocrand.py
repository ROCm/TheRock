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

# Resolve the ctest test-dir. simulator_runner.py overrides this via env so
# the simulator can run the same component out of a non-standard layout; on
# the real-GPU lane the env vars are unset and we fall back to the historical
# `<THEROCK_BIN_DIR>/rocRAND` path.
ctest_dir = os.getenv("SIMULATOR_CTEST_DIR") or f"{THEROCK_BIN_DIR}/rocRAND"

cmd = [
    "ctest",
    "--test-dir",
    ctest_dir,
    "--output-on-failure",
    "--parallel",
    "8",
]

# Simulator override: narrow ctest to the binaries that actually carry tests
# matching the preset's GTEST_FILTER. Without this the 51 rocRAND binaries
# that don't host any `basic` gtest still run, exit 0 with zero tests, and
# silently inflate ctest's pass count (see Rocjitsu_005 LastTest.log). Empty
# / unset / ".*" preserves today's behavior.
include_regex = os.getenv("SIMULATOR_CTEST_INCLUDE_REGEX", "").strip()
if include_regex and include_regex != ".*":
    cmd.extend(["-R", include_regex])

# Simulator override: drop --repeat until-pass:3. Flake-retry was useful on
# real GPUs but under the deterministic simulator it can only hide bugs by
# masking the symptom. Unset / "0" / "false" => preserve historical retries.
no_retry = os.getenv("SIMULATOR_NO_RETRY", "").strip().lower()
if no_retry not in ("1", "true", "yes", "on"):
    cmd.extend(["--repeat", "until-pass:3"])

environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")
# Only apply QUICK_TESTS if the caller hasn't already pinned GTEST_FILTER.
# This lets simulator_runner.py keep its own preset+skip list intact when it
# invokes us, while preserving today's on-device behavior (where GTEST_FILTER
# is never set in the inherited environment).
if test_type == "quick" and not environ_vars.get("GTEST_FILTER"):
    environ_vars["GTEST_FILTER"] = ":".join(QUICK_TESTS)

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
