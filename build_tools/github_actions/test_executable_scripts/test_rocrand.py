import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

SMOKE_TESTS = [
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

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/rocRAND",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "900",
    "--repeat",
    "until-pass:3",
]

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    environ_vars["GTEST_FILTER"] = ":".join(SMOKE_TESTS)

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
