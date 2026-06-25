#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Example: rocRAND test runner refactored to use BaseTestRunner.

This demonstrates the CTest + GTEST_FILTER pattern used by rocRAND.
"""

import sys
from pathlib import Path

# Add parent directory to path to import base_test_runner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base_test_runner import BaseTestRunner


class RocrandTestRunner(BaseTestRunner):
    """Test runner for rocRAND component."""

    def __init__(self):
        super().__init__(component_name="rocRAND")

        # CTest configuration
        self.parallel_jobs = 8
        self.repeat_until_pass = 3

    def get_quick_test_filters(self):
        """Define quick test patterns for rocRAND."""
        return [
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

    def build_command(self):
        """Build CTest command with optional GTEST_FILTER."""
        cmd = [
            "ctest",
            "--test-dir",
            self.get_test_directory(),
            "--output-on-failure",
            "--parallel",
            str(self.parallel_jobs),
            "--repeat",
            f"until-pass:{self.repeat_until_pass}",
        ]

        # Apply quick test filters via GTEST_FILTER environment variable
        if self.is_quick_test():
            quick_filters = self.get_quick_test_filters()
            if quick_filters:
                self.environ_vars["GTEST_FILTER"] = ":".join(quick_filters)

        return cmd


if __name__ == "__main__":
    runner = RocrandTestRunner()
    sys.exit(runner.run())
