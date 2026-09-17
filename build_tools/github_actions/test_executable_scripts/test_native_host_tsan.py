#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run explicit native CPU-only test selections under host TSAN."""

import hashlib
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from host_tsan_instrumentation import (
    native_host_tsan_environment,
    parse_gtest_listed_tests,
    require_direct_clang_tsan,
    require_gtest_execution,
    require_no_gpu_nodes,
)


COMPONENT_TESTS = {
    "rocrtst": (("intercept_queue_logic_test", ()),),
    "rocshmem": (("rocshmem_envvar_test", ()),),
    "rocrand": (
        ("test_cpp_utils", ()),
        ("test_log_normal_distribution", ()),
        ("test_normal_distribution", ()),
        ("test_poisson_distribution", ()),
        ("test_rocrand_mrg31k3p_prng", ()),
        ("test_rocrand_mrg32k3a_prng", ()),
        ("test_rocrand_mt19937_octo_engine_prng", ()),
        ("test_rocrand_linkage", ()),
        (
            "test_rocrand_generator_type",
            (
                "--gtest_filter=rocrand_generator_type_tests.rocrand_generator:"
                "rocrand_generator_type_tests.generate_test",
            ),
        ),
    ),
    "hiprand": (("test_hiprand_linkage", ()),),
    "rocsparse": (("rocsparse-unit-test", ()),),
    "rocprim": tuple(
        (name, ())
        for name in (
            "test_accumulator_t",
            "test_bit_cast",
            "test_invoke_result",
            "test_no_half_operators",
            "test_rocprim_tuple",
            "test_rocprim_types",
            "test_type_traits_interface_cpp17",
            "test_type_traits_interface_cpp20",
            "test_type_traits_interface_gnupp17",
            "test_type_traits_interface_gnupp20",
            "test_radix_key_codec",
        )
    ),
    "rocthrust": tuple(
        (name, ())
        for name in (
            "address_stability.hip",
            "alignment.hip",
            "allocator_aware_policies.hip",
            "complex_various.hip",
            "decompose.hip",
            "dependencies_aware_policies.hip",
            "discard_iterator.hip",
            "is_operator_function_object.hip",
            "metaprogramming.hip",
            "min_and_max.hip",
            "mr_disjoint_pool.hip",
            "mr_new.hip",
            "mr_pool.hip",
            "mr_pool_options.hip",
            "preprocessor.hip",
            "tuple_algorithms.hip",
        )
    ),
}

FILTERED_GTEST_INVENTORIES = {
    ("rocrand", "test_rocrand_generator_type"): (
        "rocrand_generator_type_tests.generate_test",
        "rocrand_generator_type_tests.rocrand_generator",
    ),
}

EXACT_GTEST_INVENTORIES = {
    ("rocrtst", "intercept_queue_logic_test"): (
        9,
        "427e22cbc625d6c60ae4a6707dbbcbdaf2ad49b54bc788a03e1b52f5b505bf36",
    ),
    ("rocshmem", "rocshmem_envvar_test"): (
        79,
        "5f517cf84a42fca8000429fbf5a7826bcb089ae8e5e67f070103428d49f29e4e",
    ),
}

_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s+(\d+) tests?\.")


def _inventory_digest(names: list[str]) -> str:
    normalized = "".join(f"{name}\n" for name in sorted(names))
    return hashlib.sha256(normalized.encode()).hexdigest()


def main() -> int:
    try:
        component = os.environ["TEST_COMPONENT"]
        try:
            tests = COMPONENT_TESTS[component]
        except KeyError as error:
            raise RuntimeError(
                f"unsupported native host-TSAN component: {component}"
            ) from error

        require_no_gpu_nodes()
        bin_dir = Path(os.environ["THEROCK_BIN_DIR"]).resolve()
        env = native_host_tsan_environment()
        for binary_name, arguments in tests:
            executable = bin_dir / binary_name
            require_direct_clang_tsan(executable, env)
            expected_inventory = FILTERED_GTEST_INVENTORIES.get(
                (component, binary_name)
            )
            exact_inventory = EXACT_GTEST_INVENTORIES.get((component, binary_name))
            if expected_inventory is not None or exact_inventory is not None:
                listed = subprocess.run(
                    [str(executable), *arguments, "--gtest_list_tests"],
                    capture_output=True,
                    text=True,
                    env=env,
                    check=True,
                )
                actual_inventory = tuple(
                    sorted(parse_gtest_listed_tests(listed.stdout))
                )
                if (
                    expected_inventory is not None
                    and actual_inventory != expected_inventory
                ):
                    raise RuntimeError(
                        f"{component}/{binary_name} host-TSAN inventory changed: "
                        f"expected {expected_inventory}; got {actual_inventory}"
                    )
                if exact_inventory is not None:
                    expected_count, expected_sha256 = exact_inventory
                    digest = _inventory_digest(list(actual_inventory))
                    if (
                        len(actual_inventory) != expected_count
                        or digest != expected_sha256
                    ):
                        raise RuntimeError(
                            f"{component}/{binary_name} host-TSAN inventory changed: "
                            f"expected count={expected_count}, "
                            f"sha256={expected_sha256}; got "
                            f"count={len(actual_inventory)}, sha256={digest}"
                        )
            command = [str(executable), *arguments]
            logging.info("++ Exec %s", shlex.join(command))
            executed = subprocess.run(
                command,
                cwd=bin_dir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            output = executed.stdout + executed.stderr
            print(output, end="")
            require_gtest_execution(output)
            if exact_inventory is not None:
                expected_count = exact_inventory[0]
                summaries = [int(value) for value in _PASSED_RE.findall(output)]
                if summaries != [expected_count]:
                    raise RuntimeError(
                        f"unexpected pass summary for {component}/{binary_name}: "
                        f"expected [{expected_count}], got {summaries}"
                    )
        return 0
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
