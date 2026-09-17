#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Require host-TSAN runtime linkage for an admitted component's ELF tests."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


COMPONENT_INVENTORIES = {
    "hip-tests": {
        "executables": tuple(
            f"share/hip/catch_tests/{name}"
            for name in (
                "VectorTypesTest",
                "ErrorHandlingTest",
                "ChannelDescriptorTest",
                "ComplexTest",
            )
        )
    },
    # hipFile's installed CTest discovery script replaces LD_LIBRARY_PATH and
    # hides compiler-rt. The dedicated runner freezes and executes the exact
    # GoogleTest inventory directly, so audit that same ELF directly here.
    "hipfile": {"executables": ("share/hipfile/test/internal_tests",)},
    "origami": {
        "ctest_dir": "bin/origami",
        "ctest_args": ("-R", "^origami-tests$"),
    },
    "aqlprofile": {
        "executables": tuple(
            f"share/hsa-amd-aqlprofile/tests/host-tsan/bin/{name}"
            for name in (
                "gfx9-memory-manager-test",
                "aqlprofile-test",
                "command-buffer-test",
                "counters-test",
                "pm4-factory-test",
                "logger-test",
                "aql-profile-v2-test",
                "aql-profile-v2-c-compatibility-test",
                "command-builder-test",
                "pmc-builder-test",
                "gfx9-command-builder-test",
                "spm-builder-test",
                "trace-config-test",
                "sqtt-builder-test",
                "utility_tests",
            )
        )
    },
    # The component runner builds its small parser harness after this global
    # verification step and verifies that executable itself. Verify the
    # installed instrumented library here.
    "rocdecode": {"libraries": ("lib/librocdecode.so",)},
    "rocjpeg": {"libraries": ("lib/librocjpeg.so",)},
    # The test runner builds and directly audits the two exact host test
    # executables. Verify the installed library before that runtime build.
    "rpp": {"libraries": ("lib/librpp.so",)},
    # The bin/rocgdb entry is a shell launcher. Audit every concrete Python and
    # Pythonless ELF variant because the launcher selects among them at runtime.
    "rocgdb-cpu": {"executable_globs": ("bin/rocgdb-py*",)},
    "rocprofiler-compute": {
        "executables": (
            "libexec/rocprofiler-compute/tests/test-rocprofiler-compute-tool",
            "libexec/rocprofiler-compute/tests/test-pc-sampling-collector",
        )
    },
    "rocprofiler-sdk": {
        "executables": (
            "share/rocprofiler-sdk/tests/unit-tests/bin/common-tests",
            "share/rocprofiler-sdk/tests/unit-tests/bin/codeobj-library-tests",
            "share/rocprofiler-sdk/tests/unit-tests/bin/parser-test",
        )
    },
    "rocprofiler-systems": {
        "executables": (
            "share/rocprofiler-systems/tests/unit-tests/bin/rocprof-sys-unit-tests",
        )
    },
    "rccl": {
        "executables": tuple(
            f"bin/{name}"
            for name in (
                "rccl-UnitTestsMicro",
                "rccl-UnitTestsMicroEnqueue",
                "rccl-UnitTestsMicroInit",
                "rccl-UnitTestsMicroInit-faultinj",
                "rccl-UnitTestsMicroInit-uncached",
                "rccl-UnitTestsNetTelemetry",
            )
        )
    },
    "rocrtst": {"executables": ("bin/intercept_queue_logic_test",)},
    "rocshmem": {"executables": ("bin/rocshmem_envvar_test",)},
    "rocroller": {"executables": ("bin/rocroller-tests",)},
    "rocrand": {
        "executables": tuple(
            f"bin/{name}"
            for name in (
                "test_cpp_utils",
                "test_log_normal_distribution",
                "test_normal_distribution",
                "test_poisson_distribution",
                "test_rocrand_mrg31k3p_prng",
                "test_rocrand_mrg32k3a_prng",
                "test_rocrand_mt19937_octo_engine_prng",
                "test_rocrand_linkage",
                "test_rocrand_generator_type",
            )
        )
    },
    "hiprand": {"executables": ("bin/test_hiprand_linkage",)},
    "rocsparse": {"executables": ("bin/rocsparse-unit-test",)},
    "rocprim": {
        "executables": tuple(
            f"bin/{name}"
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
        )
    },
    "rocthrust": {
        "executables": tuple(
            f"bin/{name}"
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
        )
    },
}


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) == b"\x7fELF"
    except OSError:
        return False


def discover_elf_executables(rocm_root: Path, component: str) -> list[Path]:
    try:
        inventory = COMPONENT_INVENTORIES[component]
    except KeyError as error:
        raise ValueError(f"no host-TSAN linkage inventory for {component}") from error

    candidates: set[Path] = set()
    ctest_dir = inventory.get("ctest_dir")
    if ctest_dir:
        test_dir = rocm_root / ctest_dir
        result = subprocess.run(
            [
                "ctest",
                "--test-dir",
                str(test_dir),
                "--show-only=json-v1",
                *inventory.get("ctest_args", ()),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for test in json.loads(result.stdout).get("tests", []):
            command = test.get("command", [])
            if not command:
                continue
            path = Path(command[0])
            if not path.is_absolute():
                path = test_dir / path
            if path.is_file() and os.access(path, os.X_OK) and is_elf(path):
                candidates.add(path.resolve())

    for relative in inventory.get("executables", ()):
        path = rocm_root / relative
        if not path.is_file():
            raise RuntimeError(
                f"required host-TSAN linkage executable is missing: {path}"
            )
        if not os.access(path, os.X_OK):
            raise RuntimeError(
                f"required host-TSAN linkage executable is not executable: {path}"
            )
        if not is_elf(path):
            raise RuntimeError(
                f"required host-TSAN linkage executable is not ELF: {path}"
            )
        candidates.add(path.resolve())
    for pattern in inventory.get("executable_globs", ()):
        for path in rocm_root.glob(pattern):
            if path.is_file() and os.access(path, os.X_OK) and is_elf(path):
                candidates.add(path.resolve())
    for relative in inventory.get("libraries", ()):
        path = rocm_root / relative
        if path.is_file() and is_elf(path):
            candidates.add(path.resolve())
    return sorted(candidates)


def verify_linkage(rocm_root: Path, component: str) -> list[Path]:
    readelf = rocm_root / "llvm" / "bin" / "llvm-readelf"
    if not os.access(readelf, os.X_OK):
        raise RuntimeError(f"artifact llvm-readelf not found: {readelf}")

    executables = discover_elf_executables(rocm_root, component)
    if not executables:
        raise RuntimeError(f"no ELF host-TSAN linkage targets found for {component}")

    uninstrumented = []
    for executable in executables:
        result = subprocess.run(
            [str(readelf), "--dynamic", str(executable)],
            check=True,
            capture_output=True,
            text=True,
        )
        if "libclang_rt.tsan" not in result.stdout:
            uninstrumented.append(executable)

    if uninstrumented:
        rendered = "\n  ".join(str(path) for path in uninstrumented)
        raise RuntimeError(
            "host-TSAN linkage targets lack a direct libclang_rt.tsan dependency:\n"
            f"  {rendered}"
        )
    return executables


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm-root", type=Path, required=True)
    parser.add_argument(
        "--component", choices=sorted(COMPONENT_INVENTORIES), required=True
    )
    args = parser.parse_args(argv)

    executables = verify_linkage(args.rocm_root.resolve(), args.component)
    print(
        f"Verified direct host-TSAN linkage for {len(executables)} "
        f"{args.component} ELF linkage target(s):"
    )
    for executable in executables:
        print(f"  {executable}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
