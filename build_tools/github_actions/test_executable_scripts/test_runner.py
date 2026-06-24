#!/usr/bin/env python3
"""
This is a generic test runner that can test multiple components.
This works on components in rocm-libraries/rocm-systems which use test_categories.yml for test categorization.

Environment variables used:
TEST_COMPONENT: Job name of the component to test (e.g., "miopen", "rocrand", "hiprand")
    This is automatically set by the GitHub Actions workflow from the job_name field.
    The script maps these job names to actual test directory names (e.g., "miopen" -> "MIOpen")
    Defaults to "miopen" if not set.
TEST_TYPE: Test category to run - one of "quick", "standard", "comprehensive", or "full".
    Defaults to "quick". Invalid values fall back to "quick" with an error message.
AMDGPU_FAMILIES: Parsed to extract GPU architecture (e.g., "gfx1151")

The script discovers GPU-specific labels via ctest --print-labels and runs the appropriate tests for the current GPU architecture.
"""

import sys
import subprocess
import re
import os
import platform

import logging
import shlex
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
VALID_TEST_CATEGORIES = {
    "quick",
    "standard",
    "comprehensive",
    "full",
    # ffm-specific categories
    "ffm-quick",
    "ffm-standard",
    "ffm-comprehensive",
    "ffm-full",
}
# Normalize + validate TEST_TYPE once at module load so all downstream
# consumers (apply_component_overrides at import time, main() at run
# time) see the same lower-cased, validated value. `or "quick"` covers
# both unset env var and explicitly-empty env var (which is what
# GitHub Actions inputs default to when the workflow input is left
# blank). Invalid values fall back to "quick" with an error.
_raw_test_type = os.getenv("TEST_TYPE") or "quick"
TEST_TYPE = _raw_test_type.lower()
if TEST_TYPE not in VALID_TEST_CATEGORIES:
    print(
        f"ERROR: Invalid TEST_TYPE '{_raw_test_type}'. "
        f"Must be one of: {', '.join(sorted(VALID_TEST_CATEGORIES))}. "
        f"Falling back to 'quick'.",
        file=sys.stderr,
    )
    TEST_TYPE = "quick"
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

# Map job names to actual test directory names
# The job names come from TEST_COMPONENT env var (set by GitHub Actions workflow)
# and need to be mapped to the actual directory names in THEROCK_BIN_DIR
COMPONENT_DIR_MAPPING = {
    "miopen": "MIOpen",
    "rocblas": "rocblas",
    "rocsolver": "rocsolver",
    "rocrand": "rocRAND",
    "hiprand": "hipRAND",
    "rocthrust": "rocthrust",
    "rocprim": "rocprim",
    "rocwmma": "rocwmma",
    "hipcub": "hipcub",
    "hipdnn": "hipdnn",
    "hipdnn-samples": "hipdnn_samples",
    "miopen_plugin": "miopen_legacy_plugin",
    "rocsparse": "rocsparse",
    "hipsparse": "hipsparse",
    "hipsparselt": "hipsparselt",
    "rocroller": "rocroller",
    "hipblas": "hipblas",
    # Add more mappings as needed
}

# Get the test component from environment (required - no default)
test_component_job_name = os.getenv("TEST_COMPONENT")
if not test_component_job_name:
    print(
        "ERROR: TEST_COMPONENT environment variable is required but not set.",
        file=sys.stderr,
    )
    sys.exit(1)

TEST_COMPONENT = COMPONENT_DIR_MAPPING.get(
    test_component_job_name, test_component_job_name
)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)

# Components whose category label matches MULTIPLE ctest entries (e.g. rocsparse
# registers both *_full_suite and *_ffm-full_suite under the same label). For
# these we must NOT combine the ctest `--tests-information` stride with the
# gtest GTEST_TOTAL_SHARDS sharding: the two axes compound and silently drop
# ~(1 - 1/N) of the suite (only one (entry x gtest-sub-shard) pair runs per
# shard). Instead, shard purely at the gtest case level -- every shard runs all
# ctest entries and gtest splits the cases -- which yields complete, disjoint
# coverage for any number of (gtest-binary) entries. Single-entry components are
# unaffected either way, so this is safe to keep narrowly scoped.
GTEST_ONLY_SHARDING_COMPONENTS = {"rocsparse", "hipsparse"}
use_gtest_only_sharding = test_component_job_name in GTEST_ONLY_SHARDING_COMPONENTS

# CTest runs serially by default; per-GPU overrides can be added below.
# Example: if AMDGPU_FAMILIES and "gfx1153" in AMDGPU_FAMILIES: ctest_parallel_count = 4
ctest_parallel_count = 1

# CTest per-test timeout (default 2 hours, in seconds)
# There should be a timeout set from component level, but this can be used as an override
ctest_timeout_seconds = 7200

environ_vars = os.environ.copy()
# Set the GTEST env vars for Gtest based tests
# Set ROCM_PATH for tests that rely on it
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)
ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
environ_vars["ROCM_PATH"] = str(ROCM_PATH)

# Component-specific ENV VARs/PATHs applied on top of defaults.
#
# - test_dir: The default TEST_DIR for ctest is THEROCK_BIN_DIR/TEST_COMPONENT.
#   If any component needs to override the default TEST_DIR, it can use test_dir
#   by specifying the path parts relative to ROCM_PATH.
#
# - test_dir_by_type: Optional dict mapping TEST_TYPE (quick/standard/
#   comprehensive/full) -> path components relative to ROCM_PATH. When the
#   current TEST_TYPE matches a key here, this takes precedence over the
#   plain test_dir above. Used when a component installs its ctest
#   fragments under multiple subdirectories and the routing depends on
#   the test tier (e.g. rocwmma: quick/regression run from bin/rocwmma/
#   regression to preserve the per-target emulation regression entries
#   that legacy test_rocwmma.py used).
#
# - additional_env_paths: Additional paths to prepend to the existing PATH,
#   LD_LIBRARY_PATH, etc. The path parts are relative to ROCM_PATH.
#
# - env_prepend_from_therock: Same shape as additional_env_paths, but the path
#   parts are interpreted relative to THEROCK_DIR (the source/build tree
#   checkout root) rather than ROCM_PATH (the install prefix). Use this for
#   components whose tests need to load libraries straight out of the build
#   tree, e.g. rocroller.
COMPONENT_OVERRIDES = {
    # For rocprofiler-compute, we need the following additional paths:
    # - PATH=ROCM_PATH/bin:$PATH
    # - LD_LIBRARY_PATH=ROCM_PATH/lib:ROCM_PATH/lib/rocm_sysdeps/lib:$LD_LIBRARY_PATH
    "rocprofiler-compute": {
        "test_dir": ["libexec", "rocprofiler-compute"],
        "additional_env_paths": {
            "PATH": [["bin"]],
            "LD_LIBRARY_PATH": [
                ["lib"],
                ["lib", "rocm_sysdeps", "lib"],
            ],
        },
    },
    # rocwmma installs three independent CTestTestfile.cmake fragments:
    #   bin/rocwmma/             - per-target plain runs + regression_tests
    #   bin/rocwmma/smoke/       - per-target "<target> smoke" emulation
    #   bin/rocwmma/regression/  - per-target "<target> regression" emulation
    #                              + regression_tests
    # Legacy test_rocwmma.py routed TEST_TYPE=quick (and the alias
    # TEST_TYPE=regression, which the module-level validator now folds
    # back to "quick") to the regression fragment so the per-target
    # emulation regression runs (gemm/unit/dlrm) were exercised. Mirror
    # that here so swapping to test_runner.py preserves coverage. Pairs
    # with the rocm-libraries PR that tags the "<target> regression"
    # entries with the `quick` label in bin/rocwmma/regression/
    # CTestTestfile.cmake.
    # Other TEST_TYPEs (standard/comprehensive/full) fall through to the
    # default bin/rocwmma/ fragment, matching legacy behaviour.
    "rocwmma": {
        "test_dir_by_type": {
            "quick": ["bin", "rocwmma", "regression"],
        },
    },
    # rocroller's gtests link against shared libraries that live in the
    # build tree (under THEROCK_DIR/build/...), not in the install prefix,
    # so prepend those build-tree paths to LD_LIBRARY_PATH.
    "rocroller": {
        "env_prepend_from_therock": {
            "LD_LIBRARY_PATH": [
                ["build", "core", "clr", "dist", "lib"],
                ["build", "core", "clr", "dist", "lib", "llvm", "lib"],
                ["build", "math-libs", "BLAS", "rocRoller", "dist", "lib"],
                [
                    "build",
                    "math-libs",
                    "BLAS",
                    "rocRoller",
                    "dist",
                    "lib",
                    "host-math",
                    "lib",
                ],
            ],
        },
    },
}


def _prepend_env_paths(env, base_path, additional_paths_dict):
    """Prepend paths (relative to base_path) to environment variables."""
    for env_key, path_parts_list in additional_paths_dict.items():
        new_paths = [str(base_path.joinpath(*parts)) for parts in path_parts_list]
        existing_path = env.get(env_key, "")
        env[env_key] = ":".join(filter(None, new_paths + [existing_path]))


def apply_component_overrides(
    job_name, test_type, rocm_path, therock_dir, default_test_dir, env
):
    """Apply component-specific overrides for test_dir and environment variables.

    Precedence for test_dir resolution (highest -> lowest):
      1. test_dir_by_type[test_type] - TEST_TYPE-aware route (e.g. rocwmma
         quick/regression -> bin/rocwmma/regression).
      2. test_dir - fixed override (path parts relative to rocm_path),
         applied regardless of TEST_TYPE (e.g. rocprofiler-compute ->
         libexec/rocprofiler-compute).
      3. default_test_dir - THEROCK_BIN_DIR/TEST_COMPONENT.

    Environment paths:
    - 'additional_env_paths' prepends rocm_path-relative paths to env vars.
    - 'env_prepend_from_therock' prepends therock_dir-relative (build tree)
      paths to env vars. Used by components like rocroller that load shared
      libraries straight out of the build tree.
    """
    overrides = COMPONENT_OVERRIDES.get(job_name)
    if not overrides:
        return default_test_dir

    test_dir = default_test_dir
    by_type = overrides.get("test_dir_by_type") or {}
    if test_type and test_type in by_type:
        test_dir = str(rocm_path.joinpath(*by_type[test_type]))
    elif "test_dir" in overrides:
        test_dir = str(rocm_path.joinpath(*overrides["test_dir"]))

    _prepend_env_paths(env, rocm_path, overrides.get("additional_env_paths", {}))
    _prepend_env_paths(env, therock_dir, overrides.get("env_prepend_from_therock", {}))
    return test_dir


TEST_DIR = str(Path(THEROCK_BIN_DIR) / TEST_COMPONENT)
TEST_DIR = apply_component_overrides(
    test_component_job_name,
    TEST_TYPE,
    ROCM_PATH,
    THEROCK_DIR,
    TEST_DIR,
    environ_vars,
)

logging.basicConfig(level=logging.INFO)
##############################################


def find_matching_gpu_arch(gpu_arch: str, available_gpu_archs: set[str]) -> str | None:
    """
    Find the most specific GPU architecture in the set that matches the given GPU.

    Tries in order from most specific to least specific:
    # Example:
    # find_matching_gpu_arch('gfx1151', {'gfx1151', 'gfx115X', 'gfx11X'}) gives 'gfx1151'
    # find_matching_gpu_arch('gfx1151', {'gfx1150', 'gfx94X', 'gfx11X'}) gives 'gfx11X'
    - Wildcard matches (gfx115X, gfx11X, etc.)

    Returns the matching architecture string or None if no match found.
    """
    if gpu_arch in available_gpu_archs:
        return gpu_arch

    # Start matching from the end (gfx115X) and go back till the 5th character (gfx11X)
    # Return the top matching pattern
    for i in range(len(gpu_arch) - 1, 4, -1):
        pattern = gpu_arch[:i] + "X"
        if pattern in available_gpu_archs:
            return pattern

    return None


def check_available_labels():
    """
    Discover GPU architecture labels and category exclude labels from ctest --print-labels.

    Parses labels of the form:
    - ex_gpu_{gpu_arch} (e.g. ex_gpu_gfx110X, ex_gpu_gfx950)
    - {category}_exclude (e.g. quick_exclude, standard_exclude)

    Returns (gpu_archs, exclude_labels) where:
    - gpu_archs is a set of gpu_arch strings (e.g., 'gfx110X', 'gfx115X', 'gfx950')
    - exclude_labels is a set of exclude label strings (e.g., 'quick_exclude', 'standard_exclude')
    """
    test_dir = Path(TEST_DIR)
    if not test_dir.exists() or not test_dir.is_dir():
        print(f"Error: Test directory does not exist: {test_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        list_result = subprocess.run(
            ["ctest", "-N", "--test-dir", str(test_dir)],
            capture_output=True,
            text=True,
            check=True,
        )
        total_tests = sum(
            1
            for line in list_result.stdout.splitlines()
            if re.search(r"Test\s+#\d+:", line)
        )
        if total_tests == 0:
            print(
                f"Error: No tests found in {test_dir}. Cannot run test suite.",
                file=sys.stderr,
            )
            sys.exit(1)

        result = subprocess.run(
            ["ctest", "--print-labels", "--test-dir", str(test_dir)],
            capture_output=True,
            text=True,
            check=True,
        )

        gpu_archs = set()
        exclude_labels = set()
        gpu_prefix = "ex_gpu_"
        exclude_suffix = "_exclude"
        for line in result.stdout.splitlines():
            label = line.strip()
            if label.startswith(gpu_prefix):
                gpu_arch = label[len(gpu_prefix) :]
                if gpu_arch.startswith("gfx"):
                    gpu_archs.add(gpu_arch)
            elif label.endswith(exclude_suffix):
                exclude_labels.add(label)

        return gpu_archs, exclude_labels
    except subprocess.CalledProcessError as e:
        print(f"Error running ctest --print-labels: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Error: ctest not found. Make sure CMake/CTest is installed.",
            file=sys.stderr,
        )
        sys.exit(1)


def generate_resource_spec():
    """Generate a CTest resource-spec file for components that ship the
    `generate_resource_spec` helper (currently hipcub, rocthrust, rocprim).

    These components pin each test to a GPU slot via the RESOURCE_GROUPS test
    property, which CTest only honors when a resource spec file is supplied.
    Returns the resource-spec filename to pass to ctest via --resource-spec-file.
    """
    exe_dir = Path(TEST_DIR).resolve()
    exe_name = (
        "generate_resource_spec.exe"
        if platform.system() == "Windows"
        else "generate_resource_spec"
    )
    gen_exe = exe_dir / exe_name
    if not gen_exe.is_file():
        # Component does not use CTest resource allocation; nothing to do.
        return None

    # generate_resource_spec links against the HIP runtime; prepend the ROCm
    # bin/lib dirs so it resolves on every platform (Windows via PATH, Linux
    # via LD_LIBRARY_PATH / RPATH).
    gen_env = environ_vars.copy()
    gen_env["PATH"] = os.pathsep.join(
        filter(None, [str(Path(THEROCK_BIN_DIR).resolve()), gen_env.get("PATH", "")])
    )
    gen_env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [str(ROCM_PATH / "lib"), gen_env.get("LD_LIBRARY_PATH", "")])
    )

    # Write resources.json into the test dir and pass it to ctest as a bare
    # name; ctest changes into --test-dir, so it resolves to the same file.
    resource_spec_file = "resources.json"
    gen_cmd = [str(gen_exe), str(exe_dir / resource_spec_file)]
    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(gen_cmd)}")
    try:
        subprocess.run(gen_cmd, cwd=THEROCK_DIR, check=True, env=gen_env)
    except subprocess.CalledProcessError as e:
        print(
            f"Error generating CTest resource spec via {gen_exe}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    return resource_spec_file


def build_ctest_command(
    category, gpu_arch, available_gpu_archs, exclude_labels, resource_spec_file=None
):
    """
    Build the appropriate ctest command based on the category and GPU architecture.

    Returns a list of command arguments suitable for subprocess.run()
    """
    cmd = ["ctest"]

    # Collect all exclude patterns into a list so they can be combined into
    # a single -LE regex.  Multiple -LE flags are ANDed by ctest, which would
    # only exclude tests matching ALL patterns.  We need OR semantics instead.
    le_patterns = []
    include_labels = [category]

    # Exclude tests labeled with {category}_exclude if that label exists
    category_exclude_label = f"{category}_exclude"
    if category_exclude_label in exclude_labels:
        le_patterns.append(category_exclude_label)
        print(f"# Excluding tests with label: {category_exclude_label}")

    if gpu_arch.lower() in ["generic", "none", ""]:
        le_patterns.append("ex_gpu")
    else:
        # Find the appropriate GPU suite
        matching_arch = find_matching_gpu_arch(gpu_arch, available_gpu_archs)

        if matching_arch:
            gpu_label = f"ex_gpu_{matching_arch}"
            include_labels.append(gpu_label)
            print(f"# Using GPU suite label: {gpu_label}")
        else:
            le_patterns.append("ex_gpu")
            print(f"# No GPU suite found for {gpu_arch}, excluding all ex_gpu tests")

    # Add label options together for readability: -L ... -LE ...
    for label in include_labels:
        cmd.extend(["-L", label])
    if le_patterns:
        cmd.extend(["-LE", "|".join(le_patterns)])

    # Add common ctest parameters
    cmd.extend(
        [
            "--output-on-failure",
            "--parallel",
            f"{ctest_parallel_count}",
            "--timeout",
            str(ctest_timeout_seconds),
            "--test-dir",
            TEST_DIR,
            "-V",
        ]
    )

    # Shard via the ctest entry stride only when we are NOT relying on gtest
    # case-level sharding. Applying both compounds and drops tests on multi-entry
    # suites (see GTEST_ONLY_SHARDING_COMPONENTS). For gtest-only sharding, ctest
    # runs every entry and GTEST_TOTAL_SHARDS splits the cases within each.
    if not use_gtest_only_sharding:
        cmd.extend(["--tests-information", f"{SHARD_INDEX},,{TOTAL_SHARDS}"])

    # Constrain GPU tests to the available GPU slots when the component
    # provides a resource spec. Without this, RESOURCE_GROUPS properties are
    # ignored and GPU tests run unconstrained under --parallel.
    if resource_spec_file:
        cmd.extend(["--resource-spec-file", resource_spec_file])

    return cmd


def main():
    # TEST_TYPE was normalized + validated at module load.
    category = TEST_TYPE

    # Use AMDGPU_FAMILIES from environment variable, extract gfx<xxx> part
    gpu_arch = ""
    if AMDGPU_FAMILIES:
        # Extract gfx<xxx> pattern from AMDGPU_FAMILIES string
        # Pattern matches: gfx followed by alphanumeric characters (e.g., gfx1151, gfx950, gfx11X)
        match = re.search(r"gfx[0-9a-zA-Z]+", AMDGPU_FAMILIES)
        if match:
            gpu_arch = match.group(0)
        else:
            print(
                f"# Warning: Could not extract GPU architecture from AMDGPU_FAMILIES='{AMDGPU_FAMILIES}', using default '{gpu_arch}'"
            )

    print(f"# TEST_COMPONENT: {test_component_job_name} -> Test Directory: {TEST_DIR}")
    print(f"# TEST_TYPE: {TEST_TYPE} -> Category: {category}")
    print(f"# AMDGPU_FAMILIES: {AMDGPU_FAMILIES} -> GPU Architecture: {gpu_arch}")
    print()

    # Discover available labels from ctest
    print("# Discovering available test labels...")
    available_gpu_archs, exclude_labels = check_available_labels()

    if available_gpu_archs:
        print(f"# Found {len(available_gpu_archs)} GPU suite test(s)")
        print(f"# Available GPU architectures: {sorted(available_gpu_archs)}")
    else:
        print("# Warning: No GPU specific test suites available")
    if exclude_labels:
        print(f"# Found exclude labels: {sorted(exclude_labels)}")
    print()

    # Generate a CTest resource-spec file when the component provides the
    # generate_resource_spec helper. Without a spec, CTest ignores each test's
    # RESOURCE_GROUPS property and would run GPU tests unconstrained.
    resource_spec_file = generate_resource_spec()

    # Build the ctest command
    cmd = build_ctest_command(
        category, gpu_arch, available_gpu_archs, exclude_labels, resource_spec_file
    )

    print(f"# Running: {' '.join(cmd)}")
    print()

    # Execute the command
    try:
        logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
        result = subprocess.run(cmd, cwd=THEROCK_DIR, env=environ_vars, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running ctest: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
