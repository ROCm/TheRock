#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Generic pytest test runner for pytest-based components using test_categories.yaml.

This is the pytest analog of test_runner.py (which drives ctest). It implements
the same four-tier test-filter model (quick / standard / comprehensive / full)
for components whose tests are Python/pytest rather than gtest/ctest.

Categories are defined in a test_categories.yaml shipped next to the installed
tests. Each category selects a set of test directories (``test_paths``) and an
optional pytest marker expression (``pytest_markers`` / ``exclude_markers``).
GPU-architecture exclusions are applied via ``skip-gfxXXXX`` markers. A category
may also supply ``pytest_args``: a list of extra pytest CLI options appended
verbatim to the invocation (e.g. ``-k``, or component-specific conftest options).
``{ROCM_PATH}`` tokens in those values are substituted with the install prefix.

Environment variables used:
TEST_COMPONENT: Job name of the component to test (e.g. "tensilelite").
TEST_TYPE: Test category to run; must be a category defined in
    test_categories.yaml (e.g. quick, standard, comprehensive, full). Defaults to
    "quick" when unset.
AMDGPU_FAMILIES: GPU architecture for skip-marker filtering (e.g. "gfx942").
THEROCK_BIN_DIR: Path to the installed bin/ directory; its parent is the ROCm
    install prefix used to locate share/, lib/ and llvm tooling.
JUNIT_XML_DIR: Optional. When set, pytest writes a JUnit XML report to
    {JUNIT_XML_DIR}/{TEST_COMPONENT}.xml.
"""

import logging
import os
import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Map job name -> install location, relative to the ROCm prefix (THEROCK_BIN_DIR
# parent). Components live under share/ rather than bin/ because they are Python
# packages + pytest modules, not native test executables.
INSTALLED_COMPONENTS = {
    "tensilelite": "share/hipblaslt/tensilelite",
}


def get_env_value(name, default=None):
    return os.getenv(name, default)


def get_env_int_override(name):
    """Return a positive int from env var `name`, or 0 if unset/blank/invalid.

    Used for optional overrides (per-test timeout, xdist worker count) that fall
    back to the YAML value when the env var is absent or not a positive integer.
    """
    raw = os.getenv(name)
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        logging.warning(f"Ignoring non-integer {name}={raw!r}")
        return 0
    if value < 0:
        logging.warning(f"Ignoring negative {name}={raw!r}")
        return 0
    return value


def resolve_component_path(component_name, rocm_path):
    """Return the install directory that holds the component's tests + YAML."""
    rel = INSTALLED_COMPONENTS.get(component_name)
    if rel is None:
        logging.error(
            f"Unknown pytest component '{component_name}'. "
            f"Known components: {sorted(INSTALLED_COMPONENTS)}"
        )
        sys.exit(1)
    return (rocm_path / rel).resolve()


def load_test_categories_yaml(yaml_path):
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded test categories from {yaml_path}")
        return config
    except FileNotFoundError:
        logging.error(f"test_categories.yaml not found at {yaml_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Invalid YAML syntax in {yaml_path}: {e}")
        sys.exit(1)


def extract_gpu_arch(amdgpu_families):
    """Extract the first gfx architecture token from AMDGPU_FAMILIES."""
    if not amdgpu_families:
        return None
    match = re.search(r"gfx\w+", amdgpu_families)
    if match:
        gpu_arch = match.group(0)
        logging.info(f"Detected GPU architecture: {gpu_arch}")
        return gpu_arch
    logging.warning(f"Could not parse GPU architecture from: {amdgpu_families}")
    return None


def build_marker_expression(category_config, gpu_arch):
    """
    Build a pytest -m expression from a category's marker config plus the
    GPU-arch skip markers registered in pytest.ini (skip-gfxXXXX).
    """
    include = category_config.get("pytest_markers", []) or []
    exclude = category_config.get("exclude_markers", []) or []

    terms = []
    if include:
        terms.append("(" + " or ".join(include) + ")")
    for marker in exclude:
        terms.append(f"not {marker}")

    # GPU-arch specific skip: exclude tests marked skip-gfxNNNN for this arch.
    # pytest.ini registers exact-arch skip markers (skip-gfx942, skip-gfx1151, ...),
    # so we emit the exact name only. (Wildcard family markers are not registered
    # and would never match.)
    if gpu_arch:
        terms.append(f"not skip-{gpu_arch}")

    marker_expr = " and ".join(terms)
    logging.info(f"Marker expression: {marker_expr or '(none)'}")
    return marker_expr


def run_pytest(
    test_paths, marker_expr, extra_args, junit_xml, timeout, num_workers, cwd, env
):
    existing = [p for p in test_paths if (cwd / p).exists()]
    missing = [p for p in test_paths if not (cwd / p).exists()]
    for p in missing:
        logging.warning(f"Test path does not exist, skipping: {cwd / p}")
    if not existing:
        logging.error("None of the configured test paths exist; nothing to run.")
        sys.exit(1)

    cmd = ["pytest", *existing, "-v", "--color=yes"]
    if marker_expr:
        cmd.extend(["-m", marker_expr])
    if junit_xml:
        cmd.append(f"--junit-xml={junit_xml}")
    if extra_args:
        cmd.extend(extra_args)

    # pytest-timeout / pytest-xdist are optional; only pass their flags when the
    # plugin is importable so this runner works in minimal environments too.
    if timeout and find_spec("pytest_timeout") is not None:
        cmd.append(f"--timeout={timeout}")
    elif timeout:
        logging.warning("pytest-timeout not installed; running without --timeout")

    if num_workers and num_workers > 1 and find_spec("xdist") is not None:
        cmd.append(f"--numprocesses={num_workers}")
    elif num_workers and num_workers > 1:
        logging.warning("pytest-xdist not installed; running serially")

    logging.info(
        f"Running pytest on {existing} with markers: {marker_expr or 'none'}, "
        f"extra args: {extra_args or 'none'} "
        f"(timeout={timeout}s, workers={num_workers}) in {cwd}"
    )
    return subprocess.run(cmd, cwd=str(cwd), env=env, check=False).returncode


def build_environment(rocm_path, component_name):
    """
    Replicate the install-tree environment used by the legacy test_tensilelite.py
    so that imports of Tensile / rocisa (which links libamdhip64) resolve and the
    GPU unit tests can assemble kernels with amdclang++.
    """
    env = os.environ.copy()
    component_root = resolve_component_path(component_name, rocm_path)

    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{component_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(component_root)
    )
    env["ROCM_PATH"] = str(rocm_path)

    # _rocisa links libamdhip64.so (in lib/), tensilelite-client links libomp.so
    # (in lib/llvm/lib/) — both are needed or the client segfaults at load.
    lib_path = rocm_path / "lib"
    llvm_lib_path = rocm_path / "lib" / "llvm" / "lib"
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [str(lib_path), str(llvm_lib_path), existing_ld])
    )

    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(
        filter(
            None,
            [
                str(rocm_path / "bin"),
                str(rocm_path / "lib" / "llvm" / "bin"),
                existing_path,
            ],
        )
    )
    return env


if __name__ == "__main__":
    TEST_COMPONENT_NAME = get_env_value("TEST_COMPONENT")
    TEST_TYPE = get_env_value("TEST_TYPE", "quick")
    AMDGPU_FAMILIES = get_env_value("AMDGPU_FAMILIES")
    THEROCK_BIN_DIR = get_env_value("THEROCK_BIN_DIR")

    if not TEST_COMPONENT_NAME:
        logging.error("TEST_COMPONENT environment variable is required but not set.")
        sys.exit(1)
    if not THEROCK_BIN_DIR:
        logging.error("THEROCK_BIN_DIR environment variable is required but not set.")
        sys.exit(1)

    rocm_path = Path(THEROCK_BIN_DIR).resolve().parent
    component_path = resolve_component_path(TEST_COMPONENT_NAME, rocm_path)
    if not component_path.is_dir():
        logging.error(f"Component test directory does not exist: {component_path}")
        sys.exit(1)

    logging.info(f"Component: {TEST_COMPONENT_NAME} ({component_path})")
    logging.info(f"Test category: {TEST_TYPE}")

    config = load_test_categories_yaml(component_path / "test_categories.yaml")
    all_categories = config.get("test_categories", {})
    category_config = all_categories.get(TEST_TYPE)
    if not category_config:
        logging.error(
            f"No configuration found for test category '{TEST_TYPE}'. "
            f"Available categories: {sorted(all_categories)}"
        )
        sys.exit(1)

    test_paths = category_config.get("test_paths", [])
    if not test_paths:
        logging.error(f"Category '{TEST_TYPE}' defines no test_paths")
        sys.exit(1)

    gpu_arch = extract_gpu_arch(AMDGPU_FAMILIES)
    marker_expr = build_marker_expression(category_config, gpu_arch)

    # Extra pytest CLI options for this category, with {ROCM_PATH} substitution.
    pytest_args = [
        str(arg).replace("{ROCM_PATH}", str(rocm_path))
        for arg in (category_config.get("pytest_args", []) or [])
    ]

    exec_settings = config.get("execution_settings", {})
    # Per-test timeout and worker count resolve as: env override > YAML > default.
    # test_categories.yaml ships inside the installed artifact, so these env
    # overrides let CI steps / reproduce_test_failure.py tune them without a
    # rebuild. PYTEST_TEST_TIMEOUT is the per-test timeout in seconds (passed to
    # pytest-timeout); PYTEST_NUM_WORKERS is the pytest-xdist worker count.
    timeout = get_env_int_override("PYTEST_TEST_TIMEOUT") or exec_settings.get(
        "category_timeouts", {}
    ).get(TEST_TYPE)
    # parallel_workers may be overridden per category (e.g. GPU GEMM tests want
    # more xdist workers than the default), falling back to the global setting.
    num_workers = get_env_int_override("PYTEST_NUM_WORKERS") or category_config.get(
        "parallel_workers", exec_settings.get("parallel_workers", 1)
    )

    junit_dir = get_env_value("JUNIT_XML_DIR")
    junit_xml = (
        str(Path(junit_dir) / f"{TEST_COMPONENT_NAME}.xml") if junit_dir else None
    )

    env = build_environment(rocm_path, TEST_COMPONENT_NAME)
    for key, value in (exec_settings.get("environment", {}) or {}).items():
        value = str(value).replace("{ROCM_PATH}", str(rocm_path))
        env[key] = value
        logging.info(f"Set environment variable: {key}={value}")

    sys.exit(
        run_pytest(
            test_paths,
            marker_expr,
            pytest_args,
            junit_xml,
            timeout,
            num_workers,
            component_path,
            env,
        )
    )
