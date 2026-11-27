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
``{ROCM_PATH}`` tokens in those values are substituted with the install prefix,
and ``{AMDGPU_TARGETS}`` tokens with the concrete GPU target (first entry of the
``AMDGPU_TARGETS`` env var, e.g. ``gfx1250``; when unset, the gfx token is
extracted from ``AMDGPU_FAMILIES`` via ``extract_gpu_arch()``).  This applies to
``pytest_args``, ``pytest_markers``, ``exclude_markers``, and
``execution_settings.environment``.

A category may also set ``min_executed_tests``: the fewest test cases that must
actually run (pass, fail or error; skips and xfails do not count). A GPU category
whose tests all skip, for example because each is marked for a GPU architecture
other than the runner's, exits 0 under pytest; with this floor the job fails
instead. The value is a single count for all GPU families, or a table keyed by
the family in ``AMDGPU_FAMILIES``. In a table, ``default`` covers the families
not listed; without it, an unlisted family has no floor (0 tests is expected
there)::

    min_executed_tests:
      gfx94X-dcgpu: 1
      gfx950-dcgpu: 250

Each key must be ``default`` or a family from ``amdgpu_family_matrix.py``. Any
other key fails the job on every runner, since a misspelled family never equals
``AMDGPU_FAMILIES`` and would silently drop its floor. The count comes from
pytest's JUnit XML report, which merges every pytest-xdist worker.

Environment variables used:
TEST_COMPONENT: Job name of the component to test (e.g. "tensilelite").
TEST_TYPE: Test category to run; must be a category defined in
    test_categories.yaml (e.g. quick, standard, comprehensive, full). Defaults to
    "quick" when unset.
TEST_CATEGORY: Optional. When set, overrides TEST_TYPE, for jobs that always
    run one specific category regardless of the CI tier (e.g. hw-common). If
    the installed component does not define that category yet, the run is
    skipped with a warning instead of failing.
AMDGPU_FAMILIES: GPU family (e.g. "gfx94X-dcgpu"). Its gfx token is the
    skip-marker arch, and substitutes {AMDGPU_TARGETS} if AMDGPU_TARGETS is unset.
AMDGPU_TARGETS: Optional comma-separated targets (e.g. "gfx942"). The first
    substitutes {AMDGPU_TARGETS}; a lone entry replaces the family's skip arch.
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
import tempfile
import xml.etree.ElementTree as ElementTree
from importlib.util import find_spec
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))
from amdgpu_family_matrix import (
    amdgpu_family_info_matrix,
    amdgpu_family_info_matrix_explicit_only,
    is_asan,
)
from configure_asan_env import get_asan_runtime_path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Map job name -> install location, relative to the ROCm prefix (THEROCK_BIN_DIR
# parent). Components live under share/ rather than bin/ because they are Python
# packages + pytest modules, not native test executables.
INSTALLED_COMPONENTS = {
    "tensilelite": "share/hipblaslt/tensilelite",
    "tensilelite-common": "share/hipblaslt/tensilelite",
}


def _fail(message):
    """Log an error and exit non-zero.

    Single op so callers can't log an error and forget to exit.
    """
    logging.error(message)
    sys.exit(1)


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


def resolve_test_category():
    """Return the category to run: TEST_CATEGORY if set, else TEST_TYPE."""
    return os.getenv("TEST_CATEGORY") or os.getenv("TEST_TYPE", "quick")


def lookup_category(all_categories, category):
    """Return the config for `category`, or exit if the component lacks it.

    A category pinned via TEST_CATEGORY that the installed component does not
    define yet exits 0 with a warning, so a CI job can land before the
    component change that adds its category. A missing tier is still an error.
    """
    category_config = all_categories.get(category)
    if category_config:
        return category_config
    available = sorted(all_categories)
    if os.getenv("TEST_CATEGORY"):
        message = (
            f"Test category '{category}' is not defined by this component yet "
            f"(available: {available}); skipping."
        )
        print(f"::warning title=pytest_runner::{message}")
        logging.warning(message)
        sys.exit(0)
    _fail(
        f"No configuration found for test category '{category}'. "
        f"Available categories: {available}"
    )


def resolve_component_path(component_name, rocm_path):
    """Return the install directory that holds the component's tests + YAML."""
    rel = INSTALLED_COMPONENTS.get(component_name)
    if rel is None:
        _fail(
            f"Unknown pytest component '{component_name}'. "
            f"Known components: {sorted(INSTALLED_COMPONENTS)}"
        )
    return (rocm_path / rel).resolve()


def load_test_categories_yaml(yaml_path):
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded test categories from {yaml_path}")
        return config
    except FileNotFoundError:
        _fail(f"test_categories.yaml not found at {yaml_path}")
    except yaml.YAMLError as e:
        _fail(f"Invalid YAML syntax in {yaml_path}: {e}")


def extract_gpu_arch(amdgpu_families):
    """Return the first GPU architecture named in AMDGPU_FAMILIES.

    Examples:
        "gfx942"          -> "gfx942"
        "gfx94X-dcgpu"    -> "gfx94X"          (family name; the "-dcgpu" suffix is dropped)
        "gfx1250-strict"  -> "gfx1250-strict"  (a separate target; "-strict" is kept)
    """
    if not amdgpu_families:
        return None
    # "-strict" is part of the target name (gfx1250-strict is its own target, with
    # its own skip-gfx1250-strict marker), so keep it. Other suffixes such as
    # "-dcgpu" or "-all" only group targets into a family, so drop them.
    match = re.search(r"gfx\w+(?:-strict)?", amdgpu_families)
    if match:
        gpu_arch = match.group(0)
        logging.info(f"Detected GPU architecture: {gpu_arch}")
        return gpu_arch
    logging.warning(f"Could not parse GPU architecture from: {amdgpu_families}")
    return None


def resolve_skip_arch(gpu_arch: str | None, amdgpu_targets: str | None) -> str | None:
    """Pick the arch used in the ``not skip-<arch>`` pytest marker term.

    Test configs opt out of a GPU with markers named after a single target,
    e.g. ``skip-gfx942`` or ``skip-gfx1250``. *gpu_arch* is
    ``extract_gpu_arch(AMDGPU_FAMILIES)``, often a family name such as
    ``gfx94X``, which matches no such marker.
    AMDGPU_TARGETS is the comma-separated target list that
    configure_multi_arch_ci.py builds from the family's "fetch-gfx-targets".
    When it names exactly one target, that target is the GPU under test, so use it.

    Examples (gpu_arch, AMDGPU_TARGETS -> arch):
        "gfx94X",  "gfx942"          -> "gfx942"
        "gfx125X", "gfx1250"         -> "gfx1250"
        "gfx125X", "gfx1250-strict"  -> "gfx1250-strict"
        "gfx120X", "gfx1200,gfx1201" -> "gfx120X" (unchanged: either target
            may be on the runner, and per-target skips still apply in the
            test's own conftest, which sees the detected GPU)
        "gfx942",  ""                -> "gfx942"  (unchanged)
    """
    if not gpu_arch:
        return gpu_arch
    targets = [t for t in (amdgpu_targets or "").split(",") if t]
    return targets[0] if len(targets) == 1 else gpu_arch


def build_marker_expression(category_config, gpu_arch, amdgpu_target=""):
    """
    Build a pytest -m expression from a category's marker config plus the
    GPU-arch skip markers registered in pytest.ini (skip-gfxXXXX).

    ``{AMDGPU_TARGETS}`` tokens in marker strings are substituted with the
    *amdgpu_target* value (a concrete gfx target like ``gfx1250``) so
    categories can be arch-agnostic.  When *amdgpu_target* is empty,
    any marker containing ``{AMDGPU_TARGETS}`` is dropped entirely
    (preventing malformed expressions like ``( and emu_fast)``).
    """
    gfx_target = amdgpu_target or ""
    include_markers = category_config.get("pytest_markers", []) or []
    exclude_markers = category_config.get("exclude_markers", []) or []
    if not gfx_target:
        include_markers = [
            m for m in include_markers if "{AMDGPU_TARGETS}" not in str(m)
        ]
        exclude_markers = [
            m for m in exclude_markers if "{AMDGPU_TARGETS}" not in str(m)
        ]
    include = [str(m).replace("{AMDGPU_TARGETS}", gfx_target) for m in include_markers]
    exclude = [str(m).replace("{AMDGPU_TARGETS}", gfx_target) for m in exclude_markers]

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
        _fail("None of the configured test paths exist; nothing to run.")

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


def count_executed_tests(junit_xml: Path) -> int:
    """Test cases in a pytest JUnit XML report that ran: passed, failed or errored.

    pytest records a skipped test, and an expected failure (`xfail`), as a
    `testcase` with a `skipped` child; those do not count.
    """
    root = ElementTree.parse(junit_xml).getroot()
    return sum(1 for case in root.iter("testcase") if case.find("skipped") is None)


def read_min_executed_tests(
    category_config: dict, category: str, amdgpu_families: str | None
) -> int | None:
    """The category's `min_executed_tests` floor on this family, or None.

    The value is a single count for all families, or a table keyed by family (the
    `AMDGPU_FAMILIES` string, e.g. `gfx94X-dcgpu`) in which `default` covers the
    families not listed. Exits the job on a bad key or value. Returns None when
    neither this family nor `default` has an entry.
    """
    floors = category_config.get("min_executed_tests")
    if floors is None:
        return None
    is_table = isinstance(floors, dict)
    if not is_table:
        floors = {"default": floors}
    known_families = {
        info["family"]
        for matrix in (
            amdgpu_family_info_matrix,
            amdgpu_family_info_matrix_explicit_only,
        )
        for platforms in matrix.values()
        for info in platforms.values()
    }
    for family, minimum in floors.items():
        # Checked on every runner, not only the family named: a misspelled family
        # (`gfx94x-dcgpu`) or a target (`gfx942`) would never match AMDGPU_FAMILIES.
        if family != "default" and family not in known_families:
            _fail(
                f"Category '{category}': min_executed_tests key {family!r} must be "
                f"'default' or a GPU family from amdgpu_family_matrix.py, such as "
                f"gfx94X-dcgpu (not a target such as gfx942)"
            )
        # bool is an int subclass; `true` is a typo, not a floor of 1.
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            entry = f"min_executed_tests.{family}" if is_table else "min_executed_tests"
            _fail(
                f"Category '{category}': {entry} must be a non-negative integer, "
                f"got {minimum!r}"
            )
    return floors.get(amdgpu_families, floors.get("default"))


def check_min_executed_tests(
    junit_xml: Path, minimum: int, category: str, amdgpu_families: str | None
) -> None:
    """Fail the job when fewer than `minimum` test cases in the JUnit report ran."""
    if not junit_xml.is_file():
        _fail(
            f"Category '{category}' requires at least {minimum} executed tests, "
            f"but pytest wrote no JUnit report to {junit_xml}. A --junit-xml or "
            "-p no:junitxml in the category's pytest_args overrides the runner's."
        )
    executed = count_executed_tests(junit_xml)
    logging.info(f"Executed tests (not skipped): {executed}, minimum {minimum}")
    if executed < minimum:
        _fail(
            f"Category '{category}' ran {executed} tests on "
            f"AMDGPU_FAMILIES={amdgpu_families}, below its min_executed_tests of "
            f"{minimum}; the rest of the selected tests were skipped. Check the "
            "tests' GPU architecture marks against this family, or lower its "
            "floor in test_categories.yaml."
        )


def build_environment(rocm_path, component_name):
    """
    Replicate the install-tree environment used by the legacy test_tensilelite.py
    so that imports of Tensile / rocisa (which links libamdhip64) resolve and the
    GPU unit tests can assemble kernels with amdclang++.
    """
    env = os.environ.copy()
    component_root = resolve_component_path(component_name, rocm_path)

    # Initialize ASan before Python allocates objects that instrumented imports
    # may later free. Merely exposing the runtime on LD_LIBRARY_PATH loads it
    # too late when rocisa imports instrumented ROCm libraries.
    if is_asan():
        runtime = str(get_asan_runtime_path(rocm_path))
        existing_preload = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = (
            f"{runtime}:{existing_preload}" if existing_preload else runtime
        )

    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{component_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(component_root)
    )
    env["ROCM_PATH"] = str(rocm_path)

    # _rocisa links libamdhip64.so (in lib/), tensilelite-client links libomp.so
    # (in lib/llvm/lib/<triple>/ due to GNUInstallDirs) and librocm_sysdeps_z.so
    # (in lib/rocm_sysdeps/lib/).
    lib_path = rocm_path / "lib"
    llvm_lib_path = rocm_path / "lib" / "llvm" / "lib"
    sysdeps_lib_path = rocm_path / "lib" / "rocm_sysdeps" / "lib"
    ld_paths = [str(lib_path), str(llvm_lib_path)]
    # GNUInstallDirs puts libomp.so under one or more platform-specific triple
    # subdirs (e.g. x86_64-unknown-linux-gnu/) instead of lib/llvm/lib/ directly.
    # Add every triple dir (sorted) so the result is deterministic and does not
    # depend on glob order or break when a new triple is added to the location.
    llvm_platform_dirs = sorted(
        p for p in llvm_lib_path.glob("*-linux-*") if p.is_dir()
    )
    ld_paths.extend(str(p) for p in llvm_platform_dirs)
    if sysdeps_lib_path.is_dir():
        ld_paths.append(str(sysdeps_lib_path))
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_paths.append(existing_ld)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths)

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
    TEST_COMPONENT_NAME = os.getenv("TEST_COMPONENT")
    TEST_TYPE = resolve_test_category()
    AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
    AMDGPU_TARGETS = os.getenv("AMDGPU_TARGETS")
    THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")

    if not TEST_COMPONENT_NAME:
        _fail("TEST_COMPONENT environment variable is required but not set.")
    if not THEROCK_BIN_DIR:
        _fail("THEROCK_BIN_DIR environment variable is required but not set.")

    rocm_path = Path(THEROCK_BIN_DIR).resolve().parent
    component_path = resolve_component_path(TEST_COMPONENT_NAME, rocm_path)
    if not component_path.is_dir():
        _fail(f"Component test directory does not exist: {component_path}")

    logging.info(f"Component: {TEST_COMPONENT_NAME} ({component_path})")
    logging.info(f"Test category: {TEST_TYPE}")

    config = load_test_categories_yaml(component_path / "test_categories.yaml")
    all_categories = config.get("test_categories", {})
    category_config = lookup_category(all_categories, TEST_TYPE)

    test_paths = category_config.get("test_paths", [])
    if not test_paths:
        _fail(f"Category '{TEST_TYPE}' defines no test_paths")

    gpu_arch = extract_gpu_arch(AMDGPU_FAMILIES)
    # Resolve a concrete gfx target for {AMDGPU_TARGETS} substitution.
    # AMDGPU_TARGETS (set by test_component.yml from fetch-gfx-targets)
    # carries real targets like "gfx1250". Fallback: extract_gpu_arch()
    # parses a gfx token from AMDGPU_FAMILIES — only correct when that
    # var holds a bare target (e.g. NPI sets "gfx1250"), not when it
    # holds a family name (e.g. "gfx125X-dcgpu" → "gfx125X", wrong).
    amdgpu_target = (
        AMDGPU_TARGETS.split(",")[0].strip() if AMDGPU_TARGETS else ""
    ) or (gpu_arch or "")
    marker_expr = build_marker_expression(
        category_config, resolve_skip_arch(gpu_arch, AMDGPU_TARGETS), amdgpu_target
    )

    # Extra pytest CLI options for this category.  {ROCM_PATH} and
    # {AMDGPU_TARGETS} tokens are substituted with runtime values.
    pytest_args = [
        str(arg)
        .replace("{ROCM_PATH}", str(rocm_path))
        .replace("{AMDGPU_TARGETS}", amdgpu_target)
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

    min_executed_tests = read_min_executed_tests(
        category_config, TEST_TYPE, AMDGPU_FAMILIES
    )
    junit_dir = os.getenv("JUNIT_XML_DIR")
    if not junit_dir and min_executed_tests is not None:
        # The floor is checked from the JUnit report, so write one to a temporary
        # directory when JUNIT_XML_DIR is unset.
        junit_dir = tempfile.mkdtemp(prefix="pytest_runner_junit_")
    junit_xml = (
        str(Path(junit_dir) / f"{TEST_COMPONENT_NAME}.xml") if junit_dir else None
    )

    env = build_environment(rocm_path, TEST_COMPONENT_NAME)
    for key, value in (exec_settings.get("environment", {}) or {}).items():
        value = (
            str(value)
            .replace("{ROCM_PATH}", str(rocm_path))
            .replace("{AMDGPU_TARGETS}", amdgpu_target)
        )
        env[key] = value
        logging.info(f"Set environment variable: {key}={value}")

    returncode = run_pytest(
        test_paths,
        marker_expr,
        pytest_args,
        junit_xml,
        timeout,
        num_workers,
        component_path,
        env,
    )
    # A failed run already fails the job; the floor only matters when pytest exits 0.
    if returncode == 0 and min_executed_tests is not None:
        check_min_executed_tests(
            Path(junit_xml), min_executed_tests, TEST_TYPE, AMDGPU_FAMILIES
        )
    sys.exit(returncode)
