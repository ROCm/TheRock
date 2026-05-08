# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for ROCm test runner scripts.

This module is intentionally import-safe: importing it does not read required
CI environment variables, run subprocesses, or exit the process. Runner scripts
own their project-specific paths, environment setup, timeouts, and dispatch.
"""

from dataclasses import dataclass, field, replace
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Iterable, Mapping

VALID_TEST_CATEGORIES = {"quick", "standard", "comprehensive", "full"}

_GPU_ARCH_PATTERN = re.compile(r"gfx[0-9a-zA-Z]+", re.IGNORECASE)
_ROCMINFO_NAME_PATTERN = re.compile(r"^\s*Name:\s+(gfx[0-9a-z]+)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class CTestLabels:
    gpu_archs: set[str]
    exclude_labels: set[str]


def normalize_test_category(test_type: str | None) -> str:
    """Return a valid test category, defaulting invalid or empty values to quick."""
    if not test_type:
        return "quick"
    category = test_type.strip().lower()
    if category in VALID_TEST_CATEGORIES:
        return category
    return "quick"


def extract_gpu_arch(amdgpu_families: str | None) -> str:
    """Extract the first gfx architecture token from AMDGPU_FAMILIES-style text."""
    if not amdgpu_families:
        return ""
    match = _GPU_ARCH_PATTERN.search(amdgpu_families)
    return match.group(0).lower() if match else ""


def find_matching_gpu_arch(gpu_arch: str, available_gpu_archs: set[str]) -> str | None:
    """Find the most specific available GPU architecture label for gpu_arch."""
    if gpu_arch in available_gpu_archs:
        return gpu_arch

    for i in range(len(gpu_arch) - 1, 4, -1):
        pattern = gpu_arch[:i] + "X"
        if pattern in available_gpu_archs:
            return pattern

    return None


def _positive_int(name: str, value: str | int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be an integer, got {value!r}") from e
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


@dataclass(frozen=True)
class TestRunSettings:
    """Common test runner settings parsed once and passed through helpers."""

    test_dir: str | Path
    rocm_path: str | Path | None = None
    category: str | None = "quick"
    gpu_arch: str | None = ""
    shard_index: str | int = 1
    total_shards: str | int = 1
    available_gpu_archs: frozenset[str] = field(default_factory=frozenset)
    exclude_labels: frozenset[str] = field(default_factory=frozenset)
    ctest_parallel: str | int | None = None
    ctest_timeout_seconds: str | int | None = None
    ctest_output_on_failure: bool = True
    ctest_verbose: bool = True
    extra_ctest_args: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "test_dir", Path(self.test_dir))
        if self.rocm_path is not None:
            object.__setattr__(self, "rocm_path", Path(self.rocm_path))
        object.__setattr__(self, "category", normalize_test_category(self.category))
        object.__setattr__(self, "gpu_arch", (self.gpu_arch or "").strip().lower())
        parsed_shard_index = _positive_int("shard_index", self.shard_index)
        parsed_total_shards = _positive_int("total_shards", self.total_shards)
        if parsed_shard_index > parsed_total_shards:
            raise ValueError(
                "shard_index must be less than or equal to total_shards, "
                f"got {parsed_shard_index} > {parsed_total_shards}"
            )
        object.__setattr__(self, "shard_index", parsed_shard_index)
        object.__setattr__(self, "total_shards", parsed_total_shards)

        if self.ctest_parallel is not None:
            object.__setattr__(
                self,
                "ctest_parallel",
                _positive_int("ctest_parallel", self.ctest_parallel),
            )
        if self.ctest_timeout_seconds is not None:
            object.__setattr__(
                self,
                "ctest_timeout_seconds",
                _positive_int("ctest_timeout_seconds", self.ctest_timeout_seconds),
            )

        object.__setattr__(
            self,
            "available_gpu_archs",
            frozenset(str(v) for v in self.available_gpu_archs),
        )
        object.__setattr__(
            self,
            "exclude_labels",
            frozenset(str(v) for v in self.exclude_labels),
        )
        object.__setattr__(
            self,
            "extra_ctest_args",
            tuple(str(v) for v in self.extra_ctest_args),
        )

    @classmethod
    def from_env(
        cls,
        *,
        test_dir: str | Path,
        rocm_path: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        **kwargs,
    ) -> "TestRunSettings":
        """Create settings from common CI environment variables."""
        env = os.environ if env is None else env
        return cls(
            test_dir=test_dir,
            rocm_path=rocm_path,
            category=env.get("TEST_TYPE", "quick"),
            gpu_arch=extract_gpu_arch(env.get("AMDGPU_FAMILIES")),
            shard_index=env.get("SHARD_INDEX", 1),
            total_shards=env.get("TOTAL_SHARDS", 1),
            **kwargs,
        )

    def with_ctest(
        self,
        *,
        available_gpu_archs: Iterable[str] | None = None,
        exclude_labels: Iterable[str] | None = None,
        parallel: str | int | None = None,
        timeout_seconds: str | int | None = None,
        output_on_failure: bool | None = None,
        verbose: bool | None = None,
        extra_args: Iterable[str] | None = None,
    ) -> "TestRunSettings":
        """Return settings with CTest-specific values replaced."""
        updates = {}
        if available_gpu_archs is not None:
            updates["available_gpu_archs"] = frozenset(available_gpu_archs)
        if exclude_labels is not None:
            updates["exclude_labels"] = frozenset(exclude_labels)
        if parallel is not None:
            updates["ctest_parallel"] = parallel
        if timeout_seconds is not None:
            updates["ctest_timeout_seconds"] = timeout_seconds
        if output_on_failure is not None:
            updates["ctest_output_on_failure"] = output_on_failure
        if verbose is not None:
            updates["ctest_verbose"] = verbose
        if extra_args is not None:
            updates["extra_ctest_args"] = tuple(extra_args)
        return replace(self, **updates)

    def with_ctest_labels(self, labels: CTestLabels) -> "TestRunSettings":
        """Return settings with discovered CTest labels applied."""
        return replace(
            self,
            available_gpu_archs=frozenset(labels.gpu_archs),
            exclude_labels=frozenset(labels.exclude_labels),
        )


def gtest_shard_env(
    shard_index: str | int = 1, total_shards: str | int = 1
) -> dict[str, str]:
    """Translate 1-based CI shard values into GTest sharding environment vars."""
    parsed_shard_index = _positive_int("shard_index", shard_index)
    parsed_total_shards = _positive_int("total_shards", total_shards)
    if parsed_shard_index > parsed_total_shards:
        raise ValueError(
            "shard_index must be less than or equal to total_shards, "
            f"got {parsed_shard_index} > {parsed_total_shards}"
        )
    return {
        "GTEST_SHARD_INDEX": str(parsed_shard_index - 1),
        "GTEST_TOTAL_SHARDS": str(parsed_total_shards),
    }


def ctest_shard_args(
    shard_index: str | int = 1, total_shards: str | int = 1
) -> list[str]:
    """Return CTest --tests-information args for 1-based CI shard values."""
    parsed_shard_index = _positive_int("shard_index", shard_index)
    parsed_total_shards = _positive_int("total_shards", total_shards)
    if parsed_shard_index > parsed_total_shards:
        raise ValueError(
            "shard_index must be less than or equal to total_shards, "
            f"got {parsed_shard_index} > {parsed_total_shards}"
        )
    return [
        "--tests-information",
        f"{parsed_shard_index},,{parsed_total_shards}",
    ]


def build_ctest_label_args(
    category: str | None,
    gpu_arch: str | None,
    available_gpu_archs: set[str],
    exclude_labels: set[str],
) -> list[str]:
    """Build CTest label include/exclude arguments for category and GPU labels."""
    normalized_category = normalize_test_category(category)
    normalized_gpu_arch = (gpu_arch or "").strip().lower()
    args: list[str] = ["-L", normalized_category]
    le_patterns: list[str] = []

    category_exclude_label = f"{normalized_category}_exclude"
    if category_exclude_label in exclude_labels:
        le_patterns.append(category_exclude_label)

    if normalized_gpu_arch in ("", "generic", "none"):
        le_patterns.append("ex_gpu")
    else:
        matching_arch = find_matching_gpu_arch(normalized_gpu_arch, available_gpu_archs)
        if matching_arch:
            args.extend(["-L", f"ex_gpu_{matching_arch}"])
        else:
            le_patterns.append("ex_gpu")

    if le_patterns:
        args.extend(["-LE", "|".join(le_patterns)])

    return args


def _validate_test_dir(test_dir: Path) -> None:
    if not test_dir.exists() or not test_dir.is_dir():
        raise FileNotFoundError(f"CTest test directory does not exist: {test_dir}")


def count_ctest_tests(test_dir: str | Path, runner=subprocess.run) -> int:
    """Return the number of tests reported by ctest -N for test_dir."""
    test_dir = Path(test_dir)
    _validate_test_dir(test_dir)
    result = runner(
        ["ctest", "-N", "--test-dir", str(test_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    return sum(
        1 for line in result.stdout.splitlines() if re.search(r"Test\s+#\d+:", line)
    )


def read_ctest_labels(test_dir: str | Path, runner=subprocess.run) -> set[str]:
    """Return labels printed by ctest --print-labels for test_dir."""
    test_dir = Path(test_dir)
    _validate_test_dir(test_dir)
    result = runner(
        ["ctest", "--print-labels", "--test-dir", str(test_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def parse_ctest_labels(labels: Iterable[str]) -> CTestLabels:
    """Parse CTest labels into GPU architecture and category-exclude sets."""
    gpu_archs: set[str] = set()
    exclude_labels: set[str] = set()
    gpu_prefix = "ex_gpu_"
    exclude_suffix = "_exclude"

    for raw_label in labels:
        label = raw_label.strip()
        if label.startswith(gpu_prefix):
            gpu_arch = label[len(gpu_prefix) :]
            if gpu_arch.startswith("gfx"):
                gpu_archs.add(gpu_arch)
        elif label.endswith(exclude_suffix):
            exclude_labels.add(label)

    return CTestLabels(gpu_archs=gpu_archs, exclude_labels=exclude_labels)


def discover_ctest_labels(
    test_dir: str | Path,
    runner=subprocess.run,
    *,
    require_tests: bool = True,
) -> CTestLabels:
    """Run CTest discovery and return parsed GPU/category-exclude labels."""
    if require_tests and count_ctest_tests(test_dir, runner=runner) == 0:
        raise RuntimeError(f"No CTest tests found in {Path(test_dir)}")
    return parse_ctest_labels(read_ctest_labels(test_dir, runner=runner))


def build_ctest_command(settings: TestRunSettings) -> list[str]:
    """Build a CTest command from generic test run settings."""
    cmd = ["ctest"]
    cmd.extend(
        build_ctest_label_args(
            settings.category,
            settings.gpu_arch,
            set(settings.available_gpu_archs),
            set(settings.exclude_labels),
        )
    )

    if settings.ctest_output_on_failure:
        cmd.append("--output-on-failure")
    if settings.ctest_parallel is not None:
        cmd.extend(["--parallel", str(settings.ctest_parallel)])
    if settings.ctest_timeout_seconds is not None:
        cmd.extend(["--timeout", str(settings.ctest_timeout_seconds)])
    cmd.extend(["--test-dir", str(settings.test_dir)])
    if settings.ctest_verbose:
        cmd.append("-V")
    cmd.extend(ctest_shard_args(settings.shard_index, settings.total_shards))
    cmd.extend(settings.extra_ctest_args)
    return cmd


def _prepend_env_paths(
    env: dict[str, str],
    paths_by_var: Mapping[str, Iterable[str | Path]] | None,
) -> None:
    if not paths_by_var:
        return
    for env_key, paths in paths_by_var.items():
        new_paths = [str(path) for path in paths]
        existing_path = env.get(env_key, "")
        env[env_key] = os.pathsep.join(filter(None, new_paths + [existing_path]))


def build_test_env(
    settings: TestRunSettings,
    *,
    base_env: Mapping[str, str] | None = None,
    path_prepend: Mapping[str, Iterable[str | Path]] | None = None,
    extra_env: Mapping[str, str | Path] | None = None,
) -> dict[str, str]:
    """Build a test environment with common ROCm and GTest settings."""
    env = dict(base_env or {})
    if settings.rocm_path is not None:
        env["ROCM_PATH"] = str(settings.rocm_path)
    env.update(gtest_shard_env(settings.shard_index, settings.total_shards))
    _prepend_env_paths(env, path_prepend)
    if extra_env:
        env.update({key: str(value) for key, value in extra_env.items()})
    return env


def run_ctest(
    settings: TestRunSettings,
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    runner=subprocess.run,
    check: bool = False,
    discover_labels: bool = False,
    require_tests: bool = True,
):
    """Run CTest with settings-owned common mechanics and caller-owned env/cwd."""
    if discover_labels:
        settings = settings.with_ctest_labels(
            discover_ctest_labels(
                settings.test_dir,
                runner=runner,
                require_tests=require_tests,
            )
        )
    cmd = build_ctest_command(settings)
    if cwd is None:
        cwd = settings.rocm_path
    logging.info("++ Exec [%s]$ %s", cwd or Path.cwd(), shlex.join(cmd))
    return runner(cmd, cwd=cwd, env=env, check=check)


def _rocminfo_command(rocm_bin_dir: str | Path | None = None) -> str:
    if rocm_bin_dir is not None:
        rocminfo = Path(rocm_bin_dir) / "rocminfo"
        if rocminfo.exists():
            return str(rocminfo)
    return "rocminfo"


def parse_rocminfo_gpu_archs(output: str) -> list[str]:
    """Return visible GPU architecture names from rocminfo output."""
    gpu_archs: list[str] = []
    for line in output.splitlines():
        match = _ROCMINFO_NAME_PATTERN.match(line)
        if match:
            gpu_archs.append(match.group(1).lower())
    return gpu_archs


def get_visible_gpu_count(
    env: Mapping[str, str] | None = None,
    rocm_bin_dir: str | Path | None = None,
    runner=subprocess.run,
) -> int:
    """Return the number of visible GPU architecture records in rocminfo output."""
    result = runner(
        [_rocminfo_command(rocm_bin_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )
    return len(parse_rocminfo_gpu_archs(result.stdout or ""))


def get_first_gpu_architecture(
    env: Mapping[str, str] | None = None,
    rocm_bin_dir: str | Path | None = None,
    runner=subprocess.run,
) -> str:
    """Return the first visible GPU architecture, such as gfx942."""
    result = runner(
        [_rocminfo_command(rocm_bin_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=True,
    )
    gpu_archs = parse_rocminfo_gpu_archs(result.stdout or "")
    if gpu_archs:
        gpu_arch = gpu_archs[0]
        logging.info("Detected GPU architecture: %s", gpu_arch)
        return gpu_arch
    raise RuntimeError("No GPU architecture found in rocminfo output")


def is_asan_artifact_group(artifact_group: str | None) -> bool:
    """Return whether an artifact group name describes an ASAN build."""
    return "asan" in (artifact_group or "").lower()
