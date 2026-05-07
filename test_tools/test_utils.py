# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for ROCm test runner scripts.

This module is intentionally import-safe: importing it does not read required
CI environment variables, run subprocesses, or exit the process. Runner scripts
own their project-specific paths, environment setup, timeouts, and dispatch.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import subprocess
from typing import Iterable, Mapping


VALID_TEST_CATEGORIES = {"quick", "standard", "comprehensive", "full"}

_GPU_ARCH_PATTERN = re.compile(r"gfx[0-9a-zA-Z]+", re.IGNORECASE)
_ROCMINFO_NAME_PATTERN = re.compile(
    r"^\s*Name:\s+(gfx[0-9a-z]+)\s*$", re.IGNORECASE
)


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
        matching_arch = find_matching_gpu_arch(
            normalized_gpu_arch, available_gpu_archs
        )
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
        1
        for line in result.stdout.splitlines()
        if re.search(r"Test\s+#\d+:", line)
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
