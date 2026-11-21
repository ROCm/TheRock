#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolves the developer-facing inputs of dev_pytorch_wheels.yml.

The workflow exposes GPU checkboxes and short ROCm version keywords so a
developer does not need to know family names or package versions. This script
turns them into the inputs the multi-arch PyTorch build workflows expect.

GPU checkboxes map to families from cmake/therock_amdgpu_targets.cmake:

* Datacenter (Instinct): DATACENTER_INSTINCT_FAMILIES, Linux only.
* Consumer (Radeon / Navi): CONSUMER_RADEON_FAMILIES, Linux and Windows.

Extra families are added to every selected platform. A platform that ends up
with no families is skipped (empty output).

ROCm version keywords:

* ``latest`` (default): the newest ``rocm`` version on the selected index.
* ``trunk``: the ROCm version upstream pytorch/pytorch trunk CI builds with,
  read from .ci/docker/build.sh, resolved to the newest matching release on
  the selected index (``10.1`` -> ``10.1.0``).
* Anything else is an exact version, or a series like ``10.1`` that resolves
  to its newest version on the index.

Example:

    python resolve_dev_pytorch_wheel_inputs.py \\
        --build-linux true --build-windows false \\
        --datacenter-instinct-gpus true --consumer-radeon-gpus false \\
        --extra-amdgpu-families "" --rocm-version trunk \\
        --rocm-package-index-url https://stable.repo.amd.com/rocm/whl-next/

writes these step outputs to GITHUB_OUTPUT:

    linux_amdgpu_families=gfx90a;gfx94X-dcgpu;gfx950-dcgpu;gfx125X-dcgpu
    windows_amdgpu_families=
    rocm_version=10.1.0
"""

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from packaging.version import InvalidVersion, Version

_BUILD_TOOLS_DIR = Path(__file__).resolve().parents[2] / "build_tools"
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.cmake_amdgpu_targets import amdgpu_family_map
from github_actions.github_actions_api import (
    GitHubAPIError,
    gha_append_step_summary,
    gha_fetch_text_file_contents,
    gha_set_output,
    str2bool,
)

DATACENTER_INSTINCT_FAMILIES = (
    "gfx90a",  # MI200
    "gfx94X-dcgpu",  # MI300
    "gfx950-dcgpu",  # MI350
    "gfx125X-dcgpu",  # MI450
)
CONSUMER_RADEON_FAMILIES = (
    "gfx103X-all",  # RX 6000 (RDNA2)
    "gfx110X-all",  # RX 7000 (RDNA3)
    "gfx120X-all",  # RX 9000 (RDNA4)
    "gfx1151",  # Strix Halo
)
# Instinct GPUs have no Windows PyTorch support.
WINDOWS_EXCLUDED_FAMILIES = frozenset(DATACENTER_INSTINCT_FAMILIES)

UPSTREAM_REPOSITORY = "pytorch/pytorch"
UPSTREAM_BUILD_SH_PATH = ".ci/docker/build.sh"
UPSTREAM_REF = "main"
# Image used by the ROCm jobs in upstream trunk.yml.
UPSTREAM_TRUNK_IMAGE = "pytorch-linux-noble-rocm-n-py3.11"

ROCM_PACKAGE = "rocm"


def split_families(value: str) -> list[str]:
    """Splits a semicolon- or comma-separated family list, dropping blanks."""
    return [f.strip() for f in re.split(r"[;,]", value or "") if f.strip()]


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def resolve_families(
    *,
    build_linux: bool,
    build_windows: bool,
    datacenter_instinct_gpus: bool,
    consumer_radeon_gpus: bool,
    extra_amdgpu_families: str,
    known_families: set[str],
) -> dict[str, list[str]]:
    """Returns the families to build for each platform."""
    extras = split_families(extra_amdgpu_families)
    unknown = [f for f in extras if f not in known_families]
    if unknown:
        raise ValueError(
            f"Unknown AMD GPU families {unknown}. See "
            "https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake"
        )

    selected: list[str] = []
    if datacenter_instinct_gpus:
        selected.extend(DATACENTER_INSTINCT_FAMILIES)
    if consumer_radeon_gpus:
        selected.extend(CONSUMER_RADEON_FAMILIES)
    selected = _dedupe(selected + extras)

    linux = selected if build_linux else []
    windows = (
        [f for f in selected if f not in WINDOWS_EXCLUDED_FAMILIES]
        if build_windows
        else []
    )
    if not linux and not windows:
        raise ValueError(
            "Nothing to build: select a platform and at least one GPU checkbox "
            "or extra family (Instinct families are Linux only)."
        )
    return {"linux": linux, "windows": windows}


def parse_upstream_rocm_version(
    build_sh: str, image: str = UPSTREAM_TRUNK_IMAGE
) -> str:
    """Returns ROCM_VERSION from the build.sh case branch that lists `image`."""
    in_branch = False
    for line in build_sh.splitlines():
        stripped = line.strip()
        if not in_branch:
            labels = stripped.rstrip(")").split("|")
            if stripped.endswith(")") and image in (l.strip() for l in labels):
                in_branch = True
            continue
        if stripped == ";;":
            break
        match = re.match(r"ROCM_VERSION=['\"]?([0-9][^'\"\s]*)", stripped)
        if match:
            return match.group(1)
    raise ValueError(
        f"Could not find a literal ROCM_VERSION for '{image}' in upstream "
        f"{UPSTREAM_BUILD_SH_PATH}"
    )


def parse_index_versions(project_page: str, package: str = ROCM_PACKAGE) -> list[str]:
    """Returns the distinct versions of `package` linked from a PEP 503 page."""
    pattern = re.compile(
        rf"(?<![\w.-]){re.escape(package)}-(\d[^/\"'<>#\s]*?)"
        r"(?:\.tar\.gz|-py\d[^/\"'<>#\s]*\.whl)"
    )
    versions = []
    for raw in pattern.findall(project_page):
        try:
            versions.append(str(Version(raw)))
        except InvalidVersion:
            continue
    return _dedupe(versions)


def _fetch_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "TheRock-CI"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def newest_version(versions: list[str], series: str | None = None) -> str | None:
    """Returns the newest version, optionally limited to a release series."""
    candidates = [Version(v) for v in versions]
    if series:
        prefix = Version(series).release
        candidates = [v for v in candidates if v.release[: len(prefix)] == prefix]
    return str(max(candidates)) if candidates else None


def resolve_rocm_version(
    requested: str,
    index_url: str,
    *,
    fetch_url: Callable[[str], str] | None = None,
    fetch_upstream_build_sh: Callable[[], str] | None = None,
) -> tuple[str, str]:
    """Returns (version, note) for a ROCm version keyword or exact version."""
    requested = requested.strip()
    project_url = f"{index_url.rstrip('/')}/{ROCM_PACKAGE}/"
    versions = parse_index_versions((fetch_url or _fetch_url)(project_url))
    if not versions:
        raise ValueError(f"No '{ROCM_PACKAGE}' packages found at {project_url}")

    if requested == "trunk":
        if fetch_upstream_build_sh is None:
            fetch_upstream_build_sh = lambda: gha_fetch_text_file_contents(
                UPSTREAM_REPOSITORY, UPSTREAM_BUILD_SH_PATH, UPSTREAM_REF
            )
        series = parse_upstream_rocm_version(fetch_upstream_build_sh())
        version = newest_version(versions, series)
        if version is None:
            raise ValueError(
                f"Upstream trunk uses ROCm {series}, which is not on {index_url}. "
                "Use rocm_version 'latest' or an exact version for this index."
            )
        return version, f"upstream trunk ROCm {series}"

    if requested == "latest":
        return newest_version(versions), "newest on index"

    try:
        normalized = str(Version(requested))
    except InvalidVersion as e:
        raise ValueError(
            f"rocm_version '{requested}' is not 'trunk', 'latest', or a version"
        ) from e
    if normalized in versions:
        return normalized, "exact"
    # Only a bare series ("10" or "10.1") widens to its newest match; a full
    # version such as 10.1.0 or 10.1.0a20261109 must exist exactly.
    requested_version = Version(normalized)
    is_series = len(requested_version.release) < 3 and str(
        requested_version
    ) == ".".join(str(part) for part in requested_version.release)
    in_series = newest_version(versions, normalized) if is_series else None
    if in_series:
        return in_series, f"newest {normalized} on index"
    raise ValueError(
        f"ROCm {requested} is not on {index_url}. "
        f"Newest available: {newest_version(versions)}"
    )


def _write_summary(
    families: dict[str, list[str]],
    rocm_version: str,
    rocm_note: str,
    index_url: str,
    pytorch_git_ref: str,
) -> None:
    def fmt(platform_families: list[str]) -> str:
        return f"`{';'.join(platform_families)}`" if platform_families else "skipped"

    lines = [
        "## Dev PyTorch wheels: resolved inputs",
        "",
        "| Input | Value |",
        "|-------|-------|",
        f"| ROCm/pytorch branch | `{pytorch_git_ref}` |",
        f"| Linux GPU families | {fmt(families['linux'])} |",
        f"| Windows GPU families | {fmt(families['windows'])} |",
        f"| ROCm version | `{rocm_version}` ({rocm_note}) |",
        f"| ROCm package index | {index_url} |",
    ]
    gha_append_step_summary("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Resolve dev_pytorch_wheels.yml inputs for the build workflows"
    )
    parser.add_argument("--build-linux", default="true")
    parser.add_argument("--build-windows", default="false")
    parser.add_argument("--datacenter-instinct-gpus", default="false")
    parser.add_argument("--consumer-radeon-gpus", default="false")
    parser.add_argument("--extra-amdgpu-families", default="")
    parser.add_argument("--rocm-version", default="latest")
    parser.add_argument("--rocm-package-index-url", required=True)
    parser.add_argument("--pytorch-git-ref", default="")
    args = parser.parse_args(argv)

    try:
        families = resolve_families(
            build_linux=str2bool(args.build_linux),
            build_windows=str2bool(args.build_windows),
            datacenter_instinct_gpus=str2bool(args.datacenter_instinct_gpus),
            consumer_radeon_gpus=str2bool(args.consumer_radeon_gpus),
            extra_amdgpu_families=args.extra_amdgpu_families,
            known_families=set(amdgpu_family_map()),
        )
        rocm_version, rocm_note = resolve_rocm_version(
            args.rocm_version, args.rocm_package_index_url
        )
    except (ValueError, GitHubAPIError, urllib.error.URLError) as e:
        print(f"::error::{e}")
        sys.exit(1)

    gha_set_output(
        {
            "linux_amdgpu_families": ";".join(families["linux"]),
            "windows_amdgpu_families": ";".join(families["windows"]),
            "rocm_version": rocm_version,
        }
    )
    _write_summary(
        families,
        rocm_version,
        rocm_note,
        args.rocm_package_index_url,
        args.pytorch_git_ref,
    )


if __name__ == "__main__":
    main()
