# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parses AMDGPU target and family definitions from cmake/therock_amdgpu_targets.cmake.

Provides a Python-accessible mapping of family names to constituent gfx targets,
derived from the authoritative CMake source of truth.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_CMAKE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "cmake"
    / "therock_amdgpu_targets.cmake"
)

_family_to_targets_cache: dict[Path, dict[str, list[str]]] = {}


@dataclass
class AmdgpuTargetInfo:
    gfx_target: str
    product_name: str
    families: list[str] = field(default_factory=list)


def parse_amdgpu_targets_cmake(cmake_path: Path) -> list[AmdgpuTargetInfo]:
    """Parse therock_add_amdgpu_target() calls from a cmake file.

    Returns one AmdgpuTargetInfo per call. The families list contains only the
    explicit FAMILY arguments; the implicit self-family (each target is its own
    family) is NOT included here - see build_family_to_targets().
    """
    if not cmake_path.exists():
        raise FileNotFoundError(
            f"AMDGPU targets cmake file not found: {cmake_path}\n"
            "Expected at cmake/therock_amdgpu_targets.cmake relative to repo root."
        )

    content = cmake_path.read_text()
    results: list[AmdgpuTargetInfo] = []

    # Each therock_add_amdgpu_target() call has the form:
    #   therock_add_amdgpu_target(gfx900 "Vega 10 / MI25" FAMILY dgpu-all gfx900-dgpu)
    # Parsing extracts:
    #   tokens[0]                    -> gfx target  (e.g. "gfx900")
    #   tokens[1]                    -> product name (e.g. "Vega 10 / MI25")
    #   tokens after FAMILY keyword  -> families list (e.g. ["dgpu-all", "gfx900-dgpu"])
    for call_body in re.findall(
        r"therock_add_amdgpu_target\((.*?)\)", content, re.DOTALL
    ):
        tokens = _tokenize_cmake(call_body)
        if len(tokens) < 2:
            continue

        gfx_target = tokens[0]
        product_name = tokens[1]

        families: list[str] = []
        if "FAMILY" in tokens:
            family_start = tokens.index("FAMILY") + 1
            # Collect until the next cmake keyword or end of tokens.
            cmake_keywords = {"EXCLUDE_TARGET_PROJECTS"}
            for tok in tokens[family_start:]:
                if tok in cmake_keywords:
                    break
                families.append(tok)

        results.append(AmdgpuTargetInfo(gfx_target, product_name, families))

    return results


def build_family_to_targets(infos: list[AmdgpuTargetInfo]) -> dict[str, list[str]]:
    """Build a mapping of family name -> list of gfx targets.

    Includes the implicit self-family: each target is always its own family,
    matching the CMake behaviour in therock_add_amdgpu_target().
    """
    result: dict[str, list[str]] = {}
    for info in infos:
        # The target is its own family plus all explicit FAMILY entries.
        for family in [info.gfx_target] + info.families:
            result.setdefault(family, [])
            if info.gfx_target not in result[family]:
                result[family].append(info.gfx_target)
    return result


def amdgpu_family_map(
    cmake_path: Path | None = None,
) -> dict[str, list[str]]:
    """Return the family → gfx targets map, parsing the CMake file once.

    Uses `cmake/therock_amdgpu_targets.cmake` at the repo root by default.
    Results are cached per resolved path so repeated calls are free.
    """
    resolved = (cmake_path or _DEFAULT_CMAKE_PATH).resolve()
    cached = _family_to_targets_cache.get(resolved)
    if cached is None:
        cached = build_family_to_targets(parse_amdgpu_targets_cmake(resolved))
        _family_to_targets_cache[resolved] = cached
    return cached


def expand_families(
    families: list[str],
    family_map: dict[str, list[str]],
    *,
    strict: bool = True,
) -> list[str]:
    """Expand a list of family names to the union of their gfx targets.

    Preserves input order; de-duplicates. When `strict` (the default),
    raises ValueError if any family is not present in `family_map`. With
    `strict=False`, unknown families are silently skipped.
    """
    if strict:
        unknown = [f for f in families if f not in family_map]
        if unknown:
            known = ", ".join(sorted(family_map.keys()))
            raise ValueError(
                f"Unknown AMD GPU families: {unknown}. Known families: {known}"
            )

    targets: list[str] = []
    seen: set[str] = set()
    for family in families:
        for target in family_map.get(family, []):
            if target not in seen:
                seen.add(target)
                targets.append(target)
    return targets


def _tokenize_cmake(text: str) -> list[str]:
    """Tokenize a cmake argument list, stripping comments and handling quotes."""
    # Strip inline comments (# to end of line).
    lines = []
    for line in text.splitlines():
        comment_pos = line.find("#")
        if comment_pos >= 0:
            line = line[:comment_pos]
        lines.append(line)

    # Tokenize: quoted strings or sequences of non-whitespace non-quote chars.
    tokens = re.findall(r'"[^"]*"|[^\s"]+', " ".join(lines))
    # Strip surrounding quotes from quoted tokens.
    return [t[1:-1] if t.startswith('"') else t for t in tokens]
