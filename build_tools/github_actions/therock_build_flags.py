#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turns the CI ``build_flags`` input into CMake options.

The input is a comma-separated list of ``NAME=VALUE`` pairs, e.g.
``HIPDNN_ENABLE_SDPA=ON,KPACK_SPLIT_ARTIFACTS=OFF``. A bare ``NAME`` means
``NAME=ON`` for BOOL flags.

Names and values are checked against the flags declared in ``FLAGS.cmake``, so
a typo or a bad value fails the run instead of reaching CMake. Valid flags are
rendered as ``-DTHEROCK_FLAG_<NAME>=<VALUE>``.

See ``docs/development/flags.md``. Run from the repository root::

    python build_tools/github_actions/therock_build_flags.py \\
        --build-flags=HIPDNN_ENABLE_SDPA --cmake-args
"""

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

THEROCK_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FLAGS_CMAKE = THEROCK_DIR / "FLAGS.cmake"

# Prefix of the CMake cache variable created for each declared flag.
FLAG_CMAKE_PREFIX = "THEROCK_FLAG_"

# Keywords of therock_declare_flag() that take a single value.
_SINGLE_VALUE_KEYWORDS = frozenset(
    {"NAME", "TYPE", "DEFAULT_VALUE", "DESCRIPTION", "ISSUE"}
)
# Keywords that take one or more values.
_MULTI_VALUE_KEYWORDS = frozenset({"VALID_VALUES", "CMAKE_VARS", "SUB_PROJECTS"})
# Keywords that take no value at all.
_OPTION_KEYWORDS = frozenset({"GLOBAL_PROPAGATE_FLAG"})
_ALL_KEYWORDS = _SINGLE_VALUE_KEYWORDS | _MULTI_VALUE_KEYWORDS | _OPTION_KEYWORDS

_DECLARE_FLAG_RE = re.compile(r"therock_declare_flag\s*\(([^)]*)\)", re.DOTALL)
# A CMake argument is either a double-quoted string or a run of non-space
# characters.
_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\S+')

# CMake's truthy/falsey constants, normalized to ON/OFF.
_TRUE_VALUES = frozenset({"ON", "TRUE", "YES", "Y", "1"})
_FALSE_VALUES = frozenset({"OFF", "FALSE", "NO", "N", "0"})


class BuildFlagError(ValueError):
    """Raised when a requested build flag is unknown or invalid."""


@dataclass(frozen=True)
class FlagDeclaration:
    """A single flag as declared in FLAGS.cmake."""

    name: str
    type: str  # "BOOL" or "INTEGER"
    default_value: str
    valid_values: tuple[str, ...] = ()
    description: str = ""

    @property
    def cmake_variable(self) -> str:
        return f"{FLAG_CMAKE_PREFIX}{self.name}"


def _strip_quotes(token: str) -> str:
    if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
        return token[1:-1].replace('\\"', '"')
    return token


def _parse_declaration(body: str) -> FlagDeclaration | None:
    """Parse the argument list of a single therock_declare_flag() call."""
    tokens = [_strip_quotes(t) for t in _TOKEN_RE.findall(body)]
    single: dict[str, str] = {}
    multi: dict[str, list[str]] = {}

    index = 0
    current_multi: str | None = None
    while index < len(tokens):
        token = tokens[index]
        if token in _ALL_KEYWORDS:
            current_multi = None
            if token in _SINGLE_VALUE_KEYWORDS:
                if index + 1 < len(tokens):
                    single[token] = tokens[index + 1]
                index += 2
                continue
            if token in _MULTI_VALUE_KEYWORDS:
                current_multi = token
                multi.setdefault(token, [])
            index += 1
            continue
        if current_multi:
            multi[current_multi].append(token)
        index += 1

    name = single.get("NAME")
    if not name:
        return None
    return FlagDeclaration(
        name=name,
        # TYPE BOOL is the therock_declare_flag() default.
        type=single.get("TYPE", "BOOL").upper(),
        default_value=single.get("DEFAULT_VALUE", "OFF"),
        valid_values=tuple(multi.get("VALID_VALUES", ())),
        description=single.get("DESCRIPTION", ""),
    )


def load_flag_registry(
    flags_cmake: Path | None = None,
) -> dict[str, FlagDeclaration]:
    """Load the flag registry declared in FLAGS.cmake, keyed by flag name."""
    path = Path(flags_cmake) if flags_cmake else DEFAULT_FLAGS_CMAKE
    if not path.exists():
        raise BuildFlagError(f"Flag registry not found: {path}")
    contents = path.read_text(encoding="utf-8")
    registry: dict[str, FlagDeclaration] = {}
    for match in _DECLARE_FLAG_RE.finditer(contents):
        declaration = _parse_declaration(match.group(1))
        if declaration:
            registry[declaration.name] = declaration
    if not registry:
        raise BuildFlagError(f"No therock_declare_flag() declarations found in {path}")
    return registry


def _normalize_bool(declaration: FlagDeclaration, value: str) -> str:
    upper = value.upper()
    if upper in _TRUE_VALUES:
        return "ON"
    if upper in _FALSE_VALUES:
        return "OFF"
    raise BuildFlagError(
        f"Invalid value '{value}' for BOOL build flag '{declaration.name}'. "
        f"Expected one of: ON, OFF (also accepted: "
        f"{', '.join(sorted(_TRUE_VALUES | _FALSE_VALUES))})."
    )


def _normalize_integer(declaration: FlagDeclaration, value: str) -> str:
    try:
        parsed = int(value, 10)
    except ValueError:
        raise BuildFlagError(
            f"Invalid value '{value}' for INTEGER build flag "
            f"'{declaration.name}'. Expected an integer."
        ) from None
    if declaration.valid_values:
        allowed = []
        for candidate in declaration.valid_values:
            try:
                allowed.append(int(candidate, 10))
            except ValueError:
                continue
        if parsed not in allowed:
            raise BuildFlagError(
                f"Value '{value}' is not permitted for INTEGER build flag "
                f"'{declaration.name}'. Valid values: "
                f"{', '.join(str(v) for v in allowed)}."
            )
    return str(parsed)


def normalize_flag_value(declaration: FlagDeclaration, value: str) -> str:
    """Validate one value against its flag declaration."""
    if declaration.type == "INTEGER":
        return _normalize_integer(declaration, value)
    if declaration.type == "BOOL":
        return _normalize_bool(declaration, value)
    raise BuildFlagError(
        f"Build flag '{declaration.name}' has unsupported TYPE "
        f"'{declaration.type}'; cannot validate a requested value."
    )


def parse_build_flags(
    raw: str, registry: dict[str, FlagDeclaration] | None = None
) -> dict[str, str]:
    """Parse a ``build_flags`` input into validated ``{NAME: VALUE}`` pairs.

    Raises BuildFlagError for unknown names, duplicates, malformed entries, and
    values that do not match the flag's declared TYPE / VALID_VALUES.
    """
    if not raw or not raw.strip():
        return {}
    if registry is None:
        registry = load_flag_registry()

    resolved: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            name, _, value = entry.partition("=")
            name = name.strip()
            value = value.strip()
        else:
            name = entry
            value = ""

        if not name:
            raise BuildFlagError(
                f"Malformed build flag entry '{entry}'. Expected NAME=VALUE."
            )
        declaration = registry.get(name)
        if declaration is None:
            raise BuildFlagError(
                f"Unknown build flag '{name}'. Build flags must be declared in "
                f"FLAGS.cmake. Known flags: {', '.join(sorted(registry))}."
            )
        if not value:
            if declaration.type != "BOOL":
                raise BuildFlagError(
                    f"Build flag '{name}' has TYPE {declaration.type} and "
                    f"requires an explicit value (use '{name}=<value>')."
                )
            value = "ON"
        if name in resolved:
            raise BuildFlagError(
                f"Build flag '{name}' was specified more than once in the "
                f"build_flags input."
            )
        resolved[name] = normalize_flag_value(declaration, value)
    return resolved


def format_build_flags(flags: dict[str, str]) -> str:
    """Render resolved flags back to the ``NAME=VALUE,...`` form."""
    return ",".join(f"{name}={value}" for name, value in sorted(flags.items()))


def flag_cmake_args(flags: dict[str, str]) -> list[str]:
    """Render resolved flags as ``-DTHEROCK_FLAG_<NAME>=<VALUE>`` arguments.

    These are passed as their own CMake options, so they override the flag's
    declared default and any ``BRANCH_CONFIG.json`` entry.
    """
    return [
        f"-D{FLAG_CMAKE_PREFIX}{name}={value}" for name, value in sorted(flags.items())
    ]


def build_flags_suffix(flags: dict[str, str]) -> str:
    """Return a short, stable suffix identifying a set of flag overrides.

    Namespaces artifacts and uploads so a flag build does not collide with a
    default build. Empty when no flags are requested.
    """
    if not flags:
        return ""
    canonical = format_build_flags(flags)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return f"flags-{digest}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-flags",
        default=None,
        help="Comma-separated NAME=VALUE list. Defaults to $BUILD_FLAGS.",
    )
    parser.add_argument(
        "--flags-cmake",
        type=Path,
        default=DEFAULT_FLAGS_CMAKE,
        help="Path to the FLAGS.cmake registry.",
    )
    parser.add_argument(
        "--cmake-args",
        action="store_true",
        help="Print the resolved -DTHEROCK_FLAG_<NAME>=<VALUE> arguments.",
    )
    parser.add_argument(
        "--suffix",
        action="store_true",
        help="Print the identity suffix for the resolved flags.",
    )
    parser.add_argument(
        "--gha-output",
        action="store_true",
        help="Write cmake_args/suffix/flags to $GITHUB_OUTPUT.",
    )
    args = parser.parse_args(argv)

    raw = args.build_flags
    if raw is None:
        raw = os.environ.get("BUILD_FLAGS", "")

    registry = load_flag_registry(args.flags_cmake)
    try:
        flags = parse_build_flags(raw, registry)
    except BuildFlagError as e:
        print(f"::error::Invalid build_flags input: {e}", file=sys.stderr)
        return 1

    cmake_args = " ".join(flag_cmake_args(flags))
    if args.cmake_args:
        print(cmake_args)
    if args.suffix:
        print(build_flags_suffix(flags))

    if args.gha_output:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from github_actions_api import gha_set_output

        gha_set_output(
            {
                "cmake_args": cmake_args,
                "suffix": build_flags_suffix(flags),
                "flags": format_build_flags(flags),
            }
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
