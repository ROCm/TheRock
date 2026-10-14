#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Teaches an installed JAX stack about a new ROCm major version.

The plugin wheels embed the ROCm major version in their package name
(jax_rocm7_plugin, jax_rocm10_plugin). Two places look that name up from a
hardcoded list of majors, so a newly bumped major is not found:

  * jaxlib/plugin_support.py holds _PLUGIN_MODULE_NAMES["rocm"], which gates the
    GPU kernel modules (_linalg, _solver, _sparse, _rnn, _prng, _triton,
    _hybrid). When the lookup misses, those modules resolve to None and tests
    fail with "'NoneType' object has no attribute ...". jaxlib is installed from
    upstream PyPI, so already released versions cannot be fixed at the source.

  * jax_plugins/xla_rocm<major>/__init__.py looks for its companion plugin
    package the same way. When that misses, rocm_plugin_extension stays None, no
    FFI handlers are registered, and tests fail with "No FFI handler registered
    for hipsolver_*". This one is fixed at the source in ROCm/jax, so the patch
    here is a no-op on wheels that already carry the fix.

Both patches are idempotent and rewrite the installed files before any test
imports them, and both detect the upstream fix so they no-op once a wheel
carrying it is installed. Every file is planned before any file is written, so
an unrecognized layout fails without leaving the install half patched. The
script fails loudly rather than silently skipping, so a missing patch cannot be
mistaken for a passing configuration.

Example usage:

    python patch_installed_jax_rocm_plugin_names.py --plugin-package jax_rocm10_plugin
"""

import argparse
import importlib.util
import pathlib
import re
import sys
from typing import Callable, NamedTuple

_PLUGIN_PACKAGE_RE = re.compile(r"^jax_rocm(\d+)_plugin$")

# _PLUGIN_MODULE_NAMES = {..., "rocm": ["jax_rocm7_plugin", ...], ...}
_JAXLIB_ROCM_LIST_RE = re.compile(r'("rocm"\s*:\s*\[)([^\]]*?)(\s*\])', re.DOTALL)

# Marker for the upstream fix that enumerates installed plugin packages, which
# covers every major and not just the one this run installs.
_JAXLIB_FIXED_MARKER = "_discovered_rocm_plugin_module_names"

# for pkg_name in ['jax_rocm7_plugin', 'jax_rocm60_plugin', 'jaxlib.rocm']:
_SHIM_LIST_RE = re.compile(r"(for pkg_name in \[)([^\]]*?)(\]\s*:)", re.DOTALL)

# Marker for the ROCm/jax fix that derives the major from the package name.
_SHIM_FIXED_MARKER = "rpartition('xla_rocm')"


class Rewrite(NamedTuple):
    """The new contents of one installed file, before anything is written."""

    # None when the file already resolves the plugin name correctly.
    text: str | None
    # One line describing either the change or the reason there is none.
    detail: str


class _PlannedEdit(NamedTuple):
    path: pathlib.Path
    text: str


def find_installed_module_file(module_name: str, file_name: str) -> pathlib.Path:
    """Locates a file inside an installed package without importing it."""
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(f"'{module_name}' is not installed")
    for location in spec.submodule_search_locations:
        candidate = pathlib.Path(location) / file_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"'{module_name}' does not contain '{file_name}'")


def _insert_into_list(
    text: str, match: re.Match, list_regex: re.Pattern, entry: str
) -> Rewrite:
    """Puts entry first in the matched list literal, preferring it on lookup."""
    patched = (
        text[: match.start()]
        + f"{match.group(1)}{entry}, {match.group(2).strip()}{match.group(3)}"
        + text[match.end() :]
    )
    after = list_regex.search(patched)
    return Rewrite(patched, f"{match.group(0).strip()} -> {after.group(0).strip()}")


def rewrite_jaxlib_plugin_names(text: str, plugin_package: str) -> Rewrite:
    """Adds plugin_package to jaxlib's ROCm plugin lookup list."""
    if _JAXLIB_FIXED_MARKER in text:
        return Rewrite(None, "already discovers installed ROCm plugin packages")

    match = _JAXLIB_ROCM_LIST_RE.search(text)
    if match is None:
        raise ValueError(
            "does not contain a '\"rocm\": [...]' plugin list. The upstream"
            " layout changed and this patch needs updating."
        )

    if plugin_package in match.group(2):
        return Rewrite(None, f"already lists {plugin_package}")

    return _insert_into_list(text, match, _JAXLIB_ROCM_LIST_RE, f'"{plugin_package}"')


def rewrite_pjrt_shim_plugin_names(text: str, plugin_package: str) -> Rewrite:
    """Adds plugin_package to the PJRT shim's plugin lookup list."""
    if _SHIM_FIXED_MARKER in text:
        return Rewrite(None, "already derives the plugin name from the ROCm major")

    match = _SHIM_LIST_RE.search(text)
    if match is None:
        raise ValueError(
            "does not contain a 'for pkg_name in [...]' plugin list. The layout"
            " changed and this patch needs updating."
        )

    if plugin_package in match.group(2):
        return Rewrite(None, f"already lists {plugin_package}")

    return _insert_into_list(text, match, _SHIM_LIST_RE, f"'{plugin_package}'")


def _plan_edit(
    path: pathlib.Path,
    rewrite: Callable[[str, str], Rewrite],
    plugin_package: str,
) -> _PlannedEdit | None:
    """Computes the new contents of path without writing them."""
    try:
        result = rewrite(path.read_text(), plugin_package)
    except ValueError as e:
        raise ValueError(f"{path} {e}") from e

    print(f"  {path}: {result.detail}")
    return None if result.text is None else _PlannedEdit(path, result.text)


def plan_jaxlib_edit(plugin_package: str) -> _PlannedEdit | None:
    path = find_installed_module_file("jaxlib", "plugin_support.py")
    return _plan_edit(path, rewrite_jaxlib_plugin_names, plugin_package)


def plan_pjrt_shim_edit(plugin_package: str, rocm_major: str) -> _PlannedEdit | None:
    module_name = f"jax_plugins.xla_rocm{rocm_major}"
    path = find_installed_module_file(module_name, "__init__.py")
    return _plan_edit(path, rewrite_pjrt_shim_plugin_names, plugin_package)


def main(argv: list[str]):
    p = argparse.ArgumentParser(prog="patch_installed_jax_rocm_plugin_names.py")
    p.add_argument(
        "--plugin-package",
        required=True,
        type=str,
        help="Installed plugin package name (e.g. jax_rocm10_plugin)",
    )
    p.add_argument(
        "--skip-pjrt-shim",
        action="store_true",
        help="Only patch jaxlib, leaving the PJRT shim as installed",
    )

    args = p.parse_args(argv)

    match = _PLUGIN_PACKAGE_RE.match(args.plugin_package)
    if match is None:
        p.error(
            f"--plugin-package '{args.plugin_package}' is not of the form"
            " jax_rocm<major>_plugin"
        )
    rocm_major = match.group(1)

    if importlib.util.find_spec(args.plugin_package) is None:
        raise FileNotFoundError(
            f"'{args.plugin_package}' is not installed, so there is nothing to"
            " point the installed JAX stack at"
        )

    print(f"Patching the installed JAX stack for {args.plugin_package}:")

    # Both files are planned before either is written: a layout this patch does
    # not recognize raises here, while the install is still untouched.
    planned = [plan_jaxlib_edit(args.plugin_package)]
    if not args.skip_pjrt_shim:
        planned.append(plan_pjrt_shim_edit(args.plugin_package, rocm_major))

    edits = [edit for edit in planned if edit is not None]
    for edit in edits:
        edit.path.write_text(edit.text)

    print(f"Patched {len(edits)} file(s)." if edits else "Nothing to patch.")


if __name__ == "__main__":
    main(sys.argv[1:])
