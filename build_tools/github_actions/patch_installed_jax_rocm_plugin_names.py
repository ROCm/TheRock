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
imports them. The script fails loudly rather than silently skipping, so a
missing patch cannot be mistaken for a passing configuration.

Example usage:

    python patch_installed_jax_rocm_plugin_names.py --plugin-package jax_rocm10_plugin
"""

import argparse
import importlib.util
import pathlib
import re
import sys

_PLUGIN_PACKAGE_RE = re.compile(r"^jax_rocm(\d+)_plugin$")

# _PLUGIN_MODULE_NAMES = {..., "rocm": ["jax_rocm7_plugin", ...], ...}
_JAXLIB_ROCM_LIST_RE = re.compile(r'("rocm"\s*:\s*\[)([^\]]*?)(\s*\])', re.DOTALL)

# for pkg_name in ['jax_rocm7_plugin', 'jax_rocm60_plugin', 'jaxlib.rocm']:
_SHIM_LIST_RE = re.compile(r"(for pkg_name in \[)([^\]]*?)(\]\s*:)", re.DOTALL)

# Marker for the ROCm/jax fix that derives the major from the package name.
_SHIM_FIXED_MARKER = "rpartition('xla_rocm')"


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


def patch_jaxlib_plugin_names(plugin_package: str) -> bool:
    """Adds plugin_package to jaxlib's ROCm plugin lookup list."""
    path = find_installed_module_file("jaxlib", "plugin_support.py")
    text = path.read_text()

    match = _JAXLIB_ROCM_LIST_RE.search(text)
    if match is None:
        raise ValueError(
            f"{path} does not contain a '\"rocm\": [...]' plugin list. The"
            " upstream layout changed and this patch needs updating."
        )

    if plugin_package in match.group(2):
        print(f"  {path}: already lists {plugin_package}")
        return False

    print(f"  {path}: {match.group(0).strip()}")
    patched = (
        text[: match.start()]
        + f'{match.group(1)}"{plugin_package}", {match.group(2).strip()}{match.group(3)}'
        + text[match.end() :]
    )
    path.write_text(patched)

    match = _JAXLIB_ROCM_LIST_RE.search(path.read_text())
    print(f"  {path}: -> {match.group(0).strip()}")
    return True


def patch_pjrt_shim(plugin_package: str, rocm_major: str) -> bool:
    """Adds plugin_package to the PJRT shim's plugin lookup list."""
    module_name = f"jax_plugins.xla_rocm{rocm_major}"
    path = find_installed_module_file(module_name, "__init__.py")
    text = path.read_text()

    if _SHIM_FIXED_MARKER in text:
        print(f"  {path}: already derives the plugin name from the ROCm major")
        return False

    match = _SHIM_LIST_RE.search(text)
    if match is None:
        raise ValueError(
            f"{path} does not contain a 'for pkg_name in [...]' plugin list. The"
            " layout changed and this patch needs updating."
        )

    if plugin_package in match.group(2):
        print(f"  {path}: already lists {plugin_package}")
        return False

    print(f"  {path}: {match.group(0).strip()}")
    patched = (
        text[: match.start()]
        + f"{match.group(1)}'{plugin_package}', {match.group(2).strip()}{match.group(3)}"
        + text[match.end() :]
    )
    path.write_text(patched)

    match = _SHIM_LIST_RE.search(path.read_text())
    print(f"  {path}: -> {match.group(0).strip()}")
    return True


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
    patched = patch_jaxlib_plugin_names(args.plugin_package)
    if not args.skip_pjrt_shim:
        patched |= patch_pjrt_shim(args.plugin_package, rocm_major)

    print("Patched." if patched else "Nothing to patch.")


if __name__ == "__main__":
    main(sys.argv[1:])
