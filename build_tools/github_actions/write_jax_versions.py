#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Writes jaxlib_version, jax_plugin_version, jax_pjrt_version, and jax_version to GITHUB_OUTPUT.

Fails if required JAX wheels (jax_rocm7_plugin, jax_rocm7_pjrt) are not found.

For JAX <= 0.9.0, jaxlib is built and expected in the wheelhouse.
For JAX >= 0.9.1, jaxlib is not built - it is installed from upstream PyPI
(e.g. `pip install jaxlib==0.9.1`). Only jax_rocm7_plugin and jax_rocm7_pjrt
are built.

Expected wheels:
* jaxlib (not built for JAX >= 0.9.1)
* jax_rocm7_plugin
* jax_rocm7_pjrt

The jax_version output is the base version (without the +rocm suffix),
suitable for installing the `jax` and `jaxlib` packages from PyPI.
"""

import argparse
import glob
import os
from github_actions_api import *


def _log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def parse_version_from_wheel(wheel_path: str) -> str:
    return wheel_path.split("-")[1]


def get_wheel_version(package_dist_dir: Path, wheel_name: str) -> str | None:
    _log(f"Looking for '{wheel_name}' in '{package_dist_dir}'")
    wheel_glob_pattern = f"{wheel_name}-*.whl"
    wheel_paths = glob.glob(wheel_glob_pattern, root_dir=package_dist_dir)

    if len(wheel_paths) == 0:
        _log(
            f"  WARNING: Found no '{wheel_name}' wheels matching '{wheel_glob_pattern}'"
        )
        return None
    elif len(wheel_paths) != 1:
        _log(
            f"  WARNING: Found multiple '{wheel_name}' wheels matching '{wheel_glob_pattern}', using the first from {wheel_paths}"
        )
    wheel_path = wheel_paths[0]
    _log(f"  Found wheel at '{wheel_path}'")
    wheel_version = parse_version_from_wheel(wheel_path)
    _log(f"  Parsed version '{wheel_version}'")
    return wheel_version


def get_all_jax_wheel_versions(
    package_dist_dir: Path,
) -> Mapping[str, str | Path]:
    _log(f"Looking for wheels in '{package_dist_dir}'")
    all_files = list(package_dist_dir.glob("*"))
    _log("Found files in that directory:")
    for file in all_files:
        _log(f"  {file}")

    _log("")
    all_versions = {}
    jaxlib_version = get_wheel_version(package_dist_dir, "jaxlib")
    jax_plugin_version = get_wheel_version(package_dist_dir, "jax_rocm7_plugin")
    jax_pjrt_version = get_wheel_version(package_dist_dir, "jax_rocm7_pjrt")
    _log("")

    if jaxlib_version:
        all_versions = all_versions | {"jaxlib_version": jaxlib_version}
    else:
        _log(
            "INFO: No jaxlib wheel found. For JAX >= 0.9.1, jaxlib is not built - use upstream PyPI."
        )

    if jax_plugin_version:
        all_versions = all_versions | {"jax_plugin_version": jax_plugin_version}
    else:
        raise FileNotFoundError("Did not find jax_rocm7_plugin wheel")

    if jax_pjrt_version:
        all_versions = all_versions | {"jax_pjrt_version": jax_pjrt_version}
    else:
        raise FileNotFoundError("Did not find jax_rocm7_pjrt wheel")

    # Assumption: the jax_rocm7_plugin version (e.g. "0.9.1+rocmXY") shares the
    # same base version as the upstream `jax` / `jaxlib` PyPI packages.  If the
    # plugin version scheme ever diverges, this fallback will silently produce a
    # wrong jax_version and will need an explicit override.
    base_version_source = jaxlib_version or jax_plugin_version
    all_versions = all_versions | {"jax_version": base_version_source.split("+")[0]}

    return all_versions


def main(argv: list[str]):
    env_dist_dir = os.getenv("PACKAGE_DIST_DIR")
    p = argparse.ArgumentParser(prog="write_jax_versions.py")
    p.add_argument(
        "--dist-dir",
        type=Path,
        default=Path(env_dist_dir if not env_dist_dir == None else "<no-valid-dir>"),
        help="Path where wheels are located",
    )

    args = p.parse_args(argv)
    if args.dist_dir == Path("<no-valid-dir>"):
        print(
            f"""[ERROR] No path given where to find the wheels!
        Either set environment variable 'PACKAGE_DIST_DIR' or run the command with --dist-dir=<path-to-wheels>""",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.dist_dir.exists():
        raise FileNotFoundError(f"Dist dir '{args.dist_dir}' does not exist")
    all_versions = get_all_jax_wheel_versions(args.dist_dir)
    _log("")
    gha_set_output(all_versions)


if __name__ == "__main__":
    main(sys.argv[1:])
