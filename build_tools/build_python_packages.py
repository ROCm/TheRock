#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Given ROCm artifacts directories, performs surgery to re-layout them for
distribution as Python packages and builds sdists and wheels as appropriate.

Under Linux, it is standard to run this under an appropriate manylinux container
for producing portable binaries. On Windows, it can be run natively.

See docs/packaging/python_packaging.md for more information.

Example
-------

```
./build_tools/build_python_packages.py \
    --artifact-dir ./output-linux-portable/build/artifacts \
    --dest-dir $HOME/tmp/packages
```
"""

import argparse
import functools
from pathlib import Path
import sys

from _therock_utils.artifacts import ArtifactCatalog, ArtifactName
from _therock_utils.py_packaging import Parameters, PopulatedDistPackage, build_packages


def run(args: argparse.Namespace):
    params = Parameters(
        dest_dir=args.dest_dir,
        version=args.version,
        version_suffix=args.version_suffix,
        artifacts=ArtifactCatalog(args.artifact_dir),
    )

    # Populate each target neutral library package.
    core = PopulatedDistPackage(params, logical_name="core")
    core.rpath_dep(core, "lib/llvm/lib")
    core.populate_runtime_files(
        params.filter_artifacts(
            core_artifact_filter,
            # TODO: The base package is shoving CMake redirects into lib.
            excludes=["**/cmake/**"],
        ),
    )

    # Populate each target-specific library package.
    for target_family in sorted(params.all_target_families):
        lib = PopulatedDistPackage(
            params, logical_name="libraries", target_family=target_family
        )
        lib.rpath_dep(core, "lib")
        lib.rpath_dep(core, "lib/rocm_sysdeps/lib")
        lib.rpath_dep(core, "lib/host-math/lib")
        lib.populate_runtime_files(
            params.filter_artifacts(
                filter=functools.partial(libraries_artifact_filter, target_family),
            )
        )

    # Compute these before the first build call so they can be shared with the
    # meta and devel loops below.
    all_target_families = sorted(params.all_target_families)
    multi_arch = len(all_target_families) > 1

    # Build non-devel, non-meta wheels first — the rocm and rocm-sdk-devel
    # staging dirs do not exist yet, so the default scan in build_packages
    # will not accidentally include them.
    if args.build_packages:
        build_packages(args.dest_dir, wheel_compression=args.wheel_compression)

    # One meta (rocm) sdist per target family. In a multi-arch build,
    # target_family and restrict_families=True bake THIS_TARGET_FAMILY,
    # DEFAULT_TARGET_FAMILY, and AVAILABLE_TARGET_FAMILIES for that family
    # into _dist_info.py so that determine_target_family() at install time
    # resolves only to that family's packages. In a single-arch build the
    # sdist is generic (target_family=None, no restriction) and goes
    # directly to dist/; in a multi-arch build each sdist goes to
    # dist/{target_family}/ so callers can distinguish them.
    for target_family in all_target_families:
        meta = PopulatedDistPackage(
            params,
            logical_name="meta",
            target_family=target_family if multi_arch else None,
            restrict_families=multi_arch,
        )
        if args.build_packages:
            build_packages(
                args.dest_dir,
                package_dirs=[meta.path],
                dist_dir=(
                    (args.dest_dir / "dist" / target_family) if multi_arch else None
                ),
                wheel_compression=args.wheel_compression,
            )

    # One rocm-sdk-devel wheel per target family. Each wheel is NOT generic:
    # shared libraries already materialized by the libraries runtime package
    # are embedded in the devel tarball as symlinks into that package's
    # arch-specific platform directory (e.g. _rocm_sdk_libraries_gfx120x_all),
    # so the tarball is only valid when the matching family's library wheel
    # is co-installed. In a multi-arch build each wheel goes to
    # dist/{target_family}/; in a single-arch build directly to dist/.
    for target_family in all_target_families:
        devel = PopulatedDistPackage(
            params, logical_name="devel", target_family=target_family
        )
        devel.populate_devel_files(
            addl_artifact_names=[
                # Since prim and rocwmma are header only libraries, they are not
                # included in runtime packages, but we still want them in the devel package.
                "prim",
                "rocwmma",
                # Third party dependencies needed by hipDNN consumers.
                "flatbuffers",
                "nlohmann-json",
            ],
            tarball_compression=args.devel_tarball_compression,
        )
        if args.build_packages:
            build_packages(
                args.dest_dir,
                package_dirs=[devel.path],
                dist_dir=(
                    (args.dest_dir / "dist" / target_family) if multi_arch else None
                ),
                wheel_compression=args.wheel_compression,
            )

    print(
        f"::: Finished building packages at '{args.dest_dir}' with version '{args.version}'"
    )


def core_artifact_filter(an: ArtifactName) -> bool:
    core = an.name in [
        "amd-dbgapi",
        "amd-llvm",
        "aqlprofile",
        "base",
        "core-amdsmi",
        "core-hip",
        "core-ocl",
        "core-hipinfo",
        "core-runtime",
        "hipify",
        "host-blas",
        "host-suite-sparse",
        "rocdecode",
        "rocgdb",
        "rocjpeg",
        "rocprofiler-sdk",
        "rocr-debug-agent",
        "rocr-debug-agent-tests",
        "sysdeps",
        "sysdeps-amd-mesa",
        "sysdeps-expat",
        "sysdeps-gmp",
        "sysdeps-mpfr",
        "sysdeps-ncurses",
    ] and an.component in [
        "lib",
        "run",
    ]
    # hiprtc needs to be able to find HIP headers in its same tree.
    hip_dev = an.name in [
        "core-hip",
        "core-ocl",
    ] and an.component in ["dev"]
    return core or hip_dev


def libraries_artifact_filter(target_family: str, an: ArtifactName) -> bool:
    libraries = (
        an.name
        in [
            "blas",
            "fft",
            "hipdnn",
            "miopen",
            "miopenprovider",
            "hipblasltprovider",
            "rand",
            "rccl",
        ]
        and an.component
        in [
            "lib",
        ]
        and (an.target_family == target_family or an.target_family == "generic")
    )
    return libraries


def main(argv: list[str]):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Source artifacts/ dir from a build",
    )
    p.add_argument(
        "--build-packages",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Build the resulting sdists/wheels",
    )
    p.add_argument(
        "--dest-dir",
        type=Path,
        required=True,
        help="Destination directory in which to materialize packages",
    )
    p.add_argument(
        "--version",
        default="",
        help="Package versions (defaults to an automatic dev version)",
    )
    p.add_argument(
        "--version-suffix",
        default="",
        help="Version suffix to append to package names on disk",
    )
    p.add_argument(
        "--devel-tarball-compression",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable compression of the devel tarball (slows build time but more efficient)",
    )
    p.add_argument(
        "--wheel-compression",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply compression when building wheels (disable for faster iteration or prior to recompression activities)",
    )
    args = p.parse_args(argv)

    if not args.version:
        print(f"::: Version not specified, choosing a default")
        import compute_rocm_package_version

        # Generate a default version like `7.10.0.dev0`.
        # This is a simple and predictable version, compared to using
        # `release_type="dev"`, which appends the git commit hash.
        args.version = compute_rocm_package_version.compute_version(
            custom_version_suffix=".dev0"
        )
        print(f"::: Version defaulting to {args.version}")

    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
