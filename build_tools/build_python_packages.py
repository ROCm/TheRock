#!/usr/bin/env python
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

    # Simple populate the top-level "rocm" package. This gets no platform files.
    PopulatedDistPackage(params, logical_name="meta")

    # Populate each target neutral library package.
    core = PopulatedDistPackage(params, logical_name="core").populate_runtime_files(
        params.filter_artifacts(
            core_artifact_filter,
            # TODO: The base package is shoving CMake redirects into lib.
            excludes=["**/cmake/**"],
        ),
    )

    # Populate each target-specific library package.
    for target_family in params.all_target_families:
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

    # And populate the devel package, which catches everything else.
    devel = PopulatedDistPackage(params, logical_name="devel")
    devel.populate_devel_files(
        addl_artifact_names=[
            # Since prim is a header only library, it is not included in runtime
            # packages, but we still want it in the devel package.
            "prim",
        ],
        tarball_compression=args.devel_tarball_compression,
    )

    if args.build_packages:
        build_packages(args.dest_dir, wheel_compression=args.wheel_compression)


def core_artifact_filter(an: ArtifactName) -> bool:
    core = an.name in [
        "amd-llvm",
        "base",
        "core-hip",
        "core-hipinfo",
        "core-runtime",
        "host-blas",
        "host-suite-sparse",
        "rocprofiler-sdk",
        "sysdeps",
    ] and an.component in [
        "lib",
        "run",
    ]
    # hiprtc needs to be able to find HIP headers in its same tree.
    hip_dev = an.name in [
        "core-hip",
    ] and an.component in ["dev"]
    return core or hip_dev


def libraries_artifact_filter(target_family: str, an: ArtifactName) -> bool:
    libraries = (
        an.name
        in [
            "blas",
            "fft",
            "miopen",
            "rand",
            "rccl",
        ]
        and an.component
        in [
            "lib",
        ]
        and an.target_family == target_family
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
    p.add_argument("--version", default="0.1.dev0", help="Package versions")
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
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
