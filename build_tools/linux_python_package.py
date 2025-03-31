#!/usr/bin/env python
"""Given ROCm artifacts directories, performs surgery to re-layout them for
distribution as Python packages and builds sdists and wheels as appropriate.

This process involves both a re-organization of the sources and some surgical
alterations.

Example
-------

```
./build_tools/linux_python_package.py \
    --artifact-dir ./output-linux-portable/build/artifacts \
    --dest-dir $HOME/tmp/packages
```

Note that this does do some dynamic compilation of files and it performs
patching via patchelf. It is recommended to run this in the same portable
Linux container as was used to build the SDK (so as to avoid the possibility
of accidentally referencing too-new glibc symbols).

General Procedure
-----------------
We generate three types of packages:

* Selector package: The `rocm-sdk` package is built as an sdist so that it is
  evaluated at the point of install, allowing it to perform some selection logic
  to detect dependencies and needs. Most other packages are installed by
  asking to install extras of this one (i.e. `rocm-sdk[libraries]`, etc).
* Runtime packages: Most packages are runtime packages. These are wheel files
  that are ready to be expanded directly into a site-lib directory. They contain
  only the minimal set of files needed to run. Critically, they do not contain
  symlinks (instead, only SONAME libraries are included, and any executable
  symlinks are emitted by dynamically compiling an executable that can execle
  a relative binary).
* Devel package: The `rocm-sdk-devel` package is the catch-all for everything.
  For any file already populated in a runtime package, it will include it as
  a relative symlink (also rewriting shared library soname links as needed).
  Since symlinks and non-standard attributes cannot be included in a wheel file,
  the platform contents are stored in a `_devel.tar` or `_devel.tar.xz` file.
  The installed package is extended in response to requesting a path to it
  via the `rocm-sdk` tool.

Runtime packages can either be target neutral or target specific. Target specific
packages are suffixed with their target family and the setup files have special logic
to determine what is correct to load based on the system. In the future, this
will be re-arranged to have more of a thin structure where the host code is
always emitted as target neutral and separate device code packages are loaded
as needed.

It is expected that all packages are installed in the same site-lib, as they use
relative symlinks and RPATHs that cross the top-level package boundary. The
built-in tests (via `rocm-sdk test`) verify these conditions.
"""

import argparse
import functools
import re
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

    # Simple populate the top-level "rocm-sdk" package. This gets no platform files.
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
    devel.populate_devel_files(tarball_compression=args.devel_tarball_compression)

    if args.build_packages:
        build_packages(args.dest_dir, wheel_compression=args.wheel_compression)


def core_artifact_filter(an: ArtifactName) -> bool:
    core = an.name in [
        "amd-llvm",
        "base",
        "core-hip",
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
            "prim",
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
