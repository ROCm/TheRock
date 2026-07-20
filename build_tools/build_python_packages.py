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
import json
from pathlib import Path
import sys

from _therock_utils.artifacts import ArtifactCatalog, ArtifactName
from _therock_utils.cmake_amdgpu_targets import amdgpu_family_map, expand_families
from _therock_utils.py_packaging import Parameters, PopulatedDistPackage, build_packages


def _amdgpu_families_arg(value: str) -> list[str] | None:
    """Argparse type for --linux/windows-amdgpu-families CLI flags.

    Accepts both comma and semicolon separators so the same value shape
    works from a shell prompt and from workflow YAML that passes
    BuildConfig.dist_amdgpu_families (semicolon-separated) directly.

    Returns None for empty/whitespace-only input so workflow YAML can
    pass an empty string for "platform not participating" without
    triggering cross-platform mode in Parameters.
    """
    parsed = [f.strip() for f in value.replace(";", ",").split(",") if f.strip()]
    return parsed or None


def load_therock_manifest(artifact_dir: Path) -> dict:
    """Load therock_manifest.json from the base_lib_generic artifact."""
    manifest_path = (
        artifact_dir
        / "base_lib_generic"
        / "base"
        / "aux-overlay"
        / "stage"
        / "share"
        / "therock"
        / "therock_manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"therock_manifest.json not found at {manifest_path}. "
            f"Is base_lib_generic present in {artifact_dir}?"
        )
    return json.loads(manifest_path.read_text())


def ensure_profiler_library_symlinks(profiler: PopulatedDistPackage) -> None:
    """Recreate unversioned profiler library symlinks expected by dlopen()."""
    profiler_lib_dir = profiler.platform_dir / "lib"

    for target in profiler_lib_dir.glob("librocprof-sys*.so.*"):
        if target.is_symlink():
            continue
        link = target.with_suffix("")
        if not link.exists():
            link.symlink_to(target.name)


def _platform_targets(
    *,
    linux_targets: list[str] | None,
    windows_targets: list[str] | None,
    platform_name: str,
) -> list[str] | None:
    if platform_name.startswith("linux"):
        return linux_targets
    if platform_name == "win32":
        return windows_targets
    return None


def validate_kpack_split_target_completeness(
    *,
    kpack_split: bool,
    artifact_dir: Path,
    artifacts: ArtifactCatalog,
    linux_targets: list[str] | None,
    windows_targets: list[str] | None,
    platform_name: str = sys.platform,
) -> None:
    """Validate that kpack-split artifacts cover this platform's targets."""
    if not kpack_split:
        return

    expected_targets = _platform_targets(
        linux_targets=linux_targets,
        windows_targets=windows_targets,
        platform_name=platform_name,
    )
    if expected_targets is None:
        return

    expected_target_set = set(expected_targets)
    discovered_target_set = artifacts.all_target_families

    expected = ", ".join(sorted(expected_target_set)) or "(none)"
    discovered = ", ".join(sorted(discovered_target_set)) or "(none)"
    print(f"::: KPACK_SPLIT_ARTIFACTS expected device targets: {expected}")
    print(f"::: KPACK_SPLIT_ARTIFACTS discovered artifact targets: {discovered}")

    missing_targets = sorted(expected_target_set - discovered_target_set)
    if not missing_targets:
        return

    missing = ", ".join(missing_targets)
    raise RuntimeError(
        "KPACK_SPLIT_ARTIFACTS target completeness check failed: "
        f"missing fetched artifact targets: {missing}. "
        f"Expected targets: {expected}. "
        f"Discovered artifact targets in {artifact_dir}: {discovered}. "
        "The fetched/extracted artifact catalog is incomplete; refusing to "
        "build a partial device wheel set."
    )


def _has_devel_artifacts(artifacts: ArtifactCatalog) -> bool:
    return any(an.component == "dev" for an in artifacts.artifact_names)


def validate_required_dist_packages(
    *,
    dest_dir: Path,
    version: str,
    artifacts: ArtifactCatalog,
    kpack_split: bool,
    linux_targets: list[str] | None,
    windows_targets: list[str] | None,
    platform_name: str = sys.platform,
) -> None:
    """Validate required kpack-split files in the final dist directory."""
    if not kpack_split:
        return

    required_patterns = [
        f"rocm-{version}.tar.gz",
        f"rocm_sdk_core-{version}-*.whl",
        f"rocm_sdk_libraries-{version}-*.whl",
    ]

    expected_targets = _platform_targets(
        linux_targets=linux_targets,
        windows_targets=windows_targets,
        platform_name=platform_name,
    )
    for target in expected_targets or []:
        required_patterns.append(f"rocm_sdk_device_{target}-{version}-*.whl")

    if _has_devel_artifacts(artifacts):
        required_patterns.append(f"rocm_sdk_devel-{version}-*.whl")

    dist_dir = dest_dir / "dist"
    missing_patterns = [
        pattern for pattern in required_patterns if not list(dist_dir.glob(pattern))
    ]
    if not missing_patterns:
        return

    present_files = sorted(p.name for p in dist_dir.glob("*") if p.is_file())
    raise RuntimeError(
        "Required kpack-split Python packages are missing from "
        f"{dist_dir}: {', '.join(missing_patterns)}. "
        f"Present files: {', '.join(present_files) or '(none)'}. "
        "Refusing to publish an incomplete Python package set."
    )


def run(args: argparse.Namespace):
    manifest = load_therock_manifest(args.artifact_dir)
    kpack_split = manifest.get("flags", {}).get("KPACK_SPLIT_ARTIFACTS", False)
    if kpack_split:
        print("::: Detected KPACK_SPLIT_ARTIFACTS — producing host + device wheels")

    # Cross-platform target view for the rocm sdist. Expand each platform's
    # family list to the union of its gfx targets so the
    # AVAILABLE_TARGET_FAMILIES baked into the meta sdist matches what's
    # actually published as rocm-sdk-device-<target> wheels across both
    # platforms' builds.
    # TODO: kwarg name `linux_target_families` in Parameters and the
    # LINUX/WINDOWS_TARGET_FAMILIES constants in _dist_info.py predate the
    # kpack-split semantics where the values are GPU targets. Rename to
    # *_amdgpu_targets once the wider `target_family` terminology cleanup
    # is in scope.
    family_map = None
    linux_targets: list[str] | None = None
    windows_targets: list[str] | None = None
    if args.linux_amdgpu_families is not None:
        family_map = amdgpu_family_map()
        linux_targets = expand_families(args.linux_amdgpu_families, family_map)
    if args.windows_amdgpu_families is not None:
        if family_map is None:
            family_map = amdgpu_family_map()
        windows_targets = expand_families(args.windows_amdgpu_families, family_map)

    artifacts = ArtifactCatalog(args.artifact_dir)
    validate_kpack_split_target_completeness(
        kpack_split=kpack_split,
        artifact_dir=args.artifact_dir,
        artifacts=artifacts,
        linux_targets=linux_targets,
        windows_targets=windows_targets,
    )

    params = Parameters(
        dest_dir=args.dest_dir,
        version=args.version,
        version_suffix=args.version_suffix,
        artifacts=artifacts,
        kpack_split=kpack_split,
        linux_target_families=linux_targets,
        windows_target_families=windows_targets,
    )

    # Populate each target neutral library package.
    core = PopulatedDistPackage(params, logical_name="core")
    core.rpath_dep(core, "lib/llvm/lib")
    core.rpath_dep(core, "lib/rocm_sysdeps/lib")
    core.populate_runtime_files(
        params.filter_artifacts(
            core_artifact_filter,
            # TODO: The base package is shoving CMake redirects into lib.
            excludes=[
                "**/cmake/**",
                # profiler binaries
                "bin/rocprof-*",
                # rocprofiler-systems payload
                "include/rocprofiler-systems/**",
                "lib/librocprof-sys*",
                "lib/python/site-packages/rocprofsys/**",
                "lib/rocprofiler-systems/**",
                "libexec/rocprofiler-systems/**",
                "share/**/rocprofiler-systems/**",
            ],
        ),
    )

    profiler_artifacts = params.filter_artifacts(
        profiler_artifact_filter,
        includes=[
            # rocprofiler-systems
            "bin/rocprof-sys-*",
            "include/rocprofiler-systems/**",
            "lib/librocprof-sys*",
            "lib/python/site-packages/rocprofsys/**",
            "lib/rocprofiler-systems/**",
            "libexec/rocprofiler-systems/**",
            "share/**/rocprofiler-systems/**",
            # rocprofiler-compute
            "bin/rocprof-*",
            "libexec/rocprofiler-compute/**",
            "lib/rocprofiler-compute/**",
            "share/**/rocprofiler-compute/**",
        ],
    )

    if profiler_artifacts.artifact_names:
        profiler = PopulatedDistPackage(params, logical_name="profiler")
        profiler.rpath_dep(core, "lib")
        profiler.rpath_dep(core, "lib/llvm/lib")
        profiler.rpath_dep(core, "lib/rocm_sysdeps/lib")
        profiler.populate_runtime_files(profiler_artifacts)
        ensure_profiler_library_symlinks(profiler)

        # The rocprofiler-compute artifact installs the launcher as a symlink:
        # bin/rocprof-compute -> ../libexec/rocprofiler-compute/rocprof-compute
        # However, populate_runtime_files() does not preserve symlinks and only
        # materializes the real file under libexec/. Recreate the expected bin/
        # entry here so CLI entrypoints (_exec("bin/rocprof-compute")) continue to work.
        compute_target = (
            profiler.platform_dir
            / "libexec"
            / "rocprofiler-compute"
            / "rocprof-compute"
        )
        compute_link = profiler.platform_dir / "bin" / "rocprof-compute"

        if compute_target.exists() and not compute_link.exists():
            compute_link.parent.mkdir(parents=True, exist_ok=True)
            compute_link.symlink_to("../libexec/rocprofiler-compute/rocprof-compute")
    elif sys.platform == "win32":
        print(
            "::: No profiler artifacts found on Windows; skipping rocm-profiler package"
        )
    else:
        raise RuntimeError(
            "No profiler artifacts found; refusing to build an empty rocm-profiler package"
        )

    if kpack_split:
        _run_kpack_split(args, params, core)
    else:
        _run_legacy(args, params, core)

    if args.build_packages:
        validate_required_dist_packages(
            dest_dir=args.dest_dir,
            version=args.version,
            artifacts=artifacts,
            kpack_split=kpack_split,
            linux_targets=linux_targets,
            windows_targets=windows_targets,
        )

    print(
        f"::: Finished building packages at '{args.dest_dir}' with version '{args.version}'"
    )


def _run_kpack_split(
    args: argparse.Namespace, params: Parameters, core: PopulatedDistPackage
):
    """Kpack-split mode: arch-neutral host libraries + per-ISA device wheels."""

    # Single arch-neutral libraries wheel from generic artifacts only.
    lib = PopulatedDistPackage(params, logical_name="libraries", target_family=None)
    lib.rpath_dep(core, "lib")
    lib.rpath_dep(core, "lib/rocm_sysdeps/lib")
    lib.rpath_dep(core, "lib/host-math/lib")
    lib.populate_runtime_files(
        params.filter_artifacts(
            filter=functools.partial(libraries_artifact_filter, "generic"),
        )
    )

    # Build core + libraries wheels. The rocm, rocm-sdk-devel, and
    # rocm-sdk-device staging dirs do not exist yet, so the default scan
    # in build_packages will not accidentally include them.
    if args.build_packages:
        build_packages(args.dest_dir, wheel_compression=args.wheel_compression)

    # Per-ISA device wheels. Device artifacts overlay into
    # _rocm_sdk_libraries/lib/ and may include ELF .so files (per-arch
    # MIOpen CK kernels) with dynamic deps on core.
    all_targets = sorted(params.all_target_families)
    for target in all_targets:
        dev = PopulatedDistPackage(params, logical_name="device", target_family=target)
        dev.rpath_dep(core, "lib")
        dev.rpath_dep(core, "lib/rocm_sysdeps/lib")
        dev.populate_device_files(
            params.filter_artifacts(
                filter=functools.partial(device_artifact_filter, target),
            )
        )
        if args.build_packages:
            build_packages(
                args.dest_dir,
                package_dirs=[dev.path],
                wheel_compression=args.wheel_compression,
            )

    # Single generic meta sdist.
    meta = PopulatedDistPackage(params, logical_name="meta", target_family=None)
    if args.build_packages:
        build_packages(
            args.dest_dir,
            package_dirs=[meta.path],
            wheel_compression=args.wheel_compression,
        )

    # Single arch-neutral devel wheel. Exclude test component — in kpack-split
    # mode the generic test binaries are host-only and can't run without device
    # code. Tests will be reintroduced via a dedicated package later.
    devel = PopulatedDistPackage(params, logical_name="devel", target_family=None)
    devel.populate_devel_files(
        addl_artifact_names=[
            # Header-only libraries not included in runtime packages.
            "prim",
            "rocwmma",
            "libhipcxx",
            # Third party dependencies needed by hipDNN consumers.
            "flatbuffers",
            "nlohmann-json",
            # rocshmem only provides a static library.
            "rocshmem",
            # rocjitsu emulation suite.
            "rocjitsu",
            "mirage",
        ],
        exclude_components=["test"],
        tarball_compression=args.devel_tarball_compression,
    )
    if args.build_packages:
        build_packages(
            args.dest_dir,
            package_dirs=[devel.path],
            wheel_compression=args.wheel_compression,
        )


def _run_legacy(
    args: argparse.Namespace, params: Parameters, core: PopulatedDistPackage
):
    """Legacy mode: per-family libraries wheels with embedded device code."""

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
                # rocjitsu emulation suite.
                "rocjitsu",
                "mirage",
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


def core_artifact_filter(an: ArtifactName) -> bool:
    core = an.name in [
        "amd-dbgapi",
        "amd-llvm",
        "aqlprofile",
        "base",
        "core-amdsmi",
        "core-hip",
        "core-kpack",
        "core-ocl",
        "core-hipinfo",
        "core-runtime",
        "hipfile",
        "hipify",
        "host-blas",
        "host-suite-sparse",
        "rocdecode",
        "rocgdb",
        "rocjpeg",
        "rocprofiler-sdk",
        "rocr-debug-agent",
        "sysdeps",
        "sysdeps-amd-mesa",
        "sysdeps-expat",
        "sysdeps-gmp",
        "sysdeps-mpfr",
        "sysdeps-ncurses",
        "sysdeps-util-linux",
        "wsl-rocdxg",
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
            "hipkernelprovider",
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


def profiler_artifact_filter(an: ArtifactName) -> bool:
    return an.name in [
        "rocprofiler-compute",
        "rocprofiler-systems",
    ] and an.component in ["lib", "run"]


def device_artifact_filter(target: str, an: ArtifactName) -> bool:
    """Selects per-ISA library artifacts for a specific GFX target.

    Unlike libraries_artifact_filter, this only matches the specific ISA target
    (no generic). Used in kpack-split mode for device wheel population.
    """
    return (
        an.name
        in [
            "blas",
            "fft",
            "hipdnn",
            "miopen",
            "miopenprovider",
            "hipblasltprovider",
            "hipkernelprovider",
            "rand",
            "rccl",
        ]
        and an.component == "lib"
        and an.target_family == target
    )


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
    p.add_argument(
        "--linux-amdgpu-families",
        type=_amdgpu_families_arg,
        default=None,
        help=(
            "Comma- or semicolon-separated AMD GPU families for the Linux "
            "side of a multi-arch release (e.g. 'gfx94X-dcgpu,gfx110X-all'). "
            "Expanded to GPU targets via cmake/therock_amdgpu_targets.cmake "
            "and recorded in the rocm sdist so its device-gfx* extras "
            "advertise the cross-platform union. Absent on single-platform "
            "builds."
        ),
    )
    p.add_argument(
        "--windows-amdgpu-families",
        type=_amdgpu_families_arg,
        default=None,
        help=(
            "Comma- or semicolon-separated AMD GPU families for the Windows "
            "side of a multi-arch release. See --linux-amdgpu-families."
        ),
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
