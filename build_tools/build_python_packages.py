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
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile

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


def _asan_build_id_from_manifest(manifest: dict) -> str | None:
    """Derive a stable build ID from a nightly artifact manifest."""
    package_version = manifest.get("rocm_package_version", "")
    match = re.search(r"(?:a|rc)(\d{8,})$", package_version)
    return match.group(1) if match else None


def resolve_package_version(args: argparse.Namespace, manifest: dict) -> str:
    """Resolve the package version, enforcing ASAN/release isolation."""
    import compute_rocm_package_version

    is_asan = getattr(args, "asan", False)
    asan_build_id = getattr(args, "asan_build_id", None)
    version = getattr(args, "version", "")

    if asan_build_id and not is_asan:
        raise ValueError("--asan-build-id requires --asan")

    if not is_asan:
        if version:
            return version
        print("::: Version not specified, choosing a default")
        resolved = compute_rocm_package_version.compute_version(
            custom_version_suffix=".dev0"
        )
        print(f"::: Version defaulting to {resolved}")
        return resolved

    base_version = manifest.get("rocm_version")
    if not base_version:
        raise ValueError(
            "ASAN packaging requires rocm_version in therock_manifest.json"
        )

    if version:
        expected = re.compile(
            rf"^{re.escape(base_version)}\+asan\.[a-z0-9]+(?:\.[a-z0-9]+)*$"
        )
        if not expected.fullmatch(version):
            raise ValueError(
                "--asan requires a canonical ASAN wheel version matching "
                f"{base_version}+asan.<build-id>; got {version!r}"
            )
        return version

    if asan_build_id is None:
        asan_build_id = _asan_build_id_from_manifest(manifest)
    resolved = compute_rocm_package_version.compute_version(
        release_type="asan",
        override_base_version=base_version,
        asan_build_id=asan_build_id,
    )
    print(f"::: ASAN wheel version defaulting to {resolved}")
    return resolved


def find_asan_runtime_rpath(artifacts: ArtifactCatalog) -> str:
    """Find the Clang ASAN runtime directory in staged artifacts."""
    runtime_dirs: set[str] = set()
    for relpath, entry in artifacts.pm.matches():
        relpath = PurePosixPath(relpath)
        if entry.is_file() and relpath.match(
            "lib/llvm/lib/clang/*/lib/linux/libclang_rt.asan-*.so"
        ):
            runtime_dirs.add(relpath.parent.as_posix())

    if not runtime_dirs:
        raise RuntimeError(
            "--asan was requested but no shared Clang ASAN runtime was found "
            "in the input artifacts"
        )
    if len(runtime_dirs) != 1:
        raise RuntimeError(
            "ASAN artifacts contain multiple Clang runtime directories: "
            + ", ".join(sorted(runtime_dirs))
        )
    return runtime_dirs.pop()


def _elf_dynamic_info(path: Path) -> tuple[list[str], list[str]] | None:
    """Return an ELF's NEEDED entries and RPATH/RUNPATH entries."""
    try:
        with path.open("rb") as stream:
            elf = ELFFile(stream)
            dynamic = elf.get_section_by_name(".dynamic")
            if dynamic is None:
                return ([], [])
            needed: list[str] = []
            rpaths: list[str] = []
            for tag in dynamic.iter_tags():
                if tag.entry.d_tag == "DT_NEEDED":
                    needed.append(tag.needed)
                elif tag.entry.d_tag == "DT_RPATH":
                    rpaths.extend(tag.rpath.split(":"))
                elif tag.entry.d_tag == "DT_RUNPATH":
                    rpaths.extend(tag.runpath.split(":"))
            return needed, rpaths
    except (ELFError, OSError, ValueError):
        return None


def _rpath_resolves_directory(
    *,
    binary_path: Path,
    rpaths: list[str],
    expected_dir: Path,
    binary_platform_root: Path | None = None,
    expected_platform_root: Path | None = None,
) -> bool:
    """Checks an ELF RPATH against its directory in the installed wheel set.

    Package staging directories are isolated from each other, but the contents
    of every package's ``platform/`` directory are merged into one
    site-packages directory when the wheels are installed. When platform roots
    are supplied, project both paths into that common layout before resolving
    ``$ORIGIN``. Omitting them retains the direct filesystem check used for
    paths that are already in a merged layout.
    """
    if (binary_platform_root is None) != (expected_platform_root is None):
        raise ValueError(
            "binary_platform_root and expected_platform_root must be supplied together"
        )

    if binary_platform_root is not None and expected_platform_root is not None:
        install_root = Path("/site-packages")
        try:
            binary_path = install_root / binary_path.resolve().relative_to(
                binary_platform_root.resolve()
            )
            expected_dir = install_root / expected_dir.resolve().relative_to(
                expected_platform_root.resolve()
            )
        except ValueError:
            return False

    expected_dir = expected_dir.resolve()
    for rpath in rpaths:
        for origin_syntax in ("$ORIGIN", "${ORIGIN}"):
            if not rpath.startswith(origin_syntax):
                continue
            relative = rpath[len(origin_syntax) :].lstrip("/")
            candidate = Path(os.path.normpath(binary_path.parent / relative))
            if candidate.resolve() == expected_dir:
                return True
    return False


def validate_asan_runtime_resolution(
    *,
    core: PopulatedDistPackage,
    packages: list[PopulatedDistPackage],
    runtime_rpath: str,
    require_instrumented: bool,
) -> None:
    """Validate that packaged ASAN-linked ELFs resolve the core runtime."""
    runtime_dir = core.platform_dir / runtime_rpath
    runtimes = sorted(runtime_dir.glob("libclang_rt.asan-*.so"))
    if not runtimes:
        raise RuntimeError(
            f"ASAN runtime was not packaged in rocm-sdk-core at {runtime_dir}"
        )

    instrumented_count = 0
    unresolved: list[Path] = []
    for package in packages:
        for _, path in package.files.materialized_relpaths.values():
            if not path.is_file():
                continue
            dynamic_info = _elf_dynamic_info(path)
            if dynamic_info is None:
                continue
            needed, rpaths = dynamic_info
            if not any(name.startswith("libclang_rt.asan-") for name in needed):
                continue
            instrumented_count += 1
            if not _rpath_resolves_directory(
                binary_path=path,
                rpaths=rpaths,
                expected_dir=runtime_dir,
                binary_platform_root=package.platform_dir.parent,
                expected_platform_root=core.platform_dir.parent,
            ):
                unresolved.append(path)

    if unresolved:
        paths = "\n  ".join(str(path) for path in unresolved[:20])
        extra = (
            "" if len(unresolved) <= 20 else f"\n  ... ({len(unresolved) - 20} more)"
        )
        raise RuntimeError(
            "ASAN-linked packaged ELFs cannot resolve the rocm-sdk-core "
            f"runtime directory:\n  {paths}{extra}"
        )
    if require_instrumented and not instrumented_count:
        raise RuntimeError(
            "--asan was requested but no packaged ELF links the shared ASAN runtime"
        )
    print(
        "::: ASAN RPATH validation passed: "
        f"{instrumented_count} instrumented ELF(s), runtime {runtime_rpath}"
    )


def ensure_profiler_library_symlinks(profiler: PopulatedDistPackage) -> None:
    """Recreate unversioned library symlinks for profiler runtime dependencies."""
    profiler_lib_dir = profiler.platform_dir / "lib"

    for pattern in ("librocprof-sys*.so.*", "libprofiler-hub*.so.*"):
        for target in profiler_lib_dir.glob(pattern):
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
    args.version = resolve_package_version(args, manifest)
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
    asan_runtime_rpath = (
        find_asan_runtime_rpath(artifacts) if getattr(args, "asan", False) else None
    )
    if asan_runtime_rpath:
        print(f"::: ASAN runtime detected at {asan_runtime_rpath}")
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
    if asan_runtime_rpath:
        core.rpath_dep(core, asan_runtime_rpath)
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
        includes=PROFILER_WHEEL_INCLUDES,
    )

    if profiler_artifacts.artifact_names:
        profiler = PopulatedDistPackage(params, logical_name="profiler")
        profiler.rpath_dep(core, "lib")
        profiler.rpath_dep(core, "lib/llvm/lib")
        profiler.rpath_dep(core, "lib/rocm_sysdeps/lib")
        if asan_runtime_rpath:
            profiler.rpath_dep(core, asan_runtime_rpath)
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
        _run_kpack_split(args, params, core, asan_runtime_rpath)
    else:
        _run_legacy(args, params, core, asan_runtime_rpath)

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
    args: argparse.Namespace,
    params: Parameters,
    core: PopulatedDistPackage,
    asan_runtime_rpath: str | None,
):
    """Kpack-split mode: arch-neutral host libraries + per-ISA device wheels."""

    # Single arch-neutral libraries wheel from generic artifacts only.
    lib = PopulatedDistPackage(params, logical_name="libraries", target_family=None)
    lib.rpath_dep(core, "lib")
    lib.rpath_dep(core, "lib/rocm_sysdeps/lib")
    lib.rpath_dep(core, "lib/host-math/lib")
    # rpp needs libomp, which ships in core under lib/llvm/lib.
    lib.rpath_dep(core, "lib/llvm/lib")
    if asan_runtime_rpath:
        lib.rpath_dep(core, asan_runtime_rpath)
    lib.populate_runtime_files(
        params.filter_artifacts(
            filter=functools.partial(libraries_artifact_filter, "generic"),
        )
    )

    # Build core + libraries wheels. The rocm, rocm-sdk-devel, and
    # rocm-sdk-device staging dirs do not exist yet, so the default scan
    # in build_packages will not accidentally include them.
    if asan_runtime_rpath:
        validate_asan_runtime_resolution(
            core=core,
            packages=list(params.populated_packages),
            runtime_rpath=asan_runtime_rpath,
            require_instrumented=True,
        )
    if args.build_packages:
        build_packages(args.dest_dir, wheel_compression=args.wheel_compression)

    # Per-ISA device wheels. Device artifacts overlay into
    # _rocm_sdk_libraries/lib/ and may include ELF .so files (per-arch
    # MIOpen CK kernels) with dynamic deps on core.
    # Group by base target (strip xnack suffix) to merge variants like
    # 'gfx950' and 'gfx950:xnack+' into a single device package.
    all_base_targets = sorted(set(t.split(":")[0] for t in params.all_target_families))
    for target in all_base_targets:
        dev = PopulatedDistPackage(params, logical_name="device", target_family=target)
        dev.rpath_dep(core, "lib")
        dev.rpath_dep(core, "lib/rocm_sysdeps/lib")
        if asan_runtime_rpath:
            dev.rpath_dep(core, asan_runtime_rpath)
        dev.populate_device_files(
            params.filter_artifacts(
                filter=functools.partial(device_artifact_filter, target),
            )
        )
        if asan_runtime_rpath:
            validate_asan_runtime_resolution(
                core=core,
                packages=[dev],
                runtime_rpath=asan_runtime_rpath,
                require_instrumented=False,
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
            # hipthreads only provides a static library.
            "hipthreads",
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
    args: argparse.Namespace,
    params: Parameters,
    core: PopulatedDistPackage,
    asan_runtime_rpath: str | None,
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
        # rpp needs libomp, which ships in core under lib/llvm/lib.
        lib.rpath_dep(core, "lib/llvm/lib")
        if asan_runtime_rpath:
            lib.rpath_dep(core, asan_runtime_rpath)
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
    if asan_runtime_rpath:
        validate_asan_runtime_resolution(
            core=core,
            packages=list(params.populated_packages),
            runtime_rpath=asan_runtime_rpath,
            require_instrumented=True,
        )
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
    hotswap = an.name == "rocjitsu-hotswap" and an.component == "lib"
    # hiprtc needs to be able to find HIP headers in its same tree.
    hip_dev = an.name in [
        "core-hip",
        "core-ocl",
    ] and an.component in ["dev"]
    return core or hotswap or hip_dev


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
            "rpp",
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


# File-path allowlist for the rocm-profiler wheel, applied on top of
# profiler_artifact_filter(). rocprofiler-systems' own ProfilerHub.cmake
# vendors profiler-hub as a runtime .so dependency (NEEDED libprofiler-hub.so.0);
# it stages into the same lib/ dir as librocprof-sys* but needs its own glob
# entry here, or it gets silently dropped from the wheel.
PROFILER_WHEEL_INCLUDES = [
    # rocprofiler-systems
    "bin/rocprof-sys-*",
    "include/rocprofiler-systems/**",
    "lib/librocprof-sys*",
    "lib/libprofiler-hub.so*",
    "lib/python/site-packages/rocprofsys/**",
    "lib/rocprofiler-systems/**",
    "libexec/rocprofiler-systems/**",
    "share/**/rocprofiler-systems/**",
    # rocprofiler-compute
    "bin/rocprof-*",
    "libexec/rocprofiler-compute/**",
    "lib/rocprofiler-compute/**",
    "share/**/rocprofiler-compute/**",
]


def device_artifact_filter(target: str, an: ArtifactName) -> bool:
    """Selects per-ISA library artifacts for a specific GFX target.

    Unlike libraries_artifact_filter, this only matches the specific ISA target
    (no generic). Used in kpack-split mode for device wheel population.

    Matches both the base target and any xnack variants (e.g., target='gfx950'
    matches artifacts for both 'gfx950' and 'gfx950:xnack+'), merging them into
    a single device package.
    """
    # Strip xnack suffix from artifact's target_family for comparison
    artifact_base_target = an.target_family.split(":")[0]
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
        and artifact_base_target == target
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
        "--asan",
        default=False,
        action="store_true",
        help=(
            "Build locally versioned ASAN wheels, require the shared Clang "
            "ASAN runtime, and validate cross-wheel runtime RPATHs"
        ),
    )
    p.add_argument(
        "--asan-build-id",
        default=None,
        help=(
            "Build identifier for --asan (defaults to the source artifact's "
            "nightly date, then today's date)"
        ),
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

    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
