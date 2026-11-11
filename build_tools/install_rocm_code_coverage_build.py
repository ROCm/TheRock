#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Installs Non-Instrumental TheRock Build with Replaced Instrumental Component

This script does similar to install_rocm_from_artifacts.py for installation of Non-Instrumental Build.
Additionally it replaces the --replace-<comp-name> component files / libs.

Usage:
python build_tools/install_rocm_code_coverage_build.py
    [--code-coverage-run-id CODE_COVERAGE_RUN_ID]
    [--replace-rocblas | --no-replace-rocblas]
    [--replace-rocsolver | --no-replace-rocsolver]
    [** all supported options of install_rocm_from_artifacts.py]

"""

import os
import re
import sys
import argparse
import platform
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from install_rocm_from_artifacts import main as install_from_artifacts_main
from install_rocm_from_artifacts import log as log
from artifact_manager import (
    DownloadRequest,
    create_backend_from_env,
    download_artifact,
    find_available_artifacts,
)
from _therock_utils.archive_util import open_archive_for_read
from _therock_utils.artifacts import ArtifactName
from _therock_utils.cmake_amdgpu_targets import amdgpu_family_map, expand_families
from _therock_utils.elf_phdr import pin_phdr_table

# Maps each --replace-<name> flag to its TheRock artifact and library folder.
# e.g. rocBLAS ships in the 'blas' artifact (see BUILD_TOPOLOGY.toml).
COMPONENT_MAP = {
    # component: (artifact_name, library_folder),
    "rocfft": ("fft", "rocFFT"),
    "hipfft": ("fft", "hipFFT"),
    "hipblas": ("blas", "hipBLAS"),
    "rocblas": ("blas", "rocBLAS"),
    "hipblaslt": ("blas", "hipBLASLt"),
    "hipdnn": ("hipdnn", "hipDNN"),
    "hiprand": ("rand", "hipRAND"),
    "rocrand": ("rand", "rocRAND"),
    "hipsolver": ("solver", "hipSOLVER"),
    "rocsolver": ("solver", "rocSOLVER"),
    "hipsparse": ("sparse", "hipSPARSE"),
    "rocsparse": ("sparse", "rocSPARSE"),
    "hipsparselt": ("sparse", "hipSPARSELt"),
    "hiptensor": ("hiptensor", "hipTensor"),
    "hipthreads": ("hipthreads", "hipthreads"),
    "miopen": ("miopen", "MIOpen"),
    "rocalution": ("rocalution", "rocALUTION"),
    "hipcub": ("prim", "hipCUB"),
    "rocprim": ("prim", "rocPRIM"),
    "rocthrust": ("prim", "rocThrust"),
    "rocwmma": ("rocwmma", "rocWMMA"),
    "rpp": ("rpp", "rpp"),
    "composablekernel": ("composable-kernel", "composable_kernel"),
}

# Backend library folders a hip* wrapper calls at runtime that ship in the same
# artifact. They are swapped along with the wrapper so both come from one
# build: hipRAND's linkage test compares hiprandGetVersion() (answered by
# librocrand) against the ROCRAND_VERSION it was compiled with, and fails when
# the baseline's rocRAND is a different version than the coverage build's.
SAME_ARTIFACT_BACKENDS = {
    "hipblas": ("rocBLAS",),
    "hipfft": ("rocFFT",),
    "hiprand": ("rocRAND",),
    "hipsolver": ("rocSOLVER",),
    "hipsparse": ("rocSPARSE",),
}


def replacement_folders(components: Iterable[str]) -> dict[str, list[str]]:
    """Map each artifact to swap to the library folders to extract from it.

    A selected component brings its own folder plus its SAME_ARTIFACT_BACKENDS.
    """
    artifacts: dict[str, list[str]] = {}
    for comp in components:
        name, folder = COMPONENT_MAP[comp]
        folders = artifacts.setdefault(name, [])
        for f in (folder, *SAME_ARTIFACT_BACKENDS.get(comp, ())):
            if f not in folders:
                folders.append(f)
    return artifacts


# Host libraries a component links that ship in another artifact, as
# (artifact, library folder, file name prefix under lib/). Only these files are
# swapped, from the coverage run's generic lib archive: the baseline's copies
# come from a different source commit, and their C++ interfaces need not match
# the instrumented component's. hipSPARSELt picks and launches its kernels
# through hipBLASLt's tensilelite-host and origami; against the baseline's it
# aborts with heap corruption or "No Tensile solution found".
OTHER_ARTIFACT_HOST_LIBS = {
    "hipsparselt": (
        ("blas", "hipBLASLt", "libtensilelite-host"),
        ("blas", "origami", "liborigami"),
    ),
}


def host_library_deps(
    components: Iterable[str],
) -> dict[str, list[tuple[str, str]]]:
    """Map each artifact to the (library folder, file prefix) pairs to take from it."""
    deps: dict[str, list[tuple[str, str]]] = {}
    for comp in components:
        for artifact, folder, lib_prefix in OTHER_ARTIFACT_HOST_LIBS.get(comp, ()):
            entries = deps.setdefault(artifact, [])
            if (folder, lib_prefix) not in entries:
                entries.append((folder, lib_prefix))
    return deps


def _read_passthrough_options(passthrough_argv):
    """Read options shared with install_rocm_from_artifacts without consuming them.

    Parsed non-destructively so the same argv still forwards unchanged.
    """
    reader = argparse.ArgumentParser(add_help=False)
    # --amdgpu-family and --artifact-group share a dest, mirroring install_rocm_from_artifacts.
    reader.add_argument("--artifact-group", dest="artifact_group", default=None)
    reader.add_argument("--amdgpu-family", dest="artifact_group", default=None)
    reader.add_argument("--amdgpu-targets", default="")
    reader.add_argument("--output-dir", type=Path, default=Path("./therock-build"))
    reader.add_argument("--run-github-repo", default=None)
    known, _ = reader.parse_known_args(passthrough_argv)
    return known


def _target_families(family, amdgpu_targets):
    """Build the family match set: generic + family + expanded gfx targets.

    Target-specific artifacts (blas/solver) are named per family (mono-arch) or
    per gfx target (kpack-split), so match both spellings.
    """
    families = ["generic"]
    if family:
        families.append(family)
        families.extend(expand_families([family], amdgpu_family_map(), strict=False))
    families.extend(t.strip() for t in amdgpu_targets.split(",") if t.strip())
    return families


def download_replacement_artifacts(
    code_coverage_run_id,
    artifact_names,
    opts,
    code_coverage_run_github_repo,
    code_coverage_release_type,
    host_lib_artifacts=(),
):
    """Download instrumented replacement artifacts from the code-coverage run.

    Fetches every component tar matching the requested artifact names and family,
    plus only the generic lib tar of each host_lib_artifacts entry not already
    fetched whole.
    The coverage run lives in its own repo (--code-coverage-run-github-repo),
    separate from the generic baseline's --run-github-repo. Pass it explicitly:
    the S3 bucket lookup 404s if the run id is queried against the wrong repo.
    """
    backend = create_backend_from_env(
        run_id=code_coverage_run_id,
        github_repository=code_coverage_run_github_repo,
        platform=platform.system().lower(),
        release_type=code_coverage_release_type,
    )
    log(f"Fetching replacement artifacts from {backend.base_uri}")

    available = set(backend.list_artifacts())
    target_families = _target_families(opts.artifact_group, opts.amdgpu_targets)
    log(f"Matching artifacts {sorted(artifact_names)} for families {target_families}")

    matched = find_available_artifacts(artifact_names, target_families, available)
    if not matched:
        raise IOError(
            f"ERROR: No replacement artifacts {sorted(artifact_names)} found in "
            f"{backend.base_uri} for families {target_families}"
        )

    host_lib_names = set(host_lib_artifacts) - set(artifact_names)
    if host_lib_names:
        log(f"Matching host library artifacts {sorted(host_lib_names)} (generic lib)")
        host_matched = find_available_artifacts(
            host_lib_names,
            ["generic"],
            available,
            excluded_components={"dbg", "dev", "doc", "run", "test"},
        )
        if not host_matched:
            raise IOError(
                f"ERROR: No host library artifacts {sorted(host_lib_names)} found "
                f"in {backend.base_uri}"
            )
        matched += host_matched

    dest_dir = opts.output_dir / "code-coverage-replacements"
    dest_dir.mkdir(parents=True, exist_ok=True)

    log(f"Downloading {len(matched)} replacement artifact(s) to {dest_dir}:")
    for filename in matched:
        result = download_artifact(
            DownloadRequest(
                artifact_key=filename,
                dest_path=dest_dir / filename,
                backend=backend,
            )
        )
        if result is None:
            raise IOError(f"ERROR: Failed to download {filename}")

    return dest_dir


def _replace_scoped_member(tf, member, dest_path, output_dir, relpaths):
    """Write a single archive member into the flattened install tree.

    Mirrors TheRock's file/symlink/dir/hardlink flattening so replaced files
    keep their exec bits and link structure. Removes any existing file first.
    """
    if dest_path.is_symlink() or (dest_path.exists() and not dest_path.is_dir()):
        os.unlink(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if member.isfile():
        exec_mask = member.mode & 0o111
        with tf.extractfile(member) as member_file:
            with open(dest_path, "wb") as out_file:
                out_file.write(member_file.read())
                st = os.fstat(out_file.fileno())
                if hasattr(os, "fchmod"):  # Windows has no fchmod.
                    os.fchmod(out_file.fileno(), st.st_mode | exec_mask)
    elif member.isdir():
        dest_path.mkdir(parents=True, exist_ok=True)
    elif member.issym():
        dest_path.symlink_to(member.linkname)
    elif member.islnk():
        # Hardlink target is archive-relative; strip the manifest prefix to
        # resolve it against the already-written flattened file.
        for prefix in relpaths:
            prefix_slash = prefix + "/"
            if member.linkname.startswith(prefix_slash):
                target_scoped = member.linkname[len(prefix_slash) :]
                os.link(output_dir / PurePosixPath(target_scoped), dest_path)
                break
        else:
            raise IOError(
                f"Hardlink target not in manifest: {member} -> {member.linkname}"
            )
    else:
        raise IOError(f"Unhandled tar member: {member}")


def _names_folder(folder, path):
    """Whether `path` has a component naming `folder` (e.g. rocBLAS/, rocblas.so)."""
    return re.search(rf"{folder}[^a-zA-Z]", path, flags=re.IGNORECASE)


def _is_own_stage(folder, prefix):
    """Whether manifest relpath `prefix` is one of `folder`'s stage directories."""
    return bool(
        re.search(rf"(^|/){folder}[^a-zA-Z]", prefix + "/", flags=re.IGNORECASE)
    )


def _is_component_path(folder, prefix, scoped_path, owns_archive):
    """Whether a member under manifest relpath `prefix` belongs to `folder`.

    Everything under the component's own stage directories does, named after it
    or not: the test binaries a header-only project's report measures are named
    after what they test (rocThrust's bin/merge.hip, rocPRIM's
    bin/test_block_scan). Elsewhere only a path naming the component does --
    except the device code under .kpack/. Every component in the artifact shares
    one kpack archive, stored under whichever stage directory comes first
    (prim_test's sits in hipCUB's), and host binaries look their kernels up in
    it by name. Swapping the host binaries without it leaves them loading the
    baseline's kernels, which fail once the two builds differ (a bumped
    HIPCUB_<version>_NS mangles to symbols the baseline kpack does not have).
    So it follows the host binaries: it is replaced whenever `owns_archive`,
    i.e. the component has a stage directory in this artifact.
    """
    if _names_folder(folder, scoped_path):
        return True
    if scoped_path == ".kpack" or scoped_path.startswith(".kpack/"):
        return owns_archive
    return _is_own_stage(folder, prefix)


def replace_instrumented_libraries(artifacts, dest_dir, output_dir):
    """Extract instrumented libs from downloaded archives into the install tree.

    Reads each archive's artifact_manifest.txt for relpath prefixes, then flattens
    members into output_dir -- but only those belonging to the artifact's library
    folder (e.g. rocBLAS), leaving unrelated files in place. Instrumented shared
    libraries whose program header table kpack moved get it pinned back (see
    _therock_utils/elf_phdr.py); otherwise their profile writer can segfault
    at exit before writing any counters.
    """
    archives = sorted(
        p for p in dest_dir.iterdir() if p.name.endswith((".tar.zst", ".tar.xz"))
    )
    if not archives:
        log(f"No replacement archives found in {dest_dir}")
        return

    written_files = []
    for archive in archives:
        an = ArtifactName.from_filename(archive.name)
        folders = artifacts.get(an.name) if an else None
        if not folders:
            log(f"Skipping {archive.name}: no library folder mapping")
            continue
        for folder in folders:
            log(f"Replacing '{folder}' paths from {archive.name} into {output_dir}")
            replaced = 0
            with open_archive_for_read(archive) as tf:
                relpaths = _read_manifest(tf, archive)
                owns_archive = any(_is_own_stage(folder, r) for r in relpaths)

                while member := tf.next():
                    for prefix in relpaths:
                        prefix_slash = prefix + "/"
                        if not member.name.startswith(prefix_slash):
                            continue
                        scoped_path = member.name[len(prefix_slash) :]
                        if not _is_component_path(
                            folder, prefix, scoped_path, owns_archive
                        ):
                            break
                        dest_path = output_dir / PurePosixPath(scoped_path)
                        _replace_scoped_member(
                            tf, member, dest_path, output_dir, relpaths
                        )
                        if member.isfile():
                            written_files.append(dest_path)
                        replaced += 1
                        break
            log(f"  Replaced {replaced} '{folder}' path(s) from {archive.name}")

    _pin_instrumented(written_files)


def replace_host_libraries(host_libs, dest_dir, output_dir):
    """Extract only the named host libraries from other artifacts' lib archives.

    host_libs maps an artifact to (library folder, file prefix) pairs, as built
    by host_library_deps. A member is taken when it sits in that folder's stage
    directory under lib/ and its file name starts with the prefix, so
    libtensilelite-host.so* comes over without the rest of hipBLASLt.
    """
    written_files = []
    for archive in sorted(
        p for p in dest_dir.iterdir() if p.name.endswith((".tar.zst", ".tar.xz"))
    ):
        an = ArtifactName.from_filename(archive.name)
        if not an or an.name not in host_libs:
            continue
        if an.component != "lib" or an.target_family != "generic":
            continue
        for folder, lib_prefix in host_libs[an.name]:
            log(f"Replacing '{folder}' {lib_prefix}* from {archive.name}")
            replaced = 0
            with open_archive_for_read(archive) as tf:
                relpaths = _read_manifest(tf, archive)
                while member := tf.next():
                    for prefix in relpaths:
                        if not _is_own_stage(folder, prefix):
                            continue
                        prefix_slash = prefix + "/"
                        if not member.name.startswith(prefix_slash):
                            continue
                        scoped_path = member.name[len(prefix_slash) :]
                        if not scoped_path.startswith(f"lib/{lib_prefix}"):
                            break
                        dest_path = output_dir / PurePosixPath(scoped_path)
                        _replace_scoped_member(
                            tf, member, dest_path, output_dir, relpaths
                        )
                        if member.isfile():
                            written_files.append(dest_path)
                        replaced += 1
                        break
            if not replaced:
                raise IOError(
                    f"ERROR: No {lib_prefix}* found under '{folder}' in {archive.name}"
                )
            log(f"  Replaced {replaced} path(s)")

    _pin_instrumented(written_files)


def _read_manifest(tf, archive):
    """Return the relpath prefixes listed by the archive's leading manifest."""
    manifest_member = tf.next()
    if manifest_member is None or manifest_member.name != "artifact_manifest.txt":
        raise IOError(
            f"Artifact archive {archive} must have artifact_manifest.txt "
            "as its first member"
        )
    with tf.extractfile(manifest_member) as mf_file:
        return [r for r in mf_file.read().decode().splitlines() if r]


def _pin_instrumented(written_files):
    for path in written_files:
        if pin_phdr_table(path, require_section="__llvm_prf_cnts"):
            log(f"  Pinned the program header table kpack moved in {path}")


def main(argv):
    parser = argparse.ArgumentParser(prog="code-coverage-installer")
    parser.add_argument(
        "--code-coverage-run-id",
        type=str,
        help="run id of the build from which instrumental components needs to be replaced",
    )
    # Repo owning --code-coverage-run-id (the instrumented run), separate from
    # --run-github-repo (the generic baseline). Defaults to $GITHUB_REPOSITORY.
    # parse_known_args consumes it so it does not leak into the generic install.
    parser.add_argument(
        "--code-coverage-run-github-repo",
        type=str,
        default=os.environ.get("GITHUB_REPOSITORY"),
        help=(
            "GitHub repository (owner/name) that owns --code-coverage-run-id, "
            "used to resolve the instrumented artifact backend. Defaults to "
            "$GITHUB_REPOSITORY (the coverage run's own repo). This is separate "
            "from --run-github-repo, which owns the generic --run-id build."
        ),
    )
    parser.add_argument(
        "--code-coverage-release-type",
        type=str,
        default="ci",
        help=(
            "Release type of the instrumented build identified by "
            "--code-coverage-run-id (e.g. 'ci', 'nightly'). Controls which S3 "
            "bucket is used for the replacement artifact lookup. Defaults to "
            "'ci' because instrumented builds always run as CI jobs. This is "
            "separate from the baseline's RELEASE_TYPE env var."
        ),
    )
    artifacts_group = parser.add_argument_group("replace_comps")
    for comp in COMPONENT_MAP.keys():
        artifacts_group.add_argument(
            f"--replace-{comp}",
            default=False,
            help=f"Replace '{comp}' artifacts",
            action=argparse.BooleanOptionalAction,
        )
    args, extra_args = parser.parse_known_args(argv)
    opts = _read_passthrough_options(extra_args)

    # install generic build from --run-id artifacts
    install_from_artifacts_main(extra_args)

    # collect artifacts details as per the user options
    selected = [comp for comp in COMPONENT_MAP if getattr(args, f"replace_{comp}")]
    artifacts = replacement_folders(selected)
    host_libs = host_library_deps(selected)

    if not artifacts:
        log("No --replace-* components specified; nothing to download.")
        return -1

    if not args.code_coverage_run_id:
        parser.error(
            "--code-coverage-run-id is required when using --replace-* options"
        )

    # download selected component artifacts (instrumented, from the coverage repo)
    dest_dir = download_replacement_artifacts(
        args.code_coverage_run_id,
        artifacts.keys(),
        opts,
        args.code_coverage_run_github_repo,
        args.code_coverage_release_type,
        host_lib_artifacts=host_libs.keys(),
    )

    # replace selected library folder paths in selected component artifacts
    replace_instrumented_libraries(artifacts, dest_dir, opts.output_dir)
    # and the host libraries they link from other artifacts
    replace_host_libraries(host_libs, dest_dir, opts.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
