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
import sys
import argparse
import platform
from pathlib import Path, PurePosixPath

from install_rocm_from_artifacts import main as install_from_artifacts_main
from artifact_manager import (
    DownloadRequest,
    create_backend_from_env,
    download_artifact,
    find_available_artifacts,
)
from _therock_utils.archive_util import open_archive_for_read
from _therock_utils.artifacts import ArtifactName
from _therock_utils.cmake_amdgpu_targets import amdgpu_family_map, expand_families

# Maps each --replace-<name> flag (argparse dest) to the TheRock artifact name
# that ships the instrumented library. rocBLAS is packaged in the 'blas'
# artifact and rocSOLVER in the 'solver' artifact (see BUILD_TOPOLOGY.toml).
REPLACE_ARTIFACT_MAP = {
    "replace_rocblas": "blas",
    "replace_rocsolver": "solver",
}

# The 'blas' and 'solver' artifacts ship more than one library (e.g. blas also
# carries hipBLAS). Map each artifact to the library-name substring so only the
# instrumented rocBLAS/rocSOLVER paths are replaced, leaving the rest untouched.
ARTIFACT_LIBRARY_KEYWORD = {
    name: dest[len("replace_") :] for dest, name in REPLACE_ARTIFACT_MAP.items()
}


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def _read_passthrough_options(passthrough_argv):
    """Read (without consuming) options shared with install_rocm_from_artifacts.

    These are parsed non-destructively so the same argv can still be forwarded
    to install_rocm_from_artifacts.py unchanged.
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

    blas/solver are target-specific artifacts named per family (mono-arch) or
    per gfx target (kpack-split), so both spellings must be matched.
    """
    families = ["generic"]
    if family:
        families.append(family)
        families.extend(expand_families([family], amdgpu_family_map(), strict=False))
    families.extend(t.strip() for t in amdgpu_targets.split(",") if t.strip())
    return families


def download_replacement_artifacts(code_coverage_run_id, artifact_names, opts):
    """Download the instrumented replacement artifacts from the code-coverage run.

    Uses the code coverage run ID as the run-id for the S3 backend, then fetches
    every component tar matching the requested artifact names and target family.
    """
    backend = create_backend_from_env(
        run_id=code_coverage_run_id,
        github_repository=opts.run_github_repo,
        platform=platform.system().lower(),
    )
    log(f"Fetching replacement artifacts from {backend.base_uri}")

    available = set(backend.list_artifacts())
    target_families = _target_families(opts.artifact_group, opts.amdgpu_targets)
    log(f"Matching artifacts {sorted(artifact_names)} for families {target_families}")

    matched = find_available_artifacts(artifact_names, target_families, available)
    if not matched:
        log(
            f"ERROR: No replacement artifacts {sorted(artifact_names)} found in "
            f"{backend.base_uri} for families {target_families}"
        )
        sys.exit(1)

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
            log(f"ERROR: Failed to download {filename}")
            sys.exit(1)

    return dest_dir


def _replace_scoped_member(tf, member, dest_path, output_dir, relpaths):
    """Write a single archive member into the flattened install tree.

    Mirrors the file/symlink/dir/hardlink handling used when TheRock flattens
    an artifact archive, so replaced files keep their exec bits and link
    structure. Any existing file/symlink at dest_path is removed first.
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
        # Hardlink target is archive-relative; strip its manifest prefix so it
        # resolves to the already-written file in the flattened output tree.
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


def replace_instrumented_libraries(dest_dir, output_dir):
    """Extract instrumented libs from downloaded archives into the install tree.

    For every replacement archive under dest_dir, read its artifact_manifest.txt
    to learn the relpath prefixes, then flatten (strip prefix) each member into
    output_dir -- but only members whose scoped path matches the artifact's
    library keyword (rocblas/rocsolver), so unrelated files are left in place.
    """
    archives = sorted(
        p for p in dest_dir.iterdir() if p.name.endswith((".tar.zst", ".tar.xz"))
    )
    if not archives:
        log(f"No replacement archives found in {dest_dir}")
        return

    for archive in archives:
        an = ArtifactName.from_filename(archive.name)
        keyword = ARTIFACT_LIBRARY_KEYWORD.get(an.name) if an else None
        if not keyword:
            log(f"Skipping {archive.name}: no library keyword mapping")
            continue

        log(f"Replacing '{keyword}' paths from {archive.name} into {output_dir}")
        replaced = 0
        with open_archive_for_read(archive) as tf:
            manifest_member = tf.next()
            if manifest_member is None or manifest_member.name != "artifact_manifest.txt":
                raise IOError(
                    f"Artifact archive {archive} must have artifact_manifest.txt "
                    "as its first member"
                )
            with tf.extractfile(manifest_member) as mf_file:
                relpaths = [r for r in mf_file.read().decode().splitlines() if r]

            while member := tf.next():
                for prefix in relpaths:
                    prefix_slash = prefix + "/"
                    #log(f"{member.name=} {prefix_slash=}")
                    if not member.name.startswith(prefix_slash):
                        continue
                    scoped_path = member.name[len(prefix_slash) :]
                    if keyword.lower() not in scoped_path.lower():
                        break
                    #log(f"{keyword=} {scoped_path=}")
                    dest_path = output_dir / PurePosixPath(scoped_path)
                    _replace_scoped_member(tf, member, dest_path, output_dir, relpaths)
                    replaced += 1
                    break
        log(f"  Replaced {replaced} '{keyword}' path(s) from {archive.name}")


def main(argv):
    parser = argparse.ArgumentParser(prog="code-coverage-installer")
    parser.add_argument(
        "--code-coverage-run-id",
        type=str,
        help="run id of the build from which instrumental components needs to be replaced",
    )

    artifacts_group = parser.add_argument_group("replace_comps")
    artifacts_group.add_argument(
        "--replace-rocblas",
        default=False,
        help="Replace 'blas' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--replace-rocsolver",
        default=False,
        help="Replace 'solver' artifacts",
        action=argparse.BooleanOptionalAction,
    )

    args, extra_args = parser.parse_known_args(argv)

    # install generic build from --run-id artifacts
    install_from_artifacts_main(extra_args)

    # Resolve which artifacts to replace from the --replace-* flags.
    artifact_names = {
        name for dest, name in REPLACE_ARTIFACT_MAP.items() if getattr(args, dest)
    }
    if not artifact_names:
        log("No --replace-* components specified; nothing to download.")
        return

    if not args.code_coverage_run_id:
        parser.error("--code-coverage-run-id is required when using --replace-* options")

    opts = _read_passthrough_options(extra_args)
    dest_dir = download_replacement_artifacts(
        args.code_coverage_run_id, artifact_names, opts
    )
    replace_instrumented_libraries(dest_dir, opts.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
