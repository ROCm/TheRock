#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep kpacked test host binaries paired with their device artifacts.

Kpack splitting normally puts transformed host binaries in
`<artifact>_<component>_generic` and puts `.kpack` files in per-ISA artifacts.
That is correct for runtime libraries, but unsafe for test executables in
multi-arch CI: multiple architecture jobs upload the same `_generic` artifact
name, and the last upload can be paired with a different architecture job's
`.kpack` shard. Some HIP test executables embed host-side kernel symbol names,
so a cross-job host/kpack pair can fail at runtime with missing symbols.

This post-process runs only for `test` components. It copies transformed host
binaries that contain a kpack reference marker into every per-ISA artifact from
the same split, then removes those binaries from the shared generic artifact.
Non-device test data remains in the generic artifact.
"""

import argparse
import shutil
import sys
from pathlib import Path

from rocm_kpack.format_detect import UnsupportedBinaryFormat
from rocm_kpack.kpack_transform import read_kpack_ref_marker


def _read_manifest(artifact_dir: Path) -> list[str]:
    manifest_path = artifact_dir / "artifact_manifest.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing artifact manifest: {manifest_path}")
    return [line.strip() for line in manifest_path.read_text().splitlines() if line.strip()]


def _write_manifest(artifact_dir: Path, prefixes: list[str]) -> None:
    manifest_path = artifact_dir / "artifact_manifest.txt"
    manifest_path.write_text("".join(f"{prefix}\n" for prefix in sorted(set(prefixes))))


def _has_kpack_ref(path: Path) -> bool:
    try:
        return read_kpack_ref_marker(path) is not None
    except UnsupportedBinaryFormat:
        return False


def _find_kpacked_files(generic_dir: Path, prefixes: list[str]) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for prefix in prefixes:
        prefix_dir = generic_dir / prefix
        if not prefix_dir.exists():
            continue
        for path in prefix_dir.rglob("*"):
            if path.is_file() and _has_kpack_ref(path):
                found.append((prefix, path))
    return found


def run(args: argparse.Namespace) -> None:
    artifacts_dir = args.artifacts_dir
    artifact_prefix = args.artifact_prefix
    generic_dir = artifacts_dir / f"{artifact_prefix}_generic"
    if not generic_dir.exists():
        raise FileNotFoundError(f"Missing generic split artifact: {generic_dir}")

    prefixes = _read_manifest(generic_dir)
    kpacked_files = _find_kpacked_files(generic_dir, prefixes)
    if not kpacked_files:
        print(f"No kpacked test host binaries found in {generic_dir}")
        return

    arch_dirs = [
        path
        for path in sorted(artifacts_dir.glob(f"{artifact_prefix}_*"))
        if path.is_dir()
        and path.name != f"{artifact_prefix}_generic"
        and (path / "artifact_manifest.txt").exists()
    ]
    if not arch_dirs:
        raise FileNotFoundError(
            f"Found kpacked host binaries in {generic_dir}, but no per-arch artifacts"
        )

    copied = 0
    for arch_dir in arch_dirs:
        arch_prefixes = _read_manifest(arch_dir)
        for prefix, source_path in kpacked_files:
            rel_path = source_path.relative_to(generic_dir / prefix)
            dest_path = arch_dir / prefix / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            copied += 1
            if prefix not in arch_prefixes:
                arch_prefixes.append(prefix)
        _write_manifest(arch_dir, arch_prefixes)

    for _, source_path in kpacked_files:
        source_path.unlink()

    print(
        f"Moved {len(kpacked_files)} kpacked test host binaries from "
        f"{generic_dir.name} into {len(arch_dirs)} per-arch artifacts "
        f"({copied} copies)"
    )


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing split artifact directories",
    )
    parser.add_argument(
        "--artifact-prefix",
        required=True,
        help="Artifact prefix such as fft_test or blas_test",
    )
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
