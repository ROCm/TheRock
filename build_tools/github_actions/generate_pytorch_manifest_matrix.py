#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate a PyTorch build matrix from uploaded manifest files."""

import argparse
import json
import sys
from pathlib import Path

from github_actions_api import gha_set_output


def _split_words(value: str) -> list[str]:
    return value.split() if value else []


def _read_manifest_ref(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pytorch = manifest.get("pytorch")
    if not isinstance(pytorch, dict):
        raise ValueError(f"{manifest_path}: missing pytorch manifest entry")
    ref = pytorch.get("branch")
    if not isinstance(ref, str) or not ref:
        raise ValueError(f"{manifest_path}: missing pytorch branch")
    return ref


def collect_manifest_urls(
    *, manifest_dir: Path, manifest_dir_url: str
) -> dict[str, str]:
    """Return pytorch_git_ref -> explicit uploaded manifest URL."""
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"Manifest directory not found: {manifest_dir}")

    base_url = manifest_dir_url.rstrip("/")
    manifest_urls: dict[str, str] = {}
    for manifest_path in sorted(manifest_dir.glob("*.json")):
        ref = _read_manifest_ref(manifest_path)
        if ref in manifest_urls:
            raise ValueError(f"Duplicate manifest for PyTorch ref {ref!r}")
        manifest_urls[ref] = f"{base_url}/{manifest_path.name}"

    if not manifest_urls:
        raise FileNotFoundError(f"No JSON manifests found in {manifest_dir}")
    return manifest_urls


def parse_excludes(values: list[str]) -> set[tuple[str, str]]:
    """Parse exclusions in '<pytorch_git_ref>|<python_version>' form."""
    excludes: set[tuple[str, str]] = set()
    for value in values:
        try:
            pytorch_ref, python_version = value.split("|", maxsplit=1)
        except ValueError as e:
            raise ValueError(
                f"Invalid --exclude {value!r}, expected '<pytorch_ref>|<python_version>'"
            ) from e
        excludes.add((pytorch_ref, python_version))
    return excludes


def build_matrix(
    *,
    manifest_urls: dict[str, str],
    python_versions: list[str],
    pytorch_git_refs: list[str],
    excludes: set[tuple[str, str]],
) -> dict[str, list[dict[str, str]]]:
    if not pytorch_git_refs:
        pytorch_git_refs = list(manifest_urls)

    missing_refs = [ref for ref in pytorch_git_refs if ref not in manifest_urls]
    if missing_refs:
        raise ValueError(f"Missing manifests for PyTorch refs: {missing_refs}")

    include = []
    for pytorch_ref in pytorch_git_refs:
        for python_version in python_versions:
            if (pytorch_ref, python_version) in excludes:
                continue
            include.append(
                {
                    "python_version": python_version,
                    "pytorch_git_ref": pytorch_ref,
                    "manifest_url": manifest_urls[pytorch_ref],
                }
            )

    if not include:
        raise ValueError("Generated an empty PyTorch manifest matrix")
    return {"include": include}


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PyTorch build matrix from uploaded manifest files"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Local directory containing generated manifest JSON files.",
    )
    parser.add_argument(
        "--manifest-dir-url",
        required=True,
        help="Uploaded URL for the manifest directory.",
    )
    parser.add_argument(
        "--python-versions",
        required=True,
        help="Space-separated Python versions.",
    )
    parser.add_argument(
        "--pytorch-git-refs",
        default="",
        help="Space-separated PyTorch refs. Defaults to all manifests found.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude one matrix cell in '<pytorch_ref>|<python_version>' form.",
    )
    args = parser.parse_args(argv)

    matrix = build_matrix(
        manifest_urls=collect_manifest_urls(
            manifest_dir=args.manifest_dir,
            manifest_dir_url=args.manifest_dir_url,
        ),
        python_versions=_split_words(args.python_versions),
        pytorch_git_refs=_split_words(args.pytorch_git_refs),
        excludes=parse_excludes(args.exclude),
    )
    matrix_json = json.dumps(matrix)
    print(json.dumps(matrix, indent=2))
    gha_set_output({"matrix": matrix_json})


if __name__ == "__main__":
    main(sys.argv[1:])
