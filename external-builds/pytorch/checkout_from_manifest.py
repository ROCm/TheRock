#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check out pytorch ecosystem repositories from a manifest file.

Reads a manifest JSON (produced by ``generate_pytorch_manifest_upfront.py``)
and calls the individual ``pytorch_*_repo.py checkout`` scripts with the
exact commit SHAs and repo URLs from the manifest.

This provides a single command to reproduce the same source tree that CI
builds use, given only a manifest file.

Usage::

    # Check out all projects in the manifest under ./checkouts/:
    python checkout_from_manifest.py \\
        --manifest manifests/therock-manifest_torch_linux_release-2.10.json \\
        --checkout-root ./checkouts

    # Check out only pytorch (skip audio, vision, etc.):
    python checkout_from_manifest.py \\
        --manifest manifest.json \\
        --checkout-root ./checkouts \\
        --projects pytorch
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

# Maps manifest project names to their checkout scripts.
CHECKOUT_SCRIPTS: dict[str, str] = {
    "pytorch": "pytorch_torch_repo.py",
    "pytorch_audio": "pytorch_audio_repo.py",
    "pytorch_vision": "pytorch_vision_repo.py",
    "triton": "pytorch_triton_repo.py",
    "apex": "pytorch_apex_repo.py",
}


def _split_words(value: str) -> list[str]:
    return value.replace(";", " ").split() if value else []


def log(*args, **kwargs) -> None:
    print(*args, **kwargs)
    sys.stdout.flush()


def checkout_project(
    *,
    name: str,
    source_info: dict[str, str],
    checkout_root: Path,
    no_hipify: bool,
) -> None:
    """Check out a single project using its checkout script."""
    script = THIS_DIR / CHECKOUT_SCRIPTS[name]
    checkout_dir = checkout_root / name

    commit = source_info["commit"]
    repo = source_info["repo"]

    log(f"  {name}: {repo} @ {commit[:12]}")

    cmd: list[str] = [
        sys.executable,
        str(script),
        "checkout",
        "--gitrepo-origin",
        repo,
        "--repo-hashtag",
        commit,
        "--checkout-dir",
        str(checkout_dir),
    ]

    # pytorch_torch_repo.py doesn't have --torch-dir, but the others do.
    # When checking out from a manifest we pass explicit origin/hashtag,
    # so related_commits is not needed. We still pass --torch-dir pointing
    # to the pytorch checkout so the scripts can find it if needed.
    if name != "pytorch":
        cmd.extend(["--torch-dir", str(checkout_root / "pytorch")])

    if no_hipify:
        cmd.append("--no-hipify")

    log(f"  Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def download_manifest(*, manifest_url: str, output_path: Path) -> Path:
    """Download a manifest URL to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading manifest: {manifest_url} -> {output_path}")
    urllib.request.urlretrieve(manifest_url, output_path)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to download manifest: {manifest_url}")
    return output_path


def validate_manifest(
    manifest: dict[str, object], *, expected_pytorch_git_ref: str
) -> None:
    """Validate manifest contents against workflow expectations."""
    if not expected_pytorch_git_ref:
        return
    pytorch = manifest.get("pytorch")
    if not isinstance(pytorch, dict):
        raise ValueError("Manifest is missing pytorch entry")
    actual_ref = pytorch.get("branch")
    if actual_ref != expected_pytorch_git_ref:
        raise ValueError(
            f"Manifest PyTorch ref {actual_ref!r} does not match "
            f"{expected_pytorch_git_ref!r}"
        )


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Check out pytorch repos from a manifest file"
    )
    manifest_input = parser.add_mutually_exclusive_group(required=True)
    manifest_input.add_argument(
        "--manifest",
        type=Path,
        help="Path to manifest JSON file",
    )
    manifest_input.add_argument(
        "--manifest-url",
        help="URL to a manifest JSON file to download before checkout",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help=(
            "Local path to write --manifest-url downloads to "
            "(default: <checkout-root>/pytorch_manifest.json)"
        ),
    )
    parser.add_argument(
        "--checkout-root",
        type=Path,
        required=True,
        help="Root directory for checkouts (each project gets a subdirectory)",
    )
    parser.add_argument(
        "--projects",
        default="",
        help=(
            "Semicolon- or space-separated list of projects to check out "
            "(default: all projects in the manifest)."
        ),
    )
    parser.add_argument(
        "--expected-pytorch-git-ref",
        default="",
        help="Validate manifest pytorch.branch against this ref before checkout",
    )
    parser.add_argument(
        "--no-hipify",
        action="store_true",
        default=False,
        help="Skip HIPIFY for all checkouts (e.g. for test-only runs)",
    )
    args = parser.parse_args(argv)

    manifest_path = args.manifest
    if args.manifest_url:
        manifest_path = args.manifest_output or (
            args.checkout_root / "pytorch_manifest.json"
        )
        download_manifest(manifest_url=args.manifest_url, output_path=manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, expected_pytorch_git_ref=args.expected_pytorch_git_ref)

    # Determine which projects to check out.
    available = [name for name in CHECKOUT_SCRIPTS if name in manifest]
    if args.projects:
        projects = _split_words(args.projects)
        unknown = set(projects) - set(available)
        if unknown:
            parser.error(
                f"Projects not in manifest: {unknown}. " f"Available: {available}"
            )
    else:
        projects = available

    log(f"Manifest: {manifest_path}")
    log(f"Checkout root: {args.checkout_root}")
    log(f"Projects: {projects}")
    log("")

    args.checkout_root.mkdir(parents=True, exist_ok=True)

    # Always check out pytorch first (other scripts may reference --torch-dir).
    if "pytorch" in projects:
        projects = ["pytorch"] + [p for p in projects if p != "pytorch"]

    for name in projects:
        checkout_project(
            name=name,
            source_info=manifest[name],
            checkout_root=args.checkout_root,
            no_hipify=args.no_hipify,
        )
        log("")

    log("All checkouts complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
