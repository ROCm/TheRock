#!/usr/bin/env python3
"""
Generate a manifest for PyTorch external builds.

it is called by build_prod_wheels.py after building wheels.
"""

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_wheel_name(filename: str) -> Dict[str, str]:
    # Best-effort wheel parse:
    #   {name}-{version}(-{build})?-{py}-{abi}-{plat}.whl
    if not filename.endswith(".whl"):
        return {}
    parts = filename[:-4].split("-")
    if len(parts) < 5:
        return {}
    return {
        "package_name": parts[0],
        "package_version": parts[1],
        "python_tag": parts[-3],
        "abi_tag": parts[-2],
        "platform_tag": parts[-1],
    }


def capture(cmd: List[str], cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.check_output(
            cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.STDOUT, text=True
        ).strip()
    except Exception:
        return ""


def git_head(dirpath: Optional[Path]) -> Optional[Dict[str, str]]:
    if not dirpath:
        return None
    if not (dirpath / ".git").exists():
        return None
    commit = capture(["git", "rev-parse", "HEAD"], cwd=dirpath) or None
    desc = capture(["git", "describe", "--always", "--dirty"], cwd=dirpath) or None
    remote = capture(["git", "remote", "get-url", "origin"], cwd=dirpath) or None
    if not (commit or remote or desc):
        return None
    return {"dir": str(dirpath), "commit": commit, "describe": desc, "remote": remote}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True, help="Wheel output dir")
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Directory to write manifest into (default: <output-dir>/manifests)",
    )
    ap.add_argument(
        "--artifact-group",
        default="pytorch-wheels",
        help="Manifest artifact_group label",
    )
    ap.add_argument("--rocm-sdk-version", default=None)
    ap.add_argument("--pytorch-rocm-arch", default=None)
    ap.add_argument("--version-suffix", default=None)
    ap.add_argument("--pytorch-dir", type=Path, default=None)
    ap.add_argument("--pytorch-audio-dir", type=Path, default=None)
    ap.add_argument("--pytorch-vision-dir", type=Path, default=None)
    ap.add_argument("--triton-dir", type=Path, default=None)
    args = ap.parse_args()

    output_dir = args.output_dir.resolve()
    manifest_dir = (args.manifest_dir or (output_dir / "manifests")).resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    sys_platform = platform.system().lower()
    arch = platform.machine().lower()
    platform_id = f"{sys_platform}-{arch}"

    run_id = os.getenv("GITHUB_RUN_ID")
    job_id = os.getenv("GITHUB_JOB")

    therock_repo = os.getenv("GITHUB_REPOSITORY")
    therock_server = os.getenv("GITHUB_SERVER_URL")
    therock_url = (
        f"{therock_server}/{therock_repo}" if therock_server and therock_repo else None
    )
    therock_sha = os.getenv("GITHUB_SHA")
    therock_ref = os.getenv("GITHUB_REF")

    artifacts: List[Dict[str, Any]] = []
    for p in sorted(output_dir.rglob("*.whl")):
        rel = p.relative_to(output_dir)
        meta = parse_wheel_name(p.name)
        artifacts.append(
            {
                "relative_path": str(rel).replace("\\", "/"),
                "size_bytes": p.stat().st_size,
                "labels": {
                    "framework": "pytorch",
                    "build_variant": os.getenv("BUILD_VARIANT", "release"),
                    **meta,
                },
            }
        )

    sources: Dict[str, Any] = {}
    for name, d in [
        ("pytorch", args.pytorch_dir),
        ("pytorch_audio", args.pytorch_audio_dir),
        ("pytorch_vision", args.pytorch_vision_dir),
        ("triton", args.triton_dir),
    ]:
        info = git_head(d)
        if info:
            sources[name] = info

    manifest: Dict[str, Any] = {
        "project": "TheRock",
        "component": "pytorch",
        "build_type": "external",
        "artifact_group": args.artifact_group,
        "platform": platform_id,
        "run_id": run_id,
        "job_id": job_id,
        "rocm_sdk_version": args.rocm_sdk_version,
        "pytorch_rocm_arch": args.pytorch_rocm_arch,
        "version_suffix": args.version_suffix,
        "therock": {
            "repo": therock_url,
            "commit": therock_sha,
            "ref": therock_ref,
        },
        "sources": sources,
        "artifacts": artifacts,
    }

    filename = f"therock_manifest-{args.artifact_group}-{platform_id}-{run_id}.json"
    out_path = manifest_dir / filename
    out_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[pytorch-manifest] wrote {out_path}")


if __name__ == "__main__":
    main()
