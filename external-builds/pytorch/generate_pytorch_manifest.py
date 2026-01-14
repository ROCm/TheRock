#!/usr/bin/env python3
"""
Generate a manifest for PyTorch external builds.

This is invoked by CI workflows after building wheels, and writes a JSON manifest
alongside the built artifacts.
"""

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_wheel_name(filename: str) -> Dict[str, str]:
    # Best-effort wheel parse per PEP 427:
    #   {distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
    if not filename.endswith(".whl"):
        return {}

    parts = filename[:-4].split("-")
    if len(parts) < 5:
        return {}

    distribution = parts[0]
    version = parts[1]

    python_tag = parts[-3]
    abi_tag = parts[-2]
    platform_tag = parts[-1]

    # Optional build tag: present when there are >5 dash-separated fields.
    build_tag = None
    if len(parts) > 5:
        build_tag = "-".join(parts[2:-3])

    meta: Dict[str, str] = {
        "distribution": distribution,
        "version": version,
        "python_tag": python_tag,
        "abi_tag": abi_tag,
        "platform_tag": platform_tag,
    }
    if build_tag:
        meta["build_tag"] = build_tag

    return meta


def capture(cmd: List[str], cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.check_output(
            cmd,
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except FileNotFoundError:
        print(
            f"  [WARN] Command not found: {cmd[0]} (skipping: {' '.join(cmd)})",
            flush=True,
        )
        return ""
    except subprocess.CalledProcessError as e:
        output = (e.output or "").strip()
        print(
            f"  [WARN] Command failed ({e.returncode}): {' '.join(cmd)}"
            + (f"\n    output: {output}" if output else ""),
            flush=True,
        )
        return ""
    except Exception as e:
        print(
            f"  [WARN] Unexpected error running {' '.join(cmd)}: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return ""


def git_head(dirpath: Optional[Path]) -> Optional[Dict[str, str]]:
    if not dirpath:
        print("  [WARN] git_head: no directory provided (skipping)", flush=True)
        return None

    dirpath = dirpath.resolve()

    if not dirpath.exists():
        print(
            f"  [WARN] git_head: directory does not exist: {dirpath} (skipping)",
            flush=True,
        )
        return None

    if not (dirpath / ".git").exists():
        print(
            f"  [WARN] git_head: not a git checkout (no .git): {dirpath} (skipping)",
            flush=True,
        )
        return None

    commit = capture(["git", "rev-parse", "HEAD"], cwd=dirpath) or None
    desc = capture(["git", "describe", "--always", "--dirty"], cwd=dirpath) or None
    remote = capture(["git", "remote", "get-url", "origin"], cwd=dirpath) or None

    if not (commit or desc or remote):
        print(
            f"  [WARN] git_head: unable to determine git metadata for {dirpath}",
            flush=True,
        )
        return None

    return {
        "dir": str(dirpath),
        "commit": commit,
        "describe": desc,
        "remote": remote,
    }


def _manifest_filename(
    python_version: Optional[str],
    pytorch_git_ref: Optional[str],
) -> str:
    py = (python_version or "").strip()
    if py.startswith("py"):
        py = py[2:]
    py_part = f"py{py}" if py else "py"

    if pytorch_git_ref == "nightly":
        track = "nightly"
    elif pytorch_git_ref and pytorch_git_ref.startswith("release/"):
        track = pytorch_git_ref.replace("/", "-", 1)
    else:
        track = (pytorch_git_ref or "unknown").replace("/", "-")

    return f"therock-manifest_torch_{py_part}_{track}.json"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a manifest JSON for built PyTorch wheels."
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing built wheels (searched recursively).",
    )
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Output directory for the manifest (e.g.: <output-dir>/manifests).",
    )
    ap.add_argument(
        "--artifact-group",
        required=True,
        help="Manifest artifact_group value (e.g. pytorch-wheels).",
    )
    ap.add_argument(
        "--amdgpu-family",
        default=None,
        required=True,
        help="AMDGPU family selector (e.g. gfx94X-dcgpu, gfx110X-all).",
    )
    ap.add_argument(
        "--pytorch-rocm-arch",
        default=None,
        help="PYTORCH_ROCM_ARCH used for the build (e.g. gfx942 or gfx942,gfx90a).",
    )
    ap.add_argument(
        "--rocm-sdk-version",
        default=None,
        required=True,
        help="ROCm SDK version used for the build (e.g. 7.11.0a20251124).",
    )
    ap.add_argument(
        "--version-suffix",
        default=None,
        help="Version suffix applied to built packages (e.g. +rocm7.11.0a20251124).",
    )
    ap.add_argument(
        "--pytorch-git-ref",
        default=None,
        required=True,
        help="PyTorch ref used for the build (e.g. release/2.8 or nightly).",
    )
    ap.add_argument(
        "--python-version",
        default=None,
        required=True,
        help="Python version used for the build (e.g. 3.11, 3.12).",
    )
    ap.add_argument(
        "--pytorch-dir",
        type=Path,
        default=None,
        help="Path to the PyTorch source checkout (records git metadata if present).",
    )
    ap.add_argument(
        "--pytorch-audio-dir",
        type=Path,
        default=None,
        help="Path to the torchaudio source checkout (records git metadata if present).",
    )
    ap.add_argument(
        "--pytorch-vision-dir",
        type=Path,
        default=None,
        help="Path to the torchvision source checkout (records git metadata if present).",
    )
    ap.add_argument(
        "--triton-dir",
        type=Path,
        default=None,
        help="Path to the triton source checkout (records git metadata if present).",
    )
    args = ap.parse_args()

    output_dir = args.output_dir.resolve()
    manifest_dir = (args.manifest_dir or (output_dir / "manifests")).resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    sys_platform = platform.system().lower()
    arch = platform.machine().lower()
    platform_id = f"{sys_platform}-{arch}"

    is_github = os.getenv("GITHUB_ACTIONS") == "true"

    run_id = os.getenv("GITHUB_RUN_ID") if is_github else None
    job_id = os.getenv("GITHUB_JOB") if is_github else None

    therock_repo = os.getenv("GITHUB_REPOSITORY")
    therock_server = os.getenv("GITHUB_SERVER_URL")
    therock_url = (
        f"{therock_server}/{therock_repo}"
        if is_github and therock_server and therock_repo
        else None
    )
    therock_sha = os.getenv("GITHUB_SHA") if is_github else None
    therock_ref = os.getenv("GITHUB_REF") if is_github else None

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

    if not artifacts:
        raise SystemExit(f"ERROR: No .whl files found under: {output_dir}")

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
        "amdgpu_family": args.amdgpu_family,
        "pytorch_rocm_arch": args.pytorch_rocm_arch,
        "run_id": run_id,
        "job_id": job_id,
        "rocm_sdk_version": args.rocm_sdk_version,
        "version_suffix": args.version_suffix,
        "pytorch_git_ref": args.pytorch_git_ref,
        "python_version": args.python_version,
        "therock": {
            "repo": therock_url,
            "commit": therock_sha,
            "ref": therock_ref,
        },
        "sources": sources,
        "artifacts": artifacts,
    }

    filename = _manifest_filename(
        python_version=args.python_version,
        pytorch_git_ref=args.pytorch_git_ref,
    )
    out_path = manifest_dir / filename
    out_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[pytorch-manifest] wrote {out_path}")


if __name__ == "__main__":
    main()
