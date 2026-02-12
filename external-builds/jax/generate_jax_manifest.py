#!/usr/bin/env python3
"""
Generate a manifest for JAX external builds.

Intended to be called after building JAX wheels (e.g. from a GH Actions workflow).
Writes a JSON manifest into <output-dir>/manifests by default.
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

    # Optional build tag:
    # - Exactly 5 parts → no build tag
    # - 6+ parts → everything between version and python_tag
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
            cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.STDOUT, text=True
        ).strip()
    except Exception:
        return ""


def git_head(dirpath: Optional[Path]) -> Optional[Dict[str, str]]:
    if not dirpath:
        return None
    dirpath = dirpath.resolve()
    if not (dirpath / ".git").exists():
        return None
    commit = capture(["git", "rev-parse", "HEAD"], cwd=dirpath) or None
    desc = capture(["git", "describe", "--always", "--dirty"], cwd=dirpath) or None
    remote = capture(["git", "remote", "get-url", "origin"], cwd=dirpath) or None
    if not (commit or remote or desc):
        return None
    return {"dir": str(dirpath), "commit": commit, "describe": desc, "remote": remote}


def build_therock_metadata(repo_root: Path) -> Dict[str, Any]:
    """Try GH env first; otherwise fall back to git info from the local checkout."""
    therock_repo = os.getenv("GITHUB_REPOSITORY")
    therock_server = os.getenv("GITHUB_SERVER_URL")
    therock_url = (
        f"{therock_server}/{therock_repo}" if therock_server and therock_repo else None
    )
    therock_sha = os.getenv("GITHUB_SHA")
    therock_ref = os.getenv("GITHUB_REF")

    if therock_url or therock_sha or therock_ref:
        out: Dict[str, Any] = {
            "repo": therock_url,
            "commit": therock_sha,
            "ref": therock_ref,
        }
        return {k: v for k, v in out.items() if v}

    # Local build outside GH: infer from git if available.
    info = git_head(repo_root)
    if not info:
        return {}
    out = {
        "repo": info.get("remote"),
        "commit": info.get("commit"),
        "describe": info.get("describe"),
    }
    return {k: v for k, v in out.items() if v}


def choose_manifest_wheel(wheel_paths: List[Path]) -> Path:
    """Pick the wheel whose basename will be used for manifest naming.

    Preference:
      1) jax_rocm7_pjrt-*.whl
      2) jax_rocm7_plugin-*.whl
      3) jaxlib-*.whl
      4) first wheel in sorted order
    """
    names = [p.name for p in wheel_paths]

    def pick(prefix: str) -> Optional[Path]:
        for p in wheel_paths:
            if p.name.startswith(prefix) and p.name.endswith(".whl"):
                return p
        return None

    for prefix in ("jax_rocm7_pjrt-", "jax_rocm7_plugin-", "jaxlib-"):
        p = pick(prefix)
        if p:
            return p

    # Fallback: deterministic first
    return sorted(wheel_paths, key=lambda p: p.name)[0]


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
        default="jax-wheels",
        help="Manifest artifact_group label (default: jax-wheels)",
    )

    # Build/config metadata (optional)
    ap.add_argument(
        "--rocm-sdk-version",
        default=None,
        help="ROCm SDK version used for the build (e.g. 7.10.0a20251124)",
    )
    ap.add_argument(
        "--amdgpu-family",
        default=None,
        help="AMDGPU family the wheels target (e.g. gfx94X-dcgpu)",
    )
    ap.add_argument(
        "--therock-tar-url",
        default=None,
        help="TheRock tarball URL/path used as input to the build (if applicable)",
    )

    # Source checkout info (optional)
    ap.add_argument(
        "--jax-dir",
        type=Path,
        default=None,
        help="Path to the checked-out rocm-jax repo (for git metadata)",
    )
    ap.add_argument(
        "--jax-ref",
        default=None,
        help="Repo ref/branch used for JAX checkout (e.g. rocm-jaxlib-v0.8.0)",
    )

    args = ap.parse_args()

    output_dir = args.output_dir.resolve()
    manifest_dir = (args.manifest_dir or (output_dir / "manifests")).resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    sys_platform = platform.system().lower()
    arch = platform.machine().lower()
    platform_id = f"{sys_platform}-{arch}"

    run_id = os.getenv("GITHUB_RUN_ID") or "local"
    job_id = os.getenv("GITHUB_JOB")  # may be None for local builds

    # Infer TheRock repo root (external-builds/jax/generate_jax_manifest.py -> repo root is parents[2])
    script_path = Path(__file__).resolve()
    therock_root = (
        script_path.parents[2] if len(script_path.parents) >= 3 else Path.cwd()
    )
    therock_meta = build_therock_metadata(therock_root)

    amdgpu_family = args.amdgpu_family or os.getenv("AMDGPU_FAMILY")

    wheel_paths = sorted(output_dir.rglob("*.whl"))
    if not wheel_paths:
        raise RuntimeError(f"No wheels found under: {output_dir}")

    # Collect artifacts (wheels under output dir)
    artifacts: List[Dict[str, Any]] = []
    for p in wheel_paths:
        rel = p.relative_to(output_dir)
        meta = parse_wheel_name(p.name)
        artifacts.append(
            {
                "relative_path": str(rel).replace("\\", "/"),
                "size_bytes": p.stat().st_size,
                "labels": {
                    "framework": "jax",
                    "build_variant": os.getenv("BUILD_VARIANT", "release"),
                    **({"amdgpu_family": amdgpu_family} if amdgpu_family else {}),
                    **(
                        {"rocm_sdk_version": args.rocm_sdk_version}
                        if args.rocm_sdk_version
                        else {}
                    ),
                    **({"jax_ref": args.jax_ref} if args.jax_ref else {}),
                    **meta,
                },
            }
        )

    # Repo heads for sources
    sources: Dict[str, Any] = {}
    jax_info = git_head(args.jax_dir.resolve() if args.jax_dir else None)
    if jax_info:
        sources["jax"] = jax_info

    manifest: Dict[str, Any] = {
        "project": "TheRock",
        "component": "jax",
        "build_type": "external",
        "artifact_group": args.artifact_group,
        "platform": platform_id,
        "run_id": run_id,
        **({"job_id": job_id} if job_id else {}),
        **(
            {"rocm_sdk_version": args.rocm_sdk_version} if args.rocm_sdk_version else {}
        ),
        **({"amdgpu_family": amdgpu_family} if amdgpu_family else {}),
        **({"jax_ref": args.jax_ref} if args.jax_ref else {}),
        **({"therock_tar_url": args.therock_tar_url} if args.therock_tar_url else {}),
        **({"therock": therock_meta} if therock_meta else {}),
        "sources": sources,
        "artifacts": artifacts,
    }

    chosen = choose_manifest_wheel(wheel_paths)
    manifest_stem = chosen.name[:-4]  # strip ".whl"
    filename = f"{manifest_stem}.json"

    out_path = manifest_dir / filename
    out_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[jax-manifest] wrote {out_path}")


if __name__ == "__main__":
    main()
