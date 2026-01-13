#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

def run(cmd, cwd=None, env=None):
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)

def extract_id_from_base_image(base_image: str) -> str:
    if ":" not in base_image:
        raise ValueError("BASE_IMAGE must include a tag, e.g., repo/name:tag")
    tag_part = base_image.split(":", 1)[1]
    extracted = tag_part.split("_", 1)[0]
    return extracted

def main():
    parser = argparse.ArgumentParser(description="Build and optionally push vLLM Docker images")
    parser.add_argument("--pytorch-base-image", required=True, help="Base image (e.g., rocm/pytorch-private:tag)")
    parser.add_argument("--vllm-repo", default="https://github.com/ROCm/vllm.git", help="vLLM git repository")
    parser.add_argument("--vllm-ref", required=True, help="Commit/branch/tag to checkout")
    parser.add_argument("--final-image-name", default="rocm/pytorch-private", help="Final image repo/name")
    # Backward compatibility: allow explicit rocm_suffix, but prefer computed one if rocm_version+amdgpu_family are provided
    parser.add_argument("--rocm-suffix", default="", help="Explicit suffix, e.g., rocm7.0_aiter (overrides computed suffix)")
    parser.add_argument("--rocm-version", default="", help="ROCm version used to compose rocm_suffix, e.g., rocm7.0")
    parser.add_argument("--amdgpu-family", default="", help="AMDGPU family used to compose rocm_suffix, e.g., aiter")
    parser.add_argument("--push", default="true", choices=["true", "false"], help="Whether to push final image")
    parser.add_argument("--workdir", default="", help="Optional working directory (defaults to temp dir)")
    parser.add_argument("--write-outputs", action="store_true", help="Write outputs to GITHUB_OUTPUT")
    args = parser.parse_args()

    # Normalize/resolve inputs
    base_image = args.pytorch_base_image
    final_image_name = args.final_image_name
    do_push = args.push.lower() == "true"

    # Compute rocm_suffix: explicit > computed > default
    rocm_suffix = args.rocm_suffix.strip()
    if not rocm_suffix:
        rv = args.rocm_version.strip()
        af = args.amdgpu_family.strip()
        if rv and af:
            rocm_suffix = f"{rv}_{af}"
        else:
            # Default if neither explicit nor computable is provided
            rocm_suffix = "rocm7.0_aiter"

    extracted_id = extract_id_from_base_image(base_image)
    current_date = datetime.now().strftime("%m%d")

    base_image_tag = f"vllm_base_{extracted_id}_{rocm_suffix}_{current_date}"
    final_image_tag = f"{final_image_name}:vllm_{extracted_id}_{rocm_suffix}_{current_date}"

    print(f"Extracted ID from BASE_IMAGE: {extracted_id}")
    print(f"Generated Date Stamp (MMdd): {current_date}")
    print(f"Generated Base Image Tag: {base_image_tag}")
    print(f"Generated Final Image Tag: {final_image_tag}")

    # Prepare working directory
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="vllm-build-"))

    vllm_dir = workdir / "vllm"

    # Clone and checkout vLLM
    run(["git", "clone", "--filter=blob:none", args.vllm_repo, str(vllm_dir)])
    run(["git", "fetch", "--all", "--tags", "--prune"], cwd=str(vllm_dir))
    # Try detached checkout first; fallback to branch/tag checkout
    try:
        run(["git", "checkout", "--detach", args.vllm_ref], cwd=str(vllm_dir))
    except subprocess.CalledProcessError:
        run(["git", "checkout", args.vllm_ref], cwd=str(vllm_dir))
    run(["git", "rev-parse", "--short", "HEAD"], cwd=str(vllm_dir))

    # Docker builds with BuildKit
    env = os.environ.copy()
    env["DOCKER_BUILDKIT"] = "1"

    # Build base image
    run([
        "docker", "build",
        "-f", "docker/Dockerfile.rocm_base",
        "--build-arg", f"BASE_IMAGE={base_image}",
        "-t", base_image_tag,
        "."
    ], cwd=str(vllm_dir), env=env)

    # Build final image
    run([
        "docker", "build",
        "-f", "docker/Dockerfile.rocm",
        "--build-arg", f"BASE_IMAGE={base_image_tag}",
        "-t", final_image_tag,
        "."
    ], cwd=str(vllm_dir), env=env)

    if do_push:
        run(["docker", "push", final_image_tag])

    # Emit outputs for GitHub Actions
    if args.write_outputs:
        gh_out = os.environ.get("GITHUB_OUTPUT")
        if not gh_out:
            print("WARNING: GITHUB_OUTPUT is not set; cannot export step outputs.", file=sys.stderr)
        else:
            with open(gh_out, "a", encoding="utf-8") as f:
                f.write(f"base_tag={base_image_tag}\n")
                f.write(f"vllm_tag={final_image_tag}\n")

    # Also print for logs
    print(f"base_tag={base_image_tag}")
    print(f"vllm_tag={final_image_tag}")

if __name__ == "__main__":
    main()
