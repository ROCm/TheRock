#!/usr/bin/env python3
import argparse
import subprocess
import sys
from github_actions.github_actions_utils import gha_append_step_summary
print("ARGV:", sys.argv)

def run(cmd):
    print(">", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main():
    p = argparse.ArgumentParser(description="Build and push a Docker image (no defaults).")
    p.add_argument("--context", required=True, help="Build context directory")
    p.add_argument("--dockerfile", required=True, help="Path to Dockerfile")
    p.add_argument("--tag", required=True, help="Full image tag (include registry if needed)")
    p.add_argument("--PYTHON_VERSION", required=True)
    p.add_argument("--TORCH_VERSION", required=True)
    p.add_argument("--ROCM_VERSION", required=True)
    p.add_argument("--PACKAGE_INDEX_URL", required=True)
    p.add_argument("--AMDGPU_FAMILY", required=True)
    args = p.parse_args()

    build_args = [
        "--build-arg", f"PYTHON_VERSION={args.PYTHON_VERSION}",
        "--build-arg", f"TORCH_VERSION={args.TORCH_VERSION}",
        "--build-arg", f"ROCM_VERSION={args.ROCM_VERSION}",
        "--build-arg", f"PACKAGE_INDEX_URL={args.PACKAGE_INDEX_URL}",
        "--build-arg", f"AMDGPU_FAMILY={args.AMDGPU_FAMILY}",
    ]

    # Build
    run(["docker", "build", "-t", args.tag, "-f", args.dockerfile] + build_args + [args.context])
    ## Push
    #run(["docker", "push", args.tag])

    # Append only the IMAGE (tag) to the GitHub Actions step summary
    gha_append_step_summary(f"IMAGE: {args.tag}")

if __name__ == "__main__":
    main()
