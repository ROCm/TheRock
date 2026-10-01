#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Pin, patch, then HIPIFY the integration CI PyTorch sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PINS = {
    "release/2.12": "b97872d55cf78ed4a8dd9f71ce7767e35b39f2fc",
    "release/2.13": "8e0af7fa4388411493540afa152777537bbbfc1c",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", choices=PINS, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here / "pytorch"
    subprocess.run([sys.executable, str(here / "pytorch_torch_repo.py"), "checkout", "--gitrepo-origin", "https://github.com/ROCm/pytorch.git", "--repo-hashtag", PINS[args.ref], "--no-hipify"], check=True)
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != PINS[args.ref]:
        raise SystemExit(f"Unexpected PyTorch source: {actual}")
    patches = [here / "strict-patches" / "pytorch-3667.patch", here / "strict-patches" / f"ck-empty-targets-{args.ref.split('/')[-1]}.patch"]
    for patch in patches:
        subprocess.run(["git", "-C", str(root), "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", str(root), "apply", str(patch)], check=True)
    receipt = {"ref": args.ref, "source": actual, "pr": "https://github.com/ROCm/pytorch/pull/3667", "pr_merge": "991b967f6c9966d1a93ce2515613d55d4d171e85", "adaptation": "CK-only empty-target guard from PR3639 intent; release2.12 backport, no global target removal", "patch_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in patches}, "qualification": "source overlay only; embedded AOTriton/Triton and GPU execution not qualified"}
    (root / "strict-source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    subprocess.run([sys.executable, str(here / "pytorch_torch_repo.py"), "hipify"], check=True)


if __name__ == "__main__":
    main()
