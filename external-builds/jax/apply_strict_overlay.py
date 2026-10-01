#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Apply pinned MI45x source overlays to the integration JAX checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

JAX_SHA = "7353a2f071e7935f720358a757b7e539f47a2dd1"
XLA_SHA = "40613bb4aea1f0cad475da79c570cd394563e83e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.jax_dir.resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != JAX_SHA:
        raise SystemExit(f"Unexpected JAX source: {actual}")
    patches = Path(__file__).resolve().parent / "strict-patches"
    module = root / "MODULE.bazel"
    text = module.read_text()
    anchor = '    module_name = "xla",\n'
    if text.count(anchor) != 1 or f"xla-{XLA_SHA}" not in text:
        raise SystemExit("Unexpected XLA archive override")
    jax_patch = patches / "jax-40877.patch"
    subprocess.run(["git", "-C", str(root), "apply", "--check", str(jax_patch)], check=True)
    subprocess.run(["git", "-C", str(root), "apply", str(jax_patch)], check=True)
    dest = root / "third_party" / "mi45x_strict"
    dest.mkdir()
    shutil.copyfile(patches / "xla-49806.patch", dest / "xla-49806.patch")
    (dest / "BUILD.bazel").write_text('exports_files(["xla-49806.patch"])\n')
    text = text.replace(anchor, anchor + '    patches = ["//third_party/mi45x_strict:xla-49806.patch"],\n    patch_strip = 1,\n')
    module.write_text(text)
    receipt = {"jax_base": JAX_SHA, "xla_base": XLA_SHA,
               "jax_pr": "https://github.com/jax-ml/jax/pull/40877",
               "xla_pr": "https://github.com/openxla/xla/pull/49806",
               "xla_pr_head": "5a06c6676f02493d5d4c303ee055ef30385c2073",
               "patch_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in patches.glob("*.patch")},
               "qualification": "source overlay only; no GPU execution qualification"}
    (root / "strict-source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
