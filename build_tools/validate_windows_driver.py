#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate staged Windows driver binaries before publishing them."""

import argparse
from pathlib import Path


REQUIRED_ARTIFACTS = (
    "compiler/amd-comgr/stage/bin/amd_comgr.dll",
    "compiler/amd-comgr/stage/bin/amd_comgr.pdb",
    "core/clr/stage/bin/amdhip64_*.dll",
    "core/clr/stage/bin/hiprtc[0-9]*.dll",
    "core/clr/stage/bin/hiprtc-builtins*.dll",
    "core/ocl-clr/stage/bin/amdocl64.dll",
    "core/ocl-icd/stage/bin/OpenCL.dll",
    "core/rocm-kpack/stage/bin/rocm_kpack.dll",
)


def validate_artifacts(build_dir: Path) -> None:
    for pattern in REQUIRED_ARTIFACTS:
        matches = list(build_dir.glob(pattern))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one driver artifact matching {pattern}, found {len(matches)}"
            )
        artifact = matches[0]
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"Driver artifact is not a nonempty file: {artifact}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    validate_artifacts(args.build_dir)
    print("Validated Windows driver Comgr and runtime artifacts.")


if __name__ == "__main__":
    main()
