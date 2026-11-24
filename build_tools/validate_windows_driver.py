#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate staged Windows driver binaries before publishing them."""

import argparse
from pathlib import Path
import re


REQUIRED_RUNTIME_ARTIFACTS = (
    "core/clr/stage/bin/amdhip64_*.dll",
    "core/clr/stage/bin/hiprtc[0-9]*.dll",
    "core/clr/stage/bin/hiprtc-builtins*.dll",
    "core/ocl-clr/stage/bin/amdocl64.dll",
    "core/ocl-icd/stage/bin/OpenCL.dll",
    "core/rocm-kpack/stage/bin/rocm_kpack.dll",
)


def validate_artifacts(
    build_dir: Path, *, comgr_dll_name: str = "amd_comgr.dll"
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.dll", comgr_dll_name):
        raise ValueError(f"Invalid Comgr DLL filename: {comgr_dll_name!r}")
    comgr_pdb_name = Path(comgr_dll_name).with_suffix(".pdb").name
    required_artifacts = (
        f"compiler/amd-comgr/stage/bin/{comgr_dll_name}",
        f"compiler/amd-comgr/stage/bin/{comgr_pdb_name}",
        *REQUIRED_RUNTIME_ARTIFACTS,
    )
    for pattern in required_artifacts:
        matches = list(build_dir.glob(pattern))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one driver artifact matching {pattern}, found {len(matches)}"
            )
        artifact = matches[0]
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"Driver artifact is not a nonempty file: {artifact}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--comgr-dll-name",
        default="amd_comgr.dll",
        help="Expected Comgr DLL filename, matching THEROCK_COMGR_DLL_NAME.",
    )
    args = parser.parse_args(argv)
    validate_artifacts(args.build_dir, comgr_dll_name=args.comgr_dll_name)
    print("Validated Windows driver Comgr and runtime artifacts.")


if __name__ == "__main__":
    main()
