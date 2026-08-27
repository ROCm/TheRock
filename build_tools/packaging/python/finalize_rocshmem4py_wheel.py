#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply TheRock-specific metadata to rocshmem4py wheels."""

import argparse
import email.parser
import email.policy
import functools
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

_BUILD_TOOLS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.determine_version import derive_version_suffix

_CHANGE_WHEEL_VERSION_DIR = _BUILD_TOOLS_DIR / "third_party" / "change_wheel_version"

ROCSHMEM4PY_RPATH = ":".join(
    [
        "$ORIGIN/_rocm_sdk_core/lib",
        "$ORIGIN/_rocm_sdk_core/lib/rocm_sysdeps/lib",
        "$ORIGIN/_rocm_sdk_core/lib/llvm/lib",
    ]
)


def compute_version(built_version: str, rocm_version: str) -> str:
    """Replace upstream local metadata with TheRock's ROCm build identity."""
    version = Version(built_version)
    rocm_suffix = derive_version_suffix(str(Version(rocm_version)))
    return str(Version(f"{version.public}{rocm_suffix}"))


def _set_runtime_requirement(metadata_path: Path, rocm_version: str) -> None:
    parser = email.parser.BytesParser(
        policy=email.policy.default.clone(max_line_length=0)
    )
    metadata = parser.parsebytes(metadata_path.read_bytes())
    requirements = metadata.get_all("Requires-Dist", [])
    while metadata.get("Requires-Dist") is not None:
        del metadata["Requires-Dist"]
    for requirement_text in requirements:
        requirement = Requirement(requirement_text)
        if canonicalize_name(requirement.name) != "rocm-sdk-core":
            metadata["Requires-Dist"] = requirement_text
    metadata["Requires-Dist"] = f"rocm-sdk-core=={Version(rocm_version)}"
    metadata_path.write_bytes(metadata.as_bytes())


def _update_wheel_contents(
    wheel_root: Path,
    _old_version: Version,
    _new_version: Version,
    *,
    rocm_version: str,
    patchelf: str,
) -> None:
    metadata_paths = list(wheel_root.glob("*.dist-info/METADATA"))
    extensions = list(wheel_root.glob("_rocshmem4py*.so"))
    if len(metadata_paths) != 1 or len(extensions) != 1:
        raise ValueError(
            f"Expected one METADATA file and extension under {wheel_root}; "
            f"found {metadata_paths} and {extensions}"
        )

    _set_runtime_requirement(metadata_paths[0], rocm_version)
    subprocess.run(
        [
            patchelf,
            "--set-rpath",
            ROCSHMEM4PY_RPATH,
            "--force-rpath",
            str(extensions[0]),
        ],
        check=True,
    )


def finalize_wheel(
    wheel_path: Path,
    *,
    output_dir: Path,
    rocm_version: str,
    patchelf: str,
) -> Path:
    sys.path.insert(0, str(_CHANGE_WHEEL_VERSION_DIR))
    from change_wheel_version import change_wheel_version

    _distribution, old_version, _build, _tags = parse_wheel_filename(wheel_path.name)
    version = Version(compute_version(str(old_version), rocm_version))

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_wheel = output_dir / wheel_path.name
    shutil.copy2(wheel_path, staged_wheel)
    finalized_wheel = change_wheel_version(
        wheel=staged_wheel,
        version=version.public,
        local_version=version.local,
        callback_func=functools.partial(
            _update_wheel_contents,
            rocm_version=rocm_version,
            patchelf=patchelf,
        ),
    )
    if finalized_wheel != staged_wheel:
        staged_wheel.unlink()
    return finalized_wheel


def _gpu_targets(wheel_path: Path, llvm_objdump: str) -> set[str]:
    with zipfile.ZipFile(wheel_path) as wheel:
        extensions = [
            name
            for name in wheel.namelist()
            if re.fullmatch(r"_rocshmem4py.*\.so", name)
        ]
        if len(extensions) != 1:
            raise ValueError(
                f"Expected one extension in {wheel_path}; found {extensions}"
            )
        with tempfile.TemporaryDirectory(prefix="rocshmem4py-inspect-") as temp_dir:
            extension = Path(wheel.extract(extensions[0], temp_dir))
            offloading = subprocess.check_output(
                [llvm_objdump, "--offloading", str(extension)], text=True
            )
    return set(re.findall(r"hipv\d+-amdgcn-amd-amdhsa--(gfx[0-9a-z]+)", offloading))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--expected-gpu-target", action="append", required=True)
    parser.add_argument("--patchelf", default="patchelf")
    parser.add_argument("--llvm-objdump", default="llvm-objdump")
    args = parser.parse_args(argv)

    input_wheels = sorted(args.input_dir.glob("rocshmem4py-*.whl"))
    if not input_wheels:
        parser.error(f"No rocshmem4py wheels found in {args.input_dir}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent
    ) as staging_dir_str:
        staging_dir = Path(staging_dir_str)
        finalized_wheels = [
            finalize_wheel(
                wheel,
                output_dir=staging_dir,
                rocm_version=args.rocm_version,
                patchelf=args.patchelf,
            )
            for wheel in input_wheels
        ]

        expected_gpu_targets = set(args.expected_gpu_target)
        for wheel in finalized_wheels:
            actual_gpu_targets = _gpu_targets(wheel, args.llvm_objdump)
            if actual_gpu_targets != expected_gpu_targets:
                raise ValueError(
                    f"GPU target mismatch in {wheel.name}: "
                    f"embedded={sorted(actual_gpu_targets)}, "
                    f"expected={sorted(expected_gpu_targets)}"
                )

        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        staging_dir.replace(args.output_dir)

    for wheel in finalized_wheels:
        print(f"Finalized {wheel.name}")


if __name__ == "__main__":
    main()
