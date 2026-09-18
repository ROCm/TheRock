#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Temporary CI harness for real RAND artifacts; not a full SDK release build."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

BUILD_TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUILD_TOOLS))

from _therock_utils.artifacts import ArtifactCatalog
from _therock_utils.cmake_amdgpu_targets import amdgpu_family_map, expand_families
from _therock_utils.py_packaging import Parameters, PopulatedDistPackage
from build_python_packages import (
    _run_kpack_split,
    core_artifact_filter,
    discover_llvm_host_triple,
    load_therock_manifest,
    validate_kpack_split_target_completeness,
    validate_required_dist_packages,
)
from rocm_kpack.kpack import PackedKernelArchive


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_artifacts(artifact_dir: Path, targets: list[str]) -> dict[str, str]:
    catalog = ArtifactCatalog(artifact_dir)
    if catalog.all_target_families != set(targets):
        raise ValueError(
            f"Unexpected artifact targets: {catalog.all_target_families}; expected {targets}"
        )
    expected: dict[str, str] = {}
    for target in targets:
        archives = sorted(
            path
            for path in (artifact_dir / f"rand_lib_{target}").rglob("*.kpack")
            if path.is_file()
        )
        if not archives:
            raise ValueError(f"No RAND kpack payload for {target}")
        for path in archives:
            archive = PackedKernelArchive.read(path)
            keys = {arch for entries in archive.toc.values() for arch in entries}
            if keys != {target} or set(archive.gfx_arches) != {target}:
                raise ValueError(f"Unexpected target keys in {path}: {keys}")
            if path.name in expected:
                raise ValueError(f"Duplicate archive name: {path.name}")
            expected[path.name] = digest(path)
    print(json.dumps({"targets": targets, "archive_sha256": expected}, indent=2))
    return expected


def build_sdk(
    artifact_dir: Path,
    output: Path,
    version: str,
    targets: list[str],
    expected: dict[str, str],
) -> None:
    if not load_therock_manifest(artifact_dir)["flags"]["KPACK_SPLIT_ARTIFACTS"]:
        raise ValueError("Expected kpack-split artifacts")
    artifacts = ArtifactCatalog(artifact_dir)
    validate_kpack_split_target_completeness(
        kpack_split=True,
        artifact_dir=artifact_dir,
        artifacts=artifacts,
        linux_targets=targets,
        windows_targets=None,
    )
    params = Parameters(
        dest_dir=output,
        version=version,
        version_suffix="",
        artifacts=artifacts,
        kpack_split=True,
        linux_target_families=targets,
    )
    host_triple = discover_llvm_host_triple(artifacts)
    core = PopulatedDistPackage(params, logical_name="core")
    core.rpath_dep(core, "lib/llvm/lib")
    if host_triple:
        core.rpath_dep(core, f"lib/llvm/lib/{host_triple}")
    core.rpath_dep(core, "lib/rocm_sysdeps/lib")
    core.populate_runtime_files(params.filter_artifacts(core_artifact_filter))
    _run_kpack_split(
        argparse.Namespace(
            dest_dir=output,
            build_packages=True,
            wheel_compression=False,
            devel_tarball_compression=False,
        ),
        params,
        core,
        host_triple,
    )
    validate_required_dist_packages(
        dest_dir=output,
        version=version,
        artifacts=artifacts,
        kpack_split=True,
        linux_targets=targets,
        windows_targets=None,
    )
    device_wheels = list((output / "dist").glob("rocm_sdk_device_*.whl"))
    if len(device_wheels) != 1 or not device_wheels[0].name.startswith(
        "rocm_sdk_device_gfx1250-"
    ):
        raise ValueError(f"Expected one gfx1250 device wheel: {device_wheels}")
    with zipfile.ZipFile(device_wheels[0]) as wheel:
        actual = {}
        for member in wheel.namelist():
            if member.endswith(".kpack"):
                name = Path(member).name
                if name in actual:
                    raise ValueError(f"Duplicate wheel payload: {member}")
                with wheel.open(member) as stream:
                    actual[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(
            f"Device wheel payload mismatch: {actual}; expected {expected}"
        )

    # Build an installable meta wheel for this experiment; normal output is an sdist.
    install_env = dict(os.environ, ROCM_SDK_TARGET_FAMILY=targets[0])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output / "dist"),
            str(output / "rocm"),
        ],
        env=install_env,
        check=True,
    )
    venv = output / "install-venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / "bin/python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(output / "dist"),
            *map(str, sorted((output / "dist").glob("*.whl"))),
        ],
        env=install_env,
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    report = output / "expected-archives.json"
    report.write_text(json.dumps(expected, indent=2))
    for target in targets:
        env = dict(os.environ, ROCM_SDK_TARGET_FAMILY=target)
        subprocess.run([str(venv / "bin/rocm-sdk"), "init"], env=env, check=True)
        subprocess.run(
            [
                str(python),
                "-I",
                str(Path(__file__).with_name("validate_gfx1250_rand_install.py")),
                str(report),
                target,
            ],
            env=env,
            check=True,
        )


def check_native(output: Path, package_format: str, expected: dict[str, str]) -> None:
    packages: dict[str, Path] = {}
    dependencies: dict[str, str] = {}
    for path in sorted(output.glob(f"*.{package_format}")):
        if package_format == "deb":
            name = subprocess.check_output(
                ["dpkg-deb", "--field", str(path), "Package"], text=True
            ).strip()
            requires = subprocess.check_output(
                ["dpkg-deb", "--field", str(path), "Depends"], text=True
            )
        else:
            name = subprocess.check_output(
                ["rpm", "-qp", "--queryformat", "%{NAME}", str(path)], text=True
            ).strip()
            requires = subprocess.check_output(
                ["rpm", "-qp", "--requires", str(path)], text=True
            )
        if name in packages or "gfx1250-strict" in name + requires:
            raise ValueError(f"Duplicate or unexpected package identity: {name}")
        packages[name] = path
        dependencies[name] = requires
    devices = [name for name in packages if "-gfx" in name]
    if len(devices) != 1 or not devices[0].endswith("-gfx1250"):
        raise ValueError(f"Expected one gfx1250 device package: {devices}")
    device = devices[0]
    meta = device.removesuffix("-gfx1250")
    if dependencies.get(meta, "").count(device) != 1:
        raise ValueError(f"Expected exactly one dependency on {device} in {meta}")
    print(json.dumps(dependencies, indent=2))
    package = packages[device]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "extracted"
        root.mkdir()
        if package_format == "deb":
            subprocess.run(
                ["dpkg-deb", "--extract", str(package), str(root)], check=True
            )
        else:
            archive = Path(temporary) / "payload.cpio"
            with archive.open("wb") as stream:
                subprocess.run(["rpm2cpio", str(package)], stdout=stream, check=True)
            with archive.open("rb") as stream:
                subprocess.run(
                    [
                        "cpio",
                        "--extract",
                        "--make-directories",
                        "--no-absolute-filenames",
                    ],
                    stdin=stream,
                    cwd=root,
                    check=True,
                )
        actual = {}
        for path in root.rglob("*.kpack"):
            if not path.is_file():
                continue
            if path.name in actual:
                raise ValueError(f"Duplicate native payload: {path}")
            actual[path.name] = digest(path)
        if actual != expected:
            raise ValueError(f"Native payload mismatch: {actual}; expected {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("artifacts", "sdk", "native"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("packages"))
    parser.add_argument("--version", default="0.0.1.dev20260918")
    parser.add_argument("--package-format", choices=("deb", "rpm"), default="deb")
    args = parser.parse_args()
    targets = expand_families(args.families.split(";"), amdgpu_family_map())
    if not targets or not set(targets) <= {"gfx1250", "gfx1250-strict"}:
        raise ValueError(f"Outside the RAND validation scope: {targets}")
    artifact_dir = args.artifact_dir.resolve()
    output = args.output_dir.resolve()
    expected = check_artifacts(artifact_dir, targets)
    if args.mode == "sdk":
        build_sdk(artifact_dir, output, args.version, targets, expected)
    elif args.mode == "native":
        check_native(output, args.package_format, expected)


if __name__ == "__main__":
    main()
