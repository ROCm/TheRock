#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build non-math components using the immutable successful DASan CI installation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from dasan_build import IMAGE, PINS, REPO, apply_dasan_patches, record_sources, run
from configure_stage import generate_cmake_args, get_topology

STAGES = (
    "compiler-runtime",
    "emulation",
    "runtime-tests",
    "cv-libs",
    "storage-libs",
    "debug-tools",
    "dctools-core",
    "media-libs",
    "comm-libs",
    "profiler-apps",
)
PATCHES = (
    "emulation-portability.patch",
    "emulation-fp-nan.patch",
    "mirage-fixture-races.patch",
    "rocprofiler-sdk-installed-tests.patch",
    "rdc-installed-kernel-tests.patch",
)
LAYOUT = REPO / "build_tools/dasan_components/baseline-layout.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def file_record(path):
    if path.is_symlink():
        return {"symlink": os.readlink(path)}
    return {"sha256": digest(path)}


def import_installation(archive, output):
    layout = json.loads(LAYOUT.read_text())
    if digest(archive) != layout["archive_sha256"]:
        raise RuntimeError(
            "Baseline archive differs from successful CI run 40443090683"
        )
    baseline = output / "baseline"
    baseline.mkdir()
    subprocess.run(
        ["tar", "-xJf", str(archive), "--no-same-owner", "-C", str(baseline)],
        check=True,
    )
    prefix = baseline / "rocm"
    manifest = json.loads((prefix / "share/therock/therock_manifest.json").read_text())
    if str(manifest["github_run_id"]) != str(layout["baseline_run_id"]):
        raise RuntimeError("Wrong embedded baseline run ID")
    rules = layout["rules"]
    records = {}
    stages = set()
    for source in sorted(prefix.rglob("*")):
        if source.is_dir() and not source.is_symlink():
            continue
        relative = source.relative_to(prefix).as_posix()
        # Rebuild this small overlay to record the new run's source provenance.
        if relative.startswith("share/therock/"):
            continue
        matches = [
            rule
            for rule in rules
            if relative == rule["path"]
            or (rule["directory"] and relative.startswith(rule["path"] + "/"))
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Ambiguous or missing baseline ownership: {relative}")
        records[relative] = file_record(source)
        for stage in matches[0]["stages"]:
            if stage == "base/aux-overlay/stage":
                continue
            destination = output / "build" / stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                destination.symlink_to(os.readlink(source))
            else:
                # Hardlink only immutable prebuilt stages; they are never installed over.
                os.link(source, destination)
            stages.add(stage)
    for stage in stages:
        (output / "build" / stage).with_name("stage.prebuilt").write_text(
            "Imported from ROCm/TheRock CI run 40443090683\n"
        )
    save(output / "baseline-files.json", records)
    save(
        output / "reuse.json",
        {
            "run_id": layout["baseline_run_id"],
            "artifact_id": 12707464644,
            "archive_sha256": layout["archive_sha256"],
            "image": IMAGE,
            "imported_stages": sorted(stages),
            "imported_files": len(records),
            "layout_source": layout["layout_source"],
            "payload_source": "GitHub Actions artifact only; no local build binaries",
        },
    )


def prepare(output, archive, log):
    if any((REPO / path / ".git").exists() for path in PINS):
        raise RuntimeError("Source preparation requires a fresh isolated checkout")
    for path, (_, revision) in PINS.items():
        run(["git", "update-index", "--cacheinfo", f"160000,{revision},{path}"], log)
    run(["git", "config", "submodule.rocm-systems.url", PINS["rocm-systems"][0]], log)
    fetch = [
        sys.executable,
        "build_tools/fetch_sources.py",
        "--no-include-ml-frameworks",
        "--no-include-math-libraries",
    ]
    run(fetch + ["--jobs", "12", "--depth", "1", "--no-apply-patches"], log)
    run(fetch + ["--no-update-submodules"], log)
    apply_dasan_patches(log)
    patches = [REPO / "patches/dasan/components" / name for name in PATCHES]
    systems = REPO / "rocm-systems"
    for patch in patches:
        run(["git", "apply", "--check", str(patch)], log, cwd=systems)
        run(["git", "apply", str(patch)], log, cwd=systems)
    run(
        [
            "git",
            "-c",
            "user.name=therockbot",
            "-c",
            "user.email=therockbot@amd.com",
            "commit",
            "-am",
            "Apply validated non-math component compatibility fixes",
            "--no-gpg-sign",
        ],
        log,
        cwd=systems,
    )
    record_sources(output)
    save(output / "component-patches.json", {p.name: digest(p) for p in patches})
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "rocm-systems/projects/rocprofiler-sdk/requirements-test.txt",
        ],
        log,
    )
    run(
        [
            "rustup",
            "toolchain",
            "install",
            "1.95.0",
            "--profile",
            "minimal",
            "--component",
            "clippy",
            "--component",
            "rustfmt",
        ],
        log,
    )
    import_installation(archive, output)


def configure(output, log):
    flags = {}
    for stage in STAGES:
        for rendered in generate_cmake_args(
            stage, "", "", get_topology(), platform_name="linux", manylinux=True
        ):
            arg = shlex.split(rendered)[0]
            flags[arg.split("=", 1)[0]] = arg
    for name, value in {
        "BASE": "ON",
        "COMPILER": "ON",
        "HIPDNN": "ON",
        "FLATBUFFERS": "ON",
        "NLOHMANN_JSON": "ON",
        "MPI": "ON",
        "DASAN": "ON",
        "HOST_BLAS": "OFF",
        "HOST_SUITE_SPARSE": "OFF",
        "FFTW3": "OFF",
    }.items():
        key = f"-DTHEROCK_ENABLE_{name}"
        flags[key] = f"{key}={value}"
    command = [
        "cmake",
        "-S",
        str(REPO),
        "-B",
        str(output / "build"),
        "-GNinja",
        *flags.values(),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DTHEROCK_BUNDLE_SYSDEPS=ON",
        "-DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=gfx942-gfx950",
        "-DTHEROCK_AMDGPU_TARGETS=gfx942;gfx950",
        "-DTHEROCK_DIST_AMDGPU_TARGETS=gfx942;gfx950",
        "-DTHEROCK_TEST_AMDGPU_TARGETS=gfx942;gfx950",
        "-DTHEROCK_BUILD_TESTING=ON",
        "-DTHEROCK_BUILD_COMGR_TESTS=OFF",
        "-DTHEROCK_BUILD_LLVM_TESTS=ON",
        "-DLLVM_PARALLEL_LINK_JOBS=4",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache",
    ]
    save(output / "configure.json", {"argv": command})
    run(command, log)


def verify(output):
    prefix = output / "build/dist/rocm"
    records = json.loads((output / "baseline-files.json").read_text())
    changed = [
        relative
        for relative, record in records.items()
        if not (prefix / relative).exists() or file_record(prefix / relative) != record
    ]
    compiler_rebuilt = (output / "build/compiler/amd-llvm/build/build.ninja").exists()
    save(
        output / "reuse-verification.json",
        {
            "status": "failed" if changed or compiler_rebuilt else "passed",
            "checked_files": len(records),
            "changed_or_missing": changed,
            "compiler_build_generated": compiler_rebuilt,
        },
    )
    if changed or compiler_rebuilt:
        raise RuntimeError("Baseline compiler/runtime reuse audit failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("components-output"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--phase",
        choices=["prepare", "configure", "verify", *STAGES, "hipdnn", "finish"],
        required=True,
    )
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 8, 48))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
    os.environ["CCACHE_DIR"] = str(output / "ccache")
    log = output / f"{args.phase}.log"
    if args.phase == "prepare":
        if not args.archive:
            parser.error("prepare requires --archive")
        prepare(output, args.archive.resolve(), log)
    elif args.phase == "configure":
        configure(output, log)
    elif args.phase == "verify":
        verify(output)
    else:
        targets = (
            ["therock-artifacts", "therock-dist"]
            if args.phase == "finish"
            else ["hipdnn"] if args.phase == "hipdnn" else [f"stage-{args.phase}"]
        )
        run(
            [
                "cmake",
                "--build",
                str(output / "build"),
                "--parallel",
                "2",
                "--target",
                *targets,
            ],
            log,
        )


if __name__ == "__main__":
    main()
