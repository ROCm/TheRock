#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare normalized native DASan stages without cross-run binary reuse."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from dasan_build import PINS, REPO, apply_dasan_patches, capture, run, write_json
from _therock_utils.build_topology import get_topology
from dasan_component_build import has_daddress


def compiler_options(stage: str, build: Path) -> list[str]:
    options = [
        "-DTHEROCK_AMDGPU_FAMILIES=",
        "-DTHEROCK_DIST_AMDGPU_FAMILIES=",
        "-DTHEROCK_TEST_AMDGPU_FAMILIES=",
        "-DTHEROCK_AMDGPU_TARGETS=gfx950",
        "-DTHEROCK_DIST_AMDGPU_TARGETS=gfx950",
        "-DTHEROCK_TEST_AMDGPU_TARGETS=gfx950",
        "-DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=gfx950-dcgpu",
    ]
    if stage != "compiler-runtime":
        clang = build / "compiler/amd-llvm/stage/lib/llvm/bin/clang"
        version = subprocess.check_output([str(clang), "--version"], text=True)
        match = re.search(r"clang version (\d+)", version)
        if not match:
            raise RuntimeError(f"Cannot determine imported compiler version: {version}")
        options += [
            f"-DCMAKE_C_COMPILER={clang}",
            f"-DCMAKE_CXX_COMPILER={clang.parent / 'clang++'}",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            # The compiler CMake targets reference rocm-cmake from base even
            # when the SDK stages have already been imported from this run.
            "-DTHEROCK_ENABLE_BASE=ON",
            f"-Dtherock-aux-overlay_CMAKE_ARGS=-DTHEROCK_AUX_OVERLAY_LLVM_VERSION_MAJOR={match[1]}",
        ]
    return options


def prepare(stage: str, build: Path, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    log = output / "prepare.log"
    for path, (origin, revision) in PINS.items():
        if (REPO / path / ".git").exists():
            raise RuntimeError(
                f"DASan source preparation requires fresh checkout: {path}"
            )
        run(["git", "update-index", "--cacheinfo", f"160000,{revision},{path}"], log)
    run(["git", "config", "submodule.rocm-systems.url", PINS["rocm-systems"][0]], log)
    fetch = [
        sys.executable,
        "build_tools/fetch_sources.py",
        "--stage",
        stage,
        "--source-sets",
        "base",
        "rocm-systems",
    ]
    # First fetch the exact gitlinks; apply normal compatibility patches next.
    run([*fetch, "--jobs", "12", "--depth", "1", "--no-apply-patches"], log)
    run([*fetch, "--no-update-submodules"], log)
    if stage == "compiler-runtime":
        apply_dasan_patches(log)
    sources = {}
    for path, (origin, revision) in PINS.items():
        if not (REPO / path / ".git").exists():
            continue
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=REPO / path
        )
        sources[path] = {
            "origin": origin,
            "requested_commit": revision,
            "effective_commit": capture(["git", "rev-parse", "HEAD"], REPO / path),
        }
    patch_files = sorted((REPO / "patches/dasan/llvm-project").glob("*.patch"))
    write_json(
        output / "sources.json",
        {
            "stage": stage,
            "therock_commit": capture(["git", "rev-parse", "HEAD"]),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "sources": sources,
            "submodule_inventory": capture(
                ["git", "submodule", "status", "--recursive"]
            ),
            "produced_artifacts": sorted(get_topology().get_produced_artifacts(stage)),
            "sanitizer": "DASAN",
            "binary_dependency_policy": "current workflow run only",
            "ccache_disabled": os.environ.get("CCACHE_DISABLE") == "1",
            "compiler_patches": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in patch_files
            },
        },
    )
    return compiler_options(stage, build)


def database_commands(rows: object) -> list[str]:
    """Validate the command-bearing fields of a compilation database."""
    if not isinstance(rows, list):
        raise ValueError("compilation database root must be an array")
    commands = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"entry {index} must be an object")
        command = row.get("command")
        arguments = row.get("arguments")
        if command is not None and not isinstance(command, str):
            raise ValueError(f"entry {index} command must be a string")
        if arguments is not None and (
            not isinstance(arguments, list)
            or not all(isinstance(argument, str) for argument in arguments)
        ):
            raise ValueError(f"entry {index} arguments must be an array of strings")
        if command and command.strip():
            commands.append(command)
        elif arguments:
            commands.append(shlex.join(arguments))
        else:
            raise ValueError(f"entry {index} requires a nonempty command or arguments")
    return commands


def audit(stage: str, build: Path, output: Path) -> dict:
    """Retain compiled-command evidence, including explicit bootstrap exceptions."""
    databases = []
    failures = []
    total = instrumented = 0
    for database in sorted(build.rglob("compile_commands.json")):
        record = {"path": str(database.relative_to(build))}
        databases.append(record)
        try:
            text = database.read_text(encoding="utf-8-sig")
            if not text.strip():
                record.update(
                    status="empty-not-yet-configured",
                    commands=0,
                    daddress_commands=0,
                    without_daddress=[],
                    note="No generated commands are available; not instrumentation evidence.",
                )
                continue
            commands = database_commands(json.loads(text))
        except (OSError, UnicodeError, ValueError) as error:
            message = f"{record['path']}: {type(error).__name__}: {error}"
            record.update(status="invalid", error=message)
            failures.append(message)
            continue
        good = sum(has_daddress(c) for c in commands)
        total += len(commands)
        instrumented += good
        record.update(
            {
                "status": "configured" if commands else "configured-no-commands",
                "commands": len(commands),
                "daddress_commands": good,
                "without_daddress": [c for c in commands if not has_daddress(c)],
            }
        )
    report = {
        "stage": stage,
        "native_commands": total,
        "daddress_commands": instrumented,
        "databases": databases,
        "gpu_tests_executed": False,
        "exclusions": [
            "amd-llvm, hipify, hipcc bootstrap tools",
            "system-toolchain dependencies",
            "Rust targets (mirage)",
        ],
        "failures": failures,
        "limitations": "Bootstrap compiler/system dependencies and Rust are not device-instrumented. Command evidence does not prove GPU execution.",
    }
    if stage != "compiler-runtime" and total and not instrumented:
        failures.append("Native commands exist but none enable -fsanitize=daddress")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "instrumentation-audit.json", report)
    if report["failures"]:
        raise RuntimeError(
            "DASan stage instrumentation audit failed: " + report["failures"][0]
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "audit"])
    parser.add_argument("--stage", required=True)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/logs/dasan"))
    args = parser.parse_args()
    build = args.build_dir.resolve()
    if args.action == "audit":
        audit(args.stage, build, args.output_dir)
    else:
        options = prepare(args.stage, build, args.output_dir)
        if output := os.environ.get("GITHUB_OUTPUT"):
            with open(output, "a", encoding="utf-8") as stream:
                stream.write(f"cmake_args={shlex.join(options)}\n")
        else:
            print(shlex.join(options))


if __name__ == "__main__":
    main()
