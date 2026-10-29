#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build one downstream gfx950 DASan scope using an installed compiler SDK.

Run in TheRock's manylinux image. Component tests are compiled, never executed.
Dependency artifacts are reused as built in their original workflow; only the
selected scope and missing dependencies are rebuilt from this checkout.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys

from dasan_build import REPO, capture, run, write_json
from dasan_math_build import bootstrap_sdk, probe_sdk
from _therock_utils.build_topology import BuildTopology
from _therock_utils.artifacts import (
    ArtifactName,
    ArtifactPopulator,
    prebuilt_marker_relpath,
)
from _therock_utils.pattern_match import PatternMatcher

STAGES = (
    "emulation",
    "runtime-tests",
    "comm-libs",
    "cv-libs",
    "storage-libs",
    "debug-tools",
    "dctools-core",
    "profiler-apps",
    "media-libs",
    "hipdnn",
)
TEST_ENABLED_STAGES = {
    "runtime-tests",
    "comm-libs",
    "storage-libs",
    "dctools-core",
    "profiler-apps",
}
SDK_ARTIFACTS = {"base", "amd-llvm", "hipify", "core-runtime", "core-hip", "core-kpack"}
EXTRA_ARTIFACTS = {}
# These targets do not have native CMake compilation databases. Their build
# mechanism is recorded rather than misrepresenting them as instrumented C++.
NON_NATIVE_TARGETS = {
    "mirage": "Rust project; daddress applies to LLVM C/C++/HIP compilation",
    "rocjitsu-hotswap": "restages the instrumented rocjitsu payload",
}


def select_artifacts(topology: BuildTopology, stage: str) -> tuple[set[str], set[str]]:
    if stage == "hipdnn":
        primary = {"hipdnn"}
    else:
        groups = topology.build_stages[stage].artifact_groups
        primary = {
            a.name
            for a in topology.artifacts.values()
            if a.artifact_group in groups
            and "linux" not in a.disable_platforms
            and a.platform in (None, "linux")
        }
    closure = set(primary) | EXTRA_ARTIFACTS.get(stage, set())
    pending = list(closure)
    while pending:
        for dependency in topology.artifacts[pending.pop()].artifact_deps:
            if dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)
    return primary, closure


def audit_profiler_test_payload(build_dir: Path) -> None:
    """Fail the producer if its selected full profiler suites cannot be staged."""
    artifacts = build_dir / "artifacts"
    examples = [
        path for path in artifacts.iterdir()
        if path.is_dir()
        and (name := ArtifactName.from_path(path))
        and name.name == "rocprofiler-systems-examples"
        and name.target_family in ("generic", "gfx950")
    ]
    if not examples or not any(
        (path / "artifact_manifest.txt").is_file()
        and any(
            "share/rocprofiler-systems/examples/" in line
            for line in (path / "artifact_manifest.txt").read_text().splitlines()
        )
        for path in examples
    ):
        raise RuntimeError("Profiler producer lacks rocprofiler-systems-examples test artifact")
    rocm = build_dir / "dist/rocm"
    required = (
        "libexec/rocprofiler-compute/CTestTestfile.cmake",
        "share/rocprofiler-systems/tests/CTestTestfile.cmake",
    )
    missing = [path for path in required if not (rocm / path).is_file()]
    if missing:
        raise RuntimeError("Profiler producer lacks installed full suite payload: " + ", ".join(missing))


def source_sets(topology: BuildTopology, artifacts: set[str]) -> list[str]:
    result = {"base", "rocm-systems"}
    for name in artifacts:
        group = topology.artifact_groups[topology.artifacts[name].artifact_group]
        result.update(group.source_sets)
    result.discard("compilers")  # Every compiler payload comes from the SDK.
    return sorted(result)


def source_paths(topology: BuildTopology, sets: list[str]) -> list[str]:
    names = {s.name for name in sets for s in topology.source_sets[name].submodules}
    result = []
    for name in sorted(names):
        result.append(
            capture(
                [
                    "git",
                    "config",
                    "--file",
                    ".gitmodules",
                    "--get",
                    f"submodule.{name}.path",
                ]
            )
        )
    return result


def prepare_sources(
    topology: BuildTopology, stage: str, sets: list[str], log: Path, jobs: int
) -> None:
    paths = source_paths(topology, sets)
    initialized = [path for path in paths if (REPO / path / ".git").exists()]
    if initialized:
        raise RuntimeError(
            "--prepare-sources requires a fresh checkout: " + ", ".join(initialized)
        )
    run(
        [
            sys.executable,
            str(REPO / "build_tools/fetch_sources.py"),
            "--stage",
            "math-libs" if stage == "hipdnn" else stage,
            "--source-sets",
            *sets,
            "--skip-submodules",
            "llvm-project",
            "HIPIFY",
            "spirv-llvm-translator",
            "--jobs",
            str(min(jobs, 12)),
            "--depth",
            "1",
        ],
        log,
    )


def bootstrap_hipify(rocm: Path, output: Path) -> None:
    required = ("bin/hipify-clang", "bin/hipify-perl", "libexec/hipify/hipify-perl")
    for path in required:
        if not (rocm / path).is_file():
            raise RuntimeError(f"Incomplete SDK hipify payload: {path}")
    stage = output / "build/compiler/hipify/stage"
    marker = stage.with_name("stage.prebuilt")
    identity = (
        str(rocm) + "\n" + hashlib.sha256((rocm / required[0]).read_bytes()).hexdigest()
    )
    if marker.exists():
        if marker.read_text() != identity or not all(
            (stage / p).is_file() for p in required
        ):
            raise RuntimeError(
                "Hipify SDK changed or import incomplete; use a fresh output directory"
            )
        return
    if stage.exists():
        raise RuntimeError(f"Refusing to replace existing stage: {stage}")
    matcher = PatternMatcher(includes=["bin/hipify*", "libexec/hipify/**"])
    matcher.add_basedir(rocm)
    matcher.copy_to(destdir=stage)
    marker.write_text(identity)


def import_dependencies(
    directory: Path | None, output: Path, primary: set[str], closure: set[str]
) -> dict:
    """Import only missing dependencies, never SDK or selected producer stages."""
    selected = []
    manifests = {}
    allowed = closure - primary - SDK_ARTIFACTS
    if directory:
        directory = directory.resolve()
        if not directory.is_dir():
            raise RuntimeError(f"Missing extracted dependency artifacts: {directory}")
        for path in sorted(directory.iterdir()):
            name = ArtifactName.from_path(path)
            if not path.is_dir() or not name or name.name not in allowed:
                continue
            if name.target_family not in ("generic", "gfx950"):
                continue
            manifest = path / "artifact_manifest.txt"
            payload = manifest.read_text()
            manifests[path.name] = hashlib.sha256(payload.encode()).hexdigest()
            selected.append(path)
    provenance = {
        "baseline_run_id": os.environ.get("BASELINE_RUN_ID"),
        "dependencies_run_id": os.environ.get("DEPENDENCIES_RUN_ID"),
        "directory": str(directory) if directory else None,
        "selected_artifacts": [path.name for path in selected],
        "manifest_sha256": manifests,
        "reuse_policy": "Reused dependency payloads retain their original run's source and build flags; not rebuilt with current patches.",
    }
    record = output / "dependency-import.json"
    build = output / "build"
    relpaths = set()
    for path in selected:
        for relative in (path / "artifact_manifest.txt").read_text().splitlines():
            if not relative:
                continue
            pure = PurePosixPath(relative)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or ":" in relative
                or "\\" in relative
            ):
                raise RuntimeError(f"Unsafe dependency manifest path: {relative}")
            if relative.startswith(
                (
                    "compiler/",
                    "core/clr/",
                    "core/ROCR-Runtime/",
                    "core/rocminfo/",
                    "core/rocm-kpack/",
                    "base/",
                )
            ):
                raise RuntimeError(
                    f"Dependency artifact overlaps protected SDK/base stage: {relative}"
                )
            relpaths.add(relative)
    provenance["stage_paths"] = sorted(relpaths)
    if record.exists():
        if json.loads(record.read_text()) != provenance:
            raise RuntimeError(
                "Dependency selection changed; use a fresh output directory"
            )
        for relative in relpaths:
            if (
                not (build / relative).exists()
                or not (build / prebuilt_marker_relpath(relative)).is_file()
            ):
                raise RuntimeError(f"Incomplete dependency import: {relative}")
        return provenance
    for relative in relpaths:
        if (build / relative).exists() or (
            build / prebuilt_marker_relpath(relative)
        ).exists():
            raise RuntimeError(f"Refusing to overwrite dependency stage: {relative}")
    populator = ArtifactPopulator(output_path=build)
    populator(*selected)
    for relative in relpaths:
        marker = build / prebuilt_marker_relpath(relative)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(provenance, sort_keys=True))
    write_json(record, provenance)
    return provenance


def has_daddress(command: str) -> bool:
    return any(
        "daddress" in value.split(",")
        for value in re.findall(r"-fsanitize=([^\s\"']+)", command)
    )


def audit_gpu_targets(output: Path) -> None:
    """Check effective generated lists before any nested project compiles.

    Test artifacts default to every architecture independently of the normal
    and distribution targets. Inspect generated toolchains to catch either
    accidental widening or a project's fallback architecture.
    """
    build = output / "build"
    selections = {}
    failures = []
    for path in sorted(build.rglob("*_toolchain.cmake")):
        if not path.with_name(
            path.name.replace("_toolchain.cmake", "_init.cmake")
        ).is_file():
            continue
        variables = {}
        for name, value in re.findall(
            r'^set\((AMDGPU_TARGETS|GPU_TARGETS|CMAKE_HIP_ARCHITECTURES)\s+(.*?)\s+CACHE STRING "From super-project" FORCE\)$',
            path.read_text(),
            re.MULTILINE,
        ):
            targets = value.strip('"').split(";")
            variables[name] = targets
            if targets != ["gfx950"]:
                failures.append(f"{path.relative_to(build)}: {name}={value}")
        if variables:
            selections[path.relative_to(build).as_posix()] = variables
    report = {
        "requested_target": "gfx950",
        "generated_toolchains": selections,
        "failures": failures,
    }
    write_json(output / "target-selection.json", report)
    if failures or not selections:
        raise RuntimeError(
            "GPU target selection audit failed: "
            + "; ".join(failures or ["no generated GPU toolchains"])
        )


def unsupported_device_fixture(command: str) -> bool:
    """Only explicitly targeted SPIR-V fixtures cannot use DASan."""
    return bool(
        re.search(
            r"(?:--offload-arch[= ](?:amdgcnspirv|spirv)|--target[= ]spirv|-target\s+spirv)",
            command,
        )
    )


def shell_commands(command: str) -> list[str]:
    """Do not let one instrumented command hide its uninstrumented sibling."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    commands = []
    current = []
    for token in lexer:
        if token in (";", "&&", "||", "|", "&"):
            if current:
                commands.append(shlex.join(current))
            current = []
        else:
            current.append(token)
    if current:
        commands.append(shlex.join(current))
    return commands


def assembly_device_fixture(command: str) -> bool:
    tokens = shlex.split(command)
    if any(
        value in ("assembler", "assembler-with-cpp")
        for index, value in enumerate(tokens)
        if index and tokens[index - 1] == "-x"
    ):
        return True
    if any(value in ("-xassembler", "-xassembler-with-cpp") for value in tokens):
        return True
    # Recognize an assembly input, never merely an output named .s.
    inputs = [
        value
        for index, value in enumerate(tokens)
        if not value.startswith("-")
        and (
            not index
            or tokens[index - 1] not in ("-o", "-MF", "-MT", "-MQ", "-include")
        )
    ]
    return any(Path(value).suffix in (".s", ".S") for value in inputs) and not any(
        Path(value).suffix.lower() in (".c", ".cc", ".cpp", ".cxx", ".hip", ".cl")
        for value in inputs
    )


def read_compile_commands(path: Path) -> list[dict]:
    # The subproject machinery touches an empty file for non-CMake builders.
    content = path.read_text() if path.exists() else ""
    return json.loads(content) if content.strip() else []


def audit_rocgdb(build: Path) -> dict:
    records = {}
    failures = []
    for path in sorted(
        (build / "debug-tools/rocgdb").rglob("rocgdb-build-py*/config.log")
    ):
        text = path.read_text(errors="replace")
        flags = {
            name: re.findall(rf"^{name}=(.*)$", text, re.MULTILINE)
            for name in ("CFLAGS", "CXXFLAGS", "LDFLAGS")
        }
        if not all(
            values and all(has_daddress(value) for value in values)
            for values in flags.values()
        ):
            failures.append(str(path))
        records[str(path.relative_to(build))] = flags
    binaries = {}
    stage = build / "debug-tools/rocgdb/stage"
    for path in stage.rglob("*rocgdb*"):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        dynamic = capture(["readelf", "-d", str(path)])
        needed = [line.strip() for line in dynamic.splitlines() if "NEEDED" in line]
        binaries[str(path.relative_to(build))] = needed
        if not any(
            "clang_rt.asan" in line or "clang_rt.dasan" in line for line in needed
        ):
            failures.append(str(path))
    return {
        "mechanism": "Autotools CFLAGS/CXXFLAGS/LDFLAGS and installed ELF sanitizer runtime",
        "configurations": records,
        "elf_needed": binaries,
        "passed": bool(records and binaries) and not failures,
        "failures": failures,
    }


def audit_instrumentation(
    output: Path, primary: set[str], closure: set[str], *, complete: bool
) -> None:
    build = output / "build"
    mapping = json.loads((build / "artifact_subprojects.json").read_text())
    dependencies = {
        target for name in closure - primary for target in mapping.get(name, [])
    }
    targets = sorted(
        {target for name in primary for target in mapping.get(name, [])} - dependencies
    )
    records = {}
    failures = []
    for target in targets:
        fragment = build / f"compile_commands_fragment_{target}.json"
        rows = read_compile_commands(fragment)
        if target == "therock-amd-mesa":
            # Meson generates its own database after the wrapper's configure.
            rows = [
                row
                for path in (build / "third-party/sysdeps/linux/amd-mesa").rglob(
                    "compile_commands.json"
                )
                for row in read_compile_commands(path)
            ]
        commands = []
        for row in rows:
            if Path(row.get("file", "")).suffix.lower() not in (
                ".c",
                ".cc",
                ".cpp",
                ".cxx",
                ".hip",
                ".cl",
            ):
                continue
            if "CMakeFiles/CMakeScratch/" in row.get("file", "").replace("\\", "/"):
                continue
            commands.append(row.get("command") or shlex.join(row["arguments"]))
        bad = [command for command in commands if not has_daddress(command)]
        record = {
            "native_commands": len(commands),
            "daddress_commands": len(commands) - len(bad),
            "missing_daddress": bad,
            "example_command": commands[0] if commands else None,
        }
        if target in NON_NATIVE_TARGETS:
            record["exception"] = NON_NATIVE_TARGETS[target]
        elif target == "rocgdb" and complete:
            record["autotools"] = audit_rocgdb(build)
            if not record["autotools"]["passed"]:
                failures.append(
                    "rocgdb: missing Autotools or ELF instrumentation evidence"
                )
        elif complete and not commands:
            failures.append(f"{target}: missing native compilation evidence")
        if bad:
            failures.append(f"{target}: {len(bad)} native commands omit daddress")
        # Expanded Ninja commands capture custom HIP/OpenCL compilation omitted
        # from compile_commands.json. Keep full offending commands in S3 logs.
        device_commands = set()
        for directory in {Path(row["directory"]) for row in rows}:
            if not (directory / "build.ninja").exists():
                continue
            result = subprocess.run(
                ["ninja", "-C", str(directory), "-t", "commands"],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in (
                command
                for line in result.stdout.splitlines()
                for command in shell_commands(line)
            ):
                if re.search(
                    r"(?:clang(?:\+\+)?(?:-[0-9]+)?|hipcc)(?:\s|\")", line
                ) and re.search(
                    r"--offload-arch|--cuda-gpu-arch|-mcpu=gfx|-target\s+amdgcn|-x\s*hip|-x\s+cl",
                    line,
                ):
                    device_commands.add(line)
        exempt_device = sorted(
            c for c in device_commands if unsupported_device_fixture(c)
        )
        assembly_device = sorted(
            c for c in device_commands if assembly_device_fixture(c)
        )
        missing_device = sorted(
            c
            for c in device_commands
            if not has_daddress(c)
            and not unsupported_device_fixture(c)
            and not assembly_device_fixture(c)
        )
        record.update(
            device_commands=len(device_commands),
            device_daddress_commands=sum(has_daddress(c) for c in device_commands),
            device_missing_daddress=missing_device,
            unsupported_spirv_commands=exempt_device,
            assembly_commands=assembly_device,
        )
        if missing_device:
            failures.append(
                f"{target}: {len(missing_device)} device commands omit daddress"
            )
        records[target] = record
    report = {
        "phase": "built" if complete else "configured",
        "components": records,
        "failures": failures,
        "gpu_tests_executed": False,
    }
    write_json(output / "instrumentation-audit.json", report)
    if complete and (failures or not targets):
        raise RuntimeError(
            "DASan instrumentation audit failed: "
            + "; ".join(failures or ["no selected subprojects"])
        )


def build(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rocm = args.rocm_path.resolve()
    build_dir = output / "build"
    log = output / "build.log"
    status = {
        "stage": args.stage,
        "state": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "status.json", status)
    try:
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
        os.environ["CCACHE_DIR"] = str(output / "ccache")
        os.environ["PIP_CACHE_DIR"] = str(output / "pip-cache")
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            [str(rocm / p) for p in ("lib", "lib/rocm_sysdeps/lib", "lib/llvm/lib")]
            + [os.environ.get("LD_LIBRARY_PATH", "")]
        )
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], log)
        from configure_stage import generate_cmake_args, get_topology

        topology = get_topology()
        primary, closure = select_artifacts(topology, args.stage)
        sets = source_sets(topology, closure)
        if args.prepare_sources:
            prepare_sources(topology, args.stage, sets, log, args.jobs)
        for requirement in sorted(
            {
                req
                for name in closure
                for req in topology.artifacts[name].python_requires
            }
        ):
            run(
                [sys.executable, "-m", "pip", "install", *shlex.split(requirement)], log
            )
        if args.stage == "runtime-tests":
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "pytest-cmake",
                    "jsonschema",
                    "-r",
                    "rocm-systems/projects/rocprofiler-sdk/requirements-test.txt",
                ],
                log,
            )
        bootstrap_sdk(rocm, output)
        if "hipify" in closure:
            bootstrap_hipify(rocm, output)
        import_dependencies(args.dependency_artifacts, output, primary, closure)
        llvm_major = probe_sdk(rocm, output)
        sources = {}
        for path in source_paths(topology, sets):
            if not (REPO / path / ".git").exists():
                raise RuntimeError(f"Missing source checkout: {path}")
            if capture(
                ["git", "status", "--porcelain", "--untracked-files=no"], REPO / path
            ):
                raise RuntimeError(f"Tracked source changes in {path}")
            sources[path] = capture(["git", "rev-parse", "HEAD"], REPO / path)
        write_json(
            output / "sources.json",
            {
                "therock_commit": capture(["git", "rev-parse", "HEAD"]),
                "sources": sources,
                "stage": args.stage,
                "primary_artifacts": sorted(primary),
                "dependency_closure": sorted(closure),
                "source_sets": sets,
                "gpu_target": "gfx950",
                "sanitizer": "DASAN",
                "compiler_flag": "-fsanitize=daddress",
                "build_testing": "ON" if args.stage in TEST_ENABLED_STAGES else "OFF",
                "baseline_run_id": os.environ.get("BASELINE_RUN_ID"),
                "dependencies_run_id": os.environ.get("DEPENDENCIES_RUN_ID"),
            },
        )
        stage_args = generate_cmake_args(
            "",
            "gfx950",
            "gfx950",
            topology,
            platform_name="linux",
            manylinux=True,
            artifact_names=sorted(closure),
        )
        command = [
            "cmake",
            "-S",
            str(REPO),
            "-B",
            str(build_dir),
            "-GNinja",
            *[shlex.split(arg)[0] for arg in stage_args],
            "-DTHEROCK_AMDGPU_FAMILIES=",
            "-DTHEROCK_DIST_AMDGPU_FAMILIES=",
            "-DTHEROCK_AMDGPU_TARGETS=gfx950",
            "-DTHEROCK_DIST_AMDGPU_TARGETS=gfx950",
            "-DTHEROCK_TEST_AMDGPU_TARGETS=gfx950",
            "-DTHEROCK_TEST_AMDGPU_FAMILIES=",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            f"-DCMAKE_C_COMPILER={rocm / 'lib/llvm/bin/clang'}",
            f"-DCMAKE_CXX_COMPILER={rocm / 'lib/llvm/bin/clang++'}",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            "-DTHEROCK_ENABLE_BASE=ON",
            "-DTHEROCK_ENABLE_MPI=ON",
            "-DTHEROCK_ENABLE_HOST_BLAS=OFF",
            "-DTHEROCK_ENABLE_HOST_SUITE_SPARSE=OFF",
            "-DTHEROCK_ENABLE_FFTW3=OFF",
            "-DTHEROCK_BUNDLE_SYSDEPS=ON",
            "-DTHEROCK_ENABLE_DASAN=ON",
            "-DTHEROCK_SANITIZER=DASAN",
            f"-DTHEROCK_BUILD_TESTING={'ON' if args.stage in TEST_ENABLED_STAGES else 'OFF'}",
            "-DTHEROCK_BUILD_COMGR_TESTS=OFF",
            "-DTHEROCK_BUILD_LLVM_TESTS=OFF",
            f"-Dtherock-aux-overlay_CMAKE_ARGS=-DTHEROCK_AUX_OVERLAY_LLVM_VERSION_MAJOR={llvm_major}",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache",
        ]
        write_json(output / "configure.json", {"argv": command})
        run(command, log)
        audit_gpu_targets(output)
        audit_instrumentation(output, primary, closure, complete=False)
        if not args.configure_only:
            targets = ["therock-artifacts", "therock-dist"]
            if args.stage != "hipdnn":
                targets.insert(0, f"stage-{args.stage}")
            run(
                [
                    "cmake",
                    "--build",
                    str(build_dir),
                    "--parallel",
                    str(args.jobs),
                    "--target",
                    *targets,
                ],
                log,
            )
            if args.stage == "profiler-apps":
                audit_profiler_test_payload(build_dir)
            audit_instrumentation(output, primary, closure, complete=True)
            if args.stage == "comm-libs":
                run(
                    [
                        sys.executable,
                        str(REPO / "build_tools/audit_dasan_rccl.py"),
                        "--rocm-path",
                        str(build_dir / "dist/rocm"),
                        "--llvm-bin",
                        str(rocm / "lib/llvm/bin"),
                        "--kpack-python",
                        str(REPO / "rocm-systems/shared/kpack/python"),
                        "--output-dir",
                        str(build_dir / "logs/rccl-device"),
                        "--arch",
                        "gfx950",
                    ],
                    log,
                )
            for source, filename, entry in (
                (build_dir / "dist", f"dasan-{args.stage}-gfx950.tar.gz", "rocm"),
                (build_dir, f"dasan-{args.stage}-artifacts-gfx950.tar.gz", "artifacts"),
            ):
                run(
                    ["tar", "-czf", str(output / filename), "-C", str(source), entry],
                    log,
                )
        status["state"] = "configured" if args.configure_only else "build-complete"
    except BaseException as exc:
        status.update(state="failed", error=str(exc))
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\nFAILED: {exc}\n")
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "status.json", status)
        from github_actions.collect_dasan_logs import collect_logs

        collect_logs(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--rocm-path", type=Path, required=True)
    parser.add_argument("--dependency-artifacts", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dasan-component-output")
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--prepare-sources", action="store_true")
    parser.add_argument("--configure-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build(args)


if __name__ == "__main__":
    main()
