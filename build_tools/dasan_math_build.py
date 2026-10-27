#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build gfx950 math libraries using an installed experimental DASan SDK.

Run inside TheRock's manylinux build image. The input is the extracted ``rocm``
directory from the compiler/runtime workflow installation artifact. Compiler and
runtime payloads are imported as prebuilt stages; missing host dependencies are
built normally. This does not rebuild LLVM or HIP.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import shlex
import sys

from dasan_build import REPO, capture, run, write_json
from _therock_utils.pattern_match import PatternMatcher

# Installation artifacts merge component stage roots. Restore only the files
# owned by each compiler/runtime stage, so math artifacts cannot accidentally
# incorporate an entire SDK under every component. Paths match the corresponding
# artifact TOML descriptors and CMake package providers.
SDK_STAGES = {
    "compiler/amd-llvm/stage": (
        ["lib/llvm/**"],
        "lib/llvm/bin/clang",
    ),
    "compiler/amd-comgr/stage": (
        [
            "include/amd_comgr*",
            "include/amd_comgr/**",
            "lib/libamd_comgr*",
            "lib/cmake/amd_comgr/**",
            "share/doc/amd_comgr/**",
        ],
        "lib/cmake/amd_comgr",
    ),
    "compiler/hipcc/stage": (
        ["bin/hipcc*", "bin/hipconfig*", "share/hipcc/**"],
        "bin/hipcc",
    ),
    "core/ROCR-Runtime/stage": (
        [
            "include/hsa/**",
            "include/hsakmt/**",
            "lib/libhsa-runtime*",
            "lib/libhsakmt*",
            "lib/cmake/hsa-runtime64/**",
            "lib/cmake/hsakmt/**",
            "share/doc/rocr/**",
        ],
        "lib/cmake/hsa-runtime64",
    ),
    "core/rocminfo/stage": (
        ["bin/rocminfo", "bin/rocm_agent_enumerator*"],
        "bin/rocminfo",
    ),
    "core/clr/stage": (
        [
            "include/hip/**",
            "include/hip_version.h",
            "include/hip_prof_str.h",
            "lib/libamdhip*",
            "lib/libhiprtc*",
            "lib/.hipInfo",
            "lib/cmake/hip/**",
            "lib/cmake/hip-lang/**",
            "lib/cmake/hiprtc/**",
            "share/hip/**",
            "bin/hip*",
        ],
        "lib/cmake/hip",
    ),
    "core/rocm-kpack/stage": (
        [
            "include/rocm_kpack/**",
            "lib/librocm_kpack*",
            "lib/librocm-kpack*",
            "lib/cmake/rocm-kpack/**",
        ],
        "lib/cmake/rocm-kpack",
    ),
}

SOURCE_PATHS = (
    "base/half",
    "base/rocm-cmake",
    "rocm-libraries",
    "rocm-systems",
    "math-libs/libhipcxx",
)


def prepare_sources(log: Path, jobs: int) -> None:
    # fetch_sources resets tracked files and reapplies patches. Never run it on
    # existing developer checkouts, including an interrupted previous attempt.
    initialized = [path for path in SOURCE_PATHS if (REPO / path / ".git").exists()]
    if initialized:
        raise RuntimeError(
            "--prepare-sources requires a fresh checkout; already initialized: "
            + ", ".join(initialized)
        )
    run(
        [
            sys.executable,
            str(REPO / "build_tools/fetch_sources.py"),
            "--stage",
            "math-libs",
            "--source-sets",
            "base",
            "--jobs",
            str(min(jobs, 12)),
            "--depth",
            "1",
        ],
        log,
    )


def bootstrap_sdk(rocm: Path, output: Path) -> None:
    clang = rocm / "lib/llvm/bin/clang"
    if not clang.is_file():
        raise RuntimeError(f"Missing DASan compiler: {clang}")
    digest = hashlib.sha256()
    with clang.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    identity = {"rocm_path": str(rocm), "clang_sha256": digest.hexdigest()}
    build_dir = output / "build"
    imported = {}
    for relative_stage, (patterns, required) in SDK_STAGES.items():
        if not (rocm / required).exists():
            raise RuntimeError(f"Incomplete compiler/runtime SDK: missing {required}")
        stage = build_dir / relative_stage
        marker = stage.with_name(stage.name + ".prebuilt")
        marker_text = str(identity) + "\n" + "\n".join(patterns) + "\n"
        if marker.exists():
            if marker.read_text(encoding="utf-8") != marker_text:
                raise RuntimeError(
                    f"SDK changed; use a fresh output directory: {stage}"
                )
            if not (stage / required).exists():
                raise RuntimeError(f"Incomplete imported stage: {stage / required}")
        else:
            if stage.exists():
                raise RuntimeError(f"Refusing to overwrite an existing stage: {stage}")
            matcher = PatternMatcher(includes=patterns)
            matcher.add_basedir(rocm)
            matcher.copy_to(destdir=stage)
            marker.write_text(marker_text, encoding="utf-8")
        imported[relative_stage] = patterns
    write_json(output / "bootstrap.json", {**identity, "stages": imported})


def sdk_llvm_major(rocm: Path) -> int:
    """Use the imported compiler version, not an absent LLVM source checkout."""
    version = capture([str(rocm / "lib/llvm/bin/clang"), "-dumpversion"])
    match = re.fullmatch(r"([1-9][0-9]*)(?:\.[0-9]+)*(?:git)?", version)
    if not match:
        raise RuntimeError(f"Unexpected SDK clang -dumpversion output: {version!r}")
    return int(match.group(1))


def probe_sdk(rocm: Path, output: Path) -> int:
    """Compile and link without needing a GPU on the CI builder."""
    probe = output / "sdk-probe"
    probe.mkdir(exist_ok=True)
    log = output / "sdk-probe.log"
    llvm_major = sdk_llvm_major(rocm)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"Imported SDK LLVM major version: {llvm_major}\n")
    host = probe / "host.c"
    host.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    clang = str(rocm / "lib/llvm/bin/clang")
    run([clang, "-fsanitize=daddress", str(host), "-o", str(probe / "host")], log)
    source = probe / "device.hip"
    source.write_text(
        "#include <hip/hip_runtime.h>\n"
        "__global__ void write_element(int *p, int i) { p[i] = 1; }\n"
        "int main() { return 0; }\n",
        encoding="utf-8",
    )
    flags = [
        clang,
        "-xhip",
        f"--rocm-path={rocm}",
        f"--hip-path={rocm}",
        f"--hip-device-lib-path={rocm / 'lib/llvm/amdgcn/bitcode'}",
        "-isystem",
        str(rocm / "include"),
        "--offload-arch=gfx950",
        "-fsanitize=daddress",
        "-O1",
        str(source),
    ]
    ir = probe / "device.ll"
    run([*flags, "--offload-device-only", "-S", "-emit-llvm", "-o", str(ir)], log)
    if "__dasan_report_store" not in ir.read_text(encoding="utf-8"):
        raise RuntimeError("gfx950 probe has no DASan device store instrumentation")
    run(
        [
            *flags,
            f"-L{rocm / 'lib'}",
            "-lamdhip64",
            "-o",
            str(probe / "device"),
        ],
        log,
    )
    write_json(
        output / "sdk-probe.json",
        {
            "host_link": "passed",
            "gfx950_link": "passed",
            "device_instrumentation": True,
            "llvm_major": llvm_major,
        },
    )
    return llvm_major


def record_sources(output: Path, rocm: Path, artifacts: list[str]) -> None:
    sources = {}
    for relative in SOURCE_PATHS:
        directory = REPO / relative
        if not (directory / ".git").exists():
            raise RuntimeError(f"Missing source checkout: {relative}")
        if capture(["git", "status", "--porcelain", "--untracked-files=no"], directory):
            raise RuntimeError(f"Tracked source changes in {relative}")
        sources[relative] = capture(["git", "rev-parse", "HEAD"], directory)
    write_json(
        output / "sources.json",
        {
            "therock_commit": capture(["git", "rev-parse", "HEAD"]),
            "rocm_path": str(rocm),
            "gpu_target": "gfx950",
            "sanitizer": "DASAN",
            "compiler_flag": "-fsanitize=daddress",
            "artifacts": artifacts,
            "sources": sources,
        },
    )


def build(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rocm = args.rocm_path.resolve()
    build_dir = output / "build"
    log = output / "build.log"
    status = {"started_utc": datetime.now(timezone.utc).isoformat(), "state": "running"}
    write_json(output / "status.json", status)
    try:
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
        os.environ["CCACHE_DIR"] = str(output / "ccache")
        os.environ["PIP_CACHE_DIR"] = str(output / "pip-cache")
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            [
                str(rocm / "lib"),
                str(rocm / "lib/rocm_sysdeps/lib"),
                str(rocm / "lib/llvm/lib"),
                os.environ.get("LD_LIBRARY_PATH", ""),
            ]
        )
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], log)
        if args.prepare_sources:
            prepare_sources(log, args.jobs)
        bootstrap_sdk(rocm, output)
        llvm_major = probe_sdk(rocm, output)

        from configure_stage import generate_cmake_args, get_topology

        topology = get_topology()
        artifacts = sorted(
            artifact.name
            for artifact in topology.artifacts.values()
            if artifact.artifact_group == "math-libs"
        )
        record_sources(output, rocm, artifacts)
        # The math-libs stage contains ML libraries too. Explicit artifact names
        # enable only math libraries and their transitive dependencies.
        stage_args = generate_cmake_args(
            "math-libs",
            "gfx950",
            "gfx950",
            topology,
            platform_name="linux",
            manylinux=True,
            artifact_names=artifacts,
        )
        command = [
            "cmake",
            "-S",
            str(REPO),
            "-B",
            str(build_dir),
            "-GNinja",
            *[shlex.split(arg)[0] for arg in stage_args],
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_C_COMPILER={rocm / 'lib/llvm/bin/clang'}",
            f"-DCMAKE_CXX_COMPILER={rocm / 'lib/llvm/bin/clang++'}",
            "-DTHEROCK_BUNDLE_SYSDEPS=ON",
            "-DTHEROCK_ENABLE_DASAN=ON",
            "-DTHEROCK_SANITIZER=DASAN",
            "-DTHEROCK_BUILD_TESTING=OFF",
            "-DTHEROCK_BUILD_COMGR_TESTS=OFF",
            "-DTHEROCK_BUILD_LLVM_TESTS=OFF",
            # Regenerate the overlay's manifests and gfx950 metadata, but name
            # its clang configuration alias using the imported SDK's version.
            "-Dtherock-aux-overlay_CMAKE_ARGS="
            f"-DTHEROCK_AUX_OVERLAY_LLVM_VERSION_MAJOR={llvm_major}",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache",
        ]
        write_json(output / "configure.json", {"argv": command})
        run(command, log)
        if not args.configure_only:
            run(
                [
                    "cmake",
                    "--build",
                    str(build_dir),
                    "--parallel",
                    str(args.jobs),
                    "--target",
                    "stage-math-libs",
                    "therock-artifacts",
                    "therock-dist",
                ],
                log,
            )
            for source, filename, entry in (
                (build_dir / "dist", "dasan-math-gfx950.tar.gz", "rocm"),
                (build_dir, "dasan-math-artifacts-gfx950.tar.gz", "artifacts"),
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
    parser.add_argument("--rocm-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dasan-math-output"))
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--prepare-sources", action="store_true")
    parser.add_argument("--configure-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build(args)


if __name__ == "__main__":
    main()
