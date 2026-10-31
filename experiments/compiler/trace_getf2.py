#!/usr/bin/env python3
"""Replay a captured solver specialization in Rock CI, with optional Clang tracing.

Use separate, interleaved control/trace trials to measure profiling overhead.
No performance or HIP-correctness validation has been performed on this helper.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

BASE = "5b7cb88373362ab80dc0f19605915e534f4158aa"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand(argv: list[str], cwd: Path, depth: int = 0) -> list[str]:
    if depth > 8:
        raise ValueError("Response-file nesting exceeds eight levels")
    result = []
    for value in argv:
        if value.startswith("@"):
            path = Path(value[1:])
            path = path if path.is_absolute() else cwd / path
            result += expand(shlex.split(path.read_text()), cwd, depth + 1)
        else:
            result.append(value)
    return result


def redirect(argv: list[str], output: Path) -> list[str]:
    names = {"-o": "replay.o", "-MF": "replay.d", "-MT": "replay.o", "-MQ": "replay.o"}
    result, index, outputs = [], 0, 0
    while index < len(argv):
        value = argv[index]
        if value in names:
            if index + 1 == len(argv):
                raise ValueError(f"Missing argument after {value}")
            result += [value, str(output / names[value])]
            outputs += value == "-o"
            index += 2
        else:
            if value.startswith(("-ftime-trace", "-save-temps", "-serialize-diagnostics",
                                 "--serialize-diagnostics", "-MF", "-MT", "-MQ", "-MJ", "-o")):
                raise ValueError(f"Unsupported pre-existing output/profiling option: {value}")
            result.append(value)
            index += 1
    if outputs != 1 or "-c" not in argv:
        raise ValueError("Expected CMake's separate -o argument and compilation-only command")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--device-link-trace", action="store_true")
    parser.add_argument("--device-linker", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--expected-source", choices=("getf2-small", "getri-specialized"),
                        default="getf2-small")
    args = parser.parse_args()
    if (sys.platform != "linux" or os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != "ROCm/TheRock"):
        parser.error("Compilation is allowed only inside Linux ROCm/TheRock Actions")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source_root,
                                  text=True).strip()
    if sha != BASE:
        parser.error(f"Expected frozen source SHA {BASE}, got {sha}")
    record = json.loads(args.record.read_text())
    cwd = Path(record["cwd"])
    argv = expand(record["argv"], cwd)
    patterns = {"getf2-small": ("roclapack_getf2_small_", "getf2-small-shards/"),
                "getri-specialized": ("roclapack_getri_specialized_kernels_",)}
    if not any(a.endswith(".cpp") and any(p in a for p in patterns[args.expected_source])
               for a in argv):
        parser.error(f"Expected a represented {args.expected_source} translation unit")
    if Path(argv[0]).name not in ("clang++", "amdclang++", "clang", "amdclang"):
        parser.error("Expected the pinned Clang driver")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    argv = redirect(argv, output)
    version = subprocess.check_output([argv[0], "--version"], text=True)
    help_text = ""
    if args.trace or args.device_link_trace:
        help_text = subprocess.check_output([argv[0], "--help"], text=True)
        (output / "compiler-help.txt").write_text(help_text)
    if args.trace:
        if "-ftime-trace" not in help_text:
            parser.error("Compiler does not advertise -ftime-trace")
        argv.append("-ftime-trace")
    linker_info = None
    if args.device_link_trace:
        if "-Xoffload-linker" not in help_text:
            parser.error("Compiler does not advertise -Xoffload-linker")
        linker = args.device_linker or Path(argv[0]).with_name("ld.lld")
        if not linker.is_file():
            parser.error("Supply the exact device ld.lld path captured in the baseline")
        linker_help = subprocess.check_output([str(linker), "--help"], text=True)
        (output / "device-linker-help.txt").write_text(linker_help)
        if "--time-trace=" not in linker_help:
            parser.error("Device linker does not advertise --time-trace")
        trace_path = output / "device-lld.json"
        argv += ["-Xoffload-linker", f"--time-trace={trace_path}"]
        linker_info = {"path": str(linker), "version": subprocess.check_output(
            [str(linker), "--version"], text=True)}
    # Verify parsing and driver forwarding before spending time on a compile.
    dry_run = subprocess.run(argv + ["-###"], cwd=cwd, text=True, capture_output=True)
    dry_text = dry_run.stdout + dry_run.stderr
    (output / "driver-dry-run.txt").write_text(dry_text)
    if dry_run.returncode or (args.device_link_trace and
                             f"--time-trace={trace_path}" not in dry_text):
        parser.error("Driver dry run did not accept/forward the requested flags")
    if args.check_only:
        (output / "capability.json").write_text(json.dumps({
            "source_sha": sha, "argv": argv, "compiler_version": version,
            "device_linker": linker_info, "dry_run_exit": dry_run.returncode,
            "compiled": False}, indent=2))
        return 0
    import resource
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.monotonic()
    code = subprocess.run(argv, cwd=cwd).returncode
    wall = time.monotonic() - start
    ended_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    traces = [str(p) for p in output.rglob("*.json")]
    hash_start = time.monotonic()
    objects = [{"path": str(p), "bytes": p.stat().st_size,
                "sha256": file_hash(p)}
               for p in output.glob("*.o")]
    metadata = {"source_sha": sha, "compiler_version": version, "argv": argv,
                "utc": started_utc, "end_utc": ended_utc,
                "artifact_hash_wall_s_outside_compile": time.monotonic() - hash_start,
                "expected_source": args.expected_source,
                "run_id": os.environ.get("GITHUB_RUN_ID"),
                "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                "job": os.environ.get("GITHUB_JOB"), "runner": os.environ.get("RUNNER_NAME"),
                "device_link_trace": args.device_link_trace, "device_linker": linker_info,
                "record_sha256": hashlib.sha256(args.record.read_bytes()).hexdigest(),
                "environment": {key: os.environ.get(key) for key in
                                ("PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "HIP_PATH", "HIP_DIR",
                                 "ROCM_PATH", "ROCM_DIR", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH",
                                 "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "CCACHE_DISABLE")},
                "cwd": str(cwd), "trace": args.trace, "exit_code": code, "wall_s": wall,
                "user_cpu_s": after.ru_utime - before.ru_utime,
                "system_cpu_s": after.ru_stime - before.ru_stime,
                "max_child_rss_kib_including_preflight": after.ru_maxrss, "trace_files": traces,
                "objects": objects,
                "limits": ["Environment and generated headers must match the captured build.",
                           "Inspect every host/device trace; one file may not cover all HIP phases.",
                           "Interleave unprofiled controls; observer CPU is not elapsed perturbation."]}
    (output / "replay-metadata.json").write_text(json.dumps(metadata, indent=2))
    if code == 0 and args.trace and not traces:
        raise RuntimeError("No trace produced; replay does not resolve compiler phase costs")
    if code == 0 and args.device_link_trace and not (output / "device-lld.json").is_file():
        raise RuntimeError("No device LLD trace produced; device-link cost remains unresolved")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
