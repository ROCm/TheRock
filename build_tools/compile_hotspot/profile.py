# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reproduce one CK compilation with pinned dependencies; no speedup claims."""

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

SOURCE_SHA = "5b7cb88373362ab80dc0f19605915e534f4158aa"
SOURCE = "device_grouped_conv2d_bwd_data_xdl_nhwgc_gkyxc_nhwgk_f32_instance.cpp"
ROOT = Path.cwd().resolve()
OUT = ROOT / "compile-hotspot-evidence"
BUILD = ROOT / "build"
CK = BUILD / "math-libs/composable_kernel/build"


def save(name: str, value: object) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def run(argv: list[str], *, cwd: Path = ROOT, log: str | None = None) -> str:
    print(f"[{cwd}] {shlex.join(argv)}", flush=True)
    if log:
        with (OUT / log).open("w") as stream:
            subprocess.run(
                argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True
            )
        return ""
    return subprocess.check_output(argv, cwd=cwd, text=True)


def fingerprint(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def configure(args: argparse.Namespace) -> None:
    if run(["git", "rev-parse", "HEAD"]).strip() != SOURCE_SHA:
        raise RuntimeError("Product source does not match pinned SHA")
    if args.variant == "release" and args.baseline_run_id == "41211496307":
        raise RuntimeError(
            "Normal release must not import the ASan dependency baseline"
        )
    save(
        "context.json",
        {
            "source_sha": SOURCE_SHA,
            "instrumentation_sha": os.getenv("GITHUB_SHA"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "baseline_run_id": args.baseline_run_id,
            "variant": args.variant,
            "source_basename": SOURCE,
            "runner_name": os.getenv("RUNNER_NAME"),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "cache_contract": "CCACHE_DISABLE=1; filesystem caches uncontrolled",
            "scope": "Isolated object replay; not a full math-stage timing or optimization claim.",
        },
    )
    for source in (
        Path("/proc/meminfo"),
        Path("/proc/cpuinfo"),
        Path("/sys/fs/cgroup/cpu.max"),
        Path("/sys/fs/cgroup/memory.max"),
    ):
        if source.is_file():
            (OUT / source.name).write_bytes(source.read_bytes())
    run(
        [
            sys.executable,
            "build_tools/artifact_manager.py",
            "fetch",
            f"--run-id={args.baseline_run_id}",
            "--run-github-repo=ROCm/TheRock",
            "--stage=math-libs",
            "--amdgpu-families=gfx950",
            "--output-dir=build",
            "--bootstrap",
        ],
        log="fetch-artifacts.log",
    )
    run(
        [
            sys.executable,
            "build_tools/fetch_sources.py",
            "--stage",
            "math-libs",
            "--jobs",
            "12",
            "--depth",
            "1",
        ],
        log="fetch-sources.log",
    )
    save(
        "sources.json",
        {
            "submodules": run(["git", "submodule", "status", "--recursive"]),
            "libraries_head": run(
                ["git", "-C", "rocm-libraries", "rev-parse", "HEAD"]
            ).strip(),
            "libraries_tree": run(
                ["git", "-C", "rocm-libraries", "rev-parse", "HEAD^{tree}"]
            ).strip(),
            "libraries_diff": run(["git", "-C", "rocm-libraries", "diff"]),
        },
    )
    arg_file = OUT / "configure-args.txt"
    run(
        [
            sys.executable,
            "build_tools/configure_stage.py",
            "--stage=math-libs",
            "--artifacts=composable-kernel",
            "--amdgpu-families=gfx950",
            "--dist-amdgpu-families=gfx950",
            "--manylinux",
            f"--output-cmake-args={arg_file}",
        ]
    )
    preset = (
        "linux-release-asan-debug"
        if args.variant == "asan-debug"
        else "linux-release-package"
    )
    argv = [
        "cmake",
        "-B",
        str(BUILD),
        "-S",
        ".",
        "-GNinja",
        f"--preset={preset}",
        *arg_file.read_text().splitlines(),
        "-DTHEROCK_BACKGROUND_BUILD_JOBS=9",
        "-DLLVM_PARALLEL_LINK_JOBS=4",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-Dcomposable_kernel_CMAKE_ARGS=-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    save("configure-command.json", argv)
    run(argv, log="configure.log")
    run(
        ["cmake", "--build", str(BUILD), "--target", "composable_kernel+configure"],
        log="ck-configure.log",
    )
    for path in (BUILD / "CMakeCache.txt", CK / "CMakeCache.txt", CK / "build.ninja"):
        shutil.copy2(
            path, OUT / (("super-" if path.parent == BUILD else "ck-") + path.name)
        )
    markers = [str(p.relative_to(BUILD)) for p in BUILD.rglob("stage.prebuilt")]
    if any("composable_kernel" in p for p in markers):
        raise RuntimeError("CK unexpectedly imported as prebuilt")
    save("prebuilt-markers.json", markers)


def measure(argv: list[str], cwd: Path, name: str) -> None:
    """Fresh GNU time process plus sampled aggregate process-tree RSS."""
    import psutil

    save(name + "-command.json", {"argv": argv, "cwd": str(cwd)})
    started = time.monotonic()
    peak_tree_rss = 0
    samples = []
    with (OUT / (name + ".log")).open("w") as log:
        proc = subprocess.Popen(
            ["/usr/bin/time", "-v", "-o", str(OUT / (name + "-time.txt")), *argv],
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while proc.poll() is None:
            rss = 0
            try:
                children = psutil.Process(proc.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            for child in children:
                try:
                    rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak_tree_rss = max(peak_tree_rss, rss)
            samples.append({"elapsed_s": time.monotonic() - started, "rss_bytes": rss})
            time.sleep(1)
        code = proc.wait()
    save(
        name + "-measurement.json",
        {
            "elapsed_s": time.monotonic() - started,
            "exit_code": code,
            "peak_sampled_process_tree_rss_bytes": peak_tree_rss,
            "sampling_interval_s": 1,
            "samples": samples,
            "limits": "Sampled RSS can double-count shared pages and miss short peaks; GNU time records child CPU and max RSS.",
        },
    )
    if code:
        raise RuntimeError(f"{name} failed with exit code {code}")


def replay() -> None:
    database = json.loads(run(["ninja", "-t", "compdb", "-x"], cwd=CK))
    matches = [
        entry for entry in database if Path(entry.get("file", "")).name == SOURCE
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one command for {SOURCE}, got {len(matches)}")
    entry = matches[0]
    save("original-compdb-entry.json", entry)
    argv = shlex.split(entry["command"])
    if any(token in {"&&", "||", ";", "|"} for token in argv):
        raise RuntimeError("Expected direct compiler command, not shell pipeline")
    recorded = json.loads(
        (Path(__file__).parent / "baseline-argv.json").read_text(encoding="utf-8-sig")
    )

    def normalize(command: list[str]) -> list[str]:
        if Path(command[0]).name == "ccache":
            command = command[1:]
        return [
            value.replace(str(ROOT), "<SOURCE_ROOT>").replace(
                "/__w/TheRock/TheRock", "<SOURCE_ROOT>"
            )
            for value in command
        ]

    old, new = normalize(recorded), normalize(argv)
    save(
        "historical-command-comparison.json",
        {
            "historical_baseline_run": "41211496307",
            "historical_variant": "asan-debug",
            "normalized_equal": old == new,
            "argument_diff": list(
                difflib.unified_diff(old, new, fromfile="historical", tofile="replay")
            ),
            "normalization": "Only workspace root and disabled ccache launcher removed.",
        },
    )
    cwd = Path(entry["directory"])
    target = entry["output"]
    (OUT / "target-graph.dot").write_text(run(["ninja", "-t", "graph", target], cwd=CK))
    (OUT / "target-query.txt").write_text(run(["ninja", "-t", "query", target], cwd=CK))
    (OUT / "target-commands.txt").write_text(
        run(["ninja", "-t", "commands", target], cwd=CK)
    )
    compiler = Path(argv[0])
    save(
        "compiler.json",
        {**fingerprint(compiler), "version": run([str(compiler), "--version"])},
    )
    source_path = Path(entry["file"])
    if not source_path.is_absolute():
        source_path = cwd / source_path
    save("source-file.json", fingerprint(source_path))
    for name, extra in (
        ("uninstrumented", []),
        ("profile", ["-ftime-trace", "-ftime-report"]),
    ):
        command = list(argv)
        for flag, extension in (("-o", ".o"), ("-MF", ".d")):
            if flag in command:
                command[command.index(flag) + 1] = str(OUT / (name + extension))
        command += extra
        measure(command, cwd, name)
    save(
        "result.json",
        {"status": "diagnostic replay completed", "optimization_claim": False},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["configure", "replay"])
    parser.add_argument("--baseline-run-id", default=os.getenv("BASELINE_RUN_ID"))
    parser.add_argument(
        "--variant",
        choices=["asan-debug", "release"],
        default=os.getenv("BUILD_VARIANT", "asan-debug"),
    )
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    os.environ["CCACHE_DISABLE"] = "1"
    if args.action == "configure":
        if not args.baseline_run_id or not args.baseline_run_id.isdigit():
            parser.error("A numeric baseline run ID is required")
        configure(args)
    else:
        replay()


if __name__ == "__main__":
    main()
