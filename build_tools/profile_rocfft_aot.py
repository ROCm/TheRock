#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run an opt-in, all-seven-architecture rocFFT AOT diagnostic in TheRock CI."""
import argparse
import collections
import datetime
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "5b7cb88373362ab80dc0f19605915e534f4158aa"
REFERENCE_HEAD = "82d5e8c176ce0cc00b5fd1cf631ce7790be8992b"
LIBRARIES = "f4e24d58b3f313f1a31068def1e35376e43c2f49"
SDK_RUN = "41190705863"
VERSION = "10.3.0.dev0+82d5e8c176ce0cc00b5fd1cf631ce7790be8992b"
FAMILY = "gfx103X-all"
ARCHES = [f"gfx103{i}" for i in range(7)]
PATCH_FILES = {
    "projects/rocfft/library/src/rocfft_aot_helper.cpp",
    "projects/rocfft/library/src/rtc_cache.cpp",
    "projects/rocfft/shared/work_queue.h",
    "projects/rocfft/shared/subprocess.h",
}
BATCH_PATCH_FILES = {
    "projects/rocfft/library/src/rocfft_aot_helper.cpp",
    "projects/rocfft/library/src/rtc_cache.cpp",
    "projects/rocfft/library/src/include/rtc_cache.h",
}
KNOBS = [
    "OMP_NUM_THREADS",
    "ROCFFT_RTC_MAX_SUBPROCESSES",
    "ROCFFT_RTC_PROCESS",
    "ROCFFT_RTC_CACHE_READ_DISABLE",
    "ROCFFT_RTC_CACHE_WRITE_DISABLE",
    "ROCFFT_LAYER",
    "ROCFFT_AOT_BATCH_TRANSACTION",
]


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_ci(env: dict[str, str]) -> None:
    if (
        env.get("GITHUB_ACTIONS") != "true"
        or env.get("GITHUB_REPOSITORY") != "ROCm/TheRock"
    ):
        raise ValueError("This diagnostic may run only in ROCm/TheRock CI")
    if sys.platform != "linux":
        raise ValueError("The AOT diagnostic is Linux-only")
    if env.get("RELEASE_TYPE") != "ci" or env.get("CCACHE_READONLY") != "1":
        raise ValueError("CI bucket and read-only remote compiler cache are required")
    if env.get("ROCFFT_LAYER"):
        raise ValueError("Source-dumping ROCFFT logging must not be enabled")


def capture(command: list[str], cwd: Path, env: dict[str, str]) -> str:
    return subprocess.check_output(command, cwd=cwd, env=env, text=True)


def run_logged(command: list[str], log: Path, cwd: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{utc()} START {shlex.join(command)}", flush=True)
    with log.open("w", encoding="utf-8") as output:
        output.write(shlex.join(command) + "\n")
        output.flush()
        proc = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT
        )
        while True:
            try:
                code = proc.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                print(f"{utc()} RUNNING {log.name}", flush=True)
    print(f"{utc()} END {log.name} exit={code}", flush=True)
    if code:
        print("\n".join(log.read_text(errors="replace").splitlines()[-60:]), flush=True)
        raise subprocess.CalledProcessError(code, command)


def cache_values(path: Path) -> dict[str, str]:
    return {
        m[1]: m[2]
        for line in path.read_text().splitlines()
        if (m := re.match(r"([^/#][^:]*):[^=]+=(.*)", line))
    }


def patch_paths(text: str, expected: set[str] = PATCH_FILES) -> set[str]:
    paths = re.findall(r"^diff --git a/(\S+) b/\1$", text, re.MULTILINE)
    if set(paths) != expected or len(paths) != len(expected):
        raise ValueError(
            "Instrumentation patch must touch only the reviewed host-side files"
        )
    return set(paths)


def explicit_read(path: Path) -> dict:
    try:
        return {"value": path.read_text().strip()}
    except OSError as error:
        return {"error": type(error).__name__, "errno": error.errno}


def thread_wait_channels(task_root: Path, limit: int = 4096) -> dict:
    counts = collections.Counter()
    errors = collections.Counter()
    try:
        tasks = sorted(task_root.iterdir(), key=lambda p: p.name)
    except OSError as error:
        return {"error": type(error).__name__, "errno": error.errno}
    for task in tasks[:limit]:
        result = explicit_read(task / "wchan")
        if "value" in result:
            counts[result["value"] or "<empty>"] += 1
        else:
            errors[f"{result['error']}:{result['errno']}"] += 1
    return {
        "wait_channels": dict(counts),
        "read_errors": dict(errors),
        "visible_task_count": len(tasks),
        "sampled_task_count": min(len(tasks), limit),
        "truncated": len(tasks) > limit,
        "note": "0/empty can mean running or inaccessible symbols; not a stack trace.",
    }


class WaitObserver:
    """Read-only corroboration: 1s visible pids counters, 5s exact-helper wchan."""

    def __init__(self, helper: Path, output: Path):
        self.helper, self.output = helper.resolve(), output
        if self.helper.name != "rocfft_aot_helper":
            raise ValueError("Wait-channel observation is restricted to the AOT helper")
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="rocfft-wait-observer")
        self.summary = {}

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        write_json(self.output / "observer-overhead.json", self.summary)

    def _run(self):
        import psutil
        from profile_build import cgroups

        paths, info = cgroups()
        write_json(self.output / "observer-cgroup-scope.json", info)
        count, active_wall, started = 0, 0.0, time.monotonic()
        cpu_start = time.thread_time()
        try:
            with (self.output / "external-waits.jsonl").open("w") as stream:
                while not self.stop.is_set():
                    active_start = time.monotonic()
                    row = dict(
                        utc=utc(),
                        relative_s=active_start - started,
                        pids_cgroups={
                            str(p): {
                                name: explicit_read(p / name)
                                for name in ["pids.max", "pids.current", "pids.events"]
                            }
                            for p in paths
                        },
                    )
                    if count % 5 == 0:
                        row["helpers"] = []
                        row["process_discovery_errors"] = []
                        try:
                            processes = psutil.Process().children(recursive=True)
                        except (psutil.Error, OSError) as error:
                            row["process_discovery_errors"].append(type(error).__name__)
                            processes = []
                        for process in processes:
                            try:
                                if Path(process.exe()).resolve() != self.helper:
                                    continue
                                row["helpers"].append(
                                    dict(
                                        pid=process.pid,
                                        create_time=process.create_time(),
                                        **thread_wait_channels(
                                            Path(f"/proc/{process.pid}/task")
                                        ),
                                    )
                                )
                            except (psutil.Error, OSError) as error:
                                row["process_discovery_errors"].append(
                                    type(error).__name__
                                )
                    row["observer_cpu_s"] = time.thread_time() - cpu_start
                    # A row cannot include the cost of its own serialization/flush.
                    # Report completed samples here; the final summary includes all.
                    row["observer_active_wall_s"] = active_wall
                    row["observer_active_wall_scope"] = (
                        "completed_previous_samples_including_write_flush"
                    )
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    active_wall += time.monotonic() - active_start
                    count += 1
                    self.stop.wait(1)
        except Exception as error:
            self.summary["error"] = type(error).__name__
        finally:
            self.summary.update(
                samples=count,
                thread_cpu_s=time.thread_time() - cpu_start,
                active_wall_s=active_wall,
                active_wall_scope="completed_sample_bodies_including_write_flush_excluding_wait",
                lifetime_wall_s=time.monotonic() - started,
                limitations="Observer is read-only, namespace-scoped and misses waits between snapshots.",
            )


def generator_inputs(source: Path) -> list[dict[str, str]]:
    cmake = (source / "library/src/CMakeLists.txt").read_text()
    paths = {"shared/embed-to-header-rtc.py"}
    for variable in ["kgen_embed_files", "kgen_logic_files"]:
        match = re.search(r"set\(\s*" + variable + r"\s+(.*?)\n\)", cmake, re.DOTALL)
        if not match:
            raise ValueError(
                f"Cannot establish the generator checksum inputs: {variable}"
            )
        paths.update(re.findall(r"\$\{CMAKE_SOURCE_DIR\}/([^\s)]+)", match[1]))
    return [{"path": p, "sha256": digest(source / p)} for p in sorted(paths)]


def install_instrumentation(libraries: Path, output: Path, env: dict[str, str]) -> None:
    patch = (
        ROOT
        / "patches/build-profile/rocm-libraries/0001-rocfft-aot-phase-diagnostics.patch"
    )
    batch_patch = (
        ROOT
        / "patches/build-profile/rocm-libraries/0002-rocfft-private-aot-transaction.patch"
    )
    batch_header = ROOT / "build_tools/diagnostics/rocfft_aot_transaction.h"
    patch_paths(batch_patch.read_text(), BATCH_PATCH_FILES)
    header = ROOT / "build_tools/diagnostics/rocfft_profile.h"
    patch_paths(patch.read_text())
    source = libraries / "projects/rocfft"
    before = generator_inputs(source)
    destination = source / "shared/build_profile.h"
    if destination.exists():
        raise ValueError("Unexpected preexisting profiling header")
    capture(["git", "diff", "--exit-code", "HEAD"], libraries, env)
    capture(["git", "apply", "--check", str(patch)], libraries, env)
    capture(["git", "apply", str(patch)], libraries, env)
    shutil.copyfile(header, destination)
    capture(["git", "apply", "--check", str(batch_patch)], libraries, env)
    capture(["git", "apply", str(batch_patch)], libraries, env)
    shutil.copyfile(batch_header, source / "shared/aot_transaction.h")
    after = generator_inputs(source)
    if before != after:
        raise ValueError("Instrumentation changed a generator_sum checksum input")
    changed = set(capture(["git", "diff", "--name-only"], libraries, env).splitlines())
    if changed != PATCH_FILES | BATCH_PATCH_FILES:
        raise ValueError(f"Unexpected source edits: {changed}")
    shutil.copyfile(patch, output / patch.name)
    shutil.copyfile(batch_patch, output / batch_patch.name)
    shutil.copyfile(batch_header, output / batch_header.name)
    shutil.copyfile(header, output / header.name)
    (output / "applied-source.diff").write_text(
        capture(["git", "diff", "--binary"], libraries, env)
    )
    write_json(
        output / "instrumentation.json",
        {
            "libraries_base_sha": capture(
                ["git", "rev-parse", "HEAD"], libraries, env
            ).strip(),
            "patch_sha256": digest(patch),
            "header_sha256": digest(header),
            "transaction_patch_sha256": digest(batch_patch),
            "transaction_header_sha256": digest(batch_header),
            "generator_inputs_unchanged": True,
            "generator_inputs": before,
            "note": "git apply only; source HEAD and .smrev are not rewritten",
        },
    )


def save_tree(build: Path, output: Path, prefix: str, env: dict[str, str]) -> None:
    for name in [
        "CMakeCache.txt",
        "build.ninja",
        ".ninja_log",
        "CMakeFiles/rules.ninja",
    ]:
        path = build / name
        if path.is_file():
            dest = output / prefix / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    for path in (build / "CMakeFiles").glob("*/*Compiler.cmake"):
        dest = output / prefix / path.relative_to(build)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    for tool, args in [("commands", ["-t", "commands"]), ("graph", ["-t", "graph"])]:
        (output / prefix / f"ninja-{tool}.txt").write_text(
            capture(["ninja", "-C", str(build), *args], ROOT, env)
        )


def inventory_database(path: Path, destination: Path) -> dict:
    """Read-only logical checksum excludes SQLite page layout and insertion order."""
    started = time.monotonic()
    semantic = hashlib.sha256()
    count, code_bytes = 0, 0
    generator_sums, arches, hip_versions = set(), set(), set()
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        integrity = db.execute("PRAGMA quick_check").fetchall()
        with destination.open("w", encoding="utf-8") as output:
            for kernel, arch, version, generator, code, timestamp in db.execute(
                "SELECT kernel_name,arch,hip_version,generator_sum,code,timestamp "
                "FROM cache_v1 ORDER BY kernel_name,arch,hip_version,generator_sum"
            ):
                record = dict(
                    kernel_name=kernel,
                    arch=arch,
                    hip_version=version,
                    generator_sum=generator.hex(),
                    code_bytes=len(code),
                    code_sha256=hashlib.sha256(code).hexdigest(),
                    timestamp=timestamp,
                )
                encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
                output.write(encoded + "\n")
                semantic.update((encoded + "\n").encode())
                count += 1
                code_bytes += len(code)
                generator_sums.add(generator.hex())
                arches.add(arch)
                hip_versions.add(version)
    return dict(
        path=str(path),
        bytes=path.stat().st_size,
        file_sha256=digest(path),
        semantic_sha256=semantic.hexdigest(),
        entries=count,
        code_bytes=code_bytes,
        generator_sums=sorted(generator_sums),
        arches=sorted(arches),
        hip_versions=sorted(hip_versions),
        quick_check=integrity,
        inventory_wall_s=time.monotonic() - started,
    )


def prepare(build: Path, output: Path, env: dict[str, str]) -> Path:
    for name, command in [
        (
            "fetch-sdk",
            [
                sys.executable,
                "build_tools/artifact_manager.py",
                "fetch",
                "--run-id",
                SDK_RUN,
                "--run-github-repo",
                "ROCm/TheRock",
                "--stage",
                "math-libs",
                "--amdgpu-families",
                FAMILY,
                "--output-dir",
                str(build),
                "--bootstrap",
            ],
        ),
        (
            "fetch-sources",
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
        ),
    ]:
        run_logged(command, output / f"{name}.log", ROOT, env)
    capture(["git", "merge-base", "--is-ancestor", REFERENCE_HEAD, "HEAD"], ROOT, env)
    libraries = ROOT / "rocm-libraries"
    if capture(["git", "rev-parse", "HEAD"], libraries, env).strip() != LIBRARIES:
        raise ValueError("Unexpected rocm-libraries source pin")
    install_instrumentation(libraries, output, env)
    from configure_stage import get_topology

    requirements = get_topology().get_python_requires_for_stage("math-libs")
    if requirements:
        run_logged(
            [sys.executable, "-m", "pip", "install", *requirements],
            output / "python-deps.log",
            ROOT,
            env,
        )
    stage_command = [
        sys.executable,
        "build_tools/configure_stage.py",
        "--stage",
        "math-libs",
        "--amdgpu-families",
        FAMILY,
        "--dist-amdgpu-families",
        FAMILY,
        "--manylinux",
        "--oneline",
    ]
    args = capture(stage_command, ROOT, env)
    (output / "stage-cmake-args.txt").write_text(args)
    if "-DTHEROCK_ENABLE_FFT=ON" not in args or "-DTHEROCK_ENABLE_ALL=OFF" not in args:
        raise ValueError("The full math stage must retain FFT")
    run_logged(
        [
            "cmake",
            "-B",
            str(build),
            "-S",
            str(ROOT),
            "-GNinja",
            f"-DTHEROCK_PACKAGE_VERSION={VERSION}",
            f"-DTHEROCK_ROCJITSU_TEST_JOBS={min(16, int(capture(['nproc'], ROOT, env).strip()) // 2)}",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache",
            *shlex.split(args),
        ],
        output / "configure.log",
        ROOT,
        env,
    )
    if cache_values(build / "CMakeCache.txt").get("THEROCK_PACKAGE_VERSION") != VERSION:
        raise ValueError("Package-version drift")
    save_tree(build, output, "root-before", env)
    return build / "math-libs/rocFFT/build"


def resolve_compiler(nested: Path) -> Path:
    # Toolchain files may set this as a normal variable, not a cache entry.
    candidates = set()
    value = cache_values(nested / "CMakeCache.txt").get("CMAKE_CXX_COMPILER")
    if value:
        candidates.add(Path(value).resolve())
    for metadata in (nested / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"):
        match = re.search(
            r'set\(CMAKE_CXX_COMPILER "([^"\n]+)"\)', metadata.read_text()
        )
        if match:
            candidates.add(Path(match[1]).resolve())
    if len(candidates) != 1:
        raise ValueError(f"Expected one actual generated C++ compiler: {candidates}")
    compiler = candidates.pop()
    if not compiler.is_file():
        raise ValueError(f"Generated C++ compiler does not exist: {compiler}")
    return compiler


def record_sdk_identity(nested: Path, output: Path, env: dict[str, str]) -> None:
    compiler = resolve_compiler(nested)
    write_json(
        output / "sdk-identity.json",
        {
            "repository": "ROCm/TheRock",
            "run_id": SDK_RUN,
            "package_version": VERSION,
            "compiler": str(compiler),
            "compiler_sha256": digest(compiler),
            "compiler_version": capture([str(compiler), "--version"], ROOT, env),
            "libraries_smrev": (
                (ROOT / ".rocm-libraries.smrev").read_text()
                if (ROOT / ".rocm-libraries.smrev").is_file()
                else None
            ),
        },
    )


def run_postchecks(checks: list[tuple[str, object]], output: Path) -> None:
    """Retain independent evidence even when an earlier metadata check fails."""
    results = []
    for name, check in checks:
        started = time.monotonic()
        try:
            check()
            result = {"name": name, "result": "passed"}
        except Exception as error:
            result = {"name": name, "result": "failed", "error": repr(error)}
        result["wall_s"] = time.monotonic() - started
        results.append(result)
        write_json(output / "postchecks.json", results)
    failed = [r["name"] for r in results if r["result"] == "failed"]
    if failed:
        raise RuntimeError(f"Required postchecks failed: {failed}")


def verify_rocfft(nested: Path, output: Path, env: dict[str, str]) -> None:
    values = cache_values(nested / "CMakeCache.txt")
    if values.get("ROCFFT_KERNEL_CACHE_ENABLE") != "ON":
        raise ValueError("Required shipped AOT caches are disabled")
    if values.get("GPU_TARGETS", "").split(";") != ARCHES:
        raise ValueError(
            f"Unexpected target architecture list: {values.get('GPU_TARGETS')}"
        )
    commands = {}
    for arch in ARCHES:
        target = f"library/src/rocfft_kernel_cache_{arch}.db"
        command = capture(
            ["ninja", "-C", str(nested), "-t", "commands", "-s", target], ROOT, env
        ).strip()
        if (
            "rocfft_aot_helper" not in command
            or "rocfft_rtc_helper" not in command
            or arch not in command
        ):
            raise ValueError(f"Missing original AOT helper edge: {arch}")
        commands[arch] = command
    write_json(output / "original-helper-commands.json", commands)
    save_tree(nested, output, "rocfft-generated", env)


def collect_cache_outputs(
    nested: Path, output: Path, batch_first: bool = False
) -> None:
    inventories = []
    for arch in ARCHES:
        path = nested / f"library/src/rocfft_kernel_cache_{arch}.db"
        if not path.is_file():
            raise ValueError(f"Missing required shipped cache: {arch}")
        info = inventory_database(path, output / f"inventory-{arch}.jsonl")
        if (
            info["arches"] != [arch]
            or info["quick_check"] != [("ok",)]
            or info["entries"] != 17204
        ):
            raise ValueError(f"Invalid cache output: {arch}")
        inventories.append(info)
    write_json(output / "cache-inventories.json", inventories)
    traces = list((output / "phases").glob("*.jsonl"))
    completed = []
    generator_sums = set()
    for path in traces:
        last = json.loads(path.read_text().splitlines()[-1])
        if last["event"] != "complete":
            raise ValueError(f"Incomplete phase trace: {path.name}")
        if (
            last["counters"]["cache_misses"] != 17204
            or last["counters"]["store_errors"]
        ):
            raise ValueError(
                f"Changed compilation coverage or failed cache stores: {path.name}"
            )
        expected_batch = "1" if batch_first and last["arch"] == ARCHES[0] else "0"
        if last["properties"].get("aot_batch_transaction") != expected_batch:
            raise ValueError(
                f"Unexpected first/later helper transaction policy: {path.name}"
            )
        completed.append(last["arch"])
        generator_sums.add(last["properties"].get("generator_sum"))
    if sorted(completed) != ARCHES:
        raise ValueError(
            f"Must retain one completed helper trace for each architecture: {completed}"
        )
    if len(generator_sums) != 1 or None in generator_sums:
        raise ValueError(
            "Missing or inconsistent generator_sum across architecture helpers"
        )
    if any(set(info["generator_sums"]) != generator_sums for info in inventories):
        raise ValueError(
            "Shipped cache generator_sum disagrees with the helper's recorded checksum"
        )
    write_json(
        output / "helper-binaries.json",
        [
            {"path": str(path), "sha256": digest(path)}
            for path in [
                nested / "library/src/rocfft_aot_helper",
                nested / "library/src/rocfft_rtc_helper",
            ]
        ],
    )


def require_fresh_private_cache(nested: Path) -> None:
    cache = nested / "library/src/rocfft_temp_cache.db"
    if cache.exists() or cache.is_symlink():
        raise ValueError("AOT trial requires a fresh, helper-owned temporary database")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output", type=Path, default=ROOT / "build/logs/rocfft-aot")
    parser.add_argument(
        "--batch-private-aot",
        action="store_true",
        help="Opt-in first-fresh-helper transaction discriminator; later/runtime caches unchanged",
    )
    args = parser.parse_args()
    env = dict(os.environ)
    require_ci(env)
    import resource

    limits = {}
    for name in ["RLIMIT_NPROC", "RLIMIT_NOFILE"]:
        try:
            limits[name] = {
                "soft_hard": list(resource.getrlimit(getattr(resource, name)))
            }
        except (OSError, ValueError) as error:
            limits[name] = {
                "error": type(error).__name__,
                "errno": getattr(error, "errno", None),
            }
    # Scope Git's container-ownership exception to this command's descendants.
    n = int(env.get("GIT_CONFIG_COUNT", "0"))
    env.update(
        {
            f"GIT_CONFIG_KEY_{n}": "safe.directory",
            f"GIT_CONFIG_VALUE_{n}": str(ROOT),
            "GIT_CONFIG_COUNT": str(n + 1),
        }
    )
    output, build = args.output.resolve(), args.build_dir.resolve()
    if not build.is_relative_to(ROOT) or not output.is_relative_to(build):
        raise ValueError(
            "Diagnostic output must stay under this checkout's build directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        started_utc=utc(),
        source_head=capture(["git", "rev-parse", "HEAD"], ROOT, env).strip(),
        product_head=BASELINE,
        unmodified_reference_head=REFERENCE_HEAD,
        unmodified_reference_run=SDK_RUN,
        libraries_pin=LIBRARIES,
        sdk_repository="ROCm/TheRock",
        sdk_run=SDK_RUN,
        package_version=VERSION,
        runner_label="aws-linux-scale-rocm-prod",
        runner_name=env.get("RUNNER_NAME"),
        run_id=env.get("GITHUB_RUN_ID"),
        original_rtc_knobs={name: env.get(name) for name in KNOBS},
        aot_transaction_mode="batch" if args.batch_private_aot else "autocommit",
        initial_resource_limits=limits,
        uid=os.getuid(),
        euid=os.geteuid(),
        dataset="Fresh allocation/build; default persistent temporary-cache path. Initial sizes recorded per helper.",
        limitations=[
            "Diagnostic only, no speedup comparison.",
            "Overlapping timers are not additive elapsed time.",
            "Full math-stage context retained; source instrumentation touches rocFFT host files only.",
        ],
    )
    write_json(output / "manifest.json", manifest)
    nested = None
    try:
        nested = prepare(build, output, env)
        require_fresh_private_cache(nested)
        # Worker/process/compilation policy stays fixed; transaction trial is explicit.
        child_env = dict(env)
        child_env["ROCFFT_BUILD_PROFILE_DIR"] = str(output / "phases")
        # Override inherited trial settings: the CLI is the recorded control.
        child_env["ROCFFT_AOT_BATCH_TRANSACTION"] = (
            "1" if args.batch_private_aot else "0"
        )
        for name in [
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        ]:
            child_env.pop(name, None)
        with WaitObserver(nested / "library/src/rocfft_aot_helper", output):
            run_logged(
                [
                    sys.executable,
                    "build_tools/profile_build.py",
                    "--output",
                    str(output / "build-profile"),
                    "--",
                    "cmake",
                    "--build",
                    str(build),
                    "--target",
                    "stage-math-libs",
                    "therock-artifacts",
                    "--",
                    "-k",
                    "0",
                ],
                output / "rocfft-build.log",
                ROOT,
                child_env,
            )
        run_postchecks(
            [
                ("sdk_identity", lambda: record_sdk_identity(nested, output, env)),
                (
                    "original_helper_commands",
                    lambda: verify_rocfft(nested, output, env),
                ),
                (
                    "semantic_cache_outputs",
                    lambda: collect_cache_outputs(
                        nested, output, args.batch_private_aot
                    ),
                ),
                (
                    "packaging_tests",
                    lambda: run_logged(
                        ["ctest", "--test-dir", str(build), "--output-on-failure"],
                        output / "packaging-tests.log",
                        ROOT,
                        env,
                    ),
                ),
            ],
            output,
        )
        manifest["result"] = (
            "diagnostic_completed; packaging tests passed; optimization and GPU runtime correctness unvalidated"
        )
    finally:
        if nested and (nested / "build.ninja").is_file():
            save_tree(nested, output, "rocfft-after", env)
        if (build / "build.ninja").is_file():
            save_tree(build, output, "root-after", env)
        for pattern in ["*_build.log", "*_configure.log", "*_install.log"]:
            for path in (build / "logs").glob(pattern):
                if path.is_file():
                    shutil.copyfile(path, output / path.name)
        for path in build.rglob(".ninja_log"):
            if not path.is_relative_to(output):
                dest = output / "all-ninja-logs" / path.relative_to(build)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
        for name in ["rocFFT_build.log", "rocFFT_configure.log"]:
            source = build / "logs" / name
            if source.is_file():
                shutil.copyfile(source, output / name)
        manifest["finished_utc"] = utc()
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
