#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Controlled, single-object HIP offload-job trials on TheRock CI only.

Bootstrap the existing compiler/runtime, configure hipCCL_tests and dependencies,
then compare jobs=1/2 sequentially within one allocation. No source patch,
release publication, or S3 upload is performed. Diagnostics go to GitHub artifacts.
"""

import argparse
import datetime
import filecmp
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
SOURCE_BASELINE = "5b7cb88373362ab80dc0f19605915e534f4158aa"
LIBRARIES_PIN = "f4e24d58b3f313f1a31068def1e35376e43c2f49"
SDK_RUN = "41185762902"
PACKAGE_VERSION = "10.3.0.dev0+5e7281668afc1c2c326f7ae0c73043a117ed8b53"
FAMILY = "gfx103X-all"
ARCHES = {f"gfx103{i}" for i in range(7)}
OBJECT = (
    "rocprim/benchmark/CMakeFiles/benchmark_device_segmented_radix_sort_pairs.dir/"
    "benchmark_device_segmented_radix_sort_pairs.cpp.o"
)
BENCHMARK = "rocprim/benchmark/benchmark_rocprim_device_segmented_radix_sort_pairs"
DISABLING_ENV = {
    "CC_PRINT_PROC_STAT",
    "CC_PRINT_PROC_STAT_FILE",
    "CC_PRINT_OPTIONS",
    "CC_PRINT_OPTIONS_FILE",
    "CCC_OVERRIDE_OPTIONS",
    "CLANG_CRASH_DIAGNOSTICS_DIR",
    "CCACHE_PREFIX",
    "CCACHE_PREFIX_CPP",
    "CCACHE_RECACHE",
}


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require_ci(env: dict[str, str]) -> None:
    if (
        env.get("GITHUB_ACTIONS") != "true"
        or env.get("GITHUB_REPOSITORY") != "ROCm/TheRock"
    ):
        raise RuntimeError("This trial executes exclusively in ROCm/TheRock Actions")


def trial_env(env: dict[str, str]) -> dict[str, str]:
    result = {k: v for k, v in env.items() if k not in DISABLING_ENV}
    result["CCACHE_DISABLE"] = "1"
    # No network credentials are needed by a compiler, linker, or sampler.
    for key in (
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        result.pop(key, None)
    return result


def parse_order(raw: str) -> list[int]:
    value = json.loads(raw)
    if value not in ([1, 2], [2, 1]) or any(type(n) is not int for n in value):
        raise ValueError("Only balanced paired orders [1,2] and [2,1] are supported")
    return value


def capture(command: list[str], cwd: Path, env: dict[str, str]) -> str:
    return subprocess.check_output(
        command, cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT
    )


def run_logged(command: list[str], log: Path, cwd: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        log.with_suffix(log.suffix + ".command.json"),
        {"argv": command, "cwd": str(cwd)},
    )
    print(f"{utc()} START {log.name}", flush=True)
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
        while True:
            try:
                code = process.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                print(
                    f"{utc()} ACTIVE {log.name} elapsed={time.monotonic() - start:.0f}s",
                    flush=True,
                )
    print(f"{utc()} END {log.name} exit={code}", flush=True)
    if code:
        raise subprocess.CalledProcessError(code, command)


def object_edge(text: str, target: str = OBJECT) -> tuple[list[str], int, int]:
    """Locate one exact generated object edge, never a similarly named target."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith(f"build {target}:")]
    if len(starts) != 1:
        raise ValueError(f"Expected one exact object edge, found {len(starts)}")
    start = starts[0]
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0].isspace()):
        end += 1
    return lines, start, end


def patch_flags(text: str, jobs: int, target: str = OBJECT) -> str:
    if jobs not in (1, 2):
        raise ValueError("Only the controlled jobs=1/2 comparison is supported")
    lines, start, end = object_edge(text, target)
    matches = [i for i in range(start + 1, end) if re.match(r" +FLAGS = ", lines[i])]
    if len(matches) != 1:
        raise ValueError("Object edge must have one explicit FLAGS binding")
    i = matches[0]
    if lines[i].rstrip("\r\n").endswith("$"):
        raise ValueError(
            "Unexpected continued FLAGS binding; refusing broad modification"
        )
    if re.search(r"(?:offload-jobs|parallel-jobs)=", lines[i]):
        raise ValueError("Original edge already contains an offload jobs override")
    newline = "\r\n" if lines[i].endswith("\r\n") else "\n"
    lines[i] = lines[i].rstrip("\r\n") + f" --offload-jobs={jobs}" + newline
    return "".join(lines)


def generated_path(build: Path, relative: str) -> Path:
    root = build.resolve()
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("Generated output must be a relative path inside this build")
    path = root / raw
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Generated output escaped its build directory")
    return path


def force_object_rebuild(build: Path, target: str = OBJECT) -> list[str]:
    """Remove only the one generated object and its compiler dependency file."""
    removed = []
    for relative in (target, target + ".d"):
        path = generated_path(build, relative)
        if path.exists():
            if not path.is_file():
                raise ValueError("Expected a generated regular file")
            path.unlink()
            removed.append(relative)
    return removed


def compiler_arguments(command: str, jobs: int) -> tuple[Path, list[str]]:
    tokens = shlex.split(command)
    if any(token in ("&&", ";", "|", "||") for token in tokens):
        raise ValueError("Unexpected shell compound command for the compiler edge")
    indices = [
        i
        for i, token in enumerate(tokens)
        if Path(token).name in ("clang++", "clang++.exe")
    ]
    if len(indices) != 1:
        raise ValueError("Expected one direct clang++ invocation")
    args = tokens[indices[0] + 1 :]
    forbidden = [
        a
        for a in args
        if a
        in (
            "-v",
            "-###",
            "-fdriver-only",
            "-ccc-print-phases",
            "-ccc-print-bindings",
            "--driver-mode=cl",
        )
        or a.startswith("-fproc-stat-report")
    ]
    if forbidden:
        raise ValueError(
            f"Profiling/driver option disables parallel cc1 jobs: {forbidden}"
        )
    observed = {a.split("=", 1)[1] for a in args if a.startswith("--offload-arch=")}
    if observed != ARCHES:
        raise ValueError(f"The original seven architectures must remain: {observed}")
    if [a for a in args if a.startswith("--offload-jobs=")] != [
        f"--offload-jobs={jobs}"
    ]:
        raise ValueError("Expected exactly one controlled offload-jobs option")
    if any("parallel-jobs=" in a for a in args):
        raise ValueError("The legacy parallel-jobs spelling is not this experiment")
    if "-c" not in args or not any(
        a.endswith("benchmark_device_segmented_radix_sort_pairs.cpp") for a in args
    ):
        raise ValueError("Wrong source or non-compile edge")
    return Path(tokens[indices[0]]), args


def single_command(ninja: str, build: Path, target: str, env: dict[str, str]) -> str:
    value = capture(
        [ninja, "-C", str(build), "-t", "commands", "-s", target], ROOT, env
    )
    lines = [line for line in value.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected one native command for {target}, got {len(lines)}")
    return lines[0]


def last_object_record(build: Path, target: str = OBJECT) -> str | None:
    log = build / ".ninja_log"
    if not log.exists():
        return None
    rows = [
        line
        for line in log.read_text().splitlines()
        if len(line.split("\t")) >= 5 and line.split("\t")[3] == target
    ]
    return rows[-1] if rows else None


def snapshot(build: Path, output: Path, name: str) -> None:
    for source, suffix in (
        ("build.ninja", "build.ninja"),
        ("CMakeFiles/rules.ninja", "rules.ninja"),
        (".ninja_log", "ninja.log"),
        ("CMakeCache.txt", "CMakeCache.txt"),
    ):
        path = build / source
        if path.exists():
            shutil.copyfile(path, output / f"{name}-{suffix}")


def capability_probe(compiler: Path, output: Path, env: dict[str, str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    source = output / "empty.hip"
    source.write_text(
        "// Strict empty HIP driver capability probe.\n", encoding="utf-8"
    )
    args = [
        str(compiler),
        "-x",
        "hip",
        "-nogpuinc",
        "-nogpulib",
        "-nohipwrapperinc",
        "-Werror=unused-command-line-argument",
    ]
    args.extend(f"--offload-arch={arch}" for arch in sorted(ARCHES))
    args.extend(["--offload-jobs=2", "-c", str(source), "-o", str(output / "probe.o")])
    run_logged(args, output / "actual-probe.log", ROOT, env)
    plan = capture([*args, "-###"], ROOT, env)
    (output / "rendered-probe.txt").write_text(plan, encoding="utf-8")
    if "--wrapper-jobs=2" not in plan:
        raise ValueError("Actual binary did not forward the offload concurrency option")
    device_arches = set(re.findall(r'-target-cpu["\s]+(gfx[0-9a-f]+)', plan))
    if device_arches != ARCHES:
        raise ValueError(f"Probe plan lost architecture jobs: {device_arches}")
    (output / "compiler-version.txt").write_text(
        capture([str(compiler), "--version"], ROOT, env), encoding="utf-8"
    )


def cache_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        match = re.match(r"([^/#][^:]*):[^=]+=(.*)", line)
        if match:
            values[match[1]] = match[2]
    return values


def prepare(build: Path, output: Path, env: dict[str, str]) -> Path:
    for name, command in (
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
    ):
        run_logged(command, output / f"{name}.log", ROOT, env)
    capture(["git", "merge-base", "--is-ancestor", SOURCE_BASELINE, "HEAD"], ROOT, env)
    actual_pin = capture(
        ["git", "rev-parse", "HEAD"], ROOT / "rocm-libraries", env
    ).strip()
    if actual_pin != LIBRARIES_PIN:
        raise ValueError(f"Source pointer/patch drift: {actual_pin}")
    capture(["git", "diff", "--exit-code", "HEAD"], ROOT / "rocm-libraries", env)
    from configure_stage import get_topology

    requirements = get_topology().get_python_requires_for_stage("math-libs")
    if requirements:
        run_logged(
            [sys.executable, "-m", "pip", "install", *requirements],
            output / "stage-python-deps.log",
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
    with (output / "stage-configuration-stderr.log").open(
        "w", encoding="utf-8"
    ) as error_log:
        stage_args = subprocess.check_output(
            stage_command, cwd=ROOT, env=env, text=True, stderr=error_log
        )
    (output / "stage-cmake-args.txt").write_text(stage_args, encoding="utf-8")
    configure = [
        "cmake",
        "-B",
        str(build),
        "-S",
        str(ROOT),
        "-GNinja",
        f"-DTHEROCK_PACKAGE_VERSION={PACKAGE_VERSION}",
        f"-DTHEROCK_ROCJITSU_TEST_JOBS={min(16, int(capture(['nproc'], ROOT, env).strip()) // 2)}",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache",
        *shlex.split(stage_args),
    ]
    run_logged(configure, output / "configure.log", ROOT, env)
    if (
        cache_values(build / "CMakeCache.txt").get("THEROCK_PACKAGE_VERSION")
        != PACKAGE_VERSION
    ):
        raise ValueError("The fixed baseline package version was not retained")
    run_logged(
        [
            sys.executable,
            "build_tools/profile_build.py",
            "--output",
            str(output / "configure-dependencies-profile"),
            "--",
            "cmake",
            "--build",
            str(build),
            "--target",
            "hipCCL_tests+configure",
            "--",
            "-k",
            "0",
        ],
        output / "configure-dependencies.log",
        ROOT,
        env,
    )
    nested = build / "math-libs/hipCCL_tests/build"
    values = cache_values(nested / "CMakeCache.txt")
    for key, expected in {
        "BUILD_TEST": "ON",
        "BUILD_BENCHMARK": "ON",
        "BENCHMARK_CONFIG_TUNING": "OFF",
        "BENCHMARK_AUTOTUNED_TYPES_ONLY": "OFF",
    }.items():
        if values.get(key) != expected:
            raise ValueError(f"Unexpected coverage option {key}: {values.get(key)}")
    if values.get("ROCPRIM_SEGMENTED_PAIRS_SPLIT_TU", "OFF") != "OFF":
        raise ValueError("Do not combine this experiment with TU splitting")
    snapshot(build, output, "root")
    snapshot(nested, output, "nested-original")
    return nested


def inspect_images(
    compiler: Path, binary: Path, output: Path, env: dict[str, str]
) -> dict[str, object]:
    """Try the installed inspection interface; never silently equate flags to images."""
    tool = compiler.parent / "llvm-objdump"
    result = {
        "status": "unverified",
        "expected": sorted(ARCHES),
        "reason": "Installed offload-image inspection unavailable",
    }
    if not tool.is_file():
        return result
    help_text = capture([str(tool), "--help"], ROOT, env)
    (output / "llvm-objdump-help.txt").write_text(help_text, encoding="utf-8")
    text = ""
    if "--offloading" in help_text:
        try:
            text = capture([str(tool), "--offloading", str(binary)], ROOT, env)
        except subprocess.CalledProcessError as error:
            (output / "image-inspection-error.txt").write_text(
                error.output, encoding="utf-8"
            )
    (output / "embedded-images.txt").write_text(text, encoding="utf-8")
    observed = set(re.findall(r"\bgfx[0-9a-f]+\b", text))
    copier = compiler.parent / "llvm-objcopy"
    bundler = compiler.parent / "clang-offload-bundler"
    if (
        not observed
        and "--section-headers" in help_text
        and copier.is_file()
        and bundler.is_file()
    ):
        sections = capture([str(tool), "--section-headers", str(binary)], ROOT, env)
        (output / "section-headers.txt").write_text(sections, encoding="utf-8")
        copy_help = capture([str(copier), "--help"], ROOT, env)
        bundle_help = capture([str(bundler), "--help"], ROOT, env)
        (output / "llvm-objcopy-help.txt").write_text(copy_help, encoding="utf-8")
        (output / "clang-offload-bundler-help.txt").write_text(
            bundle_help, encoding="utf-8"
        )
        if (
            re.search(r"\s\.hip_fatbin\s", sections)
            and "--dump-section" in copy_help
            and "--list" in bundle_help
        ):
            dumped, copied = (
                output / "inspection-fatbin.bin",
                output / "inspection-binary-copy",
            )
            try:
                # Supply a separate output binary: inspection must never rewrite
                # the benchmark that the native linker just produced.
                capture(
                    [
                        str(copier),
                        "--dump-section",
                        f".hip_fatbin={dumped}",
                        str(binary),
                        str(copied),
                    ],
                    ROOT,
                    env,
                )
                text = capture(
                    [str(bundler), "--list", "--type=o", f"--input={dumped}"], ROOT, env
                )
                (output / "bundled-images.txt").write_text(text, encoding="utf-8")
                observed.update(re.findall(r"\bgfx[0-9a-f]+\b", text))
            except subprocess.CalledProcessError as error:
                (output / "bundler-inspection-error.txt").write_text(
                    error.output, encoding="utf-8"
                )
            finally:
                for temporary in (dumped, copied):
                    if temporary.is_file():
                        temporary.unlink()
    if observed:
        if observed != ARCHES:
            raise ValueError(f"Embedded image coverage differs: {observed}")
        result = {
            "status": "verified",
            "expected": sorted(ARCHES),
            "observed": sorted(observed),
        }
    return result


def run_trial(
    ninja: str,
    build: Path,
    output: Path,
    original: str,
    jobs: int,
    env: dict[str, str],
    compiler: Path,
    retained: Path,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    manifest = build / "build.ninja"
    modified = patch_flags(original, jobs)
    manifest.write_text(modified, encoding="utf-8", newline="")
    command = single_command(ninja, build, OBJECT, env)
    (output / "object-command.txt").write_text(command + "\n", encoding="utf-8")
    compiler_path, arguments = compiler_arguments(command, jobs)
    (output / "rendered-command.txt").write_text(
        capture([str(compiler_path), *arguments, "-###"], build, env), encoding="utf-8"
    )
    removed = force_object_rebuild(build)
    snapshot(build, output, "before")
    profile = [
        sys.executable,
        str(ROOT / "build_tools/profile_build.py"),
        "--interval",
        "1",
        "--output",
        str(output / "object-profile"),
        "--",
        ninja,
        "-C",
        str(build),
        "-d",
        "keeprsp",
        OBJECT,
    ]
    run_logged(profile, output / "object-build.log", ROOT, env)
    if manifest.read_text() != modified:
        raise ValueError("Ninja regenerated the modified edge during the trial")
    object_file = generated_path(build, OBJECT)
    object_state = {
        "sha256": digest(object_file),
        "size_bytes": object_file.stat().st_size,
        "mtime_ns": object_file.stat().st_mtime_ns,
        "ninja_record": last_object_record(build),
    }
    if not object_state["ninja_record"]:
        raise ValueError("Native Ninja log did not record the forced object build")
    snapshot(build, output, "after-object")
    (output / "link-dry-run.txt").write_text(
        capture([ninja, "-C", str(build), "-n", BENCHMARK], ROOT, env), encoding="utf-8"
    )
    link_command = single_command(ninja, build, BENCHMARK, env)
    (output / "link-command.txt").write_text(link_command + "\n", encoding="utf-8")
    run_logged(
        [
            sys.executable,
            str(ROOT / "build_tools/profile_build.py"),
            "--output",
            str(output / "link-profile"),
            "--",
            ninja,
            "-C",
            str(build),
            "-d",
            "keeprsp",
            BENCHMARK,
        ],
        output / "benchmark-link.log",
        ROOT,
        env,
    )
    for token in shlex.split(link_command):
        if token.startswith("@"):
            response = generated_path(build, token[1:])
            if response.is_file():
                shutil.copyfile(response, output / (response.name + ".retained"))
    if (
        object_file.stat().st_mtime_ns != object_state["mtime_ns"]
        or digest(object_file) != object_state["sha256"]
        or last_object_record(build) != object_state["ninja_record"]
    ):
        raise ValueError(
            "Benchmark linking unexpectedly recompiled/modified the heavy object"
        )
    if manifest.read_text() != modified:
        raise ValueError("Ninja manifest changed during benchmark linking")
    binary = generated_path(build, BENCHMARK)
    binary_state = {"sha256": digest(binary), "size_bytes": binary.stat().st_size}
    retained.mkdir(parents=True, exist_ok=True)
    retained_binary = retained / output.name
    shutil.copy2(binary, retained_binary)
    binary_state["retained_local_path"] = str(retained_binary)
    if binary_state["size_bytes"] <= 128 * 1024 * 1024:
        shutil.copy2(binary, output / "benchmark-executable")
        binary_state["saved"] = "benchmark-executable"
    else:
        binary_state["saved"] = "hash-and-inspection-only; executable exceeds128MiB"
    snapshot(build, output, "after-link")
    result = {
        "jobs": jobs,
        "utc_complete": utc(),
        "removed_generated_outputs": removed,
        "object": object_state,
        "benchmark": binary_state,
        "object_reused_for_link": True,
        "images": inspect_images(compiler, binary, output, env),
        "object_profile": json.loads(
            (output / "object-profile/summary.json").read_text()
        ),
        "link_profile": json.loads((output / "link-profile/summary.json").read_text()),
    }
    write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", required=True, help="JSON [1,2] or [2,1]")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/logs/offload-trial"
    )
    args = parser.parse_args()
    try:
        order = parse_order(args.order)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    require_ci(dict(os.environ))
    build, output = args.build_dir.resolve(), args.output.resolve()
    if (
        build == ROOT
        or not build.is_relative_to(ROOT)
        or not output.is_relative_to(build)
    ):
        parser.error(
            "Build/output must stay in this checkout's separate build directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Container ownership can differ from checkout ownership. Scope Git trust
    # to this checkout for these child commands, without changing global config.
    config_count = int(env.get("GIT_CONFIG_COUNT", "0") or "0")
    env[f"GIT_CONFIG_KEY_{config_count}"] = "safe.directory"
    env[f"GIT_CONFIG_VALUE_{config_count}"] = str(ROOT)
    env["GIT_CONFIG_COUNT"] = str(config_count + 1)
    meta = {
        "source_baseline": SOURCE_BASELINE,
        "diagnostic_sha": capture(["git", "rev-parse", "HEAD"], ROOT, env).strip(),
        "sdk_repository": "ROCm/TheRock",
        "sdk_run": SDK_RUN,
        "package_version": PACKAGE_VERSION,
        "family": FAMILY,
        "architectures": sorted(ARCHES),
        "order": order,
        "runner_name": env.get("RUNNER_NAME"),
        "runner_label": "aws-linux-scale-rocm-prod",
        "run_id": env.get("GITHUB_RUN_ID"),
        "ccache_disable": True,
        "removed_parallelism_disabling_environment_keys": sorted(
            DISABLING_ENV.intersection(env)
        ),
        "limitations": [
            "Two paired allocations with balanced order do not establish general run-to-run variance.",
            "Compiler cache bypass does not flush OS/filesystem caches.",
            "No GPU execution checks run on the build runner.",
            "Sampled process CPU/RSS has the limitations in profile_build metadata.",
        ],
    }
    write_json(output / "experiment.json", meta)
    nested = prepare(build, output, env)
    ninja = shutil.which("ninja")
    if not ninja:
        raise RuntimeError("Ninja must be installed in the CI build container")
    manifest = nested / "build.ninja"
    original = manifest.read_text()
    benchmark_env = trial_env(env)
    probe_text = patch_flags(original, 2)
    manifest.write_text(probe_text, encoding="utf-8", newline="")
    try:
        compiler, _ = compiler_arguments(
            single_command(ninja, nested, OBJECT, benchmark_env), 2
        )
        query = capture(
            [ninja, "-C", str(nested), "-t", "query", BENCHMARK], ROOT, benchmark_env
        )
        (output / "benchmark-target-query.txt").write_text(query, encoding="utf-8")
        if OBJECT not in query:
            raise ValueError(
                "The benchmark target does not depend on the original heavy object"
            )
        write_json(
            output / "toolchain.json",
            {
                "compiler_path": str(compiler),
                "compiler_sha256": digest(compiler),
                "ninja_version": capture(
                    [ninja, "--version"], ROOT, benchmark_env
                ).strip(),
            },
        )
        capability_probe(compiler, output / "capability", benchmark_env)
        results = []
        for index, jobs in enumerate(order, 1):
            results.append(
                run_trial(
                    ninja,
                    nested,
                    output / f"trial-{index}-jobs-{jobs}",
                    original,
                    jobs,
                    benchmark_env,
                    compiler,
                    build / "offload-executables",
                )
            )
            write_json(output / "results.json", results)
        same_objects = len({result["object"]["sha256"] for result in results}) == 1
        same_binaries = filecmp.cmp(
            results[0]["benchmark"]["retained_local_path"],
            results[1]["benchmark"]["retained_local_path"],
            shallow=False,
        )
        write_json(
            output / "comparison.json",
            {
                "order": order,
                "same_object_bytes": same_objects,
                "same_benchmark_bytes": same_binaries,
                "runtime_gpu_checks_required": not same_binaries,
                "image_inspection": [result["images"] for result in results],
                "object_wall_seconds": {
                    str(result["jobs"]): result["object_profile"]["wall_s"]
                    for result in results
                },
                "speedup_status": "paired observations only; inspect traces, variance, correctness and overhead before claiming gains",
            },
        )
    finally:
        manifest.write_text(original, encoding="utf-8", newline="")


if __name__ == "__main__":
    main()
