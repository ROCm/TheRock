# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Targeted original-command replay and observational process lifecycle capture."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

try:
    import resource
except ImportError:
    resource = None

MAX_LOG_BYTES = 64 * 1024**2


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def properties(test: dict) -> dict:
    return {row["name"]: row["value"] for row in test.get("properties", [])}


def sequence(value: object) -> list[str]:
    if value is None:
        return []
    return value if isinstance(value, list) else [str(value)]


def fixture_closure(
    tests: list[dict],
    execute: str,
    verified_build_dependencies: frozenset[str] = frozenset(),
) -> list[str]:
    """Include fixture-dependent validations and all recursively required fixtures."""
    if verified_build_dependencies - {"signal-handler-test"}:
        raise ValueError("Unapproved non-CTest dependency resolution")
    by_name = {}
    for test in tests:
        by_name.setdefault(test["name"], []).append(test)
    if execute not in by_name or len(by_name[execute]) != 1:
        raise ValueError("Missing/duplicate original execution test")

    def unique(name):
        if len(by_name[name]) != 1:
            raise ValueError(
                "Ambiguous test in selected fixture/dependency namespace: " + name
            )
        return by_name[name][0]

    setups, cleanups = {}, {}
    for test in tests:
        props = properties(test)
        for key, table in [("FIXTURES_SETUP", setups), ("FIXTURES_CLEANUP", cleanups)]:
            for fixture in sequence(props.get(key)):
                table.setdefault(fixture, set()).add(test["name"])
    selected = {execute}
    # Test execution produces the fixture consumed by its original validations.
    produced = set(sequence(properties(unique(execute)).get("FIXTURES_SETUP")))
    if not produced:
        raise ValueError("Original execution has no validation fixture")
    for test in tests:
        if produced.intersection(sequence(properties(test).get("FIXTURES_REQUIRED"))):
            selected.add(test["name"])
    if len(selected) < 2:
        raise ValueError("Original validation consumers are missing")
    while True:
        previous = set(selected)
        for name in previous:
            props = properties(unique(name))
            for dependency in sequence(props.get("DEPENDS")):
                if dependency not in by_name:
                    if dependency in verified_build_dependencies:
                        continue
                    raise ValueError("Required named dependency is missing")
                selected.add(dependency)
            for fixture in sequence(props.get("FIXTURES_REQUIRED")):
                if fixture not in setups:
                    raise ValueError("Required fixture setup is missing")
                selected.update(setups[fixture])
                selected.update(cleanups.get(fixture, set()))
        if selected == previous:
            return sorted(selected)


def verify_build_target_alias(build: Path, app: Path, target: str, query: str) -> dict:
    expected = build / "bin/signal-handler/signal-handler-test"
    lines = [line.strip() for line in query.splitlines()]
    if (
        target != "signal-handler-test"
        or app != expected
        or not app.is_file()
        or app.is_symlink()
    ):
        raise ValueError("Consumer build-target identity differs")
    if lines.count(target + ":") != 1:
        raise ValueError("Ninja query lacks one exact target namespace")
    block = lines[lines.index(target + ":") + 1 :]
    if "input: phony" not in block or "outputs:" not in block:
        raise ValueError("Ninja alias shape differs")
    inputs = block[block.index("input: phony") + 1 : block.index("outputs:")]
    if app.relative_to(build).as_posix() not in inputs:
        raise ValueError(
            "Ninja target alias does not point to the original consumer app"
        )
    with app.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("Consumer app is not the expected native ELF")
    return dict(
        target=target,
        path=str(app),
        bytes=app.stat().st_size,
        sha256=sha(app),
        ninja_query_sha256=hashlib.sha256(query.encode()).hexdigest(),
        scope="Original DEPENDS property retained; exact non-CTest build target proved via stock successful build, Ninja alias and consumer app bytes",
    )


def verify_execute(test: dict, contract: dict) -> None:
    props = properties(test)
    if test["command"] != contract["original_execute_argv"]:
        raise ValueError("Generated execution argv differs from the actual failed case")
    if float(props.get("TIMEOUT", -1)) != contract["original_execute_timeout_s"]:
        raise ValueError("Original CTest deadline changed")
    if sequence(props.get("PASS_REGULAR_EXPRESSION")) != [
        contract["original_pass_regex"]
    ]:
        raise ValueError("Original pass expectation changed")


def parse_proc_stat(text: str) -> dict:
    tail = text[text.rindex(")") + 2 :].split()
    return dict(
        pid=int(text.split(" ", 1)[0]),
        start_ticks=int(tail[19]),
        state=tail[0],
        ppid=int(tail[1]),
        user_ticks=int(tail[11]),
        system_ticks=int(tail[12]),
    )


def read_proc(
    pid: int,
    include_maps: bool,
    proc_root: Path = Path("/proc"),
    expected_parent: tuple[int, int] | None = None,
    expected_start_ticks: int | None = None,
) -> dict:
    """Keep access/parse/race gaps typed; never equate unavailable with process exit."""
    path = proc_root / str(pid)
    try:
        row = parse_proc_stat((path / "stat").read_text())
        if row["pid"] != pid:
            raise ValueError("Stat PID differs from requested path")
    except (OSError, ValueError, IndexError) as error:
        return dict(
            pid=pid,
            status="unavailable",
            stage="initial-stat",
            error=error.__class__.__name__,
        )
    if expected_start_ticks is not None and row["start_ticks"] != expected_start_ticks:
        return dict(
            pid=pid,
            status="previously-owned-identity-rejected",
            expected_start_ticks=expected_start_ticks,
            observed_start_ticks=row["start_ticks"],
        )
    if expected_start_ticks is None and expected_parent is not None:
        parent_pid, parent_start = expected_parent
        if row["ppid"] != parent_pid:
            return dict(
                pid=pid,
                status="new-child-parent-rejected",
                observed_ppid=row["ppid"],
                expected_parent_pid=parent_pid,
            )
        try:
            parent = parse_proc_stat((proc_root / str(parent_pid) / "stat").read_text())
        except (OSError, ValueError, IndexError) as error:
            return dict(
                pid=pid,
                status="new-child-parent-unverified",
                expected_parent_pid=parent_pid,
                error=error.__class__.__name__,
            )
        if parent["pid"] != parent_pid or parent["start_ticks"] != parent_start:
            return dict(
                pid=pid,
                status="new-child-parent-identity-rejected",
                expected_parent_pid=parent_pid,
                expected_parent_start_ticks=parent_start,
                observed_parent_start_ticks=parent["start_ticks"],
            )
    row["status"] = "observed"
    try:
        row["exe"] = os.readlink(path / "exe")
    except OSError as error:
        row["exe_error"] = error.__class__.__name__
    for key in ["wchan", "io"]:
        try:
            row[key] = (path / key).read_text()[:8192]
        except OSError as error:
            row[key + "_error"] = error.__class__.__name__
    try:
        row["children"] = [
            int(value)
            for value in (path / "task" / str(pid) / "children").read_text().split()
        ]
    except (OSError, ValueError) as error:
        row["children_error"] = error.__class__.__name__
    if include_maps:
        try:
            maps = (path / "maps").read_text()
            row.update(maps=maps[: 2 * 1024**2], maps_truncated=len(maps) > 2 * 1024**2)
        except OSError as error:
            row["maps_error"] = error.__class__.__name__
    try:
        after = parse_proc_stat((path / "stat").read_text())
    except (OSError, ValueError, IndexError) as error:
        return dict(
            pid=pid,
            status="identity-unverified",
            stage="final-stat",
            initial_start_ticks=row["start_ticks"],
            error=error.__class__.__name__,
        )
    if after["pid"] != pid or after["start_ticks"] != row["start_ticks"]:
        return dict(
            pid=pid,
            status="identity-race-rejected",
            initial_start_ticks=row["start_ticks"],
            final_pid=after["pid"],
            final_start_ticks=after["start_ticks"],
        )
    if "exe" in row:
        try:
            after_exe = os.readlink(path / "exe")
            if after_exe != row["exe"]:
                return dict(
                    pid=pid,
                    status="exec-race-rejected",
                    start_ticks=row["start_ticks"],
                    initial_exe=row["exe"],
                    final_exe=after_exe,
                )
        except OSError as error:
            row["final_exe_error"] = error.__class__.__name__
    return row


def waited_cpu() -> dict:
    if resource is None:
        return dict(user_s=None, system_s=None)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(user_s=usage.ru_utime, system_s=usage.ru_stime)


def host_snapshot() -> dict:
    result = dict(epoch_s=time.time(), files={})
    for name in [
        "/proc/loadavg",
        "/proc/pressure/cpu",
        "/proc/pressure/io",
        "/proc/pressure/memory",
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpu.stat",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.current",
    ]:
        try:
            result["files"][name] = Path(name).read_text()[:16384]
        except OSError as error:
            result["files"][name] = dict(unavailable=error.__class__.__name__)
    return result


def launch(
    command: list[str],
    cwd: Path,
    env: dict,
    output: Path,
    observe: bool,
    interval: float,
    maps_interval: float,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    host_before = host_snapshot()
    cpu_before = waited_cpu()
    wall, own_cpu = time.monotonic(), time.process_time()
    recorded, truncated, tracked, last_maps = 0, False, {}, {}
    mapped_paths = set()
    observer_parent = None
    observer_error = None
    if observe:
        try:
            own = parse_proc_stat(
                (Path("/proc") / str(os.getpid()) / "stat").read_text()
            )
            observer_parent = (own["pid"], own["start_ticks"])
        except (OSError, ValueError, IndexError) as error:
            observer_error = dict(
                status="observer-ownership-unavailable", error=error.__class__.__name__
            )
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open(
        "wb"
    ) as stderr:
        child = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=stdout, stderr=stderr
        )
        with (output / "process.jsonl").open("w", encoding="utf-8") as log:
            while child.poll() is None:
                if not observe:
                    child.wait()
                    break
                if observer_parent is None:
                    unavailable = (
                        json.dumps(dict(pid=child.pid, **observer_error)) + "\n"
                    )
                    log.write(unavailable)
                    recorded += len(unavailable.encode())
                    child.wait()
                    break
                pending = [
                    (child.pid, observer_parent),
                    *((pid, None) for pid in tracked),
                ]
                seen = set()
                while pending:
                    pid, parent_identity = pending.pop()
                    if pid in seen:
                        continue
                    seen.add(pid)
                    now = time.monotonic()
                    include_maps = now - last_maps.get(pid, -1e20) >= maps_interval
                    row = read_proc(
                        pid,
                        include_maps,
                        expected_parent=parent_identity,
                        expected_start_ticks=tracked.get(pid),
                    )
                    if (
                        row["status"] == "observed"
                        and pid in tracked
                        and tracked[pid] != row["start_ticks"]
                    ):
                        row = dict(
                            pid=pid,
                            status="tracked-identity-change-rejected",
                            previous_start_ticks=tracked[pid],
                            current_start_ticks=row["start_ticks"],
                        )
                    if row["status"] == "observed":
                        tracked[pid] = row["start_ticks"]
                        pending.extend(
                            (number, (pid, row["start_ticks"]))
                            for number in row.get("children", [])
                        )
                    if include_maps and row["status"] == "observed":
                        last_maps[pid] = now
                        for line in row.get("maps", "").splitlines():
                            fields = line.split(maxsplit=5)
                            if (
                                len(fields) == 6
                                and fields[5].startswith("/")
                                and not fields[5].endswith(" (deleted)")
                            ):
                                mapped_paths.add(fields[5])
                    row.update(monotonic_s=now - wall, epoch_s=time.time())
                    encoded = json.dumps(row) + "\n"
                    if recorded + len(encoded.encode()) <= MAX_LOG_BYTES:
                        log.write(encoded)
                        recorded += len(encoded.encode())
                    else:
                        truncated = True
                time.sleep(interval)
        returncode = child.wait()
    elapsed, observer_cpu = time.monotonic() - wall, time.process_time() - own_cpu
    cpu_after = waited_cpu()
    return dict(
        command=command,
        returncode=returncode,
        wall_s=elapsed,
        host_before=host_before,
        host_after=host_snapshot(),
        waited_child_cpu={
            k: cpu_after[k] - v if v is not None else None
            for k, v in cpu_before.items()
        },
        controller_cpu_s=observer_cpu,
        process_observation=observe,
        process_observation_available=observer_parent is not None if observe else None,
        process_observation_preflight_error=observer_error,
        sample_interval_s=interval if observe else None,
        maps_interval_s=maps_interval if observe else None,
        process_log_bytes=recorded,
        process_log_truncated=truncated,
        mapped_paths=sorted(mapped_paths),
        observation_scope="Sampled owned descendants from main-thread children lists only; thread-spawned children, missed exec/signal/write transitions and escaped descendants can be missed. Unavailable/raced samples are typed, not process exits. No process environment read. Controller CPU is separate from waited-child CPU.",
    )


def preserve_files(
    source: Path, destination: Path, max_bytes: int = 256 * 1024**2
) -> list[dict]:
    """Copy actual test outputs before stock fixture cleanup; never normalize them."""
    destination.mkdir(parents=True, exist_ok=True)
    rows, total = [], 0
    if not source.exists():
        return rows
    for directory, dirs, names in os.walk(source, followlinks=False):
        for name in sorted(dirs + names):
            path = Path(directory) / name
            relative = path.relative_to(source)
            if path.is_symlink():
                rows.append(
                    dict(path=str(relative), type="symlink", target=os.readlink(path))
                )
                continue
            if not path.is_file():
                continue
            size = path.stat().st_size
            total += size
            row = dict(
                path=str(relative),
                type="regular",
                bytes=size,
                sha256=sha(path),
                mode=path.stat().st_mode & 0o7777,
                mtime_ns=path.stat().st_mtime_ns,
            )
            if total <= max_bytes:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                row["retained"] = True
            else:
                row["retained"] = False
                row["reason"] = (
                    "Output retention cap; full-file hash/size retained, no test file changed"
                )
            rows.append(row)
    return rows


def mapped_identities(paths: list[str]) -> list[dict]:
    rows, hashed, total = [], {}, 0
    for name in paths:
        path = Path(name)
        row = dict(path=name)
        try:
            resolved = path.resolve(strict=True)
            size = resolved.stat().st_size
            row.update(resolved_path=str(resolved), bytes=size)
            if not resolved.is_file():
                row["status"] = "not-regular-file"
            elif resolved not in hashed and (
                len(hashed) >= 512 or total + size > 4 * 1024**3
            ):
                row["status"] = "hash-cap-reached; unavailable"
            else:
                if resolved not in hashed:
                    hashed[resolved] = sha(resolved)
                    total += size
                row.update(status="file", sha256=hashed[resolved])
        except OSError as error:
            row.update(status="unavailable", error=error.__class__.__name__)
        rows.append(row)
    return rows


def run_trial(root: Path, contract: dict) -> int:
    if (
        sys.platform != "linux"
        or os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != "ROCm/TheRock"
    ):
        raise ValueError("Actual replay must run only in TheRock Linux CI")
    output = root / "profiler-lifecycle"
    before = json.loads((output / "before.json").read_text())
    if (
        not before["complete"]
        or before["artifact_run_id"] != contract["artifact_run_id"]
    ):
        raise ValueError("Missing original producer/input receipt")
    save(output / "trial-progress.json", dict(stage="stock setup", complete=False))
    driver_path = root / next(iter(contract["source_files"]))
    spec = importlib.util.spec_from_file_location("stock_profiler_driver", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.setup_env()
    save(
        output / "stock-setup-commands.json",
        dict(
            configure=driver.get_cmake_config_cmd(),
            build=driver.get_cmake_build_cmd(),
            base_ctest=driver.get_ctest_cmd(),
            nonsecret_environment={
                k: driver.environ_vars.get(k)
                for k in [
                    "LD_LIBRARY_PATH",
                    "ROCM_PATH",
                    "HIP_PATH",
                    "HIP_PLATFORM",
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "KUBE_CPU_REQUEST",
                ]
            },
        ),
    )
    setup_wall, setup_cpu = time.monotonic(), waited_cpu()
    driver.cmake_config()
    driver.cmake_build()
    setup_after = waited_cpu()
    save(
        output / "stock-setup-result.json",
        dict(
            wall_s=time.monotonic() - setup_wall,
            waited_child_cpu={
                k: setup_after[k] - v if v is not None else None
                for k, v in setup_cpu.items()
            },
            scope="Original test-app configure/build, outside all measured replays",
        ),
    )
    save(
        output / "trial-progress.json",
        dict(stage="generated command/fixture guards", complete=False),
    )
    build = driver.ROCPROFILER_SDK_TESTS_BUILD_PATH
    base = driver.get_ctest_cmd()
    show = subprocess.run(
        [*base, "--show-only=json-v1"],
        cwd=driver.ROCPROFILER_SDK_TESTS_PATH,
        env=driver.environ_vars,
        capture_output=True,
        check=True,
    )
    (output / "all-generated-tests.json").write_bytes(show.stdout)
    tests = json.loads(show.stdout)["tests"]
    execute = next(t for t in tests if t["name"] == contract["original_execute_name"])
    verify_execute(execute, contract)
    app = build / "bin/signal-handler/signal-handler-test"
    ninja = shutil.which("ninja")
    if not ninja:
        raise ValueError("Original Ninja provider unavailable for target proof")
    query = subprocess.run(
        [ninja, "-C", str(build), "-t", "query", "signal-handler-test"],
        cwd=driver.ROCPROFILER_SDK_TESTS_PATH,
        env=driver.environ_vars,
        capture_output=True,
        text=True,
        check=True,
    )
    (output / "signal-handler-ninja-query.txt").write_text(
        query.stdout, encoding="utf-8"
    )
    proof = verify_build_target_alias(build, app, "signal-handler-test", query.stdout)
    proof.update(
        ninja_query_command=query.args,
        ninja_provider=ninja,
        ninja_provider_sha256=sha(Path(ninja)),
    )
    save(output / "build-dependency-resolution.json", proof)
    app_before = dict(path=proof["path"], bytes=proof["bytes"], sha256=proof["sha256"])
    save(output / "consumer-built-app.json", app_before)
    closure = fixture_closure(tests, execute["name"], frozenset({proof["target"]}))
    regex = "^(" + "|".join(re.escape(name) for name in closure) + ")$"
    command = [*base, "--no-tests=error", "--tests-regex", regex]
    selected = subprocess.run(
        [*command, "--show-only=json-v1"],
        cwd=driver.ROCPROFILER_SDK_TESTS_PATH,
        env=driver.environ_vars,
        capture_output=True,
        check=True,
    )
    selected_data = json.loads(selected.stdout)
    if sorted(t["name"] for t in selected_data["tests"]) != closure:
        raise ValueError("Generated fixture closure and actual selected tests differ")
    save(output / "selected-original-tests.json", selected_data)
    test_output = build / "rocprofv3/signal-handler/without-signal-handling-spawn"
    expected = (
        root
        / "build/share/rocprofiler-sdk/tests/build/rocprofv3/signal-handler/without-signal-handling-spawn"
    )
    if test_output != expected or not test_output.is_relative_to(build):
        raise ValueError(
            "Original output path differs; no alternate output directory allowed"
        )
    cases = []
    for index, mode in enumerate(contract["case_order"], 1):
        save(
            output / "trial-progress.json",
            dict(stage="replay", index=index, mode=mode, complete=False),
        )
        dest = output / f"case-{index:02d}-{mode}"
        result = launch(
            command,
            driver.ROCPROFILER_SDK_TESTS_PATH,
            driver.environ_vars,
            dest,
            mode == "proc",
            contract["proc_interval_s"],
            contract["maps_interval_s"],
        )
        post_wall, post_cpu = time.monotonic(), time.process_time()
        result.update(
            index=index,
            mode=mode,
            original_test_output_directory_exists=test_output.exists(),
            original_test_outputs=preserve_files(test_output, dest / "test-outputs"),
            ctest_logs=preserve_files(build / "Testing/Temporary", dest / "ctest-logs"),
        )
        result["sampled_mapped_file_identities_after_case"] = mapped_identities(
            result["mapped_paths"]
        )
        result.update(
            post_case_capture_wall_s=time.monotonic() - post_wall,
            post_case_capture_cpu_s=time.process_time() - post_cpu,
        )
        if sha(app) != app_before["sha256"]:
            raise ValueError("Consumer application changed between cases")
        save(dest / "result.json", result)
        cases.append(result)
        save(output / "cases.json", cases)
    # Optional tracing is an additional perturbed replay, never a replacement for plain cases.
    tracer = shutil.which("strace")
    trace = dict(
        available=bool(tracer),
        attempted=False,
        reason="Stock executable absent; no installation/capability change",
    )
    if tracer:
        probe = subprocess.run(
            [
                tracer,
                "-e",
                "trace=process,signal,write,writev",
                "-o",
                str(output / "strace-access.log"),
                "/usr/bin/true",
            ],
            env=driver.environ_vars,
            capture_output=True,
        )
        trace.update(
            path=tracer,
            sha256=sha(Path(tracer)),
            access_probe_returncode=probe.returncode,
            access_probe_stderr=probe.stderr.decode(errors="replace")[:8192],
        )
        if probe.returncode == 0:
            if shutil.disk_usage(output).free < 4 * 1024**3:
                trace["reason"] = (
                    "Insufficient trace headroom; original plain/proc cases retained"
                )
            else:
                trace.update(
                    attempted=True,
                    reason="Separate additional ptrace-perturbed case; original argv/deadlines retained",
                )
                dest = output / "case-strace"
                raw_trace = root / "profiler-lifecycle-raw-trace"
                raw_trace.mkdir(exist_ok=False)
                result = launch(
                    [
                        tracer,
                        "-ff",
                        "-ttt",
                        "-T",
                        "-e",
                        "trace=process,signal,write,writev",
                        "-o",
                        str(raw_trace / "trace"),
                        *command,
                    ],
                    driver.ROCPROFILER_SDK_TESTS_PATH,
                    driver.environ_vars,
                    dest,
                    False,
                    contract["proc_interval_s"],
                    contract["maps_interval_s"],
                )
                result["original_test_outputs"] = preserve_files(
                    test_output, dest / "test-outputs"
                )
                result["ctest_logs"] = preserve_files(
                    build / "Testing/Temporary", dest / "ctest-logs"
                )
                trace_files = sorted(raw_trace.glob("trace.*"))
                total_trace_bytes = sum(path.stat().st_size for path in trace_files)
                retained = dest / "strace"
                retained.mkdir()
                trace["files"] = []
                for path in trace_files:
                    row = dict(
                        file=path.name,
                        bytes=path.stat().st_size,
                        sha256=sha(path),
                        full_trace_retained=total_trace_bytes <= MAX_LOG_BYTES,
                    )
                    if row["full_trace_retained"]:
                        shutil.copy2(path, retained / path.name)
                    else:
                        with path.open("rb") as stream:
                            (retained / (path.name + ".head")).write_bytes(
                                stream.read(65536)
                            )
                            stream.seek(max(0, row["bytes"] - 65536))
                            (retained / (path.name + ".tail")).write_bytes(
                                stream.read(65536)
                            )
                        row["scope"] = (
                            "Artifact cap: prefix/tail and full-file hash only; no complete transition proof"
                        )
                    trace["files"].append(row)
                save(dest / "result.json", result)
                trace["result"] = result
    save(output / "strace-status.json", trace)
    if sha(app) != app_before["sha256"]:
        raise ValueError("Consumer application changed after tracing")
    save(
        output / "trial-summary.json",
        dict(
            cases=len(cases),
            failures=[r["index"] for r in cases if r["returncode"]],
            consumer_app_unchanged=sha(app) == app_before["sha256"],
            original_execute_argv=execute["command"],
            original_execute_properties=properties(execute),
            fixture_closure=closure,
            source_artifact_run_id=contract["artifact_run_id"],
            scope="Subset diagnostic omits preceding full-suite/load history; non-reproduction is not a fix. Plain/proc/ptrace outcomes and overhead stay separate. No CDash submission.",
        ),
    )
    save(output / "trial-progress.json", dict(stage="complete", complete=True))
    return (
        1
        if any(row["returncode"] for row in cases)
        or (trace.get("attempted") and trace["result"]["returncode"])
        else 0
    )
