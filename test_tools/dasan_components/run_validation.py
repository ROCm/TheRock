# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Run explicitly selected installed-artifact suites and retain auditable results."""

import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def inside(path, prefix):
    return path.resolve().is_relative_to(prefix)


def environment(prefix, visible):
    env = dict(os.environ)
    for key in (
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "ASAN_OPTIONS",
        "DASAN_OPTIONS",
        "LSAN_OPTIONS",
        "UBSAN_OPTIONS",
        "HIP_CLANG_PATH",
        "HSA_TOOLS_LIB",
        "HSA_TOOLS_LIB64",
    ):
        env.pop(key, None)
    env.update(
        ROCM_PATH=str(prefix),
        HIP_PATH=str(prefix),
        THEROCK_BIN_DIR=str(prefix / "bin"),
        OUTPUT_ARTIFACTS_DIR=str(prefix),
        AMDGPU_FAMILIES="gfx950-dcgpu",
        AMDGPU_TARGETS="gfx950",
        TEST_TYPE="quick",
        TOTAL_SHARDS="1",
        SHARD_INDEX="1",
        HIP_PLATFORM="amd",
    )
    if visible is not None:
        env["ROCR_VISIBLE_DEVICES"] = visible
    env["PATH"] = ":".join(
        [
            str(prefix / "bin"),
            str(prefix / "lib/llvm/bin"),
            str(Path(sys.executable).parent),
            os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        ]
    )
    candidates = [
        prefix / "lib",
        prefix / "lib/llvm/lib",
        prefix / "lib/rocm_sysdeps/lib",
    ]
    candidates.extend((prefix / "lib/llvm/lib").glob("*-linux-gnu"))
    env["LD_LIBRARY_PATH"] = ":".join(str(p) for p in candidates if p.is_dir())
    return env


def audit(prefix, paths, env, output):
    """Reject broken symlinks and ROCm dependencies resolved outside installation."""
    records = []
    failures = []
    rocm_lib = re.compile(
        r"^(lib(?:amdhip|hsa-runtime|amd_comgr|hiprtc|rccl|rocprofiler|rocdecode|rocjpeg|rocshmem|rpp|hipfile|rdc|rocm_sysdeps)|libclang_rt\.dasan)"
    )
    for relative in paths:
        path = prefix / relative
        item = {"path": relative}
        if not path.is_file() or not inside(path, prefix):
            item["error"] = "missing or resolves outside installed prefix"
            failures.append(relative)
        else:
            item.update(resolved=str(path.resolve()), sha256=sha256(path))
            result = subprocess.run(
                ["ldd", str(path)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            item["ldd"] = result.stdout
            for line in result.stdout.splitlines():
                if "=> not found" in line:
                    failures.append(f"{relative}: {line.strip()}")
                match = re.match(r"\s*(\S+) => (\S+)", line)
                if match and rocm_lib.match(match[1]) and match[2].startswith("/"):
                    if not inside(Path(match[2]), prefix):
                        failures.append(
                            f"{relative}: external ROCm dependency {line.strip()}"
                        )
        records.append(item)
    report = {
        "status": "failed" if failures else "passed",
        "files": records,
        "failures": failures,
    }
    save(output / "provenance.json", report)
    return report


def junit_counts(paths):
    counts = dict(passed=0, failed=0, skipped=0)
    for path in paths:
        tree = ET.parse(path)
        for case in tree.iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                counts["failed"] += 1
            elif case.find("skipped") is not None:
                counts["skipped"] += 1
            else:
                counts["passed"] += 1
    return counts


def execute(suite, context, env, output):
    expand = lambda value: value.format(**context)
    name = suite["name"]
    command = [expand(value) for value in suite["command"]]
    cwd = Path(expand(suite.get("cwd", "{repo}")))
    record = {
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "timeout_seconds": suite.get("timeout", 1800),
        "status": "pending",
    }
    missing = [
        expand(value)
        for value in suite.get("requires", [])
        if not Path(expand(value)).exists()
    ]
    if missing:
        record.update(status="missing", missing=missing)
        return record
    for value in suite.get("installed_requires", []):
        path = Path(expand(value))
        if not path.exists() or not inside(path, Path(context["prefix"])):
            record.update(
                status="failed", error=f"installed path absent/outside prefix: {path}"
            )
            return record
    suite_env = dict(env)
    suite_env.update(
        {key: expand(value) for key, value in suite.get("env", {}).items()}
    )
    elf_paths = []
    prefix = Path(context["prefix"])
    for value in suite.get("installed_requires", []):
        path = Path(expand(value))
        if path.is_file():
            with path.open("rb") as source:
                if source.read(4) == b"\x7fELF":
                    elf_paths.append(str(path.relative_to(prefix)))
    if elf_paths:
        audit_output = output / (name + "-provenance")
        audit_output.mkdir()
        record["provenance"] = audit(prefix, elf_paths, suite_env, audit_output)
        if record["provenance"]["status"] != "passed":
            record.update(status="failed", error="Suite dependency audit failed")
            return record
    log = output / f"{name}.log"
    record["log"] = str(log)
    record["start_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.monotonic()
    try:
        with log.open("w") as stream:
            stream.write(json.dumps({"argv": command, "cwd": str(cwd)}) + "\n")
            stream.flush()
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=suite_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=record["timeout_seconds"])
                record.update(
                    returncode=code, status="passed" if code == 0 else "failed"
                )
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record.update(status="timeout")
        reports = [Path(expand(value)) for value in suite.get("junit", [])]
        if reports:
            record["counts"] = junit_counts(reports)
            if record["counts"]["failed"] or sum(record["counts"].values()) == 0:
                record["status"] = "failed"
        if suite.get("ctest_counts"):
            contents = log.read_text(errors="replace")
            totals = re.findall(
                r"\d+% tests passed, (\d+) tests failed out of (\d+)", contents
            )
            record["ctest_summaries"] = [
                dict(failed=int(failed), total=int(total)) for failed, total in totals
            ]
            # CTest's total includes tests it calls skipped; retain the lines instead
            # of presenting the aggregate as an executed-pass count.
            record["ctest_skip_lines"] = [
                line.strip()
                for line in contents.splitlines()
                if "Skipped" in line or "Not Run" in line
            ]
            if not totals or not sum(int(total) for _, total in totals):
                record.update(
                    status="failed", error="No nonempty CTest result summary found"
                )
            elif any(int(failed) for failed, _ in totals):
                record["status"] = "failed"
        if suite.get("summary"):
            report_path = Path(expand(suite["summary"]))
            record["suite_summary"] = json.loads(report_path.read_text())
            if record["suite_summary"].get("status") != "passed":
                record["status"] = "failed"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--visible-gpus",
        default=None,
        help="Optional explicit mask; otherwise retain CI isolation",
    )
    parser.add_argument(
        "--suite", action="append", help="Explicit suite names; otherwise only audit"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" and not args.dry_run:
        parser.error("Execution requires Linux")
    manifest = json.loads(args.manifest.read_text())
    context = {
        "prefix": str(args.prefix.resolve()),
        "repo": str(args.repo.resolve()),
        "bundle": str(args.bundle.resolve()),
        "output": str(args.output.resolve()),
        "python": sys.executable,
    }
    selected = args.suite or []
    suites = {suite["name"]: suite for suite in manifest["suites"]}
    if set(selected) - suites.keys():
        parser.error(f"unknown suites: {set(selected) - suites.keys()}")
    if args.dry_run:
        print(
            json.dumps(
                {"context": context, "selected": [suites[n] for n in selected]},
                indent=2,
            )
        )
        return 0
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be absent or empty to preserve previous results")
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix.resolve(strict=True)
    env = environment(prefix, args.visible_gpus)
    runtime_env = dict(env)
    probe = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import ctypes,sys; r=ctypes.CDLL(sys.argv[1]); n=ctypes.c_int(); rc=r.hipGetDeviceCount(ctypes.byref(n)); assert rc == 0, rc; print(n.value)",
            str(prefix / "lib/libamdhip64.so"),
        ],
        env=runtime_env,
        text=True,
    )
    context["gpu_count"] = int(probe.strip())
    if context["gpu_count"] < 2:
        raise RuntimeError("Component CI requires at least two runtime-visible GPUs")
    report = {
        "status": "running",
        "prefix": str(prefix),
        "manifest_sha256": sha256(args.manifest),
        "runner_sha256": sha256(Path(__file__)),
        "visible_gpus": args.visible_gpus,
        "environment": {
            key: env.get(key)
            for key in (
                "PATH",
                "LD_LIBRARY_PATH",
                "ROCM_PATH",
                "HIP_PATH",
                "ROCR_VISIBLE_DEVICES",
            )
        },
        "suites": [],
    }
    report["provenance"] = audit(prefix, manifest["audit_paths"], env, args.output)
    save(args.output / "summary.json", report)
    if report["provenance"]["status"] != "passed":
        report["status"] = "failed"
        save(args.output / "summary.json", report)
        return 1
    for name in selected:
        result = execute(suites[name], context, env, args.output)
        report["suites"].append(result)
        print(f"{name}: {result['status']}", flush=True)
        save(args.output / "summary.json", report)
    report["status"] = (
        "passed"
        if all(s["status"] == "passed" for s in report["suites"])
        else "incomplete"
    )
    report["coverage"] = (
        "selected suites only" if selected else "provenance audit only; no tests run"
    )
    save(args.output / "summary.json", report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
