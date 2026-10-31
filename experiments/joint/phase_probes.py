#!/usr/bin/env python3
"""Bookkeeping and isolated phase probes; execution is restricted to Rock CI."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
import time

BASE = "5b7cb88373362ab80dc0f19605915e534f4158aa"
UNSET = ("ROCM_PATH", "ROCM_DIR", "HIP_PATH", "HIP_DIR")
RUNTIME_KEYS = ("PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "ASAN_OPTIONS", "CCACHE_DISABLE", *UNSET)

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def run(argv, **kwargs):
    return subprocess.check_output(argv, text=True, **kwargs)

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def guard(source):
    if (sys.platform != "linux" or os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != "ROCm/TheRock"):
        raise RuntimeError("Execution is restricted to Linux ROCm/TheRock Actions")
    if run(["git", "rev-parse", "HEAD"], cwd=source).strip() != BASE:
        raise RuntimeError("Source checkout differs from frozen HEAD")

def preflight(source, output):
    spec = importlib.util.spec_from_file_location("observer", Path(__file__).with_name("collect_build_perf.py"))
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    models = sorted(set(re.findall(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.M)))
    affinity = sorted(os.sched_getaffinity(0))
    groups = observer.groups_snapshot(observer.cgroup_dirs())
    record = json.loads((Path(__file__).parent / "records/getri_z.json").read_text())
    compiler = Path(record["argv"][0])
    toolchain = {}
    for path in (compiler, compiler.with_name("ld.lld")):
        if path.is_file():
            toolchain[path.name] = {"path": str(path.resolve()), "sha256": digest(path),
                "bytes": path.stat().st_size, "version": run([str(path), "--version"])}
    save(output, {"utc": utc(), "source_sha": BASE, "host_logical_cpus": os.cpu_count(), "affinity": affinity,
        "models": models, "visible_cgroups": groups, "toolchain": toolchain,
        "runner_name": os.environ.get("RUNNER_NAME"),
        "pair_id": os.environ.get("PERF_PAIR_ID"), "runtime_environment": {k: os.environ.get(k) for k in RUNTIME_KEYS},
        "limits": "Logical CPUs are not physical cores; hidden ancestors and sibling interference remain unknown."})
    if len(affinity) != 192 or os.cpu_count() != 192 or not models or any("AMD EPYC 9R45" not in m for m in models):
        raise RuntimeError("CPU model/allocation differs from baseline's 192 logical CPUs and EPYC 9R45")
    if not groups or any("cpu.max" in g and not g["cpu.max"].startswith("max ") for g in groups.values()):
        raise RuntimeError("Visible CPU quota differs from baseline")

def prime(source, output, label):
    start = time.monotonic()
    manifest = output / "original-inputs.json"
    if manifest.exists():
        paths = [Path(p) for p in json.loads(manifest.read_text())["paths"]]
    else:
        roots = {source / "rocm-libraries/projects/rocsolver"}
        for record in (Path(__file__).parent / "records").glob("get*.json"):
            argv = json.loads(record.read_text())["argv"]
            for i, value in enumerate(argv):
                if value in ("-I", "-isystem", "-iquote"): roots.add(Path(argv[i + 1]))
                elif value.startswith("-I") and len(value) > 2: roots.add(Path(value[2:]))
        paths = sorted({p.resolve() for root in roots if root.is_dir() for p in root.rglob("*") if p.is_file()})
        if not paths: raise RuntimeError("Original source/include inputs missing")
        save(manifest, {"utc": utc(), "roots": sorted(map(str, roots)), "paths": list(map(str, paths)),
            "limits": "Same original path set, including patched original files; new helper files not in this set. No residency proof/cache drop."})
    rows = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in paths]
    save(output / f"prime-{label}.json", {"utc": utc(), "wall_s": time.monotonic() - start, "files": rows,
        "bytes_read": sum(x["bytes"] for x in rows), "cache_contract": "Compiler cache disabled; deliberately primed original inputs, unknown actual residency.",
        "order": os.environ.get("PERF_TRIAL_ORDER"),
        "order_limit": "Pairs 1/3 use B-C and pair 2 uses C-B; three pairs cannot perfectly balance order. Compiler/toolchain/kernel caches may remain warm."})

def reset(source, output, label):
    project = source / "build/math-libs/BLAS/rocSOLVER"
    nested = project / "build"
    assert nested.resolve() == source / "build/math-libs/BLAS/rocSOLVER/build"
    start = time.monotonic()
    result = subprocess.run(["ninja", "-C", str(nested), "-t", "clean"], capture_output=True, text=True)
    if result.returncode: raise RuntimeError(result.stderr or "Nested clean failed")
    removed = []
    for path in [nested / ".ninja_log", nested / ".ninja_deps", project / "stamp/build.stamp", project / "stamp/stage.stamp"]:
        if path.exists(): path.unlink(); removed.append(str(path))
    for name in ("stage", "dist"):
        path = project / name
        if path.exists():
            if path.is_symlink() or path.resolve().parent != project.resolve(): raise RuntimeError(f"Unsafe project reset: {path}")
            shutil.rmtree(path); removed.append(str(path))
    save(output / f"reset-{label}.json", {"utc": utc(), "wall_s": time.monotonic() - start, "stdout": result.stdout, "removed": removed,
        "scope": "All nested solver outputs and local stage/dist; dependency installations retained."})

def inventory(source, output, label):
    stage = source / "build/math-libs/BLAS/rocSOLVER/stage"
    files = []
    for path in sorted(stage.rglob("*")):
        rel = str(path.relative_to(stage))
        if path.is_symlink(): files.append({"path": rel, "symlink": os.readlink(path)})
        elif path.is_file(): files.append({"path": rel, "bytes": path.stat().st_size, "sha256": digest(path)})
    libraries = [p for p in stage.rglob("librocsolver.so*") if p.is_file() and not p.is_symlink()]
    if len(libraries) != 1: raise RuntimeError(f"Expected one real installed solver DSO: {libraries}")
    for required in ("rocsolver-test", "rocsolver-bench"):
        if not any(Path(x["path"]).name == required for x in files):
            raise RuntimeError(f"Required installed client is missing: {required}")
    library = libraries[0]
    nm = shutil.which("nm") or str(source / "build/core/clr/dist/lib/llvm/bin/llvm-nm")
    readelf = shutil.which("readelf") or str(source / "build/core/clr/dist/lib/llvm/bin/llvm-readelf")
    nm_argv = [nm, "--dynamic", "--defined-only", "--format=posix", "--no-demangle", str(library)]
    symbols = sorted(" ".join(line.split()[:2]) for line in run(nm_argv).splitlines() if len(line.split()) >= 2)
    dynamic = run([readelf, "--dynamic", str(library)])
    soname = re.findall(r"\(SONAME\).*\[(.*?)\]", dynamic)
    if len(soname) != 1 or not symbols: raise RuntimeError("Missing dynamic-symbol/SONAME contract")
    save(output / f"artifacts-{label}.json", {"utc": utc(), "stage": str(stage), "files": files,
        "library": str(library), "dynamic_defined_symbols": symbols, "soname": soname[0],
        "nm_argv": nm_argv, "readelf_dynamic": dynamic,
        "limits": "Public ELF checks do not establish numerical, device-registration or ODR correctness."})
    # Preserve the actual outer wrapper, including environment operations omitted by launcher records.
    lines = (source / "build/build.ninja").read_text().splitlines()
    edges = []
    for index, line in enumerate(lines):
        if line.startswith("build ") and "rocSOLVER/stamp/build.stamp" in line and ": CUSTOM_COMMAND" in line:
            block = []
            for value in lines[index:]:
                if block and not value: break
                block.append(value)
            if any("rocSOLVER_build.log" in value for value in block): edges.append(block)
    if len(edges) != 1: raise RuntimeError("Expected one actual outer solver build edge")
    command = next(line.split(" = ", 1)[1] for line in edges[0] if line.lstrip().startswith("COMMAND = "))
    tokens = shlex.split(command)
    env_index = next(i + 2 for i, value in enumerate(tokens[:-1]) if value == "-E" and tokens[i + 1] == "env")
    env_end = tokens.index("--", env_index)
    operations = tokens[env_index:env_end]
    if sorted(operations) != sorted(f"--unset={name}" for name in UNSET):
        raise RuntimeError(f"Outer build environment changed: {operations}")
    save(output / f"outer-build-environment-{label}.json", {"utc": utc(), "edge": edges[0],
        "environment_operations": operations, "whitelisted_parent_environment": {k: os.environ.get(k) for k in RUNTIME_KEYS}})
    commands = run(["ninja", "-C", str(source / "build/math-libs/BLAS/rocSOLVER/build"), "-t", "commands"])
    (output / f"solver-commands-{label}.txt").write_text(commands, encoding="utf-8")
    logs = output / f"logs-{label}"
    logs.mkdir(parents=True, exist_ok=True)
    for path in (source / "build/logs").glob("rocSOLVER_*.log"): shutil.copy2(path, logs / path.name)
    shutil.copy2(source / "build/math-libs/BLAS/rocSOLVER/build/CMakeCache.txt", output / f"solver-cache-{label}.txt")

def compare(output):
    a = json.loads((output / "artifacts-baseline.json").read_text())
    b = json.loads((output / "artifacts-candidate.json").read_text())
    allocation_a = json.loads((output / "allocation-baseline.json").read_text())
    allocation_b = json.loads((output / "allocation-candidate.json").read_text())
    toolchain_equal = (len(allocation_a["toolchain"]) == 2
        and allocation_a["toolchain"] == allocation_b["toolchain"])
    allocation_equal = all(allocation_a[key] == allocation_b[key]
        for key in ("host_logical_cpus", "affinity", "models", "runtime_environment"))
    paths_a, paths_b = {x["path"] for x in a["files"]}, {x["path"] for x in b["files"]}
    symbols_a, symbols_b = set(a["dynamic_defined_symbols"]), set(b["dynamic_defined_symbols"])
    links_a = {x["path"]: x["symlink"] for x in a["files"] if "symlink" in x}
    links_b = {x["path"]: x["symlink"] for x in b["files"] if "symlink" in x}
    result = {"utc": utc(), "toolchain_equal": toolchain_equal,
        "visible_allocation_and_runtime_environment_equal": allocation_equal,
        "soname_equal": a["soname"] == b["soname"], "symlinks_equal": links_a == links_b,
        "missing_required_paths": sorted(paths_a - paths_b), "extra_paths": sorted(paths_b - paths_a),
        "removed_dynamic_symbols": sorted(symbols_a - symbols_b), "added_dynamic_symbols": sorted(symbols_b - symbols_a),
        "gpu_correctness": "Not executed; supported ASan GPU allocation remains unresolved."}
    result["passed"] = toolchain_equal and allocation_equal and result["soname_equal"] and result["symlinks_equal"] and not any(result[k] for k in
        ("missing_required_paths", "extra_paths", "removed_dynamic_symbols", "added_dynamic_symbols"))
    save(output / "artifact-comparison.json", result)
    if not result["passed"]: raise RuntimeError("Public ELF or required installed output contract changed")

def probes(source, output):
    output.mkdir(parents=True, exist_ok=False)
    helper = Path(__file__).with_name("trace_getf2.py")
    records = Path(__file__).parent / "records"
    jobs = []
    start = utc()
    for name, expected in (("getri_z", "getri-specialized"), ("getf2_c", "getf2-small")):
        for trace in (False, True):
            label = f"{name}-{'trace' if trace else 'control'}"
            command = ["cmake", "-E", "env", *[f"--unset={k}" for k in UNSET], "--", sys.executable,
                str(helper), "--record", str(records / f"{name}.json"), "--source-root", str(source),
                "--expected-source", expected, "--output", str(output / label)]
            if trace: command += ["--trace", "--device-link-trace"]
            stream = (output / f"{label}.log").open("wb")
            jobs.append((label, command, subprocess.Popen(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT), stream))
    results = []
    for label, command, process, stream in jobs:
        code = process.wait(); stream.close()
        results.append({"label": label, "argv": command, "exit_code": code})
    save(output / "cohort.json", {"utc": start, "end_utc": utc(), "jobs": results,
        "limits": ["One concurrent control/trace pair per source; provisional overhead only.",
            "Four compilers co-run; interference is not separated from trace overhead.",
            "Object hashes/sizes retained; private object bytes excluded from evidence archive."]})
    return int(any(x["exit_code"] != 0 for x in results))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "prime", "reset", "inventory", "compare", "probes"))
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    source = args.source.resolve(); guard(source)
    if args.action == "preflight": preflight(source, args.output)
    elif args.action == "prime": prime(source, args.output, args.label)
    elif args.action == "reset": reset(source, args.output, args.label)
    elif args.action == "inventory": inventory(source, args.output, args.label)
    elif args.action == "compare": compare(args.output)
    else: return probes(source, args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
