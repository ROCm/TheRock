#!/usr/bin/env python3
"""Read allocated hardware in Linux ROCm/TheRock Actions; never run GPU compute.

No SDK installation, compilation, numerical tests, cleanup, device configuration,
or arbitrary environment dump. An optional existing rocminfo is diagnostic only.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import time
from typing import Any

if sys.platform == "linux":
    import resource
else:
    resource = None  # Help/syntax checks remain usable on the Windows client.

BASE = "5b7cb88373362ab80dc0f19605915e534f4158aa"
LABEL = "linux-gfx950-1gpu-ccs-ossci-rocm"
ENV_KEYS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_XNACK", "KUBE_CPU_REQUEST")
CGROUP_FILES = ("cpu.max", "cpu.stat", "cpu.weight", "cpuset.cpus.effective",
                "cpu.cfs_quota_us", "cpu.cfs_period_us", "memory.max", "memory.current",
                "memory.events", "cpu.pressure", "memory.pressure", "io.pressure", "io.stat")


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path: Path, cap: int = 65536) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            data = stream.read(cap + 1)
        return {"path": str(path), "text": data[:cap].decode(errors="replace"),
                "truncated": len(data) > cap}
    except OSError as error:
        return {"path": str(path), "error": str(error)}


def cgroup_snapshot() -> dict[str, Any]:
    membership = []
    cgroup_text = read(Path("/proc/self/cgroup"))
    mount_text = read(Path("/proc/self/mountinfo"))
    for line in cgroup_text.get("text", "").splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3:
            membership.append((set(fields[1].split(",")), fields[2]))
    dirs = set()
    for line in mount_text.get("text", "").splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields, types = left.split(), right.split()
        if len(fields) < 5 or len(types) < 3 or types[0] not in ("cgroup", "cgroup2"):
            continue
        mount = Path(fields[4])
        for controllers, name in membership:
            if types[0] == "cgroup" and not controllers.intersection(types[2].split(",")):
                continue
            relative = os.path.relpath(name, fields[3])
            current = mount if relative.startswith("..") else mount / relative
            while True:
                dirs.add(current)
                if current == mount or current.parent == current:
                    break
                current = current.parent
    return {"membership": cgroup_text,
            "visible": {str(path): {name: read(path / name) for name in CGROUP_FILES
                                    if (path / name).exists()} for path in sorted(dirs, key=str)},
            "limit": "Only visible mounted ancestry; hidden ancestor quota and sibling activity remain unknown."}


def device(path: Path) -> dict[str, Any]:
    try:
        value = path.stat()
        return {"path": str(path), "exists": True, "resolved_path": str(path.resolve()),
                "mode": stat.filemode(value.st_mode), "uid": value.st_uid, "gid": value.st_gid,
                "character_device": stat.S_ISCHR(value.st_mode),
                "major": os.major(value.st_rdev), "minor": os.minor(value.st_rdev),
                "readable_by_process": os.access(path, os.R_OK),
                "writable_by_process": os.access(path, os.W_OK)}
    except OSError as error:
        return {"path": str(path), "exists": False, "error": str(error)}


def hardware() -> dict[str, Any]:
    cpuinfo = read(Path("/proc/cpuinfo"), 1024 * 1024)
    models = sorted({line.split(":", 1)[1].strip()
                     for line in cpuinfo.get("text", "").splitlines() if line.startswith("model name")})
    kfd_root = Path("/sys/class/kfd/kfd/topology/nodes")
    try:
        nodes = [{"node": path.name, "files": {name: read(path / name)
                  for name in ("gpu_id", "name", "properties")}}
                 for path in sorted(kfd_root.iterdir()) if path.is_dir()][:64]
        topology: Any = {"path": str(kfd_root), "nodes": nodes}
    except OSError as error:
        topology = {"path": str(kfd_root), "error": str(error)}
    drm = []
    for path in sorted(Path("/sys/class/drm").glob("renderD*"))[:64]:
        drm.append({"node": path.name, "files": {name: read(path / "device" / name)
                    for name in ("vendor", "device", "subsystem_vendor", "subsystem_device", "revision", "numa_node")}})
    # Check render nodes independently: numerous card* placeholders can otherwise
    # consume a combined listing's cap before the first renderD* entry.
    render_paths = sorted(Path("/dev/dri").glob("renderD*"))
    card_paths = sorted(Path("/dev/dri").glob("card*"))
    render_limit = 256
    return {"cpu_models": models, "logical_cpus": os.cpu_count(),
            "process_affinity": sorted(os.sched_getaffinity(0)),
            "uname": list(platform.uname()), "euid": os.geteuid(), "groups": os.getgroups(),
            "rlimits": {name: list(resource.getrlimit(getattr(resource, name)))
                        for name in ("RLIMIT_AS", "RLIMIT_MEMLOCK", "RLIMIT_NOFILE", "RLIMIT_NPROC")},
            "devices": [device(Path("/dev/kfd")),
                        *[device(p) for p in render_paths[:render_limit]],
                        *[device(p) for p in card_paths[:8]]],
            "dri_enumeration": {"render_entry_count": len(render_paths),
                "render_sample_limit": render_limit,
                "render_enumeration_complete": len(render_paths) <= render_limit,
                "card_entry_count": len(card_paths), "card_sample_limit": 8},
            "kfd_topology": topology, "drm_sysfs": drm,
            "limit": "Mounted host topology can expose more devices than runtime isolation allows. No device is opened or configured by this Python code."}


def optional_rocminfo(output: Path) -> dict[str, Any]:
    tool = shutil.which("rocminfo")
    if not tool:
        return {"available": False, "reason": "No rocminfo on the existing container PATH; no installation attempted."}
    path = Path(tool)
    row: dict[str, Any] = {"available": True, "path": str(path.resolve()), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "commands": [],
        "limit": "Existing diagnostic tool only, not the pinned HEAD ROCm product, an ASan test, or a numerical validation."}
    for label, argv, timeout in (("version", [tool, "--version"], 5), ("agents", [tool], 15)):
        log = output / f"rocminfo-{label}.log"
        start = time.monotonic()
        try:
            with log.open("wb") as stream:
                process = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=timeout, preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024)))
            outcome = {"exit_code": process.returncode}
        except subprocess.TimeoutExpired:
            outcome = {"timed_out": True}
        except OSError as error:
            outcome = {"error": str(error)}
        row["commands"].append({"argv": argv, "timeout_s": timeout,
            "wall_s": time.monotonic() - start, "log": log.name, **outcome})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--try-rocminfo", action="store_true")
    args = parser.parse_args()
    if (sys.platform != "linux" or os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != "ROCm/TheRock"):
        parser.error("Probe execution is restricted to Linux ROCm/TheRock Actions")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source_root.resolve()
    result: dict[str, Any] = {"utc": utc(), "requested_runner_label": LABEL,
        "source_root": str(source), "expected_source_sha": BASE,
        "instrumentation_sha": os.environ.get("GITHUB_SHA"),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_id": os.environ.get("GITHUB_RUN_ID"), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "job": os.environ.get("GITHUB_JOB"), "runner_name": os.environ.get("RUNNER_NAME"),
        "requested_image_ref": os.environ.get("CAPACITY_PROBE_REQUESTED_IMAGE"),
        "resolved_image_digest": None,
        "image_digest_limit": "Use the retained Actions container-initialization log for any resolved platform-specific digest; no Docker socket is accessed.",
        "allocation_observed": True, "allowlisted_environment": {key: os.environ.get(key) for key in ENV_KEYS},
        "product_or_asan_correctness_performed": False,
        "scope": "Allocation, read-only device visibility and optional existing-tool enumeration only; no GPU compute, SDK installation or build."}
    try:
        sha = subprocess.check_output(["git", "-c", f"safe.directory={source}", "-C", str(source), "rev-parse", "HEAD"],
                                      text=True, stderr=subprocess.STDOUT, timeout=10).strip()
        result["source_sha"] = sha
        if sha != BASE:
            raise RuntimeError(f"Frozen source check failed: {sha}")
        result["source_guard_passed"] = True
        result["hardware"] = hardware()
        result["cgroups"] = cgroup_snapshot()
        nodes = result["hardware"]["devices"]
        has_kfd = any(d["path"] == "/dev/kfd" and d.get("character_device") for d in nodes)
        has_render = any(Path(d["path"]).name.startswith("renderD") and d.get("character_device") for d in nodes)
        result["expected_device_nodes_present"] = bool(has_kfd and has_render)
        render_complete = result["hardware"]["dri_enumeration"]["render_enumeration_complete"]
        if args.try_rocminfo:
            result["optional_rocminfo"] = optional_rocminfo(args.output)
        result["diagnostic_status"] = ("allocated_expected_nodes_present" if has_kfd and has_render
            else "allocated_expected_nodes_missing" if render_complete
            else "allocated_device_enumeration_inconclusive")
        result["end_utc"] = utc()
        (args.output / "probe-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("diagnostic_status", "source_sha", "runner_name", "expected_device_nodes_present")}, indent=2))
        return 0 if result["expected_device_nodes_present"] else 2
    except Exception as error:
        result.update({"diagnostic_status": "allocated_probe_error", "error": str(error), "end_utc": utc()})
        (args.output / "probe-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"diagnostic_status": result["diagnostic_status"], "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
