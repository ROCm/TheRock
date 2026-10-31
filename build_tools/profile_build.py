#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Sample a build process tree: --output build/logs/profile -- cmake --build build.

Requires psutil. Records commands/cwd, never environment variables. Do not pass
secrets in command arguments. Sampled CPU misses short-lived processes; POSIX
waited-child CPU is reported separately, without adding sampled CPU to it.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import psutil


def windows_cpu_metadata():
    if sys.platform != "win32":
        return None
    import ctypes as c
    from ctypes import wintypes as w

    k = c.WinDLL("kernel32", use_last_error=True)
    handle = w.HANDLE
    k.GetCurrentProcess.restype = handle
    k.GetCurrentThread.restype = handle
    process = k.GetCurrentProcess()

    class GroupAffinity(c.Structure):
        _fields_ = [("Mask", c.c_size_t), ("Group", w.WORD), ("Reserved", w.WORD * 3)]

    def error():
        return {"winerror": c.get_last_error()}

    def group_value(g):
        return {
            "group": g.Group,
            "mask_hex": hex(g.Mask),
            "mask_set_bits": int(g.Mask).bit_count(),
        }

    result = {
        "interpretation": "Raw group/job facts only; no effective CPU denominator inferred.",
        "legacy_affinity_note": "On Server2022/Windows11, legacy masks can describe the primary group while default affinities span groups.",
        "job_scope_note": "QueryInformationJobObject(NULL) sees the immediate job; ancestor job limits may remain hidden.",
    }
    try:
        k.GetActiveProcessorGroupCount.restype = w.WORD
        k.GetActiveProcessorCount.argtypes = [w.WORD]
        k.GetActiveProcessorCount.restype = w.DWORD
        count = k.GetActiveProcessorGroupCount()
        result["active_processors_per_group"] = [
            {"group": g, "count": k.GetActiveProcessorCount(g)} for g in range(count)
        ]
        k.GetProcessGroupAffinity.argtypes = [
            handle,
            c.POINTER(w.WORD),
            c.POINTER(w.WORD),
        ]
        n = w.WORD(max(64, count))
        groups = (w.WORD * n.value)()
        result["process_group_array"] = (
            list(groups)[: n.value]
            if k.GetProcessGroupAffinity(process, c.byref(n), groups)
            else error()
        )
        k.GetThreadGroupAffinity.argtypes = [handle, c.POINTER(GroupAffinity)]
        g = GroupAffinity()
        result["calling_thread_primary_group_affinity"] = (
            group_value(g)
            if k.GetThreadGroupAffinity(k.GetCurrentThread(), c.byref(g))
            else error()
        )
        k.GetProcessDefaultCpuSets.argtypes = [
            handle,
            c.POINTER(w.ULONG),
            w.ULONG,
            c.POINTER(w.ULONG),
        ]
        needed = w.ULONG()
        initial_cpu_sets_ok = k.GetProcessDefaultCpuSets(
            process, None, 0, c.byref(needed)
        )
        if needed.value:
            ids = (w.ULONG * needed.value)()
            result["process_default_cpu_set_ids"] = (
                list(ids)
                if k.GetProcessDefaultCpuSets(process, ids, needed, c.byref(needed))
                else error()
            )
        else:
            result["process_default_cpu_set_ids"] = (
                [] if initial_cpu_sets_ok else error()
            )
        k.IsProcessInJob.argtypes = [handle, handle, c.POINTER(w.BOOL)]
        in_job = w.BOOL()
        ok = k.IsProcessInJob(process, None, c.byref(in_job))
        result["in_any_job"] = bool(in_job.value) if ok else error()
        k.QueryInformationJobObject.argtypes = [
            handle,
            c.c_int,
            c.c_void_p,
            w.DWORD,
            c.POINTER(w.DWORD),
        ]
        if ok and in_job.value:

            class CpuRate(c.Structure):
                _fields_ = [("ControlFlags", w.DWORD), ("UnionValue", w.DWORD)]

            cpu, size = CpuRate(), w.DWORD()
            if k.QueryInformationJobObject(
                None, 15, c.byref(cpu), c.sizeof(cpu), c.byref(size)
            ):
                result["immediate_job_cpu_rate"] = {
                    "control_flags_hex": hex(cpu.ControlFlags),
                    "raw_union_value": cpu.UnionValue,
                    "enabled": bool(cpu.ControlFlags & 1),
                    "hard_cap": bool(cpu.ControlFlags & 4),
                    "weight_based": bool(cpu.ControlFlags & 2),
                    "min_max_rate": bool(cpu.ControlFlags & 16),
                }
            else:
                result["immediate_job_cpu_rate"] = error()
            masks = (GroupAffinity * max(64, count))()
            if k.QueryInformationJobObject(
                None, 14, masks, c.sizeof(masks), c.byref(size)
            ):
                result["immediate_job_group_affinity"] = [
                    group_value(masks[i])
                    for i in range(size.value // c.sizeof(GroupAffinity))
                ]
            else:
                result["immediate_job_group_affinity"] = error()
    except (AttributeError, OSError) as exc:
        result["collection_error"] = repr(exc)
    return result


def read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def cgroups():
    membership, mounts = read("/proc/self/cgroup"), read("/proc/self/mountinfo")
    if not membership or not mounts:
        return [], {"status": "unavailable_or_non_linux"}
    for line in membership.splitlines():
        h, controllers, member = line.split(":", 2)
        if h != "0" or controllers:
            continue
        for line in mounts.splitlines():
            left, _, right = line.partition(" - ")
            if right.split()[0] != "cgroup2":
                continue
            f = left.split()
            root, mount = Path(f[3].replace("\\040", " ")), Path(
                f[4].replace("\\040", " ")
            )
            try:
                relative = Path(member).relative_to(root)
            except ValueError:
                if member == "/":
                    relative = Path(".")
                else:
                    continue
            current, paths = mount / relative, []
            while current == mount or mount in current.parents:
                paths.append(current)
                if current == mount:
                    return paths, {
                        "status": "v2_visible_hierarchy",
                        "membership": member,
                        "mount_root": str(root),
                        "mountpoint": str(mount),
                    }
                current = current.parent
    return [], {"status": "v1_or_unresolved", "membership": membership}


GROUP_FILES = (
    "cpu.max",
    "cpu.stat",
    "cpu.pressure",
    "cpuset.cpus.effective",
    "memory.current",
    "memory.max",
    "memory.events",
    "memory.pressure",
    "io.stat",
    "io.pressure",
)


def snapshot(paths):
    return {
        str(p): {n: v for n in GROUP_FILES if (v := read(p / n)) is not None}
        for p in paths
    }


def capacity(affinity, groups):
    bounds = [len(affinity)] if affinity is not None else []
    for g in groups.values():
        spec = g.get("cpuset.cpus.effective")
        if spec:
            bounds.append(
                sum(
                    int(b) - int(a) + 1 if sep else 1
                    for part in spec.split(",")
                    for a, sep, b in [part.partition("-")]
                )
            )
        maximum = g.get("cpu.max", "").split()
        if len(maximum) == 2 and maximum[0] != "max":
            bounds.append(float(maximum[0]) / float(maximum[1]))
    return min(bounds) if bounds else None


def child_cpu():
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        return {"user_s": r.ru_utime, "system_s": r.ru_stime}
    except (ImportError, AttributeError):
        return None


def stat(text):
    return {
        p[0]: int(p[1])
        for line in (text or "").splitlines()
        if len(p := line.split()) == 2 and p[1].isdigit()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd or args.interval < 0.1:
        parser.error("provide command after -- and interval >=0.1")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stream = (out / "samples.jsonl").open("w", encoding="utf-8")

    def emit(event, **data):
        stream.write(
            json.dumps(
                dict(
                    event=event,
                    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    monotonic_s=time.monotonic(),
                    **data,
                )
            )
            + "\n"
        )
        stream.flush()

    try:
        affinity = psutil.Process().cpu_affinity()
    except (AttributeError, psutil.Error):
        affinity = None
    paths, info = cgroups()
    before_groups, before_cpu = snapshot(paths), child_cpu()
    meta = dict(
        command=cmd,
        cwd=os.getcwd(),
        platform=platform.platform(),
        psutil=psutil.__version__,
        visible_logical_cpus=psutil.cpu_count(),
        affinity=affinity,
        cgroup=info,
        cgroup_initial=before_groups,
        interval_s=args.interval,
        effective_capacity_visible_upper_bound_cpus=(
            None if sys.platform == "win32" else capacity(affinity, before_groups)
        ),
        windows_cpu_groups_and_job=windows_cpu_metadata(),
        windows_sample_policy="Per-process status and thread count omitted: psutil queries are expensive and do not identify runnable work.",
        limitations=[
            "Sampled process CPU misses processes exiting between observations.",
            "Cgroup/system counters can include other work in their scope.",
            "Namespace-hidden ancestor limits are unknown; cgroup v1 is not normalized.",
            "Windows job-object CPU limits and processor groups require ETW/host verification.",
            "RSS summed across processes double-counts shared memory.",
            "Sampler own CPU does not measure total perturbation; matched A/B is required.",
        ],
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    own_start, start = time.process_time(), time.monotonic()
    try:
        proc = subprocess.Popen(cmd)
    except OSError as error:
        emit("launch_error", error=str(error))
        stream.close()
        return 127
    try:
        root = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        root = None
    stop, known, ledger = threading.Event(), set(), {}
    overhead = {"cpu_s": 0.0, "active_wall_s": 0.0, "samples": 0, "sections": {}}

    def record_section(name, cpu, wall):
        item = overhead["sections"].setdefault(
            name, {"cpu_s": 0.0, "wall_s": 0.0, "calls": 0}
        )
        item["cpu_s"] += cpu
        item["wall_s"] += wall
        item["calls"] += 1

    def measure(name, function):
        cpu, wall = time.thread_time(), time.monotonic()
        try:
            return function()
        finally:
            record_section(name, time.thread_time() - cpu, time.monotonic() - wall)

    def sample():
        c0, w0 = time.thread_time(), time.monotonic()
        try:
            processes = (
                measure(
                    "process_tree_discovery",
                    lambda: [root] + root.children(recursive=True),
                )
                if root
                else []
            )
        except psutil.Error:
            processes = []
        rows = []
        process_cpu, process_wall = time.thread_time(), time.monotonic()
        for p in processes:
            try:
                identity = f"{p.pid}:{p.create_time():.6f}"
                with p.oneshot():
                    if identity not in known:
                        try:
                            cwd = p.cwd()
                        except psutil.Error:
                            cwd = None
                        emit(
                            "process",
                            identity=identity,
                            pid=p.pid,
                            ppid=p.ppid(),
                            create_time=p.create_time(),
                            command=p.cmdline(),
                            cwd=cwd,
                            name=p.name(),
                        )
                        known.add(identity)
                    t = p.cpu_times()
                    cpu = (
                        t.user + t.system
                    )  # Exclude children_*: descendants counted separately.
                    previous = ledger.get(identity, 0.0)
                    ledger[identity] = max(cpu, previous)
                    try:
                        io = p.io_counters()._asdict()
                    except (AttributeError, psutil.Error):
                        io = None
                    rows.append(
                        dict(
                            identity=identity,
                            cpu_user_s=t.user,
                            cpu_system_s=t.system,
                            cpu_delta_s=max(0.0, cpu - previous),
                            threads=(
                                None if sys.platform == "win32" else p.num_threads()
                            ),
                            status=(
                                "not_collected_on_windows"
                                if sys.platform == "win32"
                                else p.status()
                            ),
                            rss_bytes=p.memory_info().rss,
                            io=io,
                        )
                    )
            except psutil.Error:
                continue
        record_section(
            "process_registration_and_metrics",
            time.thread_time() - process_cpu,
            time.monotonic() - process_wall,
        )
        disk = measure("system_disk_io", psutil.disk_io_counters)
        emit(
            "sample",
            processes=rows,
            cgroups=measure("cgroup_snapshot", lambda: snapshot(paths)),
            system_cpu_times=measure("system_cpu_times", psutil.cpu_times)._asdict(),
            system_memory=measure("system_memory", psutil.virtual_memory)._asdict(),
            system_disk_io=disk._asdict() if disk else None,
        )
        overhead["cpu_s"] += time.thread_time() - c0
        overhead["active_wall_s"] += time.monotonic() - w0
        overhead["samples"] += 1

    def loop():
        while not stop.is_set():
            cycle = time.monotonic()
            try:
                sample()
            except Exception as error:
                emit("sampling_error", error=repr(error))
            stop.wait(max(0.0, args.interval - (time.monotonic() - cycle)))

    sampler = threading.Thread(target=loop, daemon=True)
    sampler.start()
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        code = proc.wait()
    finally:
        stop.set()
        sampler.join()
    after_cpu, final_groups = child_cpu(), snapshot(paths)
    exact = (
        {k: after_cpu[k] - before_cpu[k] for k in before_cpu}
        if before_cpu is not None
        else None
    )
    deltas = {}
    for path, values in final_groups.items():
        b, a = stat(before_groups.get(path, {}).get("cpu.stat")), stat(
            values.get("cpu.stat")
        )
        deltas[path] = {k: v - b[k] for k, v in a.items() if k in b}
    result = dict(
        exit_code=code,
        wall_s=time.monotonic() - start,
        observed_process_identities=len(known),
        sampled_process_cpu_lower_bound_s=sum(ledger.values()),
        posix_waited_child_cpu=exact,
        posix_waited_child_cpu_scope="Terminated/waited descendants; escaped or surviving processes excluded.",
        cgroup_cpu_stat_deltas=deltas,
        cgroup_final=final_groups,
        sampler=overhead,
        wrapper_process_cpu_s=time.process_time() - own_start,
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    emit("complete", **result)
    stream.close()
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
