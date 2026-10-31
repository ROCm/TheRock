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
        effective_capacity_visible_upper_bound_cpus=capacity(affinity, before_groups),
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
    overhead = {"cpu_s": 0.0, "active_wall_s": 0.0, "samples": 0}

    def sample():
        c0, w0 = time.thread_time(), time.monotonic()
        try:
            processes = [root] + root.children(recursive=True) if root else []
        except psutil.Error:
            processes = []
        rows = []
        for p in processes:
            try:
                identity = f"{p.pid}:{p.create_time():.6f}"
                with p.oneshot():
                    if identity not in known:
                        known.add(identity)
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
                            threads=p.num_threads(),
                            status=p.status(),
                            rss_bytes=p.memory_info().rss,
                            io=io,
                        )
                    )
            except psutil.Error:
                continue
        disk = psutil.disk_io_counters()
        emit(
            "sample",
            processes=rows,
            cgroups=snapshot(paths),
            system_cpu_times=psutil.cpu_times()._asdict(),
            system_memory=psutil.virtual_memory()._asdict(),
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
