#!/usr/bin/env python3
"""Read-only Linux build observer. Run inside the build container.
Standard library only. Does not enable compiler/link flags.
"""
import argparse, datetime, gzip, json, os, resource, shutil, signal, subprocess, sys, time
from pathlib import Path

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(path):
    try:
        return Path(path).read_text(errors='replace')
    except OSError:
        return None

def cgroup_dirs():
    result = set()
    membership = []
    for line in (read('/proc/self/cgroup') or '').splitlines():
        _, controllers, name = line.split(':', 2)
        membership.append((set(controllers.split(',')), name))
    for line in (read('/proc/self/mountinfo') or '').splitlines():
        left, right = line.split(' - ', 1)
        fields, types = left.split(), right.split()
        if types[0] not in ('cgroup', 'cgroup2'):
            continue
        root, mount = fields[3], Path(fields[4])
        for controllers, name in membership:
            if types[0] == 'cgroup' and not controllers.intersection(types[2].split(',')):
                continue
            rel = os.path.relpath(name, root)
            current = mount if rel.startswith('..') else mount / rel
            while True:
                result.add(current)
                if current == mount or current.parent == current:
                    break
                current = current.parent
    return sorted(result, key=str)

CGROUP_FILES = ('cpu.max', 'cpu.stat', 'cpu.weight', 'cpuset.cpus.effective',
                'cpuset.cpus', 'memory.max', 'memory.current', 'memory.events',
                'cpu.cfs_quota_us', 'cpu.cfs_period_us', 'cpuacct.usage',
                'memory.limit_in_bytes', 'memory.usage_in_bytes', 'io.stat',
                'cpu.pressure', 'memory.pressure', 'io.pressure')

def groups_snapshot(groups):
    return {str(p): {n: value for n in CGROUP_FILES
                     if (value := read(p / n)) is not None} for p in groups}

def run_capture(argv, output):
    start = time.monotonic()
    try:
        with output.open('wb') as f:
            code = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=120, preexec_fn=lambda: resource.setrlimit(
                                      resource.RLIMIT_FSIZE, (64*1024*1024, 64*1024*1024))).returncode
        return {'argv': argv, 'exit': code, 'wall_s': time.monotonic() - start}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'argv': argv, 'error': str(e), 'wall_s': time.monotonic() - start}

def archive_ninja(build_dir, output):
    start = time.monotonic()
    manifest = []
    roots = sorted({p.parent for p in build_dir.rglob('build.ninja')})
    for root in roots:
        dest = output / 'ninja' / root.relative_to(build_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('.ninja_log', '.ninja_deps', 'build.ninja',
                     'CMakeCache.txt', 'compile_commands.json', 'CMakeFiles/rules.ninja'):
            src = root / name
            if src.is_file():
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if src.stat().st_size <= 64 * 1024 * 1024:
                    shutil.copy2(src, target)
                else:
                    manifest.append({'skipped_large_file': str(src), 'bytes': src.stat().st_size})
        for src in root.rglob('*.rsp'):
            if any(p != root and p in src.parents for p in roots):
                continue
            target = dest / 'responses' / src.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            if src.stat().st_size <= 16 * 1024 * 1024:
                shutil.copy2(src, target)
            else:
                manifest.append({'skipped_large_response': str(src), 'bytes': src.stat().st_size})
        for tool in ('commands', 'graph', 'compdb', 'deps'):
            manifest.append(run_capture(['ninja', '-C', str(root), '-t', tool],
                                        dest / (tool + '.txt')))
    (output / 'ninja-export.json').write_text(json.dumps({
        'wall_s': time.monotonic() - start, 'commands': manifest}, indent=2))

def process_snapshot(root_pid):
    # Capture all visible processes' parent IDs first, then only the build tree.
    # /proc fields use the last ')' because comm can contain whitespace or ')'.
    records = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        raw = read(path / 'stat')
        if raw is None:
            continue
        end = raw.rfind(')')
        fields = raw[end + 2:].split()
        if len(fields) < 22:
            continue
        records[int(path.name)] = (path, raw, fields)
    wanted = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (_, _, f) in records.items():
            if int(f[1]) in wanted and pid not in wanted:
                wanted.add(pid)
                changed = True
    entries = []
    ticks = os.sysconf('SC_CLK_TCK')
    for pid in sorted(wanted):
        if pid not in records:
            continue
        path, raw, f = records[pid]
        try:
            cmd = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            status = read(path / 'status')
            entries.append({'pid': pid, 'ppid': int(f[1]), 'stat': raw,
                'status': status, 'state': f[0], 'user_s': int(f[11]) / ticks,
                'system_s': int(f[12]) / ticks, 'start_ticks': int(f[19]),
                'rss_bytes': int(f[21]) * os.sysconf('SC_PAGE_SIZE'),
                'threads': int(f[17]), 'cmdline': cmd,
                'affinity': sorted(os.sched_getaffinity(pid)),
                'io': read(path / 'io'), 'schedstat': read(path / 'schedstat'),
                'wchan': read(path / 'wchan'),
                'thread_states': {t.name: read(t / 'stat')
                    for t in sorted((path / 'task').iterdir())[:64]}})
        except (OSError, ProcessLookupError):
            continue
    return entries

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--interval', type=float, default=1.0)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.interval <= 0:
        p.error('provide command after -- and positive interval')
    args.output.mkdir(parents=True, exist_ok=False)
    groups = cgroup_dirs()
    metadata = {'utc': utc(), 'argv': command, 'cwd': str(Path.cwd()),
                'interval_s': args.interval, 'host_logical_cpus': os.cpu_count(),
                'observer_affinity': sorted(os.sched_getaffinity(0)),
                'clock_ticks': os.sysconf('SC_CLK_TCK'),
                'self_cgroup': read('/proc/self/cgroup'),
                'mountinfo': read('/proc/self/mountinfo'),
                'cpuinfo': read('/proc/cpuinfo'), 'meminfo': read('/proc/meminfo'),
                'cgroups_start': groups_snapshot(groups),
                'provenance': {k: os.environ.get(k) for k in (
                    'GITHUB_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'RUNNER_NAME',
                    'RUNNER_OS', 'KUBE_CPU_REQUEST', 'CCACHE_DISABLE',
                    'CMAKE_BUILD_PARALLEL_LEVEL', 'NINJA_STATUS')},
                'limits': {str(k): resource.getrlimit(k) for k in (
                    resource.RLIMIT_AS, resource.RLIMIT_NOFILE, resource.RLIMIT_NPROC)}}
    observer_start = resource.getrusage(resource.RUSAGE_SELF)
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    def forward(sig, _):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    sample_costs = []
    with gzip.open(args.output / 'samples.jsonl.gz', 'wt') as stream:
        while True:
            sample_start = time.monotonic()
            entries = process_snapshot(process.pid)
            stream.write(json.dumps({'utc': utc(), 'elapsed_s': time.monotonic() - start,
                'processes': entries, 'thread_detail_cap_per_process': 64,
                'cgroups': groups_snapshot(groups),
                'host_stat': read('/proc/stat'), 'host_diskstats': read('/proc/diskstats'),
                'host_pressure': {n: read('/proc/pressure/' + n)
                                  for n in ('cpu', 'memory', 'io')}}) + '\n')
            cost = time.monotonic() - sample_start
            sample_costs.append(cost)
            if process.poll() is not None:
                break
            time.sleep(max(0, args.interval - cost))
    end = time.monotonic()
    child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    observer_end = resource.getrusage(resource.RUSAGE_SELF)
    metadata.update({'exit_code': process.returncode, 'end_utc': utc(),
        'command_wall_s': end - start, 'cgroups_end': groups_snapshot(groups),
        'children_user_s': child_end.ru_utime - child_start.ru_utime,
        'children_system_s': child_end.ru_stime - child_start.ru_stime,
        'observer_user_s': observer_end.ru_utime - observer_start.ru_utime,
        'observer_system_s': observer_end.ru_stime - observer_start.ru_stime,
        'sample_cost_total_s': sum(sample_costs), 'sample_cost_max_s': max(sample_costs),
        'limitations': ['Host counters are not container allocation.',
            'Cgroup counters may include sibling processes.',
            'Hidden ancestor cgroup quota is unknown.',
            'Short-lived processes can be missed; use command_trace.py.',
            'Thread-level stat detail is capped at 64 threads per process; process CPU is complete.',
            'Child CPU includes descendants reaped by command; never sum overlapping parents.',
            'Polling wall time has up to one sample interval end detection error.',
            'Observer cost is measured; elapsed perturbation needs interleaved control runs.']})
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    archive_ninja(args.build_dir, args.output)
    return process.returncode

if __name__ == '__main__':
    sys.exit(main())
