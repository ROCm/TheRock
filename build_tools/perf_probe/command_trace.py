#!/usr/bin/env python3
"""CMake compiler/linker launcher. Set THEROCK_PERF_COMMAND_DIR absolute path.
Preserve existing launcher by placing it after this script in launcher list.
Every command gets a separate JSON file to avoid concurrent append corruption.
"""
import datetime, hashlib, json, os, resource, signal, subprocess, sys, time, uuid
from pathlib import Path

def main():
    command = sys.argv[1:]
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        raise SystemExit('command_trace.py requires a command')
    output = Path(os.environ['THEROCK_PERF_COMMAND_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    token = f'{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    responses = []
    for value in command:
        if value.startswith('@') and Path(value[1:]).is_file():
            path = Path(value[1:])
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if len(data) <= 16 * 1024 * 1024:
                (output / (digest + '.rsp')).write_bytes(data)
            responses.append({'path': str(path), 'sha256': digest, 'bytes': len(data), 'content_saved': len(data) <= 16 * 1024 * 1024})
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic_ns()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    process = subprocess.Popen(command)
    def forward(sig, _):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    code = process.wait()
    end = time.monotonic_ns()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {'argv': command, 'cwd': str(Path.cwd()), 'utc': started,
              'start_monotonic_ns': start, 'end_monotonic_ns': end,
              'wall_s': (end - start) / 1e9, 'exit_code': code,
              'user_s': after.ru_utime - before.ru_utime,
              'system_s': after.ru_stime - before.ru_stime,
              'max_rss_kib': after.ru_maxrss, 'responses': responses,
              'voluntary_context_switches': after.ru_nvcsw - before.ru_nvcsw,
              'involuntary_context_switches': after.ru_nivcsw - before.ru_nivcsw}
    (output / (token + '.json')).write_text(json.dumps(record, indent=2))
    return code if code >= 0 else 128 - code

if __name__ == '__main__':
    sys.exit(main())
