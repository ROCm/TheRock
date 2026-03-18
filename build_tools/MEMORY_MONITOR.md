# Resource Monitor

Monitors memory, GPU, storage, and CPU during CI builds.

## Usage

```bash
# Wrap a build command
python build_tools/memory_monitor.py -- cmake --build build

# With custom interval
python build_tools/memory_monitor.py --interval 10 -- ninja -C build

# One-shot check (no command)
python build_tools/memory_monitor.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONITOR_INTERVAL` | 30 | Seconds between samples |
| `MONITOR_PHASE` | Build | Label for reporting |
| `MONITOR_STORAGE_PATH` | . | Path to monitor disk usage |

## Workflow Integration

Add to any build step by prefixing the command:

```yaml
- name: Build
  run: |
    python build_tools/memory_monitor.py --phase Build -- \
      cmake --build build --target therock-dist
```

Or use environment variables for cleaner YAML:

```yaml
- name: Build
  env:
    MONITOR_PHASE: Build
  run: python build_tools/memory_monitor.py -- cmake --build build
```

## Output

During execution:
```
[10:15:30] Mem: 24.5/32.0GB (77%)~ | Load: 12.3 | GPU0: 8.2/16.0GB | Disk: 150GB free
```

Summary at end:
```
======================================================================
Resource Summary - Build
======================================================================
Duration:     45.2 min (91 samples)
Memory:       89% peak (28.5 GB), 72% avg
CPU:          85% avg, 14.2 max load
GPU card0:    12.5/16.0 GB peak (78%)
Storage:      85 GB min free

[WARNING] Memory exceeded 75%
======================================================================
```

## What It Monitors

- **Memory**: Used/total RAM, percentage, swap usage
- **CPU**: Load average, CPU utilization percentage
- **GPU**: VRAM usage per GPU (via `rocm-smi`, if available)
- **Storage**: Free disk space on build directory
