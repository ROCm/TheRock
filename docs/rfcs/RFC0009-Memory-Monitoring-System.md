# RFC0009: Memory Monitoring System for CI/CD Builds

**Status:** Draft  
**Author:** Dezhi Liao  
**Created:** January 5, 2026  
**PR:** [#2453](https://github.com/ROCm/TheRock/pull/2453)  

---

## Summary

This RFC proposes the implementation of a comprehensive memory monitoring system for TheRock CI/CD pipeline to investigate and diagnose out-of-memory (OOM) issues occurring on self-hosted GitHub runners. The system provides real-time memory tracking, detailed logging, post-build analysis capabilities, and GitHub Actions integration to identify which build phases consume excessive memory resources.

---

## Motivation

### Problem Statement

Self-hosted GitHub runners executing TheRock builds have been experiencing out-of-memory errors, causing build failures and CI instability. Without detailed memory usage tracking across different build phases, it is difficult to:

1. **Identify the root cause** of OOM failures
2. **Determine which build phases** consume the most memory
3. **Optimize resource allocation** and parallel job configurations
4. **Proactively detect** memory pressure before failures occur
5. **Analyze historical trends** in memory consumption

### Goals

1. Implement non-invasive memory monitoring during CI builds
2. Provide real-time memory usage logging with configurable intervals
3. Generate comprehensive analysis reports identifying high-memory phases
4. Integrate seamlessly with GitHub Actions workflows for both Linux and Windows
5. Support graceful shutdown mechanisms for Windows (Linux uses native `kill -SIGINT`)
6. Enable data-driven decisions for memory optimization

---

## Detailed Design

### Architecture Overview

The solution consists of four main components:

```
┌─────────────────────────────────────────────────────────────┐
│                   GitHub Actions Workflow                    │
│  (ci.yml, ci_linux.yml, ci_windows.yml, build_*.yml)       │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ├─ Start: start_memory_monitor.sh/.ps1
                    │         ↓
                    │  ┌──────────────────────────┐
                    │  │   memory_monitor.py      │
                    │  │  - Collects metrics      │
                    │  │  - Logs to JSON          │
                    │  │  - Monitors signals      │
                    │  └──────────────────────────┘
                    │         ↓
                    ├─ Stop:  stop_memory_monitor.sh/.ps1
                    │         ↓
                    │  ┌──────────────────────────┐
                    │  │  analyze_memory_logs.py  │
                    │  │  - Analyzes logs         │
                    │  │  - Generates reports     │
                    │  │  - GitHub summaries      │
                    │  └──────────────────────────┘
                    │
                    └─ Support: graceful_shutdown.py (Windows)
```

### Component 1: Memory Monitor (`memory_monitor.py`)

**Purpose:** Core monitoring daemon that tracks system and process memory usage.

**Key Features:**
- **Real-time monitoring** with configurable intervals (default: 30 seconds)
- **Thread-safe operation** using `threading.Event` for clean shutdown
- **Cross-platform support** for Linux and Windows
- **Multiple operation modes:**
  - Background monitoring (for CI integration)
  - Command wrapping (monitor specific commands)
  - One-shot sampling

**Metrics Collected:**
```python
{
    "timestamp": "ISO 8601 format",
    "phase": "Build phase name",
    # System Memory
    "total_memory_gb": float,
    "available_memory_gb": float,
    "used_memory_gb": float,
    "memory_percent": float,
    "free_memory_gb": float,
    # Peak Tracking (cumulative)
    "peak_memory_gb": float,
    "peak_swap_gb": float,
    # Swap Memory
    "total_swap_gb": float,
    "used_swap_gb": float,
    "swap_percent": float,
    # Process Memory
    "process_memory_gb": float,
    "children_memory_gb": float,
    "total_process_memory_gb": float
}
```

**Graceful Shutdown Mechanisms:**
- **Linux/Unix:** Signal handlers for `SIGTERM`, `SIGINT`, `SIGBREAK`
- **Windows:** Stop signal file detection (polled every interval)
- **Thread coordination:** Uses `threading.Event.wait()` for responsive shutdown

**Warning Thresholds:**
- **High memory:** >75% usage
- **Critical memory:** >90% usage (likely OOM risk)
- **High swap:** >50% usage (performance degradation)

**Usage Examples:**
```bash
# Background monitoring for build phase
python build_tools/memory_monitor.py \
  --background \
  --phase "Build Phase" \
  --interval 30 \
  --log-file build/logs/memory.jsonl

# Monitor a specific command
python build_tools/memory_monitor.py \
  --phase "Configure" \
  -- cmake -B build -S .

# One-shot sample
python build_tools/memory_monitor.py
```

### Component 2: Analysis Tool (`analyze_memory_logs.py`)

**Purpose:** Post-build analysis of memory logs to identify problematic phases.

**Key Features:**
- Parses JSON/JSONL log files
- Aggregates statistics by build phase
- Generates severity classifications
- Produces both console and GitHub Action summaries

**Analysis Metrics:**
- Average, min, max memory percentages per phase
- Peak memory usage in GB
- Swap usage patterns
- Duration and sample count
- Time range analysis

**Severity Classification:**
```python
CRITICAL: >= 95% memory usage
HIGH:     >= 90% memory usage
MEDIUM:   >= 75% memory usage
LOW:      < 75% memory usage
```

**Report Formats:**
1. **Console Report:** Detailed text-based analysis with tables
2. **GitHub Summary:** Markdown tables with emoji indicators
3. **File Output:** Save reports for historical analysis

**Usage Examples:**
```bash
# Analyze logs in default location
python build_tools/analyze_memory_logs.py

# Custom log directory with detailed report
python build_tools/analyze_memory_logs.py \
  --log-dir build/logs \
  --detailed

# Generate GitHub Actions summary
python build_tools/analyze_memory_logs.py \
  --github-summary

# Save report to file
python build_tools/analyze_memory_logs.py \
  --output memory_report.txt
```

### Component 3: Graceful Shutdown Utility (`graceful_shutdown.py`)

**Purpose:** Windows-specific process termination with cleanup support.

**Why Windows-Specific:**
- **Linux/Unix:** Native signal support allows using `kill -SIGINT <PID>` for graceful shutdown
- **Windows:** Limited signal support requires alternative mechanism (stop signal files)

**Key Features:**
- Creates stop signal file for Windows processes to detect
- Waits for graceful exit before force-killing
- Configurable timeout (default: 10 seconds)
- Automatic cleanup of signal files
- Falls back to SIGTERM then force kill if needed

**Shutdown Flow:**
```
1. Create stop signal file (if specified)
2. Wait for process to detect file and exit gracefully
3. Send SIGTERM (fallback if file mechanism fails)
4. Wait for graceful shutdown
5. Force kill if timeout exceeded
```

**Usage Examples:**
```bash
# Windows: Graceful shutdown with stop signal file
python build_tools/graceful_shutdown.py 12345 \
  --stop-signal-file build/logs/stop.signal \
  --timeout 10

# Linux: Use native signals instead (no utility needed)
kill -SIGINT <PID>
```

### Component 4: GitHub Actions Integration

#### Workflow Changes

**New Input Parameter:**
```yaml
monitor_memory:
  type: boolean
  description: "If enabled, memory monitoring will be performed during the build"
  default: false
```

**Integration Points:**
1. **Linux Workflows** (`build_portable_linux_artifacts.yml`, `ci_linux.yml`)
   - Start script: `start_memory_monitor.sh`
   - Stop script: `stop_memory_monitor.sh`
   - Uses SIGINT for graceful shutdown

2. **Windows Workflows** (`build_windows_artifacts.yml`, `ci_windows.yml`)
   - Start script: `start_memory_monitor.ps1`
   - Stop script: `stop_memory_monitor.ps1`
   - Uses stop signal file mechanism

**Environment Variables:**
```yaml
BUILD_DIR: build
JOB_NAME: ${{ github.job }}
PHASE: "Build Phase"
```

**Modified Build Steps:**
```yaml
- name: Start memory monitoring
  if: ${{ inputs.monitor_memory }}
  run: bash build_tools/github_actions/start_memory_monitor.sh

- name: Build therock-archives and therock-dist
  run: cmake --build ${{ env.BUILD_DIR }} --target therock-archives therock-dist

- name: Stop memory monitoring
  if: ${{ always() && inputs.monitor_memory }}
  run: bash build_tools/github_actions/stop_memory_monitor.sh
```

#### Linux Shell Scripts

**`start_memory_monitor.sh`:**
- Starts Python monitor in background
- Captures and exports PID
- Creates log directory structure
- Returns PID for cleanup

**`stop_memory_monitor.sh`:**
- Sends SIGINT to monitor process (native Linux signal handling)
- Waits for graceful exit (monitor's signal handler triggers cleanup)
- Force kills if necessary (fallback)
- Displays monitor output
- No need for graceful_shutdown.py utility

#### Windows PowerShell Scripts

**`start_memory_monitor.ps1`:**
- Conditional execution based on `$MonitorMemory` parameter
- Starts monitor with stop signal file support
- Writes PID to file for later cleanup
- Redirects output to log file

**`stop_memory_monitor.ps1`:**
- Reads PID from file
- Stops process forcefully (Windows signal limitations)
- Uses stop signal file mechanism for graceful shutdown
- Displays monitor output
- Handles missing files gracefully
- May use graceful_shutdown.py utility for better cleanup in future

### Data Flow

```
┌───────────────────────┐
│  GitHub Workflow      │
│  (Build Phase Start)  │
└──────────┬────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  start_memory_monitor.sh/.ps1        │
│  - Creates log directory             │
│  - Starts background monitor         │
│  - Captures PID                      │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  memory_monitor.py (Background)      │
│  ┌────────────────────────────────┐  │
│  │  Loop (every 30s):             │  │
│  │  1. Collect system metrics     │  │
│  │  2. Track peak usage           │  │
│  │  3. Write JSON log             │  │
│  │  4. Print warnings             │  │
│  │  5. Check stop signal          │  │
│  └────────────────────────────────┘  │
└──────────┬───────────────────────────┘
           │ (Continuous)
           │
           ▼
┌──────────────────────────────────────┐
│  build/logs/*.jsonl                  │
│  (Line-delimited JSON)               │
└──────────┬───────────────────────────┘
           │
           │ (Build Phase Complete)
           │
           ▼
┌──────────────────────────────────────┐
│  stop_memory_monitor.sh/.ps1         │
│  - Sends stop signal                 │
│  - Waits for clean exit              │
│  - Displays output                   │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  analyze_memory_logs.py              │
│  - Parse all log files               │
│  - Aggregate by phase                │
│  - Calculate statistics              │
│  - Generate report                   │
│  - Write GitHub summary              │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  GitHub Actions Step Summary         │
│  - Markdown tables                   │
│  - Severity indicators               │
│  - Actionable warnings               │
└──────────────────────────────────────┘
```

### Log File Format

**Location:** `$BUILD_DIR/logs/build_memory_log_${JOB_NAME}.jsonl`

**Format:** Line-delimited JSON (JSONL)

**Example:**
```json
{"timestamp": "2026-01-05T10:00:00", "phase": "Build Phase", "memory_percent": 45.2, "used_memory_gb": 28.5, "swap_percent": 5.1, ...}
{"timestamp": "2026-01-05T10:00:30", "phase": "Build Phase", "memory_percent": 67.8, "used_memory_gb": 42.7, "swap_percent": 12.3, ...}
{"timestamp": "2026-01-05T10:01:00", "phase": "Build Phase", "memory_percent": 89.4, "used_memory_gb": 56.3, "swap_percent": 28.7, ...}
```

### Testing Strategy

**Unit Tests:** (`build_tools/tests/`)

1. **`memory_monitor_test.py`:**
   - Stats collection validation
   - Monitoring loop functionality
   - Thread-safe stop event mechanism
   - Responsive shutdown timing
   - Stop signal file detection
   - Log file writing

2. **`graceful_shutdown_test.py`:**
   - Process termination with stop signal file
   - Timeout handling
   - Summary output validation
   - Cross-platform compatibility (Windows-specific)

**Test Coverage:**
- Memory statistics collection accuracy
- Threading behavior and shutdown responsiveness
- Stop signal file detection (Windows compatibility)
- Log file I/O and JSON formatting
- Analysis script output correctness

---

## Implementation Details

### Dependencies

**New Dependency:**
```python
psutil>=5.9.0  # Cross-platform system and process utilities
```

**Added to `requirements.txt`:**
```diff
 CppHeaderParser>=2.7.4
 build>=1.2.2
 meson>=1.7.0
+psutil>=5.9.0
 python-magic>=0.4.27; platform_system != "Windows"
 PyYAML==6.0.2
 pyzstd>=0.16.0
```

### File Structure

```
build_tools/
├── memory_monitor.py                  # Core monitoring daemon
├── analyze_memory_logs.py             # Post-build analysis tool
├── graceful_shutdown.py               # Windows-specific process termination utility
├── github_actions/
│   ├── start_memory_monitor.sh        # Linux start script
│   ├── stop_memory_monitor.sh         # Linux stop script (uses kill -SIGINT)
│   ├── start_memory_monitor.ps1       # Windows start script
│   └── stop_memory_monitor.ps1        # Windows stop script (uses stop signal file)
└── tests/
    ├── memory_monitor_test.py         # Monitor unit tests
    └── graceful_shutdown_test.py      # Shutdown utility tests (Windows-specific)

.github/workflows/
├── ci.yml                              # Main CI orchestrator
├── ci_linux.yml                        # Linux CI workflow
├── ci_windows.yml                      # Windows CI workflow
├── build_portable_linux_artifacts.yml  # Linux build workflow
└── build_windows_artifacts.yml         # Windows build workflow

build/logs/                             # Generated during builds
├── build_memory_log_*.jsonl           # Memory sample logs
├── monitor_output_*.txt               # Monitor stdout/stderr
├── monitor_pid_*.txt                  # PID files
└── stop_monitor_*.signal              # Stop signal files (Windows only)
```

### Configuration

**Environment Variables:**
- `MEMORY_MONITOR_INTERVAL`: Override default monitoring interval (seconds)
- `MEMORY_MONITOR_LOG_FILE`: Path to write detailed memory logs
- `BUILD_DIR`: Build directory location (default: `build`)
- `JOB_NAME`: GitHub job name for log file naming
- `PHASE`: Build phase name for categorization
- `GITHUB_STEP_SUMMARY`: GitHub Actions summary file path (auto-set)

**Workflow Inputs:**
```yaml
inputs:
  monitor_memory:
    type: boolean
    default: false
    description: "Enable memory monitoring during build"
```

---

## Drawbacks and Limitations

### Performance Impact

1. **Monitoring Overhead:**
   - Python process runs continuously during builds
   - Memory/CPU overhead: ~20-50 MB RAM, <1% CPU (at 30s intervals)
   - Impact is negligible compared to build resource consumption

2. **I/O Overhead:**
   - JSON logging writes (~200 bytes per sample)
   - At 30s intervals, ~6 KB/min per job
   - Minimal disk I/O impact

### Platform Limitations

1. **Windows Signal Handling:**
   - Windows has limited signal support compared to Linux/Unix
   - Cannot use `kill -SIGINT` like on Linux
   - Requires stop signal file polling as alternative mechanism
   - Slight delay in detecting stop requests (<1 interval)
   - `graceful_shutdown.py` utility provides Windows-specific graceful termination

2. **Process Monitoring Accuracy:**
   - Child process enumeration may miss short-lived processes
   - Access denied errors for system processes (handled gracefully)
   - Peak memory tracking is cumulative within monitoring session

### Operational Considerations

1. **Manual Enablement:**
   - Monitoring is opt-in via `monitor_memory` input
   - Not enabled by default to avoid unnecessary overhead on all builds
   - Must be explicitly enabled when investigating OOM issues

2. **Log Retention:**
   - Logs are stored in build directory
   - No automatic cleanup/archival mechanism
   - Users must manually manage log files for historical analysis

3. **Analysis Timing:**
   - Analysis script must be run separately post-build
   - No real-time alerts during build (only post-mortem)
   - Could be improved with real-time dashboard integration

---

## Alternatives Considered

### 1. Using Existing Monitoring Tools

**Option:** Integrate Prometheus, Datadog, or CloudWatch

**Pros:**
- Enterprise-grade monitoring
- Real-time dashboards and alerts
- Historical data retention
- Advanced querying capabilities

**Cons:**
- External dependencies and infrastructure requirements
- Cost (for cloud services)
- Complexity for simple OOM investigation
- May not have access on self-hosted runners
- Overkill for targeted debugging

**Decision:** Implement custom solution for immediate needs, with potential future integration

### 2. Kernel-Level Monitoring

**Option:** Use `perf`, `eBPF`, or kernel tracing

**Pros:**
- Zero-overhead profiling
- Detailed system-level insights
- Accurate attribution to processes
- Native signal handling on Linux

**Cons:**
- Requires elevated privileges
- Platform-specific (Linux only)
- Complex setup and analysis
- Not portable to Windows
- Overkill for memory investigation

**Decision:** Use user-space monitoring with psutil for cross-platform compatibility. Linux benefits from native signal handling (`kill -SIGINT`), while Windows uses stop signal files.

### 3. Build System Integration

**Option:** Integrate monitoring directly into CMake/build system

**Pros:**
- Tighter integration with build phases
- Automatic phase detection
- No separate script orchestration

**Cons:**
- Requires modifying build system
- Platform-specific implementation
- Harder to maintain and update
- Breaks separation of concerns

**Decision:** Keep monitoring external to build system for flexibility

### 4. GitHub Actions Native Monitoring

**Option:** Use GitHub Actions metrics and runners API

**Pros:**
- No custom code required
- Built-in support
- Integrated with GitHub UI

**Cons:**
- Limited memory metrics available
- No per-phase granularity
- Self-hosted runners have limited API support
- Cannot control sampling intervals

**Decision:** Implement custom monitoring for detailed control

---

## Migration and Rollout Plan

### Phase 1: Initial Deployment

- [ ] Implement core monitoring components with unit tests
- [ ] Add GitHub Actions integration
- [ ] Update documentation

### Phase 2: Testing and Validation (Current)

- [ ] Enable monitoring on select CI jobs
- [ ] Collect baseline memory usage data
- [ ] Validate cross-platform functionality
- [ ] Identify any integration issues
- [ ] Gather team feedback

### Phase 3: Iterative Improvement

**Short-term (1-2 weeks):**
- [ ] Analyze collected data to identify OOM root causes
- [ ] Optimize high-memory build phases
- [ ] Adjust monitoring intervals based on findings
- [ ] Add automated analysis to CI reports

**Medium-term (1-2 months):**
- [ ] Implement memory usage trends/graphs
- [ ] Add alerting for critical memory conditions
- [ ] Create dashboard for historical analysis
- [ ] Integrate with failure notifications

### Rollout Strategy

1. **Opt-in by default** - No changes to existing workflows unless explicitly enabled
2. **Gradual enablement** - Enable on subset of builds first
3. **Monitor for issues** - Watch for script failures or monitoring overhead
4. **Document findings** - Share insights with team
5. **Optimize based on data** - Make targeted improvements to reduce memory consumption

---

## Success Criteria

### Immediate Goals

1. **System Implementation:**
   - Memory monitoring system fully functional on Linux and Windows
   - Zero impact on builds when disabled
   - Minimal overhead when enabled (<1% CPU, <50 MB RAM)

2. **Data Collection:**
   - Successfully collect memory data across multiple build phases
   - Identify at least 3 high-memory build phases
   - Generate actionable analysis reports

3. **Root Cause Identification:**
   - Pinpoint specific build steps causing OOM errors
   - Quantify memory consumption by phase
   - Determine if issue is configuration or code-related

### Long-term Success

1. **OOM Reduction:**
   - Reduce OOM failures by 50% within 3 month
   - Eliminate OOM failures on builds with identified fixes

2. **Resource Optimization:**
   - Optimize runner memory allocation based on data
   - Reduce unnecessary memory usage in high-consumption phases
   - Improve parallel job configuration

3. **Developer Experience:**
   - Faster feedback on memory-related issues
   - Proactive detection before production failures
   - Better understanding of build resource requirements

---

## Security Considerations

### Data Privacy

- **No sensitive data** in memory logs (only metrics, no process contents)
- **Logs stored locally** in build directory (not transmitted)
- **Automatic cleanup** possible via build artifact retention policies

### Access Control

- **Read-only monitoring** - System observes but doesn't modify processes
- **Requires Python execution** - Already trusted in CI environment
- **No network access** - All operations are local

### Process Safety

- **Graceful degradation** - Continues on error, doesn't break builds
- **Timeouts and limits** - Prevents infinite waits or hangs
- **Signal handling** - Properly responds to termination requests

---

## Documentation Updates

### User Documentation

**README sections to add:**
- Memory monitoring system overview
- How to enable monitoring in CI
- Interpreting memory reports
- Troubleshooting OOM issues

**Developer Guide:**
- Memory monitoring architecture
- Adding new metrics
- Extending analysis capabilities
- Testing monitoring code

### Operational Documentation

**Runbook:**
- Enabling memory monitoring for investigation
- Analyzing memory logs
- Common OOM scenarios and fixes
- Performance tuning guide

---

## Open Questions and Future Work

### Current Open Questions

1. **Optimal Monitoring Interval:**
   - Is 30 seconds appropriate for all scenarios?
   - Should we make it adaptive based on memory pressure?
   - Trade-off between granularity and overhead?

2. **Log Retention:**
   - How long should logs be kept?
   - Should we implement automatic cleanup?
   - Archive to external storage for historical analysis?

3. **Alert Thresholds:**
   - Are current thresholds (75%, 90%) appropriate?
   - Should they be configurable per workflow?
   - Different thresholds for different runner sizes?

---

## Appendix A: Example Output

### Memory Monitor Summary
```
================================================================================
[SUMMARY] Memory Monitoring Summary - Phase: Build Phase
================================================================================
Duration: 1247.3 seconds
Samples collected: 42

Memory Usage:
  Average: 67.3%
  Peak: 89.4% (56.30 GB)

Swap Usage:
  Average: 15.2%
  Peak: 28.7% (4.58 GB)

[WARNING] Memory usage exceeded 75% during this phase.
================================================================================
```

### Analysis Report
```
================================================================================
MEMORY USAGE ANALYSIS REPORT
================================================================================

Total phases analyzed: 3

SUMMARY (Sorted by Peak Memory Usage)
--------------------------------------------------------------------------------
Phase                                    Peak         Avg          Severity  
--------------------------------------------------------------------------------
[!]  therock-archives Build              89.4%        67.3%        HIGH      
[~]  CMake Configure                     76.2%        54.1%        MEDIUM    
[OK] Test Packaging                      45.8%        38.2%        LOW       
--------------------------------------------------------------------------------

KEY FINDINGS
================================================================================

[!] PHASES WITH HIGH MEMORY USAGE:
  - therock-archives Build: 89.4% peak
    Consider reducing parallel jobs or increasing available memory

================================================================================
```

### GitHub Actions Summary (Markdown)
```markdown
## [WARNING] Memory Stats: Build Phase

| Metric | Value |
|:-------|------:|
| **Duration** | 1247.3s |
| **Samples Collected** | 42 |
| **Average Memory** | 67.3% |
| **Peak Memory** | 89.4% (56.30 GB) |
| **Average Swap** | 15.2% |
| **Peak Swap** | 28.7% (4.58 GB) |

> [!WARNING]
> Memory usage exceeded 75% during this phase.

> [!WARNING]
> Significant swap usage detected (28.7%). Consider increasing available memory or reducing parallel jobs.
```

---

## Appendix B: Troubleshooting Guide

### Common Issues

**Issue:** Memory monitor doesn't start
- **Check:** Python and psutil installed
- **Check:** Log directory is writable
- **Solution:** Verify dependencies in requirements.txt

**Issue:** Monitor doesn't stop gracefully on Windows
- **Cause:** Windows has limited signal support (no SIGINT like Linux)
- **Solution:** Ensure stop signal file mechanism is working properly
- **Note:** Linux uses native `kill -SIGINT` which works reliably
- **Workaround:** Increase timeout in stop script or use graceful_shutdown.py utility

**Issue:** Missing memory data in logs
- **Check:** Monitor process is running
- **Check:** Log file permissions
- **Solution:** Verify background process with PID file

**Issue:** High monitoring overhead
- **Cause:** Interval too short
- **Solution:** Increase interval to 60+ seconds
- **Note:** 30s is optimal for most cases

---

## Appendix C: Metrics Dictionary

| Metric | Unit | Description |
|--------|------|-------------|
| `total_memory_gb` | GB | Total physical RAM installed |
| `available_memory_gb` | GB | Memory available for new processes |
| `used_memory_gb` | GB | Memory currently in use |
| `memory_percent` | % | Percentage of total memory used |
| `free_memory_gb` | GB | Completely unused memory |
| `peak_memory_gb` | GB | Highest memory usage observed (cumulative) |
| `peak_swap_gb` | GB | Highest swap usage observed (cumulative) |
| `total_swap_gb` | GB | Total swap space configured |
| `used_swap_gb` | GB | Swap space currently in use |
| `swap_percent` | % | Percentage of total swap used |
| `process_memory_gb` | GB | Memory used by monitor process itself |
| `children_memory_gb` | GB | Memory used by child processes |
| `total_process_memory_gb` | GB | Combined memory of process tree |

---

**END OF RFC**
