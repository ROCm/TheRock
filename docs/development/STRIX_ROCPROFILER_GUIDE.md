# ROCProfiler-v3 and RPD Guide for Strix

## Overview

**rocprofiler-v3** is AMD's profiling tool for capturing GPU performance data.  
**rocmprofilerdata (RPD)** is the analysis tool for viewing and analyzing the captured data.

**Relationship:**
```
rocprofiler-v3 (captures data) → .rpd files → RPD tool (analyzes data)
```

---

## rocprofiler-v3: Data Collection

### Purpose
Runtime profiling tool that captures kernel traces, API calls, and hardware counters during application execution.

### Basic Usage

```bash
rocprofv3 --hip-trace --kernel-trace \
          --output-format rpd \
          -d ./traces -o profile \
          -- python app.py
```

### Key Parameters

**Tracing Options:**
- `--hip-trace` - Trace HIP API calls
- `--kernel-trace` - Trace GPU kernel executions
- `--memory-copy-trace` - Trace memory operations
- `--hsa-trace` - Trace HSA API

**Output Options:**
- `-d <dir>` - Output directory
- `-o <name>` - Output file name
- `--output-format <format>` - pftrace, json, csv, rpd

**Hardware Counters:**
- `--counter <name>` - Collect specific counter (e.g., SQ_WAVES, TCC_HIT)
- `--list-counters` - Show available counters

**Filtering:**
- `--kernel-filter <pattern>` - Filter kernels by name

### Strix-Specific Usage

**Profile on Strix:**
```bash
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
          --output-format rpd \
          -d ./strix_traces -o strix_profile \
          -- python inference.py
```

**With Hardware Counters:**
```bash
rocprofv3 --kernel-trace \
          --counter SQ_WAVES \
          --counter TCC_HIT \
          --counter GRBM_GUI_ACTIVE \
          -d ./counters -o hw_profile \
          -- python app.py
```

---

## rocmprofilerdata (RPD): Data Analysis

### Purpose
Post-processing tool for analyzing profiling data collected by rocprofiler-v3.

### Basic Usage

**View Summary:**
```bash
rpd summary profile.rpd
```

**Query Data:**
```bash
rpd query profile.rpd --kernels --sort-by duration --limit 10
```

**Generate Report:**
```bash
rpd report profile.rpd --output report.html
```

### Key Commands

**1. Summary Command**
```bash
rpd summary <file.rpd>
# Shows: Total kernels, duration, GPU utilization, memory transfers
```

**2. Query Command**
```bash
# Query kernels
rpd query <file.rpd> --kernels

# Sort by duration (slowest first)
rpd query <file.rpd> --kernels --sort-by duration --descending

# Filter by name
rpd query <file.rpd> --kernels --filter "gemm*"

# Limit results
rpd query <file.rpd> --kernels --limit 20

# Query memory operations
rpd query <file.rpd> --memory-ops
```

**3. Report Command**
```bash
# Generate HTML report
rpd report <file.rpd> --output report.html

# Generate PDF
rpd report <file.rpd> --format pdf --output report.pdf
```

**4. Convert Command**
```bash
# Convert to CSV
rpd convert <file.rpd> --format csv --output data.csv

# Convert to JSON
rpd convert <file.rpd> --format json --output data.json
```

**5. Compare Command**
```bash
# Compare two profiling runs
rpd compare baseline.rpd optimized.rpd --output comparison.html
```

---

## Identifying Slow Kernels

### Primary Parameter: `--sort-by duration`

```bash
rpd query profile.rpd --kernels --sort-by duration --descending --limit 10
```

### Sort Options

- `--sort-by duration` - Total time (identifies biggest bottlenecks)
- `--sort-by avg-duration` - Average per call (identifies inefficient kernels)
- `--sort-by max-duration` - Longest single execution (identifies outliers)
- `--sort-by calls` - Number of invocations

### Order

- `--descending` or `-d` - Slowest first (default for duration)
- `--ascending` or `-a` - Fastest first

---

## Complete Strix Profiling Workflow

### Step 1: Collect Baseline
```bash
rocprofv3 --hip-trace --kernel-trace --stats \
          --output-format rpd \
          -d ./baseline -o baseline \
          -- python model.py
```

### Step 2: Identify Hotspots
```bash
rpd query ./baseline/baseline.rpd \
    --kernels --sort-by duration --descending --limit 10
```

### Step 3: Deep Dive on Slow Kernels
```bash
# Profile specific kernel with counters
SLOW_KERNEL="slow_gemm_kernel"

rocprofv3 --kernel-trace \
          --kernel-filter "$SLOW_KERNEL*" \
          --counter SQ_WAVES \
          --counter TCC_HIT \
          -d ./analysis -o slow_kernel \
          -- python model.py
```

### Step 4: Generate Report
```bash
rpd report ./analysis/slow_kernel.rpd --output analysis.html
```

### Step 5: Optimize and Compare
```bash
# After optimization
rocprofv3 --hip-trace --kernel-trace --stats \
          --output-format rpd \
          -d ./optimized -o optimized \
          -- python model_optimized.py

# Compare
rpd compare ./baseline/baseline.rpd ./optimized/optimized.rpd \
    --output comparison.html
```

---

## rocprofiler-v3 vs RPD Differences

| Aspect | rocprofiler-v3 | RPD |
|--------|----------------|-----|
| **Purpose** | Collect profiling data | Analyze profiling data |
| **When** | Runtime (during execution) | Post-processing (after execution) |
| **Input** | Application binary/script | .rpd files |
| **Output** | .rpd, .pftrace, .json files | Reports, analysis, visualizations |
| **Action** | Traces kernels, APIs, counters | Queries, filters, compares data |
| **Type** | Profiling tool | Analysis tool |

---

## Strix-Specific Considerations

### 1. Unified Memory Architecture
Strix uses shared system memory - profile memory patterns carefully:
```bash
rocprofv3 --memory-copy-trace --hip-trace \
          -d ./uma_analysis -- python app.py
```

### 2. Power Constraints
Strix is power-limited - include stats:
```bash
rocprofv3 --hip-trace --kernel-trace --stats \
          -d ./power_analysis -- python app.py
```

### 3. Concurrent Execution
Test concurrent kernel execution:
```bash
rocprofv3 --hip-trace --kernel-trace --timestamp \
          --output-format pftrace \
          -d ./concurrent -- python concurrent_test.py
```

---

## Viewing Results

### Perfetto (Chrome Tracing)
1. Navigate to: `chrome://tracing`
2. Load: `traces/profile.pftrace`
3. View timeline with kernel executions

### JSON Analysis
```python
import json

with open('traces/profile.json') as f:
    data = json.load(f)

for event in data['traceEvents']:
    if event.get('cat') == 'kernel':
        print(f"{event['name']}: {event['dur']} us")
```

### Statistics File
```bash
cat traces/profile_stats.txt
```

---

## Key Metrics for Strix

**Performance Metrics:**
- Latency (ms): P50, P95, P99
- Throughput: FPS, Inferences/sec
- Bandwidth: GB/s

**Resource Metrics:**
- GPU Utilization: %
- Memory Usage: MB
- Power: Watts

**Hardware Counters:**
- SQ_WAVES: Wavefronts
- TCC_HIT/MISS: L2 cache
- GRBM_GUI_ACTIVE: GPU busy %

---

## Quick Reference

**Collect data:**
```bash
rocprofv3 --hip-trace --kernel-trace --output-format rpd -d ./traces -- python app.py
```

**Find slow kernels:**
```bash
rpd query traces/profile.rpd --kernels --sort-by duration --limit 10
```

**Generate report:**
```bash
rpd report traces/profile.rpd --output report.html
```

**Compare runs:**
```bash
rpd compare baseline.rpd optimized.rpd --output comparison.html
```

---

## Summary

- **rocprofiler-v3** = Data collector (runtime)
- **RPD** = Data analyst (post-processing)
- **Workflow**: Profile with rocprofv3 → Analyze with RPD → Optimize → Repeat
- **Key for Strix**: Focus on memory patterns, power efficiency, and kernel optimization

