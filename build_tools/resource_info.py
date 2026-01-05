#!/usr/bin/env python3
"""
Tooling for TheRock build system. The goal is to have single unified view across build resource utilization for TheRock components and systems locally as well as in CI.

Features:
- Acts as CMake compiler launcher (C/C++).
- Logs per-command timing + memory usage to <repo_root>/build/logs/therock-build-prof (or THEROCK_BUILD_PROF_LOG_DIR).
- Aggregates all logs into:
    - comp-summary.md   (Markdown table per-component)
    - comp-summary.html (FAQ + rendered HTML table)
- NEVER fails the build if profiling/reporting fails.
- Only propagates the real compiler's exit code.

Usage in CMake configure:

  cmake -S . -B build -GNinja \
    -DTHEROCK_AMDGPU_FAMILIES=gfx110X-all \
    -DCMAKE_C_COMPILER_LAUNCHER="${PWD}/build_tools/resource_info.py" \
    -DCMAKE_CXX_COMPILER_LAUNCHER="${PWD}/build_tools/resource_info.py"

    cmake --build build

Output:
    Summary gets created in build/logs/therock-build-prof/comp-summay.md (comp-summary.html)
"""

import os
import sys
import time
import random
import datetime
import subprocess
import shlex
import html
import re
from typing import Dict, Tuple, List
from pathlib import Path

# Best-effort resource usage (not available on Windows)
try:
    import resource  # type: ignore
except Exception:
    resource = None


# -------------------------
# Component classification
# -------------------------


CMAKEFILES_TARGET_RE = re.compile(r"(?:^|[ /])CMakeFiles/([^/]+)\.dir/")

THEROCK_COMPONENTS = [
    "rocPRIM",
    "rccl",
    "rocThrust",
    "rocWMMA",
    "rocBLAS",
    "MIOpen",
    "hipCUB",
    "rocFFT",
    "amd-llvm",
    "rocSPARSE",
    "hipBLASLt",
    "rocprofiler-systems",
    "hip-tests",
    "rocprofiler-sdk",
    "composable_kernel",
    "hipSPARSE",
    "rocSOLVER",
    "hipSPARSELt",
    "hipFFT",
    "rocRAND",
    "roctracer",
    "rdc",
    "rocRoller",
    "hipBLAS",
    "core-hiptests",
    "hipDNN",
    "rocprofiler-register",
    "hipSOLVER",
    "miopen_plugin",
    "core-runtime",
    "core-hip",
    "amd-comgr",
    "rccl-tests",
    "rocprofiler-compute",
    "hipRAND",
    "hipify",
    "core-ocl",
    "hipBLAS-common",
    "mxDataGenerator",
    "amdsmi",
    "amd-comgr-stub",
    "opencl",
    "aqlprofile",
    "rocm_smi_lib",
    "rocm-core",
    "hipcc",
    "aux-overlay",
    "rocprof-trace-decoder",
    "half",
    "rocminfo",
    "rocm-cmake",
    "flatbuffers",
    "spdlog",
    "nlohmann-json",
    "fmt",
]

FAQ_HTML = """
<h2>General FAQ</h2>

<ol>
  <li><b>rss = Resident Set Size</b>
    <p>
      It means the amount of physical RAM actually in use by a process at its peak. High rss means memory-heavy compilation or linking.
      If rss is high across many parallel jobs, we can hit swapping, cache thrashing, OOM kills in CI. High rss often limits how much
      parallelism you can safely use (-j)
    </p>
  </li>

  <li><b>max_rss_mb</b>
    <p>
      the maximum RAM a compiler or linker process had allocated in memory at any point.
    </p>
  </li>

  <li><b>Wall time sum vs component span vs estimated elapsed</b>
    <p>
      <b>wall_time_sum</b> is the <b>sum</b> of wall times across all commands for that component. In parallel builds this can be
      far larger than the total build duration because many commands run concurrently.
    </p>
    <p>
      <b>wall_time_span_min</b> is the component’s <b>elapsed span</b>:
      <code>max(end_time) - min(start_time)</code> across all commands belonging to that component. This is usually much closer to
      “how long the build spent working on that component”, though it can still overlap with other components (parallelism).
    </p>
    <p>
      <b>wall_time_est_elapsed_min</b> is an <b>estimated elapsed contribution</b> computed from average build concurrency:
      we compute <code>avg_concurrency = sum(real_s) / build_span_s</code> using timestamps, and then estimate:
      <code>wall_time_est_elapsed ≈ wall_time_sum / avg_concurrency</code>.
    </p>
    <p>
      That doesn’t mean the build is slower; it means processes are spending time waiting (I/O, scheduling, throttling, contention),
      so CPU time isn’t keeping up with wall time.
    </p>
  </li>

  <li><b>What Avg Threads (avg_threads) actually mean ?</b>
    <p><b>Case 1&gt; avg_threads ≈ 1.0</b></p>
    <p>This means process used about one CPU core for most of its runtime.</p>

    <p><b>Case 2&gt; avg_threads &lt; 1.0</b></p>
    <p><b>Meaning:</b> The process spent a lot of time waiting, not computing (I/O, contention, throttling).</p>

    <p><b>Case 3&gt; avg_threads &gt; 1.0</b></p>
    <p><b>Meaning:</b> The process used multiple CPU cores simultaneously (e.g. LTO backends, LLVM worker threads).</p>
  </li>


<li><b> What user_sum, sys_sum mean and cpu_sum mean ? (Note: all times are in minutes) </b>

    <p><b>user_sum → time spent executing your code (compiler, linker, optimizer logic) </b></p>

        <p><b>If this is high:</b>

        <p>a) the build is compute-heavy</p>

        <p>b) faster CPUs, fewer templates, or fewer TUs help</p>

        <p>c) more parallelism may help if avg_threads > 1</p>

    <p><b>sys_sum → time spent inside the operating system kernel</b></p>

        <p><b>a) If this is high:</b>

        <p>b) you’re often I/O-bound</p>

        <p>c) disk speed, filesystem, caching, or build directory layout matters</p>

        <p>d) adding more CPUs will not help much</p>

    <p><b>cpu_sum → total CPU time = user_sum_min + sys_sum_min</b></p>

        <p>a) How much CPU did this component cost overall?</p>

        <p>b) It’s the best metric for:</p>

        <p>c) capacity planning</p>

        <p>d) CI cost estimation</p>

        <p>e) "what’s expensive" comparisons between components</p>

</li>
</ol>

<h3>Key mental model</h3>
<ul>
  <li> wall_time_sum</b> → total summed command wall time (inflates with parallelism)</li>
  <li> wall_time_span</b> → component elapsed window (closest to “actual component build time”)</li>
  <li> wall_time_est_elapsed</b> → concurrency-adjusted estimate</li>
  <li> cpu_sum</b> → cost/compute spent</li>
  <li> avg_threads</b> → CPU utilization ratio per component</li>
  <li> rss</b> → limits safe parallelism</li>
</ul>
"""


def therock_components(_pwd_unused: str, cmd_str: str) -> str:
    comp = "unknown"
    lower_cmd = cmd_str.lower()

    for p in cmd_str.split():
        if p.startswith("@") and len(p) > 1:
            rsp_path = p[1:]
            try:
                lower_cmd += (
                    "\n"
                    + Path(rsp_path)
                    .read_text(encoding="utf-8", errors="ignore")
                    .lower()
                )
            except Exception:
                pass

    # CMake targets
    m = CMAKEFILES_TARGET_RE.search(cmd_str) or CMAKEFILES_TARGET_RE.search(lower_cmd)
    if m:
        target = m.group(1)
        # common target decorations
        target_norm = target.split("-", 1)[0].split("_", 1)[0].lower()

        # exact target match against known components
        for name in THEROCK_COMPONENTS:
            if name.lower() == target.lower():
                return name

        # prefix match
        for name in THEROCK_COMPONENTS:
            if name.lower() == target_norm:
                return name

    # substring match anywhere
    for name in sorted(THEROCK_COMPONENTS, key=len, reverse=True):
        if name.lower() in lower_cmd:
            return name

    return comp


# -------------------------
# Logging + measurement
# -------------------------


def run_and_log_command(log_dir: str) -> int:
    if len(sys.argv) <= 1:
        return 0

    os.makedirs(log_dir, exist_ok=True)

    pwd = os.getcwd()
    cmd_args = sys.argv[1:]
    cmd_str = " ".join(shlex.quote(arg) for arg in cmd_args)

    comp = therock_components(pwd, cmd_str)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rand = random.randint(0, 999999)
    log_file = Path(log_dir) / f"build-{ts}-{rand}-{comp}.log"

    # Start times
    start_epoch_s = time.time()
    start_wall = time.monotonic()
    start_cpu = time.process_time()

    start_self = start_child = None
    if resource is not None:
        try:
            start_self = resource.getrusage(resource.RUSAGE_SELF)
            start_child = resource.getrusage(resource.RUSAGE_CHILDREN)
        except Exception:
            start_self = start_child = None

    try:
        result = subprocess.run(cmd_args)
        returncode = result.returncode
    except OSError as e:
        returncode = 127
        cmd_str = f"{cmd_str}  # EXEC ERROR: {e!r}"

    end_wall = time.monotonic()
    real_seconds = end_wall - start_wall

    end_cpu = time.process_time()
    cpu_seconds = end_cpu - start_cpu

    user_seconds = 0.0
    sys_seconds = 0.0
    maxrss_kb = 0

    if resource is not None and start_self is not None and start_child is not None:
        try:
            end_self = resource.getrusage(resource.RUSAGE_SELF)
            end_child = resource.getrusage(resource.RUSAGE_CHILDREN)

            user_seconds = (end_child.ru_utime - start_child.ru_utime) + (
                end_self.ru_utime - start_self.ru_utime
            )
            sys_seconds = (end_child.ru_stime - start_child.ru_stime) + (
                end_self.ru_stime - start_self.ru_stime
            )
            maxrss_kb = int(end_child.ru_maxrss)
        except Exception:
            user_seconds = cpu_seconds
            sys_seconds = 0.0
            maxrss_kb = 0
    else:
        user_seconds = cpu_seconds
        sys_seconds = 0.0
        maxrss_kb = 0

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("schema=2\n")
            f.write("time_unit=seconds\n")
            f.write(f"start_epoch_s={start_epoch_s:.6f}\n")
            f.write(f"comp={comp}\n")
            f.write(f"cmd={cmd_str}\n")
            f.write(f"real_s={real_seconds:.6f}\n")
            f.write(f"user_s={user_seconds:.6f}\n")
            f.write(f"sys_s={sys_seconds:.6f}\n")
            f.write(f"maxrss_kb={maxrss_kb}\n")
            # Human-readable:
            f.write(f"real_min={(real_seconds / 60.0):.6f}\n")
            f.write(f"user_min={(user_seconds / 60.0):.6f}\n")
            f.write(f"sys_min={(sys_seconds / 60.0):.6f}\n")
    except Exception:
        pass

    return returncode


# ------------------------------------------------
# Aggregation / Resource Observability reporting
# ------------------------------------------------


def parse_log_file(path: Path, stats: Dict[Tuple[str, str], float]) -> None:
    comp = "unknown"
    start_epoch_s = None
    real_s = user_s = sys_s = None
    rss_kb = None

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line.startswith("start_epoch_s="):
                    try:
                        start_epoch_s = float(line[len("start_epoch_s=") :])
                    except ValueError:
                        pass
                elif line.startswith("comp="):
                    comp = line[len("comp=") :] or "unknown"
                elif line.startswith("real_s="):
                    try:
                        real_s = float(line[len("real_s=") :])
                    except ValueError:
                        pass
                elif line.startswith("user_s="):
                    try:
                        user_s = float(line[len("user_s=") :])
                    except ValueError:
                        pass
                elif line.startswith("sys_s="):
                    try:
                        sys_s = float(line[len("sys_s=") :])
                    except ValueError:
                        pass
                elif line.startswith("maxrss_kb="):
                    try:
                        rss_kb = float(line[len("maxrss_kb=") :])
                    except ValueError:
                        pass
    except Exception:
        return

    if real_s is None:
        return

    stats[(comp, "real_s")] = stats.get((comp, "real_s"), 0.0) + real_s
    if user_s is not None:
        stats[(comp, "user_s")] = stats.get((comp, "user_s"), 0.0) + user_s
    if sys_s is not None:
        stats[(comp, "sys_s")] = stats.get((comp, "sys_s"), 0.0) + sys_s

    # Per-component span tracking (elapsed window)
    if start_epoch_s is not None and start_epoch_s > 0.0:
        end_epoch_s = start_epoch_s + real_s

        k_start = (comp, "span_start_min")
        k_end = (comp, "span_end_max")
        if k_start not in stats or start_epoch_s < stats[k_start]:
            stats[k_start] = start_epoch_s
        if k_end not in stats or end_epoch_s > stats[k_end]:
            stats[k_end] = end_epoch_s

        # Global build span time
        g_start = ("__build__", "span_start_min")
        g_end = ("__build__", "span_end_max")
        if g_start not in stats or start_epoch_s < stats[g_start]:
            stats[g_start] = start_epoch_s
        if g_end not in stats or end_epoch_s > stats[g_end]:
            stats[g_end] = end_epoch_s

    if rss_kb is not None:
        prev = stats.get((comp, "rss_kb_max"), 0.0)
        if rss_kb > prev:
            stats[(comp, "rss_kb_max")] = rss_kb

    stats[(comp, "_seen")] = 1.0


def markdown_table_to_html(md_text: str) -> str:
    lines = [l.rstrip() for l in md_text.splitlines() if l.strip()]

    table_lines = []
    in_table = False

    for line in lines:
        if line.startswith("|") and "|" in line[1:]:
            table_lines.append(line)
            in_table = True
        elif in_table:
            break

    if len(table_lines) < 2:
        return "<p>No summary table found.</p>"

    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    rows = []

    for line in table_lines[2:]:
        cols = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cols)

    out = ["<table border='1' cellpadding='6' cellspacing='0'>"]
    out.append("<thead><tr>")
    for h in headers:
        out.append(f"<th>{html.escape(h)}</th>")
    out.append("</tr></thead><tbody>")

    for r in rows:
        out.append("<tr>")
        for c in r:
            out.append(f"<td>{html.escape(c)}</td>")
        out.append("</tr>")

    out.append("</tbody></table>")
    return "\n".join(out)


def print_summary_links(log_dir: str) -> None:
    try:
        base = Path(log_dir).resolve()
        html_path = base / "comp-summary.html"
        md_path = base / "comp-summary.md"

        print("\n[TheRock Build Summary]")
        if html_path.exists():
            print(f"HTML (absolute URI): {html_path.resolve().as_uri()}")
        if md_path.exists():
            print(f"Markdown: {md_path}")
        print()
    except Exception:
        pass


def generate_summaries(log_dir: str) -> None:
    lock_path = Path(log_dir) / ".summary.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return
    except Exception:
        return

    try:
        stats: Dict[Tuple[str, str], float] = {}
        for name in os.listdir(log_dir):
            if not name.endswith(".log"):
                continue
            parse_log_file(Path(log_dir) / name, stats)

        components = sorted({key[0] for key in stats.keys() if key[1] == "_seen"})
        if not components:
            return

        # Average build concurrency estimate from timestamps:
        # avg_concurrency ~= sum(real_s) / build_span_s
        total_wall_s_all = 0.0
        for comp in components:
            total_wall_s_all += stats.get((comp, "real_s"), 0.0)

        build_start = stats.get(("__build__", "span_start_min"))
        build_end = stats.get(("__build__", "span_end_max"))
        build_span_s = (
            (build_end - build_start)
            if (
                build_start is not None
                and build_end is not None
                and build_end >= build_start
            )
            else 0.0
        )

        avg_concurrency = (
            (total_wall_s_all / build_span_s) if build_span_s > 0.0 else 1.0
        )
        if avg_concurrency <= 0.0:
            avg_concurrency = 1.0

        rows = []
        for comp in components:
            wall_s = stats.get((comp, "real_s"), 0.0)
            user_s = stats.get((comp, "user_s"), 0.0)
            sys_s = stats.get((comp, "sys_s"), 0.0)
            cpu_s = user_s + sys_s

            wall_sum_min = wall_s / 60.0
            cpu_min = cpu_s / 60.0
            user_min = user_s / 60.0
            sys_min = sys_s / 60.0

            span_start = stats.get((comp, "span_start_min"))
            span_end = stats.get((comp, "span_end_max"))
            wall_span_min = (
                (span_end - span_start) / 60.0
                if (
                    span_start is not None
                    and span_end is not None
                    and span_end >= span_start
                )
                else 0.0
            )

            wall_est_elapsed_min = (
                wall_sum_min / avg_concurrency
                if avg_concurrency > 0.0
                else wall_sum_min
            )

            avg_threads = cpu_s / wall_s if wall_s > 0.0 else 0.0

            rss_kb = stats.get((comp, "rss_kb_max"), 0.0)
            rss_mb = rss_kb / 1024.0
            rss_gb = rss_kb / (1024.0 * 1024.0)

            rows.append(
                (
                    comp,
                    wall_sum_min,
                    wall_span_min,
                    wall_est_elapsed_min,
                    cpu_min,
                    user_min,
                    sys_min,
                    avg_threads,
                    rss_mb,
                    rss_gb,
                )
            )

        # Sort by cpu_sum_min descending
        rows.sort(key=lambda r: r[4], reverse=True)

        # Atomic write to avoid partial header-only files
        md_path = Path(log_dir) / "comp-summary.md"
        tmp_md_path = Path(log_dir) / "comp-summary.md.tmp"
        try:
            with open(tmp_md_path, "w", encoding="utf-8") as f:
                headers = [
                    "component",
                    "wall_time_sum (minutes)",
                    "wall_time_span (minutes)",
                    "wall_time_est_elapsed (minutes)",
                    "cpu_sum (minutes)",
                    "user_sum (minutes)",
                    "sys_sum (minutes)",
                    "avg_threads",
                    "max_rss_mb",
                    "max_rss_gb",
                ]
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

                for (
                    comp,
                    wall_sum_min,
                    wall_span_min,
                    wall_est_elapsed_min,
                    cpu_min,
                    user_min,
                    sys_min,
                    avg_threads,
                    rss_mb,
                    rss_gb,
                ) in rows:
                    f.write(
                        "| "
                        f"{comp} | "
                        f"{wall_sum_min:.2f} | {wall_span_min:.2f} | {wall_est_elapsed_min:.2f} | "
                        f"{cpu_min:.2f} | {user_min:.2f} | {sys_min:.2f} | "
                        f"{avg_threads:.2f} | "
                        f"{rss_mb:.2f} | {rss_gb:.4f} |\n"
                    )

            os.replace(str(tmp_md_path), str(md_path))
        except Exception:
            try:
                os.remove(str(tmp_md_path))
            except Exception:
                pass

        html_path = Path(log_dir) / "comp-summary.html"
        try:
            md_text = md_path.read_text(encoding="utf-8", errors="ignore")
            table_html = markdown_table_to_html(md_text)

            html_doc = (
                "<!doctype html>\n"
                "<html>\n<head>\n"
                '  <meta charset="utf-8" />\n'
                "  <title>TheRock Build Resource Observability Report</title>\n"
                "  <style>\n"
                "    body { font-family: Arial, sans-serif; margin: 24px; }\n"
                "    h1 { margin-bottom: 8px; }\n"
                "    table { border-collapse: collapse; margin-top: 16px; }\n"
                "    th { background: #f0f0f0; }\n"
                "    th, td { padding: 8px 12px; text-align: center; }\n"
                "  </style>\n"
                "</head>\n<body>\n"
                "<h1>TheRock Build Resource Observability Report</h1>\n"
                "<h2>Build Resource Utilization Summary</h2>\n"
                f"{table_html}\n"
                "<hr />\n"
                f"{FAQ_HTML}\n"
                "</body>\n</html>\n"
            )

            html_path.write_text(html_doc, encoding="utf-8")
        except Exception:
            pass

        print_summary_links(log_dir)

    finally:
        try:
            os.remove(str(lock_path))
        except Exception:
            pass


# --------------
# Entrypoint
# --------------


def main() -> int:
    default_log_dir = str(
        (Path(__file__).resolve().parents[1] / "build" / "logs" / "therock-build-prof")
    )
    log_dir = os.environ.get("THEROCK_BUILD_PROF_LOG_DIR", default_log_dir)

    rc = run_and_log_command(log_dir)

    # Best-effort summary generation as part of the build. Never fail build.
    try:
        generate_summaries(log_dir)
    except Exception:
        pass

    return rc


if __name__ == "__main__":
    sys.exit(main())
