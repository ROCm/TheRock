#!/usr/bin/env python3
"""
Tooling for TheRock build system. The goal is to have single unified view across build resource utilization for TheRock compoments and systems locally as well as in CI

Features:
- Acts as CMake compiler launcher (C/C++).
- Logs per-command timing + memory usage to /build/log/therock-build-resources (or THEROCK_BUILD_PROF_LOG_DIR).
- Aggregates all logs into:
    - comp-summary.csv  (per-component summary)
    - comp-summary.md   (Markdown table per-component)
- NEVER fails the build if profiling/reporting fails.
- Only propagates the real compiler's exit code.

Usage in CMake configure:

  cmake -S . -B build -GNinja \
    -DTHEROCK_AMDGPU_FAMILIES=gfx110X-all \
    -DCMAKE_C_COMPILER_LAUNCHER="${PWD}/build_tools/resource_info.py" \
    -DCMAKE_CXX_COMPILER_LAUNCHER="${PWD}/build_tools/resource_info.py"


  cmake --build build
          OR
  ninja -C build -j"$(nproc)"


Output:
    End summary for observability of build resources gets created in /build/log/comp-summary.md /build/log/comp-summary.csv file



Generic FAQ:

    1) RSS = Resident Set Size

        It means the amount of physical RAM actually in use by a process at its peak. High RSS means memory-heavy compilation or linking. If RSS is high across many parallel jobs, we can hit swapping, cache thrashing, OOM kills in CI. High RSS often limits how much parallelism you can safely use (-j)


    2) max_rss_mb = the maximum RAM a compiler or linker process had allocated in memory at any point.

    3) What Avg Threads (avg_threads) actually mean ? 

    Case 1> avg_threads ≈ 1.0


            This means process used about one CPU core for most of its runtime.

            Typical causes:

                Normal C/C++ compilation (clang/gcc front-end is mostly single-threaded)

                Serial link steps (ld, ar)

                Code generation steps

            Interpretation:

                This is expected for many build steps.

                Increasing -j will not make this step faster — only running more steps in parallel helps.



     Case 2> avg_threads < 1.0

            Meaning:

                The process spent a lot of time waiting, not computing.

            Common reasons:

                I/O waits (reading headers, writing object files)

                Lock contention

                Process startup / scheduling overhead

                Throttling / CPU starvation in CI

            Interpretation:

                The step is not CPU-bound.

                Faster disks, better caching, or reducing process churn may help more than more CPUs.


        Case 3> avg_threads > 1.0

            Meaning:

                The process used multiple CPU cores simultaneously.

                How this happens in builds:

                    Parallel LTO / ThinLTO backends

                    LLVM optimizations using worker threads

                    Compiler spawning helper threads

                    Some linkers using parallelism

            Interpretation:

                    This step benefits from more cores.

                    It can reduce total wall time but increase total CPU time.



Key mental model:

    Wall Time → affects how long the build takes

    CPU Time → affects how expensive the build is

    Avg Threads → tells how much parallelism each step actually used

    RSS → limits how many steps we can run at once


"""

import os
import sys
import time
import random
import datetime
import resource
import subprocess
import shlex
import html
from typing import Dict, Tuple
from pathlib import Path


# -------------------------
# Component classification
# -------------------------

# NOTE: These are the component/dependency names you referenced from the build_time_analysis.html page.
# The web fetch for that page failed for me in this session (server disconnected), so this list reflects
# what we previously discussed. If your HTML has additional names, just append them here.
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


def therock_components(_pwd_unused: str, cmd_str: str) -> str:
    """ when not clear to infer which component this compile belongs to."""
    comp = "unknown"

    # Expand response files if present (e.g. clang++ @file.rsp)
    lower_cmd = cmd_str.lower()
    for p in cmd_str.split():
        if p.startswith("@"):
            rsp_path = p[1:]
            try:
                lower_cmd += "\n" + Path(rsp_path).read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                pass

    # Match ROCm components
    for name in sorted(THEROCK_COMPONENTS, key=len, reverse=True):
        if name.lower() in lower_cmd:
            return name

    return comp


# -------------------------
# Logging + measurement
# -------------------------

def run_and_log_command(log_dir: str) -> int:
    """
    Run the actual compiler command (sys.argv[1:]),
    log its timing & resources, and return the compiler's exit code.
    """
    if len(sys.argv) <= 1:
        # Nothing to run – act as a no-op and don't fail the build.
        return 0

    os.makedirs(log_dir, exist_ok=True)

    pwd = os.getcwd()
    cmd_args = sys.argv[1:]
    cmd_str = " ".join(shlex.quote(arg) for arg in cmd_args)

    comp = therock_components(pwd, cmd_str)

    # Unique log filenames per component
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rand = random.randint(0, 999999)
    log_file = Path(log_dir) / f"build-{ts}-{rand}-{comp}.log"

    # Measure before compiler execution
    start_wall = time.monotonic()
    try:
        start_self = resource.getrusage(resource.RUSAGE_SELF)
        start_child = resource.getrusage(resource.RUSAGE_CHILDREN)
    except Exception:
        # If resource usage fails, fall back to simple timing
        start_self = start_child = None

    # Run the actual compiler
    try:
        result = subprocess.run(cmd_args)
        returncode = result.returncode
    except OSError as e:
        # If we can't even spawn the compiler, log that but ensure build sees failure
        returncode = 127
        cmd_str = f"{cmd_str}  # EXEC ERROR: {e!r}"

    # Measure after compiler execution
    end_wall = time.monotonic()
    real_time = end_wall - start_wall

    user_time = 0.0
    sys_time = 0.0
    maxrss_kb = 0

    try:
        if start_self is not None and start_child is not None:
            end_self = resource.getrusage(resource.RUSAGE_SELF)
            end_child = resource.getrusage(resource.RUSAGE_CHILDREN)

            user_time = (
                (end_child.ru_utime - start_child.ru_utime)
                + (end_self.ru_utime - start_self.ru_utime)
            )
            sys_time = (
                (end_child.ru_stime - start_child.ru_stime)
                + (end_self.ru_stime - start_self.ru_stime)
            )
            # ru_maxrss is in kilobytes on Linux for RUSAGE_CHILDREN
            maxrss_kb = end_child.ru_maxrss
    except Exception:
        # If we fail to read usage, leave zeros; do not break the build.
        pass

    # Write per-command log (best effort)
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"comp={comp}\n")
            f.write(f"cmd={cmd_str}\n")
            f.write(f"real={real_time:.6f}\n")
            f.write(f"user={user_time:.6f}\n")
            f.write(f"sys={sys_time:.6f}\n")
            f.write(f"maxrss_kb={maxrss_kb}\n")
    except Exception:
        # Never break the build on logging failures
        pass

    return returncode


# ------------------------------------------------
# Aggregation / Resource Observability reporting
# ------------------------------------------------

def parse_log_file(path: str, stats: Dict[Tuple[str, str], float]) -> None:
    """
    Parse a single .log file and accumulate into stats dict.

    stats key = (component, metric)
    metrics: "real", "user", "sys", "rss_kb_max"
    """
    comp = "unknown"
    real = user = sys_t = None
    rss_kb = None

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("comp="):
                    comp = line[len("comp="):] or "unknown"
                elif line.startswith("real="):
                    try:
                        real = float(line[len("real="):])
                    except ValueError:
                        pass
                elif line.startswith("user="):
                    try:
                        user = float(line[len("user="):])
                    except ValueError:
                        pass
                elif line.startswith("sys="):
                    try:
                        sys_t = float(line[len("sys="):])
                    except ValueError:
                        pass
                elif line.startswith("maxrss_kb="):
                    try:
                        rss_kb = float(line[len("maxrss_kb="):])
                    except ValueError:
                        pass
    except Exception:
        # If any one log file is corrupt, skip it.
        return

    if comp is None:
        comp = "unknown"

    # Only accumulate if we have at least real/user/sys values
    if real is not None:
        stats[(comp, "real")] = stats.get((comp, "real"), 0.0) + real
    if user is not None:
        stats[(comp, "user")] = stats.get((comp, "user"), 0.0) + user
    if sys_t is not None:
        stats[(comp, "sys")] = stats.get((comp, "sys"), 0.0) + sys_t
    if rss_kb is not None:
        prev = stats.get((comp, "rss_kb_max"), 0.0)
        if rss_kb > prev:
            stats[(comp, "rss_kb_max")] = rss_kb
    # Track component presence
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
            break  # stop after first table

    if len(table_lines) < 2:
        return "<p>No summary table found.</p>"

    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    rows = []

    for line in table_lines[2:]:  # skip header + separator
        cols = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cols)

    html = ["<table border='1' cellpadding='6' cellspacing='0'>"]
    html.append("<thead><tr>")
    for h in headers:
        html.append(f"<th>{h}</th>")
    html.append("</tr></thead><tbody>")

    for r in rows:
        html.append("<tr>")
        for c in r:
            html.append(f"<td>{c}</td>")
        html.append("</tr>")

    html.append("</tbody></table>")
    return "\n".join(html)



def generate_summaries(log_dir: str) -> None:
    """
    Scan all *.log files in log_dir and generate:
      - comp-summary.csv
      - comp-summary.md
      - Uses a simple lock file to avoid concurrent writers.
      - Any failure is masked so builds never break.
    """
    # Simple file-based lock to avoid concurrent summary writers
    lock_path = Path(log_dir) / ".summary.lock"
    try:
        # O_CREAT | O_EXCL -> fail if already exists
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        # Another process is already generating summaries; skip
        return
    except Exception:
        # If lock failed for any reason, just skip (never break build)
        return

    try:
        stats: Dict[Tuple[str, str], float] = {}
        # Gather logs
        for name in os.listdir(log_dir):
            if not name.endswith(".log"):
                continue
            path = Path(log_dir) / name
            parse_log_file(path, stats)

        # Build list of components
        components = sorted({
            key[0] for key in stats.keys() if key[1] == "_seen"
        })

        if not components:
            return  # nothing to do

        # Prepare summary data
        rows = []
        for comp in components:
            real = stats.get((comp, "real"), 0.0)
            user = stats.get((comp, "user"), 0.0)
            sys_t = stats.get((comp, "sys"), 0.0)
            cpu_sum = user + sys_t
            avg_threads = cpu_sum / real if real > 0.0 else 0.0
            rss_kb = stats.get((comp, "rss_kb_max"), 0.0)
            rss_mb = rss_kb / 1024.0
            rss_gb = rss_kb / (1024.0 * 1024.0)
            rows.append((comp, real, user, sys_t, cpu_sum, avg_threads,
                         rss_kb, rss_mb, rss_gb))

        # Sort by cpu_sum descending (like the awk version with -k5,5nr)
        rows.sort(key=lambda r: r[4], reverse=True)

        # 1) Write CSV
        csv_path = Path(log_dir) / "comp-summary.csv"
        try:
            with open(csv_path, "w", encoding="utf-8") as f:
                headers = [
                    "component",
                    "real_sum",
                    "user_sum",
                    "sys_sum",
                    "cpu_sum",
                    "avg_threads",
                    "max_rss_kb",
                    "max_rss_mb",
                    "max_rss_gb",
                ]
                f.write(",".join(headers) + "\n")
                for (comp, real, user, sys_t, cpu_sum, avg_threads,
                     rss_kb, rss_mb, rss_gb) in rows:
                    # naive CSV quoting: wrap component in quotes if needed
                    comp_str = f"\"{comp}\"" if "," in comp else comp
                    f.write(
                        f"{comp_str},"
                        f"{real:.6f},{user:.6f},{sys_t:.6f},"
                        f"{cpu_sum:.6f},{avg_threads:.6f},"
                        f"{int(rss_kb)},{rss_mb:.6f},{rss_gb:.8f}\n"
                    )
        except Exception:
            # ignore CSV failures
            pass

        # 2) Write Markdown table
        md_path = Path(log_dir) / "comp-summary.md"
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                headers = [
                    "component",
                    "real_sum",
                    "user_sum",
                    "sys_sum",
                    "cpu_sum",
                    "avg_threads",
                    "max_rss_kb",
                    "max_rss_mb",
                    "max_rss_gb",
                ]
                # Header row
                f.write("| " + " | ".join(headers) + " |\n")
                # Separator row
                f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                # Data rows
                for (comp, real, user, sys_t, cpu_sum, avg_threads,
                     rss_kb, rss_mb, rss_gb) in rows:
                    f.write(
                        "| "
                        f"{comp} | "
                        f"{real:.2f} | {user:.2f} | {sys_t:.2f} | "
                        f"{cpu_sum:.2f} | {avg_threads:.2f} | "
                        f"{int(rss_kb)} | {rss_mb:.2f} | {rss_gb:.4f} |\n"
                    )
        except Exception:
            # ignore MD failures
            pass

    finally:
        # Always release lock
        try:
            os.remove(lock_path)
        except Exception:
            pass


html_path = Path(log_dir) / "comp-summary.html"
try:
    md_text = Path(md_path).read_text(encoding="utf-8", errors="ignore")
    table_html = markdown_table_to_html(md_text)

    html_doc = (
        "<!doctype html>\n"
        "<html>\n<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <title>TheRock Build Resource Summary</title>\n"
        "  <style>\n"
        "    body { font-family: Arial, sans-serif; margin: 24px; }\n"
        "    table { border-collapse: collapse; margin-top: 16px; }\n"
        "    th { background: #f0f0f0; }\n"
        "    th, td { padding: 8px 12px; text-align: left; }\n"
        "  </style>\n"
        "</head>\n<body>\n"
        f"{FAQ_HTML}\n"
        "<hr />\n"
        "<h2>Build Resource Summary</h2>\n"
        f"{table_html}\n"
        "</body>\n</html>\n"
    )

    html_path.write_text(html_doc, encoding="utf-8")
except Exception:
    pass

# --------------
# Entrypoint
# --------------

def main() -> int:
    log_dir = os.environ.get("THEROCK_BUILD_PROF_LOG_DIR",
                             "$(pwd)/build/logs/therock-build-prof")

    # 1) Run the compiler & log per-command stats
    rc = run_and_log_command(log_dir)

    # 2) Best-effort summarize (never breaks the build)
    try:
        generate_summaries(log_dir)
    except Exception:
        # Mask all errors; build result should depend only on the compiler execution.
        pass

    # Propagate real compiler exit code
    return rc


if __name__ == "__main__":
    sys.exit(main())
