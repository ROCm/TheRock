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
    -DCMAKE_C_COMPILER_LAUNCHER="${PWD}/build_tools/resource_info3.py" \
    -DCMAKE_CXX_COMPILER_LAUNCHER="${PWD}/build_tools/resource_info3.py"


  cmake --build build
          OR
  ninja -C build -j"$(nproc)"


Output:
    End summary for observability of build resources gets created in /build/log/comp-summary.md /build/log/comp-summary.csv file
"""

import os
import sys
import time
import random
import datetime
import subprocess
import shlex
import html
from typing import Dict, Tuple
from pathlib import Path

# Best-effort resource usage (not available on Windows)
try:
    import resource  # type: ignore
except Exception:
    resource = None


# -------------------------
# Component classification
# -------------------------

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
<h2>Generic FAQ</h2>

<ol>
  <li><b>RSS = Resident Set Size</b>
    <p>
      It means the amount of physical RAM actually in use by a process at its peak. High RSS means memory-heavy compilation or linking.
      If RSS is high across many parallel jobs, we can hit swapping, cache thrashing, OOM kills in CI. High RSS often limits how much
      parallelism you can safely use (-j)
    </p>
  </li>

  <li><b>max_rss_mb</b>
    <p>
      the maximum RAM a compiler or linker process had allocated in memory at any point.
    </p>
  </li>

  <li><b>What Avg Threads (avg_threads) actually mean ?</b>
    <p><b>Case 1&gt; avg_threads ≈ 1.0</b></p>
    <p>This means process used about one CPU core for most of its runtime.</p>
    <p><b>Typical causes:</b></p>
    <ul>
      <li>Normal C/C++ compilation (clang/gcc front-end is mostly single-threaded) Serial link steps (ld, ar)</li>
      <li>Code generation steps</li>
    </ul>
    <p><b>Interpretation:</b></p>
    <ul>
      <li>This is expected for many build steps.</li>
      <li>Increasing -j will not make this step faster — only running more steps in parallel helps.</li>
    </ul>

    <p><b>Case 2&gt; avg_threads &lt; 1.0</b></p>
    <p><b>Meaning:</b> The process spent a lot of time waiting, not computing.</p>
    <p><b>Common reasons:</b></p>
    <ul>
      <li>I/O waits (reading headers, writing object files)</li>
      <li>Lock contention</li>
      <li>Process startup / scheduling overhead</li>
      <li>Throttling / CPU starvation in CI</li>
    </ul>
    <p><b>Interpretation:</b></p>
    <ul>
      <li>The step is not CPU-bound.</li>
      <li>Faster disks, better caching, or reducing process churn may help more than more CPUs.</li>
    </ul>

    <p><b>Case 3&gt; avg_threads &gt; 1.0</b></p>
    <p><b>Meaning:</b> The process used multiple CPU cores simultaneously.</p>
    <p><b>How this happens in builds:</b></p>
    <ul>
      <li>Parallel LTO / ThinLTO backends</li>
      <li>LLVM optimizations using worker threads</li>
      <li>Compiler spawning helper threads</li>
      <li>Some linkers using parallelism</li>
    </ul>
    <p><b>Interpretation:</b></p>
    <ul>
      <li>This step benefits from more cores.</li>
      <li>It can reduce total wall time but increase total CPU time.</li>
    </ul>
  </li>
</ol>

<h3>Key mental model</h3>
<ul>
  <li><b>Wall Time</b> → affects how long the build takes</li>
  <li><b>CPU Time</b> → affects how expensive the build is</li>
  <li><b>Avg Threads</b> → tells how much parallelism each step actually used</li>
  <li><b>RSS</b> → limits how many steps we can run at once</li>
</ul>
"""


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
        return 0

    os.makedirs(log_dir, exist_ok=True)

    pwd = os.getcwd()
    cmd_args = sys.argv[1:]
    cmd_str = " ".join(shlex.quote(arg) for arg in cmd_args)

    comp = therock_components(pwd, cmd_str)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rand = random.randint(0, 999999)
    log_file = Path(log_dir) / f"build-{ts}-{rand}-{comp}.log"

    start_wall = time.monotonic()

    # CPU timers (portable)
    start_cpu = time.process_time()

    # Best-effort rusage (non-portable)
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
    real_time = end_wall - start_wall

    end_cpu = time.process_time()
    cpu_time = end_cpu - start_cpu  # fallback total CPU time

    user_time = 0.0
    sys_time = 0.0
    maxrss_kb = 0

    if resource is not None and start_self is not None and start_child is not None:
        try:
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
            maxrss_kb = int(end_child.ru_maxrss)
        except Exception:
            # Fall back to total CPU time if split timing fails
            user_time = cpu_time
            sys_time = 0.0
            maxrss_kb = 0
    else:
        # Windows / no resource module: provide portable CPU total only
        user_time = cpu_time
        sys_time = 0.0
        maxrss_kb = 0

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"comp={comp}\n")
            f.write(f"cmd={cmd_str}\n")
            f.write(f"real={real_time:.6f}\n")
            f.write(f"user={user_time:.6f}\n")
            f.write(f"sys={sys_time:.6f}\n")
            f.write(f"maxrss_kb={maxrss_kb}\n")
    except Exception:
        pass

    return returncode


# ------------------------------------------------
# Aggregation / Resource Observability reporting
# ------------------------------------------------

def parse_log_file(path: Path, stats: Dict[Tuple[str, str], float]) -> None:
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
        return

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
            path = Path(log_dir) / name
            parse_log_file(path, stats)

        components = sorted({key[0] for key in stats.keys() if key[1] == "_seen"})
        if not components:
            return

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
            rows.append((comp, real, user, sys_t, cpu_sum, avg_threads, rss_kb, rss_mb, rss_gb))

        rows.sort(key=lambda r: r[4], reverse=True)

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
                for (comp, real, user, sys_t, cpu_sum, avg_threads, rss_kb, rss_mb, rss_gb) in rows:
                    comp_str = f"\"{comp}\"" if "," in comp else comp
                    f.write(
                        f"{comp_str},"
                        f"{real:.6f},{user:.6f},{sys_t:.6f},"
                        f"{cpu_sum:.6f},{avg_threads:.6f},"
                        f"{int(rss_kb)},{rss_mb:.6f},{rss_gb:.8f}\n"
                    )
        except Exception:
            pass

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
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                for (comp, real, user, sys_t, cpu_sum, avg_threads, rss_kb, rss_mb, rss_gb) in rows:
                    f.write(
                        "| "
                        f"{comp} | "
                        f"{real:.2f} | {user:.2f} | {sys_t:.2f} | "
                        f"{cpu_sum:.2f} | {avg_threads:.2f} | "
                        f"{int(rss_kb)} | {rss_mb:.2f} | {rss_gb:.4f} |\n"
                    )
        except Exception:
            pass

        html_path = Path(log_dir) / "comp-summary.html"
        try:
            md_text = md_path.read_text(encoding="utf-8", errors="ignore")
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

    finally:
        try:
            os.remove(str(lock_path))
        except Exception:
            pass


# --------------
# Entrypoint
# --------------

def main() -> int:
    log_dir = os.environ.get(
    "THEROCK_BUILD_PROF_LOG_DIR",
    str((Path(__file__).resolve().parents[1] / "build" / "logs" / "therock-build-prof"))
    )

    rc = run_and_log_command(log_dir)

    try:
        generate_summaries(log_dir)
    except Exception:
        pass

    return rc


if __name__ == "__main__":
    sys.exit(main())

