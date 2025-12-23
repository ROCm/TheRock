#!/usr/bin/env python3
"""
Tooling for TheRock build system. The goal is to have single unified view across build resource utilization for TheRock compoments and systems locally as well as in CI

Features:
- Acts as CMake compiler launcher (C/C++).
- Logs per-command timing + memory usage to /build/log/therock-build-resources (or THEROCK_BUILD_PROF_LOG_DIR).
- Aggregates all logs into:
    - comp-summary.md   (Markdown table per-component)
    - comp-summary.html (FAQ + rendered HTML table)
- NEVER fails the build if profiling/reporting fails.
- Only propagates the real compiler's exit code.

Usage in CMake configure:

  cmake -S . -B build -GNinja \
    -DTHEROCK_AMDGPU_FAMILIES=gfx110X-all \
    -DCMAKE_C_COMPILER_LAUNCHER="${PWD}/build_tools/resource_info4.py" \
    -DCMAKE_CXX_COMPILER_LAUNCHER="${PWD}/build_tools/resource_info4.py"

Output:
    End summary for observability of build resources gets created in build/log/therock-build-prof/
"""

import os
import sys
import time
import random
import datetime
import subprocess
import shlex
import html
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


def _expand_rsp_args(argv: List[str]) -> List[str]:
    expanded: List[str] = []
    for a in argv:
        if a.startswith("@") and len(a) > 1:
            p = Path(a[1:])
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                expanded.extend(shlex.split(txt, posix=os.name != "nt"))
            except Exception:
                expanded.append(a)
        else:
            expanded.append(a)
    return expanded


def _guess_kind(cmd_args: List[str]) -> str:
    # Use expanded args to avoid misclassifying when -c is in @rsp.
    ex = _expand_rsp_args(cmd_args)

    if "-c" in ex:
        return "compile"

    # If it references object files or libraries and produces -o, treat as link.
    has_o = any(s.endswith((".o", ".obj")) for s in ex)
    has_lib = any(s.endswith((".a", ".lib")) for s in ex)
    has_out = "-o" in ex

    if has_out and (has_o or has_lib):
        return "link"

    # If no -c, link is more likely than compile.
    return "link"


def therock_components(_pwd_unused: str, cmd_str: str) -> str:
    comp = "unknown"
    lower_cmd = cmd_str.lower()

    for p in cmd_str.split():
        if p.startswith("@"):
            rsp_path = p[1:]
            try:
                lower_cmd += "\n" + Path(rsp_path).read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                pass

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
    kind = _guess_kind(cmd_args)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rand = random.randint(0, 999999)
    log_file = Path(log_dir) / f"build-{ts}-{rand}-{comp}.log"

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

            user_seconds = (
                (end_child.ru_utime - start_child.ru_utime)
                + (end_self.ru_utime - start_self.ru_utime)
            )
            sys_seconds = (
                (end_child.ru_stime - start_child.ru_stime)
                + (end_self.ru_stime - start_self.ru_stime)
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

    # STORE MINUTES IN LOGS (minutes only everywhere downstream)
    real_min = real_seconds / 60.0
    user_min = user_seconds / 60.0
    sys_min = sys_seconds / 60.0

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"comp={comp}\n")
            f.write(f"kind={kind}\n")
            f.write(f"cmd={cmd_str}\n")
            f.write(f"real_min={real_min:.6f}\n")
            f.write(f"user_min={user_min:.6f}\n")
            f.write(f"sys_min={sys_min:.6f}\n")
            f.write(f"maxrss_kb={maxrss_kb}\n")
    except Exception:
        pass

    return returncode


# ------------------------------------------------
# Aggregation / Resource Observability reporting
# ------------------------------------------------

def parse_log_file(path: Path, stats: Dict[Tuple[str, str], float]) -> None:
    comp = "unknown"
    kind = "compile"
    real_min = user_min = sys_min = None
    rss_kb = None

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("comp="):
                    comp = line[len("comp="):] or "unknown"
                elif line.startswith("kind="):
                    kind = line[len("kind="):] or "compile"
                elif line.startswith("real_min="):
                    try:
                        real_min = float(line[len("real_min="):])
                    except ValueError:
                        pass
                elif line.startswith("user_min="):
                    try:
                        user_min = float(line[len("user_min="):])
                    except ValueError:
                        pass
                elif line.startswith("sys_min="):
                    try:
                        sys_min = float(line[len("sys_min="):])
                    except ValueError:
                        pass
                elif line.startswith("maxrss_kb="):
                    try:
                        rss_kb = float(line[len("maxrss_kb="):])
                    except ValueError:
                        pass
    except Exception:
        return

    if real_min is not None:
        stats[(comp, "real_min")] = stats.get((comp, "real_min"), 0.0) + real_min
    if user_min is not None:
        stats[(comp, "user_min")] = stats.get((comp, "user_min"), 0.0) + user_min
    if sys_min is not None:
        stats[(comp, "sys_min")] = stats.get((comp, "sys_min"), 0.0) + sys_min

    if kind == "link" and real_min is not None:
        stats[(comp, "link_real_min")] = stats.get((comp, "link_real_min"), 0.0) + real_min

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


def _print_summary_links(log_dir: str) -> None:
    try:
        base = Path(log_dir).resolve()
        html_path = base / "comp-summary.html"
        md_path = base / "comp-summary.md"

        print("\n[TheRock Build Summary]")
        if html_path.exists():
            print(f"HTML: file://{html_path}")
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
            path = Path(log_dir) / name
            parse_log_file(path, stats)

        components = sorted({key[0] for key in stats.keys() if key[1] == "_seen"})
        if not components:
            return

        rows = []
        for comp in components:
            wall_min = stats.get((comp, "real_min"), 0.0)
            user_min = stats.get((comp, "user_min"), 0.0)
            sys_min = stats.get((comp, "sys_min"), 0.0)
            cpu_min = user_min + sys_min

            # avg_threads = (cpu_seconds / wall_seconds) == (cpu_min / wall_min)
            avg_threads = cpu_min / wall_min if wall_min > 0.0 else 0.0

            rss_kb = stats.get((comp, "rss_kb_max"), 0.0)
            rss_mb = rss_kb / 1024.0
            rss_gb = rss_kb / (1024.0 * 1024.0)

            link_wall_min = stats.get((comp, "link_real_min"), 0.0)

            rows.append((
                comp,
                wall_min,
                cpu_min,
                user_min,
                sys_min,
                avg_threads,
                rss_kb,
                rss_mb,
                rss_gb,
                link_wall_min,
            ))

        # Sort by cpu_sum_min descending (index 2)
        rows.sort(key=lambda r: r[2], reverse=True)

        md_path = Path(log_dir) / "comp-summary.md"
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                headers = [
                    "component",
                    "wall_time_min",
                    "cpu_sum_min",
                    "user_sum_min",
                    "sys_sum_min",
                    "avg_threads",
                    "max_rss_mb",
                    "max_rss_gb",
                    "link_wall_time_min",
                ]
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                for (comp, wall_min, cpu_min, user_min, sys_min, avg_threads,
                     rss_kb, rss_mb, rss_gb, link_wall_min) in rows:
                    f.write(
                        "| "
                        f"{comp} | "
                        f"{wall_min:.2f} | {cpu_min:.2f} | {user_min:.2f} | {sys_min:.2f} | "
                        f"{avg_threads:.2f} | "
                        f"{int(rss_kb)} | {rss_mb:.2f} | {rss_gb:.4f} | "
                        f"{link_wall_min:.2f} |\n"
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

        _print_summary_links(log_dir)

    finally:
        try:
            os.remove(str(lock_path))
        except Exception:
            pass


# --------------
# Entrypoint
# --------------

def main() -> int:
    # Stable default (no need to export): <repo_root>/build/log/therock-build-prof
    default_log_dir = str((Path(__file__).resolve().parents[1] / "build" / "logs" / "therock-build-prof"))
    log_dir = os.environ.get("THEROCK_BUILD_PROF_LOG_DIR", default_log_dir)

    rc = run_and_log_command(log_dir)

    try:
        generate_summaries(log_dir)
    except Exception:
        pass

    return rc


if __name__ == "__main__":
    sys.exit(main())

