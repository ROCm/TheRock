#!/usr/bin/env python3
"""
author: shiraz.ali@amd.com

Tooling for TheRock build system. The goal is to have single unified view across build resource utilization for therock compoments and systems locally as well as in CI

Features:
- Acts as CMake compiler launcher (C/C++).
- Logs per-command timing + memory usage to /tmp/therock-build-resources (or THEROCK_BUILD_PROF_LOG_DIR).
- Aggregates all logs into:
    - comp-summary.csv  (per-component summary)
    - comp-summary.md   (Markdown table per-component)
- NEVER fails the build if profiling/reporting fails.
  
- Only propagates the real compiler's exit code.

Usage in CMake configure:

  cmake -S . -B build -GNinja \
    -DTHEROCK_AMDGPU_FAMILIES=gfx110X-all \
    -DCMAKE_C_COMPILER_LAUNCHER="${PWD}/build_tools/therock_build_observability.py" \
    -DCMAKE_CXX_COMPILER_LAUNCHER="${PWD}/build_tools/therock_build_observability.py"


  cmake --build build 
          OR 
  ninja -C build -j"$(nproc)"


Output: 
    End summary for observability of build resources gets created in /tmp/comp-summary.md /tmp/comp-summary.csv file 


"""

import os
import sys
import time
import random
import datetime
import resource
import subprocess
import shlex
from typing import Dict, Tuple


# -------------------------
# Component classification
# -------------------------

def guess_component_from_pwd_and_cmd(pwd: str, cmd_str: str) -> str:
    """ when not clear to infer which component this compile belongs to."""
    comp = "unknown"

    # Infer from build directory (PWD). Can be edited, added as needed
    if "/core/" in pwd:
        comp = "core"
    elif "/compiler/" in pwd:
        comp = "compiler"
    elif "/math-libs/" in pwd:
        comp = "math-libs"
    elif "/ml-libs/" in pwd:
        comp = "ml-libs"
    elif "/profiler/" in pwd:
        comp = "profiler"
    elif "/dctools/" in pwd:
        comp = "dctools"
    elif "/rocm-libraries/" in pwd:
        comp = "rocm-libraries"
    elif "/rocm-systems/" in pwd:
        comp = "rocm-systems"

    # Define command text (per-library)
    lower_cmd = cmd_str.lower()
    if "rocblas" in lower_cmd:
        comp = "rocblas"
    elif "rocsolver" in lower_cmd:
        comp = "rocsolver"
    elif "rocfft" in lower_cmd:
        comp = "rocfft"
    elif "miopen" in lower_cmd:
        comp = "miopen"
    elif "rccl" in lower_cmd:
        comp = "rccl"
    elif "hipblaslt" in lower_cmd:
        comp = "hipblaslt"

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

    comp = guess_component_from_pwd_and_cmd(pwd, cmd_str)

    # Unique log filename. note that the log files are not getting saved anywhere else other than /tmp and can be cleared post reporting operation 
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rand = random.randint(0, 999999)
    log_file = os.path.join(log_dir, f"build-{ts}-{rand}-{comp}.log")

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


def generate_summaries(log_dir: str) -> None:
    """
    Scan all *.log files in log_dir and generate:
      - comp-summary.csv
      - comp-summary.md
      - Uses a simple lock file to avoid concurrent writers.
      - Any failure is masked so builds never break.
    """
    # Simple file-based lock to avoid concurrent summary writers
    lock_path = os.path.join(log_dir, ".summary.lock")
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
            path = os.path.join(log_dir, name)
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
        csv_path = os.path.join(log_dir, "comp-summary.csv")
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
        md_path = os.path.join(log_dir, "comp-summary.md")
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


# --------------
# Entrypoint
# --------------

def main() -> int:
    log_dir = os.environ.get("THEROCK_BUILD_PROF_LOG_DIR",
                             "/tmp/therock-build-prof")

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
