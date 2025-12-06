#!/usr/bin/env python3
"""Analyze Ninja build times and generate HTML report.

This script parses the .ninja_log file from a build directory and generates
an HTML report showing build times for each ROCm component and dependency.

Usage:
    python analyze_build_times.py --build-dir <path> [--output <path>]

Arguments:
    --build-dir     Path to the build directory containing .ninja_log (required)
    --output        Path to output HTML file (optional)
                    Default: <build-dir>/logs/build_time_analysis.html

Examples:
    # Generate report with default output path
    python analyze_build_times.py --build-dir /path/to/build

    # Generate report with custom output path
    python analyze_build_times.py --build-dir /path/to/build --output report.html

CI Usage:
    In CI, this script is called automatically after build completion.
    The --build-dir is set to the CI build directory, and --output is optional.
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# =============================================================================
# Configuration
# =============================================================================

# Build name -> Display name mapping
NAME_MAPPING = {
    "clr": "core-hip",
    "ocl-clr": "core-ocl",
    "ROCR-Runtime": "core-runtime",
    "blas": "rocBLAS",
    "prim": "rocPRIM",
    "fft": "rocFFT",
    "rand": "rocRAND",
    "miopen": "MIOpen",
    "hipdnn": "hipDNN",
    "composable-kernel": "composable_kernel",
    "support": "mxDataGenerator",
    "host-suite-sparse": "SuiteSparse",
    "rocwmma": "rocWMMA",
    "miopen-plugin": "miopen_plugin",
}

# Top-level directories for ROCm components
ROCM_COMPONENT_DIRS = {
    "base",
    "compiler",
    "core",
    "comm-libs",
    "dctools",
    "profiler",
    "ml-libs",
}

# Regex to parse artifact filenames: <project>_<variant>[_suffix].tar.xz
# - Group 1: project name (e.g., "rocBLAS", "MIOpen")
# - Group 2: variant type (dbg=debug, dev=development, doc=documentation,
#            lib=library, run=runtime, test=test)
# - Group 3: optional suffix (e.g., "_gfx90a")
# Example: "rocBLAS_lib_gfx90a.tar.xz" -> ("rocBLAS", "lib", "_gfx90a")
ARTIFACT_REGEX = re.compile(r"(.+)_(dbg|dev|doc|lib|run|test)(_.+)?")

# Phase detection rules: (suffix/pattern, phase_name)
PHASE_RULES = [
    (lambda p: p.endswith("/stamp/configure.stamp"), "Configure"),
    (lambda p: p.endswith("/stamp/build.stamp"), "Build"),
    (lambda p: p.endswith("/stamp/stage.stamp"), "Install"),
    (lambda p: p.startswith("artifacts/") and p.endswith(".tar.xz"), "Package"),
    (lambda p: "download" in p and "stamp" in p, "Download"),
    (lambda p: "update" in p and "stamp" in p, "Update"),
]

# Category labels for grouping projects in the report
CATEGORY_ROCM = "ROCm Component"
CATEGORY_DEP = "Dependency"

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Task:
    start: int
    end: int
    output: str

    @property
    def duration(self) -> int:
        return self.end - self.start


# =============================================================================
# Parsing Functions
# =============================================================================


def parse_ninja_log(log_path: Path) -> List[Task]:
    """Parse .ninja_log and return list of Task objects."""
    tasks = []
    try:
        with open(log_path, "r") as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    tasks.append(
                        Task(start=int(parts[0]), end=int(parts[1]), output=parts[3])
                    )
    except FileNotFoundError:
        print(f"Error: Log file {log_path} not found.")
        sys.exit(1)
    return tasks


def get_phase(output_path: str) -> Optional[str]:
    """Detect build phase from output path."""
    for check, phase in PHASE_RULES:
        if check(output_path):
            return phase
    return None


def extract_name_from_artifact(filename: str) -> Optional[str]:
    """Extract project name from artifact filename."""
    base = filename.replace(".tar.xz", "")
    match = ARTIFACT_REGEX.match(base)
    name = match.group(1) if match else base
    return None if name in ("base", "sysdeps") else name


def parse_output_path(
    output_path: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract (name, category, phase) from output path."""
    phase = get_phase(output_path)
    if not phase:
        return None, None, None

    parts = output_path.split("/")
    top_dir = parts[0] if parts else ""

    # Artifact files
    if output_path.startswith("artifacts/"):
        name = extract_name_from_artifact(parts[1])
        if not name:
            return None, None, None
        category = (
            CATEGORY_DEP
            if ("sysdeps" in name or "fftw3" in name or name.startswith("host-"))
            else CATEGORY_ROCM
        )
        return NAME_MAPPING.get(name, name), category, phase

    # Third-party dependencies
    if top_dir == "third-party":
        if len(parts) > 3 and parts[1] == "sysdeps" and parts[2] in ("linux", "common"):
            name = parts[3]
        elif len(parts) > 1:
            name = parts[1]
        else:
            return None, None, None
        if name == "sysdeps":
            return None, None, None
        return NAME_MAPPING.get(name, name), CATEGORY_DEP, phase

    # ROCm components in standard directories
    if top_dir in ROCM_COMPONENT_DIRS:
        name = parts[1] if len(parts) > 1 else None
        if not name:
            return None, None, None
        return NAME_MAPPING.get(name, name), CATEGORY_ROCM, phase

    # rocm-libraries / rocm-systems
    if top_dir in ("rocm-libraries", "rocm-systems"):
        if len(parts) > 2 and parts[1] == "projects":
            name = parts[2]
            return NAME_MAPPING.get(name, name), CATEGORY_ROCM, phase
        return None, None, None

    # math-libs (special structure)
    if top_dir == "math-libs" and len(parts) > 1:
        if parts[1] == "BLAS":
            name = parts[2] if len(parts) > 2 else None
        elif parts[1] == "support" and len(parts) > 2:
            name = parts[2]
        else:
            name = parts[1]
        if not name:
            return None, None, None
        return NAME_MAPPING.get(name, name), CATEGORY_ROCM, phase

    return None, None, None


# =============================================================================
# Analysis
# =============================================================================


def analyze_tasks(
    tasks: List[Task], build_dir: Path
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Aggregate task durations by category/name/phase."""
    projects: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    seen = set()
    build_prefix = str(build_dir.resolve())

    for task in tasks:
        output = task.output
        if output.startswith(build_prefix):
            output = output[len(build_prefix) :].lstrip("/")

        key = (output, task.start, task.end)
        if key in seen:
            continue
        seen.add(key)

        name, category, phase = parse_output_path(output)
        if name:
            projects[category][name][phase] += task.duration

    return projects


# =============================================================================
# Report Generation
# =============================================================================


def get_system_info() -> Dict[str, str]:
    """Get build server system information."""
    info = {"cpu_model": "Unknown", "cpu_cores": "Unknown", "memory_gb": "Unknown"}

    # Get CPU model from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, IOError):
        pass

    # Get CPU cores
    try:
        info["cpu_cores"] = str(os.cpu_count() or "Unknown")
    except Exception:
        pass

    # Get memory from /proc/meminfo
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["memory_gb"] = f"{kb / 1024 / 1024:.1f}"
                    break
    except (FileNotFoundError, IOError, ValueError):
        pass

    return info


def calculate_wall_time(tasks: List[Task]) -> int:
    """Calculate wall clock time from earliest start to latest end."""
    if not tasks:
        return 0
    min_start = min(task.start for task in tasks)
    max_end = max(task.end for task in tasks)
    return max_end - min_start


def format_time_human(ms: int) -> str:
    """Format milliseconds to human readable time (e.g., 1h 23.5m or 45.32 min)."""
    minutes = ms / 60000
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    if hours > 0:
        return f"{hours}h {remaining_minutes:.1f}m"
    return f"{minutes:.2f} min"


def generate_system_info_html(tasks: List[Task]) -> str:
    """Generate HTML for build information section."""
    info = get_system_info()
    wall_time_str = format_time_human(calculate_wall_time(tasks))

    return f"""<div class="system-info">
    <h3>Build Information</h3>
    <p>CPU: <span>{info['cpu_model']}</span></p>
    <p>CPU Cores: <span>{info['cpu_cores']}</span></p>
    <p>Memory: <span>{info['memory_gb']} GB</span></p>
    <p>Build Duration: <span>{wall_time_str}</span></p>
</div>
"""


def format_duration(ms: int) -> str:
    """Convert milliseconds to formatted minutes string."""
    return "-" if ms == 0 else f"{ms / 60000:.2f}"


def build_table_rows(
    data: Dict[str, Dict[str, int]], phase_columns: List[str]
) -> List[tuple]:
    """Build sorted table rows from project phase data."""
    rows = []
    for name, phases in data.items():
        total = sum(phases.values())
        cols = [format_duration(phases.get(p, 0)) for p in phase_columns]
        rows.append((name, cols, format_duration(total), total))
    rows.sort(key=lambda x: x[3], reverse=True)
    return [(r[0], r[1], r[2]) for r in rows]


def generate_html_table(title: str, headers: List[str], rows: List[tuple]) -> str:
    """Generate HTML table with title, headers, and row data."""
    if not rows:
        return ""

    lines = [
        f"<h2>{title}</h2>",
        "<table>",
        "<thead><tr>",
        "".join(f"<th>{h}</th>" for h in headers),
        "</tr></thead>",
        "<tbody>",
    ]

    for name, cols, total in rows:
        cells = (
            f"<td>{name}</td>"
            + "".join(f"<td>{v}</td>" for v in cols)
            + f'<td class="total-col">{total}</td>'
        )
        lines.append(f"<tr>{cells}</tr>")

    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines) + "\n"


def generate_report(projects: Dict, tasks: List[Task], output_file: Path):
    """Generate HTML report from analyzed project data."""
    # ROCm Components table
    rocm_data = projects.get(CATEGORY_ROCM, {})
    rocm_rows = build_table_rows(
        rocm_data, ["Configure", "Build", "Install", "Package"]
    )
    rocm_html = generate_html_table(
        "ROCm Components",
        [
            "Sub-Project",
            "Configure (min)",
            "Build (min)",
            "Install (min)",
            "Package (min)",
            "Total (min)",
        ],
        rocm_rows,
    )

    # Dependencies table (combine Download + Update into single column)
    dep_data = {}
    for name, phases in projects.get(CATEGORY_DEP, {}).items():
        dep_data[name] = {
            "Download": phases.get("Download", 0) + phases.get("Update", 0),
            "Configure": phases.get("Configure", 0),
            "Build": phases.get("Build", 0),
            "Install": phases.get("Install", 0),
        }
    dep_rows = build_table_rows(dep_data, ["Download", "Configure", "Build", "Install"])
    dep_html = generate_html_table(
        "Dependencies",
        [
            "Sub-Project",
            "Download (min)",
            "Configure (min)",
            "Build (min)",
            "Install (min)",
            "Total (min)",
        ],
        dep_rows,
    )

    # Generate build info (system info + build times)
    system_html = generate_system_info_html(tasks)

    # Load template and generate output
    template_path = Path(__file__).resolve().parent / "report_build_time_template.html"
    try:
        template = template_path.read_text()
        html = (
            template.replace("{{SYSTEM_INFO}}", system_html)
            .replace("{{ROCM_TABLE}}", rocm_html)
            .replace("{{DEP_TABLE}}", dep_html)
        )
        output_file.write_text(html)
        print(f"HTML report generated at: {output_file}")
    except FileNotFoundError:
        print(f"Error: Template file not found at {template_path}")
    except Exception as e:
        print(f"Error generating report: {e}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze Ninja build times")
    parser.add_argument(
        "--build-dir", type=Path, required=True, help="Path to build directory"
    )
    parser.add_argument("--output", type=Path, help="Path to output HTML file")
    args = parser.parse_args()

    ninja_log = args.build_dir / ".ninja_log"
    if not ninja_log.exists():
        print(f"Error: {ninja_log} not found.")
        sys.exit(1)

    tasks = parse_ninja_log(ninja_log)
    projects = analyze_tasks(tasks, args.build_dir)

    output_file = args.output or args.build_dir / "logs" / "build_time_analysis.html"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    generate_report(projects, tasks, output_file)


if __name__ == "__main__":
    main()
