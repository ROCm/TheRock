#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import re
from typing import List, Dict, Tuple, Optional, Set

# --- Configuration & Constants ---

# Mapping for components with different build/package names
NAME_MAPPING = {
    'clr': 'core-hip',
    'ocl-clr': 'core-ocl',
    'ROCR-Runtime': 'core-runtime',
    'blas': 'rocBLAS',
    'prim': 'rocPRIM',
    'fft': 'rocFFT',
    'rand': 'rocRAND',
    'miopen': 'MIOpen',
    'hipdnn': 'hipDNN',
    'composable-kernel': 'composable_kernel',
    'support': 'mxDataGenerator',
    'host-suite-sparse': 'SuiteSparse',
    'rocwmma': 'rocWMMA',
    'miopen-plugin': 'miopen_plugin',
    'rccl-tests': 'rccl'
}

# Directories that are considered ROCm Components (whitelist)
ROCM_COMPONENT_DIRS = {
    'base', 'compiler', 'core', 'comm-libs', 'dctools', 'profiler', 'ml-libs'
}

# Regex to capture name and variant from artifact filenames
ARTIFACT_REGEX = re.compile(r'(.+)_(dbg|dev|doc|lib|run|test)(_.+)?')

# --- Core Logic ---

def parse_ninja_log(log_path: Path) -> List[Dict]:
    """Parses .ninja_log file."""
    tasks = []
    try:
        with open(log_path, 'r') as f:
            header = f.readline() # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                start, end, _, output, _ = parts[:5]
                tasks.append({
                    'start': int(start),
                    'end': int(end),
                    'output': output
                })
    except FileNotFoundError:
        print(f"Error: Log file {log_path} not found.")
        sys.exit(1)
    return tasks

def get_phase(output_path: str) -> Optional[str]:
    """Detects the build phase based on the output path suffix or patterns."""
    if output_path.endswith('/stamp/configure.stamp'):
        return 'Configure'
    elif output_path.endswith('/stamp/build.stamp'):
        return 'Build'
    elif output_path.endswith('/stamp/stage.stamp'):
        return 'Install'
    elif output_path.startswith('artifacts/') and output_path.endswith('.tar.xz'):
        return 'Package'
    elif 'download' in output_path and ('stamp' in output_path or output_path.endswith('.stamp')):
        return 'Download'
    elif 'update' in output_path and 'stamp' in output_path:
        return 'Update'
    return None

def parse_output_path(output_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parses the output path to identify: Name, Category, Phase."""
    phase = get_phase(output_path)
    if not phase:
        return None, None, None

    parts = output_path.split('/')
    name = "Unknown"
    category = "ROCm Component" # Default

    if output_path.startswith('artifacts/'):
        filename = parts[1]
        base = filename.replace('.tar.xz', '')
        m = ARTIFACT_REGEX.match(base)
        if m:
            name = m.group(1)
        else:
            name = base

        if name in ('base', 'sysdeps'):
            return None, None, None

        if 'sysdeps' in name or 'fftw3' in name or name.startswith('host-'):
            category = "Dependency"
    
    elif parts[0] == 'third-party':
        category = "Dependency"
        if len(parts) > 3 and parts[1] == 'sysdeps' and parts[2] in ['linux', 'common']:
            name = parts[3]
        elif len(parts) > 1:
            name = parts[1]
        if name == 'sysdeps':
             return None, None, None

    # ROCM_COMPONENT_DIRS contains 'rocm-libraries' and 'rocm-systems', but the logic above handles them specially
    # to extract the project name (3rd level) instead of the 2nd level directory.
    # 'rocm-libraries' and 'rocm-systems' are handled explicitly to support deeper directory structures
    elif parts[0] in ['rocm-libraries', 'rocm-systems']:
        category = "ROCm Component"
        if len(parts) > 2 and parts[1] == 'projects':
            name = parts[2]

    elif parts[0] in ROCM_COMPONENT_DIRS:
        category = "ROCm Component"
        if len(parts) > 1:
            name = parts[1]

    elif parts[0] == 'math-libs':
        category = "ROCm Component"
        if len(parts) > 1:
            if parts[1] == 'BLAS':
                 if len(parts) > 2:
                    name = parts[2]
                 else:
                     return None, None, None
            elif parts[1] == 'support' and len(parts) > 2:
                name = parts[2]
            else:
                name = parts[1]

    else:
        return None, None, None

    if name in NAME_MAPPING:
        name = NAME_MAPPING[name]

    return name, category, phase

def analyze_tasks(tasks: List[Dict], build_dir: Path) -> Dict:
    projects = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    seen_tasks = set()

    build_dir_abs = str(build_dir.resolve())
    build_dir_len = len(build_dir_abs)

    for task in tasks:
        output_path = task['output']
        start = task['start']
        end = task['end']

        if output_path.startswith(build_dir_abs):
            output_path = output_path[build_dir_len:].lstrip('/')

        task_key = (output_path, start, end)
        if task_key in seen_tasks:
            continue
        seen_tasks.add(task_key)

        name, category, phase = parse_output_path(output_path)
        if not name:
            continue

        duration = task['end'] - task['start']
        projects[category][name][phase] += duration

    return projects

def format_duration(ms: int) -> str:
    if ms == 0:
        return "-"
    seconds = ms / 1000.0
    return f"{seconds:.2f}"

def generate_html_table(title: str, headers: List[str], rows: List[Tuple]) -> str:
    """Generates an HTML table string."""
    if not rows:
        return ""

    html = f"<h2>{title}</h2>\n"
    html += "<table>\n<thead>\n<tr>\n"
    for h in headers:
        html += f"<th>{h}</th>\n"
    html += "</tr>\n</thead>\n<tbody>\n"

    for row in rows:
        html += "<tr>\n"
        html += f"<td>{row[0]}</td>\n" # Name
        for val in row[1]: # Columns
            html += f"<td>{val}</td>\n"
        html += f"<td class=\"total-col\">{row[2]}</td>\n" # Total
        html += "</tr>\n"

    html += "</tbody>\n</table>\n"
    return html

def generate_report(projects: Dict, output_file: Path):
    # Prepare Data for ROCm Components
    rocm_rows = []
    if "ROCm Component" in projects:
        for name, phases in projects["ROCm Component"].items():
            total = sum(phases.values())
            cols = [
                format_duration(phases['Configure']),
                format_duration(phases['Build']),
                format_duration(phases['Install']),
                format_duration(phases['Package'])
            ]
            rocm_rows.append((name, cols, format_duration(total), total))
        
        rocm_rows.sort(key=lambda x: x[3], reverse=True)
        rocm_rows = [(r[0], r[1], r[2]) for r in rocm_rows]

    rocm_table_html = generate_html_table(
        "ROCm Components", 
        ["Sub-Project", "Configure (s)", "Build (s)", "Install (s)", "Package (s)", "Total Time (s)"],
        rocm_rows
    )

    # Prepare Data for Dependencies
    dep_rows = []
    if "Dependency" in projects:
        for name, phases in projects["Dependency"].items():
            total = sum(phases.values())
            download_time = phases['Download'] + phases['Update']
            cols = [
                format_duration(download_time),
                format_duration(phases['Configure']),
                format_duration(phases['Build']),
                format_duration(phases['Install'])
            ]
            dep_rows.append((name, cols, format_duration(total), total))
        
        dep_rows.sort(key=lambda x: x[3], reverse=True)
        dep_rows = [(r[0], r[1], r[2]) for r in dep_rows]

    dep_table_html = generate_html_table(
        "Dependencies",
        ["Sub-Project", "Download (s)", "Configure (s)", "Build (s)", "Install (s)", "Total Time (s)"],
        dep_rows
    )

    # Load Template
    script_dir = Path(__file__).resolve().parent
    template_path = script_dir / "report_build_time_template.html"
    
    try:
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Replace placeholders
        full_html = template.replace("{{ROCM_TABLE}}", rocm_table_html)
        full_html = full_html.replace("{{DEP_TABLE}}", dep_table_html)

        with open(output_file, 'w') as f:
            f.write(full_html)
        print(f"HTML report generated at: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Template file not found at {template_path}")
    except Exception as e:
        print(f"Error generating report: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Ninja build times")
    parser.add_argument("--build-dir", type=Path, required=True, help="Path to build directory")
    parser.add_argument("--output", type=Path, help="Path to output HTML file")
    args = parser.parse_args()

    ninja_log = args.build_dir / ".ninja_log"
    if not ninja_log.exists():
        print(f"Error: {ninja_log} not found.")
        sys.exit(1)

    tasks = parse_ninja_log(ninja_log)
    projects = analyze_tasks(tasks, args.build_dir)

    if args.output:
        output_html = args.output
    else:
        output_html = args.build_dir / "logs" / "build_time_analysis.html"
        output_html.parent.mkdir(parents=True, exist_ok=True)

    generate_report(projects, output_html)

if __name__ == "__main__":
    main()
