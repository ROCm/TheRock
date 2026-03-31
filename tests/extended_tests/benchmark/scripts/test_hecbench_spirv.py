# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
HeCBench SPIR-V Benchmark Test

Runs HeCBench HIP benchmarks compiled with --offload-arch=amdgcnspirv
and reports results.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # For extended_tests/utils
sys.path.insert(0, str(Path(__file__).parent))  # For benchmark_base
from benchmark_base import BenchmarkBase
from utils.logger import log

THEROCK_DIR = Path(__file__).resolve().parents[4]

TIME_UNIT_PATTERN = re.compile(
    r"(?i)^(?:s|sec|secs|second|seconds|ms|msec|millisecond|milliseconds|us|usec|microsecond|microseconds|ns|nanosecond|nanoseconds)$"
)

COMMON_TIME_PATTERNS: List[str] = [
    r"(?:Average[^\n]*?(?:time|execution time)[^\n]*?:\s*)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?:Average[^\n]*?(?:time|execution time)[^\n]*?:\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%-]*)",
    r"(?:Total[^\n]*?(?:time|execution time)[^\n]*?:\s*)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?:kernel execution time[^\n]*?:\s*)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?:elapsed time[^\n]*?=\s*)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?:time\s*=\s*)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?:,\s*Time\s*=\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%-]*)",
    r"(?i)(?:average\s*time\((?P<unit>[^\n)]+)\)\s*:\s*)(?P<score>[0-9.+eE-]+)",
    r"(?i)average[^\n:]*time[^\n:]*:\s*(?P<score>[0-9.+eE-]+)\s*(?:\((?P<unit>[^\n)]+)\)|(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%-]*))?",
    r"(?i)(?:co-?execution\s*time\s*:\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%-]*)",
    r"(?i)(?:\bruntime\b[^\n:]*:\s*)(?P<score>[0-9.+eE-]+)\s*(?:\((?P<unit>[^\n)]+)\)|\[?(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)\]?)",
    r"(?i)(?P<score>[0-9.+eE-]+)\s*#\s*runtime\s*\[(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)\]",
    r"(?i)(?:compute\s+time\s*:\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:\b(?:cpu|gpu)\s*time\s*:\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:\b(?:host|device)(?:\s+execution)?\s+time\s*[:=]\s*)(?P<score>[0-9.+eE-]+)(?:\s*(?:\((?P<unit>[^\n)]+)\)|(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)))?",
    r"(?i)(?:\b(?:device\s+)?execution\s+time[^\n:]*:\s*)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?i)(?:\bdevice\s+offload\s+time\s*:\s*)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?i)(?:\bsolve\s+time\s*\((?P<unit>[^\n)]+)\)\s*:\s*)(?P<score>[0-9.+eE-]+)",
    r"(?i)(?:\bdone\s+in\s+)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)\s+for\s+[0-9]+\s+iterations",
    r"(?i)(?:\bbenchmarking[^\n.]*\.\.\.\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:total[^\n:]*execution\s+time\s*:?\s*)(?P<score>[0-9.+eE-]+)\s*(?:\((?P<unit>[^\n)]+)\)|(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*))",
    r"(?i)(?:kernel\s+ex(?:ecution|euction)\s+time[^\n=]*=\s*)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?i)(?:kernel\s+timing[^\n]*?avg\s*=\s*)(?P<score>[0-9.+eE-]+)",
    r"(?i)(?:total\s+kernel\s+execution\s+)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?i)(?:execution\s+time\s*\((?P<unit>[^\n)]+)\)\s*:\s*)(?P<score>[0-9.+eE-]+)",
    r"(?i)(?:total\s+time\s*\((?P<unit>[^\n)]+)\)\s*:\s*)(?P<score>[0-9.+eE-]+)",
    r"(?i)(?:total\s*time\s*:\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%-]*)",
    r"(?i)total\s+time[^\n:]*:\s*(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:\btotal\s+(?:trial|kernel|wall)\s+time[^\n]*?)(?P<score>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:avg\.?\s*time[^\n:]*:?\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:\bavg\.?\s*time\s+)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:\bcopy\b[^\n]*?\btakes\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:\b(?:hipmalloc(?:managed)?|hiphostmalloc|hipfree)\([^\n)]*\)\s+takes\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:\btime\s*=\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)\s*,",
    r"(?i)(?:\bmatch(?:cpu|gpu)\d+\s*:\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:\btotal\s*:\s*)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?i)(?:\btotal\s+kernel\s+execution\s+time[^\n]*?\s+)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?i)(?:\btotal\s+kernel\s+time[^\n:]*?\s+)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?i)(?:total\s+kernel\s+time\s*:\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:total\s+running\s+time\s*:\s*)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?i)(?:total\s+elapsed\s+time\s+)(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)",
    r"(?i)(?:total\s+wall\s+time\s*=\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:wall\s+time\s*=\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:\belapsed\s*:\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:elapsed\s+time[^\n:]*:\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:\b(?:enqueue\s+rate|single\s+dispatch\s+latency|batch\s+dispatch\s+latency)\s*:\s*mean\s*=\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)\|\s*time\s+(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[^\s|]+)\s*\|",
    r"(?i)(?:\bmeasured\s+time\s+for\s+sample\s*=\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)(?:\bkernel\s+time\s*=\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)\btime\b[^\n]*?\biterations\s+(?P<score>[0-9.+eE-]+)\s*\((?P<unit>[^\n)]+)\)\s*$",
    r"(?i)\bcomputed[^\n]*?\bin\s+(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)\b",
    r"(?i)(?:average\s+kernel\s+time\s*:\s*)(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%-]*)",
]

COMMON_THROUGHPUT_PATTERNS: List[str] = [
    r"(?i)^Average\s+[A-Za-z0-9_./-]+\s+(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)\s*$",
    r"(?i)\b(?P<unit_plain>GDOF/s)\s*=\s*(?P<score>[0-9.+eE-]+)",
    r"(?i)\b(?:the\s+)?average\s+performance[^\n]*?\bis\s*(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)^\s*(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>(?:[KMGTP]?B(?:ytes)?/s|MLUPS|GFLOPS/s|GDOF/s))\s*$",
    r"(?i)peer-to-peer\s+copy[^\n:]*:\s*(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)",
    r"(?i)^total\s+clocks\s*=\s*(?P<score>[0-9.+eE-]+)\s*$",
]

OUTPUT_FAILURE_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)\bverify\s*=\s*fail\b", "verification failed"),
    (r"(?i)\bverification\s*:\s*fail\b", "verification failed"),
    (r"(?i)\bincorrect\s*:\s*\[", "incorrect results reported"),
    (r"(?i)\bhip error\b", "hip runtime error reported"),
    (r"(?i)\binput hash format error\b", "input hash format error"),
    (r"(?i)\bcopying of .* failed\b", "device-to-host copy failure reported"),
]


def _load_expected_failures() -> set[str]:
    expected_failures_raw = os.getenv("HECBENCH_EXPECTED_FAILURES")
    if expected_failures_raw is None:
        return set()

    value = expected_failures_raw.strip()
    if not value or value.lower() in {"none", "off", "false", "0"}:
        return set()

    return {name.strip().lower() for name in value.split(",") if name.strip()}


def _load_run_timeout(default_seconds: int = 180) -> int:
    raw_value = os.getenv("HECBENCH_RUN_TIMEOUT", str(default_seconds)).strip()
    try:
        timeout = int(raw_value)
    except ValueError:
        log.warning(
            "Invalid HECBENCH_RUN_TIMEOUT value '%s'; using default %d seconds",
            raw_value,
            default_seconds,
        )
        return default_seconds

    if timeout <= 0:
        log.warning(
            "HECBENCH_RUN_TIMEOUT must be > 0 (got %d); using default %d seconds",
            timeout,
            default_seconds,
        )
        return default_seconds
    return timeout


def _is_time_unit(unit: str) -> bool:
    return bool(TIME_UNIT_PATTERN.match(unit.strip()))


def _extract_score_and_unit(match: re.Match[str]) -> Tuple[Optional[float], str]:
    groups = match.groupdict()
    score_text = str(groups.get("score", "")).strip().rstrip(",;")
    if not score_text:
        return None, ""
    try:
        score = float(score_text)
    except ValueError:
        return None, ""
    unit = (
        str(groups.get("unit") or groups.get("unit_plain") or "").strip().rstrip(".,;:")
    )
    unit_lower = unit.lower()
    if unit_lower in {"sec", "secs", "second", "seconds"}:
        unit = "s"
    elif unit_lower in {"msec", "millisecond", "milliseconds"}:
        unit = "ms"
    elif unit_lower in {"usec", "microsecond", "microseconds"}:
        unit = "us"
    elif unit_lower in {"nanosecond", "nanoseconds"}:
        unit = "ns"
    elif unit_lower.endswith("bytes/sec"):
        # Normalize bandwidth units for cleaner reporting and stable H/L flagging.
        prefix = unit_lower.removesuffix("bytes/sec")
        unit = f"{prefix.upper()}B/s" if prefix else "B/s"
    return score, unit


def _extract_metric_from_output(output: str) -> Tuple[Optional[float], Optional[str]]:
    lines = output.splitlines()

    # Pass 1: Prefer time metrics whenever present.
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        lower_line = line.lower()

        # Ignore obvious failure lines that can contain misleading numeric tokens.
        if any(
            token in lower_line
            for token in ("error", "failed", "traceback", "exception")
        ):
            continue

        # Some benchmarks print "... time:" and the numeric value on the next line.
        if re.search(
            r"(?i)\b(?:total\s+time|device\s+offloading\s+time|kernel\s+execution\s+time)\s*:\s*$",
            line,
        ):
            for next_line in lines[index + 1 : index + 4]:
                next_line = next_line.strip()
                if not next_line:
                    continue
                next_match = re.search(
                    r"(?P<score>[0-9.+eE-]+)\s*(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*)\s*$",
                    next_line,
                )
                if not next_match:
                    continue
                score, unit = _extract_score_and_unit(next_match)
                if score is None:
                    continue
                if unit and not _is_time_unit(unit):
                    continue
                return score, unit
            continue

        # Fast path for common line endings like "... <score> <unit>".
        if "time" in lower_line and (
            "average" in lower_line
            or lower_line.startswith("runtime")
            or lower_line.startswith("total time")
            or lower_line.startswith("co-execution time")
            or lower_line.startswith("co execution time")
            or lower_line.startswith("average kernel time")
            or ", time =" in lower_line
        ):
            tail_match = re.search(
                r"(?P<score>[0-9]+(?:\.[0-9]+)?)\s*(?:\((?P<unit>[^\n)]+)\)|(?P<unit_plain>[A-Za-z][A-Za-z0-9_/%.-]*))\s*$",
                line,
            )
            if tail_match:
                score, unit = _extract_score_and_unit(tail_match)
                if score is not None and (not unit or _is_time_unit(unit)):
                    return score, unit

        for pattern in COMMON_TIME_PATTERNS:
            try:
                match = re.search(pattern, line)
            except re.error:
                log.debug("Skipping invalid common regex: %s", pattern)
                continue
            if not match:
                continue

            score, unit = _extract_score_and_unit(match)
            if score is None:
                continue
            if not unit and (
                "host time" in lower_line
                or "device time" in lower_line
                or "kernel timing" in lower_line
            ):
                unit = "s"
            if unit and not _is_time_unit(unit):
                continue
            return score, unit

    # Pass 2: Fallback to throughput metrics only if no time metric was found.
    in_babelstream_table = False
    in_mlups_block = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        lower_line = line.lower()
        if any(
            token in lower_line
            for token in ("error", "failed", "traceback", "exception")
        ):
            continue

        # Babelstream prints a table where the unit appears in the header.
        if (
            "function" in lower_line
            and "mbytes/sec" in lower_line
            and "min" in lower_line
        ):
            in_babelstream_table = True
            continue
        if in_babelstream_table:
            row_match = re.search(
                r"^(?:Copy|Mul|Add|Triad|Dot|Nstream)\s+(?P<score>[0-9.+eE-]+)\s+[0-9.+eE-]+\s+[0-9.+eE-]+\s+[0-9.+eE-]+$",
                line,
            )
            if row_match:
                score, _ = _extract_score_and_unit(row_match)
                if score is not None:
                    return score, "MB/s"
            if lower_line.startswith("running kernels"):
                in_babelstream_table = False

        # d3q19-style benchmark prints a header and then raw MLUPS values.
        if "performance:" in lower_line and "mlups" in lower_line:
            in_mlups_block = True
            continue
        if in_mlups_block:
            mlups_match = re.search(r"^(?P<score>[0-9]+(?:\.[0-9]+)?)$", line)
            if mlups_match:
                score, _ = _extract_score_and_unit(mlups_match)
                if score is not None:
                    return score, "MLUPS"
            if lower_line.startswith("completed"):
                in_mlups_block = False

        for pattern in COMMON_THROUGHPUT_PATTERNS:
            try:
                match = re.search(pattern, line)
            except re.error:
                log.debug("Skipping invalid throughput regex: %s", pattern)
                continue
            if not match:
                continue
            score, unit = _extract_score_and_unit(match)
            if score is not None:
                if not unit and lower_line.startswith("total clocks"):
                    unit = "clocks"
                return score, unit

    return None, None


def _has_device_side_assertion(output: str) -> bool:
    return bool(re.search(r"(?i)device[-\s]?side assertion", output))


def _detect_output_failure_reason(output: str) -> Optional[str]:
    for pattern, reason in OUTPUT_FAILURE_PATTERNS:
        try:
            if re.search(pattern, output):
                return reason
        except re.error:
            continue
    return None


def _setup_rocm_env(therock_bin_dir: Optional[str] = None) -> Dict[str, str]:
    """Build an environment dict that points at the TheRock toolchain."""
    therock_bin_dir = therock_bin_dir or os.getenv("THEROCK_BIN_DIR")
    if not therock_bin_dir:
        raise RuntimeError("THEROCK_BIN_DIR environment variable is required")

    therock_bin = Path(therock_bin_dir).resolve()
    rocm_path = therock_bin.parent

    env = os.environ.copy()
    env["ROCM_PATH"] = str(rocm_path)
    env["PATH"] = f"{therock_bin}{os.pathsep}{env.get('PATH', '')}".rstrip(os.pathsep)

    if sys.platform.startswith("linux"):
        rocm_lib = str(rocm_path / "lib")
        ld_library_path = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = (
            f"{rocm_lib}{os.pathsep}{ld_library_path}" if ld_library_path else rocm_lib
        )

    return env


def _has_prebuilt_artifacts(bench_dir: Path) -> bool:
    """Return True when benchmark directory has prebuilt object artifacts."""
    if not (bench_dir / "Makefile").is_file():
        return False

    # Check for compiled object files or the final executable.
    return any(bench_dir.glob("*.o")) or (bench_dir / "main").is_file()


def _select_bench_dirs(
    src_dir: Path,
    hecbench_benchmarks: str = "",
) -> List[Path]:
    """Return benchmark directories matching the filter (or all *-hip dirs)."""
    if not hecbench_benchmarks:
        hecbench_benchmarks = os.getenv("HECBENCH_BENCHMARKS", "")

    if hecbench_benchmarks:
        names = [
            name.strip() for name in hecbench_benchmarks.split(",") if name.strip()
        ]
        selected: List[Path] = []
        for name in names:
            bench_dir = src_dir / name
            if bench_dir.is_dir():
                selected.append(bench_dir)
            else:
                log.warning("Benchmark directory not found: %s", bench_dir)
        if not selected:
            raise RuntimeError(f"None of the specified benchmarks found: {names}")
        return selected

    benches = sorted(
        bench
        for bench in src_dir.iterdir()
        if bench.is_dir() and bench.name.endswith("-hip")
    )
    if not benches:
        raise RuntimeError(f"No *-hip benchmark directories found in {src_dir}")
    return benches


def _resolve_hecbench_build_dir(therock_bin_dir: Optional[str]) -> Path:
    """Resolve prebuilt HeCBench root from known TheRock output layouts."""
    candidates: List[Path] = [
        THEROCK_DIR
        / "build"
        / "third-party"
        / "hecbench-spirv"
        / "hecbench-spirv"
        / "dist"
        / "libexec"
        / "hecbench_spirv",
        THEROCK_DIR
        / "build"
        / "third-party"
        / "hecbench-spirv"
        / "hecbench-spirv"
        / "stage"
        / "libexec"
        / "hecbench_spirv",
        THEROCK_DIR / "build" / "dist" / "rocm" / "libexec" / "hecbench_spirv",
    ]

    if therock_bin_dir:
        candidates.append(
            Path(therock_bin_dir).resolve().parent / "libexec" / "hecbench_spirv"
        )

    seen: set[Path] = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        src_dir = candidate / "src"
        if src_dir.is_dir():
            return candidate

    checked = "\n  - " + "\n  - ".join(str(path / "src") for path in unique_candidates)
    raise RuntimeError(
        "Prebuilt HeCBench benchmarks not found. Checked:" + checked + "\n"
        "Build HeCBench as part of TheRock build/install step before running this test."
    )


class HeCBenchSPIRVBenchmark(BenchmarkBase):
    """HeCBench SPIR-V benchmark test."""

    def __init__(self):
        super().__init__(
            benchmark_name="hecbench_spirv", display_name="HeCBench SPIR-V"
        )
        self._results: List[Dict[str, Any]] = []
        self.hecbench_benchmarks = os.getenv("HECBENCH_BENCHMARKS", "")
        self.run_timeout = _load_run_timeout()
        self.expected_failures = _load_expected_failures()
        self.hecbench_build_dir = _resolve_hecbench_build_dir(self.therock_bin_dir)

    def run_benchmarks(self) -> None:
        """Run prebuilt HeCBench SPIR-V benchmarks and collect results."""
        env = _setup_rocm_env(self.therock_bin_dir)

        src_dir = self.hecbench_build_dir / "src"
        if not src_dir.is_dir():
            raise RuntimeError(
                "Prebuilt HeCBench benchmarks not found. "
                f"Expected directory: {src_dir}. "
                "Build HeCBench as part of TheRock build/install step before running this test."
            )

        log.info("Using prebuilt HeCBench SPIR-V benchmark tree at %s", src_dir)

        selected_benchmarks = _select_bench_dirs(src_dir, self.hecbench_benchmarks)
        log.info(
            "Selected %d benchmark(s): %s",
            len(selected_benchmarks),
            ", ".join(bench.name for bench in selected_benchmarks),
        )

        for index, bench_dir in enumerate(selected_benchmarks, start=1):
            bench_name = bench_dir.name
            log.info(
                "Running benchmark %d/%d: %s",
                index,
                len(selected_benchmarks),
                bench_name,
            )

            if not _has_prebuilt_artifacts(bench_dir):
                log.warning(
                    "Skipping %s: prebuilt artifacts not found in %s",
                    bench_name,
                    bench_dir,
                )
                self._results.append(
                    {
                        "benchmark": bench_name,
                        "status": "skipped (no build artifacts)",
                        "output": "",
                    }
                )
                continue

            run_proc = self.execute_command(
                ["make", "run"],
                env=env,
                cwd=str(bench_dir),
                timeout=self.run_timeout,
                capture_output=True,
            )

            combined_output = f"{run_proc.stdout or ''}\n{run_proc.stderr or ''}"

            if run_proc.returncode == 124:
                log.warning(
                    "Benchmark %s exceeded timeout (%ds)", bench_name, self.run_timeout
                )
                self._results.append(
                    {
                        "benchmark": bench_name,
                        "status": "run timed out",
                        "output": combined_output,
                    }
                )
                continue

            if _has_device_side_assertion(combined_output):
                log.error(
                    "Benchmark %s failed due to device-side assertion", bench_name
                )
                self._results.append(
                    {
                        "benchmark": bench_name,
                        "status": "device-side assertion",
                        "output": combined_output,
                    }
                )
                continue

            if run_proc.returncode != 0:
                output_failure_reason = _detect_output_failure_reason(combined_output)
                if output_failure_reason:
                    log.error(
                        "Benchmark %s output indicates failure: %s",
                        bench_name,
                        output_failure_reason,
                    )
                    self._results.append(
                        {
                            "benchmark": bench_name,
                            "status": output_failure_reason,
                            "output": combined_output,
                        }
                    )
                    continue

                score, unit = _extract_metric_from_output(combined_output)
                if score is not None and score > 0.0 and bool(unit):
                    log.warning(
                        "Benchmark %s exited with code %d but reported a valid metric; marking as passed",
                        bench_name,
                        run_proc.returncode,
                    )
                    self._results.append(
                        {
                            "benchmark": bench_name,
                            "status": "passed",
                            "output": combined_output,
                        }
                    )
                    continue
                if score is not None:
                    log.warning(
                        "Benchmark %s exited with code %d and reported an incomplete metric (score=%s, unit=%s); keeping FAIL",
                        bench_name,
                        run_proc.returncode,
                        score,
                        unit,
                    )

                log.error(
                    "Run failed for %s (exit code %d)", bench_name, run_proc.returncode
                )
                self._results.append(
                    {
                        "benchmark": bench_name,
                        "status": "run failed",
                        "output": combined_output,
                    }
                )
                continue

            output_failure_reason = _detect_output_failure_reason(combined_output)
            if output_failure_reason:
                log.error(
                    "Benchmark %s output indicates failure: %s",
                    bench_name,
                    output_failure_reason,
                )
                self._results.append(
                    {
                        "benchmark": bench_name,
                        "status": output_failure_reason,
                        "output": combined_output,
                    }
                )
                continue

            log.info("Completed %s successfully", bench_name)
            self._results.append(
                {"benchmark": bench_name, "status": "passed", "output": combined_output}
            )

        log.info("HeCBench SPIR-V benchmarks execution complete")

    def parse_results(self) -> Tuple[List[Dict[str, Any]], PrettyTable]:
        """Parse benchmark results using common time patterns."""
        log.info("Parsing Results")

        num_gpus = (
            self.client.system_context.gpu_count
            if self.client.system_context and self.client.system_context.gpu_count > 0
            else 1
        )

        table = PrettyTable(
            ["TestName", "SubTests", "nGPU", "Result", "Scores", "Units", "Flag"]
        )
        test_results: List[Dict[str, Any]] = []

        for entry in self._results:
            bench_name = entry["benchmark"]
            raw_status = entry["status"]

            # Skip benchmarks that were not built — don't show them in the table.
            if raw_status.startswith("skipped"):
                log.info("Omitting skipped benchmark from results: %s", bench_name)
                continue

            status = "PASS" if raw_status == "passed" else "FAIL"

            if status == "FAIL" and bench_name.lower() in self.expected_failures:
                log.warning(
                    "Benchmark %s failed but is listed in HECBENCH_EXPECTED_FAILURES; treating as PASS",
                    bench_name,
                )
                status = "PASS"

            output = entry.get("output", "")
            score, unit = _extract_metric_from_output(output)
            if score is None:
                score = 0.0
            if unit is None:
                unit = ""

            if status == "FAIL":
                flag = "-"
            else:
                flag = "H" if unit and not _is_time_unit(unit) else "L"

            table.add_row(
                [self.benchmark_name, bench_name, num_gpus, status, score, unit, flag]
            )

            test_results.append(
                self.create_test_result(
                    self.benchmark_name,
                    bench_name,
                    status,
                    score,
                    unit,
                    flag,
                    ngpu=num_gpus,
                    detail=raw_status,
                )
            )

        return test_results, table


if __name__ == "__main__":
    # Exit with benchmark status code without throwing a traceback on expected FAIL runs.
    sys.exit(HeCBenchSPIRVBenchmark().run())
