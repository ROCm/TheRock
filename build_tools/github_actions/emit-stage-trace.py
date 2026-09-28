#!/usr/bin/env python3
"""
emit-stage-trace.py — emit a stage span + per-subproject child spans to Tempo

Creates a single trace per build stage:
  - Root span: full job duration (--start-time → now), tagged with ccache/status
  - Child spans: one per subproject parsed from build/.ninja_log, each covering
    its actual wall-clock window within the build step

Subproject timing uses relative ninja log timestamps anchored to --build-start.
If .ninja_log is missing (cancelled/failed job), only the root span is emitted.

Usage in a GHA workflow:

    # Capture timestamps:
    - run: echo "JOB_START=$(date +%s)" >> $GITHUB_ENV

    - name: Build stage
      run: |
        echo "BUILD_STEP_START=$(date +%s)" >> $GITHUB_ENV
        cmake --build "${BUILD_DIR}" --target stage-${STAGE_NAME} ...

    # At the end (if: always()):
    - env:
        OTLP_ENDPOINT: alloy-customer-dev.observability.svc.cluster.local:4317
      run: |
        CCACHE_HITS=$(ccache -s 2>/dev/null | grep -E "^\\s+Hits:" | head -1 | awk '{print $2}' || echo "0")
        CCACHE_MISSES=$(ccache -s 2>/dev/null | grep -E "^\\s+Misses:" | head -1 | awk '{print $2}' || echo "0")
        python3 -m venv /tmp/otel-venv
        /tmp/otel-venv/bin/pip install -q opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
        /tmp/otel-venv/bin/python3 build_tools/github_actions/emit-stage-trace.py \\
          --stage        "${STAGE_NAME}" \\
          --arch         "${AMDGPU_FAMILIES:-${DIST_AMDGPU_FAMILIES}}" \\
          --run-id       "${{ github.run_id }}" \\
          --repository   "${{ github.repository }}" \\
          --start-time   "${JOB_START}" \\
          --build-dir    "${BUILD_DIR}" \\
          --build-start  "${BUILD_STEP_START}" \\
          --status       "${{ job.status }}" \\
          --ccache-hits  "${CCACHE_HITS}" \\
          --ccache-misses "${CCACHE_MISSES}"

Environment variables:
    OTLP_ENDPOINT   gRPC endpoint for Alloy
                    (default: alloy-customer-dev.observability.svc.cluster.local:4317)
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from opentelemetry import context, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import NonRecordingSpan, SpanContext, SpanKind, StatusCode
from opentelemetry.trace import TraceFlags

ENDPOINT = os.environ.get(
    "OTLP_ENDPOINT",
    "alloy-customer-dev.observability.svc.cluster.local:4317",
)
CLUSTER = os.environ.get("CLUSTER", "therock-runners-dev-useast1")

# Mirrors NAME_MAPPING from analyze_build_times.py
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
    "miopenprovider": "miopenprovider",
    "hipblasltprovider": "hipblasltprovider",
}

ROCM_COMPONENT_DIRS = {
    "base",
    "compiler",
    "core",
    "comm-libs",
    "dctools",
    "profiler",
    "ml-libs",
    "media-libs",
    "cv-libs",
    "storage-libs",
}

PHASE_RULES = [
    (lambda p: p.endswith("/stamp/configure.stamp"), "configure"),
    (lambda p: p.endswith("/stamp/build.stamp"), "build"),
    (lambda p: p.endswith("/stamp/stage.stamp"), "install"),
    (lambda p: p.startswith("artifacts/") and p.endswith(".tar.xz"), "package"),
    (lambda p: "download" in p and "stamp" in p, "download"),
    (lambda p: "update" in p and "stamp" in p, "update"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emit a stage span + subproject child spans to Tempo via Alloy"
    )
    p.add_argument("--stage", required=True, help="Stage name (e.g. comm-libs)")
    p.add_argument("--arch", required=True, help="GPU arch family (e.g. gfx120X)")
    p.add_argument("--run-id", required=True, help="GitHub Actions run ID")
    p.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock"),
        help="GitHub repository (owner/name)",
    )
    p.add_argument(
        "--start-time",
        type=float,
        required=True,
        help="Job start time as Unix timestamp (seconds)",
    )
    p.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Job end time as Unix timestamp (default: now)",
    )
    p.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Build directory containing .ninja_log (for sub-stage spans)",
    )
    p.add_argument(
        "--build-start",
        type=float,
        default=None,
        help="Unix timestamp when cmake --build started (anchors ninja log offsets)",
    )
    p.add_argument(
        "--status",
        choices=["success", "failure", "cancelled"],
        default="success",
        help="Job outcome",
    )
    p.add_argument("--ccache-hits", type=int, default=None)
    p.add_argument("--ccache-misses", type=int, default=None)
    p.add_argument("--job-name", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ninja log parsing (adapted from analyze_build_times.py)
# ---------------------------------------------------------------------------


def parse_ninja_log(log_path: Path) -> list[dict]:
    records = []
    try:
        with open(log_path) as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    start_ms = int(parts[0])
                    end_ms = int(parts[1])
                    records.append(
                        {
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "output": parts[3],
                            "duration_ms": end_ms - start_ms,
                        }
                    )
    except (FileNotFoundError, ValueError):
        pass
    return records


def get_phase(output_path: str) -> str | None:
    for check, phase in PHASE_RULES:
        if check(output_path):
            return phase
    return None


def get_subproject(output_path: str, build_prefix: str) -> tuple[str | None, str | None]:
    """Return (subproject_name, phase) for a ninja output path, or (None, None)."""
    # Strip absolute build prefix if present
    if output_path.startswith(build_prefix):
        output_path = output_path[len(build_prefix):].lstrip("/")

    phase = get_phase(output_path)
    if not phase:
        return None, None

    parts = output_path.split("/")
    top = parts[0] if parts else ""

    # Artifact packaging — assign to subproject derived from filename
    if top == "artifacts" and len(parts) > 1:
        base = parts[1].replace(".tar.xz", "")
        # strip variant suffix: foo_lib_gfx94X → foo
        for suffix in ("_dbg", "_dev", "_doc", "_lib", "_run", "_test"):
            idx = base.find(suffix)
            if idx != -1:
                base = base[:idx]
                break
        name = NAME_MAPPING.get(base, base)
        return name, phase

    # Third-party dependencies
    if top == "third-party":
        if len(parts) > 3 and parts[1] == "sysdeps" and parts[2] in ("linux", "common"):
            name = parts[3]
        elif len(parts) > 1:
            name = parts[1]
        else:
            return None, None
        if name == "sysdeps":
            return None, None
        return NAME_MAPPING.get(name, name), phase

    # Standard ROCm component dirs (comm-libs, math-libs, etc.)
    if top in ROCM_COMPONENT_DIRS and len(parts) > 1:
        if top == "math-libs" and parts[1] == "BLAS" and len(parts) > 2:
            name = parts[2]
        elif top == "math-libs" and parts[1] == "support" and len(parts) > 2:
            name = parts[2]
        else:
            name = parts[1]
        return NAME_MAPPING.get(name, name), phase

    # rocm-libraries / rocm-systems external repos
    if top in ("rocm-libraries", "rocm-systems") and len(parts) > 2 and parts[1] == "projects":
        name = parts[2]
        return NAME_MAPPING.get(name, name), phase

    return None, None


def group_subprojects(
    records: list[dict], build_dir: Path
) -> dict[str, dict]:
    """Aggregate ninja records into per-subproject timing summaries."""
    build_prefix = str(build_dir.resolve())
    groups: dict[str, dict] = defaultdict(
        lambda: {
            "start_ms": float("inf"),
            "end_ms": 0,
            "total_cpu_ms": 0,
            "target_count": 0,
            "phases": defaultdict(int),
            "slowest_target": "",
            "slowest_target_ms": 0,
        }
    )

    for r in records:
        name, phase = get_subproject(r["output"], build_prefix)
        if not name or not phase:
            continue
        g = groups[name]
        g["start_ms"] = min(g["start_ms"], r["start_ms"])
        g["end_ms"] = max(g["end_ms"], r["end_ms"])
        g["total_cpu_ms"] += r["duration_ms"]
        g["target_count"] += 1
        g["phases"][phase] += r["duration_ms"]
        if r["duration_ms"] > g["slowest_target_ms"]:
            g["slowest_target_ms"] = r["duration_ms"]
            g["slowest_target"] = r["output"].split("/")[-1]

    # Finalize: fix infinity for empty groups
    for g in groups.values():
        if g["start_ms"] == float("inf"):
            g["start_ms"] = 0

    return dict(groups)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    end_ts = args.end_time or time.time()
    duration = end_ts - args.start_time
    start_ns = int(args.start_time * 1e9)
    end_ns = int(end_ts * 1e9)

    resource = Resource.create(
        {
            "service.name": "multi-arch-ci",
            "service.version": "1.0.0",
            "arch": args.arch,
            "repository": args.repository,
            "run_id": args.run_id,
            "cluster": CLUSTER,
        }
    )

    exporter = OTLPSpanExporter(endpoint=ENDPOINT, insecure=True)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("multi-arch-ci")

    stage_title = args.stage.replace("-", " ").title()
    job_name = args.job_name or (
        f"Linux::release / Build Multi-Arch Stages"
        f" / {args.stage} / Stage - {stage_title}"
    )

    # Root span attributes
    root_attrs: dict = {
        "gha.job_name": job_name,
        "gha.run_id": args.run_id,
        "gha.repository": args.repository,
        "gha.status": args.status,
        "stage": args.stage,
        "arch": args.arch,
        "duration_seconds": round(duration, 1),
    }
    if args.ccache_hits is not None:
        root_attrs["ccache.hits"] = args.ccache_hits
    if args.ccache_misses is not None:
        root_attrs["ccache.misses"] = args.ccache_misses
    if args.ccache_hits is not None and args.ccache_misses is not None:
        total = args.ccache_hits + args.ccache_misses
        root_attrs["ccache.hit_rate"] = (
            round(args.ccache_hits / total, 3) if total > 0 else 0.0
        )

    span_name = f"{args.stage} ({args.arch})"

    # Create root stage span
    root_span = tracer.start_span(
        span_name,
        kind=SpanKind.SERVER,
        start_time=start_ns,
        attributes=root_attrs,
    )
    if args.status == "failure":
        root_span.set_status(StatusCode.ERROR, "Stage failed")
    elif args.status == "cancelled":
        root_span.set_status(StatusCode.ERROR, "Stage cancelled")
    else:
        root_span.set_status(StatusCode.OK)

    # Parse ninja log and emit child spans if available
    subproject_count = 0
    ninja_log = None
    if args.build_dir:
        ninja_log = args.build_dir / ".ninja_log"

    if ninja_log and ninja_log.exists() and args.build_start is not None:
        records = parse_ninja_log(ninja_log)
        groups = group_subprojects(records, args.build_dir)

        # Set root span as parent context for all child spans
        root_ctx = trace.set_span_in_context(root_span)

        for subproject, g in sorted(groups.items(), key=lambda x: -x[1]["total_cpu_ms"]):
            wall_ms = g["end_ms"] - g["start_ms"]
            # Anchor relative ninja timestamps to absolute build start
            child_start_ns = int((args.build_start + g["start_ms"] / 1000) * 1e9)
            child_end_ns = int((args.build_start + g["end_ms"] / 1000) * 1e9)

            # Guard: don't exceed the root span's end time
            child_end_ns = min(child_end_ns, end_ns)
            child_start_ns = min(child_start_ns, child_end_ns)

            child_attrs: dict = {
                "subproject": subproject,
                "stage": args.stage,
                "arch": args.arch,
                "gha.run_id": args.run_id,
                "target_count": g["target_count"],
                "wall_duration_s": round(wall_ms / 1000, 1),
                "total_cpu_s": round(g["total_cpu_ms"] / 1000, 1),
                "slowest_target": g["slowest_target"],
                "slowest_target_s": round(g["slowest_target_ms"] / 1000, 1),
            }
            if wall_ms > 0:
                child_attrs["parallelism"] = round(g["total_cpu_ms"] / wall_ms, 1)
            for phase, ms in g["phases"].items():
                child_attrs[f"phase.{phase}_s"] = round(ms / 1000, 1)

            child_span = tracer.start_span(
                f"{subproject}",
                context=root_ctx,
                kind=SpanKind.INTERNAL,
                start_time=child_start_ns,
                attributes=child_attrs,
            )
            child_span.set_status(StatusCode.OK)
            child_span.end(end_time=child_end_ns)
            subproject_count += 1

        print(f"  subprojects: {subproject_count} child spans emitted")
    elif ninja_log and not ninja_log.exists():
        print(f"  subprojects: .ninja_log not found at {ninja_log}, skipping child spans")
    elif args.build_start is None:
        print("  subprojects: --build-start not provided, skipping child spans")

    # End root span last (after all children are exported)
    root_span.end(end_time=end_ns)

    provider.shutdown()

    print(f"Span emitted: {span_name}")
    print(f"  duration : {duration:.1f}s ({duration / 60:.1f}m)")
    print(f"  status   : {args.status}")
    print(f"  endpoint : {ENDPOINT}")
    if "ccache.hit_rate" in root_attrs:
        print(f"  ccache   : {root_attrs['ccache.hit_rate']:.1%} hit rate")

    return 0


if __name__ == "__main__":
    sys.exit(main())
