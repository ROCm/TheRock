#!/usr/bin/env python3
"""
emit-span.py — emit a single build stage span to the Alloy → Tempo pipeline

Designed to run at the END of a GHA build stage with real job metrics.
Sets explicit start/end timestamps so the span reflects actual job duration
without any sleeping.

Usage in a GHA workflow:

    # At the very start of the job, capture the start time:
    - run: echo "JOB_START=$(date +%s)" >> $GITHUB_ENV

    # At the end of the job (if: always() so it runs on failure too):
    - if: always()
      env:
        CCACHE_HITS: ...    # parsed from ccache -s output
        CCACHE_MISSES: ...
      run: |
        python3 build_tools/github_actions/emit-span.py \\
          --stage        "$STAGE_NAME" \\
          --arch         "$AMDGPU_FAMILY" \\
          --run-id       "$GITHUB_RUN_ID" \\
          --repository   "$GITHUB_REPOSITORY" \\
          --start-time   "$JOB_START" \\
          --status       "${{ job.status }}" \\
          --ccache-hits  "$CCACHE_HITS" \\
          --ccache-misses "$CCACHE_MISSES"

Environment variables:
    OTLP_ENDPOINT   gRPC endpoint for Alloy
                    (default: alloy-customer-dev.observability.svc.cluster.local:4317)
"""

import argparse
import os
import sys
import time

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import SpanKind, StatusCode

ENDPOINT = os.environ.get(
    "OTLP_ENDPOINT",
    "alloy-customer-dev.observability.svc.cluster.local:4317",
)
CLUSTER = os.environ.get("CLUSTER", "therock-runners-dev-useast1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emit a single build stage span to Tempo via Alloy"
    )
    p.add_argument(
        "--stage",
        required=True,
        help="Stage name (e.g. math-libs, compiler-runtime)",
    )
    p.add_argument(
        "--arch",
        required=True,
        help="GPU arch family (e.g. gfx120X, gfx94X)",
    )
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
        "--status",
        choices=["success", "failure", "cancelled"],
        default="success",
        help="Job outcome",
    )
    p.add_argument(
        "--ccache-hits",
        type=int,
        default=None,
        help="Number of ccache hits",
    )
    p.add_argument(
        "--ccache-misses",
        type=int,
        default=None,
        help="Number of ccache misses",
    )
    p.add_argument(
        "--targets-compiled",
        type=int,
        default=None,
        help="Total number of compilation targets",
    )
    p.add_argument(
        "--job-name",
        default=None,
        help="Full GHA job display name (auto-generated if omitted)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    end_ts = args.end_time or time.time()
    duration = end_ts - args.start_time

    # OpenTelemetry uses nanoseconds for timestamps
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
    # SimpleSpanProcessor exports synchronously — no background thread needed
    # for a single-shot script that shuts down immediately after.
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("multi-arch-ci")

    stage_title = args.stage.replace("-", " ").title()
    job_name = args.job_name or (
        f"Linux::release / Build Multi-Arch Stages"
        f" / {args.stage} / Stage - {stage_title}"
    )

    span_attrs: dict = {
        "gha.job_name": job_name,
        "gha.run_id": args.run_id,
        "gha.repository": args.repository,
        "gha.status": args.status,
        "stage": args.stage,
        "arch": args.arch,
        "duration_seconds": round(duration, 1),
    }

    if args.ccache_hits is not None:
        span_attrs["ccache.hits"] = args.ccache_hits
    if args.ccache_misses is not None:
        span_attrs["ccache.misses"] = args.ccache_misses
    if args.ccache_hits is not None and args.ccache_misses is not None:
        total = args.ccache_hits + args.ccache_misses
        span_attrs["ccache.hit_rate"] = (
            round(args.ccache_hits / total, 3) if total > 0 else 0.0
        )
    if args.targets_compiled is not None:
        span_attrs["targets_compiled"] = args.targets_compiled

    span_name = f"{args.stage} ({args.arch})"

    span = tracer.start_span(
        span_name,
        kind=SpanKind.SERVER,
        start_time=start_ns,
        attributes=span_attrs,
    )

    if args.status == "failure":
        span.set_status(StatusCode.ERROR, "Stage failed")
    elif args.status == "cancelled":
        span.set_status(StatusCode.ERROR, "Stage cancelled")
    else:
        span.set_status(StatusCode.OK)

    span.end(end_time=end_ns)

    # Flush synchronously before exit
    provider.shutdown()

    print(f"Span emitted: {span_name}")
    print(f"  duration : {duration:.1f}s ({duration / 60:.1f}m)")
    print(f"  status   : {args.status}")
    print(f"  endpoint : {ENDPOINT}")
    if "ccache.hit_rate" in span_attrs:
        print(f"  ccache   : {span_attrs['ccache.hit_rate']:.1%} hit rate")

    return 0


if __name__ == "__main__":
    sys.exit(main())
