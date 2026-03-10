"""Command-line entrypoint for miragectl."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from mirage.control_plane import (
    ControlPlaneError,
    SimulatorInstance,
    format_snapshot,
    load_status_snapshot,
    write_status_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="miragectl")
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=("boot", "status"),
        help="Control-plane action",
    )
    parser.add_argument(
        "--profile",
        help="Path to a cluster profile JSON file",
    )
    parser.add_argument(
        "--state-path",
        help="Lifecycle snapshot path emitted by a prior boot",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "boot":
            if not args.profile:
                raise ControlPlaneError("boot requires --profile")
            instance = SimulatorInstance.from_profile(args.profile)
            snapshot = instance.boot()
            if args.state_path:
                write_status_snapshot(snapshot, args.state_path)
        else:
            snapshot = _load_status_from_args(args)
    except ControlPlaneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_snapshot(snapshot, output_format=args.format))
    return 0


def _load_status_from_args(args: argparse.Namespace):
    if args.state_path:
        return load_status_snapshot(args.state_path)
    if args.profile:
        return SimulatorInstance.from_profile(args.profile).snapshot()
    raise ControlPlaneError("status requires --state-path or --profile")


if __name__ == "__main__":
    raise SystemExit(main())
