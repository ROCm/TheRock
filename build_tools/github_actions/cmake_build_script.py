#!/usr/bin/env python
"""cmake_build_script.py

This script runs TheRock's cmake build in the CI system. In addition to the cmake command, we supply telemetry options.

Usage:
python build_tools/github_actions/cmake_build_script.py [-h] [--build-variant BUILD_VARIANT] [--telemetry | --no-telemetry]
"""
import argparse
import shlex
import subprocess
import threading
import time
import sys

from github_actions_utils import _log


def telemetry_loop():
    """Background telemetry worker."""
    while True:
        try:
            _log(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            # TODO(geomin12): Add Windows alternative for these commands
            subprocess.run(["free", "-h"], check=False)

            # top -b -n1 | head -20
            top_proc = subprocess.Popen(["top", "-b", "-n1"], stdout=subprocess.PIPE)
            head_proc = subprocess.Popen(["head", "-20"], stdin=top_proc.stdout)
            top_proc.stdout.close()
            head_proc.communicate()

            time.sleep(30)

        except Exception as e:
            _log(f"[telemetry] error in telemetry loop: {e}", file=sys.stderr)
            break


def run(args):
    build_variant = args.build_variant
    telemetry = args.telemetry
    telemetry_thread = None
    stop_event = threading.Event()

    # Start telemetry if the build variant is asan or telemetry flag is enabled
    if build_variant == "asan" or telemetry:
        _log("[telemetry] starting background monitoring...")

        def telemetry_thread_wrapper():
            while not stop_event.is_set():
                telemetry_loop()

        telemetry_thread = threading.Thread(
            target=telemetry_thread_wrapper, daemon=True
        )
        telemetry_thread.start()

    # Run cmake build
    try:
        cmake_args = [
            "cmake",
            "--build",
            "build",
            "--target",
            "therock-archives",
            "therock-dist",
            "--",
            "-k",
            "0",
        ]
        _log(f"Run {shlex.join(cmake_args)}")
        subprocess.check_call(cmake_args)
    finally:
        # Shutdown telemetry
        if telemetry_thread:
            _log("[telemetry] stopping...")
            stop_event.set()
            telemetry_thread.join(timeout=5)


def main(argv):
    parser = argparse.ArgumentParser(prog="build")
    parser.add_argument(
        "--build-variant",
        type=str,
        help="The build variant that cmake will build for (ex: release, asan)",
    )
    parser.add_argument(
        "--telemetry",
        type=bool,
        default=False,
        help="If enabled, the cmake build will include telemetry logs",
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
