#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed host-TSAN canary for CPU-only test runners."""

import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from host_tsan_instrumentation import require_no_gpu_nodes


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


TSAN_OPTIONS = "halt_on_error=1:exitcode=86:history_size=7:second_deadlock_stack=1"

clean_source = r"""
#include <atomic>
#include <thread>
int main() {
  std::atomic<int> value{0};
  std::thread a([&] { value.fetch_add(1); });
  std::thread b([&] { value.fetch_add(1); });
  a.join(); b.join();
  return value.load() == 2 ? 0 : 1;
}
"""

race_source = r"""
#include <thread>
volatile int value = 0;
int main() {
  std::thread a([] { for (int i = 0; i < 100000; ++i) ++value; });
  std::thread b([] { for (int i = 0; i < 100000; ++i) ++value; });
  a.join(); b.join();
  return 0;
}
"""


def build_environment(
    runtime: Path, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    # Do not instrument this Python launcher via inheritance. Each admitted
    # native executable must carry its own TSAN instrumentation and direct
    # shared-runtime dependency.
    env.pop("LD_PRELOAD", None)
    runtime_dir = str(runtime.parent)
    env["LD_LIBRARY_PATH"] = ":".join(
        part for part in (runtime_dir, env.get("LD_LIBRARY_PATH", "")) if part
    )
    options = TSAN_OPTIONS
    symbolizer = env.get("TSAN_SYMBOLIZER_PATH")
    if symbolizer:
        options += f":external_symbolizer_path={symbolizer}"
    env["TSAN_OPTIONS"] = options
    return env


def compile_and_run(
    root: Path,
    name: str,
    source: str,
    clang: Path,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess:
    source_path = root / f"{name}.cpp"
    executable = root / name
    source_path.write_text(source)
    subprocess.run(
        [
            str(clang),
            "-O1",
            "-g",
            "-fsanitize=thread",
            "-fno-omit-frame-pointer",
            "-shared-libsan",
            "-pthread",
            str(source_path),
            "-o",
            str(executable),
        ],
        check=True,
        env=env,
    )
    return subprocess.run(executable, capture_output=True, text=True, env=env)


def run_canaries(clang: Path, env: Mapping[str, str]) -> None:
    with tempfile.TemporaryDirectory(prefix="therock-host-tsan-") as temporary:
        root = Path(temporary)
        clean = compile_and_run(root, "clean", clean_source, clang, env)
        clean_output = clean.stdout + clean.stderr
        if clean.returncode != 0 or "ThreadSanitizer" in clean_output:
            fail(
                f"clean TSAN canary failed (exit {clean.returncode}):\n"
                f"{clean_output}"
            )

        raced = compile_and_run(root, "race", race_source, clang, env)
        race_output = raced.stdout + raced.stderr
        if raced.returncode != 86 or "ThreadSanitizer: data race" not in race_output:
            fail(
                "racy TSAN canary did not produce the required deterministic report "
                f"(exit {raced.returncode}):\n{race_output}"
            )


def main(environ: Mapping[str, str] | None = None) -> int:
    require_no_gpu_nodes()
    environ = os.environ if environ is None else environ
    rocm_root = Path(environ.get("THEROCK_BIN_DIR", "")).resolve().parent
    clang = rocm_root / "llvm" / "bin" / "clang++"
    runtime = Path(environ.get("TSAN_RUNTIME_PATH", ""))
    if not clang.is_file():
        fail(f"artifact clang++ not found: {clang}")
    if not runtime.is_file():
        fail(f"TSAN runtime not found: {runtime}")

    env = build_environment(runtime, environ)
    run_canaries(clang, env)
    print("Host-TSAN clean and data-race canaries passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
