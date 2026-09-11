#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prove that the host-ASAN compiler, runtime, and failure path are active."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


CANARY_SOURCE = r"""
#include <cstdlib>
int main() {
  auto *p = static_cast<volatile int *>(std::malloc(sizeof(int)));
  std::free(const_cast<int *>(p));
  return *p;
}
"""


def main() -> int:
    bin_dir = Path(os.environ["THEROCK_BIN_DIR"]).resolve()
    compiler = bin_dir.parent / "llvm" / "bin" / "clang++"
    if not compiler.is_file():
        print(f"ERROR: host-ASAN compiler not found: {compiler}", file=sys.stderr)
        return 1

    runtime = os.environ.get("ASAN_RUNTIME_PATH", "")
    if not runtime or not Path(runtime).is_file():
        print(f"ERROR: ASAN_RUNTIME_PATH is not a file: {runtime!r}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="therock-host-asan-") as tmp:
        tmp_dir = Path(tmp)
        source = tmp_dir / "canary.cpp"
        executable = tmp_dir / "canary"
        source.write_text(CANARY_SOURCE, encoding="utf-8")
        subprocess.run(
            [str(compiler), "-fsanitize=address", "-g", str(source), "-o", str(executable)],
            check=True,
        )

        env = os.environ.copy()
        options = env.get("ASAN_OPTIONS", "")
        env["ASAN_OPTIONS"] = ":".join(
            value
            for value in (options, "halt_on_error=1", "abort_on_error=1")
            if value
        )
        result = subprocess.run(executable, env=env, capture_output=True, text=True)

    report = result.stderr + result.stdout
    if result.returncode == 0:
        print("ERROR: ASAN canary unexpectedly exited successfully", file=sys.stderr)
        return 1
    if "AddressSanitizer" not in report or "heap-use-after-free" not in report:
        print("ERROR: ASAN canary failed without the expected report", file=sys.stderr)
        print(report, file=sys.stderr)
        return 1

    print("Host-ASAN canary detected the expected heap-use-after-free")
    return 0


if __name__ == "__main__":
    sys.exit(main())
