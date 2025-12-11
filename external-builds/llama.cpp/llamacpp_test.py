#!/usr/bin/env python3

"""
Automates the execution of llama.cpp integrated tests.

It supports both full and smoke test modes, filters out tests that require
unavailable resources or are known to fail, and logs results to the console
or an optional log file.

Test binaries are discovered dynamically from the build directory.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LLAMA_BIN_DIR = Path(
    os.getenv("LLAMACPP_BUILD_DIR", SCRIPT_DIR / "llama.cpp" / "build" / "bin")
)

TESTS_TO_IGNORE = [
    # TODO: these tests require additional files/models. Disable for now.
    "test-tokenizer-1-spm",
    "test-gbnf-validator",
    "test-json-schema-to-grammar",
    "test-quantize-stats",
    "test-tokenizer-0",
    "test-chat",
    "test-tokenizer-1-bpe",
    "test-thread-safety",
    # TODO: fails due to FPE for Flash Attention ops
    # on GFX1030 GPU (AMD RX 6700 XT).
    # "test-backend-ops",
]

SMOKE_TESTS = [
    "test-llama-grammar",
    "test-arg-parser",
    "test-log",
    "test-c",
    "test-alloc",
    "test-gguf",
]


def setup_logging(log_file=None):
    """Configure logging to console and optionally to a file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def main(argv: list[str]):
    p = argparse.ArgumentParser(description="Run llama.cpp tests.")
    p.add_argument(
        "--save-logging", metavar="FILE", help="Save logs to a specified file"
    )
    args = p.parse_args(argv)

    setup_logging(args.save_logging)

    test_type = os.getenv("TEST_TYPE", "full")
    pattern = "test-*.exe" if os.name != "posix" else "test-*"
    all_tests = [
        p for p in LLAMA_BIN_DIR.glob(pattern) if p.stem not in TESTS_TO_IGNORE
    ]

    if test_type == "smoke":
        all_tests = [t for t in all_tests if t.name in SMOKE_TESTS]
    if not all_tests:
        logging.warning("No matching tests found!")
        return

    logging.info(f"Running {len(all_tests)} tests ({test_type} mode)...")

    for test in all_tests:
        logging.info(f"++ Running {test.name}")
        try:
            subprocess.run([str(test)], cwd=LLAMA_BIN_DIR, check=True)
            logging.info(f"[OK] {test.name} ran successfully")
        except subprocess.CalledProcessError as e:
            logging.error(f"[FAIL] {test.name} exited with code {e.returncode}")
            raise

    logging.info("All selected tests finished successfully.")


if __name__ == "__main__":
    main(sys.argv[1:])
