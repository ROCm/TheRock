#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
hipDNN installation consumption test.

This test verifies that hipDNN packages built by TheRock can be properly
consumed by external projects using CMake's find_package. It tests the
CMake packaging/installation correctness, not hipDNN functionality.
"""

import argparse
import logging
import os
import platform
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
TEST_PROJECT_DIR = SCRIPT_DIR / "hipdnn_install_tests"

logging.basicConfig(level=logging.INFO)


def run_tests(build_dir: Path):
    """Configure, build, and test all hipDNN packages."""
    # Locally, can set OUTPUT_ARTIFACTS_DIR=build/dist/rocm for testing
    artifacts_path = Path(OUTPUT_ARTIFACTS_DIR).resolve()
    is_windows = platform.system() == "Windows"

    # Compiler extension differs by platform
    compiler_ext = ".exe" if is_windows else ""

    # Set up environment variables for CMake/HIP
    environ_vars = os.environ.copy()
    environ_vars["HIP_PLATFORM"] = "amd"

    if is_windows:
        # Set library path for runtime (needed when running the test executables)
        rocm_lib = str(artifacts_path)

        # Windows uses PATH for DLL lookup
        path_sep = ";"
        if "PATH" in environ_vars:
            environ_vars["PATH"] = f"{rocm_lib}{path_sep}{environ_vars['PATH']}"
        else:
            environ_vars["PATH"] = rocm_lib
    else:
        rocm_lib = str(artifacts_path / "lib")

        # Linux uses LD_LIBRARY_PATH
        path_sep = ":"
        if "LD_LIBRARY_PATH" in environ_vars:
            environ_vars["LD_LIBRARY_PATH"] = (
                f"{rocm_lib}{path_sep}{environ_vars['LD_LIBRARY_PATH']}"
            )
        else:
            environ_vars["LD_LIBRARY_PATH"] = rocm_lib

    # We configure and build test projects externally (not during TheRock build)
    # to emulate how a consumer would build against the installed hipDNN artifacts.
    # This catches packaging issues that only manifest during external consumption.
    configure_cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-S",
        str(TEST_PROJECT_DIR),
        "-GNinja",
        f"-DCMAKE_PREFIX_PATH={artifacts_path}",
        f"-DCMAKE_CXX_COMPILER={artifacts_path}/lib/llvm/bin/clang++{compiler_ext}",
        f"-DCMAKE_C_COMPILER={artifacts_path}/lib/llvm/bin/clang{compiler_ext}",
        "--log-level=WARNING",
    ]

    # Windows needs a resource compiler specified
    if is_windows:
        configure_cmd.append("-DCMAKE_RC_COMPILER=rc.exe")
    logging.info(f"++ Configure: {shlex.join(configure_cmd)}")
    subprocess.run(configure_cmd, check=True, cwd=THEROCK_DIR, env=environ_vars)

    build_cmd = ["cmake", "--build", str(build_dir)]
    logging.info(f"++ Build: {shlex.join(build_cmd)}")
    subprocess.run(build_cmd, check=True, cwd=THEROCK_DIR, env=environ_vars)

    test_cmd = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "--parallel",
        "8",
    ]
    logging.info(f"++ Test: {shlex.join(test_cmd)}")
    subprocess.run(test_cmd, check=True, cwd=THEROCK_DIR, env=environ_vars)


def run_list_engines_test():
    """Test hipdnn_list_engines utility produces expected output.

    This verifies the installed utility can enumerate available hipDNN engines.
    Unlike the CMake-based tests above, this doesn't compile anything - it just
    runs the pre-built utility and validates its output.

    The test runs the utility twice:
    1. With explicit --plugin-dir argument
    2. Without --plugin-dir (using internal default path)

    If the miopen_plugin is present, both runs must contain the expected
    engine strings. If the plugin is not present, only the return code is verified.
    """
    artifacts_path = Path(OUTPUT_ARTIFACTS_DIR).resolve()
    is_windows = platform.system() == "Windows"

    exe_suffix = ".exe" if is_windows else ""
    list_engines = artifacts_path / "bin" / f"hipdnn_list_engines{exe_suffix}"

    if not list_engines.exists():
        raise FileNotFoundError(f"hipdnn_list_engines not found at: {list_engines}")

    # Plugin directory location differs by platform
    if is_windows:
        plugin_dir = artifacts_path / "bin" / "hipdnn_plugins" / "engines"
        miopen_plugin_name = "miopen_plugin.dll"
    else:
        plugin_dir = artifacts_path / "lib" / "hipdnn_plugins" / "engines"
        miopen_plugin_name = "libmiopen_plugin.so"

    # Check if miopen_plugin exists
    miopen_plugin_path = plugin_dir / miopen_plugin_name
    has_miopen_plugin = miopen_plugin_path.exists()

    if not has_miopen_plugin:
        logging.info(
            f"{miopen_plugin_name} not found in {plugin_dir}, "
            "skipping engine output validation (will only verify return code)"
        )

    # Expected engine patterns that must be present in the output (only checked if plugin exists)
    # Each pattern matches a line containing the engine name and its ID
    expected_engine_patterns = [
        r"MIOPEN_ENGINE.*0x15B46865C717A122",
        r"MIOPEN_ENGINE_DETERMINISTIC.*0xA258541A6DAA1DE3",
    ]

    def run_and_validate(cmd: list, description: str) -> None:
        """Run hipdnn_list_engines and validate output."""
        logging.info(f"++ Running ({description}): {shlex.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(
                f"hipdnn_list_engines failed with return code {result.returncode}"
            )
            logging.error(f"stdout: {result.stdout}")
            logging.error(f"stderr: {result.stderr}")
            raise RuntimeError(
                f"hipdnn_list_engines ({description}) exited with code {result.returncode}"
            )

        logging.info(f"hipdnn_list_engines ({description}) output:\n{result.stdout}")

        # Only validate output content if miopen_plugin is present
        if not has_miopen_plugin:
            logging.info(
                f"Skipping output validation ({description}) - {miopen_plugin_name} not present"
            )
            return

        if "Loaded engines:" not in result.stdout:
            logging.error(
                f"Unexpected output from hipdnn_list_engines:\n{result.stdout}"
            )
            raise RuntimeError(
                f"hipdnn_list_engines ({description}) output missing 'Loaded engines:'"
            )

        for pattern in expected_engine_patterns:
            if not re.search(pattern, result.stdout):
                logging.error(
                    f"Unexpected output from hipdnn_list_engines:\n{result.stdout}"
                )
                raise RuntimeError(
                    f"hipdnn_list_engines ({description}) output missing expected engine pattern: {pattern}"
                )

    # Run 1: Test with explicit --plugin-dir argument
    cmd_explicit = [str(list_engines), "--plugin-dir", str(plugin_dir)]
    run_and_validate(cmd_explicit, "explicit plugin-dir")

    # Run 2: Test without --plugin-dir (use internal default path)
    cmd_default = [str(list_engines)]
    run_and_validate(cmd_default, "default plugin-dir")

    if has_miopen_plugin:
        logging.info(
            "hipdnn_list_engines test passed (both explicit and default plugin-dir)"
        )
    else:
        logging.info(
            f"hipdnn_list_engines test passed (return code only - {miopen_plugin_name} not present)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test hipDNN package installation and consumption"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Build directory path (will be created if doesn't exist). "
        "If not specified, uses temporary directory that is auto-deleted.",
    )
    args = parser.parse_args()

    if not OUTPUT_ARTIFACTS_DIR:
        raise RuntimeError("OUTPUT_ARTIFACTS_DIR environment variable not set")

    logging.info(f"Using OUTPUT_ARTIFACTS_DIR: {OUTPUT_ARTIFACTS_DIR}")

    if args.build_dir:
        build_dir = args.build_dir.resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using persistent build directory: {build_dir}")
        run_tests(build_dir)
        logging.info(f"Build artifacts retained in: {build_dir}")
    else:
        logging.info("Using temporary build directory (auto-cleanup)")
        with tempfile.TemporaryDirectory() as temp_dir:
            run_tests(Path(temp_dir))

    run_list_engines_test()

    logging.info("All hipDNN install tests passed!")
