#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Test runner for TensileLite and rocisa Python tests using pre-built artifacts.

Runs against installed artifacts from the hipBLASLt test component:
  share/hipblaslt/tensilelite/Tensile/     — Tensile Python package
  share/hipblaslt/tensilelite/rocisa/       — rocisa Python package + _rocisa.abi3.so
  share/hipblaslt/tensilelite/rocisa_tests/ — rocisa pytest modules

Test order (fail fast):
- rocisa (build dependency of TensileLite)
- TensileLite unit tests
- TensileLite common GEMM tests (gfx1250 in AMDGPU_FAMILIES)

CI: All GPU archs (unit tests), GPU emulation (gfx1250 common tests)

Usage: python test_tensilelite.py
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# GPU families where unit tests are skipped. These families run under emulation
# where hip-python is unavailable and no arch-specific unit tests exist.
# TODO: move this skip logic into the pytest conftest so the wrapper stays thin.
UNIT_TEST_SKIP_FAMILIES = {"gfx1250"}

FFM_QUICK_EXCLUDE = [
    "mxf8_gfx1250.yaml",
    "mxf4_gfx1250.yaml",
    "sk_sgemm_quick.yaml",
]

# TENSILE_NUM_PYTEST_WORKERS: number of pytest-xdist processes running tests in parallel.
NUM_PYTEST_WORKERS = os.getenv("TENSILE_NUM_PYTEST_WORKERS", "16")

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR", str(THEROCK_DIR / "build" / "bin"))

rocm_path = Path(THEROCK_BIN_DIR).resolve().parent
tensilelite_root = rocm_path / "share" / "hipblaslt" / "tensilelite"

if not tensilelite_root.is_dir():
    raise FileNotFoundError(
        f"TensileLite test artifacts not found at {tensilelite_root}. "
        "Ensure the build used -DHIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS=ON."
    )

env = os.environ.copy()
existing_pythonpath = env.get("PYTHONPATH")
env["PYTHONPATH"] = (
    f"{tensilelite_root}{os.pathsep}{existing_pythonpath}"
    if existing_pythonpath
    else str(tensilelite_root)
)
env["ROCM_PATH"] = str(rocm_path)

# _rocisa links libamdhip64.so, tensilelite-client links libomp.so.
lib_path = rocm_path / "lib"
llvm_lib_path = rocm_path / "lib" / "llvm" / "lib"
existing_ld_path = env.get("LD_LIBRARY_PATH", "")
env["LD_LIBRARY_PATH"] = os.pathsep.join(
    filter(None, [str(lib_path), str(llvm_lib_path), existing_ld_path])
)

# GPU unit tests use amdclang++ to assemble kernels.
existing_path = env.get("PATH", "")
env["PATH"] = os.pathsep.join(
    filter(
        None,
        [
            str(rocm_path / "bin"),
            str(rocm_path / "lib" / "llvm" / "bin"),
            existing_path,
        ],
    )
)

# Smoke test: verify install layout and stable ABI.
logging.info("=== Verifying artifact install layout ===")
rocisa_dir = tensilelite_root / "rocisa"
logging.info(
    f"rocisa directory contents: {[f.name for f in rocisa_dir.iterdir() if not f.name.startswith('__')]}"
)
abi3_files = list(rocisa_dir.glob("*.abi3.*"))
if abi3_files:
    logging.info(f"Stable ABI confirmed: {[f.name for f in abi3_files]}")
else:
    logging.warning("No .abi3 extension found — stable ABI may not be enabled")
subprocess.check_call(
    [
        sys.executable,
        "-c",
        "import Tensile, rocisa, rocisa.instruction; "
        "print('Tensile:', Tensile.ROOT_PATH); print('rocisa:', rocisa.__file__)",
    ],
    cwd=str(THEROCK_DIR),
    env=env,
)

# === Tarball diagnostic dump (for debugging ci.yml vs PSDB tarball divergence) ===
logging.info("=== Tarball diagnostic dump ===")
import hashlib

cxx_path = rocm_path / "bin" / "amdclang++"
if cxx_path.is_file():
    result = subprocess.run([str(cxx_path), "--version"], capture_output=True, text=True, env=env)
    logging.info(f"amdclang++ version: {result.stdout.splitlines()[0] if result.returncode == 0 else 'FAILED'}")

rocisa_so = list((tensilelite_root / "rocisa").glob("_rocisa*"))
for f in rocisa_so:
    md5 = hashlib.md5(f.read_bytes()).hexdigest()
    logging.info(f"{f.name}: md5={md5}, size={f.stat().st_size}")

sol_py = tensilelite_root / "Tensile" / "SolutionStructs" / "Solution.py"
if sol_py.is_file():
    md5 = hashlib.md5(sol_py.read_bytes()).hexdigest()
    logging.info(f"Solution.py: md5={md5}")

kw_py = tensilelite_root / "Tensile" / "KernelWriter.py"
if kw_py.is_file():
    md5 = hashlib.md5(kw_py.read_bytes()).hexdigest()
    logging.info(f"KernelWriter.py: md5={md5}")

for subdir in ["gemm/gfx12", "sparse/gfx1250", "streamk/gfx1250", "gradient/gfx1250"]:
    yaml_dir = tensilelite_root / "Tensile" / "Tests" / "common" / subdir
    if yaml_dir.is_dir():
        count = len(list(yaml_dir.glob("*.yaml")))
        logging.info(f"YAML files in {subdir}: {count}")

client = rocm_path / "libexec" / "hipblaslt" / "tensilelite" / "tensilelite-client"
if client.is_file():
    md5 = hashlib.md5(client.read_bytes()).hexdigest()
    logging.info(f"tensilelite-client: md5={md5}, size={client.stat().st_size}")
    ldd = subprocess.run(["ldd", str(client)], capture_output=True, text=True, env=env)
    for line in ldd.stdout.splitlines():
        if "not found" in line:
            logging.warning(f"  MISSING: {line.strip()}")
else:
    logging.info("tensilelite-client: NOT FOUND in tarball")

logging.info("=== End diagnostic dump ===")

# rocisa tests (includes GPU tests — runner has GPU access).
logging.info("=== Running rocisa tests ===")
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        str(tensilelite_root / "rocisa_tests"),
    ],
    cwd=str(THEROCK_DIR),
    env=env,
)

# TensileLite Python unit tests (includes GPU subtile tests).
# TODO(TheRock#3288): gfx950-dcgpu is excluded from PR CI (ci.yml) due to runner
# capacity — GPU subtile tests only exercise on nightly/scheduled builds.
amdgpu_family = os.getenv("AMDGPU_FAMILIES", "")
skip_unit = UNIT_TEST_SKIP_FAMILIES & set(amdgpu_family.split(","))
if not skip_unit:
    logging.info("=== Running TensileLite unit tests ===")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            str(tensilelite_root / "Tensile" / "Tests" / "unit"),
        ],
        cwd=str(THEROCK_DIR),
        env=env,
    )
else:
    logging.info("=== Skipping unit tests (emulation mode) ===")

# TensileLite common (GEMM) tests — gfx1250 only, requires GPU or emulator.
# Scope to Tensile/Tests/common (not Tensile/Tests) to avoid rocisa singleton
# poisoning: unit test modules call validateToolchain()/makeIsaInfoMap() at
# import time, caching all-false ISA caps that break subsequent common tests.
common_tests = tensilelite_root / "Tensile" / "Tests" / "common"
client_path = rocm_path / "libexec" / "hipblaslt" / "tensilelite" / "tensilelite-client"

if common_tests.is_dir() and "gfx1250" in amdgpu_family:
    test_profile = os.getenv("TEST_PROFILE", "default")
    logging.info(
        f"=== Running TensileLite common gfx1250 tests (TEST_PROFILE={test_profile}) ==="
    )
    cxx = rocm_path / "bin" / "amdclang++"
    common_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "--durations=0",
        "-n",
        NUM_PYTEST_WORKERS,
        str(common_tests),
        "-m",
        "gfx1250 or gfx12",
        "-k",
        "gfx1250",
    ]
    if test_profile != "nightly":
        exclude = " and not ".join(["gfx1250"] + FFM_QUICK_EXCLUDE)
        common_cmd[-1] = exclude
    if client_path.is_file():
        common_cmd += [f"--prebuilt-client={client_path}"]
        common_cmd += ["--global-parameters=LibraryFormat='msgpack'"]
    if cxx.is_file():
        common_cmd += [f"--tensile-options=--cxx-compiler,{cxx},--gpu-targets,gfx1250"]
    # HSA_MODEL_NUM_THREADS: number of threads inside the FFM emulator per process.
    env["HSA_MODEL_NUM_THREADS"] = os.getenv("HSA_MODEL_NUM_THREADS", "8")

    logging.info("=== Split-test debug: pytest invocation ===")
    logging.info(f"  common_cmd: {common_cmd}")
    logging.info(f"  cwd: {THEROCK_DIR}")
    logging.info(f"  PYTHONPATH: {env.get('PYTHONPATH', '(unset)')}")
    logging.info(f"  TMPDIR: {env.get('TMPDIR', '(unset)')}")
    logging.info(f"  client_path exists: {client_path.is_file()}")
    logging.info(f"  common_tests: {common_tests}")
    logging.info(f"  conftest.py exists: {(common_tests / 'conftest.py').is_file()}")
    logging.info(f"  test_config.py exists: {(common_tests / 'test_config.py').is_file()}")
    logging.info(f"  test_config_build.py exists: {(common_tests / 'test_config_build.py').is_file()}")
    logging.info(f"  test_config_run.py exists: {(common_tests / 'test_config_run.py').is_file()}")
    logging.info(f"  artifact_helpers.py exists: {(common_tests / 'artifact_helpers.py').is_file()}")
    logging.info("=== End split-test debug ===")

    subprocess.check_call(common_cmd, cwd=str(THEROCK_DIR), env=env)
