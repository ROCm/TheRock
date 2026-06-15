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

# Curated subset for ffm-quick (~20 min test time, 30 min total with setup).
# Covers all kernel features: gemm dtypes, TDM, mixed precision, sparse, streamk,
# gradient, activation. See ~/gfx1250_test_failures.md for full timing data.
FFM_QUICK_INCLUDE = [
    # GEMM data types (8 tests, ~2.9 min)
    "fp16_use_e_gfx1250.yaml",
    "b8b8s_gfx1250.yaml",
    "f8f8s_gfx1250.yaml",
    "f6b6ss_gfx1250.yaml",
    "i8ii_gfx1250.yaml",
    "b6f4ss_gfx1250.yaml",
    "gfx12/f4_gfx1250.yaml",
    "f32_gfx1250.yaml",
    # GEMM TDM (3 tests, ~1.0 min)
    "bf6_tdm_gfx1250.yaml",
    "f4f6ss_tdm_gfx1250.yaml",
    "mxf6_tdm_gfx1250.yaml",
    # GEMM features (2 tests, ~0.6 min)
    "largeLds_gfx1250.yaml",
    "1024_vgpr_gfx1250.yaml",
    # GEMM extra dtypes (3 tests, ~2.1 min)
    "f8f8s_sr_gfx1250.yaml",
    "f8b6ss_gfx1250.yaml",
    "xfp32_gfx1250.yaml",
    # GEMM mixed precision (4 tests, ~2.4 min)
    "f6b8ss_gfx1250.yaml",
    "f8b8ss_gfx1250.yaml",
    "gfx12/f8f4ss_gfx1250.yaml",
    "gfx12/f6_tdm_gfx1250.yaml",
    # Sparse (11 tests, ~3.3 min)
    "spmm_b8f8_sb.yaml",
    "spmm_f8hs_sb.yaml",
    "spmm_i8bs_sb.yaml",
    "spmm_tdm_f16_transposes.yaml",
    "spmm_b8.yaml",
    "spmm_f8bs.yaml",
    "spmm_bf16.yaml",
    "spmm_f16_sb.yaml",
    "spmm_b8hs_sb.yaml",
    "spmm_b8f8.yaml",
    "spmm_tdm_all.yaml",
    # StreamK (2 tests, ~2.6 min)
    "sk_f8gemm_quick.yaml",
    "gfx1250/sk_hgemm_quick.yaml",
    # Gradient (8 tests, ~2.0 min)
    "hhs_dgelu_gfx1250.yaml",
    "bbs_dgelu_gfx1250.yaml",
    "bbs_bgrada_gfx1250.yaml",
    "hhs_bgrada_gfx1250.yaml",
    "bbs_bgradb_gfx1250.yaml",
    "hhs_bgradb_gfx1250.yaml",
    "bbs_bgradd_gfx1250.yaml",
    "hhs_bgradd_gfx1250.yaml",
    # Activation (2 tests, ~0.8 min)
    "bf16_activation.yaml",
    "gfx1250/f16_activation.yaml",
    # Additional sparse coverage (~6.3 min)
    "spmm_f8b8_sb.yaml",
    "spmm_b8_sb.yaml",
    "gfx1250/spmm_i8_sb.yaml",
    "spmm_b8bs_sb.yaml",
    "spmm_f8_sb.yaml",
    "spmm_i8hs.yaml",
    # Round 2: fill 20-min budget (~3.5 min extra)
    "spmm_f8b8.yaml",
    "gfx12/f6_gfx1250.yaml",
    "fp8_gfx1250.yaml",
    "spmm_fp16_ml1.yaml",
    "f4b8ss_gfx1250.yaml",
    "spmm_f8bs_sb.yaml",
    # Feature gap coverage
    "f64_gfx1250.yaml",
    "nt_th_nv_gfx1250.yaml",
]

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
        str(common_tests),
        "-m",
        "gfx1250 or gfx12",
        "-k",
        "gfx1250",
    ]
    if test_profile != "nightly":
        include = " or ".join(FFM_QUICK_INCLUDE)
        common_cmd[-1] = include
    if client_path.is_file():
        common_cmd += [f"--prebuilt-client={client_path}"]
        common_cmd += ["--global-parameters=LibraryFormat='msgpack'"]
    if cxx.is_file():
        common_cmd += [f"--tensile-options=--cxx-compiler,{cxx},--gpu-targets,gfx1250"]
    subprocess.check_call(common_cmd, cwd=str(THEROCK_DIR), env=env)
