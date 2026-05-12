#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
hipDNN Python bindings wheel install and smoke test.

This test verifies that the hipdnn-frontend wheel built by TheRock can be
installed into a fresh venv and that the basic Python API surface is
functional (import, enum access, Graph/Tensor construction, serialization).

Environment variables:
    OUTPUT_ARTIFACTS_DIR: Path to the TheRock dist/rocm output directory
                         that contains share/hipdnn/wheels/*.whl
"""

import argparse
import glob
import logging
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(
    os.environ.get("THEROCK_DIR") or SCRIPT_DIR.parent.parent.parent
).resolve()

logging.basicConfig(level=logging.INFO)


def find_wheel(artifacts_path: Path) -> Path:
    """Locate the hipdnn-frontend wheel under the artifacts directory."""
    wheel_dir = artifacts_path / "share" / "hipdnn" / "wheels"
    wheels = sorted(wheel_dir.glob("hipdnn_frontend-*.whl"))
    if not wheels:
        raise FileNotFoundError(
            f"No hipdnn-frontend wheel found in {wheel_dir}. "
            "Ensure the build was configured with -DHIPDNN_BUILD_PYTHON_BINDINGS=ON."
        )
    logging.info(f"Found wheel: {wheels[-1]}")
    return wheels[-1]


def create_venv(venv_dir: Path) -> Path:
    """Create a virtual environment and return the python executable path."""
    logging.info(f"Creating virtual environment in {venv_dir}")
    venv.create(venv_dir, with_pip=True)

    if platform.system() == "Windows":
        python = venv_dir / "Scripts" / "python.exe"
    else:
        python = venv_dir / "bin" / "python"

    if not python.exists():
        raise RuntimeError(f"venv python not found at {python}")
    return python


def install_wheel(python: Path, wheel: Path, artifacts_path: Path) -> None:
    """Install the wheel and numpy into the venv."""
    env = os.environ.copy()

    if platform.system() == "Windows":
        lib_path = str(artifacts_path)
        env["PATH"] = f"{lib_path};{env.get('PATH', '')}"
    else:
        lib_path = str(artifacts_path / "lib")
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [str(python), "-m", "pip", "install", str(wheel), "numpy"]
    logging.info(f"++ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def run_smoke_tests(python: Path, artifacts_path: Path) -> None:
    """Run inline Python smoke tests inside the venv."""
    env = os.environ.copy()

    if platform.system() == "Windows":
        lib_path = str(artifacts_path)
        env["PATH"] = f"{lib_path};{env.get('PATH', '')}"
    else:
        lib_path = str(artifacts_path / "lib")
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"

    test_script = r'''
import sys

def test_import():
    """Verify the package can be imported."""
    import hipdnn_frontend as fe
    assert hasattr(fe, "__version__"), "Missing __version__"
    print(f"  OK import hipdnn_frontend (version {fe.__version__})")

def test_data_types():
    """Verify enum bindings are accessible."""
    import hipdnn_frontend as fe
    for name in ["FLOAT", "HALF", "BFLOAT16", "INT8", "DOUBLE"]:
        assert hasattr(fe.DataType, name), f"Missing DataType.{name}"
    assert fe.DataType.FLOAT != fe.DataType.HALF
    print("  OK DataType enum values accessible")

def test_error():
    """Verify Error struct bindings."""
    import hipdnn_frontend as fe
    err = fe.Error()
    assert err.is_good(), "Default Error should be good"
    assert not err.is_bad(), "Default Error should not be bad"
    assert err.get_code() == fe.ErrorCode.OK

    err2 = fe.Error(fe.ErrorCode.INVALID_VALUE, "test error")
    assert err2.is_bad()
    assert "test error" in err2.get_message()
    print("  OK Error struct works")

def test_pointwise_mode():
    """Verify PointwiseMode enum."""
    import hipdnn_frontend as fe
    for name in ["ADD", "MUL", "RELU_FWD", "SIGMOID_FWD", "TANH_FWD"]:
        assert hasattr(fe.PointwiseMode, name), f"Missing PointwiseMode.{name}"
    print("  OK PointwiseMode enum values accessible")

def test_convolution_mode():
    """Verify ConvolutionMode enum."""
    import hipdnn_frontend as fe
    assert hasattr(fe.ConvolutionMode, "CROSS_CORRELATION")
    assert hasattr(fe.ConvolutionMode, "CONVOLUTION")
    print("  OK ConvolutionMode enum values accessible")

def test_tensor_create():
    """Verify Tensor creation and attribute access."""
    import hipdnn_frontend as fe
    t = fe.Tensor.create([1, 3, 224, 224], fe.DataType.FLOAT)
    assert t.get_dim() == [1, 3, 224, 224]
    assert t.get_data_type() == fe.DataType.FLOAT
    assert not t.get_is_virtual()
    assert t.get_volume() == 1 * 3 * 224 * 224
    print("  OK Tensor.create works")

def test_tensor_attributes():
    """Verify Tensor setters and method chaining."""
    import hipdnn_frontend as fe
    t = fe.Tensor()
    result = t.set_dim([2, 64, 32, 32]).set_data_type(fe.DataType.HALF).set_name("input")
    assert result is t, "Setters should return self for chaining"
    assert t.get_name() == "input"
    assert t.get_dim() == [2, 64, 32, 32]
    assert t.get_data_type() == fe.DataType.HALF
    print("  OK Tensor attribute setters and chaining work")

def test_tensor_uid():
    """Verify Tensor UID management."""
    import hipdnn_frontend as fe
    t = fe.Tensor.create([1, 1], fe.DataType.FLOAT)
    assert not t.has_uid(), "New tensor should not have UID"
    t.set_uid(42)
    assert t.has_uid()
    assert t.get_uid() == 42
    t.clear_uid()
    assert not t.has_uid()
    print("  OK Tensor UID management works")

def test_tensor_virtual():
    """Verify virtual tensor flag."""
    import hipdnn_frontend as fe
    t = fe.Tensor.create([1, 1], fe.DataType.FLOAT)
    t.set_is_virtual(True)
    assert t.get_is_virtual()
    t.set_is_virtual(False)
    assert not t.get_is_virtual()
    print("  OK Tensor virtual flag works")

def test_graph_create():
    """Verify Graph construction and attribute setting."""
    import hipdnn_frontend as fe
    g = fe.Graph()
    g.set_name("test_graph")
    g.set_compute_data_type(fe.DataType.FLOAT)
    g.set_io_data_type(fe.DataType.FLOAT)

    assert g.get_name() == "test_graph"
    assert g.get_compute_data_type() == fe.DataType.FLOAT
    assert g.get_io_data_type() == fe.DataType.FLOAT
    print("  OK Graph creation and attributes work")

def test_graph_chaining():
    """Verify Graph method chaining."""
    import hipdnn_frontend as fe
    g = fe.Graph()
    result = (
        g.set_name("chained")
        .set_compute_data_type(fe.DataType.FLOAT)
        .set_io_data_type(fe.DataType.HALF)
        .set_intermediate_data_type(fe.DataType.FLOAT)
    )
    assert result is g
    assert g.get_name() == "chained"
    assert g.get_io_data_type() == fe.DataType.HALF
    assert g.get_intermediate_data_type() == fe.DataType.FLOAT
    print("  OK Graph method chaining works")

def test_graph_tensor():
    """Verify creating tensors through the graph."""
    import hipdnn_frontend as fe
    g = fe.Graph()
    g.set_compute_data_type(fe.DataType.FLOAT).set_io_data_type(fe.DataType.FLOAT)

    t = g.tensor([1, 3, 224, 224], "input_tensor")
    assert t is not None
    assert t.get_name() == "input_tensor"
    assert t.get_dim() == [1, 3, 224, 224]
    print("  OK Graph.tensor() works")

def test_graph_tensor_like():
    """Verify tensor_like static method."""
    import hipdnn_frontend as fe
    t = fe.Tensor.create([2, 64, 16, 16], fe.DataType.HALF)
    t.set_name("original")

    copy = fe.Graph.tensor_like(t, "copy_tensor")
    assert copy.get_dim() == [2, 64, 16, 16]
    assert copy.get_data_type() == fe.DataType.HALF
    assert copy.get_name() == "copy_tensor"
    print("  OK Graph.tensor_like() works")

def test_engine_id_to_name():
    """Verify engine_id_to_name function."""
    import hipdnn_frontend as fe
    result = fe.engine_id_to_name(999999)
    assert isinstance(result, str)
    print("  OK engine_id_to_name works")

def test_preferred_engine_id():
    """Verify preferred engine ID set/get/clear."""
    import hipdnn_frontend as fe
    g = fe.Graph()

    assert g.get_preferred_engine_id_ext() is None

    g.set_preferred_engine_id_ext(12345)
    assert g.get_preferred_engine_id_ext() == 12345

    g.set_preferred_engine_id_ext(None)
    assert g.get_preferred_engine_id_ext() is None
    print("  OK Preferred engine ID management works")

def test_plugin_loading_mode():
    """Verify PluginLoadingMode enum."""
    import hipdnn_frontend as fe
    assert hasattr(fe, "PluginLoadingMode")
    assert hasattr(fe.PluginLoadingMode, "ADDITIVE")
    assert hasattr(fe.PluginLoadingMode, "ABSOLUTE")
    print("  OK PluginLoadingMode enum accessible")

def test_heuristic_mode():
    """Verify HeuristicMode enum."""
    import hipdnn_frontend as fe
    assert hasattr(fe.HeuristicMode, "FALLBACK")
    print("  OK HeuristicMode enum accessible")

def main():
    print("=" * 60)
    print("hipDNN Python bindings smoke tests")
    print("=" * 60)

    tests = [
        test_import,
        test_data_types,
        test_error,
        test_pointwise_mode,
        test_convolution_mode,
        test_tensor_create,
        test_tensor_attributes,
        test_tensor_uid,
        test_tensor_virtual,
        test_graph_create,
        test_graph_chaining,
        test_graph_tensor,
        test_graph_tensor_like,
        test_engine_id_to_name,
        test_preferred_engine_id,
        test_plugin_loading_mode,
        test_heuristic_mode,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    cmd = [str(python), "-c", test_script]
    logging.info("Running smoke tests...")
    subprocess.run(cmd, check=True, env=env)


def run_tests(artifacts_path: Path, venv_dir: Path) -> None:
    """Find wheel, create venv, install, and run tests."""
    wheel = find_wheel(artifacts_path)
    python = create_venv(venv_dir)
    install_wheel(python, wheel, artifacts_path)
    run_smoke_tests(python, artifacts_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Install and test hipDNN Python bindings wheel"
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        help="Directory for the test virtual environment. "
        "If not specified, uses a temporary directory that is auto-deleted.",
    )
    args = parser.parse_args()

    if not OUTPUT_ARTIFACTS_DIR:
        raise RuntimeError("OUTPUT_ARTIFACTS_DIR environment variable not set")

    artifacts_path = Path(OUTPUT_ARTIFACTS_DIR).resolve()
    logging.info(f"Using OUTPUT_ARTIFACTS_DIR: {artifacts_path}")

    if args.venv_dir:
        venv_dir = args.venv_dir.resolve()
        venv_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using persistent venv directory: {venv_dir}")
        run_tests(artifacts_path, venv_dir)
        logging.info(f"Venv retained in: {venv_dir}")
    else:
        logging.info("Using temporary venv directory (auto-cleanup)")
        with tempfile.TemporaryDirectory() as temp_dir:
            run_tests(artifacts_path, Path(temp_dir) / "venv")

    logging.info("All hipDNN Python binding tests passed!")
