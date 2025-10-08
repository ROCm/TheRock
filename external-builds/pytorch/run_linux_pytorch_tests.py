#!/usr/bin/env python3
"""Runs pytest for PyTorch unit tests with additional test exclusion

Test exclusion can be due to
- amdgpu family
- pytorch version
- general failures not yet upstream

====================================
EXPECTED INPUT ENVIRONMENT VARIABLES
====================================
INPUT_THEROCK_ROOT_DIR (optional)  - To change the root directory of TheRock
                                   - Otherwise the location is extrapolated from
                                   - the location of this script
INPUT_AMDGPU_FAMILY    (optional)  - amdgpu_family as to run the tests on
                                   - names used as in "TheRock/cmake/therock_amdgpu_targets.cmake"
                                   - if not set, can lead to test failure as not enough tests might be omitted
                                   - if not set: tries to auto-detect which amdgpu arch is available.
                                                 selects the first one returned by amdgpu-arch
INPUT_PYTORCH_VERSION  (optional)  - PyTorch version used for the tests
                                   - major.minor version as a string (e.g. "2.10")
                                   - if not set: auto-detection based on pytorch/version.txt
====================================
"""

import sys
import os
import tempfile

from skip_tests.create_skip_tests import *
from importlib.metadata import version
import pytest
from pathlib import Path

import subprocess


def setup_env(pytorch_dir):
    os.environ["PYTORCH_PRINT_REPRO_ON_FAILURE"] = "0"
    os.environ["PYTORCH_TEST_WITH_ROCM"] = "1"
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = tempfile.mkdtemp()
    os.environ["PYTORCH_TESTING_DEVICE_ONLY_FOR"] = "cuda"
    old_pythonpath = os.getenv("PYTHONPATH", "")
    test_dir = f"{pytorch_dir}/test"
    if old_pythonpath != "":
        os.environ["PYTHONPATH"] = f"{test_dir}:{old_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = f"{test_dir}"

    # we need to force update the PYTHONPATH to be part of the sys path
    # otherwise our current python process that will run pytest will NOT
    # find it and pytest will crash!
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)


def cmd_arguments(argv: list[str]):
    p = argparse.ArgumentParser(
        description="""
Runs PyTorch pytest for AMD GPUs. Skips additional tests compared to upstream.
Additional tests to be skipped can be tuned by PyTorch version and amdgpu family.
"""
    )

    amdgpu_family = os.getenv("INPUT_AMDGPU_FAMILY")
    p.add_argument(
        "--amdgpu-family",
        type=str,
        default=amdgpu_family if not amdgpu_family == None else "",
        required=False,
        help="""Amdgpu family (e.g. "gfx942").
Select (potentially) additional tests to be skipped based on the amdgpu family""",
    )

    pytorch_version = os.getenv("INPUT_PYTORCH_VERSION")
    p.add_argument(
        "--pytorch-version",
        type=str,
        default=pytorch_version if not pytorch_version == None else "",
        required=False,
        help="""Pytorch version (e.g. "2.7" or "all).
Select (potentially) additional tests to be skipped based on the Pytorch version.
'All' is also possible. Then all skip tests for all pytorch versions are included.
If no PyTorch version is given, it is auto-determined by the PyTorch used to run pytest.""",
    )

    env_root_dir = os.getenv("INPUT_THEROCK_ROOT_DIR")
    p.add_argument(
        "--the-rock-root-dir",
        default=env_root_dir if not env_root_dir == None else "",
        required=False,
        help="""Overwrites the root directory of TheRock.
By default TheRock root dir is determined based on this script's location.""",
    )

    p.add_argument(
        "--debug",
        default=False,
        required=False,
        action=argparse.BooleanOptionalAction,
        help="""Inverts the selection. Only runs skipped tests.""",
    )

    p.add_argument(
        "-k",
        default="",
        required=False,
        help="""Overwrites the pytest -k option that decides which tests should be run or skipped""",
    )

    p.add_argument(
        "--no-cache",
        default=False,
        required=False,
        action=argparse.BooleanOptionalAction,
        help="""Disable pytest caching. Useful when only having read-only access to pytorch directory""",
    )

    args = p.parse_args(argv)
    return args


if __name__ == "__main__":
    args = cmd_arguments(sys.argv[1:])

    root_dir = args.the_rock_root_dir
    # autodetect root dir via path of the script
    if root_dir == "":
        script_dir = Path(__file__).resolve().parent
        # we are in <TheRock Root Dir>/external-builds/pytorch
        root_dir = script_dir.parent.parent

    amdgpu_family = args.amdgpu_family
    # try auto determine amdgpu_arch
    if amdgpu_family == "":
        proc = subprocess.run(["amdgpu-arch"], capture=True, text=True)
        if len(proc.stderr) == 0 and proc.returncoed == 0:  # must have been successful
            amdgpu_family = proc.stdout.split("/n")[0]

    pytorch_version = args.pytorch_version
    # auto detect version by reading version string from pytorch/version.txt
    if pytorch_version == "":
        pytorch_version = version("torch").rsplit("+", 1)[0].rsplit(".", 1)[0]

    tests_to_skip = get_tests(amdgpu_family, pytorch_version, not args.debug)

    # Debugging: Get lists of tests always skipped and only run on those
    # tests_to_skip = skipped_tests.get_tests(amdgpu_family, pytorch_version, False)

    if not args.k == "":
        tests_to_skip = args.k

    pytorch_dir = f"{root_dir}/external-builds/pytorch/pytorch"
    setup_env(pytorch_dir)

    pytorch_args = [
        f"{pytorch_dir}/test/test_nn.py",
        f"{pytorch_dir}/test/test_torch.py",
        f"{pytorch_dir}/test/test_cuda.py",
        f"{pytorch_dir}/test/test_unary_ufuncs.py",
        f"{pytorch_dir}/test/test_binary_ufuncs.py",
        f"{pytorch_dir}/test/test_autograd.py",
        "--continue-on-collection-errors",
        "--import-mode=importlib",
        f"-k={tests_to_skip}",
        "-v",
        "-n=0",  # TODO does this need rework?
        # -n numprocesses, --numprocesses=numprocesses
        #         Shortcut for '--dist=load --tx=NUM*popen'.
        #         With 'logical', attempt to detect logical CPU count (requires psutil, falls back to 'auto').
        #         With 'auto', attempt to detect physical CPU count. If physical CPU count cannot be determined, falls back to 1.
        #         Forced to 0 (disabled) when used with --pdb.
    ]

    if args.no_cache:
        pytorch_args += [
            "-p",
            "no:cacheprovider",  # disable caching: useful when running in
        ]

    # pytorch_args += debug_pytorch_args

    retcode = pytest.main(pytorch_args)
    print(f"Pytest finished with return code: {retcode}")
