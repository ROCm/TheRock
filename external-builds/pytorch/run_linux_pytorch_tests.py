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
INPUT_PYTORCH_VERSION  (optional)  - PyTorch version used for the tests
                                   - major.minor version as a string (e.g. "2.10")
                                   - if not set: auto-detection based on pytorch/version.txt
====================================
"""

import sys
import os
import tempfile

from skip_tests.create_skip_tests import *
import pytest


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
    # we need to force update the PYTHONPATH to the sys path
    # otherwise our current python process that will run
    # pytest will not find it and pytest will crash!
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)


if __name__ == "__main__":
    root_dir = os.getenv("INPUT_THEROCK_ROOT_DIR", "")
    # autodetect root dir via path of the script
    if root_dir == "":
        script_dir = os.path.dirname(sys.argv[0])
        # we are in <TheRock Root Dir>/external-builds/pytorch
        root_dir = script_dir.rsplit("/", 2)[0]

    pytorch_dir = f"{root_dir}/external-builds/pytorch/pytorch"

    amdgpu_family = os.getenv("INPUT_AMDGPU_FAMILY", "")
    pytorch_version = os.getenv("INPUT_PYTORCH_VERSION", "")

    # auto detect version by reading version string from pytorch/version.txt
    if pytorch_version == "":
        with open(f"{pytorch_dir}/version.txt", "r") as file:
            line = file.read()
            # hopefully pytorch version always stays like
            # major.minor.<some more version stuff without dots>
            pytorch_version = line.rsplit(".", 1)[0].strip()

    tests_to_skip = get_tests(amdgpu_family, pytorch_version)

    # Debugging: Get lists of tests always skipped and only run on those
    # tests_to_skip = skipped_tests.get_tests(amdgpu_family, pytorch_version, False)

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

    debug_pytorch_args = [
        "-p",
        "no:cacheprovider",  # disable caching: useful when running in
        # a container but wanting to use read-only TheRock
        # from the host system via setting INPUT_THEROCK_ROOT_DIR
        "--tb=no",
        "--maxfail=9999",
    ]

    # pytorch_args += debug_pytorch_args

    retcode = pytest.main(pytorch_args)
    print(f"Pytest finished with return code: {retcode}")
