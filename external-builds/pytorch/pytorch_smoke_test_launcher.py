#!/usr/bin/env python

import pytest
import sys
import os
import subprocess
from pathlib import Path


def get_executable_pytorch_gpu_index_list() -> list[str]:
    """Get index number list for gpus that are available on computer
       and are also supported by the currently tested pytorch build.
       Returned gpu index numbers can be used for controlling the GPU
       visibility via HIP_VISIBLE_DEVICES environment variable.

    Important: If HIP_VISIBLE_DEVICES is already set before calling this script,
    this function will only see GPUs within that constraint. This allows the
    script to work within pre-configured limitations (e.g., in containers).

    Returns:
        List of AMDGPU family strings visible (e.g., [0, 1]).
        Exits on failure.
    """
    query_script = """
import sys
import torch

ret = []

try:
    # list of gpus that are installed to computer and are visible
    gpu_available_list = []
    gpu_count = torch.cuda.device_count()
    print(f"gpu_count: {gpu_count}", file=sys.stderr)
    for ii in range(gpu_count):
        device_id = f"cuda:{ii}"
        device = torch.cuda.device(device_id)
        if device:
            device_properties = torch.cuda.get_device_properties(device)
            if device_properties and hasattr(device_properties, 'gcnArchName'):
                # AMD GPUs have gcnArchName
                gpu_available_list.append(device_properties.gcnArchName)
    if len(gpu_available_list) == 0:
        print("No AMD GPUs with gcnArchName detected", file=sys.stderr)
        sys.exit(1)
    # list of gpus that installed pytorch supports
    pytorch_supported_gpu_list = torch.cuda.get_arch_list()
    if len(pytorch_supported_gpu_list) == 0:
        print("No AMD GPUs detected", file=sys.stderr)
        sys.exit(1)
    # get index number for gpus that are available and also supported by pytorch install
    for ii, gpu in enumerate(gpu_available_list):
        if gpu in pytorch_supported_gpu_list:
            ret.append(ii)
    #print("List of gpus to test:")
    for gpu in ret:
        print(f"{gpu}")
except Exception as ex:
    print(f"ERROR:{ex}", file=sys.stderr)
    sys.exit(1)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", query_script],
            capture_output=True,
            text=True,
            check=True,
        )
        visible_gpus = result.stdout.strip().split("\n")
        for ii, gpu in enumerate(visible_gpus):
            visible_gpus[ii] = visible_gpus[ii].strip()
        return visible_gpus
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to retrieve visible GPUs: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error retrieving visible GPUs: {e}")
        sys.exit(1)


def main():
    # check if smoke-tests directory exists
    sm_test_dir = Path("smoke-tests")
    if not sm_test_dir.is_dir():
        print("Could not find smoke-tests directory.", sys.stderr)
        sys.exit(1)
    # unset HIP_VISIBLE_DEVICES before we we launch
    # the subprocesses to query all gpus that can be tested
    if "HIP_VISIBLE_DEVICES" in os.environ:
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
    gpu_index_list = get_executable_pytorch_gpu_index_list()
    # run the test for GPU's one by one on own subprocess
    for ii, gpu_idx in enumerate(gpu_index_list):
        print("gpu_index: " + gpu_idx)
        pytest_cmd = [
            "python",
            "-m",
            "pytest",
            "--log-cli-level=INFO",
            "-v",
            "smoke-tests",
        ]
        # set only the tested gpu index visible
        os.environ["HIP_VISIBLE_DEVICES"] = gpu_idx
        # launch pytest in own subprocess so that HIP_VISIBLE_DEVICES settings take effect
        try:
            completed_process = subprocess.run(
                pytest_cmd,
                capture_output=True,
                text=True,
                check=False,  # Set to True to raise a CalledProcessError if the return code is non-zero
            )
            print(completed_process.stdout)
            assert completed_process.returncode == 0
        except subprocess.CalledProcessError as e:
            print("Script failed:", e.stderr)
        except FileNotFoundError:
            print(f"Error: {script_to_run} not found.")


if __name__ == "__main__":
    main()
