# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from build_python_packages import mark_amdsmi_rocm_sdk_core


def _wrapper(core_dir: Path, text: str) -> Path:
    wrapper = core_dir / "share" / "amd_smi" / "amdsmi" / "amdsmi_wrapper.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(text)
    return wrapper


def test_flag_is_set(tmp_path):
    wrapper = _wrapper(tmp_path, "x = 1\n_AMDSMI_ROCM_SDK_CORE = False\n")
    mark_amdsmi_rocm_sdk_core(tmp_path)
    assert wrapper.read_text() == "x = 1\n_AMDSMI_ROCM_SDK_CORE = True\n"


def test_wrapper_without_flag_is_unchanged(tmp_path):
    wrapper = _wrapper(tmp_path, "x = 1\n")
    mark_amdsmi_rocm_sdk_core(tmp_path)
    assert wrapper.read_text() == "x = 1\n"


def test_missing_module_is_ignored(tmp_path):
    mark_amdsmi_rocm_sdk_core(tmp_path)


def test_duplicate_flag_fails(tmp_path):
    _wrapper(tmp_path, "_AMDSMI_ROCM_SDK_CORE = False\n_AMDSMI_ROCM_SDK_CORE = False\n")
    with pytest.raises(RuntimeError):
        mark_amdsmi_rocm_sdk_core(tmp_path)
