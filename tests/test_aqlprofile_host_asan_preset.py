# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_host_asan_preset_instruments_aqlprofile():
    presets = json.loads((REPO_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    host_asan = next(
        preset
        for preset in presets["configurePresets"]
        if preset["name"] == "linux-host-asan-base"
    )

    assert host_asan["cacheVariables"]["THEROCK_SANITIZER"] == "HOST_ASAN"
    assert host_asan["cacheVariables"]["aqlprofile_SANITIZER"] == "HOST_ASAN"
