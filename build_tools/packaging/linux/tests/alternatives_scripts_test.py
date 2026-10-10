#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runs the rendered update-alternatives maintainer scripts with stubbed commands."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deb_package import generate_debian_postscripts
from packaging_utils import PackageConfig, get_package_info
from rpm_package import generate_rpm_postscripts

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="maintainer scripts are Linux-only"
)

PREFIX = "/srv/relocated/core-7.1"
OS_ID = {"deb": "debian", "rpm": "rhel"}
RPM_SECTION = {"postinst": "%post", "prerm": "%preun"}


def render(package: str, target: str, script: str, tmp_path: Path) -> str:
    config = PackageConfig(
        artifacts_dir=tmp_path,
        dest_dir=tmp_path,
        pkg_type=target,
        rocm_version="7.1.0",
        version_suffix="1",
        install_prefix=PREFIX,
        gfx_arch="",
    )
    info = get_package_info(package)
    if target == "rpm":
        return generate_rpm_postscripts(info, config)[RPM_SECTION[script]]
    generate_debian_postscripts(info, tmp_path, config)
    return (tmp_path / script).read_text()


def run(script: str, target: str, arg: str, stubs: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-e", "-s", "--", arg],
        input=f"source() {{ ID={OS_ID[target]}; }}\n{stubs}{script}",
        text=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", "RPM_INSTALL_PREFIX0": PREFIX},
    )


@pytest.mark.parametrize("package", ["amdrocm-core", "amdrocm-developer-tools"])
@pytest.mark.parametrize("target,arg", [("deb", "configure"), ("rpm", "1")])
def test_postinst_creates_opt_rocm(package, target, arg, tmp_path):
    stubs = (
        'mkdir() { [ "$*" = "-p /opt/rocm" ] && opt_rocm=1; }\n'
        'update-alternatives() { [ -n "$opt_rocm" ] && printf "%s\\n" "$*"; }\n'
    )
    result = run(render(package, target, "postinst", tmp_path), target, arg, stubs)
    assert result.returncode == 0, result.stderr
    assert f"--install /opt/rocm/share rocm-share {PREFIX}/share " in result.stdout


@pytest.mark.parametrize(
    "package", ["amdrocm-core", "amdrocm-core-devel", "amdrocm-developer-tools"]
)
@pytest.mark.parametrize(
    "target,arg,removes",
    [
        ("deb", "remove", True),
        ("deb", "upgrade", True),
        ("deb", "purge", False),
        ("rpm", "0", True),
        ("rpm", "1", False),
        ("rpm", "2", False),
    ],
)
def test_prerm_keeps_alternatives_on_rpm_upgrade(
    package, target, arg, removes, tmp_path
):
    stubs = 'update-alternatives() { printf "%s\\n" "$*"; }\n'
    result = run(render(package, target, "prerm", tmp_path), target, arg, stubs)
    assert result.returncode == 0, result.stderr
    assert ("--remove " in result.stdout) == removes
