# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from importlib import metadata
import sys

import pytest

import rocm_sdk_utils


def stage_sdk(tmp_path, monkeypatch, entries):
    """Use real distribution RECORD parsing without an ambient SDK."""
    dist_info = tmp_path / "rocm_sdk_core-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: rocm-sdk-core\nVersion: 1.0\n")
    if entries is not None:
        (dist_info / "RECORD").write_text("".join(f"{entry},,\n" for entry in entries))
    distribution = metadata.PathDistribution(dist_info)
    monkeypatch.setattr(
        rocm_sdk_utils.metadata, "distribution", lambda name: distribution
    )
    return distribution


@pytest.mark.parametrize("system", ["Linux", "Windows"])
@pytest.mark.parametrize("payload", ["_rocm_sdk_core", "_rocm_sdk_core_nonce"])
def test_record_owned_payload_without_runtime_import(
    tmp_path, monkeypatch, system, payload
):
    monkeypatch.setattr(rocm_sdk_utils.platform, "system", lambda: system)
    library = (
        "bin/amd_comgr0702.dll" if system == "Windows" else "lib/libamd_comgr.so.3"
    )
    entry = f"{payload}/{library}"
    path = tmp_path / entry
    path.parent.mkdir(parents=True)
    path.touch()
    (path.parent.parent / "__init__.py").write_text(
        "raise AssertionError('SDK payload must not be imported')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    stage_sdk(tmp_path, monkeypatch, [entry])
    before = set(sys.modules)
    assert rocm_sdk_utils.locate_comgr() == path.resolve()
    assert not any(
        name.startswith(("_rocm_sdk_core", "rocm_sdk"))
        for name in set(sys.modules) - before
    )


@pytest.mark.parametrize("override_kind", ["absolute", "relative", "symlink"])
def test_override_precedence_and_physical_path(tmp_path, monkeypatch, override_kind):
    compiler = tmp_path / "caller COMGR"
    compiler.touch()
    monkeypatch.chdir(tmp_path)
    override = str(compiler)
    if override_kind == "relative":
        override = "./caller COMGR"
    elif override_kind == "symlink":
        link = tmp_path / "compiler-link"
        try:
            link.symlink_to(compiler)
        except OSError as e:
            if sys.platform == "win32" and getattr(e, "winerror", None) == 1314:
                pytest.skip("Windows symlink privilege is unavailable")
            raise
        override = str(link)

    def forbidden_lookup(name):
        pytest.fail("explicit overrides must not consult SDK metadata")

    monkeypatch.setattr(rocm_sdk_utils.metadata, "distribution", forbidden_lookup)
    assert rocm_sdk_utils.locate_comgr(override=override) == compiler.resolve()


@pytest.mark.parametrize("override_kind", ["missing", "directory"])
def test_invalid_override_cannot_fall_back(tmp_path, monkeypatch, override_kind):
    sdk_entry = "_rocm_sdk_core/lib/libamd_comgr.so.3"
    sdk_comgr = tmp_path / sdk_entry
    sdk_comgr.parent.mkdir(parents=True)
    sdk_comgr.touch()
    stage_sdk(tmp_path, monkeypatch, [sdk_entry])
    override = tmp_path / "override"
    if override_kind == "directory":
        override.mkdir()
    with pytest.raises((OSError, RuntimeError)):
        rocm_sdk_utils.locate_comgr(override=str(override))


@pytest.mark.parametrize("override", [None, ""])
def test_missing_sdk_is_actionable(monkeypatch, override):
    def missing_distribution(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(rocm_sdk_utils.metadata, "distribution", missing_distribution)
    with pytest.raises(RuntimeError, match="rocm-sdk-core is not installed"):
        rocm_sdk_utils.locate_comgr(override=override)


@pytest.mark.parametrize(
    "entries", [None, [], ["_rocm_sdk_core/lib/libamd_comgr.so.3"]]
)
def test_missing_record_or_payload(tmp_path, monkeypatch, entries):
    stage_sdk(tmp_path, monkeypatch, entries)
    with pytest.raises(RuntimeError, match="no COMGR payload"):
        rocm_sdk_utils.locate_comgr()


def test_discovery_is_bounded_and_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(rocm_sdk_utils.platform, "system", lambda: "Linux")
    entries = [
        "other_package/lib/libamd_comgr.so",
        "_rocm_sdk_core/bin/libamd_comgr.so",
        "_rocm_sdk_core/lib/libamd_comgr.so.backup",
        "_rocm_sdk_core/lib/nested/libamd_comgr.so",
        "_rocm_sdk_core/lib/libamd_comgr.so.3",
        "_rocm_sdk_core/lib/libamd_comgr.so",
    ]
    for entry in entries + ["_rocm_sdk_core_nonce/lib/libamd_comgr.so"]:
        path = tmp_path / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    stage_sdk(tmp_path, monkeypatch, reversed(entries))
    assert (
        rocm_sdk_utils.locate_comgr()
        == (tmp_path / "_rocm_sdk_core/lib/libamd_comgr.so").resolve()
    )


def test_foreign_files_are_not_sdk_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(rocm_sdk_utils.platform, "system", lambda: "Linux")
    entries = ["other_package/lib/libamd_comgr.so", "../libamd_comgr.so"]
    stage_sdk(tmp_path, monkeypatch, entries)
    for entry in entries:
        path = tmp_path / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    with pytest.raises(RuntimeError, match="no COMGR payload"):
        rocm_sdk_utils.locate_comgr()
