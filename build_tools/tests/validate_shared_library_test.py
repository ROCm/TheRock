# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hermetic tests for validate_shared_library ELF gates.

No real libamd_comgr.so, GPU, or ROCm tree is required.

TDD notes (10.0.0 wheel characterization, spec §8 / §12.4):
- --forbid-needed against 10.0.0 libamd_comgr.so.3 is RED (DT_NEEDED
  libLLVM.so.23.0git and libclang-cpp.so.23.0git).
- --forbid-dynsym '^LLVM[A-Z]|4llvm' against that same so is GREEN: the
  dylib client imports LLVM C API (undefined), it does not define/export it.
"""

import argparse
import subprocess
from pathlib import Path

import pytest

import validate_shared_library as vsl

# Keep in sync with compiler/CMakeLists.txt FORBID_NEEDED.
_COMPILER_DYLIB_PATTERN = r"libLLVM|libclang-cpp|libclang[.]so|libLTO|libRemarks"

# Keep in sync with compiler/CMakeLists.txt FORBID_DYNSYM.
# ^LLVM[A-Z] = the C API, anchored and requiring an uppercase letter so it
# does not match `LLVM_23.0` version-node entries. 4llvm = Itanium mangling
# of the llvm:: namespace.
_LLVM_DYNSYM_PATTERN = r"^LLVM[A-Z]|4llvm"

# Copied from real `readelf -d` shape. Extra fields (SONAME, STRTAB) must
# be ignored; only (NEEDED) lines yield sonames.
_READELF_DYLIB_CLIENT = """\
Dynamic section at offset 0x123000 contains 32 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libLLVM.so.23.0git]
 0x0000000000000001 (NEEDED)             Shared library: [libclang-cpp.so.23.0git]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x000000000000000e (SONAME)             Library soname: [libamd_comgr.so.3]
"""

_READELF_LIBC_ONLY = """\
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x000000000000000e (SONAME)             Library soname: [libamd_comgr.so.3]
"""

# A version-script leak: C API, versioned C API, mangled llvm:: symbol.
# The `A LLVM_23.0` line is a version *node*, not a symbol, and must not be
# what makes this fixture red (see _NM_DEFINED_DYLIB_CLIENT).
_NM_DEFINED_WITH_LLVM = """\
0000000000012345 T LLVMInitializeAMDGPUTarget
0000000000012346 T LLVMInitializeAMDGPUTarget@@LLVM_23.0
0000000000012347 T _ZN4llvm11raw_ostream5writeEPKcm
0000000000000000 A LLVM_23.0
0000000000001111 T amd_comgr_create_data
"""

# 10.0.0 dylib-client characterization: defined dynsym has amd_comgr_* and
# version nodes only. A separate `nm -D` (without --defined-only) would show
# `U LLVMInitializeAMDGPUTarget`; --forbid-dynsym must not see that.
_NM_DEFINED_DYLIB_CLIENT = """\
0000000000001111 T amd_comgr_create_data
0000000000002222 T amd_comgr_create_action_info
0000000000000000 A LLVM_23.0
"""


def test_parse_needed_libraries_extracts_compiler_dylibs() -> None:
    assert vsl.parse_needed_libraries(_READELF_DYLIB_CLIENT) == [
        "libLLVM.so.23.0git",
        "libclang-cpp.so.23.0git",
        "libc.so.6",
    ]


def test_parse_needed_libraries_libc_only() -> None:
    assert vsl.parse_needed_libraries(_READELF_LIBC_ONLY) == ["libc.so.6"]


def test_matching_names_hits_libllvm_not_libc() -> None:
    assert vsl.matching_names(
        ["libLLVM.so.23.0git", "libc.so.6"],
        _COMPILER_DYLIB_PATTERN,
    ) == ["libLLVM.so.23.0git"]


def test_matching_names_hits_other_compiler_dylibs() -> None:
    names = [
        "libclang.so.18",
        "libLTO.so",
        "libRemarks.so",
        "libclang-cpp.so.23.0git",
        "libc.so.6",
    ]
    assert vsl.matching_names(names, _COMPILER_DYLIB_PATTERN) == [
        "libclang.so.18",
        "libLTO.so",
        "libRemarks.so",
        "libclang-cpp.so.23.0git",
    ]


def test_matching_names_libclang_so_does_not_match_libclang_cpp() -> None:
    assert vsl.matching_names(["libclang-cpp.so.23.0git"], r"libclang[.]so") == []
    assert vsl.matching_names(["libclang-cpp.so.23.0git"], _COMPILER_DYLIB_PATTERN) == [
        "libclang-cpp.so.23.0git"
    ]


def test_matching_names_libc_only_is_empty() -> None:
    assert vsl.matching_names(["libc.so.6"], _COMPILER_DYLIB_PATTERN) == []


def test_matching_names_empty_pattern_matches_nothing() -> None:
    assert vsl.matching_names(["libLLVM.so.23.0git"], "") == []


def test_parse_defined_dynsyms_matches_llvm_c_api_and_mangled() -> None:
    names = vsl.parse_defined_dynsyms(_NM_DEFINED_WITH_LLVM)
    hits = vsl.matching_names(names, _LLVM_DYNSYM_PATTERN)
    assert "LLVMInitializeAMDGPUTarget" in hits
    assert "LLVMInitializeAMDGPUTarget@@LLVM_23.0" in hits
    assert "_ZN4llvm11raw_ostream5writeEPKcm" in hits
    assert "amd_comgr_create_data" not in hits


def test_llvm_dynsym_pattern_ignores_version_nodes() -> None:
    # `nm -D --defined-only` also prints version nodes as absolute symbols.
    # ^LLVM[A-Z] must not match `LLVM_23.0` (underscore, not a letter),
    # otherwise a correct static build would false-positive.
    assert vsl.matching_names(["LLVM_23.0"], _LLVM_DYNSYM_PATTERN) == []


def test_dylib_client_defined_dynsym_has_no_llvm_exports() -> None:
    names = vsl.parse_defined_dynsyms(_NM_DEFINED_DYLIB_CLIENT)
    assert vsl.matching_names(names, _LLVM_DYNSYM_PATTERN) == []
    assert "amd_comgr_create_data" in names


def _ns(
    shared_libs: list[str],
    *,
    forbid_needed: str = "",
    forbid_dynsym: str = "",
) -> argparse.Namespace:
    return argparse.Namespace(
        shared_libs=shared_libs,
        forbid_needed=forbid_needed,
        forbid_dynsym=forbid_dynsym,
    )


def test_run_forbid_needed_exits_1_on_libllvm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(vsl.sys, "platform", "linux")
    fake_so = tmp_path / "libamd_comgr.so"
    fake_so.write_bytes(b"not-an-elf")

    monkeypatch.setattr(vsl.ctypes.cdll, "LoadLibrary", lambda _path: object())

    def fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):
        assert cmd[0] == "readelf"
        assert "-d" in cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout=_READELF_DYLIB_CLIENT, stderr=""
        )

    monkeypatch.setattr(vsl.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        vsl.run(_ns([str(fake_so)], forbid_needed=_COMPILER_DYLIB_PATTERN))
    assert exc.value.code == 1


def test_run_forbid_dynsym_green_on_dylib_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(vsl.sys, "platform", "linux")
    fake_so = tmp_path / "libamd_comgr.so"
    fake_so.write_bytes(b"not-an-elf")
    monkeypatch.setattr(vsl.ctypes.cdll, "LoadLibrary", lambda _path: object())

    def fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):
        assert cmd[0] == "nm"
        assert "-D" in cmd
        assert "--defined-only" in cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout=_NM_DEFINED_DYLIB_CLIENT, stderr=""
        )

    monkeypatch.setattr(vsl.subprocess, "run", fake_run)
    vsl.run(_ns([str(fake_so)], forbid_dynsym=_LLVM_DYNSYM_PATTERN))


def test_run_forbid_dynsym_exits_1_when_defined(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(vsl.sys, "platform", "linux")
    fake_so = tmp_path / "libamd_comgr.so"
    fake_so.write_bytes(b"not-an-elf")
    monkeypatch.setattr(vsl.ctypes.cdll, "LoadLibrary", lambda _path: object())

    def fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):
        return subprocess.CompletedProcess(
            cmd, 0, stdout=_NM_DEFINED_WITH_LLVM, stderr=""
        )

    monkeypatch.setattr(vsl.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        vsl.run(_ns([str(fake_so)], forbid_dynsym=_LLVM_DYNSYM_PATTERN))
    assert exc.value.code == 1


def test_run_missing_readelf_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(vsl.sys, "platform", "linux")
    fake_so = tmp_path / "libamd_comgr.so"
    fake_so.write_bytes(b"not-an-elf")
    monkeypatch.setattr(vsl.ctypes.cdll, "LoadLibrary", lambda _path: object())

    def fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(vsl.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        vsl.run(_ns([str(fake_so)], forbid_needed=_COMPILER_DYLIB_PATTERN))
    assert exc.value.code == 1


def test_run_tool_nonzero_exit_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # check=True raises CalledProcessError; it must be caught, not escape as
    # a bare traceback. Measured: `readelf -d` exits 1 on a non-ELF file.
    monkeypatch.setattr(vsl.sys, "platform", "linux")
    fake_so = tmp_path / "libamd_comgr.so"
    fake_so.write_bytes(b"not-an-elf")
    monkeypatch.setattr(vsl.ctypes.cdll, "LoadLibrary", lambda _path: object())

    def fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="bad ELF")

    monkeypatch.setattr(vsl.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        vsl.run(_ns([str(fake_so)], forbid_needed=_COMPILER_DYLIB_PATTERN))
    assert exc.value.code == 1


def test_main_wires_forbid_needed_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Goes through argparse, unlike the _ns() tests above. Without this, a
    # typo'd flag name or dest would leave every other test green while
    # silently disarming the CMake gate.
    monkeypatch.setattr(vsl.sys, "platform", "linux")
    fake_so = tmp_path / "libamd_comgr.so"
    fake_so.write_bytes(b"not-an-elf")
    monkeypatch.setattr(vsl.ctypes.cdll, "LoadLibrary", lambda _path: object())

    def fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):
        return subprocess.CompletedProcess(
            cmd, 0, stdout=_READELF_DYLIB_CLIENT, stderr=""
        )

    monkeypatch.setattr(vsl.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        vsl.main(["--forbid-needed", _COMPILER_DYLIB_PATTERN, str(fake_so)])
    assert exc.value.code == 1


def test_run_forbid_flags_exit_2_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vsl.sys, "platform", "win32")
    with pytest.raises(SystemExit) as exc:
        vsl.run(_ns([], forbid_needed="libLLVM"))
    assert exc.value.code == 2


def test_main_accepts_cmake_argument_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # therock_testing.cmake emits `<lib> --forbid-needed <pattern>`, i.e. the
    # positional first. The other main() test uses the opposite order, so
    # without this the shipped ordering is untested.
    monkeypatch.setattr(vsl.sys, "platform", "linux")
    fake_so = tmp_path / "libamd_comgr.so"
    fake_so.write_bytes(b"not-an-elf")
    monkeypatch.setattr(vsl.ctypes.cdll, "LoadLibrary", lambda _path: object())

    def fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):
        return subprocess.CompletedProcess(
            cmd, 0, stdout=_READELF_DYLIB_CLIENT, stderr=""
        )

    monkeypatch.setattr(vsl.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        vsl.main([str(fake_so), "--forbid-needed", _COMPILER_DYLIB_PATTERN])
    assert exc.value.code == 1
