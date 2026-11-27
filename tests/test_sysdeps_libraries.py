# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Packaging validation of the bundled sysdeps shared libraries.

Bundled system dependencies (third-party/sysdeps/) are renamed and symbol
versioned so that ROCm can ship private copies alongside system libraries.
These tests check every shipped sysdeps library (`lib<x>.so` in
lib/rocm_sysdeps/lib).

Check                    Verifies                                      On failure
-----------------------  --------------------------------------------  ----------------------
test_version_tag         provides `<SONAME>(AMDROCM_SYSDEPS_1.0...)`   FAIL (never exempted)
                         (else RPM install fails, #3085)
test_requires_satisfied  every librocm_sysdeps_* requirement is        FAIL (never exempted)
                         provided by a sysdeps library
test_soname              SONAME is the `.so.N` file name               FAIL; WARN if EXEMPT
test_naming              `.so.N` file is `librocm_sysdeps_<x>.so.<N>`  FAIL; WARN if EXEMPT
                         and matches `lib<x>.so`
test_pc_file             .pc files use `${pcfiledir}`, no absolute or  FAIL; WARN if EXEMPT
                         build paths
test_layout              `lib<x>.so` -> real `.so.N`, exactly these    WARN only
                         two entries (no `.so.N.X`, no extra links)

When a check fails:
  - test_version_tag / test_requires_satisfied: fix the library's symbol
    versioning (third-party/sysdeps/*/patch_source.sh / version script). These
    can not be exempted: the packages would not install.
  - test_soname / test_naming / test_pc_file: fix the library, or, if the
    deviation is intended or tracked, add `"<lib>.so": "reason"` (or
    `"<name>.pc"`) to EXEMPTIONS. Its problems are then reported as warnings.
    test_exemption_still_needed fails for names which no longer exist and
    warns for names with no problems left.

Usage:
    THEROCK_SYSDEPS_LIB_DIR=/path/to/install/lib/rocm_sysdeps/lib \\
        python -m pytest tests/test_sysdeps_libraries.py -v

THEROCK_SYSDEPS_LIB_DIR is the lib/rocm_sysdeps/lib directory of a ROCm install
or of the extracted sysdeps artifacts. The RPM checks need RPM's `elfdeps`
(`apt install rpm`) and are skipped without it.
"""

import functools
import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import pytest

LIB_DIR = Path(d) if (d := os.getenv("THEROCK_SYSDEPS_LIB_DIR")) else None
ELFDEPS = "/usr/lib/rpm/elfdeps" if Path("/usr/lib/rpm/elfdeps").exists() else None

SONAME_PREFIX = "librocm_sysdeps_"
VERSION_TAG = "AMDROCM_SYSDEPS_1.0"

# Libraries (or .pc files) with known packaging deviations: their soname, naming
# and pc_file problems are reported as warnings instead of failures. The RPM
# installation checks are never exempted. Remove an entry once fixed.
EXEMPTIONS: dict[str, str] = {
    "libbz2.so": "SONAME has no .N major version",
    "libsqlite3.so": "SONAME has no .N major version",
    "libgallium_drv_video.so": "VA driver plugin: loaded by path, no .N",
    "radeonsi_drv_video.so": "VA driver plugin: loaded by path, no .N",
    "libiberty.so": "not renamed to librocm_sysdeps_* or symbol versioned",
}

# Scanned once, so every test sees the same set. Linker names only: the
# librocm_sysdeps_*.so namelinks are checked through their lib<x>.so.
LIBRARIES = sorted(
    p.name
    for p in (LIB_DIR.glob("*.so") if LIB_DIR else [])
    if not p.name.startswith(SONAME_PREFIX)
)
PC_FILES = sorted(p.name for p in (LIB_DIR.glob("pkgconfig/*.pc") if LIB_DIR else []))


class SysdepsFormatWarning(UserWarning):
    """A sysdeps packaging format issue which does not break RPM installation."""


def run(*args: str | Path) -> str:
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    assert result.returncode == 0, f"{' '.join(map(str, args))}: {result.stderr}"
    return result.stdout


def read_soname(path: Path) -> str | None:
    m = re.search(r"\(SONAME\)\s+.*\[(.+)\]", run("readelf", "-d", "-W", path))
    return m.group(1) if m else None


def elfdeps(option: str, path: Path) -> list[str]:
    return run(ELFDEPS, option, path).split()


def library_path(lib: str) -> Path:
    return LIB_DIR / lib


def so_n_file(lib: str) -> Path:
    """The file `lib<x>.so` links to (itself if not a link).

    It has the same contents as `lib<x>.so`, so the ELF checks only read it.
    """
    path = library_path(lib)
    return path.with_name(path.readlink().name) if path.is_symlink() else path


@functools.cache
def all_provides() -> set[str]:
    """RPM provides of all sysdeps libraries."""
    files = {so_n_file(lib) for lib in LIBRARIES}
    return {p for f in files if f.exists() for p in elfdeps("--provides", f)}


def soname_errors(lib: str) -> list[str]:
    target = so_n_file(lib)
    soname = read_soname(target)
    if soname != target.name:
        return [f"{target.name}: SONAME is {soname}, expected {target.name}"]
    return []


def normalized_name(name: str) -> str:
    """`libnl-genl-3` and `nl_genl_3` both -> `nlgenl3`."""
    return re.sub("[-_]", "", name.removeprefix("lib"))


def naming_errors(lib: str) -> list[str]:
    name = so_n_file(lib).name
    m = re.fullmatch(rf"{SONAME_PREFIX}(.+)\.so\.[0-9]+", name)
    linker_name = lib.removesuffix(".so")
    if not m or normalized_name(m.group(1)) != normalized_name(linker_name):
        return [f"{name} is not named {SONAME_PREFIX}<x>.so.<N> for {lib}"]
    return []


def layout_errors(lib: str) -> list[str]:
    path = library_path(lib)
    target = so_n_file(lib)
    if not path.is_symlink():
        return [f"{lib} is not a symlink to the .so.N file"]
    if not target.exists():
        return [f"{lib}: symlink target {target.name} does not exist"]
    errors: list[str] = []
    real_path = target.resolve()
    if target.is_symlink():
        errors.append(f"{target.name} is a symlink to {real_path.name}, not the file")
    # Other links to the library, and leftover versioned copies such as
    # librocm_sysdeps_z.so.1.3.2 next to librocm_sysdeps_z.so.1.
    others = sorted(
        p.name
        for p in LIB_DIR.iterdir()
        if p.name not in (lib, target.name)
        and (p.resolve() == real_path or p.name.startswith(f"{target.name}."))
    )
    if others:
        errors.append(f"extra entries for the same library: {', '.join(others)}")
    return errors


def pc_file_errors(pc_file: str) -> list[str]:
    errors: list[str] = []
    lines = (LIB_DIR / "pkgconfig" / pc_file).read_text().splitlines()
    for lineno, line in enumerate(lines, 1):
        # Comments and free-text metadata fields are not used when building.
        if line.lstrip().startswith("#") or line.startswith(
            ("Name:", "Description:", "URL:", "Version:")
        ):
            continue
        if line.startswith("prefix=") and not line.startswith("prefix=${pcfiledir}"):
            errors.append(f"line {lineno}: prefix is not relative to ${{pcfiledir}}")
        # An absolute path as a value or flag: `libdir=/...`, `-L/...`,
        # `-I/...`, `-Wl,-rpath,/...` or a standalone `/...` word (not
        # `http://...`).
        elif m := re.search(r"(?:^|[=\s,])(?:-[LI])?(/[^\s,]+)", line):
            errors.append(f"line {lineno}: absolute path {m.group(1)}")
    return errors


def report(check: str, name: str, errors: list[str], strict: bool = True):
    """Fails on `errors`, or warns if not `strict` or `name` is in EXEMPTIONS."""
    if not errors:
        return
    details = "; ".join(errors)
    if name in EXEMPTIONS:
        details = f"EXEMPT ({EXEMPTIONS[name]}): {details}"
    elif strict:
        pytest.fail(details)
    warnings.warn(SysdepsFormatWarning(f"{check}[{name}]: {details}"))


pytestmark = [
    pytest.mark.skipif(LIB_DIR is None, reason="THEROCK_SYSDEPS_LIB_DIR not set"),
    pytest.mark.skipif(not shutil.which("readelf"), reason="readelf not found"),
]
needs_elfdeps = pytest.mark.skipif(not ELFDEPS, reason="RPM elfdeps not installed")


def test_libraries_found():
    """Guards against a wrong or empty THEROCK_SYSDEPS_LIB_DIR."""
    assert LIBRARIES, f"no sysdeps libraries (lib*.so) found in {LIB_DIR}"


@pytest.mark.parametrize("name", sorted(EXEMPTIONS))
def test_exemption_still_needed(name: str):
    if name in PC_FILES:
        errors = pc_file_errors(name)
    else:
        assert name in LIBRARIES, f"EXEMPTIONS entry {name} not found in {LIB_DIR}"
        errors = naming_errors(name) + soname_errors(name)
    if not errors:
        warnings.warn(SysdepsFormatWarning(f"{name} has no problems left: remove it"))


# RPM installation checks: always fail.


@needs_elfdeps
@pytest.mark.parametrize("lib", LIBRARIES)
def test_version_tag(lib: str):
    """The library provides its AMDROCM_SYSDEPS_1.0 symbol version tag."""
    target = so_n_file(lib)
    assert target.exists(), f"{lib} -> {target.name}: file does not exist"
    soname = read_soname(target)
    if not (soname or "").startswith(SONAME_PREFIX):
        pytest.skip(f"not a renamed sysdeps library (SONAME {soname})")
    provides = elfdeps("--provides", target)
    assert any(
        p.startswith(f"{soname}({VERSION_TAG}") for p in provides
    ), f"{target.name}: no {soname}({VERSION_TAG}...) in: {' '.join(provides)}"


@needs_elfdeps
@pytest.mark.parametrize("lib", LIBRARIES)
def test_requires_satisfied(lib: str):
    """Every librocm_sysdeps_* dependency is provided, as RPM checks on install."""
    target = so_n_file(lib)
    assert target.exists(), f"{lib} -> {target.name}: file does not exist"
    requires = {r for r in elfdeps("--requires", target) if r.startswith(SONAME_PREFIX)}
    missing = sorted(requires - all_provides())
    assert not missing, f"{target.name}: not provided: {' '.join(missing)}"


# Packaging rules: fail unless exempted. Layout: warn only.


@pytest.mark.parametrize("lib", LIBRARIES)
def test_soname(lib: str):
    report("soname", lib, soname_errors(lib))


@pytest.mark.parametrize("lib", LIBRARIES)
def test_naming(lib: str):
    report("naming", lib, naming_errors(lib))


@pytest.mark.parametrize("pc_file", PC_FILES)
def test_pc_file(pc_file: str):
    report("pc_file", pc_file, pc_file_errors(pc_file))


@pytest.mark.parametrize("lib", LIBRARIES)
def test_layout(lib: str):
    report("layout", lib, layout_errors(lib), strict=False)
