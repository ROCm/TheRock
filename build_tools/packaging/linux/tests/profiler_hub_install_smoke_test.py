#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Standalone install-and-load smoke test for the profiler-hub native package.

Proves the *installed* ``amdrocm-profiler-base`` artifact is actually
consumable: locates the already-built ``.deb`` (produced by
``build_package.py`` in a prior step), installs it, then configures, compiles,
and runs a trivial consumer against the installed CMake package config (not
the profiler-hub build tree) via
``build_tools/packaging/linux/tests/fixtures/profiler_hub_consumer``. This
exercises CMake ``find_package`` resolution, header availability, linking
against ``libprofiler-hub.so``, and runtime ``NEEDED``-dependency resolution
end to end -- a missing runtime dependency surfaces as a loader failure, not
just a link-time pass.

Install modes (``--install-mode``):
- ``dpkg``: real ``dpkg -i`` install into the system install prefix (matches
  production installs and CI, where this test runs in-container with root
  after the "Build Packages" step).
- ``staging``: ``dpkg-deb -x <deb> <staging_dir>`` extraction. Does not
  require root/sudo. Fallback for environments without a working
  root/sudo/apt (e.g. a bare compute node); still uses the real packaged file
  layout, it just does not touch the system install prefix. A real
  ``dpkg -i`` additionally exercises postinst scripts and apt dependency
  resolution, which this fallback does not.
- ``auto`` (default): try ``dpkg``, fall back to ``staging`` if ``dpkg``/sudo
  is unavailable or fails for a permissions reason.

Run standalone::

    python3 profiler_hub_install_smoke_test.py --packages-dir /path/to/dist

Or under pytest (CI, env-var driven; see
``multi_arch_build_native_linux_packages.yml``)::

    pytest build_tools/packaging/linux/tests/profiler_hub_install_smoke_test.py -vv --tb=long -s
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = THIS_SCRIPT_DIR / "fixtures" / "profiler_hub_consumer"

PKG_NAME = "amdrocm-profiler-base"
DEFAULT_INSTALL_PREFIX = "/opt/rocm/core"

# KEY_COMPONENTS-style structure (mirrors native_linux_package_install_test.py's
# VERIFY_KEY_COMPONENTS): paths expected under the install prefix once
# amdrocm-profiler-base is installed, checked before we even attempt to build
# the consumer. A miss here is a packaging regression, not a consumer bug.
KEY_COMPONENTS = [
    "lib/cmake/profiler-hub/profiler-hub-config.cmake",
    "include/profiler-hub",
    "lib/libprofiler-hub.so",
]

# Anchor component used to DISCOVER the real install root (unique within the
# package payload), rather than predicting it from build_package.py's
# version-suffix convention (e.g. "/opt/rocm/core" -> "/opt/rocm/core-7.15").
ANCHOR_COMPONENT = KEY_COMPONENTS[0]

RUN_TIMEOUT_SEC = 30


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, echoing it first, raising on non-zero exit (check=True)."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    kwargs.setdefault("check", True)
    return subprocess.run(cmd, **kwargs)


def find_deb(packages_dir: Path) -> Path:
    """Locate the already-built amdrocm-profiler-base .deb in packages_dir."""
    matches = sorted(Path(packages_dir).glob(f"{PKG_NAME}*.deb"))
    if not matches:
        raise FileNotFoundError(
            f"No {PKG_NAME}*.deb found in {packages_dir}. Expected "
            f"build_package.py to have already produced it (--pkg-names {PKG_NAME})."
        )
    return matches[0]


def install_via_dpkg(deb_path: Path) -> bool:
    """Real ``dpkg -i`` install. Returns True on success, False if not permitted."""
    try:
        _run(["sudo", "-n", "dpkg", "-i", str(deb_path)])
        # Resolve any dependency gaps the same way a real install would.
        _run(["sudo", "-n", "apt-get", "install", "-f", "-y"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[INFO] dpkg -i install not available/permitted here: {e}")
        return False


def install_via_staging(deb_path: Path, staging_dir: Path) -> Path:
    """Extract the real .deb payload with dpkg-deb -x (no root required)."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    _run(["dpkg-deb", "-x", str(deb_path), str(staging_dir)])
    return staging_dir


def _strip_anchor_tail(path: Path, anchor: str) -> Path:
    """Strip the trailing ``anchor`` relative path off ``path``, returning the root."""
    anchor_parts = Path(anchor).parts
    path_parts = path.parts
    if path_parts[-len(anchor_parts) :] != anchor_parts:
        raise ValueError(f"{path} does not end with anchor {anchor!r}")
    return Path(*path_parts[: -len(anchor_parts)])


def discover_install_root_dpkg(
    pkg_name: str = PKG_NAME, anchor: str = ANCHOR_COMPONENT
) -> Path:
    """Discover the real install root via ``dpkg -L <pkg_name>`` (dpkg mode ground truth).

    Asks dpkg where it actually put the anchor component, rather than predicting the
    root from build_package.py's version-suffix convention.
    """
    result = _run(
        ["dpkg", "-L", pkg_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    anchor_suffix = "/" + anchor
    matches = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().endswith(anchor_suffix)
    ]
    if not matches:
        raise RuntimeError(
            f"`dpkg -L {pkg_name}` did not list anchor component {anchor!r}; the "
            "package did not install the expected profiler-hub component."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"`dpkg -L {pkg_name}` listed anchor component {anchor!r} more than once: "
            f"{matches}"
        )
    return _strip_anchor_tail(Path(matches[0]), anchor)


def discover_install_root_staging(
    staged_root: Path, anchor: str = ANCHOR_COMPONENT
) -> Path:
    """Discover the real install root under a ``dpkg-deb -x`` staging extraction.

    Rglobs the staging tree for the anchor component, rather than predicting the root
    from build_package.py's version-suffix convention.
    """
    anchor_name = Path(anchor).name
    anchor_suffix = "/" + anchor
    matches = [
        p
        for p in Path(staged_root).rglob(anchor_name)
        if p.as_posix().endswith(anchor_suffix)
    ]
    if not matches:
        raise RuntimeError(
            f"Could not find anchor component {anchor!r} anywhere under staging dir "
            f"{staged_root}; the package did not install the expected profiler-hub "
            "component."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Anchor component {anchor!r} matched multiple paths under {staged_root}: "
            f"{matches}"
        )
    return _strip_anchor_tail(matches[0], anchor)


def _cross_check_install_prefix(install_root: Path, install_prefix: str) -> None:
    """Optional sanity net, NOT the source of truth (discovery above is).

    If the caller passed ``--install-prefix``, assert the discovered root's name is
    consistent with it (allowing a version suffix, e.g. "/opt/rocm/core" ->
    ".../core-7.15"). A mismatch here means --install-prefix and the real package
    payload disagree, which is worth failing loudly on.
    """
    expected_name = Path(install_prefix).name
    if not install_root.name.startswith(expected_name):
        raise AssertionError(
            f"Cross-check failed: discovered install root {install_root} does not "
            f"match --install-prefix {install_prefix!r} (expected name to start with "
            f"{expected_name!r}). Discovery is authoritative; this likely means "
            "--install-prefix is stale/wrong, not that the discovered root is wrong."
        )


def verify_key_components(install_root: Path) -> None:
    print("\nChecking for key profiler-hub components:")
    missing = []
    for component in KEY_COMPONENTS:
        found = (install_root / component).exists()
        print(f" [{'PASS' if found else 'FAIL'}] {component}")
        if not found:
            missing.append(component)
    if missing:
        raise AssertionError(
            f"Missing installed components under {install_root}: {missing}"
        )


def build_and_run_consumer(install_root: Path, work_dir: Path) -> None:
    build_dir = work_dir / "consumer-build"
    _run(
        [
            "cmake",
            "-S",
            str(FIXTURE_DIR),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={install_root}",
            "-GNinja",
        ]
    )
    _run(["cmake", "--build", str(build_dir), "--clean-first"])

    exe = build_dir / "profiler_hub_consumer"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [str(install_root / "lib"), env.get("LD_LIBRARY_PATH", "")])
    )
    result = _run(
        [str(exe)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=RUN_TIMEOUT_SEC,
    )
    print(result.stdout)
    if "profiler-hub storage version" not in result.stdout:
        raise AssertionError(
            f"Consumer ran but did not print expected output: {result.stdout!r}"
        )


def run_smoke_test(
    packages_dir: str,
    install_prefix: str = DEFAULT_INSTALL_PREFIX,
    install_mode: str = "auto",
) -> int:
    """Run the full install-and-load smoke test. Returns 0 on success, raises otherwise."""
    deb_path = find_deb(Path(packages_dir))
    print(f"[PASS] Found package: {deb_path}")

    with tempfile.TemporaryDirectory(prefix="profiler_hub_smoke_") as tmp:
        work_dir = Path(tmp)
        used_mode = install_mode

        if install_mode in ("dpkg", "auto"):
            if install_via_dpkg(deb_path):
                install_root = discover_install_root_dpkg()
                used_mode = "dpkg"
            elif install_mode == "dpkg":
                raise RuntimeError("--install-mode dpkg requested but dpkg -i failed")
            else:
                used_mode = "staging"

        if used_mode == "staging":
            staged_root = install_via_staging(deb_path, work_dir / "staging")
            install_root = discover_install_root_staging(staged_root)
            print(
                f"[INFO] FALLBACK: using dpkg-deb -x staging install at {install_root} "
                "(no root/sudo available here). Uses the real .deb payload but does not "
                "touch the system install prefix; a real `dpkg -i` (used in CI) "
                "additionally exercises postinst scripts and apt dependency resolution, "
                "which this fallback does not."
            )

        _cross_check_install_prefix(install_root, install_prefix)
        print(f"\n[PASS] Installed (mode={used_mode}) at: {install_root}")
        verify_key_components(install_root)
        build_and_run_consumer(install_root, work_dir)
        print("\n[PASS] profiler-hub install-and-load smoke test PASSED")
        return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--packages-dir",
        required=True,
        help=f"Directory containing the already-built {PKG_NAME}*.deb",
    )
    p.add_argument(
        "--install-prefix",
        default=DEFAULT_INSTALL_PREFIX,
        help=(
            "Optional cross-check only: the real install root is discovered directly "
            "(dpkg -L / staging rglob), not predicted from this value. If provided, "
            "asserts the discovered root's name is consistent with it (must match "
            "build_package.py --install-prefix, minus any version suffix)."
        ),
    )
    p.add_argument(
        "--install-mode",
        choices=["auto", "dpkg", "staging"],
        default="auto",
        help="See module docstring for semantics of each mode.",
    )
    return p


def test_profiler_hub_install_smoke() -> None:
    """Pytest entry: env-var driven, mirrors native_linux_package_install_test.py."""
    import pytest

    packages_dir = os.environ.get("PACKAGE_DIST_DIR", "").strip()
    if not packages_dir:
        pytest.skip(
            "Set PACKAGE_DIST_DIR to the directory containing the built "
            f"{PKG_NAME} .deb (see multi_arch_build_native_linux_packages.yml)"
        )
    install_prefix = (
        os.environ.get("INSTALL_PREFIX", "").strip() or DEFAULT_INSTALL_PREFIX
    )
    install_mode = os.environ.get("PROFILER_HUB_INSTALL_MODE", "").strip() or "auto"

    run_smoke_test(packages_dir, install_prefix, install_mode)


def test_discover_install_root_staging_versioned(tmp_path) -> None:
    """Anchor found under a version-suffixed dir -> root includes the suffix."""
    anchor_path = tmp_path / "opt" / "rocm" / "core-7.15" / ANCHOR_COMPONENT
    anchor_path.parent.mkdir(parents=True)
    anchor_path.touch()

    install_root = discover_install_root_staging(tmp_path)

    assert install_root == tmp_path / "opt" / "rocm" / "core-7.15"


def test_discover_install_root_staging_unsuffixed(tmp_path) -> None:
    """Anchor found under an unsuffixed dir -> discovery is suffix-agnostic."""
    anchor_path = tmp_path / "opt" / "rocm" / "core" / ANCHOR_COMPONENT
    anchor_path.parent.mkdir(parents=True)
    anchor_path.touch()

    install_root = discover_install_root_staging(tmp_path)

    assert install_root == tmp_path / "opt" / "rocm" / "core"


def test_discover_install_root_staging_anchor_absent(tmp_path) -> None:
    import pytest

    (
        tmp_path / "opt" / "rocm" / "core-7.15" / "lib" / "libprofiler-hub.so"
    ).parent.mkdir(parents=True)

    with pytest.raises(RuntimeError) as exc_info:
        discover_install_root_staging(tmp_path)
    assert ANCHOR_COMPONENT in str(exc_info.value)


def test_discover_install_root_dpkg(monkeypatch) -> None:
    dpkg_output = (
        "/.\n"
        "/opt/rocm/core-7.15/lib/cmake/profiler-hub/profiler-hub-config.cmake\n"
        "/opt/rocm/core-7.15/lib/libprofiler-hub.so\n"
    )

    def fake_run(cmd, **kwargs):
        assert cmd[:2] == ["dpkg", "-L"]
        return subprocess.CompletedProcess(cmd, 0, stdout=dpkg_output, stderr="")

    monkeypatch.setattr(sys.modules[__name__], "_run", fake_run)

    install_root = discover_install_root_dpkg()

    assert install_root == Path("/opt/rocm/core-7.15")


def test_discover_install_root_dpkg_anchor_absent(monkeypatch) -> None:
    import pytest

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="/opt/rocm/core-7.15/lib/libprofiler-hub.so\n", stderr=""
        )

    monkeypatch.setattr(sys.modules[__name__], "_run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        discover_install_root_dpkg()
    assert ANCHOR_COMPONENT in str(exc_info.value)


def main() -> None:
    args = _build_argument_parser().parse_args()
    sys.exit(run_smoke_test(args.packages_dir, args.install_prefix, args.install_mode))


if __name__ == "__main__":
    main()
