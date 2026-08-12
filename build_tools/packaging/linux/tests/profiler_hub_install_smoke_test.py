#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MANUAL/OPT-IN ONLY -- standalone install-and-load smoke test for profiler-hub.

No workflow step invokes this file. It is not part of CI and does not gate any
build or install-test job. The find_package/link/load proof it demonstrates has
been re-homed into ``native_linux_package_install_test.py`` as
``verify_profiler_hub_install()``, called from ``run_basic_verification()`` --
the pattern the rest of this repo's post-install verification already follows
-- so that proof now runs against a real, dependency-resolved package install
(``sudo apt install``) in the house install-test lane instead of a hand-rolled
single-``.deb`` ``dpkg -i`` that cannot resolve its own dependencies.

This file is kept only because it is the sole thing in the tree that exercises
the raw single-package ``dpkg -i`` path (as opposed to apt's repo-based,
dependency-resolving install). Run it by hand, locally, when you specifically
want to validate that path; do not wire it back into a workflow without also
fixing the dependency-resolution problem described below.

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
- ``dpkg``: real ``dpkg -i`` install into the system install prefix. Installing
  a single ``.deb`` this way does not resolve its dependency closure (``dpkg -i``
  never does); on a host missing ``amdrocm-runtime``/``amdrocm-sysdeps`` this
  fails at the configure stage until ``apt-get install -f -y`` is run to
  complete it, which this mode does automatically.
- ``staging``: ``dpkg-deb -x <deb> <staging_dir>`` extraction. Does not
  require root/sudo. Fallback for environments without a working
  root/sudo/apt (e.g. a bare compute node); still uses the real packaged file
  layout, it just does not touch the system install prefix. Because it only
  extracts profiler-hub's own .deb, files that live in its (unresolved)
  dependency packages -- e.g. ``librocm_sysdeps_dw.so.1`` from
  ``amdrocm-sysdeps`` -- are absent from the staging tree entirely, so a
  consumer linked against those symbols will not load from staging mode.
- ``auto`` (default): try ``dpkg``, fall back to ``staging`` if ``dpkg``/sudo
  is unavailable or fails for a permissions reason.

Run standalone (manual/local only)::

    python3 profiler_hub_install_smoke_test.py --packages-dir /path/to/dist

Or under pytest::

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


def _stderr_text(e: subprocess.CalledProcessError) -> str:
    """Best-effort decoded/stripped stderr from a CalledProcessError, or "" if none."""
    stderr = e.stderr
    if stderr is None:
        return ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    return stderr.strip()


def _is_sudo_denial(e: subprocess.CalledProcessError) -> bool:
    """True if ``sudo`` itself rejected the invocation, not the wrapped command.

    ``sudo -n`` exits non-zero and prints its own diagnostic prefixed with
    ``sudo:`` (e.g. ``sudo: a password is required``) *before* the wrapped
    command ever runs. A failure from the wrapped command itself -- e.g.
    dpkg's own dependency diagnostics -- has no such prefix. Conflating the
    two previously caused a real dpkg failure to be misreported as "no
    root/sudo available here" (see module context / TheRock job 93909323734).
    """
    return any(
        line.strip().startswith("sudo:") for line in _stderr_text(e).splitlines()
    )


def install_via_dpkg(deb_path: Path) -> bool:
    """Real ``dpkg -i`` install, self-healing missing dependencies via ``apt-get -f``.

    ``dpkg -i`` does not resolve dependencies: installing a single .deb whose
    dependencies are not already present unpacks fine but fails at the
    *configure* stage with "dependency problems prevent configuration". That is
    expected, ordinary ``dpkg -i`` behaviour, not a permissions problem -- so it
    is remediated in place with ``apt-get install -f -y`` (the classic
    dpkg-then-fix-deps two-step) rather than reported as one.

    Returns:
        True on success (either the initial ``dpkg -i`` or the ``apt-get -f``
        remediation). False only when root/sudo access itself is denied (sudo
        missing, or ``sudo -n`` rejecting the invocation) -- the one condition
        under which the caller should fall back to a non-root install mode --
        or when ``apt-get install -f -y`` also fails to fix the dependencies.
    """
    try:
        _run(["sudo", "-n", "dpkg", "-i", str(deb_path)], stderr=subprocess.PIPE)
        return True
    except FileNotFoundError as e:
        print(f"[INFO] sudo/dpkg not available here: {e}")
        return False
    except subprocess.CalledProcessError as e:
        if _is_sudo_denial(e):
            print(
                "[INFO] sudo -n denied (no passwordless sudo available here): "
                f"{_stderr_text(e) or e}"
            )
            return False
        print(
            f"[INFO] `dpkg -i` failed at exit {e.returncode} (dependency problems "
            "are expected here -- dpkg -i does not resolve dependencies). dpkg "
            f"stderr:\n{_stderr_text(e) or '(no stderr captured)'}"
        )
        print("[INFO] Attempting `apt-get install -f -y` to resolve dependencies...")
        try:
            _run(
                ["sudo", "-n", "apt-get", "install", "-f", "-y"], stderr=subprocess.PIPE
            )
            return True
        except FileNotFoundError as fix_e:
            print(f"[INFO] apt-get not available here: {fix_e}")
            return False
        except subprocess.CalledProcessError as fix_e:
            print(
                f"[INFO] `apt-get install -f -y` failed to fix dependencies at exit "
                f"{fix_e.returncode}. apt-get stderr:\n"
                f"{_stderr_text(fix_e) or '(no stderr captured)'}"
            )
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
                f"[INFO] FALLBACK: using dpkg-deb -x staging install at {install_root}. "
                "See the `[INFO]` lines above for why `dpkg -i` was not used here. Uses "
                "the real .deb payload but does not touch the system install prefix; a "
                "real `dpkg -i` (this file is not invoked by any CI workflow; see the "
                "module docstring) additionally exercises postinst scripts and apt "
                "dependency resolution, which this fallback does not."
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
