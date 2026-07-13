#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Manual / on-demand test suite for the promote_packages promotion script.

This exercises promote_packages.py end-to-end against a **local directory of
already-downloaded** release-candidate (RC) artifacts. It does NOT fetch
anything over the network: point it at a directory that already contains the
wheels / sdist / therock-dist tarballs for one RC (e.g. produced by
download_python_packages.py, or copied by hand) and it will run every
promotion scenario over that set.

Promotion strips the RC suffix from version strings (e.g. 7.14.0rc1 -> 7.14.0)
and, for multi-arch inputs, prunes the per-gfx device wheels down to a subset of
architectures.

It is intentionally a standalone, on-demand script (run by a human before a
release), NOT a pytest/CI target: it installs the wheels into throwaway
virtualenvs, which is too heavy for per-PR CI. Keep it runnable by hand.

WHAT IS VALIDATED
  1. Full promotion of the arch-agnostic aggregator packages
     (rocm, rocm_sdk_core, rocm_sdk_devel, rocm_sdk_libraries, torch,
     torchvision, torchaudio, triton) succeeds and installs.
  2. Multi-arch promotion with a keep-list keeps only ONE arch's per-gfx device
     wheels (amd_torch_device_gfx*, amd_torchvision_device_gfx*,
     rocm_sdk_device_gfx*) and prunes the rest, while still stripping the RC
     suffix everywhere. The kept arch and the pruned arch(es) are auto-detected
     from whatever device wheels are present.
  3. JAX wheels (jax_rocm7_plugin, jax_rocm7_pjrt) promote correctly: the RC
     suffix is stripped from the `+rocm<ver>` local segment and the wheels stay
     structurally installable (installed --no-deps, since jax/jaxlib live on
     PyPI). Note jaxlib itself is NOT published multi-arch (see ROCm/TheRock#6158).
  4. Partial promotions (only ROCm, or only torch) do NOT produce a coherent,
     installable, single-version set (expected to FAIL).
  5. Every therock-dist tarball present (per-gfx AND multiarch) is promoted by
     its filename rc->final rename (promote_targz_tarball never opens the
     archive, so this is exercised with cheap stand-ins).
  6. Promoted filenames drop the RC suffix and every wheel reports the final
     version.

The RC version, the target platform, and the set of gfx architectures are all
DERIVED from the files in the input directory (the RC token is read from the
rocm sdist / rocm_sdk_core wheel; the final version is that with the prerelease
segment stripped). Expected promoted filenames are derived by applying the same
rc->final rename to whatever was found, so there are no brittle hardcoded lists.

triton is handled specially: torch pins one exact triton build
(`Requires-Dist: triton==<ver>`) but a download can contain several triton
builds, so the pinned one MUST be present in the directory (this is asserted)
and any non-pinned triton wheels are excluded from the staged set.

PREREQUISITES
  pip install -r ./build_tools/packaging/requirements.txt

USAGE
  # Promote + verify everything already downloaded into ./rc_packages:
  python ./build_tools/packaging/tests/promote_packages_test.py --input-dir ./rc_packages

  # Force a platform (otherwise inferred from the wheel tags present):
  python ./build_tools/packaging/tests/promote_packages_test.py \
      --input-dir ./rc_packages --platform linux
"""

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

from packaging.version import Version
from pkginfo import Wheel

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import promote_packages

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
import setup_venv

PLATFORMS = ("linux", "windows")

# Arch-agnostic aggregator wheels: always promoted, never pruned by the
# keep-list. Matched by filename prefix (device wheels use the distinct
# `*_device_gfx<N>` form and are matched separately).
AGGREGATOR_WHEEL_PREFIXES = (
    "rocm_sdk_core-",
    "rocm_sdk_devel-",
    "rocm_sdk_libraries-",
    "torch-",
    "torchvision-",
    "torchaudio-",
    "triton-",
)

# JAX wheels are arch-agnostic and Linux-only. They are promoted via the
# local-version path (the RC suffix is stripped from the `+rocm<ver>` segment).
JAX_WHEEL_PREFIXES = ("jax_rocm7_plugin-", "jax_rocm7_pjrt-")

# Per-gfx device wheel marker, e.g. `rocm_sdk_device_gfx942-...` /
# `amd_torch_device_gfx1201-...`. The trailing `-` anchors the arch token so
# `gfx11` can't match `gfx1153`.
_DEVICE_ARCH_RE = re.compile(r"device_(gfx[0-9a-z]+)-")

# RC version token as it appears on the rocm sdist / rocm_sdk_core wheel.
_SDIST_VER_RE = re.compile(r"^rocm-(\d+\.\d+\.\d+(?:rc\d+|a\d+)?)\.tar\.gz$")
_CORE_VER_RE = re.compile(r"^rocm_sdk_core-(\d+\.\d+\.\d+(?:rc\d+|a\d+)?)-")


def _upstream_version(name: str) -> Version:
    """Extract the upstream version field from a wheel/sdist filename.

    torch-2.11.0+rocm7.14.0rc1-...  -> 2.11.0
    rocm_sdk_core-7.14.0rc1-...     -> 7.14.0rc1
    """
    field = name.split("-", 2)[1] if "-" in name else name
    field = field.split("+", 1)[0]
    if field.endswith(".tar.gz"):
        field = field[: -len(".tar.gz")]
    return Version(field)


def _is_aggregator(name: str) -> bool:
    if name.startswith(AGGREGATOR_WHEEL_PREFIXES):
        return name.endswith(".whl")
    # rocm-<digit>... is the meta wheel / sdist; rocm-bootstrap / rocm-profiler
    # start with a letter after `rocm-` and are intentionally excluded.
    if re.match(r"rocm-\d", name):
        return name.endswith(".whl") or name.endswith(".tar.gz")
    return False


def _device_arch(name: str) -> str | None:
    m = _DEVICE_ARCH_RE.search(name)
    return m.group(1) if m else None


def _torch_triton_pin(wheel_path: Path) -> str | None:
    """The exact triton version torch pins (`Requires-Dist: triton==<ver>`).

    A download can contain several triton builds for one RC (e.g. 3.7.0+gitXXXX,
    3.7.1+gitYYYY); only the one torch pins is a coherent match, so read it from
    torch's own metadata rather than assuming newest.
    """
    with zipfile.ZipFile(wheel_path) as z:
        meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        text = z.read(meta).decode("utf-8", errors="replace")
    m = re.search(r"^Requires-Dist:\s*triton==([^\s;]+)", text, re.MULTILINE)
    return m.group(1) if m else None


def detect_rc_version(names: set[str]) -> str:
    """Read the RC version token from the rocm sdist or rocm_sdk_core wheel."""
    for regex in (_SDIST_VER_RE, _CORE_VER_RE):
        for name in sorted(names):
            m = regex.match(name)
            if m:
                return m.group(1)
    raise RuntimeError(
        "Could not determine the RC version from the input directory: expected a "
        "rocm-<ver>.tar.gz sdist or a rocm_sdk_core-<ver>-*.whl to be present."
    )


class InputSet:
    """Classification of an input directory into promotion groups."""

    def __init__(self, input_dir: Path, platform: str | None) -> None:
        names = {p.name for p in input_dir.iterdir() if p.is_file()}
        if not names:
            raise RuntimeError(f"No files found in input directory {input_dir}")

        self.input_dir = input_dir
        self.rc_str = detect_rc_version(names)
        self.rc_version = Version(self.rc_str)
        self.final_version = Version(self.rc_version.base_version)
        self.final_str = str(self.final_version)
        self.src_version_type = "rc" if "rc" in self.rc_str else "a"
        self.platform = platform or (
            "windows" if any("win_amd64" in n for n in names) else "linux"
        )

        self.device_files = {n for n in names if _device_arch(n)}
        self.jax_files = {n for n in names if n.startswith(JAX_WHEEL_PREFIXES)}
        self.tarball_files = {
            n
            for n in names
            if n.startswith("therock-dist")
            and n.endswith(".tar.gz")
            and self.rc_str in n
        }
        self.aggregator_files = {
            n for n in names if _is_aggregator(n) and n not in self.device_files
        }
        # Keep only torch's pinned triton; drop any other triton builds so the
        # staged set stays installable.
        self.aggregator_files -= self._non_pinned_triton()

        self.present_arches = sorted(
            {a for n in self.device_files if (a := _device_arch(n))}
        )
        # Auto-detect: keep the first present arch, everything else is pruned.
        self.keep_arch = self.present_arches[0] if self.present_arches else None
        self.dropped_device_files = {
            n for n in self.device_files if _device_arch(n) != self.keep_arch
        }

    def _non_pinned_triton(self) -> set[str]:
        torch_wheels = [n for n in self.aggregator_files if n.startswith("torch-")]
        triton_wheels = {n for n in self.aggregator_files if n.startswith("triton-")}
        if not torch_wheels or not triton_wheels:
            return set()
        torch = max(torch_wheels, key=_upstream_version)
        pin = _torch_triton_pin(self.input_dir / torch)
        if not pin:
            return set()
        matched = {n for n in triton_wheels if pin in n}
        if not matched:
            raise RuntimeError(
                f"{torch} pins triton=={pin} but no matching triton wheel is "
                f"present in {self.input_dir} (found: {sorted(triton_wheels)})"
            )
        return triton_wheels - matched

    def promoted(self, names: set[str]) -> set[str]:
        return {promoted_name(n, self.rc_str, self.final_str) for n in names}


def promoted_name(rc_name: str, version: str, final_version: str) -> str:
    return rc_name.replace(version, final_version)


def checkPromotedFileNames(dir_path: Path, expected: set[str]) -> tuple[bool, str]:
    actual = {p.name for p in dir_path.glob("*") if p.name != ".venv"}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        return (
            False,
            f"Promoted file set mismatch.\n  missing: {missing}\n  unexpected: {unexpected}",
        )
    return True, ""


def checkAllWheelsSameVersion(
    dir_path: Path, expected_version: Version
) -> tuple[bool, str]:
    for file in dir_path.glob("*.whl"):
        wheel = Wheel(file)
        version = Version(wheel.version)

        local_tag = "rocm" + str(expected_version)
        if str(version) == str(expected_version) and version.local is None:
            continue  # arch-agnostic / device SDK wheels
        # torch-family wheels carry the rocm tag as a local version segment,
        # sometimes prefixed with a git hash (e.g. `git43422b04.rocm7.14.0`).
        elif version.local is not None and version.local.endswith(local_tag):
            continue
        else:
            return (
                False,
                f"{file.name} has version {version}, but expected version is {expected_version}",
            )

    return True, ""


def checkInstallation(dir_path: Path, no_deps: bool = False) -> tuple[bool, str]:
    """Note: dir_path must be a TemporaryDirectory, otherwise clean up .venv yourself.

    no_deps installs the wheels without resolving dependencies -- used for the
    JAX wheels, whose runtime deps (jax, jaxlib) live on PyPI and are not part
    of the promoted set; we only need to confirm the promoted wheels are
    structurally installable.
    """
    try:
        setup_venv.create_venv(dir_path / ".venv")
        python_exe = setup_venv.find_venv_python_exe(dir_path / ".venv")
        if python_exe is None:
            return (
                False,
                "Problem when installing temporary venv: Python executable not found",
            )

        # Install wheels/sdists, not the therock-dist tarball or the venv itself.
        packages = [
            p
            for p in dir_path.glob("*")
            if p.name != ".venv" and "therock-dist" not in p.name
        ]

        cmd = [python_exe, "-m", "pip", "install"]
        if no_deps:
            cmd.append("--no-deps")
        subprocess.run(
            cmd + packages,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    return True, ""


def _stage_inputs(src: Path, dst: Path, names: set[str]) -> None:
    for file in src.glob("*"):
        if file.is_file() and file.name in names:
            shutil.copy2(file, dst)


def _run_checks(
    tmp_dir: Path, checks: list[tuple[str, tuple[bool, str]]], expect_success: bool
) -> bool:
    """Run named checks. Returns whether the overall expectation held."""
    for func_name, res in checks:
        if expect_success and not res[0]:
            print(f"\n[ERROR] {func_name} failed:\n{res[1]}")
            return False
        if not expect_success and res[0]:
            print(
                f"\n[ERROR] {func_name} succeeded but the promotion should NOT have been coherent."
            )
            proc = subprocess.run(
                ["ls", tmp_dir], capture_output=True, encoding="utf-8"
            )
            print(proc.stdout)
            return False
    return True


def _banner(msg: str) -> None:
    line = "=" * 81
    print(f"\n{line}\n{msg}\n{line}")


def checkPromoteEverything(
    dir_path: Path,
    expected_version: Version,
    input_files: set[str],
    expected_files: set[str],
    src_version_type: str = "rc",
) -> bool:
    _banner("TEST: promotion of all aggregator packages (should SUCCEED)")
    with tempfile.TemporaryDirectory(prefix="PromoteTest-Everything-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, input_files)
        promote_packages.main(tmp_dir, delete=True, src_version_type=src_version_type)
        ok = _run_checks(
            tmp_dir,
            [
                (
                    "checkPromotedFileNames",
                    checkPromotedFileNames(tmp_dir, expected_files),
                ),
                (
                    "checkAllWheelsSameVersion",
                    checkAllWheelsSameVersion(tmp_dir, expected_version),
                ),
                ("checkInstallation", checkInstallation(tmp_dir)),
            ],
            expect_success=True,
        )
    _banner(
        "TEST DONE: promote everything. Result: " + ("SUCCESS" if ok else "FAILURE")
    )
    return ok


def checkPromoteMultiArch(
    dir_path: Path,
    expected_version: Version,
    keep_arch: str,
    input_files: set[str],
    dropped_promoted_names: set[str],
    expected_files: set[str],
    src_version_type: str = "rc",
) -> bool:
    _banner(f"TEST: multi-arch promotion keeping only {keep_arch} (should SUCCEED)")
    with tempfile.TemporaryDirectory(prefix="PromoteTest-MultiArch-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, input_files)
        promote_packages.main(
            tmp_dir,
            delete=True,
            multi_arch_targets=[keep_arch],
            src_version_type=src_version_type,
        )

        checks = [
            ("checkPromotedFileNames", checkPromotedFileNames(tmp_dir, expected_files)),
            (
                "checkAllWheelsSameVersion",
                checkAllWheelsSameVersion(tmp_dir, expected_version),
            ),
            ("checkInstallation", checkInstallation(tmp_dir)),
        ]
        ok = _run_checks(tmp_dir, checks, expect_success=True)

        # Explicitly confirm the non-kept arch device wheels were pruned.
        if ok:
            leftover = dropped_promoted_names & {p.name for p in tmp_dir.glob("*")}
            if leftover:
                print(
                    f"\n[ERROR] non-kept-arch device wheels survived: {sorted(leftover)}"
                )
                ok = False
    _banner(
        "TEST DONE: multi-arch promotion. Result: " + ("SUCCESS" if ok else "FAILURE")
    )
    return ok


def checkPromoteJax(
    dir_path: Path,
    expected_version: Version,
    input_files: set[str],
    expected_files: set[str],
    src_version_type: str = "rc",
) -> bool:
    _banner("TEST: promotion of JAX wheels (should SUCCEED)")
    if not input_files:
        print("[SKIP] no JAX wheels found for this version; skipping.")
        _banner("TEST DONE: promote JAX. Result: SKIPPED")
        return True
    with tempfile.TemporaryDirectory(prefix="PromoteTest-Jax-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, input_files)
        promote_packages.main(tmp_dir, delete=True, src_version_type=src_version_type)
        ok = _run_checks(
            tmp_dir,
            [
                (
                    "checkPromotedFileNames",
                    checkPromotedFileNames(tmp_dir, expected_files),
                ),
                (
                    "checkAllWheelsSameVersion",
                    checkAllWheelsSameVersion(tmp_dir, expected_version),
                ),
                # JAX runtime deps (jax, jaxlib) are on PyPI, not in the promoted
                # set -- install --no-deps to validate the wheels structurally.
                ("checkInstallation", checkInstallation(tmp_dir, no_deps=True)),
            ],
            expect_success=True,
        )
    _banner("TEST DONE: promote JAX. Result: " + ("SUCCESS" if ok else "FAILURE"))
    return ok


def _write_placeholder_targz(path: Path) -> None:
    """A minimal valid .tar.gz stand-in. promote_targz_tarball renames the
    therock-dist tarball by filename and never opens it, so the contents are
    irrelevant -- this avoids copying the real multi-GB artifacts."""
    with tarfile.open(path, "w:gz") as tar:
        data = b"placeholder\n"
        info = tarfile.TarInfo(name="README")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def checkPromoteTarball(
    dir_path: Path,
    tarball_names: set[str],
    version: str,
    final_version: str,
    src_version_type: str = "rc",
) -> bool:
    """Every therock-dist tarball present (per-gfx AND multiarch) is promoted by
    a filename-based rc->final rename (promote_targz_tarball), not by repacking.

    promote_targz_tarball never opens the archive, so each real (multi-GB)
    tarball is exercised with a cheap same-named stand-in. The expected promoted
    name is derived by applying the same rc->final rename to the discovered name.
    """
    _banner(
        "TEST: promotion of therock-dist tarballs (per-gfx + multiarch, should SUCCEED)"
    )
    if not tarball_names:
        print("[SKIP] no therock-dist tarballs found in the input dir; skipping.")
        _banner("TEST DONE: promote tarball. Result: SKIPPED")
        return True

    print(f"  discovered {len(tarball_names)} tarball(s):")
    for n in sorted(tarball_names):
        print(f"    {n}")

    ok = True
    with tempfile.TemporaryDirectory(prefix="PromoteTest-Tarball-") as tmp:
        tmp_dir = Path(tmp)
        for rc_name in tarball_names:
            _write_placeholder_targz(tmp_dir / rc_name)
        promote_packages.main(tmp_dir, delete=True, src_version_type=src_version_type)
        produced = {p.name for p in tmp_dir.glob("*")}
        for rc_name in sorted(tarball_names):
            fin_name = promoted_name(rc_name, version, final_version)
            if fin_name not in produced:
                print(
                    f"\n[ERROR] expected {fin_name} after promotion; got {sorted(produced)}"
                )
                ok = False
            elif rc_name in produced:
                print(f"\n[ERROR] rc tarball {rc_name} survived promotion")
                ok = False
    _banner("TEST DONE: promote tarball. Result: " + ("SUCCESS" if ok else "FAILURE"))
    return ok


def checkPartialPromotion(
    dir_path: Path,
    expected_version: Version,
    input_files: set[str],
    expected_files: set[str],
    match_files: str,
    label: str,
    src_version_type: str = "rc",
) -> bool:
    _banner(f"TEST: promotion of only {label} packages (should FAIL)")
    with tempfile.TemporaryDirectory(prefix=f"PromoteTest-Only-{label}-") as tmp:
        tmp_dir = Path(tmp)
        _stage_inputs(dir_path, tmp_dir, input_files)
        promote_packages.main(
            tmp_dir,
            match_files=match_files,
            delete=True,
            src_version_type=src_version_type,
        )
        ok = _run_checks(
            tmp_dir,
            [
                (
                    "checkPromotedFileNames",
                    checkPromotedFileNames(tmp_dir, expected_files),
                ),
                (
                    "checkAllWheelsSameVersion",
                    checkAllWheelsSameVersion(tmp_dir, expected_version),
                ),
                ("checkInstallation", checkInstallation(tmp_dir)),
            ],
            expect_success=False,
        )
    _banner(
        f"TEST DONE: promote only {label}. Result: " + ("SUCCESS" if ok else "FAILURE")
    )
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="On-demand test of RC->final package promotion over a local directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of already-downloaded RC packages (wheels / sdist / tarballs) to promote and test.",
    )
    parser.add_argument(
        "--platform",
        choices=PLATFORMS,
        default=None,
        help="Target platform (default: inferred from the wheel tags in --input-dir).",
    )
    p = parser.parse_args(sys.argv[1:])

    if not p.input_dir.is_dir():
        parser.error(f"--input-dir {p.input_dir} is not a directory")

    inputs = InputSet(p.input_dir, p.platform)

    print(
        f"Testing promotion {inputs.rc_version} -> {inputs.final_version} on "
        f"{inputs.platform} from {inputs.input_dir}"
    )
    print(
        f"  arches present: {inputs.present_arches or '(none)'}; kept: {inputs.keep_arch}"
    )
    print(
        f"  aggregators: {len(inputs.aggregator_files)}, device: {len(inputs.device_files)}, "
        f"jax: {len(inputs.jax_files)}, tarballs: {len(inputs.tarball_files)}"
    )

    multi_arch_input = inputs.aggregator_files | inputs.device_files
    expected_aggregators = inputs.promoted(inputs.aggregator_files)
    expected_multi_arch = inputs.promoted(
        multi_arch_input - inputs.dropped_device_files
    )
    dropped_promoted_names = inputs.promoted(inputs.dropped_device_files)
    expected_jax = inputs.promoted(inputs.jax_files)

    src = inputs.src_version_type

    res_everything = checkPromoteEverything(
        inputs.input_dir,
        inputs.final_version,
        inputs.aggregator_files,
        expected_aggregators,
        src,
    )

    if inputs.keep_arch is not None and inputs.dropped_device_files:
        res_multi = checkPromoteMultiArch(
            inputs.input_dir,
            inputs.final_version,
            inputs.keep_arch,
            multi_arch_input,
            dropped_promoted_names,
            expected_multi_arch,
            src,
        )
    else:
        _banner(
            "TEST: multi-arch promotion — SKIPPED "
            "(need >=2 gfx arches with device wheels to exercise pruning)"
        )
        res_multi = True

    res_only_rocm = checkPartialPromotion(
        inputs.input_dir,
        inputs.final_version,
        inputs.aggregator_files,
        expected_aggregators,
        "rocm*",
        "rocm",
        src,
    )
    res_only_torch = checkPartialPromotion(
        inputs.input_dir,
        inputs.final_version,
        inputs.aggregator_files,
        expected_aggregators,
        "*torch*",
        "torch",
        src,
    )
    res_jax = checkPromoteJax(
        inputs.input_dir, inputs.final_version, inputs.jax_files, expected_jax, src
    )
    res_tarball = checkPromoteTarball(
        inputs.input_dir, inputs.tarball_files, inputs.rc_str, inputs.final_str, src
    )

    _banner("SUMMARY")
    print(f"checkPromoteEverything:  {'SUCCESS' if res_everything else 'FAILURE'}")
    print(f"checkPromoteMultiArch:   {'SUCCESS' if res_multi else 'FAILURE'}")
    print(f"checkPromoteJax:         {'SUCCESS' if res_jax else 'FAILURE'}")
    print(f"checkPromoteTarball:     {'SUCCESS' if res_tarball else 'FAILURE'}")
    print(f"checkPromoteOnlyRocm:    {'SUCCESS' if res_only_rocm else 'FAILURE'}")
    print(f"checkPromoteOnlyTorch:   {'SUCCESS' if res_only_torch else 'FAILURE'}")
    print("=" * 81)

    if not all(
        [res_everything, res_multi, res_jax, res_tarball, res_only_rocm, res_only_torch]
    ):
        sys.exit(1)
