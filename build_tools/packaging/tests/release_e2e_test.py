#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Manual / on-demand end-to-end test of the release packaging pipeline.

Where promote_packages_test.py exercises the promotion *logic* in isolation
(curl + in-process promote_packages.main), this script drives the three real
release CLIs as subprocesses, exactly as a human does when cutting a release:

    download_python_packages.py  ->  promote_packages.py  ->  upload_release_packages.py

and inserts an install-verification step in between. It is intentionally a
standalone, on-demand script (run by hand before a release), NOT a pytest/CI
target: it pulls multiple GB of artifacts and builds a venv, which is too heavy
and too coupled to live S3 for per-PR CI.

WHAT IS VALIDATED (against a single gfx arch + Python tag to stay lean)
  1. download: the multi-arch downloader fetches the gfx-arch dependency
     closure for the RC (rocm SDK + torch/torchvision + device wheels + jax +
     triton) from the prerelease bucket.
  2. promote: promote_packages.py strips the RC suffix everywhere (verified via
     filenames AND installed metadata), including the rocm sdist and the
     rocm_sdk_libraries layout auto-detect.
  3. install (the metadata contract): `pip install torch[device-<arch>]` from a
     local index made of ONLY the promoted wheels (PyPI for generic deps)
     auto-resolves the whole ROCm chain -- rocm[libraries] -> rocm_sdk_core /
     rocm_sdk_libraries, and amd-torch-device-<arch> -> rocm_sdk_device_<arch>,
     plus triton and rocm-bootstrap -- with NO rc suffix leaking anywhere.
  4. jax installs (--no-deps, since jax/jaxlib live on PyPI) at the final
     version.
  5. upload: upload_release_packages.py runs in DRY-RUN against the testing
     bucket and reports a non-empty set of files it would upload. It NEVER
     passes --execute, so nothing is written to S3.

Only 7.14.0rc1 and later publish the torch/jax/device wheels; older RCs (e.g.
7.13.0rc1) ship only the ROCm SDK, so steps 3-4 need a >=7.14 RC.

PREREQUISITES
  pip install -r ./build_tools/packaging/requirements.txt
  Valid AWS credentials with read on therock-prerelease-python (download) and,
  for a real --execute upload only, write on therock-testing-bucket.

USAGE
  # Full dry-run pass for gfx942 / cp312 against 7.14.0rc1:
  python ./build_tools/packaging/tests/release_e2e_test.py --version 7.14.0rc1

  # Keep the work dir for inspection and pin the torch build:
  python ./build_tools/packaging/tests/release_e2e_test.py \
      --version 7.14.0rc1 --arch gfx942 --python-tag cp312 \
      --workdir /tmp/release_e2e --keep

OPEN QUESTIONS (tracked in ROCm/TheRock#6266 follow-up; resolve before merge)
  - Auto-discover the latest RC vs. require --version (currently required)?
  - Should an actual (--execute) upload to the testing bucket be part of the
    automated pass, or stay a deliberate manual follow-up after the dry-run?
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.version import Version

PACKAGING_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_SCRIPT = PACKAGING_DIR / "download_python_packages.py"
PROMOTE_SCRIPT = PACKAGING_DIR / "promote_packages.py"
UPLOAD_SCRIPT = PACKAGING_DIR / "upload_release_packages.py"

# The rocm-family packages we expect the torch metadata to pull in transitively.
# Their installed version must equal the final (rc-stripped) rocm version.
ROCM_CHAIN = [
    "rocm",
    "rocm-sdk-core",
    "rocm-sdk-libraries",
]


def _banner(msg: str) -> None:
    line = "=" * 81
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, encoding="utf-8", **kwargs)


def download_closure(version: str, arch: str, py_tag: str, out_dir: Path) -> Path:
    """Download the single-arch dependency closure for the RC. Returns the
    directory that holds the flat wheels (the downloader writes to <out>/wheels)."""
    _banner(f"STEP 1/5 download: {version} closure for {arch} / {py_tag}")
    globs = [
        f"rocm-{version}.tar.gz",
        "rocm_sdk_core-*linux_x86_64*",
        "rocm_sdk_devel-*linux_x86_64*",
        f"rocm_sdk_libraries-{version}-*linux_x86_64*",
        f"rocm_sdk_device_{arch}-*linux_x86_64*",
        f"torch-*{py_tag}*linux_x86_64*",
        f"torchvision-*{py_tag}*linux_x86_64*",
        f"amd_torch_device_{arch}-*{py_tag}*linux_x86_64*",
        f"amd_torchvision_device_{arch}-*{py_tag}*linux_x86_64*",
        f"triton-*{py_tag}*linux_x86_64*",
        f"jax_rocm7_plugin-*{py_tag}*",
        "jax_rocm7_pjrt-*",
    ]
    cmd = [
        sys.executable,
        str(DOWNLOAD_SCRIPT),
        f"--version={version}",
        "--multi-arch",
        "--bucket-prefix=v4/whl/",
        f"--output-dir={out_dir}",
    ]
    for g in globs:
        cmd += ["--include-package-glob", g]
    _run(cmd, check=True)

    wheels = out_dir / "wheels"
    if not wheels.is_dir():
        wheels = out_dir
    count = len(list(wheels.glob("*.whl"))) + len(list(wheels.glob("*.tar.gz")))
    if count == 0:
        raise RuntimeError(
            f"Downloader produced no packages in {wheels}. Does {version} publish "
            "torch/jax/device wheels? (Only >=7.14 RCs do.)"
        )
    print(f"  downloaded {count} package(s) into {wheels}")
    return wheels


def promote(wheels: Path, final_version: str) -> None:
    _banner(f"STEP 2/5 promote: strip rc -> {final_version}")
    _run(
        [
            sys.executable,
            str(PROMOTE_SCRIPT),
            f"--input-dir={wheels}",
            "--delete-old-on-success",
        ],
        check=True,
    )
    leftover = [
        p.name for p in wheels.glob("*") if "rc" in _local_or_base_version(p.name)
    ]
    if leftover:
        raise AssertionError(f"rc suffix survived promotion: {leftover}")
    print("  no rc suffix remains in any promoted filename")


def _local_or_base_version(name: str) -> str:
    """Lowercased version field of a wheel/sdist filename (base or +local)."""
    if name.endswith(".tar.gz"):
        return name[: -len(".tar.gz")].split("-", 1)[-1].lower()
    parts = name.split("-")
    return parts[1].lower() if len(parts) > 1 else name.lower()


def _pick_torch_version(wheels: Path, py_tag: str, final_version: str) -> str:
    """Newest torch upstream built for py_tag at the final rocm version."""
    candidates = []
    for p in wheels.glob(f"torch-*+rocm{final_version}-{py_tag}-*linux_x86_64.whl"):
        candidates.append(Version(p.name.split("-")[1].split("+")[0]))
    if not candidates:
        raise RuntimeError(f"No promoted torch wheel for {py_tag} in {wheels}")
    return str(max(candidates))


def check_install_torch(
    wheels: Path, arch: str, py_tag: str, final_version: str, venv: Path
) -> None:
    """Install torch[device-<arch>] from the local index and confirm the whole
    ROCm chain auto-resolves with no rc leaking."""
    _banner(f"STEP 3/5 install: torch[device-{arch}] resolves the ROCm chain")
    torch_ver = _pick_torch_version(wheels, py_tag, final_version)
    _run([sys.executable, "-m", "venv", str(venv)], check=True)
    pip = venv / "bin" / "pip"
    py = venv / "bin" / "python"
    _run([str(pip), "install", "--upgrade", "pip"], check=True, capture_output=True)
    # Local index for the rocm-family wheels; PyPI stays enabled for generic
    # deps (numpy, jinja2, ...) and the tiny unversioned rocm-bootstrap wheel.
    _run(
        [
            str(pip),
            "install",
            "--find-links",
            str(wheels),
            f"torch[device-{arch}]=={torch_ver}",
        ],
        check=True,
    )

    installed = _venv_versions(
        py,
        ROCM_CHAIN
        + [
            f"rocm-sdk-device-{arch}",
            f"amd-torch-device-{arch}",
            "triton",
            "torch",
        ],
    )
    print("  installed versions:")
    rc_leaks = []
    for name, ver in installed.items():
        if ver is None:
            raise AssertionError(
                f"expected {name} to be auto-resolved, but it is not installed"
            )
        leak = "rc" in ver.lower()
        if leak:
            rc_leaks.append(f"{name}=={ver}")
        print(f"    {name:32} {ver}{'  <-- RC LEAK' if leak else ''}")
    if rc_leaks:
        raise AssertionError(f"rc suffix leaked into installed metadata: {rc_leaks}")

    # The rocm SDK chain must match the final release version exactly.
    for name in ROCM_CHAIN + [f"rocm-sdk-device-{arch}"]:
        if installed[name] != final_version:
            raise AssertionError(
                f"{name} installed as {installed[name]}, expected {final_version}"
            )
    print("  torch metadata auto-resolved the full ROCm chain at the final version")


def check_install_jax(wheels: Path, final_version: str, venv: Path) -> None:
    _banner("STEP 4/5 install: jax wheels (--no-deps) at the final version")
    plugins = sorted(wheels.glob("jax_rocm7_plugin-*.whl"))
    pjrts = sorted(wheels.glob("jax_rocm7_pjrt-*.whl"))
    if not plugins or not pjrts:
        print("  [SKIP] no jax wheels present; skipping")
        return
    pip = venv / "bin" / "pip"
    py = venv / "bin" / "python"
    # Pick the newest upstream of each to stay single-version.
    plugin = max(plugins, key=lambda p: Version(p.name.split("-")[1].split("+")[0]))
    pjrt = max(pjrts, key=lambda p: Version(p.name.split("-")[1].split("+")[0]))
    _run([str(pip), "install", "--no-deps", str(plugin), str(pjrt)], check=True)
    installed = _venv_versions(py, ["jax-rocm7-plugin", "jax-rocm7-pjrt"])
    for name, ver in installed.items():
        if ver is None or "rc" in (ver or "").lower():
            raise AssertionError(f"jax package {name} bad version: {ver}")
        if f"+rocm{final_version}" not in ver:
            raise AssertionError(f"{name}=={ver} does not carry +rocm{final_version}")
        print(f"    {name:22} {ver}")


def check_dry_run_upload(dl_root: Path) -> None:
    """upload_release_packages.py defaults to the testing bucket AND dry-run;
    confirm it enumerates files and never touches S3 (no --execute passed)."""
    _banner("STEP 5/5 upload: dry-run to testing bucket (no --execute)")
    proc = _run(
        [
            sys.executable,
            str(UPLOAD_SCRIPT),
            f"--input-dir={dl_root}",
            "--multi-arch",
        ],
        check=True,
        capture_output=True,
    )
    out = proc.stdout
    would = len(re.findall(r"\[DRY-RUN\] Would upload:", out))
    if "DRY-RUN" not in out:
        raise AssertionError("upload did not report DRY-RUN mode")
    if would == 0:
        raise AssertionError(f"dry-run found 0 files to upload.\n{out[-2000:]}")
    print(
        f"  dry-run would upload {would} file(s) to the testing bucket (nothing written)"
    )


def _venv_versions(py: Path, names: list[str]) -> dict[str, str | None]:
    code = (
        "import importlib.metadata as m, json, sys\n"
        "out={}\n"
        "for n in sys.argv[1:]:\n"
        "    try: out[n]=m.version(n)\n"
        "    except Exception: out[n]=None\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [str(py), "-c", code, *names],
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(proc.stdout)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="On-demand end-to-end release pipeline test (download -> promote -> install -> dry-run upload)."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="RC version to test (e.g. 7.14.0rc1). Must be >=7.14 to include torch/jax.",
    )
    parser.add_argument(
        "--arch",
        default="gfx942",
        help="gfx arch to fetch and install (default: gfx942).",
    )
    parser.add_argument(
        "--python-tag",
        default="cp312",
        help="Python ABI tag for torch/device/jax wheels (default: cp312).",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working directory (default: a temp dir under the system temp).",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the work dir instead of deleting it on success.",
    )
    args = parser.parse_args(argv)

    rc_version = Version(args.version)
    final_version = str(rc_version).split("rc")[0]

    import tempfile

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="release-e2e-"))
    workdir.mkdir(parents=True, exist_ok=True)
    dl_root = workdir / "download"
    if dl_root.exists():
        shutil.rmtree(dl_root)

    print(
        f"E2E release test: {rc_version} -> {final_version} "
        f"({args.arch} / {args.python_tag}); workdir={workdir}"
    )

    ok = True
    try:
        wheels = download_closure(str(rc_version), args.arch, args.python_tag, dl_root)
        promote(wheels, final_version)
        venv = workdir / "venv"
        if venv.exists():
            shutil.rmtree(venv)
        check_install_torch(wheels, args.arch, args.python_tag, final_version, venv)
        check_install_jax(wheels, final_version, venv)
        check_dry_run_upload(dl_root)
    except (subprocess.CalledProcessError, AssertionError, RuntimeError) as e:
        ok = False
        detail = getattr(e, "stderr", None) or str(e)
        print(f"\n[ERROR] {type(e).__name__}: {detail}")

    _banner("SUMMARY: " + ("SUCCESS" if ok else "FAILURE"))
    if ok and not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    elif args.keep:
        print(f"  work dir retained at {workdir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
