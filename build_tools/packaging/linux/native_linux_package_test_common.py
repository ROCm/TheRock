# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for native Linux package install and uninstall test scripts.

Used by ``native_linux_package_install_test.py`` and
``native_linux_package_uninstall_test.py`` so metapackage naming, OS detection,
and subprocess streaming behave identically across install and uninstall steps.

Constants:
    ENV_NATIVE_LINUX_INSTALL_ROCM_VERSION: Workflow env key mapped to
        ``--rocm-version`` on install/uninstall CLIs when versioned metapackage
        names are required.
    UNINSTALL_TIMEOUT_SEC: Maximum seconds for apt/dnf/zypper remove commands
        during uninstall (large stacks may install hundreds of packages).
    VERIFY_KEY_COMPONENTS: Relative paths under the install prefix checked
        after uninstall for informational prefix cleanup (warn-only; Step 4b
        pass/fail is based on the package-manager query only).
"""

import re
import subprocess
import sys

from packaging_utils import normalize_target_list

ENV_NATIVE_LINUX_INSTALL_ROCM_VERSION = "NATIVE_LINUX_INSTALL_ROCM_VERSION"

UNINSTALL_TIMEOUT_SEC = 600

VERIFY_KEY_COMPONENTS = [
    "bin/rocminfo",
    "bin/hipcc",
    "bin/clinfo",
    "include/hip/hip_runtime.h",
    "lib/libamdhip64.so",
    "lib/cmake/profiler-hub/profiler-hub-config.cmake",
]

# Directories the native install must create under the prefix. ``lib/cmake``
# is where downstream ``find_package`` looks. These are fail-closed.
INSTALL_LAYOUT_DIRS = (
    "bin",
    "lib",
    "include",
    "share",
    "lib/cmake",
)

# Metapackages installed and removed together, in this order.
METAPACKAGE_BASES = ("amdrocm", "amdrocm-core-sdk", "amdrocm-hpc-sdk")


def metapackage_variant_suffix(build_variant: str) -> str:
    """Return the metapackage name variant segment for a CI build-variant label.

    ASan-family labels (``asan``, ``host-asan``, ``asan-debug``, ``host-asan-debug``)
    all collapse to ``-asan``, matching ``packaging_utils.update_package_name``.
    Other labels (e.g. ``release``) return an empty string.

    Args:
        build_variant: Build variant from CLI or ``BUILD_VARIANT`` env.

    Returns:
        ``-asan`` when the variant is ASan-family; otherwise ``""``.
    """
    variant = build_variant.strip().lower()
    if "asan" in variant:
        return "-asan"
    return ""


def derive_package_type(os_profile: str) -> str:
    """Derive native package type from an OS profile string.

    Args:
        os_profile: OS profile (e.g. ``ubuntu2404``, ``rhel8``, ``sles16``).

    Returns:
        ``deb`` for Debian/Ubuntu profiles, ``rpm`` for RHEL/SLES/Alma/CentOS/AZL.

    Raises:
        ValueError: If ``os_profile`` does not match a supported prefix.
    """
    os_profile_lower = os_profile.lower()
    if os_profile_lower.startswith(("ubuntu", "debian")):
        return "deb"
    if os_profile_lower.startswith(("rhel", "sles", "almalinux", "centos", "azl")):
        return "rpm"
    raise ValueError(
        f"Unable to derive package type from OS profile: {os_profile}. "
        "Supported profiles: ubuntu*, debian*, rhel*, sles*, almalinux*, centos*, azl*"
    )


def major_minor_rocm_version_from_input(rocm_version: str | None) -> str | None:
    """Normalize a ROCm release string to major.minor for metapackage names.

    Accepts optional leading ``v`` and patch segments (``7.13.1`` → ``7.13``).

    Args:
        rocm_version: User or CI ROCm version, or ``None`` / empty for unversioned
            metapackages.

    Returns:
        ``major.minor`` string, or ``None`` when input is absent or blank.

    Raises:
        ValueError: When a non-empty value cannot be parsed as major.minor.
    """
    if rocm_version is None:
        return None
    s = str(rocm_version).strip()
    if not s:
        return None
    if s.lower().startswith("v"):
        s = s[1:].lstrip()
    match = re.match(r"^(\d+)\.(\d+)", s)
    if not match:
        raise ValueError(
            "Invalid ROCm version "
            f"{rocm_version!r}: expected major.minor (e.g. 7.13 or 7.13.1)."
        )
    return f"{int(match.group(1))}.{int(match.group(2))}"


def _metapackage_name(
    base: str, variant_sep: str, ver: str | None, arch: str | None
) -> str:
    """Return one metapackage name with the shared suffix rules.

    ``variant_sep`` is ``-asan`` or empty. ``ver`` and ``arch`` are left out of
    the name when they are ``None``, so this covers versioned, arch-specific,
    and unversioned targets. ``amdrocm-hpc-sdk`` uses the same order as
    ``amdrocm`` and ``amdrocm-core-sdk``: optional ``-asan``, then major.minor,
    then ``-arch``.

    Args:
        base: Metapackage stem, such as ``amdrocm-hpc-sdk``.
        variant_sep: ``-asan`` for an ASan-family build, otherwise ``""``.
        ver: Major.minor version, or ``None`` for an unversioned name.
        arch: GPU architecture, or ``None`` when the name has no arch suffix.

    Returns:
        The package name, for example ``amdrocm-hpc-sdk-asan7.13-gfx94x``.
    """
    name = f"{base}{variant_sep}"
    if ver:
        name = f"{name}{ver}"
    if arch:
        name = f"{name}-{arch}"
    return name


def build_metapackage_names(
    *,
    gfx_arch: str | list[str] | None,
    rocm_version: str | None,
    build_variant: str = "",
) -> list[str]:
    """Build the install or uninstall metapackage targets.

    The list is ``amdrocm``, ``amdrocm-core-sdk``, then ``amdrocm-hpc-sdk``.
    The HPC SDK is opt-in in ``package.json`` and is not pulled in by
    ``amdrocm``. Naming rules (must match between install and uninstall):

    - With ``gfx_arch`` and ``rocm_version``: one trio per arch
      (``amdrocm{variant}{ver}-{arch}``, and the same suffix on
      ``amdrocm-core-sdk`` and ``amdrocm-hpc-sdk``).
    - With ``rocm_version`` only: the three names with ``{variant}{ver}``.
    - With neither: unversioned ``amdrocm{variant}``,
      ``amdrocm-core-sdk{variant}``, and ``amdrocm-hpc-sdk{variant}``.
    - ASan-family build variants (``asan``, ``host-asan``, ``asan-debug``,
      ``host-asan-debug``) insert ``-asan`` before the version suffix
      (e.g. ``amdrocm-asan7.13-gfx94x``). Labels such as ``release`` do not
      alter names.

    Args:
        gfx_arch: GPU architecture(s); normalized via ``normalize_target_list``.
        rocm_version: ROCm release for versioned names; major.minor is used.
        build_variant: Optional CI build-variant label (ASan-family collapses to ``-asan``).

    Returns:
        Ordered list of metapackage names to install or remove.
    """
    gfx_arch_list = normalize_target_list(gfx_arch, lowercase=True, dedupe=True)
    rocm_version_major_minor = major_minor_rocm_version_from_input(rocm_version)
    variant_sep = metapackage_variant_suffix(build_variant)
    ver = rocm_version_major_minor

    if gfx_arch_list and ver:
        names: list[str] = []
        for arch in gfx_arch_list:
            names.extend(
                _metapackage_name(base, variant_sep, ver, arch)
                for base in METAPACKAGE_BASES
            )
        return names
    if ver:
        return [
            _metapackage_name(base, variant_sep, ver, None)
            for base in METAPACKAGE_BASES
        ]
    return [
        _metapackage_name(base, variant_sep, None, None) for base in METAPACKAGE_BASES
    ]


def is_sles(os_profile: str) -> bool:
    """Return True when ``os_profile`` denotes SUSE Linux Enterprise Server.

    Args:
        os_profile: OS profile string (case-insensitive).

    Returns:
        True for profiles starting with ``sles``.
    """
    return os_profile.lower().startswith("sles")


def is_rocm_related_package_name(name: str) -> bool:
    """Return True if a package name looks like a native Linux ROCm package.

    Used when scanning ``dpkg -l`` or ``rpm -qa`` output during uninstall
    verification (Step 4b).

    Args:
        name: Package name from the system package manager.

    Returns:
        True when ``rocm`` or ``amdrocm`` appears in the name (case-insensitive).
    """
    lower = name.lower()
    return "rocm" in lower or "amdrocm" in lower


def run_streaming(cmd: list[str], timeout_sec: int) -> int:
    """Run a subprocess with merged stdout/stderr streamed line-by-line.

    Args:
        cmd: Command argv (executable plus arguments).
        timeout_sec: Maximum seconds to wait for process completion.

    Returns:
        Process exit code on normal completion.

    Raises:
        subprocess.TimeoutExpired: When the process exceeds ``timeout_sec``; the
            child is killed before the exception is re-raised.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in process.stdout:
            print(line.rstrip())
            sys.stdout.flush()
        return process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.kill()
        raise
