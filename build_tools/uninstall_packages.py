#!/usr/bin/env python3
import argparse
import sys
from packaging.linux.package_load import LoadPackages, logger
import subprocess
import os
import platform


def run_cmd(cmd):
    """Execute a shell command and return (stdout, stderr, rc)."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def detect_pkg_type():
    """Detect if system uses deb or rpm packages."""
    system = platform.system().lower()
    if system != "linux":
        logger.error("Unsupported OS: Only Linux systems are supported.")
        sys.exit(1)

    distro_id = ""
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("ID="):
                    distro_id = line.strip().split("=")[1].strip('"').lower()
                    break

    deb_like = {"debian", "ubuntu", "mint", "pop"}
    rpm_like = {"rhel", "centos", "fedora", "rocky", "almalinux", "sles", "suse", "opensuse-leap"}

    if distro_id in deb_like:
        return "deb"
    elif distro_id in rpm_like:
        return "rpm"

    # Fallback detection
    stdout, _, _ = run_cmd("which dpkg")
    if stdout:
        return "deb"
    stdout, _, _ = run_cmd("which rpm")
    if stdout:
        return "rpm"

    logger.error("Unable to detect package manager (deb or rpm).")
    sys.exit(1)


def uninstall_by_run_id(pkg_list,run_id):
    """Uninstall the given list of packages."""
    pkg_type = detect_pkg_type()
    logger.info(f"Detected package type: {pkg_type.upper()}")

    if not pkg_list:
        logger.info("No packages provided for uninstallation.")
        return

    logger.info(f"Preparing to uninstall {len(pkg_list)} package(s): {pkg_list}")

    for pkg in reversed(pkg_list):
        pkg = pkg.strip()
        if not pkg:
            continue

        if pkg_type == "deb":
            uninstall_cmd = f"sudo dpkg -r {pkg}"
        elif pkg_type == "rpm":
            uninstall_cmd = f"sudo rpm -e {pkg}"
        else:
            logger.error("Unsupported package manager type detected.")
            sys.exit(1)

        logger.info(f"Removing {pkg} ...")
        _, err, rc = run_cmd(uninstall_cmd)
        if rc == 0:
            logger.info(f"Successfully removed {pkg}")
        else:
            logger.error(f"Failed to remove {pkg}: {err}")


def main():
    parser = argparse.ArgumentParser(description="Uninstall ROCm native build packages by run_id.")
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="run_id to match installed package names (e.g., 16418185899).",
    )
    parser.add_argument("--package_json", required=True, help="Path to package.json.")
    parser.add_argument(
        "--version",
        choices=["true", "false"],
        default="false",
        help="If true, install only versioned packages.",
    )
    parser.add_argument(
        "--composite",
        choices=["true", "false"],
        default="false",
        help="Install composite packages only.",
    )
    parser.add_argument(
        "--amdgpu_family",
        type=str,
        required=False,
        help="Specify AMD GPU family (e.g., gfx94x).",
    )
    parser.add_argument(
        "--rocm-version",
        type=str,
        required=True,
        help="Specify ROCm version (e.g., 7.0.0).",
    )


    args = parser.parse_args()
    run_id = args.run_id

    version_flag = args.version.lower() == "true"
    composite_flag = args.composite.lower() == "true"
    amdgpu_family = args.amdgpu_family
    rocm_version = args.rocm_version

    pm = LoadPackages(args.package_json,version_flag, amdgpu_family,rocm_version)
    non_comp, comp = pm.list_composite_packages()


    # Select package list
    if composite_flag:
        logger.info(f"Count of Composite packages: {len(comp)}")
        sorted_packages = pm.sort_packages_by_dependencies(comp)
    else:
        logger.info(f"Count of non Composite packages: {len(non_comp)}")
        sorted_packages = pm.sort_packages_by_dependencies(non_comp)

    logger.info("=== Starting package uninstallation ===")

    try:
        pm.uninstall_packages(sorted_packages,composite_flag)
        logger.info("Uninstallation process completed.")
    except Exception as e:
        logger.error(f"Uninstallation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

