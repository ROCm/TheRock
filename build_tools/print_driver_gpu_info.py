#!/usr/bin/env python3
"""
Sanity check script for CI runners.

On Linux:
  - print Linux kernel version
  - print linux-firmware package version
  - print amdgpu driver source (in-kernel vs DKMS/external)
  - run "amd-smi static"
  - run "amd-smi firmware" (shows GPU firmware versions)
  - run "rocminfo"

On Windows:
  - run "hipInfo.exe"

This script prints only raw command output.
"""

import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import List, Optional

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
# TODO(#2964): Remove gfx950-dcgpu once amdsmi static does not timeout
unsupported_amdsmi_families = ["gfx1151", "gfx950-dcgpu"]


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# Common system paths where commands like modinfo might be located
SYSTEM_BIN_PATHS = [
    Path("/sbin"),
    Path("/usr/sbin"),
    Path("/bin"),
    Path("/usr/bin"),
]


def find_command(command: str) -> Optional[str]:
    """Find a command in PATH or common system directories."""
    # First try PATH
    resolved = shutil.which(command)
    if resolved:
        return resolved

    # Then try common system paths (not always in PATH for non-root users)
    for path in SYSTEM_BIN_PATHS:
        candidate = path / command
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    return None


def run_command(args: List[str | Path], cwd: Optional[Path] = None) -> None:
    args = [str(arg) for arg in args]
    if cwd is None:
        cwd = Path.cwd()

    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")

    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        log(proc.stdout.rstrip())
    except FileNotFoundError:
        log(f"{args[0]}: command not found")


def run_command_with_search(
    label: str,
    command: str,
    args: List[str],
    extra_command_search_paths: List[Path],
) -> None:
    """
    Run a command, searching in extra paths first, then PATH.

    Example:
        run_command_with_search(
            label="amd-smi static",
            command="amd-smi",
            args=["static"],
            extra_command_search_paths=[bin_dir],
        )
    """
    # Try explicit directories first (e.g. THEROCK_DIR/build/bin)
    for base in extra_command_search_paths:
        candidate = base / command
        if candidate.exists():
            log(f"\n=== {label} ===")
            run_command([candidate] + args)
            return

    # Then fall back to PATH
    resolved = shutil.which(command)
    if resolved:
        log(f"\n=== {label} ===")
        run_command([resolved] + args)
        return

    # Nothing found
    log(f"\n=== {label} ===")
    log(f"{command}: command not found")


def is_running_in_container() -> bool:
    """Detect if we're running inside a container (Docker, podman, etc.)."""
    # Check for /.dockerenv file
    if Path("/.dockerenv").exists():
        return True

    # Check cgroup for container indicators
    try:
        cgroup = Path("/proc/1/cgroup").read_text()
        if "docker" in cgroup or "containerd" in cgroup or "podman" in cgroup:
            return True
        # In cgroup v2, check if we're not in the root cgroup
        if cgroup.strip() == "0::/":
            # Could be cgroup v2, check for container env vars
            pass
    except Exception:
        pass

    # Check for container environment variables
    if os.getenv("container") or os.getenv("KUBERNETES_SERVICE_HOST"):
        return True

    return False


def print_kernel_version() -> None:
    """Print the Linux kernel version."""
    log("\n=== Linux Kernel Version ===")
    log(f"Kernel: {platform.release()}")
    log(f"Full uname: {platform.platform()}")


def print_linux_firmware_version() -> None:
    """Print the version of the installed linux-firmware package."""
    log("\n=== Linux Firmware Package ===")

    found_package_manager = False

    # Try dpkg (Debian/Ubuntu)
    dpkg = find_command("dpkg-query")
    if dpkg:
        found_package_manager = True
        try:
            proc = subprocess.run(
                [dpkg, "-W", "-f", "${Version}", "linux-firmware"],
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                log(f"linux-firmware (dpkg): {proc.stdout.strip()}")
                return
            else:
                log(f"linux-firmware: not installed (dpkg-query at {dpkg})")
                if proc.stderr.strip():
                    log(f"  {proc.stderr.strip()}")
        except Exception as e:
            log(f"dpkg-query error: {e}")

    # Try rpm (Fedora/RHEL)
    rpm = find_command("rpm")
    if rpm:
        found_package_manager = True
        try:
            proc = subprocess.run(
                [rpm, "-q", "linux-firmware", "--queryformat", "%{VERSION}-%{RELEASE}"],
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                log(f"linux-firmware (rpm): {proc.stdout.strip()}")
                return
            else:
                log(f"linux-firmware: not installed (rpm at {rpm})")
                if proc.stderr.strip():
                    log(f"  {proc.stderr.strip()}")
        except Exception as e:
            log(f"rpm error: {e}")

    # Fallback: check if /lib/firmware exists and show amdgpu firmware info
    firmware_dir = Path("/lib/firmware/amdgpu")
    in_container = is_running_in_container()

    if firmware_dir.exists():
        try:
            fw_files = list(firmware_dir.glob("*.bin"))
            log(f"Firmware directory exists: {firmware_dir}")
            log(f"  Number of amdgpu firmware files: {len(fw_files)}")
            # Try to get modification time of a firmware file as a proxy for version
            if fw_files:
                newest = max(fw_files, key=lambda p: p.stat().st_mtime)
                import datetime

                mtime = datetime.datetime.fromtimestamp(newest.stat().st_mtime)
                log(f"  Newest firmware file: {newest.name} ({mtime.isoformat()})")
        except Exception as e:
            log(f"Error reading firmware directory: {e}")
    else:
        if in_container:
            log("Note: Running in container, firmware is loaded by host kernel")
        elif not found_package_manager:
            log("No supported package manager found (dpkg-query, rpm)")
            log("  /lib/firmware/amdgpu does not exist")


def print_amdgpu_driver_source() -> None:
    """Check whether the amdgpu driver is from the kernel or external (DKMS)."""
    log("\n=== AMDGPU Driver Source ===")

    modinfo = find_command("modinfo")
    if modinfo:
        log(f"Using modinfo: {modinfo}")
        try:
            proc = subprocess.run(
                [modinfo, "amdgpu"],
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if proc.returncode == 0:
                # Parse modinfo output
                info = {}
                for line in proc.stdout.splitlines():
                    if ":" in line:
                        key, _, value = line.partition(":")
                        # Only keep first occurrence of each key
                        if key.strip() not in info:
                            info[key.strip()] = value.strip()

                filename = info.get("filename", "")
                version = info.get("version", "unknown")

                # Determine source based on module path
                if "updates/dkms" in filename or "/dkms/" in filename:
                    source = "DKMS (external)"
                elif "extra/" in filename or "/extra/" in filename:
                    source = "External (extra modules)"
                elif "kernel/drivers" in filename:
                    source = "In-kernel"
                elif "(builtin)" in filename.lower() or not filename:
                    source = "Built-in (compiled into kernel)"
                else:
                    source = "Unknown"

                log(f"Driver version: {version}")
                log(f"Driver source: {source}")
                log(f"Module path: {filename}")
                return
            else:
                log(f"modinfo amdgpu failed (returncode: {proc.returncode})")
                if proc.stderr.strip():
                    log(f"  {proc.stderr.strip()}")
        except Exception as e:
            log(f"modinfo error: {e}")
    else:
        log("modinfo: not found (kmod package not installed)")
        log(f"  Searched: PATH, {', '.join(str(p) for p in SYSTEM_BIN_PATHS)}")

    # Fallback: check sysfs for amdgpu module info
    sysfs_amdgpu = Path("/sys/module/amdgpu")
    if sysfs_amdgpu.exists():
        log("Fallback: amdgpu module detected via sysfs")
        # Check for version in sysfs
        version_file = sysfs_amdgpu / "version"
        if version_file.exists():
            try:
                version = version_file.read_text().strip()
                log(f"  Driver version (sysfs): {version}")
            except Exception:
                pass

        # Try to determine source from /proc/modules
        try:
            proc_modules = Path("/proc/modules").read_text()
            for line in proc_modules.splitlines():
                if line.startswith("amdgpu "):
                    log(f"  /proc/modules entry: {line}")
                    break
        except Exception:
            pass

        # Check kernel release to help identify module path
        kernel_release = platform.release()
        possible_paths = [
            f"/lib/modules/{kernel_release}/updates/dkms/amdgpu.ko*",
            f"/lib/modules/{kernel_release}/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko*",
            f"/lib/modules/{kernel_release}/extra/amdgpu.ko*",
        ]
        for pattern in possible_paths:
            matches = list(Path("/").glob(pattern.lstrip("/")))
            if matches:
                path = matches[0]
                if "dkms" in str(path):
                    source = "DKMS (external)"
                elif "extra" in str(path):
                    source = "External (extra modules)"
                else:
                    source = "In-kernel"
                log(f"  Driver source: {source}")
                log(f"  Module path: {path}")
                return
        log("  Could not determine driver source from filesystem")
    else:
        log("amdgpu: module not loaded (/sys/module/amdgpu not found)")


def run_sanity(os_name: str) -> None:
    THIS_SCRIPT_DIR = Path(__file__).resolve().parent
    THEROCK_DIR = THIS_SCRIPT_DIR.parent
    bin_dir = Path(os.getenv("THEROCK_BIN_DIR", THEROCK_DIR / "build" / "bin"))

    log("=== Sanity check: driver / GPU info ===")

    if os_name.lower() == "windows":
        # Windows: only hipInfo.exe
        run_command_with_search(
            label="hipInfo.exe",
            command="hipInfo.exe",
            args=[],
            extra_command_search_paths=[bin_dir],
        )
    else:
        # Linux: kernel version, firmware pkg, driver source, amd-smi, rocminfo
        print_kernel_version()
        print_linux_firmware_version()
        print_amdgpu_driver_source()

        # TODO(#2789): Remove conditional once amdsmi supports gfx1151
        if AMDGPU_FAMILIES not in unsupported_amdsmi_families:
            run_command_with_search(
                label="amd-smi static",
                command="amd-smi",
                args=["static"],
                extra_command_search_paths=[bin_dir],
            )
            run_command_with_search(
                label="amd-smi firmware",
                command="amd-smi",
                args=["firmware"],
                extra_command_search_paths=[bin_dir],
            )
        run_command_with_search(
            label="rocminfo",
            command="rocminfo",
            args=[],
            extra_command_search_paths=[bin_dir],
        )

    log("\n=== End of sanity check ===")


def main(argv: Optional[List[str]] = None) -> int:
    detected = platform.system()
    run_sanity(detected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
