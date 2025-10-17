#!/usr/bin/env python3
import argparse
import os
import platform
import subprocess

def detect_os_family():
    """
    Detect the OS family (debian/redhat/suse/unknown) based on /etc/os-release
    """
    os_release = {}
    try:
        with open("/etc/os-release", "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    os_release[key] = value.strip('"')
    except FileNotFoundError:
        print("/etc/os-release not found. Falling back to platform detection.")
        system_name = platform.system().lower()
        if "linux" in system_name:
            return "linux"
        return "unknown"

    os_id = os_release.get("ID", "").lower()
    os_like = os_release.get("ID_LIKE", "").lower()

    if "ubuntu" in os_id or "debian" in os_id or "debian" in os_like:
        return "debian"
    elif "rhel" in os_id or "centos" in os_id or "fedora" in os_id or "redhat" in os_like:
        return "redhat"
    elif "suse" in os_id or "sles" in os_id:
        return "suse"
    else:
        return "unknown"


def install_packages(artifacts_dir):
    """
    Install packages from the specified artifacts directory depending on OS family
    """
    os_family = detect_os_family()
    print(f"Detected OS family: {os_family}")

    if not os.path.isdir(artifacts_dir):
        print(f"Artifacts directory not found: {artifacts_dir}")
        return

    debs = [f for f in os.listdir(artifacts_dir) if f.endswith(".deb")]
    rpms = [f for f in os.listdir(artifacts_dir) if f.endswith(".rpm")]

    if os_family == "debian" and debs:
        print("Installing Debian packages...")
        for pkg in debs:
            pkg_path = os.path.join(artifacts_dir, pkg)
            print(f"Running: sudo dpkg -i {pkg_path}")
            subprocess.run(["sudo", "dpkg", "-i", pkg_path], check=False)
        print("Running: sudo apt-get -f install -y")
        subprocess.run(["sudo", "apt-get", "-f", "install", "-y"], check=False)

    elif os_family == "redhat" and rpms:
        print("Installing RPM packages...")
        for pkg in rpms:
            pkg_path = os.path.join(artifacts_dir, pkg)
            print(f"➡️ Running: sudo rpm -ivh --replacepkgs {pkg_path}")
            subprocess.run(["sudo", "rpm", "-ivh", "--replacepkgs", pkg_path], check=False)

    elif os_family == "suse" and rpms:
        print("Installing SUSE RPM packages...")
        for pkg in rpms:
            pkg_path = os.path.join(artifacts_dir, pkg)
            print(f"➡️ Running: sudo zypper --non-interactive install --replacepkgs {pkg_path}")
            subprocess.run(["sudo", "zypper", "--non-interactive", "install", "--replacepkgs", pkg_path], check=False)

    else:
        print("No supported packages found or unsupported OS.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install ROCm native build packages.")
    parser.add_argument("--artifacts-dir", required=True, help="Directory where built packages are located.")
    parser.add_argument("--os", help="Optional manual override for OS type (debian or rhel)")
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    os_type = args.os.lower() if args.os else detect_os_family()

    print(f"Detected or provided OS type: {os_type}")
    print(f"Using artifacts from: {artifacts_dir}")

    if not os.path.isdir(artifacts_dir):
        print(f"Directory not found: {artifacts_dir}")
        sys.exit(1)

    install_packages(args.artifacts_dir)

