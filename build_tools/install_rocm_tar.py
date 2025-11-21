#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

def run(cmd, cwd=None):
    print(f"+ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def main():
    therock_tar_url = os.environ.get("THEROCK_TAR_URL", "").strip()
    if not therock_tar_url:
        print("THEROCK_TAR_URL not provided")
        sys.exit(1)

    workdir = Path.cwd() / "therock-tarball"
    install_dir = workdir / "install"
    workdir.mkdir(exist_ok=True)
    install_dir.mkdir(exist_ok=True)
    print(f"Working in {workdir}")

    # Download tarball
    run(f'wget -q "{therock_tar_url}"', cwd=str(workdir))

    # Find tarball
    tars = list(workdir.glob("*.tar.gz"))
    if not tars:
        print("No .tar.gz downloaded")
        sys.exit(1)
    tarball = tars[0].name
    print(f"Found tarball: {tarball}")

    # Extract version from filename
    m = re.search(r'(\d+\.\d+\.\w+\d+)', tarball)
    if not m:
        print("Could not extract ROCm version from tarball name")
        sys.exit(1)
    version = m.group(1)
    print(f"Parsed ROCm version: {version}")

    # Extract tarball
    run(f'tar -xf "{tarball}" -C install', cwd=str(workdir))

    # Move into /opt/rocm-<version> and create symlinks
    dest = Path(f"/opt/rocm-{version}")
    run(f'sudo mkdir -p "{dest}"')
    run(f'sudo mv "{install_dir}"/* "{dest}"')
    run(f'sudo ln -sfn "{dest}" /opt/rocm')
    run(f'sudo ln -sfn /opt/rocm /etc/alternatives/rocm')

    print("ROCm installation configured at /opt/rocm with alternatives link")

if __name__ == "__main__":
    sys.exit(main())
