#!/usr/bin/env python3
"""
Minimal ROCm tarball installer using requests with simple logging (message only)

Usage:
    python3 install_rocm_tar.py <tar_url> <rocm_version>

Example:
    python3 install_rocm_tar.py \
      "https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx94X-dcgpu-7.10.0a20251109.tar.gz" \
      "7.10.0a20251109"
"""

import sys
import os
import logging
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import requests

# Log message
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run(cmd, cwd=None):
    logger.info("+ %s", cmd)
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def download_tarball(url, dest_dir) -> Path:
    logger.info("Downloading: %s", url)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(urlparse(url).path) or "rocm.tar.gz"
    outfile = dest_dir / filename
    tmpfile = outfile.with_suffix(outfile.suffix + ".part")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(tmpfile, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmpfile.rename(outfile)
    logger.info("Downloaded to %s", outfile)
    return outfile


def main():
    # Expect exactly two args: tar_url and rocm_version
    tar_url = sys.argv[1]
    rocm_version = sys.argv[2]

    workdir = Path.cwd() / "therock-tarball"
    install_dir = workdir / "install"
    workdir.mkdir(exist_ok=True)
    install_dir.mkdir(exist_ok=True)
    logger.info("Working in %s", workdir)

    tar_path = download_tarball(tar_url, workdir)

    # Extract tarball
    run(f'tar -xf "{tar_path.name}" -C install', cwd=str(workdir))

    # Install to /opt/rocm-<rocm_version> and create symlinks
    dest = Path(f"/opt/rocm-{rocm_version}")
    run(f'sudo mkdir -p "{dest}"')
    run(f'sudo mv "{install_dir}"/* "{dest}"')
    run(f'sudo ln -sfn "{dest}" /opt/rocm')
    run(f'sudo ln -sfn /opt/rocm /etc/alternatives/rocm')

    logger.info("ROCm installation configured at /opt/rocm-%s", rocm_version)

if __name__ == "__main__":
    main()
