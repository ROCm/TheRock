"""
This script runs the Linux build steps
"""

import logging
import os
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO)
THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent


def build_therock_dist():
    logging.info(f"Building therock-dist")
    cmd = "cmake --build build --target therock-dist"
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


def build_therock_archives():
    logging.info(f"Building therock-archives")
    cmd = "cmake --build build --target therock-archives"
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


def test_therock_packaging():
    github_repository = os.getenv("GITHUB_REPOSITORY", "ROCm/TheRock")
    _, repo_name = github_repository.split("/")
    if repo_name == "TheRock":
        logging.info(f"Running TheRock test packaging")
        cmd = "ctest --test-dir build --output-on-failure"
        subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


if __name__ == "__main__":
    build_therock_dist()
    build_therock_archives()
    test_therock_packaging()
