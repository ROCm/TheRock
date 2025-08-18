"""Given ROCm artifacts directories, performs the download and extraction of artifacts

```
./fetch_artifacts.py --artifact-dir ./output-linux-portable/build/artifacts \
        --gfx-arch gfx94X
```
"""

import urllib.request
import tarfile
import shutil

# from pathlib import Path
# import sys
import os

from packaging_utils import *

# Directory for downloading tar artifacts
ARTIFACTS_DOWNLOAD_DIR = f"{os.getcwd()}/artifacts_tar"
# Directory for extracting tar artifacts
ARTIFACTS_EXTRACT_DIR = f"{os.getcwd()}/artifacts"

# TBD - Will get it as user input? full path or just build id?
# ARTIFACT_URL = "https://therock-artifacts.s3.amazonaws.com"
# BUILD_ID = "16418185899"
# BUILD_ID_SUFFIX= "-linux"

############### Download artifacts #####################
# Function will find the artifacts corresponding to the package
# Download the artifacts
# Extract the artifacts
def download_and_extract_artifacts(artifact_uri: str, pkg_name, gfx_arch):
    """Function will find the artifacts corresponding to the package
    Download the artifacts
    Extract the artifacts"""

    os.makedirs(ARTIFACTS_DOWNLOAD_DIR, exist_ok=True)
    pkg_info = get_package_info(pkg_name)
    artifact_prefix = pkg_info["Artifact"]
    if str(pkg_info.get("Gfxarch", "false")).strip().lower() == "true":
        artifact_suffix = gfx_arch + "-dcgpu.tar.xz"
    else:
        artifact_suffix = "generic.tar.xz"

    # Iterate through each artifact and download
    for component in pkg_info["Components"]:
        artifact_name = f"{artifact_prefix}_{component}_{artifact_suffix}"
        print(artifact_name)
        destination_path = f"{ARTIFACTS_DOWNLOAD_DIR}/{artifact_name}"
        download_uri = f"{artifact_uri}/{artifact_name}"
        print(f"Downloading {download_uri} to {destination_path}...")

        if os.path.exists(destination_path):
            print(f"{destination_path} already exists. Skipping download.")
        else:
            try:
                urllib.request.urlretrieve(download_uri, destination_path)
                print("Download complete.")
            except Exception as e:
                print(f"Download failed: {e}")
                return

        # Extract the downloaded artifact
        extract_directory = f"{ARTIFACTS_EXTRACT_DIR}/{artifact_prefix}_{component}"
        if os.path.exists(extract_directory):
            print("Already extracted, skipping.")
            continue
        print(f"Extracting {destination_path} to {extract_directory}...")
        if tarfile.is_tarfile(destination_path):
            print("Valid tar file.")
        else:
            print("Invalid tar file.")
            continue

        try:
            with tarfile.open(destination_path, "r:xz") as tar:
                tar.extractall(path=extract_directory)
                print("Extraction complete.")
        except tarfile.ReadError as e:
            print(f"Extraction failed: {e}. The file might not be a valid tar.xz file.")
        except Exception as e:
            print(f"An error occurred during extraction: {e}")


def clean_artifacts_download_dir():
    """Clean download directory"""
    if os.path.exists(ARTIFACTS_DOWNLOAD_DIR) and os.path.isdir(ARTIFACTS_DOWNLOAD_DIR):
        shutil.rmtree(ARTIFACTS_DOWNLOAD_DIR)
        print(f"Removed directory: {ARTIFACTS_DOWNLOAD_DIR}")


def clean_artifacts_extract_dir():
    """Clean artifacts extraction directory"""
    if os.path.exists(ARTIFACTS_EXTRACT_DIR) and os.path.isdir(ARTIFACTS_EXTRACT_DIR):
        shutil.rmtree(ARTIFACTS_EXTRACT_DIR)
        print(f"Removed directory: {ARTIFACTS_EXTRACT_DIR}")
