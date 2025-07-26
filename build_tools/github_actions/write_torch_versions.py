"""Writes torch_version, torchaudio_version, torchvision_version, and triton_version to GITHUB_OUTPUT.

Fails if any wheels that were expected for the platform were not set.

Currently,
* Windows expects torch and torchaudio
* Linux expects torch, torchaudio, torchvision, and triton
"""

import argparse
import os
import glob
import platform

from github_actions_utils import *


def _log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def parse_version_from_wheel(wheel_path: Path) -> str:
    return wheel_path.split("-")[1]


def get_wheel_version(package_dist_dir: Path, wheel_name: str) -> str | None:
    _log(f"Looking for '{wheel_name} in '{package_dist_dir}'")
    wheel_glob_pattern = f"{wheel_name}-*.whl"
    wheel_paths = glob.glob(wheel_glob_pattern, root_dir=package_dist_dir)

    if len(wheel_paths) == 0:
        _log(
            f"  WARNING: Found no '{wheel_name}' wheels matching '{wheel_glob_pattern}'"
        )
        return None
    elif len(wheel_paths) != 1:
        _log(
            f"  WARNING: Found multiple '{wheel_name}' wheels matching '{wheel_glob_pattern}', using the first from {wheel_paths}"
        )
    wheel_path = wheel_paths[0]
    _log(f"  Found wheel at '{wheel_path}'")
    wheel_version = parse_version_from_wheel(wheel_path)
    _log(f"  Parsed version '{wheel_version}'")
    return wheel_version


def get_all_wheel_versions(
    package_dist_dir: Path, os: str = platform.system()
) -> Mapping[str, str | Path]:
    _log(f"Looking for wheels in '{package_dist_dir}'")
    all_files = list(package_dist_dir.glob("*"))
    _log("Found files in that directory:")
    for file in all_files:
        _log(f"  {file}")

    _log("")
    all_versions = {}
    torch_version = get_wheel_version(package_dist_dir, "torch")
    torchaudio_version = get_wheel_version(package_dist_dir, "torchaudio")
    torchvision_version = get_wheel_version(package_dist_dir, "torchvision")
    triton_version = get_wheel_version(package_dist_dir, "pytorch_triton_rocm")
    _log("")

    if torch_version:
        all_versions = all_versions | {"torch_version": torch_version}
    else:
        raise FileNotFoundError("Did not find torch wheel")

    if torchaudio_version:
        all_versions = all_versions | {"torchaudio_version": torchaudio_version}
    else:
        raise FileNotFoundError("Did not find torchaudio wheel")

    if torchvision_version:
        all_versions = all_versions | {"torchvision_version": torchvision_version}
    elif os.lower() == "windows":
        _log(
            "Did not find torchvision (that's okay, is not currently built on Windows)"
        )
    else:
        raise FileNotFoundError("Did not find torchvision wheel")

    if triton_version:
        all_versions = all_versions | {"triton_version": triton_version}
    elif os.lower() == "windows":
        _log("Did not find triton (that's okay, is not currently built on Windows)")
    else:
        raise FileNotFoundError("Did not find triton wheel")

    return all_versions


def main(argv: list[str]):
    p = argparse.ArgumentParser(prog="write_torch_versions.py")
    p.add_argument(
        "--dist-dir",
        type=Path,
        default=Path(os.getenv("PACKAGE_DIST_DIR")),
        help="Path where wheels are located",
    )
    args = p.parse_args(argv)

    if not args.dist_dir.exists():
        raise FileNotFoundError(f"Dist dir '{args.dist_dir}' does not exist")

    all_versions = get_all_wheel_versions(args.dist_dir)
    _log("")
    gha_set_output(all_versions)


if __name__ == "__main__":
    main(sys.argv[1:])
