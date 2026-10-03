#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Download a build run's Windows MSI packages into a local directory.

Separates the download (network) step from the install/uninstall tests, which
consume the already-downloaded files via ``--msi-dir``. CI runs this once and
then runs the tests against the output directory; a developer can do the same
and re-run the tests repeatedly without re-downloading.

The MSIs live at the public artifacts prefix ``{run_id}-windows/packages/msi/``
and are named ``<output_stem>.msi`` per the generator's PACKAGES table, so no S3
listing or credentials are needed (works from forks).

Example::

    python download_msi_packages.py \\
        --artifact-run-id 36648157930 --release-type nightly \\
        --packages runtime --dest-dir ./msis
"""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_msi_wxs import PACKAGES, expected_msi_filename  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _therock_utils.workflow_outputs import WorkflowOutputRoot  # noqa: E402

PLATFORM = "windows"
DEFAULT_ARTIFACT_GITHUB_REPO = "ROCm/TheRock"


def resolve_msi_prefix_url(
    artifact_run_id: str, artifact_github_repo: str, release_type: str
) -> str:
    """Return the public HTTPS URL of the run's msi package prefix.

    Resolves the artifacts bucket from the run id + release type without any
    GitHub API call (lookup_workflow_run=False).
    """
    root = WorkflowOutputRoot.from_workflow_run(
        run_id=artifact_run_id,
        platform=PLATFORM,
        github_repository=artifact_github_repo,
        release_type=release_type,
        lookup_workflow_run=False,
    )
    return root.native_windows_packages("msi").https_url


def download_msis(prefix_url: str, packages: list[str], dest_dir: Path) -> list[Path]:
    """Download each package's MSI from the public prefix into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for package in packages:
        filename = expected_msi_filename(package)
        url = f"{prefix_url.rstrip('/')}/{filename}"
        dest = dest_dir / filename
        print(f"Downloading {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.URLError as e:
            raise FileNotFoundError(
                f"could not download MSI for package {package!r} from {url}: {e}. "
                "The build run may not have produced this package, or its MSIs "
                "were not uploaded to the public prefix (fork build runs do not "
                "upload)."
            ) from e
        downloaded.append(dest)
    return downloaded


def _split_packages(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download a build run's Windows MSI packages to a directory."
    )
    parser.add_argument(
        "--artifact-run-id",
        required=True,
        help="Run id whose published Windows MSIs to download.",
    )
    parser.add_argument(
        "--artifact-github-repo",
        default=DEFAULT_ARTIFACT_GITHUB_REPO,
        help=f"Repository that owns the run id. Default: {DEFAULT_ARTIFACT_GITHUB_REPO}.",
    )
    parser.add_argument(
        "--release-type",
        required=True,
        help="Release type selecting the source artifacts bucket (e.g. ci, nightly).",
    )
    parser.add_argument(
        "--packages",
        default="runtime",
        help="Comma-separated packages to download (e.g. 'runtime,core').",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        required=True,
        help="Directory to download the MSIs into.",
    )
    args = parser.parse_args(argv)

    packages = _split_packages(args.packages)
    if not packages:
        parser.error("--packages was empty")
    unknown = [p for p in packages if p not in PACKAGES]
    if unknown:
        parser.error(f"unknown package(s): {', '.join(unknown)}")

    prefix_url = resolve_msi_prefix_url(
        args.artifact_run_id, args.artifact_github_repo, args.release_type
    )
    paths = download_msis(prefix_url, packages, args.dest_dir)
    print(f"Downloaded {len(paths)} MSI(s) to {args.dest_dir}")


if __name__ == "__main__":
    main()
