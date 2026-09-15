#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Picks the published JAX nightly that a tip build of the ROCm plugin pairs with.

A build of jax-ml/jax@main with ML_WHEEL_TYPE=release would produce a plugin
versioned like the next release (0.11.2), which PyPI does not carry, so the
test job could not install a matching jax/jaxlib. The plugin is built as a
nightly instead: its base version must equal a jax/jaxlib nightly that the JAX
nightly index already carries, so the version string alone tells any consumer
what to install alongside it (jax-ml's own ROCm lane pairs plugins with a
published jaxlib nightly the same way).

This script reads the base version from the checkout's jax/version.py, lists
the jaxlib nightlies on the index for that base version, and picks the newest
build date that has a manylinux x86_64 wheel for the requested Python version.
Pinning to a date that is already published avoids racing jax-ml's daily
publish.

Example:

    python resolve_jax_nightly_version.py \\
        --jax-source-dir jax-source --python-version 3.12

writes these step outputs to GITHUB_OUTPUT:

    build_date=20260914
    jax_nightly_version=0.11.2.dev20260914
"""

import argparse
import datetime
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

from packaging.utils import InvalidWheelFilename, parse_wheel_filename

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.github_actions_api import gha_set_output

# The PEP 503 simple index that jax-ml publishes nightly wheels to. The test
# side installs jax/jaxlib from the same index.
JAX_NIGHTLY_INDEX_URL = (
    "https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/"
)

# The nightly package whose presence proves a build date is installable. jax
# itself is a pure Python wheel that is published alongside it.
JAXLIB_PROJECT = "jaxlib"

_JAX_VERSION_RE = re.compile(r'^_version\s*=\s*"(?P<version>[^"]+)"', re.MULTILINE)


class ResolveError(Exception):
    """Raised when no usable nightly can be resolved."""


@dataclass(frozen=True)
class NightlyWheel:
    """One jaxlib nightly wheel on the index, as far as this script cares."""

    base_version: str
    build_date: str
    python_tag: str
    platform: str


def read_jax_base_version(jax_source_dir: Path) -> str:
    """Returns the `_version` string from jax/version.py in a JAX checkout."""
    version_file = jax_source_dir / "jax" / "version.py"
    if not version_file.is_file():
        raise ResolveError(f"No jax/version.py under '{jax_source_dir}'")
    match = _JAX_VERSION_RE.search(version_file.read_text(encoding="utf-8"))
    if match is None:
        raise ResolveError(f"Could not find `_version = \"...\"` in '{version_file}'")
    return match.group("version")


class _AnchorHrefParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.hrefs.append(value)


def parse_index_wheel_filenames(index_html: str) -> list[str]:
    """Returns the wheel filenames linked from a PEP 503 simple index page.

    Each anchor's href is a URL whose last path segment is the filename, with
    an optional `#sha256=...` fragment.
    """
    parser = _AnchorHrefParser()
    parser.feed(index_html)
    filenames = []
    for href in parser.hrefs:
        filename = unquote(urlsplit(href).path.rsplit("/", 1)[-1])
        if filename.endswith(".whl"):
            filenames.append(filename)
    return filenames


def _build_date_from_dev_number(dev: int | None) -> str | None:
    """Returns the YYYYMMDD string a `.devN` number encodes, or None."""
    if dev is None:
        return None
    build_date = str(dev)
    if len(build_date) != 8:
        return None
    try:
        datetime.datetime.strptime(build_date, "%Y%m%d")
    except ValueError:
        return None
    return build_date


def parse_nightly_wheels(wheel_filenames: list[str]) -> list[NightlyWheel]:
    """Keeps the jaxlib wheels whose version is `<base>.devYYYYMMDD`.

    Release wheels, pre-releases, post-releases, local versions, and wheels of
    other projects are dropped.
    """
    nightlies = []
    for filename in wheel_filenames:
        try:
            name, version, _build, tags = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            continue
        if name != JAXLIB_PROJECT:
            continue
        if version.pre is not None or version.post is not None or version.local:
            continue
        build_date = _build_date_from_dev_number(version.dev)
        if build_date is None:
            continue
        for tag in tags:
            nightlies.append(
                NightlyWheel(
                    base_version=version.base_version,
                    build_date=build_date,
                    python_tag=f"{tag.interpreter}-{tag.abi}",
                    platform=tag.platform,
                )
            )
    return nightlies


def python_tag_for_version(python_version: str) -> str:
    """Maps "3.12" to the "cp312-cp312" interpreter-abi pair of a regular build."""
    major, minor = python_version.strip().split(".", 1)
    interpreter = f"cp{int(major)}{int(minor)}"
    return f"{interpreter}-{interpreter}"


def is_manylinux_x86_64(platform: str) -> bool:
    return platform.startswith("manylinux") and platform.endswith("x86_64")


def resolve_nightly_build_date(
    nightlies: list[NightlyWheel],
    *,
    base_version: str,
    python_version: str,
) -> str:
    """Returns the newest build date with a usable jaxlib wheel, as YYYYMMDD."""
    python_tag = python_tag_for_version(python_version)
    build_dates = {
        wheel.build_date
        for wheel in nightlies
        if wheel.base_version == base_version
        and wheel.python_tag == python_tag
        and is_manylinux_x86_64(wheel.platform)
    }
    if not build_dates:
        available = sorted({wheel.base_version for wheel in nightlies})
        raise ResolveError(
            f"No {JAXLIB_PROJECT} {base_version}.devYYYYMMDD manylinux x86_64 "
            f"wheel for Python {python_version} on the nightly index "
            f"(base versions with nightlies: {available})"
        )
    return max(build_dates)


def fetch_index_page(index_url: str, project: str, timeout_seconds: int) -> str:
    """Downloads the simple index page for one project."""
    url = f"{index_url.rstrip('/')}/{project}/"
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8")
    except HTTPError as e:
        raise ResolveError(f"HTTP {e.code} fetching {url}: {e.reason}") from e
    except URLError as e:
        raise ResolveError(f"Network error fetching {url}: {e.reason}") from e
    except TimeoutError as e:
        raise ResolveError(f"Timed out after {timeout_seconds}s fetching {url}") from e


def resolve_jax_nightly_version(
    *,
    jax_source_dir: Path,
    python_version: str,
    index_url: str,
    timeout_seconds: int,
) -> tuple[str, str]:
    """Returns (build_date, nightly_version) for the checkout and Python version."""
    base_version = read_jax_base_version(jax_source_dir)
    print(f"JAX base version in '{jax_source_dir}': {base_version}")
    index_html = fetch_index_page(index_url, JAXLIB_PROJECT, timeout_seconds)
    nightlies = parse_nightly_wheels(parse_index_wheel_filenames(index_html))
    print(f"Found {len(nightlies)} {JAXLIB_PROJECT} nightly wheel tags on {index_url}")
    build_date = resolve_nightly_build_date(
        nightlies, base_version=base_version, python_version=python_version
    )
    return build_date, f"{base_version}.dev{build_date}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pick the published JAX nightly a tip plugin build pairs with"
    )
    parser.add_argument(
        "--jax-source-dir",
        type=Path,
        required=True,
        help="JAX checkout holding jax/version.py",
    )
    parser.add_argument(
        "--python-version",
        type=str,
        required=True,
        help="Python version the plugin is built for, e.g. 3.12",
    )
    parser.add_argument(
        "--index-url",
        type=str,
        default=JAX_NIGHTLY_INDEX_URL,
        help=f"PEP 503 simple index of JAX nightlies (default: {JAX_NIGHTLY_INDEX_URL})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Timeout for fetching the index page (default: 60)",
    )
    args = parser.parse_args(argv)

    try:
        build_date, nightly_version = resolve_jax_nightly_version(
            jax_source_dir=args.jax_source_dir,
            python_version=args.python_version,
            index_url=args.index_url,
            timeout_seconds=args.timeout_seconds,
        )
    except ResolveError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Resolved JAX nightly {nightly_version} (build date {build_date})")
    gha_set_output({"build_date": build_date, "jax_nightly_version": nightly_version})
    return 0


if __name__ == "__main__":
    sys.exit(main())
