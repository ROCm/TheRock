#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Finds the newest nightly torch wheel on a package index for one GPU target.

Multi-arch torch nightlies are published as a `torch` wheel plus one
`amd-torch-device-<target>` wheel per GPU target, with the same version. The
index also carries release lines (for example 2.13.x), so "highest version"
is not enough: only versions on the nightly line (`X.Y.0a0+rocm...`) count,
and only when both wheels exist for the requested Python and platform. The
device wheel can be uploaded after `torch`, so the newest `torch` alone may not
be installable yet.

The `rocm...aYYYYMMDD` local version carries the nightly date. The script fails
when the newest match is older than --max-age-days, so a stalled nightly
pipeline does not keep re-testing the same wheel.

Example:

    python resolve_latest_pytorch_nightly.py \\
        --index-url https://nightly.repo.amd.com/rocm/whl-next/ \\
        --device-target gfx942 --python-version 3.12

writes this step output to GITHUB_OUTPUT:

    torch_version=2.16.0a0+rocm10.3.0a20261119
"""

import argparse
import datetime
import html.parser
import re
import sys
import urllib.request
from pathlib import Path

from packaging.tags import Tag
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import Version

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.github_actions_api import gha_set_output

_NIGHTLY_DATE_RE = re.compile(r"a(?P<date>\d{8})$")


class _LinkTextParser(html.parser.HTMLParser):
    """Collects the text of every <a> element on a PEP 503 project page."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []
        self._in_link = False

    def handle_starttag(self, tag: str, attrs) -> None:
        self._in_link = tag == "a"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._in_link = False

    def handle_data(self, data: str) -> None:
        if self._in_link and data.strip():
            self.names.append(data.strip())


def fetch_project_page(index_url: str, project: str) -> str:
    url = f"{index_url.rstrip('/')}/{project}/"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def wheel_versions(page_html: str, tag: Tag) -> set[Version]:
    """Returns the versions of wheels on a project page that support `tag`."""
    parser = _LinkTextParser()
    parser.feed(page_html)
    versions = set()
    for name in parser.names:
        if not name.endswith(".whl"):
            continue
        try:
            _, version, _, tags = parse_wheel_filename(name)
        except InvalidWheelFilename:
            continue
        if tag in tags:
            versions.add(version)
    return versions


def is_nightly(version: Version) -> bool:
    return version.pre == ("a", 0) and version.release[-1] == 0


def nightly_date(version: Version) -> datetime.date | None:
    m = _NIGHTLY_DATE_RE.search(version.local or "")
    if not m:
        return None
    return datetime.datetime.strptime(m.group("date"), "%Y%m%d").date()


def newest_nightly(
    torch_versions: set[Version],
    device_versions: set[Version],
    today: datetime.date,
    max_age_days: int,
) -> Version:
    """Returns the newest nightly version present for both torch and the device.

    Raises ValueError when there is none, or when it is older than
    `max_age_days`.
    """
    candidates = sorted(v for v in torch_versions & device_versions if is_nightly(v))
    if not candidates:
        raise ValueError("No nightly torch version has both wheels on the index")
    newest = candidates[-1]
    date = nightly_date(newest)
    if date is None:
        raise ValueError(f"Cannot read a nightly date from {newest}")
    age = (today - date).days
    if age > max_age_days:
        raise ValueError(
            f"Newest nightly {newest} is {age} days old (limit {max_age_days})"
        )
    return newest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Find the newest nightly torch wheel for one GPU target"
    )
    parser.add_argument(
        "--index-url",
        required=True,
        help="PEP 503 simple index holding torch and amd-torch-device-* wheels",
    )
    parser.add_argument(
        "--device-target",
        required=True,
        help="GPU target whose device wheel must exist (e.g. gfx942)",
    )
    parser.add_argument(
        "--python-version",
        default="3.12",
        help="CPython version the wheels must support (default: 3.12)",
    )
    parser.add_argument(
        "--platform-tag",
        default="linux_x86_64",
        help="Wheel platform tag (default: linux_x86_64)",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=3,
        help="Fail when the newest nightly is older than this (default: 3)",
    )
    args = parser.parse_args(argv)

    cp = "cp" + args.python_version.replace(".", "")
    tag = Tag(cp, cp, args.platform_tag)
    torch_versions = wheel_versions(fetch_project_page(args.index_url, "torch"), tag)
    device_versions = wheel_versions(
        fetch_project_page(args.index_url, f"amd-torch-device-{args.device_target}"),
        tag,
    )
    try:
        version = newest_nightly(
            torch_versions,
            device_versions,
            today=datetime.datetime.now(datetime.timezone.utc).date(),
            max_age_days=args.max_age_days,
        )
    except ValueError as e:
        sys.exit(f"error: {e}")
    print(f"Newest nightly for {args.device_target} {cp}: {version}")
    gha_set_output({"torch_version": str(version)})


if __name__ == "__main__":
    main()
