#!/usr/bin/env python3
"""
Download .whl files from a CloudFront URL amdgpu_family into a specified directory.

USAGE:
    python download_from_cloudfront.py \
        --cloudfront_url ${PACKAGE_INDEX_URL} \
        --amdgpu_family ${amdgpu_family} \
        --dir ${WHEELHOUSE_DIR} \
        --list_whls '["jax_pjrt.whl","jax_plugin.whl","jaxlib.whl"]'

EXAMPLE:
    python download_from_cloudfront.py \
        --cloudfront_url https://rocm.nightlies.amd.com/v2-staging  \
        --amdgpu_family gfx94X-dcgpu \
        --dir wheelhouse \
        --list_whls $jax_whl_list

This will:
    - Create wheelhouse if it does not exist.
    - Download:
        https://rocm.nightlies.amd.com/v2-staging/gfx94X-dcgpu/jax_pjrt.whl
        https://rocm.nightlies.amd.com/v2-staging/gfx94X-dcgpu/jax_plugin.whl
        https://rocm.nightlies.amd.com/v2-staging/gfx94X-dcgpu/jaxlib.whl
    - Save all files into wheelhouse
"""

import argparse
import ast
import logging
from pathlib import Path
from urllib.parse import quote, urljoin
import shutil
import requests
from github_actions.github_actions_utils import gha_append_step_summary

LOG = logging.getLogger(__name__)


def download_one(session, url: str, dest: Path, bufsize: int = 64 * 1024, timeout: int = 30):

    # Stream-download url to dest using a temporary .part file.
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tmp.open("wb") as fh:
            shutil.copyfileobj(r.raw, fh, length=bufsize)

    tmp.replace(dest)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser("Simple wheel downloader")
    p.add_argument("--cloudfront_url", required=True)
    p.add_argument("--amdgpu_family", required=True)
    p.add_argument("--dir", required=True, dest="wheelhouse_dir")
    p.add_argument(
        "--list_whls",
        required=True,
        help='Python list literal of filenames, e.g. \'["jax_pjrt.whl","jax_plugin.whl","jaxlib.whl"]\'',
    )
    args = p.parse_args()

    try:
        whl_list = ast.literal_eval(args.list_whls)
        if not isinstance(whl_list, list):
            raise ValueError
    except Exception:
        LOG.error("Invalid --list_whls; provide a Python list literal like '[\"jax_pjrt.whl\",\"jax_plugin.whl\",\"jaxlib.whl\"]'")
        raise SystemExit(1)

    base = args.cloudfront_url.rstrip("/") + "/"
    family = args.amdgpu_family.strip("/")
    out_dir = Path(args.wheelhouse_dir)

    session = requests.Session()

    failures = []
    for name in whl_list:
        encoded = quote(name, safe="")
        url = urljoin(base, f"{family}/{encoded}")
        dest = out_dir / name

        LOG.info("Downloading %s -> %s", url, dest)
        try:
            download_one(session, url, dest)
            msg = f"Downloaded {name} -> {out_dir}"
            LOG.info(msg)
            gha_append_step_summary(msg)
        except Exception as e:
            msg = f"Failed {name}: {e}"
            LOG.error(msg)
            gha_append_step_summary(msg)
            failures.append(name)

    if failures:
        LOG.error("Failed downloads: %s", ", ".join(failures))
        raise SystemExit(1)

    LOG.info("All downloads complete.")


if __name__ == "__main__":
    main()
