#!/usr/bin/env python

# Copyright Facebook, Inc. and its affiliates.
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Forked from https://github.com/pytorch/test-infra/blob/6105c6f94a6055fffdbbb7319f8fb10a45dae644/s3_management/manage.py

"""
This script supports both:
- Static prefix updates (default behavior)
- Explicit prefix override via CLI
- Optional auto-detection of prefixes from S3 (guarded by flag)

Auto-detection is not enabled by default to preserve backward compatibility
with existing CI workflows.
"""

import argparse
import base64
import concurrent.futures
import dataclasses
import functools
import time

from os import path, makedirs, getenv
from collections import defaultdict
from typing import Iterable, List, Type, Dict, Set, TypeVar, Optional
from re import sub, match
from packaging.version import parse as _parse_version, Version, InvalidVersion

import boto3
import botocore


S3 = boto3.resource('s3')
CLIENT = boto3.client('s3')

# Bucket configuration is resolved via initialize_bucket().
# Must be provided via --bucket (CLI) or S3_BUCKET_PY (env).

# Bucket globals configured via initialize_bucket()
BUCKET_NAME: str | None = None
BUCKET = None
INDEX_BUCKETS: Set = set()

def initialize_bucket(bucket_name: str | None) -> None:
    """
    initialize_bucket() is required for both CLI and Lambda usage.
    Do not move bucket initialization exclusively into main(), as the
    Lambda wrapper imports and calls update_pep503_index() directly.
    """
    # Resolve and configure the S3 bucket used for index updates.
    # CLI --bucket takes precedence over S3_BUCKET_PY; fails if neither is set.
    global BUCKET_NAME, BUCKET, INDEX_BUCKETS
    # Bucket resolution order:
    # 1. CLI --bucket (explicit override)
    # 2. S3_BUCKET_PY environment variable
    # Raises if neither is provided.
    BUCKET_NAME = bucket_name or getenv("S3_BUCKET_PY")
    if not BUCKET_NAME:
        raise RuntimeError("Bucket must be provided via --bucket or S3_BUCKET_PY")

    BUCKET = S3.Bucket(BUCKET_NAME)
    INDEX_BUCKETS = {BUCKET}

ACCEPTED_FILE_EXTENSIONS = ("whl", "zip", "tar.gz")
PREFIXES = [
    # Note: v2-staging first, in case issues are observed while the script runs
    # and the developer wants to more safely cancel the script.
    "v2-staging/gfx110X-all",
    "v2-staging/gfx1151",
    "v2-staging/gfx1152",
    "v2-staging/gfx1153",
    "v2-staging/gfx900",
    "v2-staging/gfx90a",
    "v2-staging/gfx908",
    "v2-staging/gfx906",
    "v2-staging/gfx120X-all",
    "v2-staging/gfx94X-dcgpu",
    "v2-staging/gfx950-dcgpu",
    "v2/gfx110X-all",
    "v2/gfx1151",
    "v2/gfx1152",
    "v2/gfx1153",
    "v2/gfx900",
    "v2/gfx90a",
    "v2/gfx908",
    "v2/gfx906",
    "v2/gfx120X-all",
    "v2/gfx94X-dcgpu",
    "v2/gfx950-dcgpu",
]

CUSTOM_PREFIX = getenv('CUSTOM_PREFIX')
if CUSTOM_PREFIX:
    PREFIXES.append(CUSTOM_PREFIX)

# NOTE: This refers to the name on the wheels themselves and not the name of
# package as specified by setuptools, for packages with "-" (hyphens) in their
# names you need to convert them to "_" (underscores) in order for them to be
# allowed here since the name of the wheels is compared here
PACKAGE_ALLOW_LIST = {x.lower() for x in [
    # ---- ROCm ----
    "rocm",
    "rocm_bootstrap",
    "rocm_sdk",
    "rocm_sdk_core",
    "rocm_sdk_devel",
    "rocm_profiler",
    "rocm_sdk_libraries",
    # ---- triton ----
    "triton",
    # ---- triton additional packages ----
    "Arpeggio",
    "caliper_reader",
    "contourpy",
    "cycler",
    "dill",
    "fonttools",
    "kiwisolver",
    "llnl-hatchet",
    "matplotlib",
    "pandas",
    "pydot",
    "pyparsing",
    "pytz",
    "textx",
    "tzdata",
    "importlib_metadata",
    "importlib_resources",
    "zipp",
    # ----
    "Pillow",
    "apex",
    "certifi",
    "charset_normalizer",
    "cmake",
    "colorama",
    "fbgemm_gpu",
    "fbgemm_gpu_genai",
    "filelock",
    "fsspec",
    "idna",
    "iopath",
    "intel_openmp",
    "Jinja2",
    "lit",
    "lightning_utilities",
    "MarkupSafe",
    "mpmath",
    "mkl",
    "mypy_extensions",
    "nestedtensor",
    "networkx",
    "numpy",
    "packaging",
    "portalocker",
    "pyre_extensions",
    "requests",
    "sympy",
    "tbb",
    "torch",
    "torcharrow",
    "torchaudio",
    "torchcodec",
    "torchcsprng",
    "torchdata",
    "torchdistx",
    "torchmetrics",
    "torchrec",
    "torchtext",
    "torchtune",
    "torchvision",
    "torchvision_extra_decoders",
    "triton",
    "tqdm",
    "typing_extensions",
    "typing_inspect",
    "urllib3",
    "xformers",
    "executorch",
    "setuptools",
    "setuptools_scm",
    "wheel",
    # ---- JAX ----
    "jax",
    "jaxlib",
    "jax_rocm7_plugin",
    "jax_rocm7_pjrt",
]}

S3IndexType = TypeVar('S3IndexType', bound='S3Index')


@dataclasses.dataclass(frozen=False)
@functools.total_ordering
class S3Object:
    key: str
    orig_key: str
    checksum: Optional[str]
    size: Optional[int]
    pep658: Optional[str]

    def __hash__(self):
        return hash(self.key)

    def __str__(self):
        return self.key

    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        return self.key < other.key


def safe_parse_version(ver_str: str) -> Version:
    try:
        return _parse_version(ver_str)
    except InvalidVersion:
        return Version("0.0.0")


class S3Index:
    def __init__(self: S3IndexType, objects: List[S3Object], prefix: str) -> None:
        self.objects = objects
        self.prefix = prefix.rstrip("/")
        self.html_name = "index.html"
        # should dynamically grab subdirectories like whl/test/cu101
        # so we don't need to add them manually anymore
        self.subdirs = {
            path.dirname(obj.key) for obj in objects if path.dirname != prefix
        }

    def nightly_packages_to_show(self: S3IndexType) -> List[S3Object]:
        """Finding packages to show based on a threshold we specify

        Basically takes our S3 packages, normalizes the version for easier
        comparisons, then iterates over normalized versions until we reach a
        threshold and then starts adding package to delete after that threshold
        has been reached

        After figuring out what versions we'd like to hide we iterate over
        our original object list again and pick out the full paths to the
        packages that are included in the list of versions to delete
        """
        # also includes versions without GPU specifier (i.e. cu102) for easier
        # sorting, sorts in reverse to put the most recent versions first
        all_sorted_packages = sorted(
            {self.normalize_package_version(obj) for obj in self.objects},
            key=lambda name_ver: safe_parse_version(name_ver.split('-', 1)[-1]),
            reverse=True,
        )
        packages: Dict[str, int] = defaultdict(int)
        to_hide: Set[str] = set()
        for obj in all_sorted_packages:
            full_package_name = path.basename(obj)
            package_name = full_package_name.split('-')[0]
            pkg = package_name.lower()
            BLACKLISTED_PACKAGES = {
                "rocm_sdk_libraries_dev",
                "rocm_sdk_libraries_doc",
                "rocm_sdk_libraries_run",
            }
            if pkg in BLACKLISTED_PACKAGES:
                print(f"[BLACKLISTED] {package_name}")
                to_hide.add(obj)
                continue
            # Allow legacy packages + explicitly handle dynamic multi-arch packages
            if (
                pkg in PACKAGE_ALLOW_LIST
                or pkg.startswith("rocm_sdk_device_")
                or pkg.startswith("rocm_sdk_libraries_")
                or pkg.startswith("amd_torch_device")
                or pkg.startswith("amd_torchvision_device")
            ):
                packages[package_name] += 1
            else:
                print(f"[FILTERED OUT] {package_name}")
                to_hide.add(obj)
                continue
        return list(set(self.objects).difference({
            obj for obj in self.objects
            if self.normalize_package_version(obj) in to_hide
        }))

    def is_obj_at_root(self, obj: S3Object) -> bool:
        return path.dirname(obj.key) == self.prefix

    def _resolve_subdir(self, subdir: Optional[str] = None) -> str:
        if not subdir:
            subdir = self.prefix
        # make sure we strip any trailing slashes
        return subdir.rstrip("/")

    def gen_file_list(
        self,
        subdir: Optional[str] = None,
        package_name: Optional[str] = None
    ) -> Iterable[S3Object]:
        objects = self.objects
        subdir = self._resolve_subdir(subdir) + '/'
        for obj in objects:
            if package_name is not None and self.obj_to_package_name(obj) != package_name:
                continue
            if self.is_obj_at_root(obj) or obj.key.startswith(subdir):
                yield obj

    def get_package_names(self, subdir: Optional[str] = None) -> List[str]:
        return sorted({self.obj_to_package_name(obj) for obj in self.gen_file_list(subdir)})

    def normalize_package_version(self: S3IndexType, obj: S3Object) -> str:
        # removes the GPU specifier from the package name as well as
        # unnecessary things like the file extension, architecture name, etc.
        return sub(
            r"%2B.*",
            "",
            "-".join(path.basename(obj.key).split("-")[:2])
        )

    def obj_to_package_name(self, obj: S3Object) -> str:
        return path.basename(obj.key).split('-', 1)[0].lower()

    def to_simple_package_html(
        self,
        subdir: Optional[str],
        package_name: str
    ) -> str:
        """Generates a string that can be used as the package simple HTML index
        """
        out: List[str] = []
        # Adding html header
        out.append('<!DOCTYPE html>')
        out.append('<html>')
        out.append('  <body>')
        out.append('    <h1>Links for {}</h1>'.format(package_name.lower().replace("_", "-")))
        for obj in sorted(self.gen_file_list(subdir, package_name)):
            # Do not include checksum for nightly packages, see
            # https://github.com/pytorch/test-infra/pull/6307
            maybe_fragment = f"#sha256={obj.checksum}" if obj.checksum and not obj.orig_key.startswith("whl/nightly") else ""
            attributes = ""
            if obj.pep658:
                pep658_sha = f"sha256={obj.pep658}"
                # pep714 renames the attribute to data-core-metadata
                attributes = (
                    f' data-dist-info-metadata="{pep658_sha}" data-core-metadata="{pep658_sha}"'
                )
            # Mark networkx versions with Python requirements (pytorch/pytorch#152191)
            # networkx 3.4.2 for Python 3.10, 3.5+ for Python 3.11+
            if obj.key.endswith("networkx-3.4.2-py3-none-any.whl"):
                attributes += ' data-requires-python="&gt;=3.10"'
            elif "networkx-" in obj.key and obj.key.endswith("-py3-none-any.whl"):
                attributes += ' data-requires-python="&gt;=3.11"'

            stripped_key = obj.key.split("/")[-1]

            out.append(
                f'    <a href="../{stripped_key}{maybe_fragment}"{attributes}>{path.basename(obj.key).replace("%2B","+")}</a><br/>'
            )
        # Adding html footer
        out.append('  </body>')
        out.append('</html>')
        out.append(f'<!--TIMESTAMP {int(time.time())}-->')
        return '\n'.join(out)

    def to_simple_packages_html(
        self,
        subdir: Optional[str],
    ) -> str:
        """Generates a string that can be used as the simple HTML index
        """
        out: List[str] = []
        # Adding html header
        out.append('<!DOCTYPE html>')
        out.append('<html>')
        out.append('  <body>')
        for pkg_name in sorted(self.get_package_names(subdir)):
            out.append(f'    <a href="{pkg_name.lower().replace("_","-")}/">{pkg_name.replace("_","-")}</a><br/>')
        # Adding html footer
        out.append('  </body>')
        out.append('</html>')
        out.append(f'<!--TIMESTAMP {int(time.time())}-->')
        return '\n'.join(out)

    def upload_pep503_htmls(self) -> None:
        for subdir in self.subdirs:
            index_html = self.to_simple_packages_html(subdir=subdir)
            for bucket in INDEX_BUCKETS:
                print(f"INFO Uploading {subdir}/index.html to {bucket.name}")
                bucket.Object(
                    key=f"{subdir}/index.html"
                ).put(
                    # ACL='public-read',
                    CacheControl='no-cache,no-store,must-revalidate',
                    ContentType='text/html',
                    Body=index_html
                )
            for pkg_name in self.get_package_names(subdir=subdir):
                compat_pkg_name = pkg_name.lower().replace("_", "-")
                index_html = self.to_simple_package_html(subdir=subdir, package_name=pkg_name)
                for bucket in INDEX_BUCKETS:
                    print(f"INFO Uploading {subdir}/{compat_pkg_name}/index.html to {bucket.name}")
                    bucket.Object(
                        key=f"{subdir}/{compat_pkg_name}/index.html"
                    ).put(
                        # ACL='public-read',
                        CacheControl='no-cache,no-store,must-revalidate',
                        ContentType='text/html',
                        Body=index_html
                    )

    def save_pep503_htmls(self) -> None:
        for subdir in self.subdirs:
            print(f"INFO Saving {subdir}/index.html")
            makedirs(subdir, exist_ok=True)
            with open(path.join(subdir, "index.html"), mode="w", encoding="utf-8") as f:
                f.write(self.to_simple_packages_html(subdir=subdir))
            for pkg_name in self.get_package_names(subdir=subdir):
                makedirs(path.join(subdir, pkg_name), exist_ok=True)
                with open(path.join(subdir, pkg_name, "index.html"), mode="w", encoding="utf-8") as f:
                    f.write(self.to_simple_package_html(subdir=subdir, package_name=pkg_name))

    def compute_sha256(self) -> None:
        for obj in self.objects:
            if obj.checksum is not None:
                continue
            print(f"Updating {obj.orig_key} of size {obj.size} with SHA256 checksum")
            s3_obj = BUCKET.Object(key=obj.orig_key)
            s3_obj.copy_from(CopySource={"Bucket": BUCKET.name, "Key": obj.orig_key},
                             Metadata=s3_obj.metadata, MetadataDirective="REPLACE",
                             ACL="public-read",
                             ChecksumAlgorithm="SHA256")

    @classmethod
    def fetch_object_names(cls: Type[S3IndexType], prefix: str) -> List[str]:
        obj_names = []
        paginator = CLIENT.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)
        for page in page_iterator:
            for obj in page['Contents']:
                is_acceptable = (path.dirname(obj['Key']) == prefix) and obj['Key'].endswith(ACCEPTED_FILE_EXTENSIONS)
                if not is_acceptable:
                    continue
                obj_names.append(obj['Key'])
        return obj_names

    def fetch_metadata(self: S3IndexType) -> None:
        # Add PEP 503-compatible hashes to URLs to allow clients to avoid spurious downloads, if possible.
        regex_multipart_upload = r"^[A-Za-z0-9+/=]+=-[0-9]+$"
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            for idx, future in {
                idx: executor.submit(
                    lambda key: CLIENT.head_object(
                        Bucket=BUCKET.name, Key=key, ChecksumMode="Enabled"
                    ),
                    obj.orig_key,
                )
                for (idx, obj) in enumerate(self.objects)
                if obj.size is None
            }.items():
                response = future.result()
                raw = response.get("ChecksumSHA256")
                if raw and match(regex_multipart_upload, raw):
                    # Possibly part of a multipart upload, making the checksum incorrect
                    print(f"WARNING: {self.objects[idx].orig_key} has bad checksum: {raw}")
                    raw = None
                sha256 = raw and base64.b64decode(raw).hex()
                # For older files, rely on checksum-sha256 metadata that can be added to the file later
                if sha256 is None:
                    sha256 = response.get("Metadata", {}).get("checksum-sha256")
                if sha256 is None:
                    sha256 = response.get("Metadata", {}).get("x-amz-meta-checksum-sha256")
                self.objects[idx].checksum = sha256
                if size := response.get("ContentLength"):
                    self.objects[idx].size = int(size)

    def fetch_pep658(self: S3IndexType) -> None:
        def _fetch_metadata(key: str) -> str:
            try:
                response = CLIENT.head_object(
                    Bucket=BUCKET.name, Key=f"{key}.metadata", ChecksumMode="Enabled"
                )
                sha256 = base64.b64decode(response.get("ChecksumSHA256")).hex()
                return sha256
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return None
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            metadata_futures = {
                idx: executor.submit(
                    _fetch_metadata,
                    obj.orig_key,
                )
                for (idx, obj) in enumerate(self.objects)
            }
            for idx, future in metadata_futures.items():
                response = future.result()
                if response is not None:
                    self.objects[idx].pep658 = response

    @classmethod
    def from_S3(cls: Type[S3IndexType], prefix: str, with_metadata: bool = True) -> S3IndexType:
        prefix = prefix.rstrip("/")
        obj_names = cls.fetch_object_names(prefix)

        def sanitize_key(key: str) -> str:
            return key.replace("+", "%2B")

        rc = cls([S3Object(key=sanitize_key(key),
                           orig_key=key,
                           checksum=None,
                           size=None,
                           pep658=None) for key in obj_names], prefix)
        rc.objects = rc.nightly_packages_to_show()
        if with_metadata:
            rc.fetch_metadata()
            rc.fetch_pep658()
        return rc

def update_pep503_index(prefix: str, compute_sha256: bool = False, upload: bool = True):
    """
    Regenerates the PEP 503 simple index for a given S3 prefix.
    Fetches valid artifacts, applies allow-list filtering, optionally updates
    checksums, and uploads (or saves) the generated index.html files.

    This function is used both by the CLI (via main()) and by the AWS Lambda
    wrapper (lambda_function.py).

    IMPORTANT:
        `initialize_bucket()` must be called prior to invoking this function.

        This function is not fully self-contained and depends on the
        module-level globals `BUCKET_NAME`, `BUCKET`, and `INDEX_BUCKETS`
        being initialized beforehand via `initialize_bucket()`.

        Both the CLI flow and the Lambda wrapper perform this initialization
        before calling `update_pep503_index()`.

    Returns:
        int: The number of S3 objects included in the index after filtering.
    """
    print(f"Processing prefix: {prefix}")

    # Record start time to measure S3 fetch duration
    stime = time.time()

    # Bucket must already be initialized via initialize_bucket().
    # S3Index.from_S3() relies on module-level BUCKET_NAME / BUCKET.
    idx = S3Index.from_S3(prefix=prefix, with_metadata=True)
    
    # Record end time and compute elapsed duration
    etime = time.time()
    print(f"Fetched {len(idx.objects)} objects for '{prefix}' in {etime-stime:.2f}s")

    if compute_sha256:
        idx.compute_sha256()
    elif not upload:
        idx.save_pep503_htmls()
    else:
        idx.upload_pep503_htmls()

    return len(idx.objects)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Manage S3 HTML indices for PyTorch")
    parser.add_argument(
        "prefix",
        nargs="?",
        default=None,
        help="Prefix to update (overrides defaults)"
    )
    parser.add_argument("--bucket", type=str, help="S3 bucket name")
    parser.add_argument(
        "--auto-detect-prefixes",
        action="store_true",
        help=(
            "Automatically detect architecture prefixes under the given base "
            "path using S3 CommonPrefixes. Disabled by default."
    )
    )
    parser.add_argument(
        "--starting-from",
        type=str,
        help=(
            "Base prefix for auto-detection (e.g. v2/, v2-staging/, v3/whl/). "
            "Required when using --auto-detect-prefixes."
    )
    )
    parser.add_argument("--do-not-upload", action="store_true")
    parser.add_argument("--compute-sha256", action="store_true")
    return parser

def resolve_prefixes(args) -> List[str]:
    """
    Determines which prefixes to update.

    Priority:
    1. Explicit CLI prefix
    2. Auto-detection (if enabled)
    3. Static PREFIXES list (+ CUSTOM_PREFIX)
    """
    # Backward compatibility: support old "all" behavior
    if args.prefix == "all":
        return PREFIXES

    # Explicit CLI prefix wins
    if args.prefix:
        return [args.prefix]

    # Auto-detection (guarded by flag)
    if args.auto_detect_prefixes:
        base = args.starting_from
        if not base:
            raise RuntimeError("--starting-from must be provided when using --auto-detect-prefixes")

        return detect_prefixes_from_bucket(base)

    # Default static list
    return PREFIXES

def detect_prefixes_from_bucket(base_prefix: str) -> List[str]:
    """
    Detects architecture prefixes dynamically by listing common prefixes
    under a base path (e.g. v2/, v2-staging/, v3/whl/).
    """

    print(f"INFO: Auto-detecting prefixes under '{base_prefix}'")

    prefixes = set()

    paginator = CLIENT.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(
        Bucket=BUCKET_NAME,
        Prefix=base_prefix,
        Delimiter="/"
    )

    for page in page_iterator:
        for common_prefix in page.get("CommonPrefixes", []):
            prefixes.add(common_prefix["Prefix"].rstrip("/"))

    detected = sorted(prefixes)
    print(f"INFO: Detected prefixes: {detected}")

    return detected

def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    initialize_bucket(args.bucket)

    action = "Saving indices" if args.do_not_upload else "Uploading indices"
    if args.compute_sha256:
        action = "Computing checksums"

    prefixes = resolve_prefixes(args)

    print(f"INFO: {action} for prefixes: {prefixes}")

    for prefix in prefixes:
        update_pep503_index(
            prefix=prefix,
            compute_sha256=args.compute_sha256,
            upload=not args.do_not_upload
        )

if __name__ == "__main__":
    main()
