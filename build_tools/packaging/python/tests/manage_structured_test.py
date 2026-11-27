# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the structured product-local PEP 503 index generator.

All tests operate on in-memory inputs. No AWS credentials or network access
are required.
"""

import base64
import os
import re
import sys
from pathlib import Path

import boto3
import botocore.exceptions
import pytest
from botocore.stub import Stubber

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import manage_structured
from manage_structured import (
    _MULTIPART_CHECKSUM_RE,
    IndexPage,
    PackageDir,
    PackageFile,
    build_index_pages,
    create_parser,
    discover_packages,
    fetch_metadata,
    generate_structured_index,
    list_package_dirs,
    pep503_normalize,
    render_package_page,
    render_root_page,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_HREF_RE = re.compile(r'href="([^"]*)"')


def _hrefs(html: str) -> list[str]:
    return _HREF_RE.findall(html)


def _file(
    key: str, *, checksum: str | None = None, pep658: str | None = None
) -> PackageFile:
    return PackageFile(
        key=key,
        filename=key.split("/")[-1],
        checksum=checksum,
        pep658=pep658,
        size=None,
    )


_ListPage = dict[str, list[dict[str, str]]]


class _InMemoryS3Client:
    """S3 stand-in serving list_objects_v2 pages from a fixed set of keys.

    Listing follows S3 delimiter semantics: keys with the delimiter after the
    prefix collapse into CommonPrefixes, the rest are returned as Contents.
    Each page holds at most `page_size` entries. head_object returns a fixed
    size for existing keys and raises a 404 ClientError otherwise.
    """

    def __init__(self, keys: list[str], page_size: int = 1000) -> None:
        self._keys = sorted(keys)
        self._page_size = page_size
        self.list_calls: list[dict[str, str | None]] = []
        self.head_keys: list[str] = []
        self.put_calls: list[dict[str, object]] = []

    def get_paginator(self, operation_name: str) -> "_InMemoryS3Client":
        assert operation_name == "list_objects_v2"
        return self

    def paginate(
        self, *, Bucket: str, Prefix: str, Delimiter: str | None = None
    ) -> list[_ListPage]:
        self.list_calls.append(
            {"Bucket": Bucket, "Prefix": Prefix, "Delimiter": Delimiter}
        )
        entries: list[tuple[str, str]] = []
        seen_prefixes: set[str] = set()
        for key in self._keys:
            if not key.startswith(Prefix):
                continue
            remainder = key[len(Prefix) :]
            if Delimiter and Delimiter in remainder:
                common = Prefix + remainder.split(Delimiter, 1)[0] + Delimiter
                if common not in seen_prefixes:
                    seen_prefixes.add(common)
                    entries.append(("CommonPrefixes", common))
            else:
                entries.append(("Contents", key))

        pages: list[_ListPage] = []
        for start in range(0, len(entries), self._page_size):
            page: _ListPage = {}
            for kind, value in entries[start : start + self._page_size]:
                field = "Prefix" if kind == "CommonPrefixes" else "Key"
                page.setdefault(kind, []).append({field: value})
            pages.append(page)
        return pages or [{}]

    def head_object(
        self, *, Bucket: str, Key: str, ChecksumMode: str | None = None
    ) -> dict[str, object]:
        self.head_keys.append(Key)
        if Key not in self._keys:
            raise botocore.exceptions.ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )
        return {"ContentLength": 1}

    def put_object(self, **kwargs: object) -> None:
        self.put_calls.append(kwargs)


class _CannedListClient:
    """Returns fixed CommonPrefixes pages, for prefixes S3 itself never emits."""

    def __init__(self, prefixes: list[str]) -> None:
        self._prefixes = prefixes

    def get_paginator(self, operation_name: str) -> "_CannedListClient":
        return self

    def paginate(self, **kwargs: str) -> list[_ListPage]:
        return [{"CommonPrefixes": [{"Prefix": p} for p in self._prefixes]}]


def _put_bodies(client: _InMemoryS3Client) -> dict[str, str]:
    bodies: dict[str, str] = {}
    for call in client.put_calls:
        body = call["Body"]
        assert isinstance(body, bytes)
        bodies[str(call["Key"])] = body.decode("utf-8")
    return bodies


# A product root shaped like nightly pytorch whl-next: several packages, each
# with wheels and PEP 658 sidecars, plus the generated index pages.
_PYTORCH_ROOT = "v5/rocm/pytorch/whl-next"
_PYTORCH_KEYS = [
    f"{_PYTORCH_ROOT}/index.html",
    f"{_PYTORCH_ROOT}/torch/index.html",
    f"{_PYTORCH_ROOT}/torch/torch-2.10.0+rocm7.14.0-cp310-cp310-linux_x86_64.whl",
    f"{_PYTORCH_ROOT}/torch/torch-2.10.0+rocm7.14.0-cp310-cp310-linux_x86_64.whl.metadata",
    f"{_PYTORCH_ROOT}/torch/torch-2.10.0+rocm7.14.0-cp312-cp312-linux_x86_64.whl",
    f"{_PYTORCH_ROOT}/torchvision/index.html",
    f"{_PYTORCH_ROOT}/torchvision/torchvision-0.25.0-cp310-cp310-linux_x86_64.whl",
    f"{_PYTORCH_ROOT}/amd-torch-device-gfx1170/index.html",
    f"{_PYTORCH_ROOT}/amd-torch-device-gfx1170/"
    "amd_torch_device_gfx1170-2.10.0-py3-none-any.whl",
]


# ---------------------------------------------------------------------------
# pep503_normalize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Foo_Bar", "foo-bar"),
        ("torch.scatter", "torch-scatter"),
        ("a--b__c", "a-b-c"),
        ("torch_scatter", "torch-scatter"),
        ("rocm_sdk_core", "rocm-sdk-core"),
        # Already normalized -> passthrough.
        ("torch", "torch"),
        ("rocm-sdk-core", "rocm-sdk-core"),
        # Mixed run of separators collapses to a single dash.
        ("a-_.-b", "a-b"),
        ("UPPER", "upper"),
    ],
)
def test_pep503_normalize(raw: str, expected: str) -> None:
    assert pep503_normalize(raw) == expected


def test_pep503_normalize_is_idempotent() -> None:
    once = pep503_normalize("Foo_._Bar")
    assert pep503_normalize(once) == once


# ---------------------------------------------------------------------------
# discover_packages
# ---------------------------------------------------------------------------


def test_discover_groups_files_by_package_dir() -> None:
    keys = [
        "pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
        "pytorch/whl/torch/torch-2.10.0-cp312-cp312-linux_x86_64.whl",
        "pytorch/whl/torchvision/torchvision-0.21.0-cp310-cp310-linux_x86_64.whl",
    ]
    packages = discover_packages(keys, root="pytorch/whl")
    assert [p.name for p in packages] == ["torch", "torchvision"]
    torch = packages[0]
    assert len(torch.files) == 2
    assert all(f.filename.startswith("torch-") for f in torch.files)


def test_discover_filters_by_extension() -> None:
    keys = [
        "core/whl/rocm-sdk-core/rocm_sdk_core-1.0.0-py3-none-any.whl",
        "core/whl/rocm-sdk-core/rocm_sdk_core-1.0.0.tar.gz",
        "core/whl/rocm-sdk-core/rocm_sdk_core-1.0.0.zip",
        "core/whl/rocm-sdk-core/README.md",
        "core/whl/rocm-sdk-core/rocm_sdk_core-1.0.0.whl.metadata",
    ]
    packages = discover_packages(keys, root="core/whl")
    assert len(packages) == 1
    names = sorted(f.filename for f in packages[0].files)
    assert names == [
        "rocm_sdk_core-1.0.0-py3-none-any.whl",
        "rocm_sdk_core-1.0.0.tar.gz",
        "rocm_sdk_core-1.0.0.zip",
    ]


def test_discover_one_level_down_only() -> None:
    keys = [
        # Valid: <root>/<pkg>/<file>
        "pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
        # File directly under root -> ignored (no package dir).
        "pytorch/whl/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
        # Nested one level too deep -> ignored.
        "pytorch/whl/torch/extra/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
        # Root index page -> ignored.
        "pytorch/whl/index.html",
        # Package index page -> ignored.
        "pytorch/whl/torch/index.html",
    ]
    packages = discover_packages(keys, root="pytorch/whl")
    assert len(packages) == 1
    assert packages[0].name == "torch"
    assert len(packages[0].files) == 1


def test_discover_package_scoping() -> None:
    keys = [
        "pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
        "pytorch/whl/torchvision/torchvision-0.21.0-cp310-cp310-linux_x86_64.whl",
    ]
    packages = discover_packages(keys, root="pytorch/whl", package="torch")
    assert [p.name for p in packages] == ["torch"]


def test_discover_directory_authoritative_name() -> None:
    # The package name comes from the directory, not re-derived per file.
    keys = [
        "core/whl/rocm-sdk-core/rocm_sdk_core-1.0.0-py3-none-any.whl",
    ]
    packages = discover_packages(keys, root="core/whl")
    assert packages[0].name == "rocm-sdk-core"


def test_discover_raises_on_unnormalized_directory() -> None:
    keys = [
        "pytorch/whl/Torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
    ]
    with pytest.raises(ValueError, match="not normalized"):
        discover_packages(keys, root="pytorch/whl")


def test_discover_raises_on_filename_dir_mismatch() -> None:
    # A torchvision wheel misfiled under the torch/ directory must fail.
    keys = [
        "pytorch/whl/torch/torchvision-0.21.0-cp310-cp310-linux_x86_64.whl",
    ]
    with pytest.raises(ValueError, match="does not match package directory"):
        discover_packages(keys, root="pytorch/whl")


def test_discover_accepts_underscore_vs_dash_filename() -> None:
    # Wheel dist names escape to underscores (PEP 427); directory uses '-'.
    keys = [
        "pytorch/whl/torch-scatter/torch_scatter-1.0.0-cp310-cp310-linux_x86_64.whl",
    ]
    packages = discover_packages(keys, root="pytorch/whl")
    assert packages[0].name == "torch-scatter"


def test_discover_accepts_hyphenated_sdist() -> None:
    # Sdist project names can contain '-'; spec-aware parsing handles them
    # correctly where first-hyphen tokenization would not.
    keys = [
        "deps/whl/llnl-hatchet/llnl-hatchet-2024.1.tar.gz",
    ]
    packages = discover_packages(keys, root="deps/whl")
    assert packages[0].name == "llnl-hatchet"


def test_discover_raises_on_hyphenated_sdist_in_wrong_dir() -> None:
    # llnl-hatchet placed under llnl/ (what first-hyphen split would produce)
    # must be rejected.
    keys = [
        "deps/whl/llnl/llnl-hatchet-2024.1.tar.gz",
    ]
    with pytest.raises(ValueError, match="does not match package directory"):
        discover_packages(keys, root="deps/whl")


def test_discover_accepts_local_version_plus_wheel() -> None:
    # Local-version wheels carry a literal '+' (PEP 440). Keys are stored raw,
    # so parse_wheel_filename receives the '+' it expects and the wheel is
    # grouped correctly. A '%2B'-encoded key here would fail to parse.
    keys = [
        "pytorch/whl/torch/torch-2.10.0+rocm7.14.0-cp310-cp310-linux_x86_64.whl",
    ]
    packages = discover_packages(keys, root="pytorch/whl")
    assert [p.name for p in packages] == ["torch"]
    assert (
        packages[0].files[0].filename
        == "torch-2.10.0+rocm7.14.0-cp310-cp310-linux_x86_64.whl"
    )


# ---------------------------------------------------------------------------
# render_package_page: same-directory links
# ---------------------------------------------------------------------------


def test_package_page_uses_same_directory_links() -> None:
    pkg = PackageDir(
        name="torch",
        files=[_file("pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl")],
    )
    html = render_package_page(pkg)
    hrefs = _hrefs(html)
    assert hrefs == ["torch-2.10.0-cp310-cp310-linux_x86_64.whl"]
    # No parent-relative links and no escaping the package directory.
    assert "../" not in html
    for href in hrefs:
        target = href.split("#", 1)[0]
        assert "/" not in target


def test_package_page_omits_checksum_by_default() -> None:
    # Default: skip_checksum=True, so no #sha256 fragment even when available.
    pkg = PackageDir(
        name="torch",
        files=[
            _file(
                "pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
                checksum="abc123",
            )
        ],
    )
    html = render_package_page(pkg)
    assert "#sha256=" not in html


def test_package_page_includes_checksum_when_requested() -> None:
    pkg = PackageDir(
        name="torch",
        files=[
            _file(
                "pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
                checksum="abc123",
            )
        ],
    )
    html = render_package_page(pkg, skip_checksum=False)
    assert "#sha256=abc123" in html
    for href in _hrefs(html):
        assert "/" not in href.split("#", 1)[0]


def test_package_page_includes_pep658_attributes() -> None:
    pkg = PackageDir(
        name="torch",
        files=[
            _file(
                "pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
                pep658="deadbeef",
            )
        ],
    )
    html = render_package_page(pkg)
    assert 'data-dist-info-metadata="sha256=deadbeef"' in html
    assert 'data-core-metadata="sha256=deadbeef"' in html


def test_package_page_escapes_plus_in_href_only() -> None:
    # Keys are stored raw (literal '+'); the href percent-encodes it to %2B
    # while the display text keeps the raw '+'.
    pkg = PackageDir(
        name="torch",
        files=[
            _file(
                "pytorch/whl/torch/torch-2.10.0+rocm7.14.0-cp310-cp310-linux_x86_64.whl"
            )
        ],
    )
    html = render_package_page(pkg)
    assert "torch-2.10.0+rocm7.14.0-cp310-cp310-linux_x86_64.whl</a>" in html
    assert 'href="torch-2.10.0%2Brocm7.14.0-cp310-cp310-linux_x86_64.whl' in html


def test_package_page_rejects_slash_in_filename() -> None:
    # The same-directory invariant: a filename with '/' would escape the
    # package directory and must be rejected rather than rendered.
    pkg = PackageDir(
        name="torch",
        files=[
            PackageFile(
                key="pytorch/whl/torch/sub/torch-2.10.0.whl",
                filename="sub/torch-2.10.0.whl",
                checksum=None,
                pep658=None,
                size=None,
            )
        ],
    )
    with pytest.raises(ValueError, match="escape package directory"):
        render_package_page(pkg)


def test_package_page_networkx_requires_python() -> None:
    pkg = PackageDir(
        name="networkx",
        files=[
            _file("deps/whl/networkx/networkx-3.4.2-py3-none-any.whl"),
            _file("deps/whl/networkx/networkx-3.5-py3-none-any.whl"),
        ],
    )
    html = render_package_page(pkg)
    assert "networkx-3.4.2-py3-none-any.whl" in html
    # 3.4.2 -> >=3.10, other 3.x -> >=3.11
    assert 'data-requires-python="&gt;=3.10"' in html
    assert 'data-requires-python="&gt;=3.11"' in html


# ---------------------------------------------------------------------------
# render_root_page
# ---------------------------------------------------------------------------


def test_root_page_lists_local_packages() -> None:
    packages = [
        PackageDir(name="torch", files=[]),
        PackageDir(name="torchvision", files=[]),
    ]
    html = render_root_page(packages)
    hrefs = _hrefs(html)
    assert hrefs == ["torch/", "torchvision/"]


# ---------------------------------------------------------------------------
# Root-only generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("root", [_PYTORCH_ROOT, f"{_PYTORCH_ROOT}/"])
def test_list_package_dirs_uses_single_delimited_listing(root: str) -> None:
    client = _InMemoryS3Client(_PYTORCH_KEYS)

    packages = list_package_dirs(client, "bucket", root)

    assert [p.name for p in packages] == [
        "amd-torch-device-gfx1170",
        "torch",
        "torchvision",
    ]
    assert all(p.files == [] for p in packages)
    assert client.list_calls == [
        {"Bucket": "bucket", "Prefix": f"{_PYTORCH_ROOT}/", "Delimiter": "/"}
    ]


def test_list_package_dirs_merges_and_sorts_across_pages() -> None:
    keys = [f"core/whl/pkg-{i:03d}/pkg_{i:03d}-1.0-py3-none-any.whl" for i in range(7)]
    client = _InMemoryS3Client(list(reversed(keys)), page_size=2)

    packages = list_package_dirs(client, "bucket", "core/whl")

    assert [p.name for p in packages] == [f"pkg-{i:03d}" for i in range(7)]


def test_list_package_dirs_follows_s3_continuation_tokens() -> None:
    # Exercises the real botocore paginator: the second request must carry
    # the continuation token and keep the delimiter.
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    request = {"Bucket": "bucket", "Prefix": "core/whl/", "Delimiter": "/"}
    with Stubber(client) as stubber:
        stubber.add_response(
            "list_objects_v2",
            {
                "IsTruncated": True,
                "NextContinuationToken": "token-1",
                "CommonPrefixes": [{"Prefix": "core/whl/rocm-sdk-core/"}],
            },
            request,
        )
        stubber.add_response(
            "list_objects_v2",
            {
                "IsTruncated": False,
                "CommonPrefixes": [{"Prefix": "core/whl/numpy/"}],
            },
            {**request, "ContinuationToken": "token-1"},
        )

        packages = list_package_dirs(client, "bucket", "core/whl")

        stubber.assert_no_pending_responses()
    assert [p.name for p in packages] == ["numpy", "rocm-sdk-core"]


def test_list_package_dirs_empty_root() -> None:
    client = _InMemoryS3Client(["other/whl/torch/torch-2.10.0-py3-none-any.whl"])

    assert list_package_dirs(client, "bucket", "pytorch/whl") == []


def test_list_package_dirs_ignores_objects_directly_under_root() -> None:
    client = _InMemoryS3Client(
        [
            "pytorch/whl/index.html",
            "pytorch/whl/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
            "pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
        ]
    )

    packages = list_package_dirs(client, "bucket", "pytorch/whl")

    assert [p.name for p in packages] == ["torch"]


def test_list_package_dirs_does_not_match_sibling_roots() -> None:
    # "whl/" must not pick up directories under the sibling "whl-next/" root.
    client = _InMemoryS3Client(
        [
            "pytorch/whl/torch/torch-2.10.0-py3-none-any.whl",
            "pytorch/whl-next/torchaudio/torchaudio-2.10.0-py3-none-any.whl",
        ]
    )

    packages = list_package_dirs(client, "bucket", "pytorch/whl")

    assert [p.name for p in packages] == ["torch"]


def test_list_package_dirs_lists_directories_without_artifacts() -> None:
    # Directory existence is the only signal: a directory holding just an
    # index page, sidecars, or nested objects is still listed.
    client = _InMemoryS3Client(
        [
            "pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
            "pytorch/whl/only-index/index.html",
            "pytorch/whl/only-sidecar/only_sidecar-1.0-py3-none-any.whl.metadata",
            "pytorch/whl/only-nested/sub/only_nested-1.0-py3-none-any.whl",
        ]
    )

    packages = list_package_dirs(client, "bucket", "pytorch/whl")

    assert [p.name for p in packages] == [
        "only-index",
        "only-nested",
        "only-sidecar",
        "torch",
    ]


@pytest.mark.parametrize("dir_name", ["Torch", "torch_vision", "torch.vision"])
def test_list_package_dirs_rejects_non_normalized_name(dir_name: str) -> None:
    client = _InMemoryS3Client([f"pytorch/whl/{dir_name}/index.html"])

    with pytest.raises(ValueError, match="not normalized"):
        list_package_dirs(client, "bucket", "pytorch/whl")


@pytest.mark.parametrize(
    "dir_name",
    [
        "torch vision",
        'torch"vision',
        "<torch>",
        "-torch",
        "torch-",
        "torch+rocm",
        "torch\n",
    ],
)
def test_list_package_dirs_rejects_invalid_name(dir_name: str) -> None:
    client = _InMemoryS3Client([f"pytorch/whl/{dir_name}/index.html"])

    with pytest.raises(ValueError, match="Invalid package directory name"):
        list_package_dirs(client, "bucket", "pytorch/whl")


def test_list_package_dirs_rejects_empty_directory_name() -> None:
    # "pytorch/whl//x" yields the CommonPrefix "pytorch/whl//".
    client = _InMemoryS3Client(["pytorch/whl//torch-2.10.0-py3-none-any.whl"])

    with pytest.raises(ValueError, match="Malformed package directory prefix"):
        list_package_dirs(client, "bucket", "pytorch/whl")


@pytest.mark.parametrize("prefix", ["other/whl/torch/", "pytorch/whl/torch"])
def test_list_package_dirs_rejects_unexpected_prefix(prefix: str) -> None:
    client = _CannedListClient([prefix])

    with pytest.raises(ValueError, match="Malformed package directory prefix"):
        list_package_dirs(client, "bucket", "pytorch/whl")


def test_list_package_dirs_propagates_s3_errors() -> None:
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    with Stubber(client) as stubber:
        stubber.add_client_error(
            "list_objects_v2", service_error_code="AccessDenied", http_status_code=403
        )
        with pytest.raises(botocore.exceptions.ClientError, match="AccessDenied"):
            list_package_dirs(client, "bucket", "core/whl")


def test_generate_root_only_uploads_only_root_page() -> None:
    client = _InMemoryS3Client(_PYTORCH_KEYS)

    count = generate_structured_index(
        client=client,
        bucket_name="bucket",
        root=_PYTORCH_ROOT,
        root_only=True,
    )

    assert count == 3
    assert client.head_keys == []
    assert [call["Delimiter"] for call in client.list_calls] == ["/"]
    assert len(client.put_calls) == 1
    put = client.put_calls[0]
    assert put["Bucket"] == "bucket"
    assert put["Key"] == f"{_PYTORCH_ROOT}/index.html"
    assert put["ContentType"] == "text/html"
    assert put["CacheControl"] == "no-cache,no-store,must-revalidate"
    assert _hrefs(_put_bodies(client)[f"{_PYTORCH_ROOT}/index.html"]) == [
        "amd-torch-device-gfx1170/",
        "torch/",
        "torchvision/",
    ]


def test_generate_root_only_matches_full_mode_root_page() -> None:
    full_client = _InMemoryS3Client(_PYTORCH_KEYS)
    generate_structured_index(
        client=full_client, bucket_name="bucket", root=_PYTORCH_ROOT
    )
    root_only_client = _InMemoryS3Client(_PYTORCH_KEYS)
    generate_structured_index(
        client=root_only_client,
        bucket_name="bucket",
        root=_PYTORCH_ROOT,
        root_only=True,
    )

    root_key = f"{_PYTORCH_ROOT}/index.html"
    full_root = _put_bodies(full_client)[root_key]
    root_only_root = _put_bodies(root_only_client)[root_key]
    assert _hrefs(root_only_root) == _hrefs(full_root)
    # Full mode does a HEAD per artifact and sidecar and rewrites every
    # package page; root-only does neither.
    assert full_client.head_keys
    assert len(full_client.put_calls) == 4
    assert root_only_client.head_keys == []
    assert len(root_only_client.put_calls) == 1


def test_generate_root_only_empty_root_writes_empty_page() -> None:
    client = _InMemoryS3Client([])

    count = generate_structured_index(
        client=client, bucket_name="bucket", root="pytorch/whl", root_only=True
    )

    assert count == 0
    assert _hrefs(_put_bodies(client)["pytorch/whl/index.html"]) == []


def test_generate_root_only_invalid_directory_uploads_nothing() -> None:
    client = _InMemoryS3Client(
        [
            "pytorch/whl/torch/torch-2.10.0-py3-none-any.whl",
            "pytorch/whl/Torch/torch-2.10.0-py3-none-any.whl",
        ]
    )

    with pytest.raises(ValueError, match="not normalized"):
        generate_structured_index(
            client=client, bucket_name="bucket", root="pytorch/whl", root_only=True
        )
    assert client.put_calls == []


def test_generate_root_only_saves_only_root_page_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _InMemoryS3Client(_PYTORCH_KEYS)

    generate_structured_index(
        client=client,
        bucket_name="bucket",
        root=_PYTORCH_ROOT,
        root_only=True,
        upload=False,
    )

    written = sorted(
        p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()
    )
    assert written == [f"{_PYTORCH_ROOT}/index.html"]
    assert client.put_calls == []
    html = (tmp_path / _PYTORCH_ROOT / "index.html").read_text(encoding="utf-8")
    assert _hrefs(html) == ["amd-torch-device-gfx1170/", "torch/", "torchvision/"]


def test_generate_rejects_root_only_with_package() -> None:
    client = _InMemoryS3Client(_PYTORCH_KEYS)

    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_structured_index(
            client=client,
            bucket_name="bucket",
            root=_PYTORCH_ROOT,
            package="torch",
            root_only=True,
        )
    assert client.list_calls == []
    assert client.put_calls == []


def test_parser_root_only_defaults_off() -> None:
    args = create_parser().parse_args(["pytorch/whl"])

    assert args.root_only is False
    assert args.package is None


def test_parser_accepts_root_only() -> None:
    args = create_parser().parse_args(["pytorch/whl", "--root-only"])

    assert args.root_only is True
    assert args.package is None


@pytest.mark.parametrize(
    "argv",
    [
        ["pytorch/whl", "--root-only", "--package", "torch"],
        ["pytorch/whl", "--package=torch", "--root-only"],
    ],
)
def test_parser_rejects_root_only_with_package(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        create_parser().parse_args(argv)


def test_main_root_only_uploads_only_root_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _InMemoryS3Client(_PYTORCH_KEYS)
    monkeypatch.setattr(manage_structured.boto3, "client", lambda service: client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manage_structured.py",
            _PYTORCH_ROOT,
            "--bucket",
            "bucket",
            "--upload",
            "--root-only",
        ],
    )

    manage_structured.main()

    assert client.head_keys == []
    assert [call["Key"] for call in client.put_calls] == [f"{_PYTORCH_ROOT}/index.html"]


# ---------------------------------------------------------------------------
# build_index_pages: root inclusion + scoping
# ---------------------------------------------------------------------------


def _two_packages() -> list[PackageDir]:
    return [
        PackageDir(
            name="torch",
            files=[
                _file("pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl")
            ],
        ),
        PackageDir(
            name="torchvision",
            files=[
                _file(
                    "pytorch/whl/torchvision/torchvision-0.21.0-cp310-cp310-linux_x86_64.whl"
                )
            ],
        ),
    ]


def test_build_index_pages_full_sweep_includes_root() -> None:
    pages = build_index_pages(_two_packages(), root="pytorch/whl", write_root=True)
    keys = {p.key for p in pages}
    assert "pytorch/whl/index.html" in keys
    assert "pytorch/whl/torch/index.html" in keys
    assert "pytorch/whl/torchvision/index.html" in keys


def test_build_index_pages_scoped_skips_root() -> None:
    packages = [p for p in _two_packages() if p.name == "torch"]
    pages = build_index_pages(packages, root="pytorch/whl", write_root=False)
    keys = {p.key for p in pages}
    assert "pytorch/whl/index.html" not in keys
    assert keys == {"pytorch/whl/torch/index.html"}
    assert all(isinstance(p, IndexPage) for p in pages)


# ---------------------------------------------------------------------------
# Multipart checksum guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, should_match",
    [
        # Composite multipart checksums -- must be discarded.
        ("abc123==-1", True),
        ("abc123==-12", True),
        ("AAAA+/===-2", True),
        # Normal base64 SHA256 -- must not match.
        ("47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=", False),
        ("abc123==", False),
        ("", False),
    ],
)
def test_multipart_checksum_regex(raw: str, should_match: bool) -> None:
    assert bool(_MULTIPART_CHECKSUM_RE.match(raw)) == should_match


# ---------------------------------------------------------------------------
# fetch_metadata: HEAD-request guard conditions
# ---------------------------------------------------------------------------


class _FakeS3Client:
    """Records head_object calls and returns a canned response per key."""

    def __init__(self, responses: dict[str, dict]) -> None:
        self._responses = responses
        self.head_keys: list[str] = []

    def head_object(self, *, Bucket: str, Key: str, ChecksumMode: str) -> dict:
        self.head_keys.append(Key)
        return self._responses[Key]


# A valid base64-encoded 32-byte SHA256 digest.
_VALID_CHECKSUM_B64 = base64.b64encode(bytes(range(32))).decode("ascii")
_VALID_CHECKSUM_HEX = bytes(range(32)).hex()


def test_fetch_metadata_populates_size_and_checksum() -> None:
    files = [_file("core/whl/numpy/numpy-2.0.0-cp312-cp312-linux_x86_64.whl")]
    client = _FakeS3Client(
        {
            files[0].key: {
                "ChecksumSHA256": _VALID_CHECKSUM_B64,
                "ContentLength": 4096,
            }
        }
    )
    fetch_metadata(client, "bucket", files)
    assert files[0].size == 4096
    assert files[0].checksum == _VALID_CHECKSUM_HEX


def test_fetch_metadata_fetches_when_size_preset_but_checksum_missing() -> None:
    # Regression: the HEAD guard must fire when the checksum is missing even if
    # size is already populated, since one HEAD fills both fields.
    files = [_file("core/whl/numpy/numpy-2.0.0-cp312-cp312-linux_x86_64.whl")]
    files[0].size = 4096  # pre-populated; checksum still None
    client = _FakeS3Client(
        {files[0].key: {"ChecksumSHA256": _VALID_CHECKSUM_B64, "ContentLength": 4096}}
    )
    fetch_metadata(client, "bucket", files)
    assert client.head_keys == [files[0].key]
    assert files[0].checksum == _VALID_CHECKSUM_HEX


def test_fetch_metadata_skips_when_both_present() -> None:
    files = [_file("core/whl/numpy/numpy-2.0.0-cp312-cp312-linux_x86_64.whl")]
    files[0].size = 4096
    files[0].checksum = _VALID_CHECKSUM_HEX
    client = _FakeS3Client({})  # any HEAD would KeyError
    fetch_metadata(client, "bucket", files)
    assert client.head_keys == []


def test_fetch_metadata_uses_raw_key_for_head() -> None:
    # Keys are stored raw; the HEAD request must use the literal '+' key, not a
    # percent-encoded one, or S3 would 404.
    key = "pytorch/whl/torch/torch-2.10.0+rocm7.14.0-cp310-cp310-linux_x86_64.whl"
    files = [_file(key)]
    client = _FakeS3Client(
        {key: {"ChecksumSHA256": _VALID_CHECKSUM_B64, "ContentLength": 10}}
    )
    fetch_metadata(client, "bucket", files)
    assert client.head_keys == [key]


def test_fetch_metadata_discards_multipart_checksum() -> None:
    files = [_file("core/whl/numpy/numpy-2.0.0-cp312-cp312-linux_x86_64.whl")]
    client = _FakeS3Client(
        {files[0].key: {"ChecksumSHA256": "abc123==-4", "ContentLength": 10}}
    )
    fetch_metadata(client, "bucket", files)
    # Composite multipart checksum is discarded; no metadata fallback present.
    assert files[0].checksum is None
    assert files[0].size == 10


def test_fetch_metadata_falls_back_to_object_metadata() -> None:
    files = [_file("core/whl/numpy/numpy-2.0.0-cp312-cp312-linux_x86_64.whl")]
    client = _FakeS3Client(
        {
            files[0].key: {
                "ContentLength": 10,
                "Metadata": {"checksum-sha256": "deadbeef"},
            }
        }
    )
    fetch_metadata(client, "bucket", files)
    assert files[0].checksum == "deadbeef"
