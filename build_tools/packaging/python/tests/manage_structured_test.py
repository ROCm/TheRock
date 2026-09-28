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

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from manage_structured import (
    _MULTIPART_CHECKSUM_RE,
    IndexPage,
    PackageDir,
    PackageFile,
    build_index_pages,
    discover_packages,
    fetch_metadata,
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
