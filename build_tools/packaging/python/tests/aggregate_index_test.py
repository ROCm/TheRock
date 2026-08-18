# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ROCm aggregate Python index ownership tooling."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from aggregate_index import (
    IndexValidationError,
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    ROUTES_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION,
    generate_outputs,
    load_ownership_manifest,
    main,
    parse_ownership_manifest,
    validate_product_indexes,
)


def _valid_manifest() -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "streams": {
            "known": ["dev", "nightly", "rc", "stable", "stable-staging"],
            "default": ["dev", "nightly", "rc", "stable", "stable-staging"],
        },
        "python_indexes": [
            {
                "public_base": "/rocm/whl-next",
                "packages": {
                    "torch": {"owner_path": "pytorch/whl-next"},
                    "rocm-sdk-core": {"owner_path": "core/whl-next"},
                    "rocm-sdk-devel": {"owner_path": "core/whl-next"},
                    "apex": {"owner_path": "pytorch/whl-next"},
                },
            }
        ],
    }


def _write_file(root: Path, relative_path: str, text: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_product_root(root: Path, owner_path: str, package_names: list[str]) -> Path:
    links = "\n".join(
        f'    <a href="{package_name}/">{package_name}</a><br/>'
        for package_name in package_names
    )
    return _write_file(
        root,
        f"rocm/{owner_path}/index.html",
        f"""<!DOCTYPE html>
<html>
  <body>
{links}
  </body>
</html>
""",
    )


def _write_package_page(root: Path, owner_path: str, package_name: str) -> Path:
    return _write_file(
        root,
        f"rocm/{owner_path}/{package_name}/index.html",
        f"""<!DOCTYPE html>
<html>
  <body>
    <a href="{package_name}-1.0.0-py3-none-any.whl">
      {package_name}-1.0.0-py3-none-any.whl
    </a>
  </body>
</html>
""",
    )


def _write_valid_content_root(root: Path) -> None:
    _write_product_root(root, "core/whl-next", ["extra-core", "rocm-sdk-core"])
    _write_product_root(root, "pytorch/whl-next", ["apex", "torch"])
    _write_package_page(root, "core/whl-next", "rocm-sdk-core")
    _write_package_page(root, "pytorch/whl-next", "apex")
    _write_package_page(root, "pytorch/whl-next", "torch")


def _write_complete_content_root(root: Path) -> None:
    _write_valid_content_root(root)
    _write_product_root(
        root,
        "core/whl-next",
        ["extra-core", "rocm-sdk-core", "rocm-sdk-devel"],
    )
    _write_package_page(root, "core/whl-next", "rocm-sdk-devel")


def _generated_output_bytes(output_dir: Path) -> dict[str, bytes]:
    relative_paths = [
        "rocm/whl-next/index.html",
        "rocm-whl-next-routes.json",
        "validation.json",
    ]
    return {
        relative_path: (output_dir / relative_path).read_bytes()
        for relative_path in relative_paths
    }


def _assert_no_generated_outputs(output_dir: Path) -> None:
    assert not (output_dir / "rocm/whl-next/index.html").exists()
    assert not (output_dir / "rocm-whl-next-routes.json").exists()
    assert not (output_dir / "validation.json").exists()


def test_checked_in_manifest_loads() -> None:
    manifest_path = Path(__file__).parents[1] / "rocm_whl_next_ownership.yaml"
    manifest = load_ownership_manifest(manifest_path)
    index = manifest.python_indexes[0]

    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION
    assert index.public_base == "/rocm/whl-next"
    assert manifest.streams.known == (
        "dev",
        "nightly",
        "rc",
        "stable",
        "stable-staging",
    )
    assert "jax-rocm7-plugin" not in index.packages
    assert "jax-rocm7-pjrt" not in index.packages
    assert index.packages["jax-rocm10-plugin"].owner_path == "jax/whl-next"
    assert index.packages["rocm-sdk-core"].owner_path == "core/whl-next"
    assert index.packages["torch"].owner_path == "pytorch/whl-next"


def test_checked_in_manifest_uses_known_owner_paths() -> None:
    manifest_path = Path(__file__).parents[1] / "rocm_whl_next_ownership.yaml"
    manifest = load_ownership_manifest(manifest_path)
    index = manifest.python_indexes[0]
    known_owner_paths = {"core/whl-next", "jax/whl-next", "pytorch/whl-next"}

    assert {
        *[package.owner_path for package in index.packages.values()],
    } == known_owner_paths


def test_parse_sorts_packages_by_owner_path_then_name() -> None:
    data = _valid_manifest()
    manifest = parse_ownership_manifest(data)
    index = manifest.python_indexes[0]

    assert list(index.packages) == [
        "rocm-sdk-core",
        "rocm-sdk-devel",
        "apex",
        "torch",
    ]
    assert [package.name for package in index.ordered_packages()] == [
        "rocm-sdk-core",
        "rocm-sdk-devel",
        "apex",
        "torch",
    ]


def test_validate_product_indexes_success(tmp_path: Path) -> None:
    _write_valid_content_root(tmp_path)
    manifest = parse_ownership_manifest(_valid_manifest())

    validated = validate_product_indexes(
        manifest,
        tmp_path,
        stream="nightly",
        require_all_manifest_packages=False,
    )

    assert validated.public_base == "/rocm/whl-next"
    assert list(validated.packages) == ["rocm-sdk-core", "apex", "torch"]
    assert list(validated.unpublished_packages) == ["rocm-sdk-devel"]
    rocm_sdk_core = validated.packages["rocm-sdk-core"]
    assert rocm_sdk_core.owner_path == "core/whl-next"
    assert rocm_sdk_core.product_root_index == (
        tmp_path / "rocm/core/whl-next/index.html"
    )
    assert rocm_sdk_core.package_index == (
        tmp_path / "rocm/core/whl-next/rocm-sdk-core/index.html"
    )


def test_validate_product_indexes_allows_extra_package_by_default(
    tmp_path: Path,
) -> None:
    _write_valid_content_root(tmp_path)
    manifest = parse_ownership_manifest(_valid_manifest())

    validate_product_indexes(
        manifest,
        tmp_path,
        stream="nightly",
        require_all_manifest_packages=False,
    )


def test_validate_product_indexes_strict_completeness_rejects_extra_package(
    tmp_path: Path,
) -> None:
    _write_valid_content_root(tmp_path)
    manifest = parse_ownership_manifest(_valid_manifest())

    with pytest.raises(
        IndexValidationError, match="absent from the ownership manifest"
    ):
        validate_product_indexes(
            manifest,
            tmp_path,
            stream="nightly",
            strict_completeness=True,
            require_all_manifest_packages=False,
        )


def test_stream_package_generates_exact_route_and_validation_source(
    tmp_path: Path,
) -> None:
    data = _valid_manifest()
    index = data["python_indexes"][0]
    assert isinstance(index, dict)
    packages = index["packages"]
    assert isinstance(packages, dict)
    packages.clear()
    packages["rocm-sdk-device-gfx1201"] = {
        "owner_path": "core/whl-next",
        "streams": ["dev", "nightly"],
    }
    manifest = parse_ownership_manifest(data)
    content_root = tmp_path / "content"
    output_dir = tmp_path / "out"
    _write_product_root(content_root, "core/whl-next", ["rocm-sdk-device-gfx1201"])
    _write_package_page(content_root, "core/whl-next", "rocm-sdk-device-gfx1201")

    outputs = generate_outputs(
        manifest,
        output_dir,
        stream="nightly",
        content_root=content_root,
    )

    route_table = json.loads(outputs.route_table.read_text(encoding="utf-8"))
    validation_report = json.loads(
        outputs.validation_report.read_text(encoding="utf-8")
    )
    assert route_table["stream"] == "nightly"
    assert route_table["routes"] == [
        {
            "owner_path": "core/whl-next",
            "package": "rocm-sdk-device-gfx1201",
            "target": "/rocm/core/whl-next/rocm-sdk-device-gfx1201/",
        }
    ]
    assert validation_report["packages"] == [
        {
            "name": "rocm-sdk-device-gfx1201",
            "owner_path": "core/whl-next",
            "product_root_index": "/rocm/core/whl-next/index.html",
            "package_index": "/rocm/core/whl-next/rocm-sdk-device-gfx1201/index.html",
            "source": "exact",
        }
    ]


def test_stream_filtered_package_is_excluded_from_manifest_only_output(
    tmp_path: Path,
) -> None:
    data = _valid_manifest()
    index = data["python_indexes"][0]
    assert isinstance(index, dict)
    packages = index["packages"]
    assert isinstance(packages, dict)
    packages["rocm-sdk-device-gfx900"] = {
        "owner_path": "core/whl-next",
        "streams": ["dev", "nightly"],
    }
    manifest = parse_ownership_manifest(data)
    output_dir = tmp_path / "out"

    outputs = generate_outputs(manifest, output_dir, stream="stable")

    route_table = json.loads(outputs.route_table.read_text(encoding="utf-8"))
    validation_report = json.loads(
        outputs.validation_report.read_text(encoding="utf-8")
    )
    assert route_table["stream"] == "stable"
    assert "rocm-sdk-device-gfx900" not in {
        route["package"] for route in route_table["routes"]
    }
    assert validation_report["generation_mode"] == "manifest"
    assert validation_report["content_validated"] is False
    assert {package["source"] for package in validation_report["packages"]} == {
        "manifest"
    }


@pytest.mark.parametrize(
    "extra_arg",
    [
        "--allow-unpublished",
        "--strict-completeness",
    ],
)
def test_manifest_only_generate_rejects_content_validation_flags(
    tmp_path: Path,
    extra_arg: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "ownership.yaml"
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")

    exit_code = main(
        [
            "generate",
            "--manifest",
            os.fspath(manifest_path),
            "--stream",
            "nightly",
            "--output-dir",
            os.fspath(output_dir),
            extra_arg,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Content validation flags require --content-root" in captured.err
    _assert_no_generated_outputs(output_dir)


def test_strict_completeness_rejects_package_disabled_for_stream(
    tmp_path: Path,
) -> None:
    data = _valid_manifest()
    index = data["python_indexes"][0]
    assert isinstance(index, dict)
    packages = index["packages"]
    assert isinstance(packages, dict)
    packages.clear()
    packages["rocm-sdk-device-gfx900"] = {
        "owner_path": "core/whl-next",
        "streams": ["dev", "nightly"],
    }
    manifest = parse_ownership_manifest(data)
    _write_product_root(tmp_path, "core/whl-next", ["rocm-sdk-device-gfx900"])
    _write_package_page(tmp_path, "core/whl-next", "rocm-sdk-device-gfx900")

    with pytest.raises(
        IndexValidationError, match="absent from the ownership manifest"
    ):
        validate_product_indexes(
            manifest,
            tmp_path,
            stream="stable",
            strict_completeness=True,
        )


def test_generate_outputs_writes_routed_artifacts(tmp_path: Path) -> None:
    content_root = tmp_path / "content"
    output_dir = tmp_path / "out"
    _write_complete_content_root(content_root)
    manifest = parse_ownership_manifest(_valid_manifest())

    outputs = generate_outputs(
        manifest,
        output_dir,
        stream="nightly",
        content_root=content_root,
    )

    assert outputs.aggregate_root == output_dir / "rocm/whl-next/index.html"
    assert outputs.route_table == output_dir / "rocm-whl-next-routes.json"
    assert outputs.validation_report == output_dir / "validation.json"

    aggregate_root = outputs.aggregate_root.read_text(encoding="utf-8")
    assert (
        aggregate_root
        == """<!DOCTYPE html>
<html>
  <body>
    <a href="apex/">apex</a><br/>
    <a href="rocm-sdk-core/">rocm-sdk-core</a><br/>
    <a href="rocm-sdk-devel/">rocm-sdk-devel</a><br/>
    <a href="torch/">torch</a><br/>
  </body>
</html>
"""
    )

    route_table = json.loads(outputs.route_table.read_text(encoding="utf-8"))
    assert route_table == {
        "schema_version": ROUTES_SCHEMA_VERSION,
        "public_base": "/rocm/whl-next",
        "stream": "nightly",
        "routes": [
            {
                "owner_path": "core/whl-next",
                "package": "rocm-sdk-core",
                "target": "/rocm/core/whl-next/rocm-sdk-core/",
            },
            {
                "owner_path": "core/whl-next",
                "package": "rocm-sdk-devel",
                "target": "/rocm/core/whl-next/rocm-sdk-devel/",
            },
            {
                "owner_path": "pytorch/whl-next",
                "package": "apex",
                "target": "/rocm/pytorch/whl-next/apex/",
            },
            {
                "owner_path": "pytorch/whl-next",
                "package": "torch",
                "target": "/rocm/pytorch/whl-next/torch/",
            },
        ],
    }

    validation_report = json.loads(
        outputs.validation_report.read_text(encoding="utf-8")
    )
    assert validation_report["schema_version"] == VALIDATION_SCHEMA_VERSION
    assert validation_report["public_base"] == "/rocm/whl-next"
    assert validation_report["stream"] == "nightly"
    assert validation_report["generation_mode"] == "content"
    assert validation_report["content_validated"] is True
    assert validation_report["package_count"] == 4
    assert validation_report["unpublished_package_count"] == 0
    assert validation_report["owners"] == {
        "core/whl-next": {"package_count": 2},
        "pytorch/whl-next": {"package_count": 2},
    }
    assert [package["name"] for package in validation_report["packages"]] == [
        "rocm-sdk-core",
        "rocm-sdk-devel",
        "apex",
        "torch",
    ]
    assert {package["source"] for package in validation_report["packages"]} == {"exact"}
    assert validation_report["unpublished_packages"] == []


def test_generate_outputs_requires_manifest_packages_by_default(
    tmp_path: Path,
) -> None:
    content_root = tmp_path / "content"
    output_dir = tmp_path / "out"
    _write_valid_content_root(content_root)
    manifest = parse_ownership_manifest(_valid_manifest())

    with pytest.raises(IndexValidationError, match="missing canonical package link"):
        generate_outputs(
            manifest,
            output_dir,
            stream="nightly",
            content_root=content_root,
        )
    _assert_no_generated_outputs(output_dir)


def test_generate_outputs_can_allow_unpublished_packages(tmp_path: Path) -> None:
    content_root = tmp_path / "content"
    output_dir = tmp_path / "out"
    _write_valid_content_root(content_root)
    manifest = parse_ownership_manifest(_valid_manifest())

    outputs = generate_outputs(
        manifest,
        output_dir,
        stream="nightly",
        content_root=content_root,
        require_all_manifest_packages=False,
    )

    validation_report = json.loads(
        outputs.validation_report.read_text(encoding="utf-8")
    )
    assert validation_report["package_count"] == 3
    assert validation_report["unpublished_package_count"] == 1
    assert [package["name"] for package in validation_report["packages"]] == [
        "rocm-sdk-core",
        "apex",
        "torch",
    ]
    assert validation_report["unpublished_packages"] == [
        {
            "name": "rocm-sdk-devel",
            "owner_path": "core/whl-next",
            "product_root_index": "/rocm/core/whl-next/index.html",
        }
    ]


def test_generate_outputs_strict_completeness_rejects_extra_package(
    tmp_path: Path,
) -> None:
    content_root = tmp_path / "content"
    output_dir = tmp_path / "out"
    _write_valid_content_root(content_root)
    manifest = parse_ownership_manifest(_valid_manifest())

    with pytest.raises(
        IndexValidationError, match="absent from the ownership manifest"
    ):
        generate_outputs(
            manifest,
            output_dir,
            stream="nightly",
            content_root=content_root,
            strict_completeness=True,
        )
    _assert_no_generated_outputs(output_dir)


def test_generate_outputs_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    content_root = tmp_path / "content"
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"
    _write_complete_content_root(content_root)
    manifest = parse_ownership_manifest(_valid_manifest())

    generate_outputs(
        manifest,
        first_output_dir,
        stream="nightly",
        content_root=content_root,
    )
    generate_outputs(
        manifest,
        second_output_dir,
        stream="nightly",
        content_root=content_root,
    )

    assert _generated_output_bytes(first_output_dir) == _generated_output_bytes(
        second_output_dir
    )


def test_generate_outputs_is_deterministic_across_content_roots(
    tmp_path: Path,
) -> None:
    first_content_root = tmp_path / "first-content"
    second_content_root = tmp_path / "second-content"
    first_output_dir = tmp_path / "first-output"
    second_output_dir = tmp_path / "second-output"
    _write_complete_content_root(first_content_root)
    _write_complete_content_root(second_content_root)
    manifest = parse_ownership_manifest(_valid_manifest())

    generate_outputs(
        manifest,
        first_output_dir,
        stream="nightly",
        content_root=first_content_root,
    )
    generate_outputs(
        manifest,
        second_output_dir,
        stream="nightly",
        content_root=second_content_root,
    )

    assert _generated_output_bytes(first_output_dir) == _generated_output_bytes(
        second_output_dir
    )


def test_generate_outputs_overwrite_is_deterministic(tmp_path: Path) -> None:
    content_root = tmp_path / "content"
    output_dir = tmp_path / "out"
    _write_complete_content_root(content_root)
    manifest = parse_ownership_manifest(_valid_manifest())

    generate_outputs(
        manifest,
        output_dir,
        stream="nightly",
        content_root=content_root,
    )
    first_output = _generated_output_bytes(output_dir)
    generate_outputs(
        manifest,
        output_dir,
        stream="nightly",
        content_root=content_root,
    )

    assert _generated_output_bytes(output_dir) == first_output


def test_main_validate_manifest_prints_default_streams_deterministically(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "ownership.yaml"
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")

    exit_code = main(["validate-manifest", "--manifest", os.fspath(manifest_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "default_streams: dev, nightly, rc, stable, stable-staging\n" in captured.out


def test_main_generate_default_writes_complete_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content_root = tmp_path / "content"
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "ownership.yaml"
    _write_complete_content_root(content_root)
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")

    exit_code = main(
        [
            "generate",
            "--manifest",
            os.fspath(manifest_path),
            "--stream",
            "nightly",
            "--output-dir",
            os.fspath(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert (output_dir / "rocm/whl-next/index.html").is_file()
    assert (output_dir / "rocm-whl-next-routes.json").is_file()
    assert (output_dir / "validation.json").is_file()


def test_main_generate_requires_manifest_packages_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content_root = tmp_path / "content"
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "ownership.yaml"
    _write_valid_content_root(content_root)
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")

    exit_code = main(
        [
            "generate",
            "--manifest",
            os.fspath(manifest_path),
            "--stream",
            "nightly",
            "--content-root",
            os.fspath(content_root),
            "--output-dir",
            os.fspath(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "missing canonical package link" in captured.err
    assert "Traceback" not in captured.err
    _assert_no_generated_outputs(output_dir)


def test_main_generate_allow_unpublished_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content_root = tmp_path / "content"
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "ownership.yaml"
    _write_valid_content_root(content_root)
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")

    exit_code = main(
        [
            "generate",
            "--manifest",
            os.fspath(manifest_path),
            "--stream",
            "nightly",
            "--content-root",
            os.fspath(content_root),
            "--output-dir",
            os.fspath(output_dir),
            "--allow-unpublished",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "warning: 1 manifest package" in captured.err
    assert (output_dir / "rocm/whl-next/index.html").is_file()
    assert (output_dir / "rocm-whl-next-routes.json").is_file()
    assert (output_dir / "validation.json").is_file()


def test_main_generate_strict_completeness_allow_unpublished_fails_on_extra_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content_root = tmp_path / "content"
    output_dir = tmp_path / "out"
    manifest_path = tmp_path / "ownership.yaml"
    _write_valid_content_root(content_root)
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")

    exit_code = main(
        [
            "generate",
            "--manifest",
            os.fspath(manifest_path),
            "--stream",
            "nightly",
            "--content-root",
            os.fspath(content_root),
            "--output-dir",
            os.fspath(output_dir),
            "--allow-unpublished",
            "--strict-completeness",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "absent from the ownership manifest" in captured.err
    assert "Traceback" not in captured.err
    _assert_no_generated_outputs(output_dir)


def test_main_validate_content_default_requires_manifest_packages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content_root = tmp_path / "content"
    manifest_path = tmp_path / "ownership.yaml"
    _write_valid_content_root(content_root)
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")

    exit_code = main(
        [
            "validate-content",
            "--manifest",
            os.fspath(manifest_path),
            "--stream",
            "nightly",
            "--content-root",
            os.fspath(content_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "missing canonical package link" in captured.err
    assert "Traceback" not in captured.err


def test_main_validate_content_allow_unpublished_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content_root = tmp_path / "content"
    manifest_path = tmp_path / "ownership.yaml"
    _write_valid_content_root(content_root)
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")

    exit_code = main(
        [
            "validate-content",
            "--manifest",
            os.fspath(manifest_path),
            "--stream",
            "nightly",
            "--content-root",
            os.fspath(content_root),
            "--allow-unpublished",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "warning: 1 manifest package" in captured.err
    assert "unpublished_packages: 1" in captured.out


def test_main_validate_content_strict_completeness_with_allow_unpublished_fails_on_extra_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content_root = tmp_path / "content"
    manifest_path = tmp_path / "ownership.yaml"
    _write_valid_content_root(content_root)
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")

    exit_code = main(
        [
            "validate-content",
            "--manifest",
            os.fspath(manifest_path),
            "--stream",
            "nightly",
            "--content-root",
            os.fspath(content_root),
            "--allow-unpublished",
            "--strict-completeness",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "absent from the ownership manifest" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        [
            "generate",
            "--manifest",
            "ownership.yaml",
            "--stream",
            "nightly",
            "--content-root",
            "content",
            "--output-dir",
            "out",
            "--require-all-manifest-packages",
        ],
        [
            "validate-content",
            "--manifest",
            "ownership.yaml",
            "--stream",
            "nightly",
            "--content-root",
            "content",
            "--strict",
        ],
    ],
)
def test_main_rejects_unsupported_content_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    argv = [
        (
            os.fspath(tmp_path / value)
            if value in {"ownership.yaml", "content", "out"}
            else value
        )
        for value in argv
    ]
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "unrecognized arguments" in captured.err


def test_main_reports_manifest_errors_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "ownership.yaml"
    manifest_path.write_text(
        """
schema_version: 1
streams:
  known: [nightly]
  default: [nightly]
python_indexes: []
""".lstrip(),
        encoding="utf-8",
    )

    exit_code = main(["validate-manifest", "--manifest", os.fspath(manifest_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "schema_version must be 2" in captured.err
    assert "Traceback" not in captured.err


def test_main_reports_malformed_yaml_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "ownership.yaml"
    manifest_path.write_text("schema_version: [", encoding="utf-8")

    exit_code = main(["validate-manifest", "--manifest", os.fspath(manifest_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "while parsing" in captured.err
    assert "Traceback" not in captured.err


def test_main_reports_unhashable_yaml_key_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "ownership.yaml"
    manifest_path.write_text("? [schema_version]\n: 1\n", encoding="utf-8")

    exit_code = main(["validate-manifest", "--manifest", os.fspath(manifest_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "unhashable key" in captured.err
    assert "Traceback" not in captured.err


def test_main_reports_missing_manifest_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "missing.yaml"

    exit_code = main(["validate-manifest", "--manifest", os.fspath(manifest_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "cannot read ownership manifest" in captured.err
    assert "Traceback" not in captured.err


def test_main_reports_index_validation_errors_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content_root = tmp_path / "content"
    manifest_path = tmp_path / "ownership.yaml"
    manifest_path.write_text(
        """
schema_version: 2
streams:
  known: [nightly]
  default: [nightly]
python_indexes:
  - public_base: /rocm/whl-next
    packages:
      torch:
        owner_path: pytorch/whl-next
""".lstrip(),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "validate-content",
            "--manifest",
            os.fspath(manifest_path),
            "--stream",
            "nightly",
            "--content-root",
            os.fspath(content_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: Missing product root index:")
    assert "Traceback" not in captured.err


def test_validate_product_indexes_rejects_missing_product_root(
    tmp_path: Path,
) -> None:
    _write_product_root(tmp_path, "pytorch/whl-next", ["apex", "torch"])
    _write_package_page(tmp_path, "core/whl-next", "rocm-sdk-core")
    _write_package_page(tmp_path, "pytorch/whl-next", "apex")
    _write_package_page(tmp_path, "pytorch/whl-next", "torch")
    manifest = parse_ownership_manifest(_valid_manifest())

    with pytest.raises(IndexValidationError, match="Missing product root index"):
        validate_product_indexes(manifest, tmp_path, stream="nightly")


def test_validate_product_indexes_records_unpublished_package_by_default(
    tmp_path: Path,
) -> None:
    _write_product_root(tmp_path, "core/whl-next", ["extra-core"])
    _write_product_root(tmp_path, "pytorch/whl-next", ["apex", "torch"])
    _write_package_page(tmp_path, "core/whl-next", "rocm-sdk-core")
    _write_package_page(tmp_path, "pytorch/whl-next", "apex")
    _write_package_page(tmp_path, "pytorch/whl-next", "torch")
    manifest = parse_ownership_manifest(_valid_manifest())

    validated = validate_product_indexes(
        manifest,
        tmp_path,
        stream="nightly",
        require_all_manifest_packages=False,
    )

    assert list(validated.packages) == ["apex", "torch"]
    assert list(validated.unpublished_packages) == [
        "rocm-sdk-core",
        "rocm-sdk-devel",
    ]


def test_validate_product_indexes_requires_manifest_packages_when_requested(
    tmp_path: Path,
) -> None:
    _write_product_root(tmp_path, "core/whl-next", ["extra-core"])
    _write_product_root(tmp_path, "pytorch/whl-next", ["apex", "torch"])
    _write_package_page(tmp_path, "core/whl-next", "rocm-sdk-core")
    _write_package_page(tmp_path, "pytorch/whl-next", "apex")
    _write_package_page(tmp_path, "pytorch/whl-next", "torch")
    manifest = parse_ownership_manifest(_valid_manifest())

    with pytest.raises(IndexValidationError, match="missing canonical package link"):
        validate_product_indexes(
            manifest,
            tmp_path,
            stream="nightly",
            require_all_manifest_packages=True,
        )


def test_validate_product_indexes_rejects_duplicate_package_link(
    tmp_path: Path,
) -> None:
    _write_product_root(tmp_path, "core/whl-next", ["rocm-sdk-core", "rocm-sdk-core"])
    _write_product_root(tmp_path, "pytorch/whl-next", ["apex", "torch"])
    _write_package_page(tmp_path, "core/whl-next", "rocm-sdk-core")
    _write_package_page(tmp_path, "pytorch/whl-next", "apex")
    _write_package_page(tmp_path, "pytorch/whl-next", "torch")
    manifest = parse_ownership_manifest(_valid_manifest())

    with pytest.raises(IndexValidationError, match="expected exactly one package link"):
        validate_product_indexes(manifest, tmp_path, stream="nightly")


@pytest.mark.parametrize(
    "href, text, match",
    [
        ("../rocm-sdk-core/", "rocm-sdk-core", "must name one package directory"),
        ("/rocm/core/whl-next/rocm-sdk-core/", "rocm-sdk-core", "must not be absolute"),
        ("https://example.com/rocm-sdk-core/", "rocm-sdk-core", "must be relative"),
        ("rocm-sdk-core/#sha256=abc", "rocm-sdk-core", "query or fragment"),
        ("rocm-sdk-core", "rocm-sdk-core", "must end with '/'"),
        ("Rocm_Sdk_Core/", "Rocm_Sdk_Core", "must use normalized package name"),
        ("foo<script>/", "foo<script>", "index.html: Package links"),
        ("foo bar/", "foo bar", "valid normalized Python project name"),
        ("ünïcode/", "ünïcode", "valid normalized Python project name"),
        ("rocm-sdk-core/", "wrong", "link text"),
    ],
)
def test_validate_product_indexes_rejects_bad_product_root_links(
    tmp_path: Path, href: str, text: str, match: str
) -> None:
    _write_file(
        tmp_path,
        "rocm/core/whl-next/index.html",
        f'<html><body><a href="{href}">{text}</a></body></html>',
    )
    _write_product_root(tmp_path, "pytorch/whl-next", ["apex", "torch"])
    _write_package_page(tmp_path, "core/whl-next", "rocm-sdk-core")
    _write_package_page(tmp_path, "pytorch/whl-next", "apex")
    _write_package_page(tmp_path, "pytorch/whl-next", "torch")
    manifest = parse_ownership_manifest(_valid_manifest())

    with pytest.raises(IndexValidationError, match=match):
        validate_product_indexes(manifest, tmp_path, stream="nightly")


def test_validate_product_indexes_rejects_missing_package_page(
    tmp_path: Path,
) -> None:
    _write_valid_content_root(tmp_path)
    (tmp_path / "rocm/pytorch/whl-next/torch/index.html").unlink()
    manifest = parse_ownership_manifest(_valid_manifest())

    with pytest.raises(IndexValidationError, match="Missing package index"):
        validate_product_indexes(
            manifest,
            tmp_path,
            stream="nightly",
            require_all_manifest_packages=False,
        )


def test_validate_product_indexes_rejects_empty_package_page(
    tmp_path: Path,
) -> None:
    _write_valid_content_root(tmp_path)
    (tmp_path / "rocm/pytorch/whl-next/torch/index.html").write_text(
        "", encoding="utf-8"
    )
    manifest = parse_ownership_manifest(_valid_manifest())

    with pytest.raises(IndexValidationError, match="Empty package index"):
        validate_product_indexes(
            manifest,
            tmp_path,
            stream="nightly",
            require_all_manifest_packages=False,
        )


@pytest.mark.parametrize(
    "data, match",
    [
        ({}, "missing required key"),
        (
            {
                "schema_version": 1,
                "streams": {"known": ["nightly"], "default": ["nightly"]},
                "python_indexes": [],
            },
            "schema_version must be 2",
        ),
        (
            {
                "schema_version": True,
                "streams": {"known": ["nightly"], "default": ["nightly"]},
                "python_indexes": [],
            },
            "schema_version must be an integer",
        ),
        (
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "streams": {"known": ["nightly"], "default": ["nightly"]},
                "python_indexes": [],
            },
            "exactly one /rocm/whl-next index",
        ),
        (
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "streams": {"known": ["nightly"], "default": ["nightly"]},
                "python_indexes": [
                    {
                        "public_base": "/rocm/whl-next",
                        "packages": {},
                    }
                ],
            },
            "packages must not be empty",
        ),
    ],
)
def test_rejects_malformed_top_level_data(data: object, match: str) -> None:
    with pytest.raises(ManifestError, match=match):
        parse_ownership_manifest(data)


@pytest.mark.parametrize(
    "streams, match",
    [
        ({"known": [], "default": ["nightly"]}, "known must not be empty"),
        ({"known": ["nightly"], "default": []}, "default must not be empty"),
        (
            {"known": ["nightly", "nightly"], "default": ["nightly"]},
            "duplicate stream",
        ),
        (
            {"known": ["Nightly"], "default": ["Nightly"]},
            "lowercase alphanumeric",
        ),
        (
            {"known": ["release.1"], "default": ["release.1"]},
            "lowercase alphanumeric",
        ),
        (
            {"known": ["nightly"], "default": ["nightly", "rc"]},
            "unknown stream",
        ),
    ],
)
def test_rejects_invalid_stream_config(streams: dict[str, object], match: str) -> None:
    data = _valid_manifest()
    data["streams"] = streams

    with pytest.raises(ManifestError, match=match):
        parse_ownership_manifest(data)


def test_rejects_unknown_package_stream() -> None:
    data = _valid_manifest()
    index = data["python_indexes"][0]
    assert isinstance(index, dict)
    packages = index["packages"]
    assert isinstance(packages, dict)
    packages["torch"] = {
        "owner_path": "pytorch/whl-next",
        "streams": ["nightly", "unknown"],
    }

    with pytest.raises(ManifestError, match="unknown stream"):
        parse_ownership_manifest(data)


def test_rejects_patterns_key_in_schema_v2() -> None:
    data = _valid_manifest()
    index = data["python_indexes"][0]
    assert isinstance(index, dict)
    index["patterns"] = {"foo*": {"owner_path": "core/whl-next"}}

    with pytest.raises(ManifestError, match="unknown key"):
        parse_ownership_manifest(data)


@pytest.mark.parametrize(
    "public_base, match",
    [
        ("/rocm/whl-next/", "must not end"),
        ("/other/whl-next", "contained under /rocm"),
        ("/rocm/../whl-next", "path segment"),
        ("/rocm/whl", "must be '/rocm/whl-next'"),
    ],
)
def test_rejects_invalid_public_base(public_base: str, match: str) -> None:
    data = _valid_manifest()
    index = data["python_indexes"][0]
    assert isinstance(index, dict)
    index["public_base"] = public_base

    with pytest.raises(ManifestError, match=match):
        parse_ownership_manifest(data)


@pytest.mark.parametrize(
    "package_name, match",
    [
        ("Torch", "package name must be PEP 503 normalized"),
        ("torch_gpu", "package name must be PEP 503 normalized"),
        ("", "valid normalized Python project name"),
        ("a/b", "valid normalized Python project name"),
        ("foo bar", "valid normalized Python project name"),
        ("foo<script>", "valid normalized Python project name"),
        ('"onclick"', "valid normalized Python project name"),
        ("ünïcode", "valid normalized Python project name"),
    ],
)
def test_rejects_unnormalized_package_names(package_name: str, match: str) -> None:
    data = _valid_manifest()
    index = data["python_indexes"][0]
    assert isinstance(index, dict)
    packages = index["packages"]
    assert isinstance(packages, dict)
    packages.clear()
    packages[package_name] = {"owner_path": "pytorch/whl-next"}

    with pytest.raises(ManifestError, match=match):
        parse_ownership_manifest(data)


@pytest.mark.parametrize(
    "owner_path, match",
    [
        ("/rocm/core/whl-next", "must be relative"),
        ("core/../whl-next", "path segment"),
        ("https://example.com/rocm/core/whl-next", "URL scheme"),
        ("core//whl-next", "path segment"),
        ("core/whl next", "unsupported path segment"),
    ],
)
def test_rejects_invalid_owner_paths(owner_path: str, match: str) -> None:
    data = _valid_manifest()
    index = data["python_indexes"][0]
    assert isinstance(index, dict)
    packages = index["packages"]
    assert isinstance(packages, dict)
    packages["torch"] = {"owner_path": owner_path}

    with pytest.raises(ManifestError, match=match):
        parse_ownership_manifest(data)


def test_rejects_unknown_keys() -> None:
    data = _valid_manifest()
    data["extra"] = "unsupported"

    with pytest.raises(ManifestError, match="unknown key"):
        parse_ownership_manifest(data)


def test_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    manifest_path = tmp_path / "ownership.yaml"
    manifest_path.write_text(
        """
schema_version: 1
python_indexes:
  - public_base: /rocm/whl-next
    packages:
      torch:
        owner_path: pytorch/whl-next
      torch:
        owner_path: core/whl-next
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="duplicate key 'torch'"):
        load_ownership_manifest(manifest_path)


def test_rejects_malformed_yaml(tmp_path: Path) -> None:
    manifest_path = tmp_path / "ownership.yaml"
    manifest_path.write_text("schema_version: [", encoding="utf-8")

    with pytest.raises(ManifestError, match="while parsing"):
        load_ownership_manifest(manifest_path)


def test_rejects_unhashable_yaml_key(tmp_path: Path) -> None:
    manifest_path = tmp_path / "ownership.yaml"
    manifest_path.write_text("? [schema_version]\n: 1\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="unhashable key"):
        load_ownership_manifest(manifest_path)
