# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import update_dependencies
from update_dependencies import (
    PACKAGES_PER_PROJECT,
    _run_structured,
    get_dependency_package_names,
    get_project_paths,
    get_selected_packages,
    is_wheel_allowed,
    main,
    normalize_package_name,
    resolve_target_prefixes,
    structured_dependency_key,
)


class FakeBucket:
    name = "test-bucket"


# ---------------------------------------------------------------------------
# Allowed wheels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pkg",
    [
        # linux_x86_64, various supported CPython versions
        "numpy-2.0.0-cp310-cp310-linux_x86_64.whl",
        "numpy-2.0.0-cp311-cp311-linux_x86_64.whl",
        "numpy-2.0.0-cp312-cp312-linux_x86_64.whl",
        "numpy-2.0.0-cp313-cp313-linux_x86_64.whl",
        "numpy-2.0.0-cp314-cp314-linux_x86_64.whl",
        # manylinux variants
        "numpy-2.0.0-cp310-cp310-manylinux_2_17_x86_64.whl",
        "numpy-2.0.0-cp312-cp312-manylinux2014_x86_64.whl",
        "pillow-10.0.0-cp311-cp311-manylinux_2_28_x86_64.whl",
        # pure-Python / platform-independent
        "sympy-1.13.0-py3-none-any.whl",
        "filelock-3.15.0-py3-none-any.whl",
        # Windows x64 — was not excluded by the old blacklist
        "torch-2.3.0-cp312-cp312-win_amd64.whl",
        "numpy-2.0.0-cp310-cp310-win_amd64.whl",
    ],
)
def test_allowed(pkg: str) -> None:
    assert is_wheel_allowed(pkg), f"Expected allowed: {pkg}"


# ---------------------------------------------------------------------------
# Rejected platform tags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pkg",
    [
        # win32 and win_arm64
        "numpy-2.0.0-cp312-cp312-win32.whl",
        "numpy-2.0.0-cp312-cp312-win_arm64.whl",
        # musllinux
        "numpy-2.0.0-cp312-cp312-musllinux_1_1_x86_64.whl",
        "numpy-2.0.0-cp312-cp312-musllinux_1_2_aarch64.whl",
        # macOS — including the tricky _x86_64 suffix variant
        "numpy-2.0.0-cp312-cp312-macosx_10_9_x86_64.whl",
        "numpy-2.0.0-cp312-cp312-macosx_11_0_arm64.whl",
        "numpy-2.0.0-cp312-cp312-macosx_12_0_universal2.whl",
        # aarch64 / ARM
        "numpy-2.0.0-cp312-cp312-manylinux_2_17_aarch64.whl",
        "numpy-2.0.0-cp312-cp312-linux_aarch64.whl",
        # i686 / 32-bit x86
        "numpy-2.0.0-cp312-cp312-manylinux_2_17_i686.whl",
        "numpy-2.0.0-cp312-cp312-linux_i686.whl",
        # iOS
        "numpy-2.0.0-cp312-cp312-iphoneos_17_0_arm64.whl",
        "numpy-2.0.0-cp312-cp312-iphonesimulator_17_0_x86_64.whl",
        # RISC-V
        "numpy-2.0.0-cp312-cp312-linux_riscv64.whl",
    ],
)
def test_rejected_platform(pkg: str) -> None:
    assert not is_wheel_allowed(pkg), f"Expected rejected: {pkg}"


# ---------------------------------------------------------------------------
# Rejected Python tags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pkg",
    [
        # Too old
        "numpy-1.26.0-cp39-cp39-linux_x86_64.whl",
        # PyPy
        "numpy-2.0.0-pp310-pypy310_pp73-linux_x86_64.whl",
        "numpy-2.0.0-pp310-pypy310_pp73-manylinux_2_17_x86_64.whl",
        # Free-threaded and future versions
        "numpy-2.0.0-cp313t-cp313t-linux_x86_64.whl",
        "numpy-2.0.0-cp314t-cp314t-linux_x86_64.whl",
        # Python 2 and py2.py3 universal tags
        "six-1.16.0-py2-none-any.whl",
        "six-1.16.0-py2.py3-none-any.whl",
    ],
)
def test_rejected_python(pkg: str) -> None:
    assert not is_wheel_allowed(pkg), f"Expected rejected: {pkg}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pkg",
    [
        # Not a wheel
        "numpy-2.0.0.tar.gz",
        "numpy-2.0.0-cp312-cp312-linux_x86_64.zip",
        # Malformed (too few parts)
        "numpy-2.0.0-linux_x86_64.whl",
        # Empty string
        "",
    ],
)
def test_rejected_non_wheel_or_malformed(pkg: str) -> None:
    assert not is_wheel_allowed(pkg), f"Expected rejected: {pkg}"


# ---------------------------------------------------------------------------
# Package/project helpers
# ---------------------------------------------------------------------------


def test_normalize_package_name() -> None:
    assert normalize_package_name("ml_dtypes") == "ml-dtypes"
    assert normalize_package_name("typing_extensions") == "typing-extensions"
    assert normalize_package_name("MarkupSafe") == "markupsafe"
    assert normalize_package_name("foo.bar_baz") == "foo-bar-baz"


def test_get_project_paths() -> None:
    assert get_project_paths() == ["jax", "rocm", "torch"]


def test_get_dependency_package_names() -> None:
    assert "setuptools" in get_dependency_package_names("rocm")
    assert "jinja2" in get_dependency_package_names("torch")
    assert "ml_dtypes" in get_dependency_package_names("jax")


# ---------------------------------------------------------------------------
# Prefix resolution
# ---------------------------------------------------------------------------


def test_resolve_target_prefixes_explicit_prefix() -> None:
    assert resolve_target_prefixes(
        bucket=FakeBucket(),
        explicit_prefix="v4/whl/",
    ) == ["v4/whl"]


def test_resolve_target_prefixes_requires_prefix_or_auto_detect() -> None:
    with pytest.raises(
        RuntimeError,
        match="Must provide either --prefix or --auto-detect-prefixes with --base-prefix",
    ):
        resolve_target_prefixes(bucket=FakeBucket())


def test_resolve_target_prefixes_base_prefix_requires_auto_detect() -> None:
    with pytest.raises(
        RuntimeError,
        match="--auto-detect-prefixes must be provided when using --base-prefix",
    ):
        resolve_target_prefixes(
            bucket=FakeBucket(),
            base_prefix="v2/",
        )


def test_resolve_target_prefixes_auto_detect_requires_base_prefix() -> None:
    with pytest.raises(
        RuntimeError,
        match="--base-prefix must be provided when using --auto-detect-prefixes",
    ):
        resolve_target_prefixes(
            bucket=FakeBucket(),
            auto_detect_prefixes=True,
        )


# ---------------------------------------------------------------------------
# structured_dependency_key
# ---------------------------------------------------------------------------


def test_structured_dependency_key_composition() -> None:
    key = structured_dependency_key(
        index="whl",
        pkg_name="numpy",
        filename="numpy-2.0.0-cp312-cp312-linux_x86_64.whl",
    )
    assert key == "v5/core/whl/numpy/numpy-2.0.0-cp312-cp312-linux_x86_64.whl"


def test_structured_dependency_key_pure_python() -> None:
    key = structured_dependency_key(
        "whl", "networkx", "networkx-3.4.2-py3-none-any.whl"
    )
    assert key == "v5/core/whl/networkx/networkx-3.4.2-py3-none-any.whl"


def test_structured_dependency_key_whl_next() -> None:
    key = structured_dependency_key(
        "whl-next", "numpy", "numpy-2.0.0-cp312-cp312-linux_x86_64.whl"
    )
    assert key == "v5/core/whl-next/numpy/numpy-2.0.0-cp312-cp312-linux_x86_64.whl"


def test_structured_dependency_key_underscore_name_dashed_dir() -> None:
    # The package directory is dashed; the wheel filename keeps underscores.
    key = structured_dependency_key(
        "whl", "ml_dtypes", "ml_dtypes-0.5.0-cp312-cp312-linux_x86_64.whl"
    )
    assert key == "v5/core/whl/ml-dtypes/ml_dtypes-0.5.0-cp312-cp312-linux_x86_64.whl"


def test_structured_dependency_key_rejects_bad_index() -> None:
    with pytest.raises(ValueError, match="index="):
        structured_dependency_key("wheels", "numpy", "numpy-2.0.0.whl")


# ---------------------------------------------------------------------------
# get_selected_packages
# ---------------------------------------------------------------------------


def test_get_selected_packages_all() -> None:
    selected = get_selected_packages(package="all")
    assert selected == PACKAGES_PER_PROJECT
    # Packages from every project are present.
    projects = {info["project"] for info in selected.values()}
    assert projects == {"jax", "rocm", "torch"}


def test_get_selected_packages_filters_by_project() -> None:
    selected = get_selected_packages(package="torch")
    assert selected
    assert all(info["project"] == "torch" for info in selected.values())
    assert "numpy" in selected
    # jax-only deps must be excluded.
    assert "ml_dtypes" not in selected


def test_get_selected_packages_unknown_project_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported package"):
        get_selected_packages(package="nope")


def test_get_selected_packages_dependency_filter() -> None:
    selected = get_selected_packages(
        package="torch", dependency_names=frozenset({"numpy"})
    )
    assert list(selected) == ["numpy"]


def test_get_selected_packages_dependency_filter_across_all() -> None:
    selected = get_selected_packages(
        package="all", dependency_names=frozenset({"ml_dtypes", "numpy"})
    )
    assert set(selected) == {"ml_dtypes", "numpy"}


def test_get_selected_packages_unknown_dependency_raises() -> None:
    with pytest.raises(ValueError, match="Unknown --dependency-package"):
        get_selected_packages(
            package="torch", dependency_names=frozenset({"not-a-dep"})
        )


# ---------------------------------------------------------------------------
# _run_structured early validation
# ---------------------------------------------------------------------------


def test_run_structured_rejects_bad_index() -> None:
    # Guards programmatic callers that bypass argparse choices: fail fast
    # before any network/upload work rather than mid-run.
    with pytest.raises(ValueError, match="index="):
        _run_structured(
            bucket=FakeBucket(),
            selected_packages={"numpy": PACKAGES_PER_PROJECT["numpy"]},
            index="wheels",
            dry_run=True,
            only_pypi=False,
        )


# ---------------------------------------------------------------------------
# CLI validation: structured mode rejects flat-only prefix flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--prefix", "v4/whl"],
        ["--auto-detect-prefixes"],
        ["--base-prefix", "v2/"],
    ],
)
def test_structured_rejects_flat_prefix_flags(
    monkeypatch: pytest.MonkeyPatch, extra_args: list[str]
) -> None:
    argv = ["prog", "--structured", "--bucket", "b"] + extra_args
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        main()


def test_structured_defaults_package_to_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(update_dependencies, "run_update_dependencies", fake_run)
    monkeypatch.setattr(sys, "argv", ["prog", "--structured", "--bucket", "b"])
    main()
    assert captured["package"] == "all"
    assert captured["structured"] is True
    assert captured["index"] == "whl"


def test_flat_defaults_package_to_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(update_dependencies, "run_update_dependencies", fake_run)
    monkeypatch.setattr(sys, "argv", ["prog", "--bucket", "b", "--prefix", "v4/whl"])
    main()
    assert captured["package"] == "torch"
    assert captured["structured"] is False
