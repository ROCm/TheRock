# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Source override routing in fetch_sources, with Git operations mocked."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import fetch_sources


TEST_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SUBMODULE_PATHS = {
    "rocm-systems": "rocm-systems",
    "rocm-libraries": "rocm-libraries",
    "rocgdb": "debug-tools/rocgdb/source",
}


@pytest.fixture
def args() -> SimpleNamespace:
    return SimpleNamespace(
        remote=False,
        update_submodules=True,
        depth=None,
        progress=False,
        jobs=None,
        dvc_projects=[],
        apply_patches=False,
        external_repo_config=None,
    )


@pytest.fixture
def git_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Stub Git I/O; keep the override and ordinary update logic under test."""
    monkeypatch.setattr(fetch_sources, "THEROCK_DIR", tmp_path)
    monkeypatch.setattr(fetch_sources, "ALWAYS_SUBMODULE_PATHS", [])
    monkeypatch.setattr(
        fetch_sources,
        "get_enabled_sources",
        Mock(return_value=(list(SUBMODULE_PATHS), [])),
    )
    monkeypatch.setattr(
        fetch_sources, "get_submodule_path", SUBMODULE_PATHS.__getitem__
    )
    monkeypatch.setattr(fetch_sources, "resolve_reference_dir", Mock(return_value=None))
    gitmodules = "\n".join(
        f"submodule.{name}.path {path}" for name, path in SUBMODULE_PATHS.items()
    )

    def read_git_output(command: list[str], *, cwd: Path, text: bool) -> str:
        if command[:2] == ["git", "config"]:
            return gitmodules
        if command == ["git", "rev-parse", "HEAD"]:
            return TEST_COMMIT
        pytest.fail(f"Unexpected Git read: {command}")

    monkeypatch.setattr(fetch_sources.subprocess, "check_output", read_git_output)
    commands = Mock()
    monkeypatch.setattr(fetch_sources, "run_command", commands)
    return commands


def test_no_config_uses_normal_submodule_update(
    args: SimpleNamespace, git_commands: Mock
) -> None:
    fetch_sources.run(args)

    # All submodules should be initialized normally.
    git_commands.assert_any_call(
        ["git", "submodule", "update", "--init", "--"] + list(SUBMODULE_PATHS.values()),
        cwd=fetch_sources.THEROCK_DIR,
    )
    # No fancy cache/config commands to change submodule pins.
    commands = [invocation.args[0] for invocation in git_commands.call_args_list]
    assert not any("--cacheinfo" in command for command in commands)
    assert not any(command[:2] == ["git", "config"] for command in commands)


def test_only_configured_submodule_gets_new_pin(
    args: SimpleNamespace, git_commands: Mock
) -> None:
    args.external_repo_config = {
        "repository": "ROCm/rocm-systems",
        "ref": TEST_COMMIT,
        "submodule_path": "rocm-systems",
    }

    fetch_sources.run(args)

    # Only the configured submodule gets a new pin; all remotes stay unchanged.
    pin_updates = [
        invocation.args[0]
        for invocation in git_commands.call_args_list
        if "--cacheinfo" in invocation.args[0]
    ]
    assert pin_updates == [
        ["git", "update-index", "--cacheinfo", f"160000,{TEST_COMMIT},rocm-systems"]
    ]
    commands = [invocation.args[0] for invocation in git_commands.call_args_list]
    assert not any(command[:2] == ["git", "config"] for command in commands)
    assert not any(command[:2] == ["git", "remote"] for command in commands)

    # All selected submodules then use `git submodule update --init` with indexed pins.
    git_commands.assert_any_call(
        ["git", "submodule", "update", "--init", "--"] + list(SUBMODULE_PATHS.values()),
        cwd=fetch_sources.THEROCK_DIR,
    )

    # The pin override must take effect before the checkout.
    update_index = next(
        i
        for i, command in enumerate(commands)
        if command[:3] == ["git", "submodule", "update"]
    )
    assert commands.index(pin_updates[0]) < update_index


def test_unselected_override_does_not_add_a_checkout(
    args: SimpleNamespace,
    git_commands: Mock,
) -> None:
    fetch_sources.get_enabled_sources.return_value = (["rocm-libraries"], [])

    args.external_repo_config = {
        "repository": "ROCm/rocm-systems",
        "ref": TEST_COMMIT,
        "submodule_path": "rocm-systems",
    }
    fetch_sources.run(args)

    # Stage the systems pin without adding systems to the selected checkouts.
    commands = [invocation.args[0] for invocation in git_commands.call_args_list]
    checkouts = [command for command in commands if command[:2] == ["git", "submodule"]]
    assert checkouts == [
        ["git", "submodule", "update", "--init", "--", "rocm-libraries"]
    ]
    assert not any(command[:2] == ["git", "config"] for command in commands)
    git_commands.assert_any_call(
        ["git", "update-index", "--cacheinfo", f"160000,{TEST_COMMIT},rocm-systems"],
        cwd=fetch_sources.THEROCK_DIR,
    )


@pytest.mark.parametrize(
    "field,value", [("ref", "main"), ("submodule_path", "missing")]
)
def test_invalid_override_does_not_change_sources(
    git_commands: Mock, field: str, value: str
) -> None:
    config = {
        "ref": TEST_COMMIT,
        "submodule_path": "rocm-systems",
    }
    config[field] = value

    with pytest.raises(ValueError):
        fetch_sources.apply_external_repo_override(config)

    git_commands.assert_not_called()


@pytest.mark.parametrize("explicit_config", [False, True])
def test_cli_passes_decoded_config_through_args(
    explicit_config: bool,
) -> None:
    config = {
        "ref": TEST_COMMIT,
        "submodule_path": "rocm-systems",
    }
    requested = {**config, "ref": "a" * 40}
    argv = ["--external-repo-config", json.dumps(requested)] if explicit_config else []
    with (
        patch("fetch_sources.os.getenv", return_value=json.dumps(config)) as getenv,
        patch("fetch_sources.run") as run,
    ):
        fetch_sources.main(argv)

    getenv.assert_called_once_with("EXTERNAL_REPO_CONFIG")
    assert run.call_args.args[0].external_repo_config == (
        requested if explicit_config else config
    )
