# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for explicit phase selection in run_resource_probe_phase.py."""

import argparse
from pathlib import Path
import sys

import pytest

import run_resource_probe_phase as phase_wrapper


def _args(*, enabled: str, selected: frozenset[str], phase: str) -> argparse.Namespace:
    return argparse.Namespace(
        enabled=enabled,
        selected_phases=selected,
        phase=phase,
        interval_seconds=5.0,
        output_dir=Path("telemetry"),
        storage_path=Path("workspace"),
        detail_profile="diagnostic",
        command=["cmake", "--build", "build"],
    )


def test_disabled_executes_original_command():
    args = _args(
        enabled="false", selected=frozenset({"build-stage"}), phase="build-stage"
    )
    assert phase_wrapper.command_for(args) == args.command


def test_unselected_phase_executes_original_command():
    args = _args(enabled="true", selected=frozenset({"configure"}), phase="build-stage")
    assert phase_wrapper.command_for(args) == args.command


def test_selected_phase_builds_probe_command():
    args = _args(
        enabled="true", selected=frozenset({"build-stage"}), phase="build-stage"
    )
    command = phase_wrapper.command_for(args)
    assert command[0] == sys.executable
    assert command[1].endswith("resource_probe.py")
    assert command[-3:] == ["cmake", "--build", "build"]
    assert command[command.index("--phase") + 1] == "build-stage"
    assert command[command.index("--detail-profile") + 1] == "diagnostic"


def test_phase_list_rejects_unknown_or_non_string_values():
    with pytest.raises(argparse.ArgumentTypeError):
        phase_wrapper.parse_selected_phases('["build-stage", "secret-phase"]')
    with pytest.raises(argparse.ArgumentTypeError):
        phase_wrapper.parse_selected_phases('["build-stage", 3]')
