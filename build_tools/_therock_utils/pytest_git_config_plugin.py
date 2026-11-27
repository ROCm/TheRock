# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Isolate Git configuration across TheRock's pytest suites."""

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Isolate fixture repositories from host Git settings and identities."""
    monkeypatch = pytest.MonkeyPatch()
    config.add_cleanup(monkeypatch.undo)

    # Ignore machine-wide Git settings so tests behave consistently across hosts.
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    # Ignore user Git settings, including author identities, hooks, and commit
    # signing that may require access to a hardware key.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)

    # Supply both identities so fixture commits do not inherit identity variables
    # or derive names and email addresses from the host account. Tests can override
    # these values when they need specific authors or committers.
    monkeypatch.setenv("GIT_AUTHOR_NAME", "TheRock Tests")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "TheRock Tests")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.com")
