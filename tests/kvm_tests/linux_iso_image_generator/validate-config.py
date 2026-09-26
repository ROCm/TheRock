#!/usr/bin/env python3
"""Validate Rocky ISO build configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REQUIRED_KEYS = [
    "rocky_version",
    "source_iso_url",
    "source_iso_sha256",
    "source_iso_path",
    "output_iso_path",
    "kickstart_template",
    "kickstart_output",
    "lang",
    "keyboard",
    "timezone",
    "hostname",
    "user_name",
    "disable_user_password_auth",
    "disable_sudo_password_prompt",
    "enable_sshd",
    "allow_ping_icmp",
    "generate_ssh_key_if_missing",
    "ssh_authorized_public_key",
    "ssh_private_key_path",
    "ssh_public_key_path",
]

ALLOW_EMPTY_STRING = {"source_iso_sha256", "ssh_authorized_public_key"}


def validate_config(cfg: dict[str, object]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    empty: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in cfg:
            missing.append(key)
            continue

        value = cfg[key]
        if value is None:
            missing.append(key)
            continue

        if isinstance(value, str) and key not in ALLOW_EMPTY_STRING and value.strip() == "":
            empty.append(key)
    return missing, empty


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate build-rockylinux-8_10-config.json required keys for Rocky ISO automation."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default="./build-rockylinux-8_10-config.json",
        help="Path to JSON config file (default: ./build-rockylinux-8_10-config.json)",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if not config_path.is_file():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in config file: {config_path}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    if not isinstance(cfg, dict):
        print("Config root must be a JSON object.", file=sys.stderr)
        return 1

    missing, empty = validate_config(cfg)
    if missing or empty:
        print("Invalid build-rockylinux-8_10-config.json:", file=sys.stderr)
        if missing:
            print("Missing required keys:", file=sys.stderr)
            for entry in missing:
                print(f"  - {entry}", file=sys.stderr)
        if empty:
            print("Required keys with empty values:", file=sys.stderr)
            for entry in empty:
                print(f"  - {entry}", file=sys.stderr)
        return 1

    print(f"Configuration is valid: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
