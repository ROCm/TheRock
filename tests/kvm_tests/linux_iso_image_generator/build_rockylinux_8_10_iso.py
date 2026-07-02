#!/usr/bin/env python3
"""Build a Rocky Linux 8.10 unattended ISO from build-config.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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


class BuildConfigError(Exception):
    """Raised when configuration is missing required keys."""


class BuildRuntimeError(Exception):
    """Raised when a runtime build step fails."""


def _bool_to_text(value: object) -> str:
    return str(value).lower()


def _require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise BuildRuntimeError(f"Missing required command: {command}")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _run_capture(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return f"{result.stdout}\n{result.stderr}"


def _load_config(config_path: Path) -> dict[str, object]:
    if not config_path.is_file():
        raise BuildConfigError(f"Config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as exc:
        raise BuildConfigError(f"Invalid JSON in config file {config_path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise BuildConfigError("Config root must be a JSON object.")
    return cfg


def _validate_config(cfg: dict[str, object]) -> None:
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

    if missing or empty:
        parts: list[str] = ["Invalid build-config.json:"]
        if missing:
            parts.append("Missing required keys:")
            parts.extend(f"  - {entry}" for entry in missing)
        if empty:
            parts.append("Required keys with empty values:")
            parts.extend(f"  - {entry}" for entry in empty)
        raise BuildConfigError("\n".join(parts))


def _download_file(url: str, destination: Path, retries: int = 3, delay_seconds: int = 2) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "rockylinux-iso-builder/1.0"})
            with urlopen(request, timeout=120) as response, destination.open("wb") as out_file:
                shutil.copyfileobj(response, out_file)
            return
        except (HTTPError, URLError, TimeoutError) as exc:
            if destination.exists():
                destination.unlink()
            if attempt >= retries:
                raise BuildRuntimeError(f"Failed downloading {url}: {exc}") from exc
            time.sleep(delay_seconds)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_kickstart(template_path: Path, output_path: Path, values: dict[str, str]) -> None:
    if not template_path.is_file():
        raise BuildRuntimeError(f"Kickstart template not found: {template_path}")

    content = template_path.read_text(encoding="utf-8")
    for key, value in values.items():
        content = content.replace(f"{{{{{key}}}}}", value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _detect_iso_builder() -> tuple[list[str], str]:
    if shutil.which("mkisofs"):
        return ["mkisofs"], "mkisofs"
    if shutil.which("xorrisofs"):
        return ["xorrisofs"], "xorrisofs"
    if shutil.which("xorriso"):
        return ["xorriso", "-as", "mkisofs"], "xorriso"
    raise BuildRuntimeError(
        "Neither mkisofs nor xorrisofs is installed. "
        "Install either mkisofs (genisoimage) or xorrisofs (xorriso)."
    )


def _extract_iso_tree(source_iso: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("xorriso"):
        _run(
            [
                "xorriso",
                "-osirrox",
                "on",
                "-indev",
                str(source_iso),
                "-extract",
                "/",
                str(destination_dir),
            ]
        )
        _make_tree_writable(destination_dir)
        return
    if shutil.which("bsdtar"):
        _run(["bsdtar", "-C", str(destination_dir), "-xf", str(source_iso)])
        _make_tree_writable(destination_dir)
        return
    raise BuildRuntimeError(
        "Cannot extract source ISO: install xorriso or bsdtar."
    )


def _make_tree_writable(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        try:
            current_dir.chmod(current_dir.stat().st_mode | 0o200)
        except OSError:
            pass
        for name in dirnames:
            p = current_dir / name
            try:
                p.chmod(p.stat().st_mode | 0o200)
            except OSError:
                pass
        for name in filenames:
            p = current_dir / name
            try:
                p.chmod(p.stat().st_mode | 0o200)
            except OSError:
                pass


def _append_kernel_arg(line: str, kernel_arg: str) -> str:
    if kernel_arg in line:
        return line
    if line.endswith("\n"):
        return f"{line[:-1]} {kernel_arg}\n"
    return f"{line} {kernel_arg}"


def _append_kernel_args(line: str, kernel_args: list[str]) -> str:
    updated = line
    for kernel_arg in kernel_args:
        updated = _append_kernel_arg(updated, kernel_arg)
    return updated


def _patch_bootloader_configs(staging_dir: Path, kernel_args: list[str]) -> None:
    changed_any = False

    isolinux_cfg = staging_dir / "isolinux" / "isolinux.cfg"
    if isolinux_cfg.is_file():
        lines = isolinux_cfg.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines: list[str] = []
        changed = False
        current_label = ""
        for line in lines:
            label_match = re.match(r"^\s*label\s+(\S+)\s*$", line)
            if label_match:
                current_label = label_match.group(1)
            if re.match(r"^\s*timeout\s+\d+\s*$", line):
                new_line = "timeout 50\n"
                if new_line != line:
                    changed = True
                new_lines.append(new_line)
                continue
            if re.match(r"^\s*append\s+", line):
                new_line = _append_kernel_args(line, kernel_args)
                if new_line != line:
                    changed = True
                new_lines.append(new_line)
            elif "menu default" in line and current_label == "check":
                # Remove media-check default selection for faster unattended boot.
                changed = True
                continue
            else:
                new_lines.append(line)

        # Ensure the non-media-check install entry is default in BIOS menu.
        linux_label_idx = next(
            (i for i, l in enumerate(new_lines) if re.match(r"^\s*label\s+linux\s*$", l)),
            None,
        )
        if linux_label_idx is not None:
            next_label_idx = next(
                (
                    i
                    for i in range(linux_label_idx + 1, len(new_lines))
                    if re.match(r"^\s*label\s+\S+\s*$", new_lines[i])
                ),
                len(new_lines),
            )
            has_menu_default = any(
                re.match(r"^\s*menu default\s*$", new_lines[i])
                for i in range(linux_label_idx + 1, next_label_idx)
            )
            if not has_menu_default:
                new_lines.insert(linux_label_idx + 1, "  menu default\n")
                changed = True

        if changed:
            isolinux_cfg.write_text("".join(new_lines), encoding="utf-8")
            changed_any = True

    grub_cfg_paths = [
        staging_dir / "EFI" / "BOOT" / "grub.cfg",
        staging_dir / "boot" / "grub2" / "grub.cfg",
    ]
    for grub_cfg in grub_cfg_paths:
        if not grub_cfg.is_file():
            continue
        lines = grub_cfg.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = []
        changed = False
        for line in lines:
            if re.match(r'^\s*set\s+default="?\d+"?\s*$', line):
                new_line = 'set default="0"\n'
                if new_line != line:
                    changed = True
                new_lines.append(new_line)
                continue
            if re.match(r"^\s*set\s+timeout=\d+\s*$", line):
                new_line = "set timeout=5\n"
                if new_line != line:
                    changed = True
                new_lines.append(new_line)
                continue
            if re.match(r"^\s*linux(efi)?\s+", line):
                new_line = _append_kernel_args(line, kernel_args)
                if new_line != line:
                    changed = True
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        if changed:
            grub_cfg.write_text("".join(new_lines), encoding="utf-8")
            changed_any = True

    if not changed_any:
        raise BuildRuntimeError(
            "Failed to patch bootloader configs with required kernel args."
        )


def _get_iso_volume_id(source_iso: Path) -> str:
    if shutil.which("xorriso"):
        info = _run_capture(["xorriso", "-indev", str(source_iso), "-pvd_info"])
        match = re.search(r"Volume id\s*:\s*'([^']+)'", info)
        if match:
            return match.group(1)

    if shutil.which("isoinfo"):
        info = _run_capture(["isoinfo", "-d", "-i", str(source_iso)])
        match = re.search(r"Volume id:\s*(.+)", info)
        if match:
            return match.group(1).strip()

    stem = source_iso.stem.upper()
    fallback = re.sub(r"[^A-Z0-9_]", "_", stem)[:32]
    return fallback or "ROCKY_AUTO"


def _find_isohybrid_mbr() -> str | None:
    candidates = [
        "/usr/lib/ISOLINUX/isohdpfx.bin",
        "/usr/lib/syslinux/isohdpfx.bin",
        "/usr/lib/syslinux/modules/bios/isohdpfx.bin",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


def _build_iso_with_builder(
    builder_prefix: list[str],
    builder_name: str,
    staging_dir: Path,
    output_iso_path: Path,
    volume_id: str,
) -> None:
    command = builder_prefix + [
        "-o",
        str(output_iso_path),
        "-V",
        volume_id,
        "-J",
        "-joliet-long",
        "-R",
        "-T",
        "-eltorito-boot",
        "isolinux/isolinux.bin",
        "-eltorito-catalog",
        "isolinux/boot.cat",
        "-no-emul-boot",
        "-boot-load-size",
        "4",
        "-boot-info-table",
        "-eltorito-alt-boot",
        "-e",
        "images/efiboot.img",
        "-no-emul-boot",
    ]

    # xorriso/xorrisofs support isohybrid flags; classic mkisofs typically does not.
    if builder_name in {"xorrisofs", "xorriso"}:
        command.append("-isohybrid-gpt-basdat")
        isohybrid_mbr = _find_isohybrid_mbr()
        if isohybrid_mbr is not None:
            command.extend(["-isohybrid-mbr", isohybrid_mbr])

    command.append(str(staging_dir))

    if output_iso_path.exists():
        output_iso_path.unlink()

    _run(command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Rocky Linux 8.10 unattended ISO.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="./build-config.json",
        help="Path to JSON config file (default: ./build-config.json)",
    )
    args = parser.parse_args()

    try:
        _require_command("ssh-keygen")
        builder_prefix, builder_name = _detect_iso_builder()

        config_path = Path(args.config_path)
        cfg = _load_config(config_path)
        _validate_config(cfg)

        rocky_version = str(cfg["rocky_version"])
        source_iso_url = str(cfg["source_iso_url"])
        source_iso_sha256 = str(cfg["source_iso_sha256"])
        source_iso_path = Path(str(cfg["source_iso_path"]))
        output_iso_path = Path(str(cfg["output_iso_path"]))
        kickstart_template = Path(str(cfg["kickstart_template"]))
        kickstart_output = Path(str(cfg["kickstart_output"]))
        lang_value = str(cfg["lang"])
        keyboard = str(cfg["keyboard"])
        timezone = str(cfg["timezone"])
        hostname = str(cfg["hostname"])
        user_name = str(cfg["user_name"])
        disable_user_password_auth = _bool_to_text(cfg["disable_user_password_auth"])
        disable_sudo_password_prompt = _bool_to_text(cfg["disable_sudo_password_prompt"])
        enable_sshd = _bool_to_text(cfg["enable_sshd"])
        allow_ping_icmp = _bool_to_text(cfg["allow_ping_icmp"])
        generate_ssh_key_if_missing = _bool_to_text(cfg["generate_ssh_key_if_missing"])
        ssh_authorized_public_key = str(cfg["ssh_authorized_public_key"])
        ssh_private_key_path = Path(str(cfg["ssh_private_key_path"]))
        ssh_public_key_path = Path(str(cfg["ssh_public_key_path"]))

        output_iso_path.parent.mkdir(parents=True, exist_ok=True)
        kickstart_output.parent.mkdir(parents=True, exist_ok=True)
        ssh_private_key_path.parent.mkdir(parents=True, exist_ok=True)
        ssh_public_key_path.parent.mkdir(parents=True, exist_ok=True)

        if ssh_authorized_public_key.strip() == "":
            if not ssh_private_key_path.is_file() or not ssh_public_key_path.is_file():
                if generate_ssh_key_if_missing != "true":
                    raise BuildRuntimeError(
                        "SSH key files are missing and auto-generation is disabled.\n"
                        f"Expected private key: {ssh_private_key_path}\n"
                        f"Expected public key : {ssh_public_key_path}"
                    )
                print(f"Generating passwordless SSH keypair for user {user_name}...")
                _run(
                    [
                        "ssh-keygen",
                        "-q",
                        "-t",
                        "ed25519",
                        "-N",
                        "",
                        "-C",
                        f"{user_name}@{hostname}",
                        "-f",
                        str(ssh_private_key_path),
                    ]
                )
            else:
                print("Using existing SSH keypair:")
                print(f"  Private: {ssh_private_key_path}")
                print(f"  Public : {ssh_public_key_path}")
            ssh_authorized_public_key = ssh_public_key_path.read_text(encoding="utf-8").strip()
        else:
            print("Using ssh_authorized_public_key from config.")

        if not ssh_authorized_public_key.startswith("ssh-"):
            raise BuildRuntimeError("Configured SSH public key does not look valid.")

        if source_iso_path.is_file():
            print(f"Using cached source ISO: {source_iso_path}")
        else:
            print(f"Downloading Rocky Linux {rocky_version} ISO...")
            _download_file(source_iso_url, source_iso_path)

        if source_iso_sha256.strip():
            print("Verifying source ISO checksum...")
            actual_sha256 = _sha256_file(source_iso_path)
            if actual_sha256 != source_iso_sha256:
                raise BuildRuntimeError(
                    "Checksum mismatch for source ISO\n"
                    f"Expected: {source_iso_sha256}\n"
                    f"Actual  : {actual_sha256}"
                )

        print("Rendering kickstart file...")
        volume_id = _get_iso_volume_id(source_iso_path)
        _render_kickstart(
            kickstart_template,
            kickstart_output,
            {
                "LANG": lang_value,
                "KEYBOARD": keyboard,
                "TIMEZONE": timezone,
                "HOSTNAME": hostname,
                "USER_NAME": user_name,
                "DISABLE_SUDO_PASSWORD_PROMPT": disable_sudo_password_prompt,
                "SSH_AUTHORIZED_PUBLIC_KEY": ssh_authorized_public_key,
                "ENABLE_SSHD": enable_sshd,
                "ALLOW_PING_ICMP": allow_ping_icmp,
                "DISABLE_USER_PASSWORD_AUTH": disable_user_password_auth,
            },
        )

        print("Rebuilding unattended ISO image...")
        print(f"Using ISO builder: {builder_name}")
        with tempfile.TemporaryDirectory(prefix="rocky-iso-build-") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            staging_dir = tmp_dir / "iso-root"
            _extract_iso_tree(source_iso_path, staging_dir)

            (staging_dir / "ks.cfg").write_text(
                kickstart_output.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            _patch_bootloader_configs(
                staging_dir,
                [
                    "inst.ks=cdrom:/ks.cfg",
                    "inst.text",
                    "console=ttyS0,115200n8",
                ],
            )
            _build_iso_with_builder(
                builder_prefix=builder_prefix,
                builder_name=builder_name,
                staging_dir=staging_dir,
                output_iso_path=output_iso_path,
                volume_id=volume_id,
            )

        print()
        print("Build completed successfully.")
        print(f"Source ISO : {source_iso_path}")
        print(f"Kickstart  : {kickstart_output}")
        print(f"Output ISO : {output_iso_path}")
        print(f"SSH private key for {user_name}: {ssh_private_key_path}")
        print(f"SSH public key for {user_name} : {ssh_public_key_path}")
        return 0
    except (BuildConfigError, BuildRuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
