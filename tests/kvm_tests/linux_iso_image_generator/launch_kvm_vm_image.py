#!/usr/bin/env python3
"""Launch a Rocky Linux qcow2 VM with QEMU/KVM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(f"Missing command: {command}")


def _load_pci_ids(config_path: Path) -> list[str]:
    if not config_path.is_file():
        raise RuntimeError(f"Config file not found: {config_path}")

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in config file {config_path}: {exc}") from exc

    if not isinstance(cfg, dict):
        raise RuntimeError("Config root must be a JSON object.")

    pci_ids_raw = cfg.get("pci_passthrough_device_ids", [])
    if isinstance(pci_ids_raw, str):
        pci_ids = [pci_ids_raw.strip()] if pci_ids_raw.strip() else []
    elif isinstance(pci_ids_raw, list):
        pci_ids = [str(entry).strip() for entry in pci_ids_raw if str(entry).strip()]
    else:
        pci_ids = []
    return pci_ids


def _validate_pci_devices(pci_ids: list[str]) -> None:
    if not Path("/dev/vfio/vfio").exists():
        raise RuntimeError("/dev/vfio/vfio not present. VFIO may not be enabled on host.")

    for pci_id in pci_ids:
        device_path = Path(f"/sys/bus/pci/devices/{pci_id}")
        if not device_path.is_dir():
            raise RuntimeError(f"PCI device not found: {pci_id}")

        driver_link = device_path / "driver"
        driver_name = "unbound"
        if driver_link.is_symlink():
            driver_name = Path(os.readlink(driver_link)).name
        if driver_name != "vfio-pci":
            raise RuntimeError(
                f"PCI device {pci_id} is bound to '{driver_name}', expected 'vfio-pci'."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch a Rocky Linux qcow2 image in QEMU/KVM."
    )
    parser.add_argument(
        "image",
        nargs="?",
        metavar="QCOW_IMAGE",
        default="./output/Rocky-8.10-x86_64-autoinstall.qcow2",
        help=(
            "Optional positional path to qcow2 image "
            "(default: ./output/Rocky-8.10-x86_64-autoinstall.qcow2)"
        ),
    )
    parser.add_argument(
        "--pci-p",
        action="store_true",
        help=(
            "Use PCI passthrough mode (alias: --pci-p; "
            "reads pci_passthrough_device_ids from "
            "./build-rockylinux-8_10-config.json)."
        ),
    )
    args = parser.parse_args()

    qcow_path = Path(args.image)
    ram_mb = _env("RAM_MB", "4096")
    cpus = _env("CPUS", "2")
    ssh_fwd_port = _env("SSH_FWD_PORT", "2222")
    vm_name = _env("VM_NAME", "rocky810-qcow")
    headless = _env("HEADLESS", "0")
    accel_mode = _env("ACCEL_MODE", "kvm" if args.pci_passthrough else "kvm:tcg")

    try:
        _require_command("qemu-system-x86_64")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not qcow_path.is_file():
        print(f"QCOW image not found: {qcow_path}", file=sys.stderr)
        print(
            "Build it first with: "
            "python3 ./build_kvm_vm_image.py ./build-rockylinux-8_10-config.json",
            file=sys.stderr,
        )
        return 1

    vfio_args: list[str] = []
    if args.pci_passthrough:
        config_path = Path("./build-rockylinux-8_10-config.json")
        try:
            pci_ids = _load_pci_ids(config_path)
            if not pci_ids:
                print("No PCI IDs configured.", file=sys.stderr)
                print(
                    "Set build-rockylinux-8_10-config.json key: "
                    "pci_passthrough_device_ids",
                    file=sys.stderr,
                )
                print(
                    'Example: "pci_passthrough_device_ids": '
                    '["0000:01:00.0", "0000:01:00.1"]',
                    file=sys.stderr,
                )
                return 1
            _validate_pci_devices(pci_ids)
            for pci_id in pci_ids:
                vfio_args.extend(["-device", f"vfio-pci,host={pci_id}"])
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    print(f"Launching VM '{vm_name}'")
    print(f"Disk: {qcow_path}")
    print(f"SSH:  host 127.0.0.1:{ssh_fwd_port} -> guest :22")
    if args.pci_passthrough:
        print("Mode: PCI passthrough")
    else:
        print("Mode: Virtual display (no PCI passthrough)")
    print()
    if args.pci_passthrough:
        print("Using passthrough devices from build config.")
    else:
        print("No GPU PCI passthrough is used. VM display is virtual only.")
    print()
    print("Connect with:")
    print("ssh -i ./output/ssh/therock_ed25519 \\")
    print("  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \\")
    print(f"  -p {ssh_fwd_port} therock@127.0.0.1")

    command = [
        "qemu-system-x86_64",
        "-name",
        vm_name,
        "-machine",
        f"{'q35,' if args.pci_passthrough else ''}accel={accel_mode}",
        "-m",
        ram_mb,
        "-smp",
        cpus,
        "-drive",
        f"file={qcow_path},if=virtio,format=qcow2",
        "-netdev",
        f"user,id=n1,hostfwd=tcp::{ssh_fwd_port}-:22",
        "-device",
        "virtio-net-pci,netdev=n1",
    ]
    if args.pci_passthrough:
        command.extend(["-cpu", "host", "-vga", "none", "-display", "none", "-serial", "mon:stdio"])
        command.extend(vfio_args)
    elif headless == "1":
        command.extend(["-display", "none", "-serial", "mon:stdio"])
    else:
        command.extend(["-device", "virtio-vga", "-display", "gtk,gl=off"])

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
