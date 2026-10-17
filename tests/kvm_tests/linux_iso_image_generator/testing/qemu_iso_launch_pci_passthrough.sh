#!/usr/bin/env bash
set -euo pipefail

# qemu_iso_launch_pci_passthrough.sh
# Launch Rocky Linux autoinstall ISO with QEMU/KVM using PCI passthrough.
#
# PCI IDs are read from build-rockylinux-8_10-config.json key:
#   "pci_passthrough_device_ids": ["0000:01:00.0", "0000:01:00.1"]
#
# Usage:
#   ./testing/qemu_iso_launch_pci_passthrough.sh [CONFIG_PATH] [ISO_PATH] [DISK_PATH]
#
# Optional environment variables:
#   RAM_MB=8192                Memory for VM in MB
#   CPUS=4                     Number of virtual CPUs
#   DISK_SIZE=60G              Created qcow2 size when DISK_PATH is missing
#   SSH_FWD_PORT=2222          Host TCP port forwarded to guest 22
#   VM_NAME=rocky810-pci-test  VM name shown by QEMU
#   ACCEL_MODE=kvm             Acceleration mode, e.g. kvm or kvm:tcg
#
# Notes:
# - This script requires the listed PCI IDs to already be bound to vfio-pci.
# - No virtual GPU is added (`-vga none`), because display is expected via passthrough GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${1:-${REPO_ROOT}/build-rockylinux-8_10-config.json}"
ISO_PATH_OVERRIDE="${2:-}"
DISK_PATH="${3:-${REPO_ROOT}/output/rocky810-test-pci.qcow2}"

RAM_MB="${RAM_MB:-8192}"
CPUS="${CPUS:-4}"
DISK_SIZE="${DISK_SIZE:-60G}"
SSH_FWD_PORT="${SSH_FWD_PORT:-2222}"
VM_NAME="${VM_NAME:-rocky810-pci-test}"
ACCEL_MODE="${ACCEL_MODE:-kvm}"

to_abs_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf "%s" "${value}"
  else
    printf "%s/%s" "${REPO_ROOT}" "${value#./}"
  fi
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "Missing command: python3" >&2
  exit 1
fi

if ! command -v qemu-system-x86_64 >/dev/null 2>&1; then
  echo "Missing command: qemu-system-x86_64" >&2
  exit 1
fi

if ! command -v qemu-img >/dev/null 2>&1; then
  echo "Missing command: qemu-img" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

eval "$(
python3 - "${CONFIG_PATH}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
if not isinstance(cfg, dict):
    raise SystemExit("Config root must be a JSON object.")

pci_ids_raw = cfg.get("pci_passthrough_device_ids", [])
if isinstance(pci_ids_raw, str):
    pci_ids = [pci_ids_raw.strip()] if pci_ids_raw.strip() else []
elif isinstance(pci_ids_raw, list):
    pci_ids = [str(entry).strip() for entry in pci_ids_raw if str(entry).strip()]
else:
    pci_ids = []

output_iso = str(cfg.get("output_iso_path", "./output/Rocky-8.10-x86_64-autoinstall.iso"))
ssh_private_key = str(cfg.get("ssh_private_key_path", "./output/ssh/therock_ed25519"))

print(f"CFG_OUTPUT_ISO={shlex.quote(output_iso)}")
print(f"CFG_SSH_PRIVATE_KEY={shlex.quote(ssh_private_key)}")
print(f"CFG_PCI_IDS={shlex.quote(','.join(pci_ids))}")
PY
)"

if [[ -z "${CFG_PCI_IDS}" ]]; then
  echo "No PCI IDs configured." >&2
  echo "Set build-rockylinux-8_10-config.json key: pci_passthrough_device_ids" >&2
  echo "Example: \"pci_passthrough_device_ids\": [\"0000:01:00.0\", \"0000:01:00.1\"]" >&2
  exit 1
fi

ISO_PATH="${ISO_PATH_OVERRIDE:-${CFG_OUTPUT_ISO}}"
ISO_PATH="$(to_abs_path "${ISO_PATH}")"
DISK_PATH="$(to_abs_path "${DISK_PATH}")"
SSH_KEY_PATH="$(to_abs_path "${CFG_SSH_PRIVATE_KEY}")"

if [[ ! -f "${ISO_PATH}" ]]; then
  echo "ISO not found: ${ISO_PATH}" >&2
  echo "Build it first with: python3 ./build_kvm_vm_image.py ./build-rockylinux-8_10-config.json" >&2
  exit 1
fi

if [[ ! -f "${SSH_KEY_PATH}" ]]; then
  echo "SSH private key not found: ${SSH_KEY_PATH}" >&2
  echo "Generate it by running the ISO build first." >&2
  exit 1
fi

if [[ ! -e "/dev/vfio/vfio" ]]; then
  echo "/dev/vfio/vfio not present. VFIO may not be enabled on host." >&2
  exit 1
fi

IFS=',' read -r -a PCI_IDS <<<"${CFG_PCI_IDS}"
VFIO_ARGS=()
for pci_id in "${PCI_IDS[@]}"; do
  if [[ ! -d "/sys/bus/pci/devices/${pci_id}" ]]; then
    echo "PCI device not found: ${pci_id}" >&2
    exit 1
  fi

  driver_name="unbound"
  if [[ -L "/sys/bus/pci/devices/${pci_id}/driver" ]]; then
    driver_name="$(basename "$(readlink "/sys/bus/pci/devices/${pci_id}/driver")")"
  fi
  if [[ "${driver_name}" != "vfio-pci" ]]; then
    echo "PCI device ${pci_id} is bound to '${driver_name}', expected 'vfio-pci'." >&2
    echo "Bind device to vfio-pci first, then retry." >&2
    exit 1
  fi

  VFIO_ARGS+=(-device "vfio-pci,host=${pci_id}")
done

mkdir -p "$(dirname "${DISK_PATH}")"
if [[ ! -f "${DISK_PATH}" ]]; then
  echo "Creating VM disk: ${DISK_PATH} (${DISK_SIZE})"
  qemu-img create -f qcow2 "${DISK_PATH}" "${DISK_SIZE}" >/dev/null
fi

echo "Launching VM '${VM_NAME}' with PCI passthrough"
echo "Config: ${CONFIG_PATH}"
echo "ISO:    ${ISO_PATH}"
echo "Disk:   ${DISK_PATH}"
echo "VFIO:   ${CFG_PCI_IDS}"
echo "SSH:    host 127.0.0.1:${SSH_FWD_PORT} -> guest :22"
echo
echo "Connect after install completes:"
echo "ssh -i ${SSH_KEY_PATH} \\"
echo "  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \\"
echo "  -p ${SSH_FWD_PORT} therock@127.0.0.1"

exec qemu-system-x86_64 \
  -name "${VM_NAME}" \
  -machine "q35,accel=${ACCEL_MODE}" \
  -cpu host \
  -m "${RAM_MB}" \
  -smp "${CPUS}" \
  -drive "file=${DISK_PATH},if=virtio,format=qcow2" \
  -cdrom "${ISO_PATH}" \
  -boot once=d,menu=on \
  -netdev "user,id=n1,hostfwd=tcp::${SSH_FWD_PORT}-:22" \
  -device virtio-net-pci,netdev=n1 \
  -vga none \
  -display none \
  -serial mon:stdio \
  "${VFIO_ARGS[@]}"
