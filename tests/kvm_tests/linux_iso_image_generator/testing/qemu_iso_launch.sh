#!/usr/bin/env bash
set -euo pipefail

# qemu_iso_launch.sh
# Launch Rocky Linux autoinstall ISO with QEMU/KVM without GPU PCI passthrough.
#
# This script intentionally uses a virtual/emulated display device only.
# It does NOT include vfio GPU passthrough flags such as:
#   -device vfio-pci,...
#
# Usage:
#   ./testing/qemu_iso_launch.sh [ISO_PATH] [DISK_PATH]
#
# Optional environment variables:
#   RAM_MB=4096           Memory for VM in MB
#   CPUS=2                Number of virtual CPUs
#   DISK_SIZE=60G         Created qcow2 size when DISK_PATH is missing
#   SSH_FWD_PORT=2222     Host TCP port forwarded to guest 22
#   VM_NAME=rocky810-test VM name shown by QEMU
#   HEADLESS=0            Set to 1 for no GUI window
#   ACCEL_MODE=kvm:tcg    kvm preferred, fallback to tcg
#
# Examples:
#   ./testing/qemu_iso_launch.sh
#   HEADLESS=1 ./testing/qemu_iso_launch.sh
#   RAM_MB=8192 CPUS=4 SSH_FWD_PORT=2200 ./testing/qemu_iso_launch.sh
#
# SSH login after install finishes:
#   ssh -i ./output/ssh/therock_ed25519 \
#     -o StrictHostKeyChecking=no \
#     -o UserKnownHostsFile=/dev/null \
#     -p 2222 therock@127.0.0.1
#
# If SSH is not ready yet, poll until it comes up:
#   for i in $(seq 1 120); do
#     ssh -i ./output/ssh/therock_ed25519 \
#       -o BatchMode=yes \
#       -o StrictHostKeyChecking=no \
#       -o UserKnownHostsFile=/dev/null \
#       -o ConnectTimeout=5 \
#       -p 2222 therock@127.0.0.1 "echo SSH_OK && id" && break
#     sleep 10
#   done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ISO_PATH="${1:-${REPO_ROOT}/output/Rocky-8.10-x86_64-autoinstall.iso}"
DISK_PATH="${2:-${REPO_ROOT}/output/rocky810-test.qcow2}"

RAM_MB="${RAM_MB:-4096}"
CPUS="${CPUS:-2}"
DISK_SIZE="${DISK_SIZE:-60G}"
SSH_FWD_PORT="${SSH_FWD_PORT:-2222}"
VM_NAME="${VM_NAME:-rocky810-test}"
HEADLESS="${HEADLESS:-0}"
ACCEL_MODE="${ACCEL_MODE:-kvm:tcg}"

if ! command -v qemu-system-x86_64 >/dev/null 2>&1; then
  echo "Missing command: qemu-system-x86_64" >&2
  exit 1
fi

if ! command -v qemu-img >/dev/null 2>&1; then
  echo "Missing command: qemu-img" >&2
  exit 1
fi

if [[ ! -f "${ISO_PATH}" ]]; then
  echo "ISO not found: ${ISO_PATH}" >&2
  echo "Build it first with: python3 ./build_kvm_vm_image.py ./build-rockylinux-8_10-config.json" >&2
  exit 1
fi

mkdir -p "$(dirname "${DISK_PATH}")"
if [[ ! -f "${DISK_PATH}" ]]; then
  echo "Creating VM disk: ${DISK_PATH} (${DISK_SIZE})"
  qemu-img create -f qcow2 "${DISK_PATH}" "${DISK_SIZE}" >/dev/null
fi

echo "Launching VM '${VM_NAME}'"
echo "ISO:  ${ISO_PATH}"
echo "Disk: ${DISK_PATH}"
echo "SSH:  host 127.0.0.1:${SSH_FWD_PORT} -> guest :22"
echo
echo "No GPU PCI passthrough is used. VM display is virtual only."
echo
echo "When install completes, connect with:"
echo "ssh -i ${REPO_ROOT}/output/ssh/therock_ed25519 \\"
echo "  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \\"
echo "  -p ${SSH_FWD_PORT} therock@127.0.0.1"

QEMU_DISPLAY_ARGS=()
if [[ "${HEADLESS}" == "1" ]]; then
  QEMU_DISPLAY_ARGS=(-display none -serial mon:stdio)
else
  # Virtual GPU/display only (no physical GPU passthrough).
  QEMU_DISPLAY_ARGS=(-device virtio-vga -display gtk,gl=off)
fi

exec qemu-system-x86_64 \
  -name "${VM_NAME}" \
  -machine accel="${ACCEL_MODE}" \
  -m "${RAM_MB}" \
  -smp "${CPUS}" \
  -drive "file=${DISK_PATH},if=virtio,format=qcow2" \
  -cdrom "${ISO_PATH}" \
  -boot once=d,menu=on \
  -netdev "user,id=n1,hostfwd=tcp::${SSH_FWD_PORT}-:22" \
  -device virtio-net-pci,netdev=n1 \
  "${QEMU_DISPLAY_ARGS[@]}"
