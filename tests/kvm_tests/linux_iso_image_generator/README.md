# Rocky Linux 8.10 Unattended ISO Builder

This workspace creates a repeatable Rocky Linux 8.10 installer ISO using:

- a JSON config file (`build-rockylinux-8_10-config.json`)
- a Kickstart template (`templates/rocky8.ks.tmpl`)
- a non-interactive build script (`build_kvm_vm_image.py`)

## What this produces

An output ISO that boots the Rocky installer and uses embedded Kickstart values for unattended installation.

Configured defaults include:

- user `therock`
- `therock` account created locked (no password login)
- user in `wheel`
- passwordless sudo for `therock`
- `sshd` enabled
- ping (ICMP echo) allowed
- generated passwordless SSH keypair for `therock`
- key-only SSH login policy for all users (password auth disabled)

## Prerequisites (on the build host)

Install required tools:

```bash
sudo apt-get update
sudo apt-get install -y genisoimage xorriso isolinux syslinux-utils openssh-client python3
```

The build script prefers `mkisofs` (from `genisoimage`) and falls back to `xorrisofs`/`xorriso`.
On Rocky/RHEL build hosts, install equivalent packages that provide either `mkisofs` or `xorrisofs`.

## Configure

Edit `build-rockylinux-8_10-config.json`:

- all keys are required; the build script does not apply defaults
- optionally set `source_iso_sha256` for checksum verification
- customize hostname/timezone/lang/paths as needed
- SSH behavior:
  - leave `ssh_authorized_public_key` empty to auto-generate keypair
  - generated keys default to `./output/ssh/therock_ed25519` and `.pub`
  - set `ssh_authorized_public_key` if you want to inject an existing public key
  - password-based SSH login is disabled globally in generated images
  - keep `disable_user_password_auth: true` to also lock local password auth for `therock`

## Build

From this directory:

```bash
python3 ./build_kvm_vm_image.py ./build-rockylinux-8_10-config.json
```

By default, the builder now does both steps:

1) builds the unattended ISO, and
2) launches a QEMU/KVM VM from that ISO (same behavior as `testing/qemu_iso_launch.sh`).

In default mode, the script waits until guest customization is reachable/validated over SSH
(`id`, `sudo -n true`, `systemctl is-active sshd`), then it powers off the guest automatically.
This prevents leaving QEMU running unintentionally and leaves a finalized installed qcow2 image.

If you want only the ISO build (no VM run), use:

```bash
python3 ./build_kvm_vm_image.py --bare-image ./build-rockylinux-8_10-config.json
```

If you want QEMU to keep running (disable auto-shutdown mode), use:

```bash
python3 ./build_kvm_vm_image.py --run ./build-rockylinux-8_10-config.json
```

Output ISO path (default):

`./output/Rocky-8.10-x86_64-autoinstall.iso`

Installed qcow2 image path (default):

`./output/Rocky-8.10-x86_64-autoinstall.qcow2`

Optional config override in `build-rockylinux-8_10-config.json`:

```json
"output_installed_image_path": "./output/my-rocky-final.qcow2"
```

Generated SSH keys (default):

- private key: `./output/ssh/therock_ed25519`
- public key: `./output/ssh/therock_ed25519.pub`

Example login after install:

```bash
ssh -i ./output/ssh/therock_ed25519 therock@<installed-host-ip>
```

Verify password-based SSH auth is disabled on the guest:

```bash
ssh -i ./output/ssh/therock_ed25519 therock@<installed-host-ip> \
  "sudo sshd -T | rg 'passwordauthentication|kbdinteractiveauthentication|challengeresponseauthentication|pubkeyauthentication|permitemptypasswords'"
```

Expected effective values include:

- `passwordauthentication no`
- `kbdinteractiveauthentication no`
- `challengeresponseauthentication no`
- `pubkeyauthentication yes`
- `permitemptypasswords no`

## Launch with QEMU/KVM (No GPU passthrough)

Use the helper script:

```bash
./testing/qemu_iso_launch.sh
```

This launch path uses virtual display devices only and does not use GPU PCI passthrough (`vfio-pci`).
It forwards host `127.0.0.1:2222` to guest `:22` by default.
The helper script defaults to a `60G` disk to satisfy current partitioning requirements.
If you override disk size manually, keep it at least about `40G` (recommended `60G`).

The same QEMU launch behavior is now integrated into `build_kvm_vm_image.py`.
Use `--run` to keep the VM running, or `--bare-image` to skip VM launch.

After installer completes and the VM boots into the installed OS, connect with:

```bash
ssh -i ./output/ssh/therock_ed25519 \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -p 2222 therock@127.0.0.1
```

If SSH is not ready yet, poll until it becomes available:

```bash
for i in $(seq 1 120); do
  ssh -i ./output/ssh/therock_ed25519 \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=5 \
    -p 2222 therock@127.0.0.1 "echo SSH_OK && id" && break
  sleep 10
done
```

## Launch finalized qcow2 image

Use the Python helper to boot the installed image directly:

```bash
python3 ./launch_kvm_vm_image.py
```

Optional positional argument:

```bash
python3 ./launch_kvm_vm_image.py ./output/Rocky-8.10-x86_64-autoinstall.qcow2
```

Use PCI passthrough mode (same vfio checks/behavior as `testing/qemu_iso_launch_pci_passthrough.sh`):

```bash
python3 ./launch_kvm_vm_image.py --pci-p
```

In passthrough mode, device IDs are read from `build-rockylinux-8_10-config.json` key
`pci_passthrough_device_ids`, and each device must already be bound to `vfio-pci`.

Environment knobs are the same as `testing/qemu_iso_launch.sh`:

- `RAM_MB` (default `4096`)
- `CPUS` (default `2`)
- `SSH_FWD_PORT` (default `2222`)
- `VM_NAME` (default `rocky810-qcow`)
- `HEADLESS` (`1` for no GUI)
- `ACCEL_MODE` (default `kvm:tcg`)

## Launch with QEMU/KVM (PCI passthrough)

1) Set PCI IDs in `build-rockylinux-8_10-config.json`:

```json
"pci_passthrough_device_ids": ["0000:01:00.0", "0000:01:00.1"]
```

2) Make sure those devices are bound to `vfio-pci` on the host.

3) Launch with passthrough script:

```bash
./testing/qemu_iso_launch_pci_passthrough.sh
```

This script reads PCI IDs from `build-rockylinux-8_10-config.json` and fails early if IDs are missing,
not present on host, or not bound to `vfio-pci`.
It also defaults to a `60G` disk to satisfy current partitioning requirements.

## Notes

- The Kickstart template uses `clearpart --all --initlabel` with explicit partition rules, which wipes target disks during install.
- `/boot` is explicitly set to `1536 MiB` (1.5 GiB).
- `/` has minimum size `30720 MiB` (30 GiB) and then grows to fill remaining space.
- `/home` is not created as a separate filesystem; user homes live under `/`.
- SSH key authorization for `therock` is configured by Kickstart `sshkey` during install.
- Boot configs are patched to default to direct install (skip media-check default) and include serial console args for headless VM troubleshooting.
