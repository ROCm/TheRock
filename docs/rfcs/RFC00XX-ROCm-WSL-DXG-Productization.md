---
author: Saad Rahim (saadrahim)
created: 2026-05-22
modified: 2026-05-22
status: draft
---

# ROCm WSL2 Support via DXG

## Overview

This RFC describes the work required to productize ROCm support on Windows Subsystem for Linux 2 (WSL2)
using the DXG (DirectX Graphics) runtime path, while remaining fully compliant with
[RFC0009 – OS Packaging Requirements](./RFC0009-OS-Packaging-Requirements.md).

ROCm execution under WSL2 relies on Microsoft's DXG kernel interface (`/dev/dxg`) to route GPU requests
from Linux user space to the Windows graphics stack. The DXG backend is implemented in the ROCr runtime
under `projects/rocr-runtime/libhsakmt/src/dxg`
([source](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime/libhsakmt/src/dxg)).

Productization moves the source of truth for DXG enablement from a user- or shell-scoped environment
variable into a persistent ROCm configuration file read by the ROCr runtime itself. This removes the
dependence on environment propagation across shells, `ssh` invocations, container runtimes, and
application launchers.

> **Beta scope.** ROCr will introduce the `/etc/rocm/rocm.conf` configuration file in **beta**, scoped
> initially to `HSA_ENABLE_DXG_DETECTION`. If the mechanism proves successful in production, the intent
> is to extend the same file to provide a persistent alternative to **all environment variables in the
> ROCm Core SDK**, so that runtime tuning no longer depends on environment propagation. The format,
> parser, search order, and caching design defined in this RFC are deliberately general so that adding
> further keys does not require structural changes.

Our goals are to:

1. **Enable ROCr to detect WSL2 from a persistent configuration file under `/etc`, not from per-shell
   environment state.**
1. **Comply with [RFC0009](./RFC0009-OS-Packaging-Requirements.md) directory, RPATH, and relocatability
   rules.**
1. **Bundle `librocdxg.so` inside the ROCm Core SDK prefix without leaking into system library paths.**
1. **Provide identical behavior for DEB, RPM, and tarball installs, with tarballs remaining
   configuration-free.**
1. **Preserve `HSA_ENABLE_DXG_DETECTION` as a manual override for debugging, containers, and tarball
   workflows.**
1. **Pilot a general configuration-file mechanism (beta) that, on success, can replace all ROCm Core SDK
   environment variables with persistent file-based settings.**

## Scope

### In Scope

- ROCr runtime behavior for selecting the DXG backend on WSL2.
- Persistent configuration file location, format, and ownership.
- WSL2 detection logic and when it runs.
- DEB, RPM, and tarball packaging changes for `librocdxg.so` and the detector script.
- Updates to `build_tools/install_rocm_from_artifacts.py` and `RELEASES.md`.

### Out of Scope

- GPU driver packaging (out of scope per [RFC0009](./RFC0009-OS-Packaging-Requirements.md)).
- Native Linux GPU detection or KFD device handling.
- Windows-native ROCm packaging.
- Python pip / wheelnext packaging (covered separately).
- Per-application runtime tuning unrelated to DXG.

## Directory Layout

Per [RFC0009](./RFC0009-OS-Packaging-Requirements.md), the ROCm Core SDK installs under
`/opt/rocm/core-X.Y` with a `/opt/rocm/core` symlink to the active version. All paths below are written
relative to `/opt/rocm/core`; the resolved path on disk includes the `-X.Y` suffix.

### `librocdxg.so` Installation

| Install method | Path                              |
| :------------- | :-------------------------------- |
| Tarball        | `<ROCM_PATH>/lib/librocdxg.so`    |
| DEB / RPM      | `/opt/rocm/core/lib/librocdxg.so` |

The library must not be installed into `/usr/lib`, `/usr/lib64`, or `/lib*`. RPATH must remain
`$ORIGIN`-based per [RFC0009](./RFC0009-OS-Packaging-Requirements.md).

### Detector Script Installation

| Install method | Path                                     |
| :------------- | :--------------------------------------- |
| Tarball        | `<ROCM_PATH>/libexec/rocm-detect-env`    |
| DEB / RPM      | `/opt/rocm/core/libexec/rocm-detect-env` |

### System Configuration

| Path                        | Role                                                                                                                                                                                                                                                    |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/etc/rocm/rocm.conf`       | Canonical ROCm configuration file read by ROCr.                                                                                                                                                                                                         |
| `/etc/default/rocm`         | Debian/Ubuntu fallback location (read if canonical absent).                                                                                                                                                                                             |
| `/etc/sysconfig/rocm`       | RHEL/Fedora/SLES fallback location (read if canonical absent).                                                                                                                                                                                          |
| `/etc/tmpfiles.d/rocm.conf` | `systemd-tmpfiles` entry that creates `/run/rocm/` on boot for the parsed-config cache. Placed under `/etc` rather than `/usr/lib/tmpfiles.d/` to comply with [RFC0009](./RFC0009-OS-Packaging-Requirements.md)'s prohibition on writes to `/usr/lib*`. |
| `/run/rocm/rocm.conf.cache` | Runtime cache of parsed configuration (`tmpfs`, volatile across reboots).                                                                                                                                                                               |

Configuration under `/etc` is treated as system configuration and is permitted outside the ROCm prefix,
consistent with [RFC0009](./RFC0009-OS-Packaging-Requirements.md) which restricts library installation
paths but not system configuration.

## ROCr Runtime Behavior

At initialization, ROCr determines whether to enable the DXG backend by consulting the following sources
in order. The first source that yields a definite value wins.

| Order | Source                             | Notes                                                                      |
| :---- | :--------------------------------- | :------------------------------------------------------------------------- |
| 1     | `HSA_ENABLE_DXG_DETECTION` env var | Manual override. Values `1`/`true`/`yes` enable, `0`/`false`/`no` disable. |
| 2     | `/etc/rocm/rocm.conf`              | Canonical persistent configuration.                                        |
| 3     | `/etc/default/rocm`                | Debian/Ubuntu fallback.                                                    |
| 4     | `/etc/sysconfig/rocm`              | RHEL/Fedora/SLES fallback.                                                 |
| 5     | Default                            | DXG disabled (native Linux behavior).                                      |

The configuration file is parsed by ROCr in-process. No shell sourcing, environment-file plumbing, or
consumer cooperation is required. A workload launched from an interactive shell, a desktop shortcut, a
`cron` job, or a container observes identical behavior provided the file is visible.

### Performance Requirements

Reading the configuration file must have minimal impact on application startup times. The ROCr
implementation must:

- Perform a single, lightweight file read operation during initialization.
- Use simple parsing logic (plain `KEY=value`, no complex syntax requiring heavy parsers).
- Avoid blocking I/O or network operations during configuration loading.
- Cache the parsed result in a memory-backed filesystem (preferably `/run/rocm/rocm.conf.cache` on
  `tmpfs`, falling back to `/dev/shm/rocm.conf.cache`) so subsequent process invocations on the same
  boot can skip re-parsing `/etc/rocm/rocm.conf`. The cache is invalidated when its `mtime` is older
  than the source file's `mtime`, and is automatically discarded on reboot since `tmpfs` does not
  persist.
- Fail fast with sensible defaults if the configuration file is missing or unreadable.

The design prioritizes startup performance over configuration flexibility. Complex configuration
requirements should be addressed through separate mechanisms rather than expanding the scope of this
file.

## Persistent Configuration File

### Format

The configuration file uses an **INI-style sectioned `KEY=value` format**. Each section corresponds to
one ROCm Core SDK component, and each key within a section names a setting owned by that component. Keys
mirror the existing environment-variable names so migration is mechanical and a reader familiar with the
env-var surface can read the file without a translation table.

Lexical rules:

| Rule                  | Detail                                                                                                                                                                                                                                                                                                                                         |
| :-------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Section header        | `[name]` on its own line. Names are lowercase ASCII, `[a-z][a-z0-9_-]*`.                                                                                                                                                                                                                                                                       |
| Key/value             | `KEY=value`, one per line. Keys are case-sensitive and match the env-var spelling (typically `UPPER_SNAKE_CASE`).                                                                                                                                                                                                                              |
| Comments              | Lines beginning with `#` or `;` (after optional leading whitespace). Inline comments are **not** supported — a `#` inside a value is part of the value.                                                                                                                                                                                        |
| Whitespace            | Leading/trailing whitespace on keys, values, and section headers is stripped. Blank lines are ignored.                                                                                                                                                                                                                                         |
| Quoting               | Values may optionally be wrapped in `"..."` or `'...'` to preserve leading/trailing whitespace. Quotes are stripped.                                                                                                                                                                                                                           |
| Escapes               | None. The file is not shell-sourced; `$VAR`, backticks, and backslash escapes are literal.                                                                                                                                                                                                                                                     |
| Booleans              | `1`/`true`/`yes`/`on` and `0`/`false`/`no`/`off` (case-insensitive).                                                                                                                                                                                                                                                                           |
| Duplicate keys        | Last occurrence within a section wins. A warning is logged.                                                                                                                                                                                                                                                                                    |
| Unknown sections/keys | Ignored with a debug-level log message. Forward compatibility: future components may introduce new sections without breaking older runtimes.                                                                                                                                                                                                   |
| Pre-section keys      | Keys appearing **before** any `[section]` header are assigned to an implicit `[global]` section. `[global]` is reserved for cross-component settings; component-specific keys placed there are ignored with a warning.                                                                                                                         |
| Per-key documentation | Every key written by the package detector, by `rocm-detect-env`, or by any future Core SDK installer **must** be preceded by a one-to-two-sentence comment that describes the behavior the key controls, the accepted values, and the effective default when the key is absent. Hand-edited keys are encouraged to follow the same convention. |

### Sections

Sections are namespaced one-per-library. The initial set of well-known sections is:

| Section         | Component / library                        | Typical key prefixes (env-var equivalents) |
| :-------------- | :----------------------------------------- | :----------------------------------------- |
| `[global]`      | Cross-component settings                   | `ROCM_*`                                   |
| `[rocr]`        | ROCr runtime (HSA, libhsakmt, DXG backend) | `HSA_*`, `ROCR_*`, `HSAKMT_*`              |
| `[hip]`         | HIP runtime and CLR                        | `HIP_*`, `AMD_*`, `GPU_*`                  |
| `[rccl]`        | RCCL collective communication              | `NCCL_*`, `RCCL_*`                         |
| `[rocblas]`     | rocBLAS                                    | `ROCBLAS_*`                                |
| `[hipblas]`     | hipBLAS / hipBLASLt                        | `HIPBLAS_*`, `HIPBLASLT_*`                 |
| `[rocsolver]`   | rocSOLVER / hipSOLVER                      | `ROCSOLVER_*`, `HIPSOLVER_*`               |
| `[rocsparse]`   | rocSPARSE / hipSPARSE                      | `ROCSPARSE_*`, `HIPSPARSE_*`               |
| `[rocfft]`      | rocFFT / hipFFT                            | `ROCFFT_*`, `HIPFFT_*`                     |
| `[rocrand]`     | rocRAND / hipRAND                          | `ROCRAND_*`, `HIPRAND_*`                   |
| `[miopen]`      | MIOpen                                     | `MIOPEN_*`                                 |
| `[hipdnn]`      | hipDNN                                     | `HIPDNN_*`                                 |
| `[rocprim]`     | rocPRIM / rocThrust / hipCUB               | `ROCPRIM_*`, `HIPCUB_*`                    |
| `[rocprofiler]` | rocprofiler-sdk and tracer                 | `ROCPROFILER_*`, `ROCP_*`                  |
| `[rocgdb]`      | rocGDB                                     | `ROCGDB_*`, `AMDGPU_GDB_*`                 |
| `[amdsmi]`      | amd-smi                                    | `AMDSMI_*`                                 |
| `[opencl]`      | OpenCL runtime                             | `CL_*`, `OCL_*`                            |
| `[openmp]`      | OpenMP offload                             | `LIBOMPTARGET_*`, `OMP_TARGET_*`           |
| `[rocdecode]`   | rocDecode                                  | `ROCDECODE_*`                              |
| `[rocjpeg]`     | rocJPEG                                    | `ROCJPEG_*`                                |
| `[rdc]`         | ROCm Data Center tool                      | `RDC_*`                                    |

Each component reads only its own section (and `[global]` for shared keys). Components must document the
keys they consume; the table above lists *prefixes* that are conventional but not enforced — a section
may carry any key its owning component agrees to honor.

### Override Precedence

When the same logical setting is expressible both as an environment variable and as a sectioned key, the
order from highest priority to lowest is:

1. Environment variable (e.g. `HSA_ENABLE_DXG_DETECTION=1` in the process environment).
1. Key in `/etc/rocm/rocm.conf`, in its component section.
1. Key in `/etc/default/rocm` or `/etc/sysconfig/rocm` (fallback locations, same sectioned format).
1. User-scoped file `$XDG_CONFIG_HOME/rocm/rocm.conf` (Python-install scenarios; see §Python Packages).
1. Component-internal default.

### Example

A representative `/etc/rocm/rocm.conf` after install on a WSL2 host:

```ini
# /etc/rocm/rocm.conf
# Generated by rocm-detect-env on 2026-05-22 — environment: WSL2
# Format: INI. Section per ROCm Core SDK component, KEY=value within.
# Lines beginning with '#' or ';' are comments. Every key carries a
# one-to-two-sentence comment describing its behaviour.

[global]
# Informational path to the active ROCm Core SDK prefix. Consumed only by
# diagnostic tools; the runtime prefix is resolved at link time via RPATH.
ROCM_PATH=/opt/rocm

[rocr]
# Enables the DXG backend so ROCr routes GPU calls through /dev/dxg on WSL2.
# Accepted values: 1/true/yes (enable) or 0/false/no (disable). Default: 0.
HSA_ENABLE_DXG_DETECTION=1

[hip]
# Restricts HIP device enumeration to the listed comma-separated indices.
# Default: all visible GPUs are enumerated. Example shown is commented out.
# HIP_VISIBLE_DEVICES=0

[rccl]
# Sets the RCCL/NCCL log verbosity. Accepted values: VERSION, WARN, INFO, TRACE.
# Default: WARN. Increase only when debugging collective communication issues.
# NCCL_DEBUG=WARN
```

A representative file on a native Linux host:

```ini
# /etc/rocm/rocm.conf
# Generated by rocm-detect-env on 2026-05-22 — environment: native Linux

[rocr]
# Enables the DXG backend so ROCr routes GPU calls through /dev/dxg on WSL2.
# Accepted values: 1/true/yes (enable) or 0/false/no (disable). Default: 0.
HSA_ENABLE_DXG_DETECTION=0
```

The native-Linux case writes the DXG key explicitly so ROCr always observes a definite value rather than
relying on absence.

### Parsing Requirements

The parser is shipped as part of the ROCm Core SDK (initially inside ROCr) and must:

- Be implemented in C/C++ with no external dependencies beyond libc, so it links cleanly into every Core
  SDK component.
- Stream the file in a single pass — no second read for section discovery.
- Allocate at most one heap buffer for the full file contents (sized by `fstat`), then build the
  in-memory map by pointing into that buffer with NUL-terminated key/value spans.
- Be reentrant and thread-safe; concurrent first-use from multiple threads must not double-parse the
  file.
- Tolerate UTF-8 BOM and both LF and CRLF line endings (CRLF matters on WSL2 where users may edit the
  file from Windows).
- Reject files larger than 1 MiB with a logged error and fall back to defaults; the file is
  configuration, not data.

### Package Ownership

The file must be marked as a configuration file so that local admin edits survive package upgrades:

| Format | Mechanism                                           |
| :----- | :-------------------------------------------------- |
| DEB    | List `/etc/rocm/rocm.conf` in `debian/conffiles`.   |
| RPM    | Mark `/etc/rocm/rocm.conf` as `%config(noreplace)`. |

### Detection Logic

WSL2 is detected by checking, in order:

1. `/proc/sys/kernel/osrelease` contains `microsoft` or `WSL` (case-insensitive).
1. `/dev/dxg` exists.
1. `/proc/version` contains `microsoft` (fallback).

If any check succeeds the host is treated as WSL2 and `HSA_ENABLE_DXG_DETECTION=1` is written. Otherwise
the host is treated as native Linux and `HSA_ENABLE_DXG_DETECTION=0` is written.

### Runtime Cache Directory

The runtime cache lives under `/run/rocm/`, which is wiped on every boot since `/run` is `tmpfs`.
Packages provision the directory by shipping a `systemd-tmpfiles` snippet at `/etc/tmpfiles.d/rocm.conf`:

```
# /etc/tmpfiles.d/rocm.conf
# Type Path        Mode UID  GID  Age Argument
d      /run/rocm   0755 root root -   -
```

`systemd-tmpfiles --create /etc/tmpfiles.d/rocm.conf` is invoked from the package scriptlets immediately
after install so the directory exists without waiting for a reboot. On WSL2 distributions running
without systemd, ROCr falls back to `mkdir -p /run/rocm` on first cache write (or, if `/run/rocm/`
cannot be created, to `/dev/shm/rocm.conf.cache`).

### Reference Detector

```sh
#!/bin/sh
# /opt/rocm/core/libexec/rocm-detect-env
# Detect runtime environment once and persist ROCm config.
set -eu

out=/etc/rocm/rocm.conf
mkdir -p "$(dirname "$out")"

if grep -qiE 'microsoft|wsl' /proc/sys/kernel/osrelease 2>/dev/null \
   || [ -e /dev/dxg ]; then
    env_kind="WSL2"
    dxg_value=1
else
    env_kind="native Linux"
    dxg_value=0
fi

# Only overwrite if file is missing or auto-generated; preserve admin edits.
if [ ! -e "$out" ] || head -n1 "$out" | grep -q 'Generated by rocm-detect-env'; then
    umask 022
    {
        printf '# Generated by rocm-detect-env on %s — environment: %s\n' \
               "$(date -Is)" "$env_kind"
        printf '# Edit freely; removing the "Generated by" header above\n'
        printf '# prevents future regeneration by package upgrades.\n'
        printf '# Format: INI. Section per ROCm Core SDK component.\n'
        printf '\n'
        printf '[rocr]\n'
        printf '# Enables the DXG backend so ROCr routes GPU calls through /dev/dxg on WSL2.\n'
        printf '# Accepted values: 1/true/yes (enable) or 0/false/no (disable). Default: 0.\n'
        printf 'HSA_ENABLE_DXG_DETECTION=%s\n' "$dxg_value"
    } >"$out"
fi
```

### When Detection Runs

| Trigger                  | Behavior                                                     |
| :----------------------- | :----------------------------------------------------------- |
| DEB `postinst configure` | Invokes `rocm-detect-env` to populate `/etc/rocm/rocm.conf`. |
| RPM `%post`              | Invokes `rocm-detect-env` to populate `/etc/rocm/rocm.conf`. |
| Tarball install          | Detector shipped; user runs it manually if desired.          |
| ROCr initialization      | Reads the file. Does **not** re-run detection.               |

The WSL2-vs-native property is fixed for the life of the installation, so detection runs at most once
per install or upgrade.

## Package-Specific Behavior

### DEB

- Install `librocdxg.so` under `/opt/rocm/core/lib`.
- Install `/opt/rocm/core/libexec/rocm-detect-env`.
- Install `/etc/tmpfiles.d/rocm.conf` containing the `d /run/rocm 0755 root root - -` entry.
- Invoke `rocm-detect-env` from `postinst` on `configure`.
- Invoke `systemd-tmpfiles --create /etc/tmpfiles.d/rocm.conf` from `postinst` (guarded by a check that
  `systemd-tmpfiles` is present, since WSL2 distros may lack systemd).
- Register `/etc/rocm/rocm.conf` and `/etc/tmpfiles.d/rocm.conf` in `debian/conffiles`.
- Do **not** ship `/etc/profile.d/rocm-wsl.sh` or any equivalent shell hook.

### RPM

- Install `librocdxg.so` under `/opt/rocm/core/lib`.
- Install `/opt/rocm/core/libexec/rocm-detect-env`.
- Install `/etc/tmpfiles.d/rocm.conf` containing the `d /run/rocm 0755 root root - -` entry.
- Invoke `rocm-detect-env` from `%post`.
- Invoke `systemd-tmpfiles --create /etc/tmpfiles.d/rocm.conf` from `%post` (guarded by a check that
  `systemd-tmpfiles` is present).
- Mark `/etc/rocm/rocm.conf` and `/etc/tmpfiles.d/rocm.conf` as `%config(noreplace)`.
- Do **not** ship `/etc/profile.d/rocm-wsl.sh` or any equivalent shell hook.

### Tarball

- Ship `librocdxg.so` under `<ROCM_PATH>/lib`.
- Ship `rocm-detect-env` under `<ROCM_PATH>/libexec`.
- Do not modify system configuration at unpack time.
- Document the two manual paths in `RELEASES.md`:
  1. `sudo <ROCM_PATH>/libexec/rocm-detect-env` to populate `/etc/rocm/rocm.conf`.
  1. `export HSA_ENABLE_DXG_DETECTION=1` for ephemeral use.

### Python Packages

- Detect WSL2 during install.
- When `/etc/rocm/rocm.conf` is not writable, write to `$XDG_CONFIG_HOME/rocm/rocm.conf` (default
  `~/.config/rocm/rocm.conf`).
- ROCr's configuration-file search order will be extended to consult the user-scoped path after the
  system paths and before falling back to native behavior.

## Installer Script Update

File: `build_tools/install_rocm_from_artifacts.py`

Requirements:

- Resolve libraries under the ROCm prefix; do not assume native AMDGPU drivers.
- When run with sufficient privileges, invoke `rocm-detect-env` to populate `/etc/rocm/rocm.conf`.
- Honor `HSA_ENABLE_DXG_DETECTION` if already set in the invoking environment.

## RELEASES.md Update

Add the following under tarball documentation:

```markdown
### WSL2 (Windows Subsystem for Linux) Support

ROCm package installs (DEB/RPM) auto-detect WSL2 at install time and write
the DXG setting to /etc/rocm/rocm.conf. ROCr reads this file directly at
initialization, so no shell, service, or container configuration is required.

Tarball installs under WSL2 should either run the bundled detector once:

    sudo <ROCM_PATH>/libexec/rocm-detect-env

or export the override variable manually:

    export HSA_ENABLE_DXG_DETECTION=1
```

## Compatibility

| Concern                                           | Resolution                                                                                                                                                     |
| :------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Existing `HSA_ENABLE_DXG_DETECTION` users         | Retained as highest-priority override; existing workflows unchanged.                                                                                           |
| Native Linux hosts                                | File written with `HSA_ENABLE_DXG_DETECTION=0`; behavior unchanged.                                                                                            |
| Admin-edited config                               | `conffiles` / `%config(noreplace)` preserve edits across upgrades; the `Generated by` header sentinel prevents the detector from clobbering hand-edited files. |
| Tarball installs                                  | No `/etc` writes at unpack time; detector and override both available.                                                                                         |
| Containers without `/etc/rocm/`                   | Override variable remains the recommended path.                                                                                                                |
| WSL2 distros without `systemd`/`systemd-tmpfiles` | ROCr falls back to `mkdir -p /run/rocm` on first cache write, then to `/dev/shm/rocm.conf.cache` if `/run/rocm/` is unavailable.                               |

## References

- [RFC0009 – OS Packaging Requirements](./RFC0009-OS-Packaging-Requirements.md)
- [ROCr DXG backend source](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime/libhsakmt/src/dxg)
