# ROCm Runtime MSI — Usage Guide

## System Requirements

|              |                                           |
| ------------ | ----------------------------------------- |
| OS           | Windows 11 / Windows Server 2019 or later |
| Architecture | x86-64                                    |
| Privileges   | Administrator                             |
| Disk space   | ~200 MB                                   |

## Install

The MSI has no graphical UI and is intended for scripted and enterprise
deployment. All `msiexec` commands must be run from an elevated prompt.

**Silent install (default location):**

```bat
msiexec /i amdrocm-runtime.msi /qn
```

**Silent install with a log file:**

```bat
msiexec /i amdrocm-runtime.msi /qn /l*v "%TEMP%\rocm-install.log"
```

**Silent install to a custom directory:**

```bat
msiexec /i amdrocm-runtime.msi /qn TARGETDIR="D:\ROCm\"
```

**Silent install with long-path support disabled:**

```bat
msiexec /i amdrocm-runtime.msi /qn ENABLE_LONG_PATHS=0
```

**Silent install with legacy System32 DLLs enabled:**

```bat
msiexec /i amdrocm-runtime.msi /qn LEGACY_SYSTEM32_INSTALL=1
```

## What Gets Installed

| Item                         | Default location                                                               |
| ---------------------------- | ------------------------------------------------------------------------------ |
| Runtime DLLs and executables | `C:\Program Files\AMD\ROCm\runtime-<version>\bin\`                             |
| Import libraries (`.lib`)    | `C:\Program Files\AMD\ROCm\runtime-<version>\lib\`                             |
| System PATH entry            | `...\bin` appended to the machine-wide PATH (default install only — see below) |
| SDK-discovery registry key   | `HKLM\Software\AMD\ROCm\<X.Y>` (`InstallDir`, `Version`)                       |

The machine PATH entry is added **only when installing to the default location**
(`C:\Program Files\AMD\ROCm\...`), which is admin-write-only. Installing to a
custom location (via `--install-root` at generation time, or a different
`INSTALLFOLDER` / `TARGETDIR` at install time) **skips** the PATH entry, because
putting a potentially user-writable directory on the machine-wide PATH is a
binary-planting risk. In that case, locate ROCm via the SDK-discovery registry
key below instead of relying on PATH.

The SDK-discovery key is **shared** across all ROCm packages installed at the
same `<X.Y>` version (per
[RFC0014](/docs/rfcs/RFC0014-Windows-Packaging-Requirements.md)): tools locate
ROCm via `InstallDir` / `Version` without needing to know which packages are
present. It is written in the native 64-bit registry view and removed when the
last package at that version is uninstalled. The package's `ProductCode` is not
stored here — each MSI registers its own under the standard Add/Remove Programs
(`HKLM\...\Uninstall\{ProductCode}`) hive.

A default install also removes the legacy ROCm DLL names from
`C:\Windows\System32\` so they cannot shadow the copies under `bin\`; see
[Legacy System32 DLLs](#legacy-system32-dlls).

### Optional: Long Path Support

Passing `ENABLE_LONG_PATHS=1` additionally writes:

| Registry key                                       | Value              | Data        |
| -------------------------------------------------- | ------------------ | ----------- |
| `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem` | `LongPathsEnabled` | `1` (DWORD) |

This lifts the 260-character `MAX_PATH` limit system-wide. Requires Windows 11
or Windows Server 2019 or later. A reboot is needed for the change to propagate to all running
processes; processes started after the installer exits pick it up immediately.

The installer checks whether `LongPathsEnabled` was already set to `1` before
installing. If it was already enabled (by the user or another installer), the
key is left untouched on uninstall. If the ROCm installer set it, it is removed
on uninstall.

### Legacy System32 DLLs

By default the installer places the ROCm DLLs only under the install dir's
`bin\` (reachable via the machine `PATH`), and **removes** the package's legacy
DLL names from `C:\Windows\System32\` — on both install and uninstall — so a
stale copy left by an AMD driver, an older ROCm installer, or a prior
`LEGACY_SYSTEM32_INSTALL=1` install cannot shadow the runtime shipped under `bin\`. For
the `runtime` package these five names are cleaned up: `amdhip64_6.dll`,
`amdhip64_7.dll`, `amd_comgr.dll`, `amd_comgr_2.dll`, and `rocm_kpack.dll`.

To instead install those DLLs **into** `System32` (for applications that load
ROCm DLLs from `System32` rather than from `PATH`), pass `LEGACY_SYSTEM32_INSTALL=1`:

```bat
msiexec /i amdrocm-runtime.msi /qn LEGACY_SYSTEM32_INSTALL=1
```

This copies the same five DLLs into `System32` and skips the cleanup. The two
behaviors are mutually exclusive — a given install either scrubs the System32
names (default) or populates them (`LEGACY_SYSTEM32_INSTALL=1`), never both. The System32
copies written by `LEGACY_SYSTEM32_INSTALL=1` are shared, reference-counted components and
are **not** removed when that install is later uninstalled.

## Upgrade

Install the new MSI directly — no manual uninstall required:

```bat
msiexec /i amdrocm-runtime-<new-version>.msi /qn
```

Installing an older version over a newer one is blocked. Uninstall the current
version first if a downgrade is needed.

## Uninstall

**Using the MSI file:**

```bat
msiexec /x amdrocm-runtime.msi /qn
```

**Using the product code (when the MSI file is unavailable):**

```bat
:: Find the product code
wmic product where "Name like '%ROCm%'" get Name,Version,IdentifyingNumber

:: Uninstall by product code
msiexec /x {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX} /qn
```

**Using Settings:** Apps > Installed apps > find the ROCm entry (e.g.
"AMD ROCm Runtime", or "AMD ROCm Core Runtime" for the core package) >
Uninstall.

Uninstall removes all installed files, the PATH entry, and all registry keys
written by the installer. Files created after installation are not removed.

## Troubleshooting

### PATH not updated

PATH changes apply to newly opened terminals only. Open a new prompt and check:

```bat
echo %PATH%
```

`...\AMD\ROCm\runtime-<version>\bin` should appear in the output.

### Installation fails

Capture a verbose log and check the last error:

```bat
msiexec /i amdrocm-runtime.msi /qn /l*v "%TEMP%\rocm-install.log"
findstr /i "error" "%TEMP%\rocm-install.log"
```

### Long path errors

Long-path support is enabled by default. If it was explicitly disabled, re-enable it:

```bat
msiexec /i amdrocm-runtime.msi /qn ENABLE_LONG_PATHS=1
```

Or set the registry key manually and reboot:

```bat
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

### Legacy DLL conflicts

DLLs such as `amdhip64_6.dll` in `C:\Windows\System32\` take precedence over the
copies in Program Files and can shadow the installed runtime. A default install
removes these names from `System32` automatically (see
[Legacy System32 DLLs](#legacy-system32-dlls)), so a plain reinstall or repair
clears a stale copy:

```bat
msiexec /fvomus amdrocm-runtime.msi /qn
```

A System32 copy only persists intentionally when it was placed by
`LEGACY_SYSTEM32_INSTALL=1` (those are left in place on uninstall). To confirm what is
present:

```bat
dir C:\Windows\System32\amdhip64_*.dll
dir C:\Windows\System32\amd_comgr_*.dll
```

## See Also

- [MSI generator script usage](msi-generator-usage.md)
