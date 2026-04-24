# ROCm Windows MSI Installer

Silent, per-machine MSI installer for the ROCm runtime (`bin/` and `lib/`).

## Requirements

| Requirement | Details |
|---|---|
| OS | Windows 11 / Windows Server 2019 or later |
| WiX Toolset | v4 — `winget install WiXToolset.WiX` |
| TheRock build | `build/dist/rocm/` must exist and contain the artifacts listed in `amdrocm-runtimes-artifacts.txt` |

## Build the MSI

From the repo root:

```cmd
python build_tools\packaging\windows\generate_msi_wxs.py
wix build build_tools\packaging\windows\amdrocm-runtimes.wxs -o build_tools\packaging\windows\amdrocm-runtimes.msi
```

## Install

Standard install (installs to `C:\Program Files\AMD\ROCm\<version>\`):

```cmd
msiexec /i amdrocm-runtimes.msi /qn
```

Enable Windows long path support during install:

```cmd
msiexec /i amdrocm-runtimes.msi /qn ENABLE_LONG_PATHS=1
```

## Uninstall

```cmd
msiexec /x amdrocm-runtimes.msi /qn
```

Or uninstall by product code:

```cmd
msiexec /x {PRODUCT-CODE-GUID} /qn
```

## Upgrade

Run the new installer directly — `MajorUpgrade` is configured to remove the previous version automatically:

```cmd
msiexec /i amdrocm-runtimes-<new-version>.msi /qn
```

## Legacy DLL Cleanup

The installer automatically removes legacy ROCm DLLs (`amdhip64_*.dll`, `amd_comgr_*.dll`) from `%SystemRoot%\System32` during installation.

## Further Reading

- [MSI installer usage reference](amdrocm-runtimes-msi-usage.md)
- [Generator script usage reference](msi-generator-usage.md)
