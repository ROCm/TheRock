# ROCm Runtime MSI — Usage Guide

## System Requirements

| | |
|---|---|
| OS | Windows 11 / Windows Server 2019 or later |
| Architecture | x86-64 |
| Privileges | Administrator |
| Disk space | ~200 MB |

## Install

The MSI has no graphical UI and is intended for scripted and enterprise
deployment. All `msiexec` commands must be run from an elevated prompt.

**Silent install (default location):**

```bat
msiexec /i amdrocm-runtimes.msi /qn
```

**Silent install with a log file:**

```bat
msiexec /i amdrocm-runtimes.msi /qn /l*v "%TEMP%\rocm-install.log"
```

**Silent install to a custom directory:**

```bat
msiexec /i amdrocm-runtimes.msi /qn TARGETDIR="D:\ROCm\"
```

**Silent install with long-path support enabled:**

```bat
msiexec /i amdrocm-runtimes.msi /qn ENABLE_LONG_PATHS=1
```

## What Gets Installed

| Item | Default location |
|---|---|
| Runtime DLLs and executables | `C:\Program Files\AMD\ROCm\runtimes-<version>\bin\` |
| Import libraries (`.lib`) | `C:\Program Files\AMD\ROCm\runtimes-<version>\lib\` |
| System PATH entry | `...\bin` appended to the machine-wide PATH |
| Install-dir registry key | `HKLM\Software\AMD\ROCm\<version>\InstallDir` |

### Optional: Long Path Support

Passing `ENABLE_LONG_PATHS=1` additionally writes:

| Registry key | Value | Data |
|---|---|---|
| `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem` | `LongPathsEnabled` | `1` (DWORD) |

This lifts the 260-character `MAX_PATH` limit system-wide. Requires Windows 11
or Windows Server 2019 or later. A reboot is needed for the change to propagate to all running
processes; processes started after the installer exits pick it up immediately.
The key is removed on uninstall.

## Upgrade

Install the new MSI directly — no manual uninstall required:

```bat
msiexec /i amdrocm-runtimes-<new-version>.msi /qn
```

Installing an older version over a newer one is blocked. Uninstall the current
version first if a downgrade is needed.

## Uninstall

**Using the MSI file:**

```bat
msiexec /x amdrocm-runtimes.msi /qn
```

**Using the product code (when the MSI file is unavailable):**

```bat
:: Find the product code
wmic product where "Name like 'ROCm%'" get Name,Version,IdentifyingNumber

:: Uninstall by product code
msiexec /x {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX} /qn
```

**Using Settings:** Apps > Installed apps > ROCm Runtime > Uninstall.

Uninstall removes all installed files, the PATH entry, and all registry keys
written by the installer. Files created after installation are not removed.

## Troubleshooting

### PATH not updated

PATH changes apply to newly opened terminals only. Open a new prompt and check:

```bat
echo %PATH%
```

`...\AMD\ROCm\runtimes-<version>\bin` should appear in the output.

### Installation fails

Capture a verbose log and check the last error:

```bat
msiexec /i amdrocm-runtimes.msi /qn /l*v "%TEMP%\rocm-install.log"
findstr /i "error" "%TEMP%\rocm-install.log"
```

### Long path errors

Enable long-path support at install time:

```bat
msiexec /i amdrocm-runtimes.msi /qn ENABLE_LONG_PATHS=1
```

Or set the registry key manually and reboot:

```bat
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

### Legacy DLL conflicts

Older ROCm installers placed DLLs such as `amdhip64_6.dll` directly into
`C:\Windows\System32\`, where they take precedence over the copies in Program
Files. The installer removes them automatically. If problems persist, check
manually:

```bat
dir C:\Windows\System32\amdhip64_*.dll
dir C:\Windows\System32\amd_comgr_*.dll
```

Delete any files found, then repair the installation:

```bat
msiexec /fvomus amdrocm-runtimes.msi /qn
```

## See Also

- [MSI generator script usage](msi-generator-usage.md)
