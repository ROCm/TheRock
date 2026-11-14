#!/usr/bin/env pwsh
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

<#
.SYNOPSIS
    Checks or sets up the MSYS2 environment that builds rocgdb on Windows.

.DESCRIPTION
    rocgdb and the autotools sysdeps it links (gmp, mpfr, expat, ncurses
    and libiconv) are built by MSYS2's UCRT64 gcc, through MSYS2's shell.

    With no options, this reports whether -InstallDir holds an MSYS2
    installation ready for the build, meaning that it has the packages that
    the build and the gdb testsuite use.  It also reports whether /tmp and
    the home directory are configured as below, which only -Pack requires.
    It changes nothing, and exits with status 1 if the installation is not
    ready for the build.

    With -Install, it downloads MSYS2's base archive, extracts it to
    -InstallDir, brings it up to date, and installs those packages.  Run
    again over an installation it created, it skips the download and
    brings the installation up to date.  Configure TheRock with
    -DTHEROCK_MSYS2_ROOT=<InstallDir> unless it is the default, C:\msys64.

    The installation is configured to hold no per-user state: /tmp maps to
    the Windows user's temp directory, and the home directory to the
    Windows one.  It can then be moved or copied to another directory, and
    does not need to be writable.  Because that changes how every MSYS2
    shell of that installation behaves, -Install leaves alone an existing
    installation it did not create, unless given -Force.

    With -Pack, it packs the installation into an archive, for unpacking
    elsewhere instead of running this script there, and refuses an
    installation that is not ready for the build or whose /tmp and home
    directory are not configured as above.  On its own, -Pack packs the
    installation as it is, without updating it, so that an installation
    checked by hand is what gets packed.  With -Install, it packs once the
    installation is set up, so that the archive holds nothing but what this
    script did.  Unpacking it into another directory and testing that
    copy then tests the archive itself.

    See docs/development/windows_support.md.

.PARAMETER InstallDir
    The MSYS2 installation to check, set up or pack.  Defaults to
    C:\msys64.

.PARAMETER Install
    Create the installation, or bring one this script created up to date.

.PARAMETER Force
    With -Install, also update and reconfigure an existing installation
    that this script did not create.  It is then marked as set up by this
    script, and later runs need no -Force.

.PARAMETER Pack
    Pack the installation into this archive, replacing any existing file,
    after setting it up if -Install is also given.
    The suffix picks the compression, as with tar -a: .tar.zst, .tar.xz,
    .tar.gz or .zip.  The archive holds a single top-level directory named
    like -InstallDir.

.PARAMETER Help
    Show this help.

.EXAMPLE
    .\build_tools\setup_msys2.ps1

.EXAMPLE
    .\build_tools\setup_msys2.ps1 -Install

.EXAMPLE
    .\build_tools\setup_msys2.ps1 -Install -InstallDir D:\msys64

.EXAMPLE
    .\build_tools\setup_msys2.ps1 -InstallDir D:\msys64 -Pack D:\msys2-therock.tar.zst

.EXAMPLE
    .\build_tools\setup_msys2.ps1 -Install -InstallDir D:\msys64 -Pack D:\msys2-therock.tar.zst
#>
# -Pack shares a parameter set with -Install and -Force.  Were it also in a set
# of its own, PowerShell could not tell which set -Pack alone selects.
[CmdletBinding(DefaultParameterSetName = "Check")]
param(
    [Parameter(ParameterSetName = "Check")]
    [Parameter(ParameterSetName = "Change")]
    [string]$InstallDir = "C:\msys64",
    [Parameter(ParameterSetName = "Change")]
    [switch]$Install,
    [Parameter(ParameterSetName = "Change")]
    [switch]$Force,
    [Parameter(ParameterSetName = "Change")]
    [string]$Pack = "",
    [Parameter(ParameterSetName = "Help")]
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Help) {
    Get-Help $PSCommandPath -Detailed
    exit 0
}
# The parameter sets keep -Force away from the other modes, but cannot require
# -Install for it without PowerShell prompting for it.
if ($Force -and -not $Install) {
    throw "-Force only applies with -Install"
}

# The MSYS2 base archive, pinned by release and hash.  This follows the
# "Other Systems" recipe at https://www.msys2.org/docs/ci/.  The packages are
# not pinned: pacman installs and updates to whatever the MSYS2 mirrors serve
# when this runs.
$Msys2Release = "2026-09-27"
$Msys2Archive = "msys2-base-x86_64-$($Msys2Release -replace '-', '').sfx.exe"
$Msys2Url = "https://github.com/msys2/msys2-installer/releases/download/$Msys2Release/$Msys2Archive"
$Msys2Sha256 = "ad336cccfda47758b5e15cda993fbba421115cb0b126697daef1ee4dfe37209f"

# What rocgdb and the autotools sysdeps build with, and what the gdb
# testsuite runs under (dejagnu).  The base system every MSYS2 installation
# has supplies bash, sed, grep, gawk and the like.
$Packages = @(
    "bison"
    "dejagnu"
    "diffutils"
    "flex"
    "m4"
    "make"
    "mingw-w64-ucrt-x86_64-gcc"
    "mingw-w64-ucrt-x86_64-pkgconf"
    "texinfo"
)

# The /etc/fstab and /etc/nsswitch.conf lines that keep per-user state out of
# the installation.  See
# https://cygwin.com/cygwin-ug-net/using.html#usertemp and
# https://cygwin.com/cygwin-ug-net/ntsec.html#ntsec-mapping-nsswitch-home
$TmpEntry = "none /tmp usertemp binary,posix=0 0 0"
$HomeEntry = "db_home: env windows cygwin desc"

# Written into installations this script creates.  -Install only updates and
# reconfigures an existing installation that has it, so that it does not
# change someone's own MSYS2 behind their back.
$MarkerRelPath = "etc\therock-setup-msys2"

# Make a path given on the command line absolute.  A relative one is taken
# relative to PowerShell's current location, as the user means it, instead of
# [System.IO.Path]::GetFullPath()'s process working directory, which
# Set-Location does not change.
function Resolve-UserPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

$InstallDir = Resolve-UserPath $InstallDir
$Marker = Join-Path $InstallDir $MarkerRelPath

# Run a command in the MSYS2 installation's bash.
function Invoke-Msys2 {
    param([Parameter(Mandatory = $true)][string]$Command)

    # Windows PowerShell passes a double quote inside an argument to a native
    # program unescaped, and pwsh 7.3 and later escapes it, so no single way of
    # writing one works under both.  The commands here have no need for one.
    if ($Command.Contains('"')) {
        throw "Invoke-Msys2 does not support double quotes: $Command"
    }
    & "$InstallDir\usr\bin\env.exe" LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 `
        /usr/bin/bash -lc $Command
    if ($LASTEXITCODE -ne 0) {
        throw "MSYS2 command failed (exit code $LASTEXITCODE): $Command"
    }
}

# Report whether $InstallDir is ready for the build, without changing it, and
# whether /tmp and the home directory are configured as -Install does.
# Returns $true if it is ready for the build, and with -ForPack, if /tmp and
# the home directory are configured too.  The build itself needs only the
# packages, so an MSYS2 of someone's own with the packages is ready for it.
function Test-Msys2Installation {
    param([switch]$ForPack)
    if (-not (Test-Path "$InstallDir\usr\bin\bash.exe")) {
        Write-Host "$InstallDir holds no MSYS2 installation."
        return $false
    }
    $ready = $true
    if (Test-Path $Marker) {
        Write-Host "$InstallDir holds an MSYS2 installation set up by this script."
    } else {
        Write-Host "$InstallDir holds an MSYS2 installation not set up by this script."
    }

    # pacman.exe run directly instead of through a login shell, which would
    # initialize a fresh installation.  -Qq lists those of the packages that
    # are installed, and complains about the others on stderr.  Windows
    # PowerShell turns a native program's redirected stderr lines into error
    # records, which the script-wide "Stop" would make fatal on the first
    # missing package, so the call runs in a scope that lets them pass.
    $installed = @(& {
        $ErrorActionPreference = "Continue"
        & "$InstallDir\usr\bin\pacman.exe" -Qq @Packages 2>$null
    })
    $missing = @($Packages | Where-Object { $installed -notcontains $_ })
    if ($missing.Count -eq 0) {
        Write-Host "  packages: all $($Packages.Count) installed"
    } else {
        Write-Host "  packages: missing $($missing -join ', ')"
        $ready = $false
    }

    $fstab = @(Get-Content "$InstallDir\etc\fstab" -ErrorAction SilentlyContinue)
    if ($fstab -contains $TmpEntry) {
        Write-Host "  /tmp: the Windows temp directory"
    } else {
        Write-Host "  /tmp: inside the installation (not configured, which only -Pack requires)"
        if ($ForPack) { $ready = $false }
    }
    $nsswitch = @(Get-Content "$InstallDir\etc\nsswitch.conf" -ErrorAction SilentlyContinue)
    if ($nsswitch -contains $HomeEntry) {
        Write-Host "  home directory: the Windows one"
    } else {
        Write-Host "  home directory: inside the installation (not configured, which only -Pack requires)"
        if ($ForPack) { $ready = $false }
    }
    return $ready
}

# Pack $InstallDir into the archive $Pack names, replacing any existing file.
# Throws if the installation is not ready for the build, or its /tmp and home
# directory are not configured, as an archive of it would not be relocatable.
function Compress-Msys2Installation {
    if (-not (Test-Msys2Installation -ForPack)) {
        throw "$InstallDir is not ready to be packed"
    }
    $archivePath = Resolve-UserPath $Pack
    Write-Host "Packing $InstallDir into $archivePath"
    if (Test-Path $archivePath) {
        Remove-Item $archivePath
    }
    # Windows' own tar, so that the archive is made by the same tool that a
    # machine without MSYS2 unpacks it with.  /tmp and /home hold only what
    # running the installation left behind.
    $tar = Join-Path $env:SystemRoot "System32\tar.exe"
    $leaf = Split-Path -Leaf $InstallDir
    & $tar -a -c -f $archivePath -C (Split-Path -Parent $InstallDir) `
        --exclude "$leaf/tmp/*" --exclude "$leaf/home/*" $leaf
    if ($LASTEXITCODE -ne 0) {
        throw "Packing $InstallDir failed (exit code $LASTEXITCODE)"
    }
    Write-Host "Done.  $InstallDir is packed into $archivePath"
}

if ($Pack -and -not $Install) {
    Compress-Msys2Installation
    exit 0
}

if (-not $Install) {
    if (Test-Msys2Installation) {
        Write-Host "Ready for the build."
        exit 0
    }
    Write-Host ("Not ready for the build.  Install the missing packages with pacman, " +
        "or run with -Install to set the installation up, or -Help for usage.")
    exit 1
}

if (Test-Path "$InstallDir\usr\bin\bash.exe") {
    if (-not (Test-Path $Marker) -and -not $Force) {
        throw ("$InstallDir holds an MSYS2 installation that this script did not set up.  " +
            "-Install would update it and change where its /tmp and home directory are.  " +
            "Run with -Force to do that anyway, or choose another -InstallDir.")
    }
    Write-Host "Updating the existing MSYS2 installation in $InstallDir"
} else {
    if (Test-Path $InstallDir) {
        throw "$InstallDir exists, but holds no MSYS2 installation"
    }

    # The archive always extracts to a directory named msys64, so extract
    # beside the requested directory and rename if that has another name.
    $parent = Split-Path -Parent $InstallDir
    $extracted = Join-Path $parent "msys64"
    if ($extracted -ne $InstallDir -and (Test-Path $extracted)) {
        throw "$extracted is in the way of extracting MSYS2 for $InstallDir"
    }
    New-Item -ItemType Directory -Force -Path $parent | Out-Null

    $archive = Join-Path ([System.IO.Path]::GetTempPath()) $Msys2Archive
    Write-Host "Downloading $Msys2Url"
    # Where .NET picks the protocols itself (SystemDefault), leave it be.
    # Elsewhere Windows PowerShell may offer only SSL 3 and TLS 1.0, which
    # GitHub refuses, so add TLS 1.2.  SystemDefault is compared as its value,
    # 0, because the name only exists from .NET 4.7, and the older setups are
    # the ones this is for.
    $protocols = [Net.ServicePointManager]::SecurityProtocol
    if ([int]$protocols -ne 0) {
        [Net.ServicePointManager]::SecurityProtocol =
            $protocols -bor [Net.SecurityProtocolType]::Tls12
    }
    (New-Object System.Net.WebClient).DownloadFile($Msys2Url, $archive)
    $hash = (Get-FileHash -Algorithm SHA256 $archive).Hash
    if ($hash -ne $Msys2Sha256) {
        Remove-Item $archive
        throw "$Msys2Archive has SHA256 $hash, expected $Msys2Sha256"
    }

    Write-Host "Extracting to $parent"
    & $archive -y "-o$parent" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Extracting $Msys2Archive failed (exit code $LASTEXITCODE)"
    }
    Remove-Item $archive
    if ($extracted -ne $InstallDir) {
        Move-Item $extracted $InstallDir
    }
}
Set-Content -Path $Marker -Value "Set up by TheRock's build_tools/setup_msys2.ps1."

# The first login initializes the installation, pacman's keyring included.
Write-Host "Initializing"
Invoke-Msys2 " "

# Twice: the first round may update only pacman and the core system, which
# then has to restart before the rest can be updated.
Write-Host "Updating"
Invoke-Msys2 "pacman --noconfirm --needed -Syuu"
Invoke-Msys2 "pacman --noconfirm --needed -Syuu"

Write-Host "Installing packages"
Invoke-Msys2 "pacman --noconfirm --needed -S $($Packages -join ' ')"

# The downloaded package files are not needed once installed, and would only
# make the installation bigger to copy.
Invoke-Msys2 "pacman --noconfirm -Scc"

# Record exactly what was installed, readable without running anything.
Invoke-Msys2 "pacman -Q > /package_list.txt"

Write-Host "Configuring /tmp and the home directory"
Invoke-Msys2 "grep -Fqx '$TmpEntry' /etc/fstab || echo '$TmpEntry' >> /etc/fstab"
Invoke-Msys2 ("if grep -q '^db_home:' /etc/nsswitch.conf; then " +
    "sed -i 's|^db_home:.*|$HomeEntry|' /etc/nsswitch.conf; " +
    "else echo '$HomeEntry' >> /etc/nsswitch.conf; fi")

Write-Host ""
Write-Host "Done.  MSYS2 is set up in $InstallDir"

if ($Pack) {
    Write-Host ""
    Compress-Msys2Installation
}
