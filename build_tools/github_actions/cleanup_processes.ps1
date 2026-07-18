# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# cleanup_processes.ps1
#
# Optional Environment variable inputs:
#   GITHUB_WORKSPACE - Path to github workspace, typically to a git repo directory
#                      in which to look for a `build` folder with running executables
#                      (if undefined, defaults to equivalent glob of `**\build\**\*.exe`)
# Will exit with error code if not all processes are stopped for some reason
#
# This powershell script is used on non-ephemeral Windows runners to stop
# any processes that still exist after a Github actions job has completed
# typically due to timing out. Additionally, it will clean up the system
# environment to restore the system to a clean state for the next job.
#
# It's written in powershell per the specifications
# for github pre or post job scripts in this article:
# https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/run-scripts
#
#
# The typical next steps in the workflow would be to "Set up job" and "Checkout Repository"
# which refers to Github's official `actions/checkout` step that will delete the repo
# directory specified in the GITHUB_WORKSPACE environment variable. This will only
# succeed if no processes are running in that directory.
#
# This script will defer the step of deleting the repo directory to `actions/checkout`
#

#### Helper functions ####

function Get-Process-Filter ([String]$RegexStr)  {
    # Skip processes that have already exited: PowerShell's Get-Process can
    # surface stale handles for processes that died naturally but whose
    # process objects are still cached, and WMI Terminate is a silent no-op
    # on them. Without this guard, Wait-Process-Filter exhausts all of its
    # bounded retries on the zombie handle and the script then reports
    # "Failed to stop" on a process that already finished. The -and
    # short-circuits the .MainModule access for exited processes, where
    # that property is unreliable.
    Get-Process | Where-Object { -not $_.HasExited -and $_.MainModule.FileName -Match $RegexStr }
}
function Get-Process-Info ($PSobj) {
    # Note in powershell this output gets buffered and returned from this function
    return "[pid:$($PSobj.id)][HasExited:$($PSobj.HasExited)] $($PSobj.MainModule.ModuleName)"
}
function Wait-Process-Filter ([String]$RegexStr, [int] $Tries, [int] $Seconds = 1) {
    Write-Host "[*] Waiting up to $($Tries * $Seconds) seconds for processes to stop..."
    $ps_list_len = 0
    for($i = 0; $i -lt $Tries; $i++) {
        Start-Sleep -Seconds $Seconds
        $ps_list = Get-Process-Filter($RegexStr)
        $ps_list_len = $ps_list.Count
        if($ps_list_len -gt 0) {
            Write-Host "    > Waiting for $ps_list_len processes..."
            $ps_list | % {
                Write-Host "      $(Get-Process-Info $_)"
            }
        } else {
            Write-Host "    > Found no processes after waiting $(($i+1) * $Seconds) second(s)"
            return $true;
        }
    }
    Write-Host "    > Found $ps_list_len processes after waiting $($i * $Seconds) second(s)"
    return $false;
}


#### Script Start ####
echo "[*] ==== Starting cleanup_processes.ps1 ===="
# https://superuser.com/questions/749243/detect-if-powershell-is-running-as-administrator/756696#756696
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::
            GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
echo "[*] Checking if elevated: $isAdmin | current user: $currentUser"

#### Cleanup system environment ####
# Only perform system environment cleanup if running as NT AUTHORITY\* (any windows service account)
# as a safeguard from accidentally running this cleanup on a normal user account
if ($currentUser -match "NT AUTHORITY") {
    echo "[*] Running as a Windows Service (NT AUTHORITY\*) - Cleaning up system environment..."

    # Remove ~/.gitconfig file in case it was corrupted during a previous job
    $gitConfigPath = Join-Path $env:USERPROFILE ".gitconfig"

    echo "[*] > Checking for `"~/.gitconfig`" file at: $gitConfigPath"
    if (Test-Path $gitConfigPath) {
        echo "[*] >> .gitconfig file found, removing after logging contents:"

        # Returns non-zero exit code if invalid and outputs "fatal: bad config line 1..." to stderr
        git config --global --list
        try {
            Remove-Item -Path $gitConfigPath -Force -ErrorAction Stop
            echo "[+] >> Successfully removed .gitconfig"
        } catch {
            echo "[-] >> Warning: Failed to remove .gitconfig: $_"
        }
    } else {
        echo "[*] >> .gitconfig file not found at: $gitConfigPath"
    }

    # Remove temp uv setup archives and empty folders
    $uvTempDir = [System.IO.Path]::GetTempPath() # As System User: \Windows\SystemTemp
    echo "[*] > Checking for uv.zip archives under: $uvTempDir"

    $uvZipFiles = @(Get-ChildItem "$uvTempDir\**\uv.zip" -ErrorAction SilentlyContinue)
    $uvZipCount = $uvZipFiles.Count
    $uvZipBytes = ($uvZipFiles | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $uvZipBytes) { $uvZipBytes = 0 }
    $uvZipMB = [math]::Round($uvZipBytes / 1MB, 2)
    echo "[*] >> Found $uvZipCount uv.zip file(s), total size: $uvZipMB MB"

    if ($uvZipCount -gt 0) {
        try {
            $uvZipFiles | Remove-Item -Force -ErrorAction Stop
            echo "[+] >> Removed $uvZipCount uv.zip file(s)"
        } catch {
            echo "[-] >> Warning: Failed to remove uv.zip file(s): $_"
        }
    }

    # Remove empty GUID-like directories in temp, that uv setup most likely created
    echo "[*] > Checking for empty GUID directories under: $uvTempDir"
    $guidDirs = Get-ChildItem -Path $uvTempDir -Directory -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Name.Length -eq 36 -and ($_.Name.Split("-").Count -eq 5) }
    $emptyGuidDirs = @($guidDirs | Where-Object { (Get-ChildItem -LiteralPath $_.FullName).Count -eq 0 })
    $emptyGuidCount = $emptyGuidDirs.Count
    echo "[*] >> Found $emptyGuidCount empty GUID folder(s)"

    if ($emptyGuidCount -gt 0) {
        try {
            $emptyGuidDirs | Remove-Item -Force -ErrorAction Stop
            echo "[+] >> Removed $emptyGuidCount empty GUID folder(s)"
        } catch {
            echo "[-] >> Warning: Failed to remove empty GUID folder(s): $_"
        }
    }

    # Remove pip cache http-v2 files older than 1 day
    $pipHttpCacheDirs = @()
    if ($env:PIP_CACHE_DIR) {
        $pipHttpCacheDirs += Join-Path $env:PIP_CACHE_DIR "http-v2"
    }
    $pipHttpCacheDirs += Join-Path $env:LOCALAPPDATA "pip\cache\http-v2"

    foreach ($httpCacheDir in $pipHttpCacheDirs) {
        if (Test-Path $httpCacheDir) {
            echo "[*] > Checking pip http-v2 cache under: $httpCacheDir"

            $oneDayAgo = (Get-Date).AddDays(-1)
            $oldFiles = @(Get-ChildItem -Path $httpCacheDir -File -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.LastWriteTime -lt $oneDayAgo })

            $oldFileCount = $oldFiles.Count
            $oldFileBytes = ($oldFiles | Measure-Object -Property Length -Sum).Sum
            if ($null -eq $oldFileBytes) { $oldFileBytes = 0 }
            $oldFileMB = [math]::Round($oldFileBytes / 1MB, 2)

            echo "[*] >> Found $oldFileCount file(s) older than 1 day, total size: $oldFileMB MB"

            if ($oldFileCount -gt 0) {
                try {
                    $oldFiles | Remove-Item -Force -ErrorAction Stop
                    echo "[+] >> Removed $oldFileCount old file(s) from http-v2 cache"
                } catch {
                    echo "[-] >> Warning: Failed to remove old http-v2 cache files: $_"
                }

                # Remove empty directories
                $emptyDirs = @(Get-ChildItem -Path $httpCacheDir -Directory -Recurse -ErrorAction SilentlyContinue |
                    Where-Object { (Get-ChildItem -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue).Count -eq 0 } |
                    Sort-Object -Property FullName -Descending)

                if ($emptyDirs.Count -gt 0) {
                    try {
                        $emptyDirs | Remove-Item -Force -ErrorAction Stop
                        echo "[+] >> Removed $($emptyDirs.Count) empty directory(ies) from http-v2 cache"
                    } catch {
                        echo "[-] >> Warning: Failed to remove empty directories: $_"
                    }
                }
            }
        } else {
            echo "[*] > Pip http-v2 cache not found at: $httpCacheDir"
        }
    }
} else {
    echo "[*] Not Running as a Windows Service (NT AUTHORITY\*) - skipping system environment cleanup"
}

#### Cleanup stale git submodule working trees ####
# These runners are non-ephemeral and shared across super-repos whose checkouts
# pin DIFFERENT submodule commits. `actions/checkout` runs
# `git submodule foreach --recursive` during its auth setup/teardown, descending
# into leftover submodule working trees from a prior job.
# `git reset --hard` does not clear nested submodule working trees, so a newer
# nested submodule left on disk can remain under an older superproject whose
# .gitmodules lacks that URL. The recursive foreach then aborts with
# `fatal: No url found for submodule path ...` (exit 128) and the job dies
# before its own checkout begins.
#
# We cannot rely on `git submodule deinit`: a prior job can leave the workspace
# partially corrupted (e.g. submodule.<name>.url registered in .git/config while
# .git/modules/<name> is missing), which makes deinit abort with
# `could not lock config file .git/modules/<name>/config`. Instead we physically
# empty the top-level submodule working trees listed in .gitmodules. The
# recursive foreach only descends into populated submodule dirs, so emptying them
# leaves nothing for it to enter, independent of .git/modules health. We keep
# .git/modules objects (repopulation stays a local checkout, no network re-clone)
# and never touch ccache/deps (those are not submodule paths).
$workspaceRepo = $env:GITHUB_WORKSPACE
$gitmodulesPath = if ($workspaceRepo) { Join-Path $workspaceRepo ".gitmodules" } else { $null }
if ($workspaceRepo -and (Test-Path (Join-Path $workspaceRepo ".git")) -and $gitmodulesPath -and (Test-Path $gitmodulesPath)) {
    echo "[*] Clearing stale git submodule working trees in workspace: $workspaceRepo"
    # Read submodule paths via `git config -f`, which parses .gitmodules as a
    # standalone config file without touching repo state (no index, no
    # .git/modules), so it cannot trip on the corruption we are cleaning up.
    $submodulePaths = git config -f $gitmodulesPath --get-regexp '^submodule\..*\.path$' |
        ForEach-Object { ($_ -split '\s+', 2)[1].Trim() }
    foreach ($subPath in $submodulePaths) {
        $fullSubPath = Join-Path $workspaceRepo $subPath
        if (Test-Path $fullSubPath) {
            echo "      removing $subPath"
            try {
                Remove-Item -Recurse -Force -LiteralPath $fullSubPath -ErrorAction Stop
            } catch {
                echo "      WARNING: could not fully remove ${subPath}: $_"
            }
        }
    }
    echo "[*] >> Submodule working-tree cleanup finished"
} else {
    echo "[*] No git repo / .gitmodules at workspace root, skipping submodule cleanup"
}

#### Cleanup Processes ####
echo "[*] Cleaning up processes..."

# Note use of single '\' for Windows path separator, but it's escaped as '\\' for regex
$regex_build_exe = "\build\.*[.]exe"

# Some test runners have been setup with differing working directories
# etc. "C:\runner" vs "B:\actions-runner" so use the workspace env var
if($ENV:GITHUB_WORKSPACE -ne $null) {
    echo "[*] GITHUB_WORKSPACE env var defined: $ENV:GITHUB_WORKSPACE"
    $regex_build_exe = "$ENV:GITHUB_WORKSPACE\build\.*[.]exe"
} else {
    echo "[*] GITHUB_WORKSPACE env var undefined, using default regex"
}
$regex_build_exe = $regex_build_exe.Replace("\","\\")
echo "[*] Checking for running build executables filtered by .exe regex: $regex_build_exe"

$IsAllStopped = $false

$ps_list = Get-Process-Filter($regex_build_exe)
$ps_list_begin_len = $ps_list.Count

# exit early if no processes found
if($ps_list_begin_len -eq 0) {
    echo "[+] No executables to clean up."
    exit 0
}

# First Attempt with powershell `Stop-Process`
echo "[*] Found $ps_list_begin_len running build executable(s):"
$ps_list | % { echo "    > $($_.MainModule.FileName)"}

echo "[*] Attempting to stop executable(s) with WMI: "
$ps_list | ForEach-Object {
    #https://stackoverflow.com/questions/40585754/powershell-wont-terminate-hung-process
    echo "    > $(Get-Process-Info $_)"
    (Get-WmiObject win32_process -Filter "ProcessId = '$($_.id)'").Terminate() | Out-Null
}
$IsAllStopped = Wait-Process-Filter -RegexStr $regex_build_exe -Tries 5


# Second Attempt with `WMI` (if any processes are still running)
if(!$IsAllStopped) {
    $ps_list = Get-Process-Filter -RegexStr $regex_build_exe
    if($ps_list.Count -gt 0) {
        echo "[*] Attempting to stop any remaining executable(s) forcefully with 'Stop-Process':"
        $ps_list | ForEach-Object {
            #https://stackoverflow.com/questions/40585754/powershell-wont-terminate-hung-process
            echo "    > $(Get-Process-Info $_)"
            Stop-Process $_ -Force
        }
    }
    $IsAllStopped = Wait-Process-Filter -RegexStr $regex_build_exe -Tries 5
}

# Query list of processes again to see whether processes may have hung
# only if at the beginning of the script there were found processes to be stopped
if(!$IsAllStopped) {
    $ps_list = Get-Process-Filter -RegexStr $regex_build_exe
    if($ps_list.Count -gt 0) {
        echo "[-] Failed to stop executable(s): "
        $ps_list | ForEach-Object {
            Write-Host "    > $(Get-Process-Info $_)"
        }
        exit 1
    }
} else {
    echo "[+] All executable(s) were stopped."
}
