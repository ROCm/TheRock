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
# typically due to timing out. It's written in powershell per the specifications
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
    Get-Process | Where-Object { $_.MainModule.FileName -Match $RegexStr }
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

# Note use of single '\' for Windows path separator, but it's escaped as '\\' for regex
$regex_build_exe = "\build\.*[.]exe"

# https://superuser.com/questions/749243/detect-if-powershell-is-running-as-administrator/756696#756696
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::
            GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
echo "[*] Checking if elevated: $isAdmin | current user: $currentUser"

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
