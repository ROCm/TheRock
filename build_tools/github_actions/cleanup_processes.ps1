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

$ps_list = Get-Process | Where-Object { $_.MainModule.FileName -Match $regex_build_exe }
$ps_list_len = $ps_list.Count

if($ps_list.Count -gt 0) {
    echo "[*] Found $ps_list_len running build executable(s):"
    $ps_list | % { echo "    > $($_.MainModule.FileName) "}

    echo "[*] Attempting to stop executable(s) forcefully..."
    $ps_list | ForEach-Object {
        echo "    > $($_.MainModule.ModuleName)"
        Stop-Process $_ -Force
    }

    $ps_list = Get-Process | Where-Object { $_.MainModule.FileName -Match $regex_build_exe }
    if($ps_list.Count -gt 0) {
        echo "[-] Failed to stop executable(s): "
        $ps_list | ForEach-Object {
            echo "    > $($_.MainModule.ModuleName)"
        }
        exit 1
    } else {
        echo "[+] All $ps_list_len executable(s) were stopped."
    }

} else {
    echo "[+] No executables to clean up."
}
