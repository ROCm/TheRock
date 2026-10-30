# Preflight for the shark69 lld-link reproducer (ROCm/TheRock#9469).
# Reports host/toolchain facts, picks a build root on the fixed drive with the most
# free space, and fails early if the machine cannot run a math-libs stage build.
param([int]$MinFreeGB = 100)
$ErrorActionPreference = 'Continue'
$fail = @()

$cpu = Get-CimInstance Win32_Processor
$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$ramGB = [math]::Round($cs.TotalPhysicalMemory / 1GB)
$logical = ($cpu | Measure-Object NumberOfLogicalProcessors -Sum).Sum
$cores = ($cpu | Measure-Object NumberOfCores -Sum).Sum
"HOST $env:COMPUTERNAME user=$env:USERNAME"
"CPU  $(($cpu | Select-Object -First 1).Name.Trim()) sockets=$(@($cpu).Count) cores=$cores logical=$logical"
"RAM  ${ramGB} GB"
"OS   $($os.Caption) $($os.Version) build=$($os.BuildNumber)"
$ucode = (Get-ItemProperty 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0' -EA SilentlyContinue).'Update Revision'
if ($ucode) { "UCODE $([BitConverter]::ToString($ucode))" }
"HYPERVISOR present=$($cs.HypervisorPresent) model=$($cs.Model)"
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
"ADMIN $admin"
$lp = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -EA SilentlyContinue).LongPathsEnabled
"LONGPATHS $lp"
try { $mp = Get-MpComputerStatus -EA Stop; "DEFENDER realtime=$($mp.RealTimeProtectionEnabled)" } catch { "DEFENDER unknown" }

foreach ($t in 'git', 'bash', 'python', 'cmake', 'ninja', 'ccache', 'perl', 'choco', 'gh', '7z') {
    $c = Get-Command $t -EA SilentlyContinue
    if ($c) { "TOOL $t = $($c.Source)" } else { "TOOL $t = MISSING" }
}
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vs = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -latest -property installationPath
    $vsv = & $vswhere -products * -latest -property installationVersion
    "VS   path=$vs version=$vsv"
    if (-not $vs) { $fail += 'MSVC VC.Tools.x86.x64 not installed' }
} else { "VS   vswhere MISSING"; $fail += 'Visual Studio / Build Tools not installed' }
$csc = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
"CSC  $(Test-Path $csc)"
if (-not (Test-Path $csc)) { $fail += 'in-box csc.exe missing (needed for xproc_probe)' }

$drives = Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Sort-Object FreeSpace -Descending
foreach ($d in $drives) { "DISK $($d.DeviceID) free=$([math]::Round($d.FreeSpace/1GB))GB size=$([math]::Round($d.Size/1GB))GB fs=$($d.FileSystem)" }
$best = $drives | Select-Object -First 1
$freeGB = [math]::Round($best.FreeSpace / 1GB)
if ($freeGB -lt $MinFreeGB) { $fail += "most free fixed drive $($best.DeviceID) has ${freeGB}GB < ${MinFreeGB}GB" }
# Short root keeps TheRock's deep paths under MAX_PATH, like B:\build in CI.
$root = "$($best.DeviceID)\lr"
# Leave ~2.5 GB RAM per compile job; heavy HIP compiles can exceed 2 GB each.
$jobs = [math]::Max(8, [math]::Min($logical, [math]::Floor($ramGB / 2.5)))
"PICK root=$root jobs=$jobs"

$busy = Get-Process -EA SilentlyContinue | Where-Object { $_.ProcessName -match '^(clang|clang\+\+|lld-link|ninja|cl|link)$' }
"OTHER_BUILD_PROCS $(@($busy).Count)"

if ($env:GITHUB_OUTPUT) {
    "root=$root" | Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
    "jobs=$jobs" | Out-File -FilePath $env:GITHUB_OUTPUT -Append -Encoding utf8
}
if ($fail) { "PREFLIGHT_FAIL: $($fail -join '; ')"; exit 1 }
"PREFLIGHT_OK"
