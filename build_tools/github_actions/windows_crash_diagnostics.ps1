# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("Setup", "Collect")]
  [string]$Mode,

  [Parameter(Mandatory = $true)]
  [string]$OutputDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$outputPath = [System.IO.Path]::GetFullPath($OutputDir)
$dumpPath = Join-Path $outputPath "crash-dumps"
$startTimePath = Join-Path $outputPath "start-time-utc.txt"

function Write-SystemSnapshot {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  try {
    "CapturedUtc: $([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $Path -Encoding utf8
    "`nOperatingSystem:" | Add-Content -Path $Path -Encoding utf8
    Get-CimInstance Win32_OperatingSystem |
      Select-Object Caption, Version, BuildNumber, TotalVisibleMemorySize, FreePhysicalMemory, TotalVirtualMemorySize, FreeVirtualMemory |
      Format-List | Out-String -Width 4096 | Add-Content -Path $Path -Encoding utf8

    "`nComputerSystem:" | Add-Content -Path $Path -Encoding utf8
    Get-CimInstance Win32_ComputerSystem |
      Select-Object Name, NumberOfLogicalProcessors, NumberOfProcessors, TotalPhysicalMemory |
      Format-List | Out-String -Width 4096 | Add-Content -Path $Path -Encoding utf8

    "`nPageFiles:" | Add-Content -Path $Path -Encoding utf8
    Get-CimInstance Win32_PageFileUsage |
      Select-Object Name, AllocatedBaseSize, CurrentUsage, PeakUsage |
      Format-List | Out-String -Width 4096 | Add-Content -Path $Path -Encoding utf8

    "`nPerformance:" | Add-Content -Path $Path -Encoding utf8
    Get-CimInstance Win32_PerfFormattedData_PerfOS_System |
      Select-Object Processes, Threads, SystemCallsPersec |
      Format-List | Out-String -Width 4096 | Add-Content -Path $Path -Encoding utf8
    Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory |
      Select-Object AvailableMBytes, CommittedBytes, CommitLimit, PercentCommittedBytesInUse, PoolNonpagedBytes, PoolPagedBytes |
      Format-List | Out-String -Width 4096 | Add-Content -Path $Path -Encoding utf8

    "`nTopProcesses:" | Add-Content -Path $Path -Encoding utf8
    Get-Process |
      Sort-Object WorkingSet64 -Descending |
      Select-Object -First 50 Id, ProcessName, HandleCount,
        @{Name = "Threads"; Expression = { $_.Threads.Count }},
        @{Name = "WorkingSetBytes"; Expression = { $_.WorkingSet64 }},
        @{Name = "PrivateMemoryBytes"; Expression = { $_.PrivateMemorySize64 }} |
      Format-Table -AutoSize | Out-String -Width 4096 | Add-Content -Path $Path -Encoding utf8
  }
  catch {
    "Failed to collect system snapshot: $_" | Add-Content -Path $Path -Encoding utf8
  }
}

function Export-EventChannel {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Channel,

    [Parameter(Mandatory = $true)]
    [string]$Name,

    [Parameter(Mandatory = $true)]
    [string]$Query
  )

  $textPath = Join-Path $outputPath "$Name.txt"
  $eventLogPath = Join-Path $outputPath "$Name.evtx"
  try {
    $textOutput = & wevtutil.exe qe $Channel "/q:$Query" /f:text /rd:true 2>&1
    $textExitCode = $LASTEXITCODE
    $textOutput | Out-File -FilePath $textPath -Encoding utf8
    "wevtutil text exit code: $textExitCode" | Add-Content -Path $textPath -Encoding utf8

    if ($textExitCode -eq 0) {
      $exportOutput = & wevtutil.exe epl $Channel $eventLogPath "/q:$Query" /ow:true 2>&1
      if ($LASTEXITCODE -ne 0) {
        "wevtutil EVTX export exit code: $LASTEXITCODE" | Add-Content -Path $textPath -Encoding utf8
        $exportOutput | Add-Content -Path $textPath -Encoding utf8
      }
    }
  }
  catch {
    "Failed to export '$Channel': $_" | Out-File -FilePath $textPath -Encoding utf8
  }
}

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

if ($Mode -eq "Setup") {
  New-Item -ItemType Directory -Path $dumpPath -Force | Out-Null
  [DateTime]::UtcNow.ToString("o") | Set-Content -Path $startTimePath -Encoding ascii

  $localDumpsRoot = "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps"
  foreach ($executable in @("python.exe", "python3.exe")) {
    $key = Join-Path $localDumpsRoot $executable
    New-Item -Path $key -Force | Out-Null
    New-ItemProperty -Path $key -Name DumpFolder -PropertyType ExpandString -Value $dumpPath -Force | Out-Null
    New-ItemProperty -Path $key -Name DumpType -PropertyType DWord -Value 2 -Force | Out-Null
    New-ItemProperty -Path $key -Name DumpCount -PropertyType DWord -Value 20 -Force | Out-Null
    Get-ItemProperty -Path $key | Format-List
  }

  Write-SystemSnapshot -Path (Join-Path $outputPath "system-start.txt")
  Write-Host "Configured full Python crash dumps in $dumpPath"
  exit 0
}

$startedAt = [DateTime]::UtcNow.AddHours(-12)
if (Test-Path $startTimePath) {
  $startedAt = [DateTime]::Parse((Get-Content -Path $startTimePath -Raw)).ToUniversalTime()
}
$elapsedMilliseconds = [Math]::Ceiling(([DateTime]::UtcNow - $startedAt).TotalMilliseconds + 60000)
$eventQuery = "*[System[TimeCreated[timediff(@SystemTime) <= $elapsedMilliseconds]]]"

Write-SystemSnapshot -Path (Join-Path $outputPath "system-end.txt")
Export-EventChannel -Channel "Application" -Name "events-application" -Query $eventQuery
Export-EventChannel -Channel "Microsoft-Windows-WER-Diag/Operational" -Name "events-wer" -Query $eventQuery
Export-EventChannel -Channel "Microsoft-Windows-Windows Defender/Operational" -Name "events-defender" -Query $eventQuery
Export-EventChannel -Channel "Microsoft-Windows-CodeIntegrity/Operational" -Name "events-code-integrity" -Query $eventQuery

$dumpManifestPath = Join-Path $outputPath "crash-dumps.txt"
if (Test-Path $dumpPath) {
  Get-ChildItem -Path $dumpPath -File -Recurse |
    Select-Object FullName, Length, CreationTimeUtc, LastWriteTimeUtc |
    Format-Table -AutoSize | Out-String -Width 4096 | Out-File -FilePath $dumpManifestPath -Encoding utf8
}
else {
  "Crash dump directory not found: $dumpPath" | Out-File -FilePath $dumpManifestPath -Encoding utf8
}

$localDumpsRoot = "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps"
foreach ($executable in @("python.exe", "python3.exe")) {
  $key = Join-Path $localDumpsRoot $executable
  if (Test-Path $key) {
    Remove-Item -Path $key -Recurse -Force
  }
}

Write-Host "Collected Windows crash diagnostics in $outputPath"
exit 0
