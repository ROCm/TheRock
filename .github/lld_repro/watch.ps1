# Background watcher for the shark69 lld-link reproducer (ROCm/TheRock#9469).
# Waits until a finished clang/lld-linked sub-project with >= $MinObjs objects exists
# in the live build tree, then loops repro_link.ps1 against it while the build keeps
# running, until $StopFile appears.
param([Parameter(Mandatory)][string]$BuildRoot, [Parameter(Mandatory)][string]$ReproRoot,
      [Parameter(Mandatory)][string]$StopFile, [Parameter(Mandatory)][string]$Probe,
      [Parameter(Mandatory)][string]$Log, [int]$MinObjs = 80, [int]$Par = 32, [int]$PollSec = 120)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
function W([string]$m) { Add-Content -Path $Log -Value ("{0:u} {1}" -f (Get-Date).ToUniversalTime(), $m) }
W "WATCH start build=$BuildRoot repro=$ReproRoot minobjs=$MinObjs par=$Par"
$round = 0
while (-not (Test-Path $StopFile)) {
    $round++
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $here 'repro_link.ps1') `
        -BuildRoot $BuildRoot -ReproRoot $ReproRoot -StopFile $StopFile -MinObjs $MinObjs `
        -Probe $Probe -Par $Par -Iters 100000000 -Map 2>&1 | ForEach-Object { W "r$round $_" }
    $rc = $LASTEXITCODE
    W "ROUND $round exit=$rc"
    if (Test-Path $StopFile) { break }
    # 2 = no candidate yet / setup issue, 3 = baseline link failed: retry after the build progresses.
    Start-Sleep -Seconds $PollSec
}
W "WATCH stop rounds=$round"
