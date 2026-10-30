# Real-toolchain reproducer for #5024, run INSIDE a live runner pod.
# Adapted for shark69 (ROCm/TheRock#9469): -BuildRoot/-ReproRoot instead of B:, -StopFile, -MinObjs.
# Copies the objects + .rsp of the largest already-built link target of a finished
# sub-project into B:\_repro (the live build tree is never modified), then loops:
#   1. rewrite every object in place from $Par short-lived processes (ccache-restore shape)
#   2. immediately run the target's real link command (from `ninja -t commands`)
# and counts links that fail with "undefined symbol".
param([int]$Iters = 100, [int]$Par = 32, [string]$Probe = 'C:\xproc_probe2.exe', [string]$Sub = '',
      [string]$Mode = 'plain', [switch]$PreRead, [int]$DelayMs = 0, [string]$LinkEnv = '', [switch]$Map, [int]$MaxEvents = 0, [switch]$Reuse,
      [int]$MaxMinutes = 0, [string]$Shim = '', [string]$ShimDir = 'C:\lldzero', [string]$ShimEnv = '',
      [int]$LinkAffinity = 0, [string]$AffLauncher = 'C:\lldzero\affinity_launch.exe',
      [string]$BuildRoot = 'B:\build', [string]$ReproRoot = 'B:\_repro', [string]$StopFile = '', [int]$MinObjs = 0)
if (-not (Test-Path $Probe)) { $Probe = Join-Path $ShimDir 'xproc_probe2.exe' }
if (-not (Test-Path $Probe)) { 'NO_PROBE'; exit 2 }
$ErrorActionPreference = 'Stop'
# AKS windef runners create B: with `subst` in the runner entrypoint; subst mappings are
# per logon session, so a kubectl-exec session has to map it again (same backing dir).
if (-not (Test-Path 'B:\') -and (Test-Path 'C:\scratch-b')) { subst B: C:\scratch-b | Out-Null }
$ninja = @('C:\ProgramData\chocolatey\lib\ninja\tools\ninja.exe', 'C:\ProgramData\chocolatey\bin\ninja.exe') | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $ninja) { $ninja = (Get-Command ninja -EA SilentlyContinue).Source }
if (-not $ninja) { $ninja = Get-ChildItem C:\home\runner\_work\_tool, C:\hostedtoolcache, C:\Python*, $env:LOCALAPPDATA -Recurse -Filter ninja.exe -EA SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName }
if (-not $ninja) { 'NO_NINJA'; exit 2 }
"ENV cpu=$((Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()) os=$((Get-CimInstance Win32_OperatingSystem).Version) host=$env:COMPUTERNAME ninja=$ninja"

function LinkEdges([string]$dir) {
    # yields @{target; nobj} for every exe/dll link edge in build.ninja
    foreach ($ln in [IO.File]::ReadLines((Join-Path $dir 'build.ninja'))) {
        if ($ln -match '^build (.+?)(?<!\$): (CXX|C)_(EXECUTABLE)_LINKER\S* (.*)$') {
            $n = ([regex]::Matches($Matches[4], '\.obj(?=\s|$)')).Count
            [pscustomobject]@{ target = ($Matches[1] -split '(?<!\$) ')[0].Replace('$:', ':').Replace('$ ', ' '); nobj = $n }
        }
    }
}
if (-not $Sub) {
    $cands = Get-ChildItem $BuildRoot -Recurse -Depth 4 -Filter build.stamp -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -eq 'stamp' } | ForEach-Object { Join-Path $_.Directory.Parent.FullName 'build' } |
        Where-Object { Test-Path (Join-Path $_ 'build.ninja') } |
        # Only clang/lld-linked sub-projects; host tools use MSVC link.exe, which needs vcvars.
        # The compiler comes from a toolchain file, so CMakeCache.txt does not name it; the link rule does.
        Where-Object { Select-String -Quiet -Path (Join-Path $_ 'CMakeFiles\rules.ninja') -Pattern 'clang|lld-link' }
    $best = $null; $bestN = 0
    foreach ($c in $cands) { foreach ($e in LinkEdges $c) { if ($e.nobj -gt $bestN) { $bestN = $e.nobj; $best = @($c, $e.target) } } }
    if (-not $best -or $bestN -lt $MinObjs) { "NO_CANDIDATE best=$bestN min=$MinObjs"; exit 2 }
    $Sub = $best[0]; $target = $best[1]
} else {
    $target = (LinkEdges $Sub | Sort-Object nobj -Descending | Select-Object -First 1).target
}
"PICK sub=$Sub target=$target"
# Re-run only this link edge in place so ninja regenerates (and keeps) its .rsp.
# Inputs are unchanged, so the relinked output is equivalent to what was there.
$tpath = Join-Path $Sub $target
Move-Item $tpath "$tpath.reprobak" -Force -ErrorAction SilentlyContinue
$kr = & $ninja -C $Sub -d keeprsp $target 2>&1
if ($LASTEXITCODE -ne 0) {
    # Put the original output back so a failed pick never damages the live build tree.
    if (Test-Path "$tpath.reprobak") { Move-Item "$tpath.reprobak" $tpath -Force }
    "KEEPRSP_FAILED"; $kr | Select-Object -Last 10; exit 2
}
Remove-Item "$tpath.reprobak" -Force -ErrorAction SilentlyContinue
$bn = Get-Content (Join-Path $Sub 'build.ninja')
$rspName = $null
for ($i = 0; $i -lt $bn.Count; $i++) {
    if ($bn[$i].StartsWith("build $($target.Replace(':', '$:').Replace(' ', '$ '))") -or $bn[$i].StartsWith("build $target")) {
        for ($j = $i + 1; $j -lt [Math]::Min($i + 40, $bn.Count) -and $bn[$j].StartsWith(' '); $j++) {
            if ($bn[$j] -match 'RSP_FILE = CMakeFiles\\(\S+\.rsp)') { $rspName = $Matches[1]; break }
        }
        break
    }
}
if (-not $rspName -or -not (Test-Path (Join-Path $Sub "CMakeFiles\$rspName"))) { "NO_RSP for $target ($rspName)"; exit 2 }
$cmd = (& $ninja -C $Sub -t commands $target | Select-Object -Last 1)

$dst = Join-Path $ReproRoot (Split-Path (Split-Path $Sub) -Leaf)
if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
New-Item -ItemType Directory -Force (Join-Path $dst 'CMakeFiles') | Out-Null
$rspText = Get-Content -Raw (Join-Path $Sub "CMakeFiles\$rspName")
$tokens = [regex]::Matches($rspText + ' ' + $cmd, '[^\s"=:]+\.(obj|lib|res|def|exp|manifest)\b') | ForEach-Object { $_.Value } | Sort-Object -Unique
$rel = @()
foreach ($o in $tokens) {
    if ([IO.Path]::IsPathRooted($o)) { continue }
    $s = Join-Path $Sub $o; if (-not (Test-Path $s -PathType Leaf)) { continue }
    $d = Join-Path $dst $o
    New-Item -ItemType Directory -Force (Split-Path $d) | Out-Null
    Copy-Item $s $d
    if ($o -like '*.obj') { $rel += $d }
}
Copy-Item (Join-Path $Sub "CMakeFiles\$rspName") (Join-Path $dst "CMakeFiles\$rspName")
# Pre-create every output directory the link writes (relative /out:, /implib:, /pdb:, --intdir=).
foreach ($m in [regex]::Matches($cmd, '(?:/out:|/implib:|/pdb:|--intdir=|\s-o\s+|-Wl,--out-implib,|-Wl,/implib:|-Wl,/pdb:)("?)([^\s",]+)')) {
    $p = $m.Groups[2].Value; if (-not [IO.Path]::IsPathRooted($p)) { $p = Join-Path $dst $p }
    $dir = if ($m.Value -like '--intdir=*') { $p } else { Split-Path $p }
    New-Item -ItemType Directory -Force $dir | Out-Null
}
$list = Join-Path $dst '_objs.txt'; Set-Content -Path $list -Value $rel -Encoding ascii
$bytes = ($rel | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum).Sum
"REPRO sub=$Sub target=$target objs=$($rel.Count) MB=$([int]($bytes/1MB)) iters=$Iters par=$Par"
"CMD " + $cmd.Substring(0, [Math]::Min(400, $cmd.Length))

Set-Content -Path (Join-Path $dst '_link.bat') -Value ('@' + $cmd) -Encoding ascii

# ---- lld-link injection shim (optional) ----
# $Shim in off|detect|fix installs lldzero_launch.exe as lld-link.exe in a hardlinked
# mirror of the toolchain bin dir, so clang/clang-linker-wrapper spawn the launcher,
# which injects lldzero_hook.dll into the real lld-link. The resource-dir and hip-path
# in the link command are explicit absolute paths, so only the linker is interposed.
if ($Shim) {
    $clangExe = [regex]::Match($cmd, '([A-Za-z]:[\\/][^\s"]*?(?:clang\+\+|clang-cl|clang)\.exe)').Groups[1].Value
    if (-not $clangExe) { "SHIM_NO_CLANG"; exit 4 }
    $binDir = Split-Path $clangExe
    $tcBin = Join-Path $dst 'tcbin'
    if (Test-Path $tcBin) { Remove-Item $tcBin -Recurse -Force }
    New-Item -ItemType Directory -Force $tcBin | Out-Null
    foreach ($fsrc in Get-ChildItem $binDir -File) {
        $link = Join-Path $tcBin $fsrc.Name
        if ($fsrc.Name -ieq 'lld-link.exe') { continue } # replaced by launcher
        cmd /c mklink /H "`"$link`"" "`"$($fsrc.FullName)`"" > $null 2>&1
        if (-not (Test-Path $link)) { Copy-Item $fsrc.FullName $link }
    }
    # real lld-link kept as lld-link-real.exe; launcher installed as lld-link.exe
    $realLld = Join-Path $binDir 'lld-link.exe'
    cmd /c mklink /H "`"$(Join-Path $tcBin 'lld-link-real.exe')`"" "`"$realLld`"" > $null 2>&1
    if (-not (Test-Path (Join-Path $tcBin 'lld-link-real.exe'))) { Copy-Item $realLld (Join-Path $tcBin 'lld-link-real.exe') }
    Copy-Item (Join-Path $ShimDir 'lldzero_launch.exe') (Join-Path $tcBin 'lld-link.exe')
    Copy-Item (Join-Path $ShimDir 'lldzero_hook.dll')   (Join-Path $tcBin 'lldzero_hook.dll')
    # point the link command's clang at the mirrored bin
    $cmd2 = $cmd.Replace($binDir, $tcBin)
    Set-Content -Path (Join-Path $dst '_link.bat') -Value ('@' + $cmd2) -Encoding ascii
    $env:LLDZERO_MODE = $Shim
    $env:LLDZERO_LOG = (Join-Path $dst '_shim.log')
    if ($Shim -eq 'off') { Remove-Item Env:LLDZERO_MODE -ErrorAction SilentlyContinue; $env:LLDZERO_MODE = 'off' }
    Remove-Item $env:LLDZERO_LOG -ErrorAction SilentlyContinue
    # extra hook/launcher settings, e.g. -ShimEnv 'LLDZERO_AFFINITY=1;LLDZERO_FRESHVA=1'
    foreach ($kv in ($ShimEnv -split ';' | Where-Object { $_ -match '=' })) { $k, $v = $kv -split '=', 2; Set-Item "Env:$($k.Trim())" $v.Trim() }
    "SHIM mode=$Shim tcbin=$tcBin clang=$clangExe env=[$ShimEnv]"
}

# Baseline link must succeed, otherwise the copy is incomplete and results are meaningless.
if ($LinkEnv) { $env:_LINK_ = $LinkEnv }
if ($Map) { $env:_LINK_ = (($env:_LINK_ + ' /map:' + (Join-Path $dst '_cur.map')).Trim()) }
"VARIANT mode=$Mode preread=$PreRead delayms=$DelayMs _LINK_=[$env:_LINK_]"
Push-Location $dst
# -LinkAffinity N: run the whole link step (clang++ and the lld-link it spawns, by
# inheritance) pinned to N random CPUs via affinity_launch.exe; nothing is injected.
if ($LinkAffinity -gt 0 -and -not (Test-Path $AffLauncher)) { 'NO_AFFLAUNCHER'; Pop-Location; exit 2 }
if ($LinkAffinity -gt 0) { $env:AFFINITY_LOG = Join-Path $dst '_aff.log'; Remove-Item $env:AFFINITY_LOG -EA SilentlyContinue }
function Run-Link {
    if ($LinkAffinity -gt 0) { & $AffLauncher --cpus $LinkAffinity cmd.exe /c "_link.bat 2>&1" }
    else { cmd /c "_link.bat 2>&1" }
}
$out = Run-Link; $rc = $LASTEXITCODE
if ($rc -ne 0) { "BASELINE_FAILED rc=$rc"; $out | Select-Object -Last 15; Pop-Location; exit 3 }
$outFile = [regex]::Match($cmd, '(?:/out:|\s-o\s+)"?([^\s"]+)').Groups[1].Value; if (-not [IO.Path]::IsPathRooted($outFile)) { $outFile = Join-Path $dst $outFile }
$baseSize = (Get-Item $outFile).Length; "BASELINE_OK out=$outFile size=$baseSize"
function MapObjs([string]$m) { $h = @{}; foreach ($l in [IO.File]::ReadLines($m)) { if ($l -match '\s(\S+\.obj)\s*$') { $h[$Matches[1]] = 1 } }; return $h }
if ($Map) { Copy-Item (Join-Path $dst '_cur.map') (Join-Path $dst '_base.map') -Force; $baseObjs = MapObjs (Join-Path $dst '_base.map'); "BASEMAP objs=$($baseObjs.Count)" }
$sizediff = 0; $ok = 0; $undef = 0; $other = 0; $rwfail = 0; $sw = [Diagnostics.Stopwatch]::StartNew(); $done = 0
# Sum ZERO_SEEN / RESCUED across every lld-link invocation (one ATTACH/DETACH pair each).
function AffStats { if ($LinkAffinity -gt 0 -and (Test-Path $env:AFFINITY_LOG)) { $a = Get-Content $env:AFFINITY_LOG; "aff_launches=$(@($a).Count) aff_ok=$(@($a | Select-String 'ok=1').Count)" } else { "aff_launches=0" } }
function ShimStats {
    if (-not $Shim -or -not (Test-Path $env:LLDZERO_LOG)) { return @(0, 0, 0, 0, 0, 0, 0) }
    $t = Get-Content $env:LLDZERO_LOG
    $cp = ($t | Select-String 'COPY_ACTIVE').Count # link processes that used ReadFile copies
    @(($t | Select-String 'ZERO_SEEN').Count, ($t | Select-String '^RESCUED_').Count, ($t | Select-String 'ATTACH ').Count, $cp,
      ($t | Select-String '^(MAPZERO_BOTH|COPY_ZERO)').Count, ($t | Select-String '^LATE_CHANGE').Count, ($t | Select-String '^UNRESCUED_').Count)
}
for ($k = 1; $k -le $Iters; $k++) {
    if ($Mode -ne 'none') { & $Probe rw $list $Par $Mode | Out-Null; if ($LASTEXITCODE -ne 0) { $rwfail++ } }
    if ($PreRead) { & $Probe pr $list | Out-Null }
    if ($DelayMs) { Start-Sleep -Milliseconds $DelayMs }
    $out = Run-Link; $rc = $LASTEXITCODE; $done = $k
    $ev = $false
    if ($rc -eq 0) { $ok++; $sz = (Get-Item $outFile).Length; if ($sz -ne $baseSize) { $sizediff++; $ev = $true; "SIZEDIFF iter=$k size=$sz base=$baseSize" } }
    elseif ($out -match 'undefined symbol') { $undef++; $ev = $true; "UNDEF iter=$k"; $out | Select-String 'undefined symbol|referenced by' | Select-Object -First 4 }
    else { $other++; "OTHER iter=$k rc=$rc"; $out | Select-Object -Last 5 }
    if ($ev) {
        & $Probe zs $list
        if ($Map -and (Test-Path (Join-Path $dst '_cur.map'))) { $c = MapObjs (Join-Path $dst '_cur.map'); $miss = $baseObjs.Keys | Where-Object { -not $c.ContainsKey($_) }; "  MISSING_FROM_MAP: $($miss -join ', ')" }
        if ($MaxEvents -and ($sizediff + $undef) -ge $MaxEvents) { break }
    }
    if ($k % 100 -eq 0) { $s = ShimStats; "PROGRESS links=$k ok=$ok sizediff=$sizediff undef=$undef other=$other rwfail=$rwfail shim_zero=$($s[0]) shim_rescued=$($s[1]) shim_copied=$($s[3]) shim_zboth=$($s[4]) shim_late=$($s[5]) shim_unresc=$($s[6]) $(AffStats) objreads=$($k*$rel.Count) secs=$([int]$sw.Elapsed.TotalSeconds)" }
    if ($MaxMinutes -and $sw.Elapsed.TotalMinutes -ge $MaxMinutes) { break }
    if ($StopFile -and (Test-Path $StopFile)) { "STOPFILE iter=$k"; break }
}
Pop-Location
$s = ShimStats
"RESULT mode=$Mode shim=$Shim preread=$PreRead delayms=$DelayMs link=[$env:_LINK_] links=$done ok=$ok sizediff=$sizediff undef=$undef other=$other rwfail=$rwfail shim_zero=$($s[0]) shim_rescued=$($s[1]) shim_links=$($s[2]) shim_copied=$($s[3]) shim_zboth=$($s[4]) shim_late=$($s[5]) shim_unresc=$($s[6]) linkaff=$LinkAffinity $(AffStats) objreads=$($done*$rel.Count) secs=$([int]$sw.Elapsed.TotalSeconds)"
