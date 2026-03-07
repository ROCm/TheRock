# Check multiple event logs for driver-related events
Write-Host "=== System Log (last 30) ==="
Get-WinEvent -LogName System -MaxEvents 30 | ForEach-Object {
    Write-Host "$($_.TimeCreated) [$($_.LevelDisplayName)] $($_.Source) ID=$($_.Id)"
    $msg = $_.Message
    if ($msg.Length -gt 300) { $msg = $msg.Substring(0,300) }
    Write-Host "  $msg"
}

Write-Host "`n=== Microsoft-Windows-Kernel-PnP/Configuration ==="
try {
    Get-WinEvent -LogName 'Microsoft-Windows-Kernel-PnP/Configuration' -MaxEvents 20 | ForEach-Object {
        Write-Host "$($_.TimeCreated) [$($_.LevelDisplayName)] ID=$($_.Id)"
        $msg = $_.Message
        if ($msg.Length -gt 300) { $msg = $msg.Substring(0,300) }
        Write-Host "  $msg"
    }
} catch { Write-Host "  (not available)" }

Write-Host "`n=== Device Install Log ==="
$setupLog = "$env:WINDIR\INF\setupapi.dev.log"
if (Test-Path $setupLog) {
    Get-Content $setupLog -Tail 80
}
