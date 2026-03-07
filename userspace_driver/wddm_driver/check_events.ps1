$events = Get-WinEvent -LogName System -MaxEvents 200 | Where-Object {
    $_.Message -match 'display|dxgkrnl|7551|amdgpu|video|gpu|graphics' -or
    $_.Source -match 'display|dxgkrnl|video|Kernel-PnP'
}
foreach ($e in $events) {
    Write-Host "--- $($e.TimeCreated) [$($e.LevelDisplayName)] Source=$($e.Source) ID=$($e.Id)"
    Write-Host $e.Message.Substring(0, [Math]::Min(500, $e.Message.Length))
    Write-Host ""
}
