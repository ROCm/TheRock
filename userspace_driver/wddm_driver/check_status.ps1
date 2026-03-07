$dev = Get-CimInstance Win32_PnPEntity | Where-Object { $_.DeviceID -like '*7551*' }
if ($dev) {
    Write-Host "Device: $($dev.Name)"
    Write-Host "Status: $($dev.Status)"
    Write-Host "ConfigManagerErrorCode: $($dev.ConfigManagerErrorCode)"

    # Check problem status in registry
    $hwKey = "HKLM:\SYSTEM\CurrentControlSet\Enum\$($dev.DeviceID)"
    if (Test-Path $hwKey) {
        $props = Get-ItemProperty $hwKey -ErrorAction SilentlyContinue
        Write-Host "Problem Status: $($props.ProblemStatus)"
    }
} else {
    Write-Host "Device not found"
}

# Also check system event log for recent display errors
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='*dxgkrnl*','*display*'; StartTime=(Get-Date).AddMinutes(-5)} -MaxEvents 10 -ErrorAction SilentlyContinue | Format-List TimeCreated, Id, Message
