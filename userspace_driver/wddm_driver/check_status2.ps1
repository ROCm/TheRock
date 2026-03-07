# Check problem status from device properties
$devId = 'PCI\VEN_1002&DEV_7551&SUBSYS_242F1458&REV_C0\6&2a7c2239&0&00000019'
$regPath = "HKLM:\SYSTEM\CurrentControlSet\Enum\$devId"
if (Test-Path $regPath) {
    Get-ItemProperty $regPath | Format-List Problem*, ConfigFlags, Driver
}

# Check driver store for our inf
pnputil /enum-drivers | Select-String 'amdgpu_wddm' -Context 2,5

# Check dxgkrnl and display events in last 10 minutes
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddMinutes(-10); Level=1,2,3} -MaxEvents 20 -ErrorAction SilentlyContinue | Where-Object { $_.Message -like '*display*' -or $_.Message -like '*dxgk*' -or $_.Message -like '*AMD*' -or $_.Message -like '*amdgpu*' -or $_.ProviderName -like '*dxgkrnl*' } | Format-List TimeCreated, Id, ProviderName, Message
