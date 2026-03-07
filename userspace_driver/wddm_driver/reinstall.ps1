# Clean reinstall of WDDM driver
$devcon = "C:\Program Files (x86)\Windows Kits\10\Tools\10.0.26100.0\x64\devcon.exe"
$hwid = "PCI\VEN_1002&DEV_7551"
$inf = "D:\R\userspace_driver\wddm_driver\package\amdgpu_wddm.inf"

# Clear diagnostic markers
reg delete HKLM\SOFTWARE\AmdGpuWddm /f 2>$null

# Remove device
& $devcon remove $hwid 2>$null

# Find and delete old driver from store
$drivers = pnputil /enum-drivers 2>&1
$oem = $null
$lines = $drivers -split "`n"
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match 'amdgpu_wddm') {
        for ($j = $i - 1; $j -ge 0; $j--) {
            if ($lines[$j] -match '(oem\d+\.inf)') {
                $oem = $Matches[1]
                break
            }
        }
    }
}
if ($oem) {
    Write-Host "Deleting old driver: $oem"
    pnputil /delete-driver $oem /force
}

# Rescan
& $devcon rescan
Start-Sleep 2

# Install
& $devcon updateni $inf $hwid
Start-Sleep 3

# Show results
Write-Host "`n=== Registry Markers ==="
if (Test-Path 'HKLM:\SOFTWARE\AmdGpuWddm') {
    Get-ItemProperty 'HKLM:\SOFTWARE\AmdGpuWddm' | Format-List -Property * -Exclude PS*
} else {
    Write-Host "No markers found"
}

Write-Host "`n=== Device Status ==="
pnputil /enum-devices /class Display | Select-String '7551' -Context 0,8
