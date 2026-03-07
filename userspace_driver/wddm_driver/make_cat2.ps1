$PKG = "D:\R\userspace_driver\wddm_driver\package"
$WDK = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"

$cdf = @"
[CatalogHeader]
Name=amdgpu_wddm.cat
ResultDir=$PKG
PublicVersion=0x00000001
CatalogVersion=2
HashAlgorithms=SHA256
PageHashes=false
EncodingType=0x00010001
CATATTR1=0x10010001:OSAttr:2:10.0

[CatalogFiles]
<HASH>amdgpu_wddm.inf=$PKG\amdgpu_wddm.inf
<HASH>amdgpu_wddm.sys=$PKG\amdgpu_wddm.sys
"@

[System.IO.File]::WriteAllText("$PKG\amdgpu_wddm.cdf", $cdf.Replace("`n", "`r`n"))

& "$WDK\makecat.exe" -v "$PKG\amdgpu_wddm.cdf"
Write-Host "makecat exit: $LASTEXITCODE"

if ($LASTEXITCODE -eq 0) {
    & "$WDK\signtool.exe" sign /s PrivateCertStore /n "AMDGPU Test" /fd sha256 "$PKG\amdgpu_wddm.cat"
    & "$WDK\signtool.exe" sign /s PrivateCertStore /n "AMDGPU Test" /fd sha256 "$PKG\amdgpu_wddm.sys"
    Write-Host "All signed successfully"
} else {
    Write-Host "makecat failed - trying inf2cat instead"
    & "$WDK\inf2cat.exe" /driver:"$PKG" /os:10_X64 /verbose 2>&1
    if ($LASTEXITCODE -eq 0) {
        & "$WDK\signtool.exe" sign /s PrivateCertStore /n "AMDGPU Test" /fd sha256 "$PKG\amdgpu_wddm.cat"
        & "$WDK\signtool.exe" sign /s PrivateCertStore /n "AMDGPU Test" /fd sha256 "$PKG\amdgpu_wddm.sys"
        Write-Host "All signed successfully (via inf2cat)"
    } else {
        Write-Host "inf2cat also failed"
    }
}
