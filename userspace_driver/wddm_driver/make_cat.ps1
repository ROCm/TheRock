$PKG = "D:\R\userspace_driver\wddm_driver\package"
$WDK = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"

$lines = @(
    "[CatalogHeader]",
    "Name=amdgpu_wddm.cat",
    "ResultDir=$PKG",
    "PublicVersion=0x00000001",
    "CatalogVersion=2",
    "HashAlgorithms=SHA256",
    "PageHashes=false",
    "",
    "[CatalogFiles]",
    "amdgpu_wddm.inf=$PKG\amdgpu_wddm.inf",
    "amdgpu_wddm.sys=$PKG\amdgpu_wddm.sys"
)

$lines -join "`r`n" | Set-Content "$PKG\amdgpu_wddm.cdf" -NoNewline -Encoding ASCII

& "$WDK\makecat.exe" "$PKG\amdgpu_wddm.cdf"
Write-Host "makecat exit: $LASTEXITCODE"

if (Test-Path "$PKG\amdgpu_wddm.cat") {
    & "$WDK\signtool.exe" sign /s PrivateCertStore /n "AMDGPU Test" /fd sha256 "$PKG\amdgpu_wddm.cat"
    & "$WDK\signtool.exe" sign /s PrivateCertStore /n "AMDGPU Test" /fd sha256 "$PKG\amdgpu_wddm.sys"
    Write-Host "All signed successfully"
} else {
    Write-Host "ERROR: catalog not created"
}
