$PKG = "D:\R\userspace_driver\wddm_driver\package"
$SYS = "D:\R\userspace_driver\wddm_driver\x64\Release\amdgpu_wddm.sys"
$INF = "D:\R\userspace_driver\wddm_driver\amdgpu_wddm.inf"
$WDK = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"

# Copy files to package
Copy-Item $SYS "$PKG\amdgpu_wddm.sys" -Force
Copy-Item $INF "$PKG\amdgpu_wddm.inf" -Force

# Create catalog CDF
@"
[CatalogHeader]
Name=amdgpu_wddm.cat
ResultDir=$PKG
PublicVersion=0x00000001
CatalogVersion=2
HashAlgorithms=SHA256
PageHashes=false

[CatalogFiles]
amdgpu_wddm.inf=$PKG\amdgpu_wddm.inf
amdgpu_wddm.sys=$PKG\amdgpu_wddm.sys
"@ | Out-File "$PKG\amdgpu_wddm.cdf" -Encoding ASCII

# Make catalog
& "$WDK\makecat.exe" "$PKG\amdgpu_wddm.cdf"
Write-Host "makecat exit: $LASTEXITCODE"

# Sign .sys and .cat
& "$WDK\signtool.exe" sign /s PrivateCertStore /n "AMDGPU Test" /fd sha256 "$PKG\amdgpu_wddm.sys"
& "$WDK\signtool.exe" sign /s PrivateCertStore /n "AMDGPU Test" /fd sha256 "$PKG\amdgpu_wddm.cat"
Write-Host "Signing complete"
