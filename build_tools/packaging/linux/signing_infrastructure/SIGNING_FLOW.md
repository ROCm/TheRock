# Complete Package Signing Flow

## Overview

This document explains the complete end-to-end signing flow for both RPM packages and repository metadata (DEB and RPM).

---

## Workflow Steps

```
GitHub Actions Workflow
├─ 1. Get OIDC Token (if release_type set)
├─ 2. Build Packages (build_package.py)
├─ 3. Install gpgshim (RPM only)
├─ 4. Sign RPM Packages (rpmsign + gpgshim)
├─ 5. Upload Package repo to S3 (upload_package_repo.py)
│    ├─ Upload packages to S3
│    ├─ Generate repository metadata
│    ├─ **Sign metadata** (integrated)
│    └─ Upload signed metadata to S3
└─ 6. Generate index.html files
```

---

## Detailed Flow

### Step 1: Get OIDC Token

**When:** Only if `release_type` is set (dev, nightly, prerelease, release)

**What happens:**
```bash
OIDC_TOKEN=$(curl -sSL \
  -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
  "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" \
  | jq -r '.value')

echo "token=$OIDC_TOKEN" >> $GITHUB_OUTPUT
```

**Result:** OIDC token stored in `steps.oidc.outputs.token`

---

### Step 2: Build Packages

**Script:** `build_package.py`

**What happens:**
- Builds unsigned DEB or RPM packages
- Packages placed in `$PACKAGE_DIST_DIR`

**Result:** Unsigned packages ready for signing

---

### Step 3: Install gpgshim (RPM only)

**When:** Only for RPM builds with release_type set

**What happens:**
```bash
cp build_tools/packaging/linux/gpgshim ~/.local/bin/gpgshim
chmod +x ~/.local/bin/gpgshim
```

**Result:** gpgshim available in PATH

---

### Step 4: Sign RPM Packages

**When:** Only for RPM builds with release_type set

**Environment Variables:**
- `GPG_SIGNING_SERVER` - From GitHub Secrets
- `GPG_SERVER_TOKEN` - OIDC token from Step 1

**What happens:**
```bash
find . -name "*.rpm" -type f | while read rpm_file; do
  rpmsign --addsign \
    --define "_gpg_path $HOME/.local/bin/gpgshim" \
    "$rpm_file"

  rpm --checksig "$rpm_file"
done
```

**gpgshim flow:**
1. rpmsign calls gpgshim instead of system gpg
2. gpgshim reads `GPG_SERVER_TOKEN` environment variable
3. gpgshim sends RPM header (~4KB) + token to signing server
4. Server validates OIDC token and signs
5. gpgshim caches signature
6. rpmsign calls gpgshim second time for full package
7. gpgshim returns cached signature (no server call)
8. rpmsign embeds signature into RPM

**Result:** All RPM packages have embedded signatures

---

### Step 5: Upload Package Repo to S3

**Script:** `upload_package_repo.py`

**Environment Variables:**
- `GPG_SIGNING_SERVER` - From GitHub Secrets
- `GPG_SERVER_TOKEN` - OIDC token from Step 1

**Command:**
```bash
python upload_package_repo.py \
  --pkg-type deb \
  --s3-bucket therock-dev-packages \
  --amdgpu-family gfx94X-dcgpu \
  --artifact-id 123456 \
  --job dev \
  --gpg-signing-server "$GPG_SIGNING_SERVER" \
  --gpg-server-token "$GPG_SERVER_TOKEN"
```

**What happens (DEB):**
```python
# 1. Upload packages to S3
upload_to_s3(package_dir, bucket, prefix)

# 2. Regenerate metadata (merge with existing S3 metadata)
regenerate_deb_metadata_from_s3(s3, bucket, prefix, uploaded_packages)
  # 2a. Download existing metadata from S3
  # 2b. Merge Packages files
  # 2c. Generate Release file with checksums
  generate_release_file_with_checksums(release_file, job_type, dists_dir)

  # 2d. **SIGN RELEASE FILE** (if credentials provided)
  if gpg_signing_server and gpg_server_token:
    sign_deb_release_file(release_file, gpg_signing_server, gpg_server_token)
      # Creates InRelease (clearsigned)
      sign_metadata_file(release_file, "InRelease", server, token, clearsign=True)

      # Creates Release.gpg (detached)
      sign_metadata_file(release_file, "Release.gpg", server, token, clearsign=False)

  # 2e. Upload signed metadata to S3
  upload_deb_metadata_to_s3(s3, bucket, prefix, dists_dir, release_file)
```

**What happens (RPM):**
```python
# 1. Upload packages to S3
upload_to_s3(package_dir, bucket, prefix)

# 2. Regenerate metadata (merge with existing S3 metadata)
regenerate_rpm_metadata_from_s3(s3, bucket, prefix, uploaded_packages)
  # 2a. Download existing repodata from S3
  # 2b. Generate metadata for new packages
  createrepo_c --no-database --simple-md-filenames .

  # 2c. Merge old and new metadata
  mergerepo_c --repo old_repo --repo new_repo --outputdir merged_repo

  # 2d. **SIGN REPOMD.XML** (if credentials provided)
  if gpg_signing_server and gpg_server_token:
    sign_rpm_repomd_files(merged_arch_dir, gpg_signing_server, gpg_server_token)
      # Creates repomd.xml.asc (detached signature)
      sign_metadata_file(repomd_file, "repomd.xml.asc", server, token)

  # 2e. Upload signed metadata to S3
  for metadata_file in merged_repodata.iterdir():
    s3.upload_file(metadata_file, bucket, s3_key)
```

**Result:**
- Packages uploaded to S3
- Repository metadata generated, signed, and uploaded
- **DEB**: `Release`, `InRelease`, `Release.gpg` in S3
- **RPM**: `repomd.xml`, `repomd.xml.asc` in S3

---

## Signing Functions

### `sign_metadata_file(metadata_file, output_file, server_url, token, clearsign=False)`

**Purpose:** Sign any metadata file using remote signing server

**Flow:**
```python
# 1. Read metadata file
with open(metadata_file, 'rb') as f:
    metadata_data = f.read()

# 2. Prepare signing request
payload = {
    'data': base64.b64encode(metadata_data).decode('ascii'),
    'digest_algo': 'SHA256',
    'armor': True,
    'clearsign': clearsign  # For InRelease
}

headers = {
    'Authorization': f'Bearer {token}',  # ← OIDC token here
    'Content-Type': 'application/json'
}

# 3. Send to signing server
request = Request(server_url, data=json_data, headers=headers)
response = urlopen(request, timeout=60)

# 4. Extract signature
result = json.loads(response.read())
signature = base64.b64decode(result['signature'])

# 5. Write signature to output file
with open(output_file, 'wb') as f:
    f.write(signature)
```

---

### `sign_deb_release_file(release_file, server_url, token)`

**Purpose:** Sign DEB Release file (creates both InRelease and Release.gpg)

**Flow:**
```python
# Create InRelease (clearsigned)
sign_metadata_file(
    release_file,
    release_file.parent / "InRelease",
    server_url,
    token,
    clearsign=True
)

# Create Release.gpg (detached signature)
sign_metadata_file(
    release_file,
    release_file.parent / "Release.gpg",
    server_url,
    token,
    clearsign=False
)
```

---

### `sign_rpm_repomd_files(rpm_dir, server_url, token)`

**Purpose:** Sign all RPM repomd.xml files (one per architecture)

**Flow:**
```python
# Find all repomd.xml files
repomd_files = list(rpm_dir.rglob("repodata/repomd.xml"))

for repomd_file in repomd_files:
    # Create detached signature (repomd.xml.asc)
    sign_metadata_file(
        repomd_file,
        repomd_file.parent / "repomd.xml.asc",
        server_url,
        token,
        clearsign=False
    )
```

---

## Signing Server Flow

When signing server receives request:

```
1. Extract OIDC token from Authorization header
   ↓
2. Validate OIDC token (verify signature, check expiration)
   ↓
3. Authorize request based on OIDC claims:
   - Check repository: ROCm/TheRock or ROCm/rockrel
   - Check branch: refs/heads/main, etc.
   - Check workflow: build_native_linux_packages.yml
   ↓
4. Sign data with GPG
   ↓
5. Return signature to client
```

---

## What Gets Signed

| Item | Signed? | Method | Signature Type |
|------|---------|--------|----------------|
| **RPM packages** | ✅ Yes | gpgshim + rpmsign | Embedded in RPM |
| **DEB packages** | ❌ No | N/A | N/A |
| **DEB Release file** | ✅ Yes | upload_package_repo.py | InRelease (clearsigned) + Release.gpg (detached) |
| **RPM repomd.xml** | ✅ Yes | upload_package_repo.py | repomd.xml.asc (detached) |

---

## Files Created

### DEB Repository Structure (on S3)

```
s3://therock-dev-packages/v3/packages/deb/
├── pool/
│   └── main/
│       ├── rocm-hip-runtime_8.0.0_amd64.deb
│       ├── rocm-hip-sdk_8.0.0_amd64.deb
│       └── ...
└── dists/
    └── stable/
        ├── main/
        │   ├── binary-amd64/
        │   │   ├── Packages
        │   │   └── Packages.gz
        ├── Release              ← Metadata with checksums
        ├── InRelease            ← Clearsigned Release ✅ SIGNED
        └── Release.gpg          ← Detached signature ✅ SIGNED
```

### RPM Repository Structure (on S3)

```
s3://therock-dev-packages/v3/packages/rpm/
└── x86_64/
    ├── rocm-hip-runtime-8.0.0.el8.x86_64.rpm  ✅ SIGNED (embedded)
    ├── rocm-hip-sdk-8.0.0.el8.x86_64.rpm      ✅ SIGNED (embedded)
    ├── ...
    └── repodata/
        ├── repomd.xml           ← Repository metadata
        ├── repomd.xml.asc       ← Detached signature ✅ SIGNED
        ├── primary.xml.gz
        ├── filelists.xml.gz
        └── other.xml.gz
```

---

## User Experience

### Installing DEB Packages

```bash
# Import GPG public key
wget -O - https://therock-dev-packages.s3.amazonaws.com/keys/therock-dev-public.gpg | sudo apt-key add -

# Add repository
echo "deb [arch=amd64] https://therock-dev-packages.s3.amazonaws.com/v3/packages/deb stable main" | \
  sudo tee /etc/apt/sources.list.d/therock.list

# Update (verifies InRelease signature automatically)
sudo apt update

# Install (metadata signature already verified by apt)
sudo apt install rocm-hip-runtime
```

### Installing RPM Packages

```bash
# Import GPG public key
sudo rpm --import https://therock-dev-packages.s3.amazonaws.com/keys/therock-dev-public.gpg

# Add repository
sudo cat > /etc/yum.repos.d/therock.repo <<EOF
[therock-dev]
name=AMD ROCm TheRock Development
baseurl=https://therock-dev-packages.s3.amazonaws.com/v3/packages/rpm/x86_64
enabled=1
gpgcheck=1
gpgkey=https://therock-dev-packages.s3.amazonaws.com/keys/therock-dev-public.gpg
repo_gpgcheck=1
EOF

# Install (verifies both package signature and repomd.xml.asc)
sudo yum install rocm-hip-runtime
```

---

## Key Advantages

1. **RPM Packages:** Individual packages signed → yum verifies on install
2. **DEB Metadata:** Release file signed → apt verifies repository authenticity
3. **RPM Metadata:** repomd.xml signed → yum verifies repository integrity
4. **Automatic:** Signing happens automatically when `release_type` is set
5. **Secure:** OIDC tokens instead of stored secrets
6. **Efficient:** gpgshim optimization (4KB transfer instead of 1GB+)
7. **Integrated:** Metadata signing built into upload_package_repo.py

---

## Troubleshooting

### Signing not happening?

Check:
1. Is `release_type` set in workflow inputs?
2. Is `GPG_SIGNING_SERVER` secret configured?
3. Is OIDC token generation step succeeding?
4. Are signing parameters being passed to upload_package_repo.py?

### Signature verification failing?

Check:
1. Is the correct public key imported?
2. Is the signing server using the correct GPG key?
3. Is the OIDC token valid (not expired)?
4. Are the signed files actually uploaded to S3?

### Metadata signing skipped?

Check upload_package_repo.py logs for:
- "Repository metadata signing: ENABLED" (should appear)
- "Signing DEB Release file" or "Signing RPM repomd.xml files"
- Any error messages from signing functions
