# Workflow Integration Summary

## Changes Made to `build_native_linux_packages.yml`

### Overview
The workflow has been updated to include OIDC-based package signing for both RPM and DEB packages. Signing occurs automatically when `release_type` is set (dev, nightly, prerelease, or release).

---

## New Workflow Steps

### 1. Get OIDC Token (Step added after "Build Packages")

```yaml
- name: Get OIDC token for package signing
  id: oidc
  if: env.RELEASE_TYPE != ''
  run: |
    OIDC_TOKEN=$(curl -sSL \
      -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
      "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" \
      | jq -r '.value')

    echo "::add-mask::$OIDC_TOKEN"
    echo "token=$OIDC_TOKEN" >> $GITHUB_OUTPUT
```

**What it does:**
- Requests OIDC token from GitHub Actions OIDC provider
- Token valid for ~10 minutes
- Token contains: repository, branch, workflow, actor, run_id
- Masked in logs for security
- Stored in step output for use in signing steps

**When it runs:** Only when `release_type` is set (not for CI builds)

---

### 2. Install gpgshim (RPM only)

```yaml
- name: Install gpgshim for RPM signing
  if: inputs.native_package_type == 'rpm' && env.RELEASE_TYPE != ''
  run: |
    mkdir -p ~/.local/bin
    cp build_tools/packaging/linux/gpgshim ~/.local/bin/gpgshim
    chmod +x ~/.local/bin/gpgshim
```

**What it does:**
- Copies gpgshim from repository to runner's local bin
- Makes it executable
- gpgshim acts as drop-in GPG replacement for rpmsign

**When it runs:** Only for RPM builds with release_type set

---

### 3. Sign RPM Packages

```yaml
- name: Sign RPM packages
  if: inputs.native_package_type == 'rpm' && env.RELEASE_TYPE != ''
  env:
    GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
    GPG_SERVER_TOKEN: ${{ steps.oidc.outputs.token }}  # ← OIDC token here
    PATH: /home/runner/.local/bin:${{ env.PATH }}
  run: |
    cd ${{ env.PACKAGE_DIST_DIR }}

    find . -name "*.rpm" -type f | while read rpm_file; do
      rpmsign --addsign \
        --define "_gpg_path $HOME/.local/bin/gpgshim" \
        "$rpm_file"

      rpm --checksig "$rpm_file"
    done
```

**What it does:**
- Signs all RPM packages using gpgshim
- gpgshim reads `GPG_SERVER_TOKEN` env var (contains OIDC token)
- gpgshim sends token to signing server
- Server validates OIDC token and signs packages
- Verifies signatures after signing

**When it runs:** Only for RPM builds with release_type set

**Key environment variables:**
- `GPG_SIGNING_SERVER` - Signing server URL (from GitHub Secrets)
- `GPG_SERVER_TOKEN` - OIDC token (from step above)

---

### 4. Upload Package repo to S3 (with integrated metadata signing)

```yaml
- name: Upload Package repo to S3
  id: upload-packages
  env:
    GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
    GPG_SERVER_TOKEN: ${{ steps.oidc.outputs.token }}  # ← OIDC token here
  run: |
    # Build command with optional signing parameters
    CMD="python ./build_tools/packaging/linux/upload_package_repo.py \
      --pkg-type ${{ inputs.native_package_type }} \
      --s3-bucket ${{ env.S3_BUCKET_NATIVE }} \
      --amdgpu-family ${{ inputs.artifact_group }} \
      --artifact-id ${{ env.ARTIFACT_RUN_ID }} \
      --job ${{ inputs.release_type }}"

    # Add signing parameters if release_type is set (enables signing)
    if [ -n "${{ env.RELEASE_TYPE }}" ]; then
      CMD="$CMD --gpg-signing-server \"$GPG_SIGNING_SERVER\" --gpg-server-token \"$GPG_SERVER_TOKEN\""
      echo "Repository metadata signing: ENABLED"
    fi

    # Execute command
    eval $CMD
```

**What it does:**
- Uploads packages to S3
- Generates repository metadata inside `upload_package_repo.py`:
  - **DEB**: Creates `Release` file with checksums
  - **RPM**: Creates `repomd.xml` using createrepo_c/mergerepo_c
- **Signs metadata automatically** if release_type is set:
  - **DEB**: Signs Release → creates `InRelease` (clearsigned) and `Release.gpg` (detached)
  - **RPM**: Signs repomd.xml → creates `repomd.xml.asc` (detached)
- Uploads signed metadata files to S3

**When signing happens:**
- Only when `release_type` is set (not for CI builds)
- Signing integrated into `upload_package_repo.py` script
- Happens **after** metadata generation, **before** upload to S3

**Key Flow:**
1. upload_package_repo.py uploads packages
2. Regenerates metadata (merges with existing S3 metadata)
3. **Calls signing functions** if credentials provided
4. Uploads signed metadata to S3

**Note:** Individual DEB packages are NOT signed - only the repository metadata (Release file)

---


## Required GitHub Secrets

Add these secrets to your repository (Settings → Secrets and variables → Actions):

### 1. GPG_SIGNING_SERVER
**Value:** `https://signing.yourdomain.com/sign`

**Description:** URL of the signing server endpoint

**Example:** `https://signing.internal.amd.com/api/v1/sign`

**Note:** This is the only required secret - OIDC tokens are generated automatically by GitHub Actions

---

## Workflow Behavior by Release Type

| Release Type | Signing Enabled | Key Used | Branch Allowed |
|--------------|-----------------|----------|----------------|
| *(not set)* | ❌ No | N/A | Any |
| `dev` | ✅ Yes | therock-dev | Any |
| `nightly` | ✅ Yes | therock-nightly | main, develop |
| `prerelease` | ✅ Yes | therock-release | main, release/* |
| `release` | ✅ Yes | therock-release | main, release/* |

**Note:** Signing server enforces branch restrictions based on OIDC token's `ref` claim.

---

## Files Created

### 1. `build_tools/packaging/linux/sign_repo_metadata.py`
**Purpose:** Sign DEB Release files and RPM repomd.xml using remote signing server

**Usage:**
```bash
python3 sign_repo_metadata.py \
  --metadata-file Release \
  --output InRelease \
  --server https://signing.example.com/sign \
  --token $OIDC_TOKEN \
  --clearsign
```

**Features:**
- Supports clearsigned (InRelease) and detached (Release.gpg) signatures
- Uses OIDC or JWT tokens for authentication
- Error handling and detailed logging
- Timeout configuration

---

## How Signing Works

### OIDC Token Flow

```
1. GitHub Actions generates OIDC token
   ↓
2. Workflow sets GPG_SERVER_TOKEN env var
   ↓
3. rpmsign calls gpgshim
   ↓
4. gpgshim reads GPG_SERVER_TOKEN env var
   ↓
5. gpgshim sends token to signing server
   ↓
6. Server validates OIDC token:
   - Verifies GitHub's RS256 signature
   - Checks repository: ROCm/TheRock or ROCm/rockrel
   - Checks branch: refs/heads/main, etc.
   - Checks workflow: build_native_linux_packages.yml
   ↓
7. Server signs package with GPG
   ↓
8. Server returns signature
   ↓
9. gpgshim writes signature to output
   ↓
10. rpmsign embeds signature in RPM
```

### Server-Side Authorization

Based on OIDC token claims, server enforces:

**Development role (`dev`):**
- Repositories: ROCm/TheRock, ROCm/rockrel
- Branches: Any (`refs/heads/*`, `refs/pull/*`)
- Workflows: build_native_linux_packages.yml, multi_arch_build_native_linux_packages.yml
- Key: therock-dev@amd.com

**Release role (`release`, `prerelease`):**
- Repositories: ROCm/TheRock, ROCm/rockrel
- Branches: main, release/* only
- Workflows: build_native_linux_packages.yml only
- Key: therock-release@amd.com

---

## Testing the Integration

### 1. Trigger Manual Workflow

```bash
# Trigger with release_type=dev (signs with dev key)
gh workflow run build_native_linux_packages.yml \
  --ref main \
  -f native_package_type=rpm \
  -f rocm_version=8.0.0 \
  -f artifact_run_id=123456789 \
  -f release_type=dev
```

### 2. Monitor Workflow Logs

Look for these log entries:

```
✅ OIDC token obtained (valid for ~10 minutes)
✅ gpgshim installed at ~/.local/bin/gpgshim
Found 42 RPM packages to sign
  Signing: rocm-hip-runtime-8.0.0.rpm
  ✅ Signed: rocm-hip-runtime-8.0.0.rpm
✅ All 42 RPM packages signed successfully
```

### 3. Verify Signatures

Download signed package from S3:
```bash
aws s3 cp s3://therock-dev-packages/v3/packages/rpm/rocm-hip-runtime.rpm ./

# Verify signature
rpm --checksig rocm-hip-runtime.rpm
# Expected: rocm-hip-runtime.rpm: digests signatures OK
```

### 4. Check Server Audit Logs

On signing server:
```bash
tail -f /var/log/gpg-signing/audit.log
```

Look for:
```json
{
  "timestamp": "2024-03-09T10:30:00Z",
  "auth_type": "oidc",
  "repository": "ROCm/TheRock",
  "ref": "refs/heads/main",
  "workflow": ".github/workflows/build_native_linux_packages.yml",
  "actor": "username",
  "run_id": "123456789",
  "role": "development",
  "key_id": "therock-dev@amd.com",
  "status": "SUCCESS"
}
```

---

## Troubleshooting

### Issue: "OIDC token obtained" but signing fails

**Possible causes:**
1. `GPG_SIGNING_SERVER` secret not configured
2. Signing server not reachable from GitHub Actions
3. Signing server not configured for OIDC (PyJWT not installed)

**Solution:**
```bash
# Test connectivity from workflow
- name: Test signing server
  run: curl -v ${{ secrets.GPG_SIGNING_SERVER }}/health
```

---

### Issue: "Repository not authorized"

**Cause:** Repository not in `allowed_repositories` list in server's `authorization.json`

**Solution:** Update server's `authorization.json`:
```json
{
  "roles": {
    "development": {
      "allowed_repositories": [
        "ROCm/TheRock",
        "ROCm/rockrel"  // ← Add your repository
      ]
    }
  }
}
```

---

### Issue: "Workflow not authorized"

**Cause:** Workflow file path not in `allowed_workflows` list

**Solution:** Update server's `authorization.json`:
```json
{
  "roles": {
    "development": {
      "allowed_workflows": [
        ".github/workflows/build_native_linux_packages.yml",  // ← Add workflow
        ".github/workflows/multi_arch_build_native_linux_packages.yml"
      ]
    }
  }
}
```

---

### Issue: Signing step skipped

**Cause:** `release_type` not set or is empty

**Check workflow run:**
- Look for: `if: env.RELEASE_TYPE != ''` condition
- Verify `release_type` input parameter is provided
- Check: `RELEASE_TYPE: ${{ inputs.release_type || '' }}` in env

**Solution:**
```bash
# Always provide release_type when triggering workflow
gh workflow run build_native_linux_packages.yml \
  -f release_type=dev  # ← Must be set
```

---

## End User Package Installation

After packages are signed and uploaded to S3, users can install with signature verification:

### RPM (RHEL/Rocky/AlmaLinux)

```bash
# Import GPG public key
sudo rpm --import https://therock-dev-packages.s3.us-east-2.amazonaws.com/keys/therock-dev-public.gpg

# Install package (signature verified automatically)
sudo yum install rocm-hip-runtime
```

### DEB (Ubuntu/Debian)

```bash
# Import GPG public key
wget -O - https://therock-dev-packages.s3.us-east-2.amazonaws.com/keys/therock-dev-public.gpg | sudo apt-key add -

# Add repository with signature verification enabled
echo "deb [arch=amd64] https://therock-dev-packages.s3.us-east-2.amazonaws.com/v3/packages/deb stable main" | \
  sudo tee /etc/apt/sources.list.d/therock.list

# Update and install (signature verified automatically)
sudo apt update
sudo apt install rocm-hip-runtime
```

---

## References

- **OIDC Flow Explanation:** `docs/oidc-flow-explained.md`
- **Deployment Guide:** `DEPLOYMENT_GUIDE.md`
- **Token Reference:** `docs/oidc-token-reference.md`
- **Security Analysis:** `GITHUB_SECRETS_SECURITY_ANALYSIS.md`
- **Implementation Summary:** `OIDC_IMPLEMENTATION_SUMMARY.md`
