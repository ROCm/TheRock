# Package Signing POC Analysis & Implementation Plan

**Branch:** `users/isparry/remote_signing`
**Commit:** `83ed5f3616c8f76a9b00b4571be63fa8be3ea73d`
**Date:** March 2026

## Executive Summary

The POC implements an **innovative GPG shim architecture** that dramatically reduces network data transfer for package signing. Instead of sending multi-GB RPM files to a remote signing server, it sends only the RPM header (~4KB) and caches the signature for reuse.

**Key Innovation:** Exploit rpmsign's dual GPG invocation pattern using ppid-based signature caching.

**Network Savings:** **250x reduction** - 1GB RPM requires only ~4KB transfer instead of 1GB+

---

## POC Architecture Overview

### Components

**1. Client: `gpgshim`** (`build_tools/packaging/linux/gpgshim`)
- Drop-in GPG replacement that intercepts rpmsign calls
- Python 3.6+ with zero external dependencies (stdlib only)
- Implements ppid-based signature caching for RPM v4 signing optimization
- JWT Bearer token authentication
- TLS/HTTPS support with custom CA certs
- Exponential backoff retry logic

**2. Server: `signing-server.py`** (`signing_infrastructure/server/`)
- Multi-threaded HTTP server for GPG signing
- Receives data (header), returns GPG signature
- JWT authentication with role-based authorization (RBAC)
- Rate limiting, audit logging, DoS protection
- TLS 1.2+ support
- Thread semaphore for concurrent operation limits

**3. Authentication Module: `auth.py`**
- HMAC-SHA256 JWT token validation
- Role-based key access control
- Per-role rate limiting
- Comprehensive audit trail

**4. Documentation** (`signing_infrastructure/docs/`)
- `README.md` - Architecture and usage
- `authentication.md` - Auth system design
- `deployment-aws.md` - AWS deployment (ALB + EC2)
- `github-actions-integration.md` - CI/CD integration
- `manual-signing.md` - Admin manual signing workflow
- `config/README.md` - Configuration reference

---

## How RPM Signing Optimization Works

### The Problem

`rpmsign` calls GPG **twice** to sign RPMs:
1. **First call:** Sign RPM header (~4KB)
2. **Second call:** Sign full RPM (up to 1GB+)

**Naive solution:** Send both to remote server = **1GB+ network transfer**

### POC Solution: ppid-Based Signature Caching

```
rpmsign process (ppid=12345)
  │
  ├─ Call 1: gpgshim (pid=12346, ppid=12345)
  │   ├─ Read RPM header from stdin (~4KB)
  │   ├─ POST to signing server: base64(header)
  │   ├─ Receive signature (~256 bytes)
  │   ├─ Cache to /tmp/gpgshim-cache-12345.sig  ← KEY: cache by ppid!
  │   └─ Return signature to rpmsign
  │
  └─ Call 2: gpgshim (pid=12347, ppid=12345)  ← SAME ppid!
      ├─ Read full RPM from stdin (1GB) - MUST consume pipe
      ├─ Check cache: /tmp/gpgshim-cache-12345.sig exists!
      ├─ NO SERVER CALL - use cached signature
      ├─ Delete cache file
      └─ Return cached signature to rpmsign
```

**Result:** Only ~4KB sent to server (header from Call 1), signature reused for Call 2.

**Network transfer:** Header (4KB) + Signature (256 bytes) = **~4.3KB total** instead of 1GB+

---

## GitHub Actions Integration Pattern

```yaml
steps:
  - name: Install gpgshim
    run: |
      curl -o ~/.local/bin/gpgshim https://signing-server/gpgshim
      chmod +x ~/.local/bin/gpgshim

  - name: Sign RPMs
    env:
      GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
      GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
    run: |
      # rpmsign uses gpgshim instead of system gpg
      rpmsign --addsign \
        --define "_gpg_path $HOME/.local/bin/gpgshim" \
        dist/*.rpm
```

**No changes to build scripts required** - just configure RPM `_gpg_path` macro.

---

## Security Features

| Layer | Feature | Implementation |
|-------|---------|----------------|
| **Authentication** | JWT Bearer tokens | HMAC-SHA256 signed, configurable expiration |
| **Authorization** | Role-based access | Per-role allowed keys, digest algorithms |
| **Rate Limiting** | Per-client quotas | Configurable requests/hour per role |
| **DoS Protection** | Request size limits | Default 10KB max, configurable timeout |
| **Audit** | Comprehensive logging | All requests logged with client_id, role, key, timestamp |
| **Transport** | TLS/HTTPS | TLS 1.2+ only, custom CA cert support |
| **Input Validation** | key_id sanitization | Prevents command injection, directory traversal |
| **Key Isolation** | GNUPGHOME | Separate GPG keyring per environment |

---

## POC vs Original Plan Comparison

| Aspect | Original Plan | POC Implementation | Winner |
|--------|---------------|-------------------|--------|
| **Network Transfer** | Send full package (~1GB) | Send header only (~4KB) | ✅ POC (250x better) |
| **Client Integration** | Modify build_package.py | Drop-in GPG replacement | ✅ POC (simpler) |
| **Dependencies** | FastAPI, uvicorn, SQLAlchemy, Vault client | Python stdlib only | ✅ POC (zero deps) |
| **Authentication** | GitHub OIDC | JWT tokens | ↔️ Tie (both secure) |
| **Key Management** | HashiCorp Vault/AWS KMS | GPG keyring (can add Vault) | ↔️ Tie |
| **Database** | PostgreSQL for audit | File-based audit log | ↔️ Original (better audit queries) |
| **Deployment** | Kubernetes | Simple HTTP server | ✅ POC (simpler) |
| **Documentation** | Plan only | Comprehensive docs + tests | ✅ POC |

---

## Challenges & Gaps

### 1. DEB Package Signing
- **✅ RESOLVED:** DEB repositories only need Release file signing, not individual package signing
- apt verifies: Release signature → package checksums in Release file
- No individual .deb package signatures needed (unlike RPM)

### 2. Repository Metadata Signing
- DEB: `Release`, `InRelease`, `Release.gpg` files
- RPM: `repomd.xml.asc`
- **Gap:** POC doesn't include metadata signing scripts
- **Need:** Client-side script to call signing server API for metadata files

### 3. Token Management
- JWT tokens have expiration
- **Gap:** No automated token rotation mechanism documented
- **Need:** Token rotation procedure for production

### 4. Monitoring & Alerting
- Audit logs to file
- **Gap:** No integration with monitoring systems (Prometheus, Grafana, CloudWatch)
- **Need:** Metrics export, alerting on unauthorized attempts

---

## Implementation Plan

### Phase 1: Signing Server Deployment (Week 1-2)

#### 1.1 Deploy Signing Server Infrastructure

**Deployment: AWS EC2 + Application Load Balancer**

```bash
# 1. Launch EC2 instance (Ubuntu 22.04 LTS)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --security-group-ids sg-signing-server \
  --key-name signing-server-key \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=gpg-signing-server}]'

# 2. Install dependencies
ssh ubuntu@signing-server
sudo apt-get update
sudo apt-get install -y python3 gnupg2

# 3. Setup GPG keyring
sudo mkdir -p /etc/gpg-signing/keyring
sudo chmod 700 /etc/gpg-signing/keyring
export GNUPGHOME=/etc/gpg-signing/keyring

# 4. Import signing keys (one per release type)
gpg --import dev-signing-key.asc
gpg --import nightly-signing-key.asc
gpg --import release-signing-key.asc

# 5. Create systemd service
sudo cat > /etc/systemd/system/gpg-signing.service <<EOF
[Unit]
Description=GPG Signing Server
After=network.target

[Service]
Type=simple
User=gpg-signing
WorkingDirectory=/opt/gpg-signing
ExecStart=/usr/bin/python3 /opt/gpg-signing/server/signing-server.py \
  --host 0.0.0.0 \
  --port 8443 \
  --keyring /etc/gpg-signing/keyring \
  --enable-auth \
  --secrets-file /etc/gpg-signing/config/secrets.json \
  --authz-config /etc/gpg-signing/config/authorization.json \
  --enable-tls \
  --cert-file /etc/ssl/certs/signing-server.crt \
  --key-file /etc/ssl/private/signing-server.key \
  --audit-log /var/log/gpg-signing/audit.log
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable gpg-signing
sudo systemctl start gpg-signing
```

#### 1.2 Generate and Configure Signing Keys

For each release type: dev, nightly, prerelease, release

```bash
# Generate key (on secure workstation, not on server)
gpg --full-generate-key
# Real name: AMD ROCm TheRock <RELEASE_TYPE> Signing Key
# Email: therock-<RELEASE_TYPE>@amd.com
# Key type: RSA 4096
# Expiration: 1 year

# Export private key (encrypted)
gpg --armor --export-secret-keys therock-dev@amd.com > dev-signing-key.asc

# Export public key
gpg --armor --export therock-dev@amd.com > therock-dev-public.gpg

# Securely transfer private key to server
scp dev-signing-key.asc signing-server:/tmp/
ssh signing-server "sudo -u gpg-signing gpg --import /tmp/dev-signing-key.asc && rm /tmp/dev-signing-key.asc"
```

#### 1.3 Configure Authentication

**Create `config/secrets.json`:**
```json
{
  "clients": {
    "github-actions-dev": {
      "secret": "<generate-with-tools/generate-token.py>",
      "description": "GitHub Actions - development builds"
    },
    "github-actions-nightly": {
      "secret": "<generate-with-tools/generate-token.py>",
      "description": "GitHub Actions - nightly builds"
    },
    "github-actions-release": {
      "secret": "<generate-with-tools/generate-token.py>",
      "description": "GitHub Actions - release builds"
    }
  }
}
```

**Create `config/authorization.json`:**
```json
{
  "roles": {
    "development": {
      "allowed_keys": ["therock-dev@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 10000
    },
    "nightly": {
      "allowed_keys": ["therock-nightly@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 5000
    },
    "release": {
      "allowed_keys": ["therock-release@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 1000
    }
  }
}
```

**Generate JWT tokens:**
```bash
# Generate 30-day token for GitHub Actions (dev)
./tools/generate-token.py --generate \
  --client-id github-actions-dev \
  --role development \
  --secret-file config/secrets.json \
  --expires-hours 720

# Store token in GitHub Secrets as GPG_SERVER_TOKEN_DEV
```

---

### Phase 2: GitHub Actions Integration - RPM Signing (Week 2-3)

#### 2.1 Modify Multi-Arch Workflow

**File:** `.github/workflows/multi_arch_build_native_linux_packages.yml`

**Add after "Build Packages" step:**

```yaml
- name: Install gpgshim for RPM signing
  if: inputs.native_package_type == 'rpm' && env.RELEASE_TYPE != ''
  run: |
    mkdir -p ~/.local/bin
    # Copy gpgshim from repository (already committed in POC branch)
    cp build_tools/packaging/linux/gpgshim ~/.local/bin/gpgshim
    chmod +x ~/.local/bin/gpgshim

- name: Configure GPG signing environment
  if: inputs.native_package_type == 'rpm' && env.RELEASE_TYPE != ''
  run: |
    # Determine signing server token based on release type
    case "${{ env.RELEASE_TYPE }}" in
      dev)
        echo "GPG_SERVER_TOKEN=${{ secrets.GPG_SERVER_TOKEN_DEV }}" >> $GITHUB_ENV
        ;;
      nightly)
        echo "GPG_SERVER_TOKEN=${{ secrets.GPG_SERVER_TOKEN_NIGHTLY }}" >> $GITHUB_ENV
        ;;
      release|prerelease)
        echo "GPG_SERVER_TOKEN=${{ secrets.GPG_SERVER_TOKEN_RELEASE }}" >> $GITHUB_ENV
        ;;
      *)
        echo "Unknown release type: ${{ env.RELEASE_TYPE }}"
        exit 1
        ;;
    esac

    # Set signing server URL
    echo "GPG_SIGNING_SERVER=${{ secrets.GPG_SIGNING_SERVER }}" >> $GITHUB_ENV

- name: Sign RPM packages
  if: inputs.native_package_type == 'rpm' && env.RELEASE_TYPE != ''
  env:
    PATH: /home/runner/.local/bin:${{ env.PATH }}
  run: |
    cd ${{ env.PACKAGE_DIST_DIR }}/rpm

    # Find all RPMs
    find . -name "*.rpm" | while read rpm_file; do
      echo "Signing: $(basename $rpm_file)"

      # rpmsign will call gpgshim instead of system gpg
      rpmsign --addsign \
        --define "_gpg_path $HOME/.local/bin/gpgshim" \
        "$rpm_file"

      # Verify signature
      rpm --checksig "$rpm_file"
    done

    echo "✅ All RPM packages signed"
```

**Add GitHub Secrets:**
- Navigate to repository Settings → Secrets and variables → Actions
- Add:
  - `GPG_SIGNING_SERVER` = `https://signing-server.example.com/sign`
  - `GPG_SERVER_TOKEN_DEV` = `<JWT token for dev>`
  - `GPG_SERVER_TOKEN_NIGHTLY` = `<JWT token for nightly>`
  - `GPG_SERVER_TOKEN_RELEASE` = `<JWT token for release>`

#### 2.2 Upload Public Keys to S3

```yaml
- name: Upload GPG public key to S3
  if: inputs.native_package_type == 'rpm' && env.RELEASE_TYPE != ''
  run: |
    # Download public key from signing server or use pre-stored key
    curl -o therock-${{ env.RELEASE_TYPE }}-public.gpg \
      https://signing-server.example.com/keys/therock-${{ env.RELEASE_TYPE }}-public.gpg

    # Upload to S3 (same bucket as packages)
    aws s3 cp therock-${{ env.RELEASE_TYPE }}-public.gpg \
      s3://${{ steps.s3_config.outputs.s3_bucket }}/v3/keys/therock-${{ env.RELEASE_TYPE }}-public.gpg \
      --acl public-read
```

---

### Phase 3: Repository Metadata Signing (Week 3)

**Note:** For DEB repositories, we only need to sign the repository metadata (Release file), not individual .deb packages. The apt package manager verifies:
1. Release file signature (GPG)
2. Package checksums listed in Release file (SHA256)

This is different from RPM where individual packages contain embedded signatures.

#### 3.1 Create Metadata Signing Script

**File:** `build_tools/packaging/linux/sign_repo_metadata.py`

```python
#!/usr/bin/env python3
"""
Sign repository metadata files (Release, repomd.xml) using remote signing server.
"""

import sys
import os
import json
import base64
from pathlib import Path
from urllib.request import Request, urlopen

def sign_metadata(metadata_file: Path, key_id: str, server_url: str, token: str, clearsign: bool = False):
    """Sign repository metadata file."""

    with open(metadata_file, 'rb') as f:
        metadata_data = f.read()

    payload = {
        'data': base64.b64encode(metadata_data).decode('ascii'),
        'key_id': key_id,
        'digest_algo': 'SHA256',
        'armor': True,  # Metadata signatures are ASCII-armored
        'clearsign': clearsign
    }

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    request = Request(
        server_url,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers,
        method='POST'
    )

    with urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode('utf-8'))

    signature = base64.b64decode(result['signature'])
    return signature

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sign repository metadata')
    parser.add_argument('--metadata-file', required=True, help='Metadata file to sign')
    parser.add_argument('--output', required=True, help='Output signature file')
    parser.add_argument('--key-id', required=True, help='GPG key ID')
    parser.add_argument('--server', required=True, help='Signing server URL')
    parser.add_argument('--token', required=True, help='Authentication token')
    parser.add_argument('--clearsign', action='store_true', help='Create clearsigned output')

    args = parser.parse_args()

    signature = sign_metadata(
        Path(args.metadata_file),
        args.key_id,
        args.server,
        args.token,
        args.clearsign
    )

    with open(args.output, 'wb') as f:
        f.write(signature)

    print(f'✅ Signed metadata: {args.metadata_file} → {args.output}')
```

#### 3.2 Integrate Metadata Signing into upload_package_repo.py

**File:** `build_tools/packaging/linux/upload_package_repo.py`

Add function and modify upload functions to call signing when environment variables are set.

---

## Verification Steps

### 1. Verify Signing Server

```bash
# Test authentication
TOKEN=$(./tools/generate-token.py --generate --client-id test --role development --secret-file config/secrets.json --expires-hours 1)

curl -X POST https://signing-server.example.com/sign \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"data": "'$(echo "test" | base64)'", "key_id": "therock-dev@amd.com", "digest_algo": "SHA256", "armor": true}'

# Should return JSON with signature
```

### 2. Verify RPM Signing

```bash
# Trigger workflow
gh workflow run multi_arch_build_native_linux_packages.yml -f native_package_type=rpm -f release_type=dev

# Download and verify
aws s3 cp s3://therock-dev-packages/v3/packages/rpm/x86_64/rocm-hip-runtime.rpm ./
rpm --checksig rocm-hip-runtime.rpm
# Should show: gpg OK
```

### 3. Verify DEB Repository Metadata Signing

```bash
# Trigger workflow
gh workflow run multi_arch_build_native_linux_packages.yml -f native_package_type=deb -f release_type=dev

# Download and verify Release file signatures
aws s3 cp s3://therock-dev-packages/v3/packages/deb/dists/stable/InRelease ./
aws s3 cp s3://therock-dev-packages/v3/packages/deb/dists/stable/Release.gpg ./
aws s3 cp s3://therock-dev-packages/v3/packages/deb/dists/stable/Release ./

# Import public key
wget -O - https://therock-dev-packages.s3.amazonaws.com/v3/keys/therock-dev-public.gpg | gpg --import

# Verify signatures
gpg --verify InRelease
gpg --verify Release.gpg Release
# Both should show: Good signature from "AMD ROCm TheRock Dev Signing Key"
```

### 4. Verify End-to-End Installation

```bash
# RPM (RHEL/Rocky/AlmaLinux)
wget -O - https://therock-dev-packages.s3.amazonaws.com/v3/keys/therock-dev-public.gpg | sudo rpm --import -
sudo yum install rocm-hip-runtime  # Should verify signature

# DEB (Ubuntu/Debian)
wget -O - https://therock-dev-packages.s3.amazonaws.com/v3/keys/therock-dev-public.gpg | sudo apt-key add -
sudo apt install rocm-hip-runtime  # Should verify signature
```

---

## Critical Files Summary

### Files Already in POC (Ready to Use)

1. **`build_tools/packaging/linux/gpgshim`** - GPG shim client
2. **`build_tools/packaging/linux/signing_infrastructure/`** - Complete signing server infrastructure

### Files to Create

1. **`build_tools/packaging/linux/sign_repo_metadata.py`** - Repository metadata signing (DEB Release, RPM repomd.xml)
2. **`docs/packaging/PACKAGE_SIGNING.md`** - User documentation

### Files to Modify

1. **`.github/workflows/multi_arch_build_native_linux_packages.yml`**
   - Add gpgshim installation step (RPM only)
   - Add signing environment configuration
   - Add RPM package signing step (using gpgshim)
   - Add public key upload step

2. **`build_tools/packaging/linux/upload_package_repo.py`**
   - Add `sign_deb_release_file()` function (signs Release → creates InRelease + Release.gpg)
   - Add `sign_rpm_repomd()` function (signs repomd.xml → creates repomd.xml.asc)
   - Modify `upload_deb_metadata()` to call signing after Release generation
   - Modify `regenerate_rpm_metadata_from_s3()` to call signing after repomd.xml generation

---

## Timeline

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1-2 | Signing Server Deployment | Production signing server running, keys configured, authentication setup |
| 2-3 | RPM Package Signing | gpgshim integrated into workflow, RPM packages signed |
| 3 | Repository Metadata Signing | DEB Release + RPM repomd.xml signing integrated |
| 4 | Testing & Validation | All tests passing, end-to-end verification complete |
| 5 | Production Hardening | Monitoring, alerting, token rotation configured |
| 6 | Documentation | User docs, runbooks, internal docs complete |

**Total:** 6 weeks (~1.5 months)

**Note:** Timeline reduced from 7 to 6 weeks since DEB packages don't require individual signing.

---

## Advantages of POC Approach

✅ **Minimal network transfer** - Only sends digest/header, not full package (250x reduction)
✅ **Drop-in compatibility** - Works with existing rpmsign workflow
✅ **Zero dependencies** - Client uses only Python stdlib
✅ **Production-ready** - Comprehensive auth, rate limiting, audit logging
✅ **Well documented** - Extensive docs for deployment, integration, manual signing
✅ **Tested** - Includes test suite, benchmarking, integration tests
✅ **Flexible deployment** - Can run on AWS, Azure, on-prem
✅ **No build script changes** - Just configure `_gpg_path` RPM macro

---

## Recommended Next Steps

1. **Deploy signing server** to AWS/Azure infrastructure
2. **Test RPM signing** with POC gpgshim in dev environment
3. **Develop DEB signing** script (direct API approach)
4. **Add metadata signing** scripts for Release/repomd.xml
5. **Integrate into workflows** following plan above
6. **Add monitoring** (Prometheus, CloudWatch)
7. **Implement token rotation** for production
