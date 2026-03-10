# GitHub Actions OIDC Integration Guide

This guide shows how to integrate GitHub Actions OIDC authentication with the signing server.

## Overview

GitHub Actions OIDC provides **keyless authentication** with short-lived tokens (10 minutes) that contain rich workflow metadata:
- Repository name
- Branch/ref
- Workflow file path
- Actor (triggering user)
- Run ID, run number, etc.

The signing server validates these tokens using GitHub's public JWKS (JSON Web Key Set) and enforces workflow-based authorization.

## Prerequisites

1. **Server Side:**
   - Install PyJWT: `pip install PyJWT[crypto]>=2.8.0`
   - Configure `authorization.json` with OIDC rules (see `config/authorization-example.json`)
   - Set environment variable: `OIDC_AUDIENCE=amd-signing-service`

2. **Workflow Side:**
   - Add `id-token: write` permission to workflow
   - Request OIDC token from GitHub Actions

## GitHub Actions Workflow Integration

### Step 1: Add OIDC Permission

```yaml
jobs:
  build_and_sign:
    name: Build and sign packages
    runs-on: ubuntu-24.04

    # REQUIRED: Enable OIDC token generation
    permissions:
      id-token: write
      contents: read
```

### Step 2: Request OIDC Token

```yaml
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Get OIDC token from GitHub
        id: oidc
        run: |
          # Request OIDC token with signing server as audience
          OIDC_TOKEN=$(curl -sSL \
            -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
            "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" \
            | jq -r '.value')

          # Mask token in logs
          echo "::add-mask::$OIDC_TOKEN"

          # Export for subsequent steps
          echo "token=$OIDC_TOKEN" >> $GITHUB_OUTPUT
```

### Step 3: Configure gpgshim with OIDC Token

```yaml
      - name: Install gpgshim
        run: |
          mkdir -p ~/.local/bin
          cp build_tools/packaging/linux/gpgshim ~/.local/bin/gpgshim
          chmod +x ~/.local/bin/gpgshim

      - name: Sign RPM packages
        env:
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
          GPG_SERVER_TOKEN: ${{ steps.oidc.outputs.token }}
        run: |
          # gpgshim will use the OIDC token for authentication
          rpmsign --addsign \
            --define "_gpg_path $HOME/.local/bin/gpgshim" \
            dist/*.rpm
```

### Complete Example Workflow

```yaml
name: Build and Sign Native Packages (OIDC)

on:
  workflow_dispatch:
    inputs:
      native_package_type:
        description: 'Package type (rpm or deb)'
        required: true
        type: choice
        options:
          - rpm
          - deb
      rocm_version:
        description: 'ROCm version'
        required: true
        type: string

jobs:
  build_and_sign:
    name: Build ${{ inputs.native_package_type }} packages
    runs-on: ubuntu-24.04

    # Enable OIDC token generation
    permissions:
      id-token: write
      contents: read

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Get OIDC token from GitHub
        id: oidc
        run: |
          OIDC_TOKEN=$(curl -sSL \
            -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
            "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" \
            | jq -r '.value')

          echo "::add-mask::$OIDC_TOKEN"
          echo "token=$OIDC_TOKEN" >> $GITHUB_OUTPUT

      - name: Build packages
        run: |
          # Your package build logic here
          python build_tools/packaging/linux/build_package.py \
            --dest-dir ./dist \
            --rocm-version ${{ inputs.rocm_version }} \
            --pkg-type ${{ inputs.native_package_type }}

      - name: Install gpgshim for signing
        run: |
          mkdir -p ~/.local/bin
          cp build_tools/packaging/linux/gpgshim ~/.local/bin/gpgshim
          chmod +x ~/.local/bin/gpgshim

      - name: Sign packages with OIDC authentication
        env:
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
          GPG_SERVER_TOKEN: ${{ steps.oidc.outputs.token }}
        run: |
          if [ "${{ inputs.native_package_type }}" == "rpm" ]; then
            # Sign RPM packages
            find dist -name "*.rpm" | while read rpm; do
              rpmsign --addsign \
                --define "_gpg_path $HOME/.local/bin/gpgshim" \
                "$rpm"
              echo "✅ Signed: $(basename $rpm)"
            done
          elif [ "${{ inputs.native_package_type }}" == "deb" ]; then
            # Sign DEB repository metadata
            python build_tools/packaging/linux/sign_repo_metadata.py \
              --metadata-file dist/deb/dists/stable/Release \
              --output dist/deb/dists/stable/InRelease \
              --server "$GPG_SIGNING_SERVER" \
              --token "$GPG_SERVER_TOKEN" \
              --clearsign
          fi

      - name: Upload packages
        run: |
          # Your S3 upload logic here
          aws s3 sync dist/ s3://therock-packages/
```

## OIDC Token Payload Example

When the signing server receives an OIDC token, it decodes to:

```json
{
  "jti": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
  "sub": "repo:ROCm/TheRock:ref:refs/heads/main",
  "aud": "amd-signing-service",
  "ref": "refs/heads/main",
  "sha": "abc123def456...",
  "repository": "ROCm/TheRock",
  "repository_owner": "ROCm",
  "repository_owner_id": "123456",
  "run_id": "123456789",
  "run_number": "42",
  "run_attempt": "1",
  "actor": "john.doe",
  "workflow": ".github/workflows/build_native_linux_packages.yml",
  "head_ref": "",
  "base_ref": "",
  "event_name": "workflow_dispatch",
  "ref_type": "branch",
  "job_workflow_ref": "ROCm/TheRock/.github/workflows/build_native_linux_packages.yml@refs/heads/main",
  "iss": "https://token.actions.githubusercontent.com",
  "nbf": 1709736000,
  "exp": 1709737200,
  "iat": 1709736300
}
```

## Server-Side Authorization

The signing server uses these claims to enforce authorization:

### 1. Repository Restriction

```json
{
  "roles": {
    "release": {
      "allowed_repositories": ["ROCm/TheRock"]
    }
  }
}
```

**Effect:** Only workflows from `ROCm/TheRock` can sign with the `release` role.

### 2. Branch Restriction

```json
{
  "roles": {
    "release": {
      "allowed_refs": [
        "refs/heads/main",
        "refs/heads/release/*"
      ]
    }
  }
}
```

**Effect:** Only `main` branch and `release/*` branches can use the `release` key.

### 3. Workflow Restriction

```json
{
  "roles": {
    "release": {
      "allowed_workflows": [
        ".github/workflows/build_native_linux_packages.yml"
      ]
    }
  }
}
```

**Effect:** Only the official build workflow can sign packages, preventing malicious workflow injection.

### 4. Dynamic Role Mapping

```json
{
  "oidc_role_mapping": {
    "refs/heads/main": "release",
    "refs/heads/release/*": "release",
    "refs/heads/*": "development",
    "refs/pull/*": "development"
  }
}
```

**Effect:**
- Main branch → uses `release` key
- Feature branches → uses `development` key
- Pull requests → uses `development` key

## Audit Trail

OIDC authentication provides rich audit logs:

```json
{
  "timestamp": "2024-03-06T10:30:00Z",
  "auth_type": "oidc",
  "repository": "ROCm/TheRock",
  "ref": "refs/heads/main",
  "workflow": ".github/workflows/build_native_linux_packages.yml",
  "actor": "john.doe",
  "run_id": "123456789",
  "run_number": "42",
  "event_name": "workflow_dispatch",
  "job_workflow_ref": "ROCm/TheRock/.github/workflows/build_native_linux_packages.yml@refs/heads/main",
  "role": "release",
  "key_id": "therock-release@amd.com",
  "digest_algo": "SHA256",
  "package_name": "rocm-hip-runtime.rpm",
  "package_hash": "abc123...",
  "status": "SUCCESS"
}
```

This enables:
- **Attribution:** Who triggered the signing?
- **Source tracking:** Which workflow and commit?
- **Anomaly detection:** Unexpected workflow or branch signing attempts

## Security Benefits vs JWT

| Feature | JWT (GitHub Secrets) | OIDC |
|---------|---------------------|------|
| **Secrets stored** | Yes (long-lived tokens) | No (GitHub-issued, short-lived) |
| **Token lifetime** | Days/weeks | 10 minutes |
| **Rotation required** | Manual | Automatic (every run) |
| **Workflow metadata** | Must be added manually | Built-in |
| **Tamper-proof** | Depends on secret protection | Cryptographically signed by GitHub |
| **Revocation** | Manual secret rotation | Automatic expiration |
| **Trust boundary** | Workflow generates token | GitHub generates token |

## Troubleshooting

### Error: "OIDC token validation failed"

**Cause:** PyJWT not installed or GitHub JWKS unreachable.

**Solution:**
```bash
pip install PyJWT[crypto]>=2.8.0
```

Check server logs for details:
```bash
tail -f /var/log/gpg-signing/audit.log
```

### Error: "Repository 'ROCm/TheRock' not authorized"

**Cause:** `authorization.json` doesn't include the repository.

**Solution:** Update `authorization.json`:
```json
{
  "roles": {
    "development": {
      "allowed_repositories": ["ROCm/TheRock"]
    }
  }
}
```

### Error: "Workflow not authorized"

**Cause:** Workflow file path not in `allowed_workflows`.

**Solution:** Add workflow to allowed list:
```json
{
  "roles": {
    "release": {
      "allowed_workflows": [
        ".github/workflows/your-workflow-name.yml"
      ]
    }
  }
}
```

### Error: "Cannot determine role from ref"

**Cause:** Branch doesn't match any pattern in `oidc_role_mapping`.

**Solution:** Add pattern to mapping:
```json
{
  "oidc_role_mapping": {
    "refs/heads/your-branch-pattern/*": "development"
  }
}
```

## Migration from JWT to OIDC

See [GITHUB_OIDC_MIGRATION_GUIDE.md](../GITHUB_OIDC_MIGRATION_GUIDE.md) for detailed migration steps.

**Quick summary:**
1. Install PyJWT on server: `pip install PyJWT[crypto]`
2. Update `authorization.json` with OIDC rules
3. Update workflow to request OIDC token (add `id-token: write`)
4. Keep JWT support enabled during transition
5. Verify OIDC works, then deprecate JWT tokens

## Testing OIDC Locally

You cannot generate GitHub OIDC tokens locally (they're issued by GitHub Actions only), but you can test the server's OIDC validation:

```bash
# Start server with OIDC enabled
cd build_tools/packaging/linux/signing_infrastructure
pip install PyJWT[crypto]
python3 server/signing-server.py \
  --enable-auth \
  --secrets-file config/secrets.json \
  --authz-config config/authorization.json \
  --port 8080

# Test with mock OIDC token (use tools/generate-mock-oidc-token.py if available)
# Or use act (local GitHub Actions) to get real OIDC tokens:
act workflow_dispatch -j build_and_sign
```

## Next Steps

- Read [JWT_VS_OIDC_WORKFLOW_RESTRICTION.md](../JWT_VS_OIDC_WORKFLOW_RESTRICTION.md) for comparison
- See [GITHUB_OIDC_MIGRATION_GUIDE.md](../GITHUB_OIDC_MIGRATION_GUIDE.md) for migration plan
- Review [SECURITY_COMPARISON.md](../SECURITY_COMPARISON.md) for security analysis
