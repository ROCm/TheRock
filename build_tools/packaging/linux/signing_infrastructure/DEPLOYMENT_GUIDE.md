# Signing Server Deployment Guide

Complete step-by-step guide to deploy the GPG signing server and integrate with GitHub Actions.

## Table of Contents

1. [Server Setup](#server-setup)
2. [GPG Key Generation](#gpg-key-generation)
3. [Server Configuration](#server-configuration)
4. [Server Deployment](#server-deployment)
5. [GitHub Actions Integration](#github-actions-integration)
6. [Testing & Verification](#testing--verification)
7. [Troubleshooting](#troubleshooting)

---

## Server Setup

### Prerequisites

- Linux server (Ubuntu 22.04 LTS recommended)
- Python 3.8 or higher
- GPG 2.x installed
- Network access from GitHub Actions runners
- (Optional) Domain name and SSL certificate for HTTPS

### Step 1: Install Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and GPG
sudo apt-get install -y python3 python3-pip gnupg2

# Create dedicated user for signing service
sudo useradd -r -m -s /bin/bash gpg-signing
sudo mkdir -p /opt/gpg-signing
sudo chown gpg-signing:gpg-signing /opt/gpg-signing
```

### Step 2: Deploy Server Code

```bash
# Switch to signing user
sudo su - gpg-signing

# Create directory structure
mkdir -p /opt/gpg-signing/{server,config,tools,logs}
cd /opt/gpg-signing

# Copy server files from repository
# (Assuming you have the repository cloned)
cp /path/to/TheRock/build_tools/packaging/linux/signing_infrastructure/server/*.py server/
cp /path/to/TheRock/build_tools/packaging/linux/gpgshim .

# Install OIDC dependencies (optional, for OIDC support)
pip3 install PyJWT[crypto]>=2.8.0

# Or install from requirements.txt
pip3 install -r requirements.txt
```

---

## GPG Key Generation

### Step 3: Generate Signing Keys

Generate separate GPG keys for each release type (development, nightly, release).

```bash
# Switch to signing user
sudo su - gpg-signing

# Set GPG home directory
export GNUPGHOME=/opt/gpg-signing/keyring
mkdir -p $GNUPGHOME
chmod 700 $GNUPGHOME

# Generate development key
gpg --full-generate-key

# Interactive prompts:
# - Kind of key: (1) RSA and RSA
# - Key size: 4096
# - Key expiration: 1y (1 year)
# - Real name: AMD ROCm TheRock Development Signing Key
# - Email: therock-dev@amd.com
# - Comment: Development builds
# - Passphrase: (generate strong passphrase, store securely)

# Repeat for nightly key
gpg --full-generate-key
# - Real name: AMD ROCm TheRock Nightly Signing Key
# - Email: therock-nightly@amd.com
# - Comment: Nightly builds

# Repeat for release key
gpg --full-generate-key
# - Real name: AMD ROCm TheRock Release Signing Key
# - Email: therock-release@amd.com
# - Comment: Production releases
```

### Step 4: Export Public Keys

```bash
# Export public keys for distribution
mkdir -p /opt/gpg-signing/keys
gpg --armor --export therock-dev@amd.com > /opt/gpg-signing/keys/therock-dev-public.gpg
gpg --armor --export therock-nightly@amd.com > /opt/gpg-signing/keys/therock-nightly-public.gpg
gpg --armor --export therock-release@amd.com > /opt/gpg-signing/keys/therock-release-public.gpg

# List keys to verify
gpg --list-keys
```

**Expected output:**
```
pub   rsa4096 2024-03-06 [SC] [expires: 2025-03-06]
      ABC123DEF456...
uid           [ultimate] AMD ROCm TheRock Development Signing Key <therock-dev@amd.com>
sub   rsa4096 2024-03-06 [E] [expires: 2025-03-06]
```

---

## Server Configuration

### Step 5: Configure JWT Secrets (Optional - for JWT auth)

```bash
cd /opt/gpg-signing/config

# Generate secrets file
cat > secrets.json <<'EOF'
{
  "clients": {
    "github-actions-dev": {
      "secret": "REPLACE_WITH_RANDOM_SECRET_1",
      "description": "GitHub Actions - development builds"
    },
    "github-actions-nightly": {
      "secret": "REPLACE_WITH_RANDOM_SECRET_2",
      "description": "GitHub Actions - nightly builds"
    },
    "github-actions-release": {
      "secret": "REPLACE_WITH_RANDOM_SECRET_3",
      "description": "GitHub Actions - release builds"
    }
  }
}
EOF

# Generate random secrets
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# Copy output and replace REPLACE_WITH_RANDOM_SECRET_1
# Repeat for each secret

# Secure the secrets file
chmod 600 secrets.json
```

### Step 6: Configure Authorization

```bash
cd /opt/gpg-signing/config

# Copy example config
cp authorization-example.json authorization.json

# Edit to match your repositories
nano authorization.json
```

**Update `authorization.json`:**
```json
{
  "oidc_role_mapping": {
    "refs/heads/main": "release",
    "refs/heads/release/*": "release",
    "refs/heads/*": "development",
    "refs/pull/*": "development"
  },

  "roles": {
    "development": {
      "description": "Development builds",
      "allowed_keys": ["therock-dev@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 10000,

      "allowed_repositories": [
        "ROCm/TheRock",
        "ROCm/rockrel"
      ],
      "allowed_refs": [
        "refs/heads/*",
        "refs/pull/*"
      ],
      "allowed_workflows": [
        ".github/workflows/multi_arch_build_native_linux_packages.yml",
        ".github/workflows/build_native_linux_packages.yml"
      ]
    },

    "release": {
      "description": "Production releases",
      "allowed_keys": ["therock-release@amd.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 1000,

      "allowed_repositories": [
        "ROCm/TheRock",
        "ROCm/rockrel"
      ],
      "allowed_refs": [
        "refs/heads/main",
        "refs/heads/release/*",
        "refs/tags/v*"
      ],
      "allowed_workflows": [
        ".github/workflows/multi_arch_build_native_linux_packages.yml"
      ]
    }
  }
}
```

### Step 7: Configure Systemd Service

```bash
# Create systemd service file
sudo nano /etc/systemd/system/gpg-signing.service
```

**Service file:**
```ini
[Unit]
Description=GPG Signing Server
After=network.target

[Service]
Type=simple
User=gpg-signing
Group=gpg-signing
WorkingDirectory=/opt/gpg-signing

# Environment variables
Environment="GNUPGHOME=/opt/gpg-signing/keyring"
Environment="GPG_BINARY=gpg"
Environment="AUTH_ENABLED=true"
Environment="SECRETS_FILE=/opt/gpg-signing/config/secrets.json"
Environment="AUTHZ_CONFIG_FILE=/opt/gpg-signing/config/authorization.json"
Environment="AUDIT_LOG_FILE=/opt/gpg-signing/logs/audit.log"
Environment="MAX_REQUEST_SIZE=10240"
Environment="MAX_THREADS=10"
Environment="GPG_TIMEOUT=30"
Environment="OIDC_AUDIENCE=amd-signing-service"

# Start server
ExecStart=/usr/bin/python3 /opt/gpg-signing/server/signing-server.py \
    --host 0.0.0.0 \
    --port 8443

# Restart policy
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/gpg-signing/logs

[Install]
WantedBy=multi-user.target
```

### Step 8: Configure TLS/HTTPS (Recommended)

**Option A: Use reverse proxy (Nginx/Apache)**

```bash
# Install Nginx
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Configure Nginx
sudo nano /etc/nginx/sites-available/signing-server
```

**Nginx config:**
```nginx
server {
    listen 443 ssl http2;
    server_name signing.yourdomain.com;

    # SSL certificate (use Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/signing.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/signing.yourdomain.com/privkey.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;

    # Proxy to signing server
    location / {
        proxy_pass http://127.0.0.1:8443;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name signing.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/signing-server /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Get SSL certificate
sudo certbot --nginx -d signing.yourdomain.com
```

**Option B: Direct TLS (Python server)**

Modify the systemd service to use TLS:
```ini
ExecStart=/usr/bin/python3 /opt/gpg-signing/server/signing-server.py \
    --host 0.0.0.0 \
    --port 8443 \
    --enable-tls \
    --cert-file /path/to/cert.pem \
    --key-file /path/to/key.pem
```

---

## Server Deployment

### Step 9: Start the Server

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable gpg-signing
sudo systemctl start gpg-signing

# Check status
sudo systemctl status gpg-signing

# View logs
sudo journalctl -u gpg-signing -f
```

**Expected output:**
```
● gpg-signing.service - GPG Signing Server
     Loaded: loaded (/etc/systemd/system/gpg-signing.service; enabled)
     Active: active (running) since Wed 2024-03-06 10:00:00 UTC
```

### Step 10: Configure Firewall

```bash
# Allow HTTPS traffic
sudo ufw allow 443/tcp
sudo ufw allow 8443/tcp  # If not using reverse proxy

# Restrict access to GitHub Actions IP ranges (optional but recommended)
# GitHub Actions IP ranges: https://api.github.com/meta
curl -s https://api.github.com/meta | jq -r '.actions[]' | while read ip; do
    sudo ufw allow from $ip to any port 443 proto tcp
done

# Enable firewall
sudo ufw enable
sudo ufw status
```

### Step 11: Upload Public Keys to S3

```bash
# Upload public keys for end users to download
aws s3 cp /opt/gpg-signing/keys/therock-dev-public.gpg \
    s3://your-bucket/keys/therock-dev-public.gpg \
    --acl public-read

aws s3 cp /opt/gpg-signing/keys/therock-nightly-public.gpg \
    s3://your-bucket/keys/therock-nightly-public.gpg \
    --acl public-read

aws s3 cp /opt/gpg-signing/keys/therock-release-public.gpg \
    s3://your-bucket/keys/therock-release-public.gpg \
    --acl public-read
```

---

## GitHub Actions Integration

### OIDC Token Flow Explanation

Here's how the OIDC token flows from GitHub Actions through gpgshim to the signing server:

```
┌─────────────────────────────────────────────────────────────┐
│ GitHub Actions Workflow                                      │
│                                                               │
│ 1. Request OIDC token from GitHub                           │
│    OIDC_TOKEN=$(curl $ACTIONS_ID_TOKEN_REQUEST_URL...)      │
│                                                               │
│ 2. Set as environment variable                               │
│    env:                                                       │
│      GPG_SERVER_TOKEN: ${{ steps.oidc.outputs.token }}      │
│                                                               │
│ 3. Invoke rpmsign (which calls gpgshim)                     │
│    rpmsign --define "_gpg_path gpgshim" file.rpm            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ gpgshim (Client - reads environment variables)               │
│                                                               │
│ Line 40: self.auth_token = os.environ.get('GPG_SERVER_TOKEN')│
│ Line 30: self.signing_server = os.environ.get('GPG_SIGNING_SERVER')│
│                                                               │
│ Line 248-249: Add token to HTTP headers                     │
│   if self.auth_token:                                        │
│       headers['Authorization'] = 'Bearer {}'.format(token)   │
│                                                               │
│ Send signing request:                                        │
│   POST https://signing-server/sign                           │
│   Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6Ikp...   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Signing Server                                               │
│                                                               │
│ 1. Extract token from Authorization header                  │
│    token = auth_header[7:]  # Remove "Bearer " prefix       │
│                                                               │
│ 2. Try OIDC validation first (if PyJWT installed)           │
│    payload = validate_github_oidc_token(token)               │
│    - Fetch GitHub's JWKS (RS256 public keys)                │
│    - Verify signature cryptographically                     │
│    - Validate claims: audience, issuer, expiration          │
│    - Extract: repository, ref, workflow, actor              │
│                                                               │
│ 3. If OIDC validation succeeds:                             │
│    - Authorize based on OIDC claims                         │
│    - Check: repository in allowed_repositories              │
│    - Check: ref matches allowed_refs                        │
│    - Check: workflow in allowed_workflows                   │
│                                                               │
│ 4. If OIDC fails, try JWT validation (fallback)            │
│    payload = validate_jwt_token(token, secrets)              │
│    - Verify HMAC-SHA256 signature                           │
│    - Check: client_id, role                                 │
│                                                               │
│ 5. Sign package with GPG and return signature               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Key Points:**
- ✅ **gpgshim is token-agnostic** - it doesn't know or care if token is JWT or OIDC
- ✅ **Same environment variable** - both JWT and OIDC use `GPG_SERVER_TOKEN`
- ✅ **Server auto-detects** - tries OIDC first (RS256), falls back to JWT (HMAC-SHA256)
- ✅ **No code changes** - switching from JWT to OIDC only changes how token is generated in workflow
- ✅ **Backward compatible** - existing JWT workflows continue to work

**Environment Variables Read by gpgshim:**
- `GPG_SERVER_TOKEN` - Authentication token (JWT or OIDC) [**THIS IS WHERE OIDC TOKEN GOES**]
- `GPG_SIGNING_SERVER` - Server URL (e.g., `https://signing.yourdomain.com/sign`)
- `GPG_KEY_ID` - (Optional) GPG key ID to use
- `GPG_TIMEOUT` - (Optional) Request timeout in seconds
- `GPG_MAX_RETRIES` - (Optional) Max retry attempts
- `GPG_VERIFY_SSL` - (Optional) Enable/disable SSL verification

### Step 12: Configure GitHub Secrets

In your repository (ROCm/TheRock or ROCm/rockrel), add these secrets:

**Navigate to:** Settings → Secrets and variables → Actions → New repository secret

**Add the following secrets:**

1. **GPG_SIGNING_SERVER**
   - Value: `https://signing.yourdomain.com/sign`

2. **GPG_SERVER_TOKEN_DEV** (if using JWT)
   - Generate token:
     ```bash
     cd /opt/gpg-signing/tools
     python3 generate-token.py --generate \
       --client-id github-actions-dev \
       --role development \
       --secret-file ../config/secrets.json \
       --expires-hours 720
     ```
   - Copy the token output

3. **GPG_SERVER_TOKEN_NIGHTLY** (if using JWT)
   - Same as above, but with `--role nightly`

4. **GPG_SERVER_TOKEN_RELEASE** (if using JWT)
   - Same as above, but with `--role release`

**Note:** If using OIDC, you don't need to store tokens in secrets!

### Step 13: Update GitHub Actions Workflow

**File:** `.github/workflows/build_native_linux_packages.yml`

**Option A: Using OIDC (Recommended)**

```yaml
name: Build and Sign Native Packages

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

    # REQUIRED: Enable OIDC token generation
    permissions:
      id-token: write
      contents: read

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      # Get OIDC token from GitHub
      - name: Get OIDC token for signing
        id: oidc
        run: |
          OIDC_TOKEN=$(curl -sSL \
            -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
            "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=amd-signing-service" \
            | jq -r '.value')

          echo "::add-mask::$OIDC_TOKEN"
          echo "token=$OIDC_TOKEN" >> $GITHUB_OUTPUT

      # Your build steps here
      - name: Build packages
        run: |
          python build_tools/packaging/linux/build_package.py \
            --dest-dir ./dist \
            --rocm-version ${{ inputs.rocm_version }} \
            --pkg-type ${{ inputs.native_package_type }}

      # Install gpgshim
      - name: Install gpgshim for RPM signing
        if: inputs.native_package_type == 'rpm'
        run: |
          mkdir -p ~/.local/bin
          cp build_tools/packaging/linux/gpgshim ~/.local/bin/gpgshim
          chmod +x ~/.local/bin/gpgshim

      # Sign RPM packages
      - name: Sign RPM packages
        if: inputs.native_package_type == 'rpm'
        env:
          # gpgshim reads these environment variables:
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
          # ⭐ THIS IS WHERE THE OIDC TOKEN IS USED - gpgshim reads GPG_SERVER_TOKEN env var
          GPG_SERVER_TOKEN: ${{ steps.oidc.outputs.token }}  # OIDC token from step above
          PATH: /home/runner/.local/bin:${{ env.PATH }}
        run: |
          cd dist

          # Sign all RPM packages
          find . -name "*.rpm" | while read rpm_file; do
            echo "Signing: $(basename $rpm_file)"

            # rpmsign calls gpgshim (via _gpg_path)
            # gpgshim reads GPG_SERVER_TOKEN env var and sends it to signing server
            rpmsign --addsign \
              --define "_gpg_path $HOME/.local/bin/gpgshim" \
              "$rpm_file"

            # Verify signature
            rpm --checksig "$rpm_file"
            echo "✅ Signed: $(basename $rpm_file)"
          done

      # Sign DEB repository metadata
      - name: Sign DEB repository metadata
        if: inputs.native_package_type == 'deb'
        env:
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
          GPG_SERVER_TOKEN: ${{ steps.oidc.outputs.token }}
        run: |
          # Your DEB metadata signing script
          python build_tools/packaging/linux/sign_repo_metadata.py \
            --metadata-file dist/deb/dists/stable/Release \
            --output dist/deb/dists/stable/InRelease \
            --server "$GPG_SIGNING_SERVER" \
            --token "$GPG_SERVER_TOKEN" \
            --clearsign

      # Upload packages
      - name: Upload packages
        run: |
          # Your S3 upload logic
          aws s3 sync dist/ s3://your-bucket/packages/
```

**Option B: Using JWT Tokens**

```yaml
# Same as OIDC but replace OIDC token step with:

      - name: Set JWT token based on branch
        id: jwt
        run: |
          if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
            echo "token=${{ secrets.GPG_SERVER_TOKEN_RELEASE }}" >> $GITHUB_OUTPUT
          else
            echo "token=${{ secrets.GPG_SERVER_TOKEN_DEV }}" >> $GITHUB_OUTPUT
          fi

      # Then use ${{ steps.jwt.outputs.token }} instead of OIDC token
```

---

## Testing & Verification

### Step 14: Test Server Locally

```bash
# Test health endpoint
curl https://signing.yourdomain.com/health
# Expected: {"status": "ok"}

# Test signing (with JWT)
TOKEN="your-jwt-token"
curl -X POST https://signing.yourdomain.com/sign \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data": "'$(echo "test data" | base64)'",
    "key_id": "therock-dev@amd.com",
    "digest_algo": "SHA256",
    "armor": true
  }'

# Expected: {"signature": "...", "key_id": "...", "digest_algo": "SHA256"}
```

### Step 15: Test from GitHub Actions

```bash
# Trigger workflow manually
gh workflow run build_native_linux_packages.yml \
  --ref main \
  -f native_package_type=rpm \
  -f rocm_version=8.0.0

# Monitor workflow
gh run view --log

# Check for successful signing messages
# Look for: "✅ Signed: package.rpm"
```

### Step 16: Verify Audit Logs

```bash
# On signing server
sudo tail -f /opt/gpg-signing/logs/audit.log

# Should show entries like:
# {"timestamp":"2024-03-06T10:30:00Z","auth_type":"oidc","repository":"ROCm/TheRock",...}
```

### Step 17: Test Package Installation

```bash
# Download signed package
aws s3 cp s3://your-bucket/packages/rpm/rocm-hip.rpm ./

# Import public key
wget https://your-bucket.s3.amazonaws.com/keys/therock-dev-public.gpg
sudo rpm --import therock-dev-public.gpg

# Verify signature
rpm --checksig rocm-hip.rpm
# Expected: rocm-hip.rpm: digests signatures OK

# Install package (signature verified automatically)
sudo yum install ./rocm-hip.rpm
```

---

## Troubleshooting

### Server Not Starting

```bash
# Check logs
sudo journalctl -u gpg-signing -n 50

# Common issues:
# 1. Port already in use
sudo netstat -tlnp | grep 8443

# 2. GPG keyring permissions
ls -la /opt/gpg-signing/keyring
chmod 700 /opt/gpg-signing/keyring

# 3. Missing dependencies
pip3 list | grep PyJWT
```

### Authentication Failures

```bash
# Check audit log
tail -f /opt/gpg-signing/logs/audit.log | grep AUTH_FAILED

# For OIDC:
# - Verify PyJWT installed: pip3 show PyJWT
# - Check OIDC_AUDIENCE matches in workflow
# - Verify id-token: write permission in workflow

# For JWT:
# - Verify token not expired
# - Check secrets.json has correct client_id
# - Verify token in GitHub Secrets is correct
```

### Authorization Failures

```bash
# Check audit log
tail -f /opt/gpg-signing/logs/audit.log | grep DENIED

# Common reasons:
# 1. Repository not in allowed_repositories
# 2. Branch not in allowed_refs
# 3. Workflow not in allowed_workflows
# 4. Wrong key_id for role

# Verify authorization.json
cat /opt/gpg-signing/config/authorization.json | jq '.roles.development'
```

### Signing Failures

```bash
# On server, check GPG keys
sudo su - gpg-signing
export GNUPGHOME=/opt/gpg-signing/keyring
gpg --list-keys

# Test GPG signing manually
echo "test" | gpg --clearsign --local-user therock-dev@amd.com

# Check GPG permissions
ls -la $GNUPGHOME
```

### Network Issues

```bash
# Test connectivity from GitHub Actions
# Add to workflow:
- name: Test signing server connectivity
  run: |
    curl -v https://signing.yourdomain.com/health

# Check firewall rules
sudo ufw status numbered

# Check Nginx logs (if using reverse proxy)
sudo tail -f /var/log/nginx/error.log
```

---

## Monitoring & Maintenance

### Monitor Server Health

```bash
# Check service status
sudo systemctl status gpg-signing

# Monitor resource usage
htop

# Check disk space (audit logs can grow)
df -h /opt/gpg-signing/logs

# Rotate logs
sudo logrotate -f /etc/logrotate.d/gpg-signing
```

### Log Rotation

Create `/etc/logrotate.d/gpg-signing`:
```
/opt/gpg-signing/logs/audit.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 gpg-signing gpg-signing
    postrotate
        systemctl reload gpg-signing
    endscript
}
```

### Key Rotation

**Before keys expire (check expiration):**
```bash
gpg --list-keys
# Look for [expires: YYYY-MM-DD]

# Extend expiration (if needed)
gpg --edit-key therock-dev@amd.com
# At prompt: expire
# Follow prompts to extend

# Or generate new key and transition
# 1. Generate new key
# 2. Update authorization.json to allow both keys
# 3. Update workflows to use new key
# 4. After transition period, remove old key
```

---

## Security Best Practices

1. **Restrict network access** - Use firewall rules to allow only GitHub Actions IPs
2. **Use OIDC** - Prefer OIDC over JWT tokens (no secrets stored)
3. **Monitor audit logs** - Set up alerts for unauthorized attempts
4. **Regular key rotation** - Rotate signing keys annually
5. **Secure GPG keyring** - Proper permissions (700) and backups
6. **Use HTTPS** - Always use TLS for server communication
7. **Separate keys per environment** - Different keys for dev/nightly/release
8. **Protected branches** - Use `ref_protected` check in authorization

---

## Quick Reference

### Server Commands

```bash
# Start/Stop/Restart
sudo systemctl start gpg-signing
sudo systemctl stop gpg-signing
sudo systemctl restart gpg-signing

# View logs
sudo journalctl -u gpg-signing -f
tail -f /opt/gpg-signing/logs/audit.log

# Test signing
curl -X POST https://signing.yourdomain.com/sign ...
```

### GitHub Actions

```bash
# Trigger workflow
gh workflow run build_native_linux_packages.yml \
  -f native_package_type=rpm \
  -f rocm_version=8.0.0

# View workflow run
gh run view --log
gh run list --workflow=build_native_linux_packages.yml
```

### Key Management

```bash
# List keys
export GNUPGHOME=/opt/gpg-signing/keyring
gpg --list-keys

# Export public key
gpg --armor --export therock-dev@amd.com

# Test signing
echo "test" | gpg --clearsign --local-user therock-dev@amd.com
```

---

## Support & Documentation

- **OIDC Integration:** `docs/github-oidc-integration.md`
- **Token Reference:** `docs/oidc-token-reference.md`
- **JWT vs OIDC:** `JWT_VS_OIDC_WORKFLOW_RESTRICTION.md`
- **Security Analysis:** `GITHUB_SECRETS_SECURITY_ANALYSIS.md`
- **Implementation Summary:** `OIDC_IMPLEMENTATION_SUMMARY.md`
