# GitHub Actions Integration Guide

This guide covers integrating the GPG remote signing server with GitHub Actions CI/CD pipelines for automated RPM signing.

## Architecture

```
GitHub Actions Runner (Azure VM)
    |
    | 1. Build RPM
    v
RPM artifacts (unsigned)
    |
    | 2. Install gpgshim
    | 3. Configure GPG_SIGNING_SERVER
    | 4. Set GPG_SERVER_TOKEN (from GitHub Secrets)
    v
rpmsign --addsign package.rpm
    |
    | HTTPS POST with JWT token
    v
AWS Signing Server
    |
    | Validate token, sign digest
    v
Signed RPM
```

## Quick Start

### 1. Store Secrets in GitHub

Navigate to repository Settings → Secrets and variables → Actions

**Add these secrets:**

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `GPG_SIGNING_SERVER` | `https://signing.company.com/sign` | Signing server URL |
| `GPG_SERVER_TOKEN_PROD` | `eyJhbGci...` | Production signing token |
| `GPG_SERVER_TOKEN_DEV` | `eyJhbGci...` | Development signing token |

**Optional secrets:**

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `GPG_SERVER_CA_CERT` | `-----BEGIN CERTIFICATE-----...` | Custom CA cert (if not using public CA) |

### 2. Basic Workflow

Create `.github/workflows/build-and-sign.yml`:

```yaml
name: Build and Sign RPM

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-and-sign:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install build dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y rpm rpmbuild rpmdevtools

      - name: Build RPM
        run: |
          # Your RPM build commands here
          rpmbuild -ba mypackage.spec

      - name: Install gpgshim
        run: |
          curl -sSL https://github.com/yourorg/gpg-signing-server/releases/latest/download/gpgshim \
            -o gpgshim
          chmod +x gpgshim
          mkdir -p ~/.local/bin
          mv gpgshim ~/.local/bin/

      - name: Sign RPM (production)
        if: github.ref == 'refs/heads/main'
        env:
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
          GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
          PATH: ${{ github.workspace }}/.local/bin:${{ env.PATH }}
        run: |
          rpmsign --addsign --define "_gpg_path $HOME/.local/bin/gpgshim" \
            ~/rpmbuild/RPMS/*/*.rpm

      - name: Sign RPM (development)
        if: github.ref != 'refs/heads/main'
        env:
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
          GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_DEV }}
          PATH: ${{ github.workspace }}/.local/bin:${{ env.PATH }}
        run: |
          rpmsign --addsign --define "_gpg_path $HOME/.local/bin/gpgshim" \
            ~/rpmbuild/RPMS/*/*.rpm

      - name: Verify signature
        run: |
          rpm -qip ~/rpmbuild/RPMS/*/*.rpm
          rpm --checksig ~/rpmbuild/RPMS/*/*.rpm

      - name: Upload signed RPM
        uses: actions/upload-artifact@v4
        with:
          name: signed-rpms
          path: ~/rpmbuild/RPMS/*/*.rpm
```

## Advanced Workflows

### Production vs Development Signing

Use different tokens and keys for different environments:

```yaml
name: Multi-Environment Signing

on:
  push:
    branches: [main, develop]
    tags: ['v*']

jobs:
  sign-rpm:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Determine environment
        id: env
        run: |
          if [[ "${{ github.ref }}" == refs/tags/v* ]]; then
            echo "env=production" >> $GITHUB_OUTPUT
            echo "token_secret=GPG_SERVER_TOKEN_PROD" >> $GITHUB_OUTPUT
          elif [[ "${{ github.ref }}" == refs/heads/main ]]; then
            echo "env=staging" >> $GITHUB_OUTPUT
            echo "token_secret=GPG_SERVER_TOKEN_STAGING" >> $GITHUB_OUTPUT
          else
            echo "env=development" >> $GITHUB_OUTPUT
            echo "token_secret=GPG_SERVER_TOKEN_DEV" >> $GITHUB_OUTPUT
          fi

      - name: Build RPM
        run: |
          # Your build steps
          make build-rpm

      - name: Setup gpgshim
        run: |
          curl -sSL https://signing.company.com/client/gpgshim -o ~/.local/bin/gpgshim
          chmod +x ~/.local/bin/gpgshim

      - name: Sign RPM
        env:
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
          GPG_SERVER_TOKEN: ${{ secrets[steps.env.outputs.token_secret] }}
        run: |
          export PATH="$HOME/.local/bin:$PATH"
          rpmsign --addsign --define "_gpg_path $HOME/.local/bin/gpgshim" \
            dist/*.rpm

      - name: Tag with environment
        run: |
          for rpm in dist/*.rpm; do
            mv "$rpm" "${rpm%.rpm}-${{ steps.env.outputs.env }}.rpm"
          done

      - name: Upload to artifact registry
        run: |
          # Upload to JFrog Artifactory, Nexus, etc.
          echo "Uploading to ${{ steps.env.outputs.env }} repository"
```

### Parallel Signing

Sign multiple RPMs concurrently to speed up builds:

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      rpms: ${{ steps.find-rpms.outputs.rpms }}
    steps:
      - name: Build all RPMs
        run: make build-all

      - name: Find RPMs
        id: find-rpms
        run: |
          rpms=$(find dist/ -name "*.rpm" -printf "%f\n" | jq -R -s -c 'split("\n")[:-1]')
          echo "rpms=$rpms" >> $GITHUB_OUTPUT

      - name: Upload RPMs
        uses: actions/upload-artifact@v4
        with:
          name: unsigned-rpms
          path: dist/*.rpm

  sign:
    needs: build
    runs-on: ubuntu-latest
    strategy:
      matrix:
        rpm: ${{ fromJson(needs.build.outputs.rpms) }}
      max-parallel: 10
    steps:
      - name: Download RPM
        uses: actions/download-artifact@v4
        with:
          name: unsigned-rpms

      - name: Setup gpgshim
        run: |
          curl -sSL https://signing.company.com/client/gpgshim -o ~/.local/bin/gpgshim
          chmod +x ~/.local/bin/gpgshim

      - name: Sign ${{ matrix.rpm }}
        env:
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
          GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
        run: |
          export PATH="$HOME/.local/bin:$PATH"
          rpmsign --addsign --define "_gpg_path $HOME/.local/bin/gpgshim" \
            "${{ matrix.rpm }}"

      - name: Upload signed RPM
        uses: actions/upload-artifact@v4
        with:
          name: signed-${{ matrix.rpm }}
          path: ${{ matrix.rpm }}
```

### Caching gpgshim

Cache the gpgshim binary to avoid downloading on every run:

```yaml
steps:
  - name: Cache gpgshim
    id: cache-gpgshim
    uses: actions/cache@v4
    with:
      path: ~/.local/bin/gpgshim
      key: gpgshim-${{ hashFiles('.github/workflows/build.yml') }}

  - name: Download gpgshim
    if: steps.cache-gpgshim.outputs.cache-hit != 'true'
    run: |
      mkdir -p ~/.local/bin
      curl -sSL https://signing.company.com/client/gpgshim \
        -o ~/.local/bin/gpgshim
      chmod +x ~/.local/bin/gpgshim

  - name: Sign RPMs
    env:
      GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
      GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
      PATH: /home/runner/.local/bin:${{ env.PATH }}
    run: |
      rpmsign --addsign --define "_gpg_path $HOME/.local/bin/gpgshim" \
        dist/*.rpm
```

## Azure Build Servers

If using GitHub Actions with self-hosted runners on Azure VMs:

### 1. Setup Azure VM

**Create VM:**
```bash
az vm create \
  --resource-group github-runners \
  --name github-runner-01 \
  --image Ubuntu2204 \
  --size Standard_D2s_v3 \
  --admin-username azureuser \
  --ssh-key-values ~/.ssh/id_rsa.pub
```

**Install dependencies:**
```bash
ssh azureuser@<vm-ip>

# Install Docker and GitHub Actions runner
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker azureuser

# Download runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.313.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.313.0/actions-runner-linux-x64-2.313.0.tar.gz
tar xzf actions-runner-linux-x64-2.313.0.tar.gz

# Configure runner
./config.sh --url https://github.com/yourorg/yourrepo --token <token>

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

### 2. Network Configuration

**Allow outbound HTTPS to signing server:**
```bash
az network nsg rule create \
  --resource-group github-runners \
  --nsg-name github-runner-nsg \
  --name allow-signing-server \
  --priority 100 \
  --destination-address-prefixes <signing-server-ip> \
  --destination-port-ranges 443 \
  --access Allow \
  --protocol Tcp
```

**Update AWS security group to allow Azure VM IPs:**
```bash
# Get Azure VM public IP
az vm show -d --resource-group github-runners --name github-runner-01 \
  --query publicIps -o tsv

# Add to AWS ALB security group
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp \
  --port 443 \
  --cidr <azure-vm-ip>/32
```

### 3. Self-Hosted Runner Workflow

```yaml
name: Build on Azure Runner

on:
  push:
    branches: [main]

jobs:
  build-and-sign:
    runs-on: [self-hosted, linux, azure]

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install gpgshim (if not cached)
        run: |
          if [ ! -f ~/.local/bin/gpgshim ]; then
            mkdir -p ~/.local/bin
            curl -sSL https://signing.company.com/client/gpgshim \
              -o ~/.local/bin/gpgshim
            chmod +x ~/.local/bin/gpgshim
          fi

      - name: Build and sign
        env:
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
          GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
        run: |
          export PATH="$HOME/.local/bin:$PATH"
          make build-rpm
          make sign-rpm
```

## Token Management

### Generating Tokens for GitHub Actions

```bash
# On signing server or local machine with access to secrets.json

# Production token (30 days)
./generate-token.py --generate \
    --client-id github-actions-prod \
    --role production \
    --secret-file config/secrets.json \
    --expires-hours 720

# Development token (30 days)
./generate-token.py --generate \
    --client-id github-actions-dev \
    --role development \
    --secret-file config/secrets.json \
    --expires-hours 720

# Copy output tokens to GitHub Secrets
```

### Token Rotation in GitHub Actions

When rotating tokens, update GitHub Secrets:

1. **Generate new token** on signing server
2. **Update GitHub Secret**:
   - Go to repository Settings → Secrets → Actions
   - Click "GPG_SERVER_TOKEN_PROD"
   - Click "Update secret"
   - Paste new token value
3. **Test with workflow run**
4. **Old token expires automatically** (based on exp field)

### Automated Token Rotation

Use a scheduled workflow to check token expiration:

```yaml
name: Check Token Expiration

on:
  schedule:
    - cron: '0 9 * * 1'  # Every Monday at 9 AM
  workflow_dispatch:

jobs:
  check-token:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Download token utility
        run: |
          curl -sSL https://signing.company.com/client/generate-token.py \
            -o generate-token.py
          chmod +x generate-token.py

      - name: Check token expiration
        env:
          TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
        run: |
          # Decode token (doesn't require secret)
          python3 generate-token.py decode --token "$TOKEN" > token-info.json

          # Extract expiration timestamp
          exp=$(jq -r '.payload.exp' token-info.json)
          now=$(date +%s)
          days_remaining=$(( (exp - now) / 86400 ))

          echo "Token expires in $days_remaining days"

          if [ $days_remaining -lt 7 ]; then
            echo "::warning::Token expires in $days_remaining days - rotation needed"
            exit 1
          fi
```

## Troubleshooting

### Common Issues

#### 1. Authentication Failed (401)

**Symptoms:**
```
Error: Authentication failed: 401 Unauthorized
```

**Causes:**
- Expired token
- Invalid token (corrupted, wrong secret)
- Missing Authorization header

**Solutions:**
```bash
# Check token expiration
echo "$GPG_SERVER_TOKEN" | base64 -d | jq .

# Generate new token
./generate-token.py --generate \
    --client-id github-actions-prod \
    --role production \
    --secret-file config/secrets.json

# Update GitHub Secret with new token
```

#### 2. Authorization Failed (403)

**Symptoms:**
```
Error: Forbidden: Not authorized for this key
```

**Causes:**
- Token has valid role, but role doesn't allow requested key
- Wrong role for the signing key

**Solutions:**
```bash
# Check which role the token has
./generate-token.py decode --token "$TOKEN"

# Verify authorization config allows this key for this role
cat config/authorization.json | jq '.roles["production"].allowed_keys'

# Generate token with correct role or add key to role
```

#### 3. Rate Limited (429)

**Symptoms:**
```
Error: Rate limit exceeded: 429 Too Many Requests
```

**Causes:**
- Exceeded max_requests_per_hour for role
- Multiple parallel builds using same token

**Solutions:**
```bash
# Check rate limit for role
cat config/authorization.json | jq '.roles["production"].max_requests_per_hour'

# Wait for rolling window to reset (up to 1 hour)
# Or increase rate limit in authorization.json

# For parallel builds, use separate tokens with separate rate limits
```

#### 4. Connection Timeout

**Symptoms:**
```
Error: Connection timed out after 30 seconds
```

**Causes:**
- Network connectivity issue
- Signing server down
- Firewall blocking traffic

**Solutions:**
```bash
# Test connectivity from runner
curl -v https://signing.company.com/health

# Check AWS ALB target health
aws elbv2 describe-target-health \
    --target-group-arn arn:aws:elasticloadbalancing:...

# Verify security group allows GitHub Actions IPs
# GitHub Actions IP ranges: https://api.github.com/meta
curl https://api.github.com/meta | jq -r '.actions[]'
```

#### 5. SSL Certificate Verification Failed

**Symptoms:**
```
Error: SSL certificate verification failed
```

**Causes:**
- Self-signed certificate
- Corporate CA not in system trust store
- Certificate expired

**Solutions:**

**Option 1: Add CA cert to GitHub Secret**
```yaml
steps:
  - name: Setup CA cert
    run: |
      echo "${{ secrets.GPG_SERVER_CA_CERT }}" > /tmp/ca.crt

  - name: Sign with custom CA
    env:
      GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
      GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
      GPG_SERVER_CA_CERT: /tmp/ca.crt
    run: |
      rpmsign --addsign --define "_gpg_path $HOME/.local/bin/gpgshim" dist/*.rpm
```

**Option 2: Disable SSL verification (NOT RECOMMENDED for production)**
```yaml
env:
  GPG_VERIFY_SSL: "false"
```

#### 6. rpmsign Doesn't Use gpgshim

**Symptoms:**
```
Error: gpg: signing failed: No secret key
```

**Causes:**
- rpmsign using system gpg instead of gpgshim
- _gpg_path not set correctly

**Solutions:**
```bash
# Verify gpgshim is in PATH
which gpgshim

# Use absolute path
rpmsign --addsign --define "_gpg_path $(which gpgshim)" mypackage.rpm

# Or set PATH before rpmsign
export PATH="$HOME/.local/bin:$PATH"
rpmsign --addsign --define "_gpg_path gpgshim" mypackage.rpm
```

### Debug Mode

Enable verbose logging in workflow:

```yaml
steps:
  - name: Sign RPM (debug)
    env:
      GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
      GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
      GPGSHIM_DEBUG: "1"
    run: |
      set -x  # Enable bash debug mode
      rpmsign --addsign --define "_gpg_path $HOME/.local/bin/gpgshim" \
        --verbose \
        dist/*.rpm
```

Check server audit logs:
```bash
# SSH to signing server
ssh ec2-user@signing-server

# Check recent audit entries
sudo tail -100 /var/log/gpg-signing/audit.log | jq .

# Filter by client_id
sudo cat /var/log/gpg-signing/audit.log | \
  jq 'select(.client_id == "github-actions-prod")'
```

## Security Best Practices

### 1. Least Privilege Tokens

Use separate tokens with minimal permissions:

```yaml
# Don't use production token for PR builds
- name: Sign (production)
  if: github.ref == 'refs/heads/main'
  env:
    GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
  run: make sign

# Use development token for PRs
- name: Sign (development)
  if: github.event_name == 'pull_request'
  env:
    GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_DEV }}
  run: make sign
```

### 2. Short-Lived Tokens for Manual Workflows

For workflow_dispatch (manual triggers), use short-lived tokens:

```yaml
on:
  workflow_dispatch:
    inputs:
      admin_token:
        description: 'Admin signing token (4 hour expiry)'
        required: true

jobs:
  manual-sign:
    runs-on: ubuntu-latest
    steps:
      - name: Sign with admin token
        env:
          GPG_SERVER_TOKEN: ${{ github.event.inputs.admin_token }}
        run: make sign
```

### 3. Don't Log Tokens

Never log tokens in workflow output:

```yaml
# BAD - logs token
- run: echo "Using token: $GPG_SERVER_TOKEN"

# GOOD - mask token
- run: echo "::add-mask::$GPG_SERVER_TOKEN"
  env:
    GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
```

GitHub automatically masks secrets, but be careful with derived values.

### 4. Restrict Secret Access

Use environment-specific secrets with protection rules:

```yaml
jobs:
  sign-production:
    runs-on: ubuntu-latest
    environment: production  # Requires approval
    steps:
      - name: Sign
        env:
          GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
        run: make sign
```

Configure environment protection:
- Settings → Environments → production
- Add required reviewers
- Add deployment branches rule (only main)

## Performance Optimization

### Concurrent Signing

Sign multiple RPMs in parallel:

```yaml
- name: Sign all RPMs concurrently
  env:
    GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
    GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
  run: |
    export PATH="$HOME/.local/bin:$PATH"

    # Sign up to 10 RPMs concurrently
    find dist/ -name "*.rpm" | \
      xargs -P 10 -I {} rpmsign --addsign --define "_gpg_path gpgshim" {}
```

**Before:** 100 RPMs × 50ms each = 5 seconds
**After:** 100 RPMs ÷ 10 parallel = 0.5 seconds

### Caching Strategies

Cache signing server response (if digests are deterministic):

```yaml
- name: Cache signed RPMs
  uses: actions/cache@v4
  with:
    path: dist/*.rpm
    key: signed-rpms-${{ hashFiles('rpmbuild/RPMS/**/*.rpm') }}

- name: Sign only if cache miss
  if: steps.cache.outputs.cache-hit != 'true'
  run: make sign
```

## Monitoring and Alerts

### Workflow Success Rate

Track signing failures:

```yaml
- name: Sign RPMs
  id: sign
  continue-on-error: true
  run: make sign

- name: Report failure
  if: steps.sign.outcome == 'failure'
  run: |
    curl -X POST https://monitoring.company.com/alert \
      -d "workflow=build-and-sign" \
      -d "status=failed" \
      -d "run_id=${{ github.run_id }}"
```

### Token Expiration Monitoring

Alert when token is close to expiration (see "Automated Token Rotation" above).

## Example: Complete Production Workflow

```yaml
name: Production RPM Build and Sign

on:
  push:
    tags: ['v*']

jobs:
  build-and-sign:
    runs-on: ubuntu-latest
    environment: production

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set version from tag
        id: version
        run: echo "version=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT

      - name: Install build tools
        run: |
          sudo apt-get update
          sudo apt-get install -y rpm rpmbuild rpmdevtools

      - name: Build RPM
        run: |
          make build-rpm VERSION=${{ steps.version.outputs.version }}

      - name: Setup gpgshim
        run: |
          mkdir -p ~/.local/bin
          curl -sSL https://signing.company.com/client/gpgshim \
            -o ~/.local/bin/gpgshim
          chmod +x ~/.local/bin/gpgshim

      - name: Sign RPM
        env:
          GPG_SIGNING_SERVER: ${{ secrets.GPG_SIGNING_SERVER }}
          GPG_SERVER_TOKEN: ${{ secrets.GPG_SERVER_TOKEN_PROD }}
          PATH: /home/runner/.local/bin:${{ env.PATH }}
        run: |
          rpmsign --addsign --define "_gpg_path gpgshim" dist/*.rpm

      - name: Verify signature
        run: |
          rpm --checksig dist/*.rpm
          rpm -qip dist/*.rpm | grep Signature

      - name: Upload to artifact registry
        run: |
          curl -u "${{ secrets.ARTIFACTORY_USER }}:${{ secrets.ARTIFACTORY_TOKEN }}" \
            -T "dist/mypackage-${{ steps.version.outputs.version }}.rpm" \
            "https://artifactory.company.com/rpms/production/"

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*.rpm
          generate_release_notes: true
```

This workflow:
- Only runs on version tags (v1.0.0, v2.1.3, etc.)
- Requires production environment approval
- Builds RPM with version from tag
- Signs with production key
- Verifies signature
- Uploads to artifact registry
- Creates GitHub release with signed RPM
