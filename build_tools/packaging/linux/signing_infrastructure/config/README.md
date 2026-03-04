# Configuration Files

This directory contains configuration templates and examples for the GPG signing server authentication system.

## Files

### `authorization.json` (Production File - DO COMMIT)

Defines roles and their permissions. This file should be committed to version control.

**Purpose:** Maps roles to allowed signing keys and digest algorithms.

**Structure:**
```json
{
  "roles": {
    "role-name": {
      "allowed_keys": ["key-id-1", "key-id-2"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 1000,
      "description": "Human-readable description"
    }
  },
  "audit": {
    "enabled": true,
    "log_file": "/var/log/gpg-signing/audit.log",
    "log_level": "INFO",
    "retention_days": 365
  }
}
```

**Key Fields:**
- `allowed_keys`: List of GPG key IDs or email addresses allowed for this role
  - Empty list `[]` means "allow any key" (use with caution!)
  - Can use hex key IDs: `"ABCD1234ABCD1234"`
  - Can use email addresses: `"signer@company.com"`
- `allowed_digest_algos`: Permitted hash algorithms (SHA256, SHA384, SHA512, SHA1, etc.)
  - Empty list `[]` means "allow any algorithm" (not recommended)
- `max_requests_per_hour`: Rate limit (rolling 1-hour window)
  - Set to `0` to disable rate limiting for this role
- `description`: Human-readable note (for documentation)

**Setup:**
```bash
# Copy example and customize
cp authorization.json.example authorization.json

# Edit to add your actual key IDs and roles
vim authorization.json

# Validate JSON syntax
python3 -m json.tool authorization.json > /dev/null && echo "Valid JSON"
```

### `secrets.json` (Confidential - DO NOT COMMIT)

Contains client credentials for authentication. **Never commit this file to version control.**

**Purpose:** Maps client IDs to their shared secrets used for JWT token signing.

**Structure:**
```json
{
  "client-id": {
    "secret": "base64-encoded-256-bit-secret",
    "description": "Human-readable description",
    "created": "YYYY-MM-DD",
    "expires": "YYYY-MM-DD"
  }
}
```

**Setup:**
```bash
# Copy example as starting point
cp secrets.json.example secrets.json

# Generate secrets for each client
./generate-token.py --generate-secret --client-id github-actions-prod

# Add the output to secrets.json

# Set restrictive permissions
chmod 600 secrets.json

# Verify it's ignored by git
git status | grep -q secrets.json && echo "WARNING: File not ignored!" || echo "OK: Ignored"
```

**Security Notes:**
- File should have `0600` permissions (readable only by owner)
- Store production secrets in a secrets manager (Vault, AWS Secrets Manager, etc.)
- Rotate secrets quarterly or when compromised
- Use different secrets for each environment (prod/dev/test)

### `*.example` Files (Templates - COMMIT THESE)

Example configuration files showing the expected format. These are safe to commit.

## Role Definitions

### Production Role
- **Purpose:** CI/CD pipelines signing production releases
- **Keys:** Production signing key only
- **Rate Limit:** 1000 req/hour (typical CI/CD load)
- **Token Lifetime:** 7-30 days (long-lived for automation)

### Development Role
- **Purpose:** Development and test builds
- **Keys:** Development signing key only
- **Rate Limit:** 5000 req/hour (higher for frequent builds)
- **Token Lifetime:** 7-30 days

### Admin Role
- **Purpose:** Manual signing by administrators
- **Keys:** Both production and development keys
- **Rate Limit:** 100 req/hour (manual use doesn't need high limits)
- **Token Lifetime:** 4-24 hours (short-lived for security)

### Emergency Role
- **Purpose:** Break-glass emergency access
- **Keys:** All keys (empty `allowed_keys` list)
- **Rate Limit:** 50 req/hour (restricted)
- **Token Lifetime:** 1-4 hours (very short)
- **Use Case:** Disaster recovery, critical hotfixes

## Workflow

### Initial Setup

1. **Create authorization config:**
   ```bash
   cp authorization.json.example authorization.json
   # Edit to add your actual GPG key IDs
   ```

2. **Create secrets file:**
   ```bash
   cp secrets.json.example secrets.json
   chmod 600 secrets.json
   ```

3. **Generate client secrets:**
   ```bash
   # For each client in secrets.json:
   ./generate-token.py --generate-secret --client-id github-actions-prod
   # Copy output into secrets.json
   ```

4. **Commit authorization, ignore secrets:**
   ```bash
   git add authorization.json
   git add *.example
   echo "config/secrets.json" >> .gitignore  # Already there
   git commit -m "Add authorization config"
   ```

### Adding a New Client

1. **Generate secret:**
   ```bash
   ./generate-token.py --generate-secret --client-id new-client-id
   ```

2. **Add to secrets.json:**
   ```json
   {
     "new-client-id": {
       "secret": "output-from-previous-command",
       "description": "Description of this client",
       "created": "2026-03-02",
       "expires": "2027-03-02"
     }
   }
   ```

3. **Generate token for client:**
   ```bash
   ./generate-token.py --generate \
       --client-id new-client-id \
       --role production \
       --secret-file config/secrets.json \
       --expires-hours 720  # 30 days
   ```

4. **Store token securely:**
   - GitHub Secrets for CI/CD
   - Secrets manager for production
   - Environment variable for manual use

### Adding a New Role

1. **Edit authorization.json:**
   ```json
   {
     "roles": {
       "new-role": {
         "allowed_keys": ["key-id"],
         "allowed_digest_algos": ["SHA256"],
         "max_requests_per_hour": 100,
         "description": "Description of role"
       }
     }
   }
   ```

2. **Restart signing server** (to reload config)

3. **Generate tokens with new role:**
   ```bash
   ./generate-token.py --generate \
       --client-id client-id \
       --role new-role \
       --secret-file config/secrets.json
   ```

## Security Best Practices

### File Permissions
```bash
# Authorization file: readable by all (contains no secrets)
chmod 644 authorization.json

# Secrets file: readable only by signing server process owner
chmod 600 secrets.json
chown signing-server:signing-server secrets.json
```

### Secrets Rotation

Rotate secrets quarterly or when:
- Employee leaves who had access
- Secret may have been compromised
- Compliance requires rotation

**Rotation process:**
1. Generate new secret for client
2. Update `secrets.json` with new secret
3. Generate new tokens using new secret
4. Update GitHub Secrets / environment
5. Old tokens automatically expire (don't need manual revocation)

### Monitoring

Check audit logs regularly:
```bash
# Count signatures by role today
grep $(date +%Y-%m-%d) /var/log/gpg-signing/audit.log | \
  jq -r .role | sort | uniq -c

# Find denied requests
grep '"action": "DENIED"' /var/log/gpg-signing/audit.log | tail -20

# Check rate limiting
grep '"action": "RATE_LIMITED"' /var/log/gpg-signing/audit.log
```

### Backup

Back up secrets file securely:
```bash
# Encrypt before backup
gpg --encrypt --recipient admin@company.com secrets.json
# Store secrets.json.gpg in secure vault

# Restore when needed
gpg --decrypt secrets.json.gpg > secrets.json
chmod 600 secrets.json
```

## Troubleshooting

### "Unknown role" error

**Cause:** Client token has a role not defined in `authorization.json`.

**Fix:** Add the role to `authorization.json` or regenerate token with correct role.

### "Not authorized for key" error

**Cause:** Role's `allowed_keys` list doesn't include the requested key.

**Fix:** Add the key ID to the role's `allowed_keys` list, or use a different role.

### Server doesn't see updated config

**Cause:** Server caches config in memory.

**Fix:** Restart the signing server:
```bash
# Find server process
ps aux | grep signing-server

# Gracefully shutdown via /quit endpoint
curl -X POST http://localhost:8080/quit

# Or kill and restart
kill <pid>
./signing-server.py --enable-auth --secrets-file config/secrets.json \
    --authz-config config/authorization.json --keyring /path/to/.gnupg
```

## Examples

See `authorization.json.example` and `secrets.json.example` for complete examples.

For more details on token management, see:
- `../docs/manual-signing.md` - Manual signing workflow
- `./generate-token.py --help` - Token utility usage
