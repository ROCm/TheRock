# Manual Signing with the Remote Signing Server

This guide explains how administrators should manually sign RPMs using the remote signing infrastructure.

## Quick Answer

**Q: Should I copy the production key to my machine for manual signing?**

**A: No. Use the signing service with a personal admin token instead.**

## Why Not Copy the Production Key?

The remote signing architecture keeps the production signing key **centralized and secured** on the signing server.

### Security Benefits of Using the Service

- ✅ **Centralized key storage** - Key lives only on signing server
- ✅ **Audit trail** - Operations logged with your client_id  
- ✅ **Revocable access** - Tokens can be expired/revoked
- ✅ **Time-limited** - Tokens expire automatically
- ✅ **No key exposure** - Key can't be copied or stolen
- ✅ **Compliance ready** - All signatures tracked
- ✅ **Future-proof** - Works with HSM/KMS migration

### Risks of Copying the Key

- ❌ **No audit trail** - Can't determine who signed what
- ❌ **Can't revoke** - Key is out of control once copied
- ❌ **Key sprawl** - Multiple copies = multiple attack vectors  
- ❌ **Defeats architecture** - Undermines the security model
- ❌ **Compliance issues** - No visibility into key usage

## Recommended: Admin Token Workflow

### 1. Create Your Admin Client

```bash
./generate-token.py --generate-secret --client-id admin-yourname
```

Add to `config/secrets.json`:
```json
{
  "admin-yourname": {
    "secret": "generated-secret-here",
    "description": "Admin token for manual signing",
    "created": "2026-03-02"
  }
}
```

### 2. Add Admin Role

Edit `config/authorization.json`:
```json
{
  "roles": {
    "admin": {
      "allowed_keys": ["prod-signing-key@company.com"],
      "allowed_digest_algos": ["SHA256", "SHA512"],
      "max_requests_per_hour": 100
    }
  }
}
```

### 3. Generate Your Token

```bash
./generate-token.py --generate \
    --client-id admin-yourname \
    --role admin \
    --secret-file config/secrets.json \
    --expires-hours 24
```

### 4. Sign RPMs

```bash
# Setup
export GPG_SIGNING_SERVER='https://signing-server:8443/sign'
export GPG_SERVER_TOKEN='your-token-here'
export PATH="$HOME/.local/bin:$PATH"

# Sign
rpmsign --addsign mypackage.rpm
```

### 5. Verify Audit Trail

```bash
grep "admin-yourname" /var/log/gpg-signing/audit.log
```

## Token Management Best Practices

### Token Lifetime
- Manual signing: 4-24 hours
- CI/CD: 7-30 days  
- Emergency: 1-4 hours

### Secure Storage
- ✅ Environment variables
- ✅ Secrets managers (Vault, AWS Secrets Manager)
- ❌ Hardcoded in scripts
- ❌ Plain text files
- ❌ Shared in Slack/email

## Emergency: When You Might Need the Key

Very limited scenarios:

1. **Disaster Recovery** - Encrypted backup in vault
2. **Initial Setup** - Importing key to new server
3. **Server Unavailable** - Emergency only, requires approval

If you must use local key:
```bash
export GNUPGHOME=/tmp/emergency-gpg
mkdir -p $GNUPGHOME && chmod 700 $GNUPGHOME
gpg --import /secure/vault/key.gpg
rpmsign --addsign mypackage.rpm
rm -rf $GNUPGHOME  # IMMEDIATELY clean up
```

## Troubleshooting

### "Authentication failed"
```bash
# Validate token
./generate-token.py --validate "$GPG_SERVER_TOKEN" --secret-file config/secrets.json

# Generate new one
./generate-token.py --generate --client-id admin-yourname --role admin \
    --secret-file config/secrets.json --expires-hours 24
```

### "Authorization failed"  
Check that admin role includes your key in `authorization.json`.

### "Rate limit exceeded"
Wait 1 hour or ask admin to increase limit.

## Summary

**For manual RPM signing:**

✅ **DO:**
- Get a personal admin token
- Use gpgshim with the signing service
- Keep tokens short-lived (4-24 hours)
- Check audit logs

❌ **DON'T:**
- Copy the production key to your workstation
- Share tokens between users
- Commit tokens to git
- Use CI/CD tokens for manual work

The signing service provides centralized security, audit trails, and revocable access. Use it!
