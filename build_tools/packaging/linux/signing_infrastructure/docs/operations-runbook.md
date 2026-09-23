# Signing Server — Operations Runbook

**Component:** Remote GPG Signing Service
**Related docs:** `signing-server-design.md`, `use-case-flows.md`

This runbook covers the full lifecycle of the signing server: initial provisioning, day-to-day operations, key rotation, and emergency procedures. All commands are AWS CLI unless noted otherwise.

______________________________________________________________________

## Prerequisites

Before starting, confirm the following exist:

| Prerequisite                                                    | How to verify                                                    |
| --------------------------------------------------------------- | ---------------------------------------------------------------- |
| AWS account and region decided                                  | `aws sts get-caller-identity`                                    |
| VPC with a private subnet (no internet gateway)                 | AWS Console → VPC                                                |
| EC2 instance launched in the private subnet                     | Running state in EC2 console                                     |
| Security group `sg-signing-server` created                      | Inbound: TCP 443 from build runner SG and operator VPN CIDR only |
| GPG key pair generated offline                                  | `gpg --list-secret-keys` on the air-gapped machine               |
| Operator has AWS CLI configured with sufficient IAM permissions | `aws iam get-user`                                               |

Set your region once for all commands in this session:

```bash
export AWS_REGION=us-east-1   # replace with your region
export AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "Account: $AWS_ACCOUNT  Region: $AWS_REGION"
```

______________________________________________________________________

## Section 1 — Initial Provisioning (One-Time Setup)

Run these steps in order. Each step depends on the previous one.

### 1.1 Create the Signing Server IAM Role

**Must be done first** — the KMS key policy references this role ARN.

```bash
# Create the role
aws iam create-role \
  --role-name role-signing-server \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach permissions — Secrets Manager, KMS, CloudWatch Logs
aws iam put-role-policy \
  --role-name role-signing-server \
  --policy-name signing-server-policy \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Sid": "SecretsManagerReadGPGKeys",
        "Effect": "Allow",
        "Action": "secretsmanager:GetSecretValue",
        "Resource": "arn:aws:secretsmanager:*:*:secret:signing/gpg/*"
      },
      {
        "Sid": "KMSDecryptForSecretsManager",
        "Effect": "Allow",
        "Action": ["kms:Decrypt", "kms:GenerateDataKey"],
        "Resource": "*"
      },
      {
        "Sid": "CloudWatchLogs",
        "Effect": "Allow",
        "Action": ["logs:PutLogEvents", "logs:CreateLogStream", "logs:CreateLogGroup"],
        "Resource": "*"
      }
    ]
  }'

# Create instance profile (EC2 wrapper for the role)
aws iam create-instance-profile \
  --instance-profile-name role-signing-server

aws iam add-role-to-instance-profile \
  --instance-profile-name role-signing-server \
  --role-name role-signing-server

# Confirm role ARN
aws iam get-role --role-name role-signing-server \
  --query Role.Arn --output text
# Expected: arn:aws:iam::123456789:role/role-signing-server
```

______________________________________________________________________

### 1.2 Create the KMS Customer Managed Key (CMK)

**Must be done after role creation** — key policy references the role ARN.

```bash
# Get the role ARN for use in key policy
SIGNING_ROLE_ARN=$(aws iam get-role \
  --role-name role-signing-server \
  --query Role.Arn --output text)

PROVISIONER_ARN=$(aws sts get-caller-identity --query Arn --output text)
# Note: if this is an assumed-role ARN (arn:aws:sts::...:assumed-role/...)
# convert it to IAM role ARN: arn:aws:iam::<account>:role/<role-name>

# Create the CMK
KEY_ID=$(aws kms create-key \
  --description "AMD signing server GPG key protection" \
  --key-usage ENCRYPT_DECRYPT \
  --key-spec SYMMETRIC_DEFAULT \
  --query KeyMetadata.KeyId \
  --output text)

echo "CMK Key ID: $KEY_ID"

# Create alias
aws kms create-alias \
  --alias-name alias/amd-signing-gpg-key \
  --target-key-id $KEY_ID

# Set key policy
aws kms put-key-policy \
  --key-id $KEY_ID \
  --policy-name default \
  --policy "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Sid\": \"AllowKeyAdministration\",
        \"Effect\": \"Allow\",
        \"Principal\": {\"AWS\": \"arn:aws:iam::${AWS_ACCOUNT}:root\"},
        \"Action\": \"kms:*\",
        \"Resource\": \"*\"
      },
      {
        \"Sid\": \"AllowProvisionerToStore\",
        \"Effect\": \"Allow\",
        \"Principal\": {\"AWS\": \"${PROVISIONER_ARN}\"},
        \"Action\": [\"kms:GenerateDataKey\"],
        \"Resource\": \"*\"
      },
      {
        \"Sid\": \"AllowSigningServerToDecrypt\",
        \"Effect\": \"Allow\",
        \"Principal\": {\"AWS\": \"${SIGNING_ROLE_ARN}\"},
        \"Action\": [\"kms:Decrypt\", \"kms:GenerateDataKey\"],
        \"Resource\": \"*\"
      }
    ]
  }"

# Verify
aws kms describe-key --key-id alias/amd-signing-gpg-key \
  --query KeyMetadata.KeyState --output text
# Expected: Enabled
```

______________________________________________________________________

### 1.3 Store GPG Private Keys in Secrets Manager

**Must be done after KMS CMK creation** — Secrets Manager uses the CMK to encrypt.

Run these commands from the machine where the GPG key was generated (or copy the exported key file securely):

```bash
# Export the private key to a temp file
gpg --export-secret-keys --armor <key-email-or-id> > /tmp/private.asc

# Verify it looks correct
head -1 /tmp/private.asc
# Expected: -----BEGIN PGP PRIVATE KEY BLOCK-----

# Store in Secrets Manager — SM handles KMS encryption internally
aws secretsmanager create-secret \
  --name signing/gpg/therock-release \
  --description "GPG private key for therock-release package signing" \
  --kms-key-id alias/amd-signing-gpg-key \
  --secret-string file:///tmp/private.asc \
  --region $AWS_REGION

# IMPORTANT: shred the local plaintext copy
shred -vzu /tmp/private.asc
echo "Plaintext key deleted"

# Verify stored correctly — should start with -----BEGIN PGP
aws secretsmanager get-secret-value \
  --secret-id signing/gpg/therock-release \
  --query SecretString \
  --output text | head -1
# Expected: -----BEGIN PGP PRIVATE KEY BLOCK-----
```

Repeat for each signing tier (`therock-dev`, `therock-nightly`, `therock-release`).

______________________________________________________________________

### 1.4 Attach IAM Role to EC2 Instance

**Must be done before starting the signing server** — server needs IAM credentials to call Secrets Manager.

```bash
# Get your instance ID (run on the EC2 instance using IMDSv2)
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id)
echo $INSTANCE_ID

# Or from your workstation — list running instances
aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query "Reservations[].Instances[].[InstanceId,PrivateIpAddress]" \
  --output table

# Attach the role (run from workstation)
aws ec2 associate-iam-instance-profile \
  --instance-id <instance-id> \
  --iam-instance-profile Name=role-signing-server

# Verify on the EC2 instance (wait ~10 seconds after attaching)
aws sts get-caller-identity
# Expected: Arn contains "assumed-role/role-signing-server"
```

______________________________________________________________________

### 1.5 Clone Repo and Run Setup Script

**Final step — for provisioning a brand-new instance only.** Run on the EC2 instance. All AWS prerequisites must be in place.

> **This `git clone` is a one-time bootstrap action for a fresh instance, not an update mechanism.** Never `git pull` on an already-running production signing server to pick up code or config changes — see §8 "Instance Replacement Procedure" below for how updates are actually deployed. Treating a live instance's checkout as something you periodically re-pull creates exactly the "patched in place, no clean rollback, no reproducible build" problem §8 exists to avoid.

```bash
# Install git
sudo apt-get install -y git   # Ubuntu
# sudo yum install -y git     # Amazon Linux

# Clone a specific, known-good commit or tag — never a floating branch.
# A branch name (including "main") can move between when you test an AMI/
# instance and when you actually launch from it, so two instances built
# "from the same branch" at different times are not guaranteed to run the
# same code. Pin to a tag or commit SHA instead:
sudo git clone https://github.com/ROCm/TheRock.git /opt/therock-signing
cd /opt/therock-signing
sudo git checkout <release-tag-or-commit-sha>   # e.g. v1.4.2, or a specific SHA

# Run setup script
sudo bash build_tools/packaging/linux/signing_infrastructure/tools/setup-server.sh \
  --secret signing/gpg/therock-release \
  --region $AWS_REGION
```

The script:

1. Installs `gnupg2`, `python3`, `python3-venv`
1. Creates `/opt/signing-server-venv` with `boto3` and `PyJWT`
1. Mounts `tmpfs` at `/var/gpg-keyring` and adds to `/etc/fstab`
1. Copies `signing-server.py` and `auth.py` to `/opt/signing-server/`
1. Generates a self-signed TLS certificate in `/opt/signing-server/certs/`
1. Installs and enables the `signing-server` systemd service

______________________________________________________________________

### 1.6 Start and Verify

```bash
# Start the server
sudo systemctl start signing-server

# Watch startup logs
sudo journalctl -u signing-server -f
```

Expected startup output:

```
Fetching GPG key from Secrets Manager: signing/gpg/therock-release
  Imported key from 'signing/gpg/therock-release'
Keyring verified: 1 secret key(s) available
GPG Signing Server
Listening: https://0.0.0.0:443
Auth: DISABLED (Phase 1 — VPC Security Groups)
TLS: ENABLED
```

```bash
# Verify health endpoint
curl -k https://localhost/health
# Expected: {"status": "ok"}

# Test a signing request
echo "test" > /tmp/test.txt
python3 /opt/therock-signing/build_tools/packaging/linux/signing_infrastructure/tools/sign-file \
  --server https://localhost \
  --key-id <your-key-email> \
  --file /tmp/test.txt \
  --armor \
  --no-verify-ssl

cat /tmp/test.txt.asc
# Expected: -----BEGIN PGP SIGNATURE-----
```

______________________________________________________________________

## Section 2 — Day-to-Day Operations

### 2.1 Check Server Status

```bash
# Service status
sudo systemctl status signing-server

# Live log stream
sudo journalctl -u signing-server -f

# Health check
curl -k https://localhost/health

# Confirm key is in keyring
sudo GNUPGHOME=/var/gpg-keyring gpg --list-secret-keys
```

### 2.2 Restart the Server

```bash
sudo systemctl restart signing-server
# Server will re-fetch GPG key from Secrets Manager on startup
```

### 2.3 View Audit Logs

Audit logs are written to stdout → captured by systemd journal → forwarded to CloudWatch:

```bash
# Local: view recent signing requests
sudo journalctl -u signing-server --since "1 hour ago" \
  | grep '"action"' | python3 -m json.tool

# CloudWatch (from workstation)
aws logs filter-log-events \
  --log-group-name /amd/signing-server/audit \
  --filter-pattern '{ $.action = "SIGNED" }' \
  --start-time $(date -d '1 hour ago' +%s000)
```

### 2.4 Check Rate Limit Status

Rate limit hits appear in the audit log as `RATE_LIMITED`:

```bash
sudo journalctl -u signing-server --since today | grep RATE_LIMITED
```

______________________________________________________________________

## Section 3 — GPG Key Rotation

Perform key rotation when a key expires, is compromised, or on a scheduled rotation policy. This does NOT require server downtime in Phase 2 (scheduled sync picks it up automatically).

```bash
# Step 1: Generate new GPG key pair on an isolated machine
gpg --batch --gen-key <<EOF
Key-Type: RSA
Key-Length: 4096
Name-Real: TheRock Release
Name-Email: therock-release-new@amd.com
Expire-Date: 2y
%no-protection
%commit
EOF

# Step 2: Export new private key
gpg --export-secret-keys --armor therock-release-new@amd.com > /tmp/new-private.asc

# Step 3: Update Secrets Manager secret
aws secretsmanager put-secret-value \
  --secret-id signing/gpg/therock-release \
  --secret-string file:///tmp/new-private.asc \
  --region $AWS_REGION

# Step 4: Shred local copy
shred -vzu /tmp/new-private.asc

# Step 5: Export and distribute NEW public key
gpg --export --armor therock-release-new@amd.com > /tmp/new-public.asc
# Upload to keyserver / distribute to RPM and APT client configs

# Step 6a: Phase 1 — restart server to pick up new key
sudo systemctl restart signing-server

# Step 6b: Phase 2 — wait for scheduled sync (every 6 hours)
# No restart needed
```

______________________________________________________________________

## Section 4 — Emergency Procedures

### 4.1 Emergency Key Revocation

Use when a GPG private key is suspected compromised:

```bash
# Step 1: Disable CMK immediately — all future SM fetches fail
aws kms disable-key --key-id alias/amd-signing-gpg-key

# Step 2: Restart server — startup fails, signing server goes down
sudo systemctl restart signing-server
# Server will fail to fetch key → exit → systemd keeps restarting
# Signing is now effectively stopped

# Step 3: Revoke old public key and generate new key pair
# (follow Section 3 for key generation)

# Step 4: Re-enable CMK (or create new CMK)
aws kms enable-key --key-id alias/amd-signing-gpg-key

# Step 5: Store new key in Secrets Manager (Section 1.3)

# Step 6: Restart server — picks up new key
sudo systemctl restart signing-server
```

### 4.2 Break-Glass Shell Access

For genuine emergencies where direct server access is required. Normally SSH and SSM are disabled.

```bash
# Step 1: Get approval (4-eyes sign-off, create incident ticket)

# Step 2: Temporarily enable SSM Session Manager
# (Modify role-signing-server IAM policy to allow ssm:StartSession)
aws iam put-role-policy \
  --role-name role-signing-server \
  --policy-name break-glass-ssm \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["ssm:StartSession"],
      "Resource": "*"
    }]
  }'

# Step 3: Start session (fully recorded in CloudWatch)
aws ssm start-session --target <instance-id>

# Step 4: Conduct investigation

# Step 5: IMMEDIATELY revoke break-glass access after session
aws iam delete-role-policy \
  --role-name role-signing-server \
  --policy-name break-glass-ssm

# Step 6: Update incident ticket with findings
```

### 4.3 Server Instance Replacement

When replacing a failed or outdated instance:

```bash
# Step 1: Launch new EC2 instance (same subnet, same security group)
# Step 2: Attach role-signing-server instance profile
# Step 3: Clone repo and run setup-server.sh (Section 1.5)
# Step 4: Verify health: curl -k https://new-instance-ip/health
# Step 5: Update GPG_SIGNING_SERVER secret/env var to new instance IP
# Step 6: Terminate old instance
aws ec2 terminate-instances --instance-ids <old-instance-id>
```

______________________________________________________________________

## Section 5 — Troubleshooting

| Error                                              | Cause                                          | Fix                                                                                                                                      |
| -------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `Unable to locate credentials`                     | No IAM role attached to EC2 instance           | Attach `role-signing-server` instance profile (Section 1.4)                                                                              |
| `Failed to fetch secret ... AccessDeniedException` | IAM role lacks `secretsmanager:GetSecretValue` | Check role policy has the Secrets Manager statement                                                                                      |
| `KMS AccessDeniedException`                        | Role not in CMK key policy                     | Update KMS key policy to include role ARN (Section 1.2)                                                                                  |
| `/health` returns `503`                            | Key not loaded into keyring                    | Check `journalctl -u signing-server` for import error                                                                                    |
| `gpg --import failed`                              | Secret content is not a PGP private key        | Verify with: `aws secretsmanager get-secret-value --secret-id signing/gpg/therock-release --query SecretString --output text \| head -1` |
| `Address already in use`                           | Port 443 already bound                         | `sudo lsof -i :443` to find the process                                                                                                  |
| `Connection refused` from build runner             | Wrong IP or SG blocking                        | Confirm `sg-signing-server` allows TCP 443 from build runner SG                                                                          |
| `Invalid key_id format`                            | key_id contains disallowed characters          | key_id must match `[a-zA-Z0-9@.\-_ <>]+`                                                                                                 |
| `externally-managed-environment` pip error         | Python 3.12+ on Debian/Ubuntu                  | Re-run setup-server.sh — it uses venv automatically                                                                                      |
| `python3-venv` not found                           | Missing OS package                             | `sudo apt install python3.X-venv` where X matches `python3 --version`                                                                    |

______________________________________________________________________

## Section 6 — Provisioning Order Reference

```
1. Create IAM role (role-signing-server)
   └── Role ARN needed for KMS key policy

2. Create KMS CMK (alias/amd-signing-gpg-key)
   └── CMK needed for Secrets Manager encryption
   └── Key policy references role ARN from step 1

3. Store GPG private keys in Secrets Manager
   └── Uses CMK from step 2 for encryption
   └── Secret must exist before server starts

4. Attach IAM role to EC2 instance
   └── Instance needs credentials before boto3 calls work

5. Run setup-server.sh on EC2 instance
   └── Fetches key from SM (needs role from step 4)
   └── Starts signing server

6. Verify: curl -k https://localhost/health → {"status": "ok"}
```

Skipping or reordering any step produces a specific error:

| Wrong order                          | Error seen                                          |
| ------------------------------------ | --------------------------------------------------- |
| KMS before role                      | `InvalidPrincipalException` when setting key policy |
| SM before KMS                        | `KMSNotFoundException` when creating secret         |
| Start server before role attached    | `Unable to locate credentials`                      |
| Start server before SM secret exists | `ResourceNotFoundException` fetching secret         |

______________________________________________________________________

## Section 7 — Phase 1b: Cross-Cloud Client Access

Additive to the Phase 1 setup above — see `signing-server-design.md` §4.1b for the full design and `signing-server-requirements.md` §2A for requirements. This section covers (a) one-time provisioning of the API Gateway/VPC Link/NLB front door, and (b) the repeatable procedure for onboarding each new signing client (a repo/IAM role pair).

### 7.1 One-Time Provisioning

```
1. Create internal NLB + target group, register the existing signing server EC2 instance
   aws elbv2 create-load-balancer --name signing-nlb --type network --scheme internal \
     --subnets <private-subnet-ids>
   aws elbv2 create-target-group --name signing-targets --protocol TCP --port 443 \
     --target-type instance --vpc-id <vpc-id>
   aws elbv2 register-targets --target-group-arn <tg-arn> --targets Id=<signing-server-instance-id>
   aws elbv2 create-listener --load-balancer-arn <nlb-arn> --protocol TCP --port 443 \
     --default-actions Type=forward,TargetGroupArn=<tg-arn>

2. Create a VPC Link to the NLB
   aws apigateway create-vpc-link --name signing-vpc-link \
     --target-arns <nlb-arn>
   └── Creates ENIs in the NLB's subnets — note their Security Group for step 3

3. Update the signing server's Security Group
   Add an inbound rule on sg-signing-server allowing TCP 443 from the VPC Link's
   Security Group/ENIs only (do not open it to the whole VPC CIDR)

4. Create the REST API (v1 — required for the resource policy in step 5)
   aws apigateway create-rest-api --name therock-signing-api
   aws apigateway put-method --rest-api-id <api-id> --resource-id <root-id> \
     --http-method POST --authorization-type AWS_IAM
   aws apigateway put-integration --rest-api-id <api-id> --resource-id <root-id> \
     --http-method POST --type HTTP_PROXY --integration-http-method POST \
     --connection-type VPC_LINK --connection-id <vpc-link-id> \
     --uri https://<nlb-dns-name>/sign

5. Attach a resource policy allow-listing authorized client IAM role ARNs
   (see 7.2 step 3 below — the policy is maintained incrementally as clients are onboarded)

6. Deploy the API to a stage (e.g. "prod")
   aws apigateway create-deployment --rest-api-id <api-id> --stage-name prod

7. Verify: no client onboarded yet, so any invoke attempt (even with valid AWS
   credentials) should return 403 from the resource policy — confirms the
   deny-by-default posture before the first client is added
```

### 7.2 Onboarding a New Client (per repository)

Repeat for each repo/role that needs cross-cloud signing access. The default assumption (see §4.1b) is that the client's existing GitHub-OIDC-federated IAM role — the same one used for S3 artifact uploads — is reused; substitute a dedicated signing-only role if that assumption doesn't hold for this client.

```
1. Identify (or create) the client's IAM role
   Existing role: build_tools/_therock_utils/s3_buckets.py (e.g. "therock-nightly")
   New role: trust policy scoped to repo:<org>/<repo>:ref:<ref> via the
   token.actions.githubusercontent.com:sub condition — copy the shape of an
   existing S3-upload role's trust policy

2. Attach an execute-api:Invoke permission scoped to this API's /sign route only
   aws iam put-role-policy --role-name <role-name> --policy-name signing-invoke \
     --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
     "Action":"execute-api:Invoke","Resource":"arn:aws:execute-api:<region>:<acct>:<api-id>/prod/POST/sign"}]}'

3. Add the role ARN to the API Gateway resource policy allow-list, redeploy the stage
   aws apigateway update-rest-api --rest-api-id <api-id> --patch-operations \
     op=replace,path=/policy,value='<policy JSON with role ARN added to Principal list>'
   aws apigateway create-deployment --rest-api-id <api-id> --stage-name prod

4. Add an entry to authorization.json's "clients" map, mapping the role name to
   an existing (or new) "role" in the "roles" block with the intended
   allowed_keys/allowed_digest_algos/max_requests_per_hour — never reuse
   TheRock's key aliases for a different repo's client

5. Test with a non-production key/tier first:
   - Trigger the client's workflow on a test branch with TRUST_APIGW_HEADER
     disabled on a staging signing server, confirm the SigV4 request round-trips
   - Only then enable against the production signing server / production tier

6. To revoke a client: remove its entry from authorization.json's "clients" map
   AND remove its ARN from the API Gateway resource policy (either alone is
   sufficient to deny it — do both for defense-in-depth and a clean audit trail)
```

| Symptom                                                             | Likely cause                                                                                                                                                     |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `403` from API Gateway itself (never reaches signing server logs)   | Role not on the resource policy allow-list, or missing `execute-api:Invoke` permission                                                                           |
| `403` from the signing server (`Forbidden: Unknown signing client`) | Role reached the server but has no entry in `authorization.json`'s `clients` map                                                                                 |
| `401` "missing client identity" from the signing server             | `TRUST_APIGW_HEADER=true` but the API Gateway integration isn't mapping `$context.identity.userArn` into `X-Signing-Client-Role`                                 |
| Works from AWS-hosted runners, fails from Azure-hosted ones         | Check the Azure runner actually received `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN` from the OIDC exchange step before the signing step ran |

______________________________________________________________________

## Section 8 — Instance Replacement Procedure (Required Before Production Cutover)

`signing-server-design.md` §4 states the intended model — **an instance is replaced for any code or config change, never patched in place** — but until now no concrete procedure existed for actually doing that, and §1.5's `git clone` step reads like an ongoing update mechanism if followed literally. This section is that missing procedure, and closes both gaps together.

### 8.1 Why "patch in place" is rejected here specifically

- The server explicitly has no SSH or SSM access except under the audited break-glass procedure (`signing-server-design.md` §4) — there is no routine, low-friction way to reach a shell on a live instance to run `git pull` anyway, by design.
- A floating branch (including `main`) does not give a reproducible answer to "what code is actually running on this instance right now" — two instances built from the same branch name at different times can silently differ.
- In-place mutation of a signing server has no clean rollback: if a bad config change causes signing failures, the previous known-good state has already been overwritten.

### 8.2 The gap this must close before Phase 1 goes to production

**Phase 1 clients call the signing server's raw private IP directly** (§3.1a, EXT-1/EXT-2 — no ALB exists until Phase 2, no DNS name is used anywhere in the current design). This means replacing the instance today would require updating every caller's configured endpoint (the `GPG_SIGNING_SERVER` value baked into CI secrets and operator tooling) at the same moment the instance changes — operationally fragile and easy to get wrong under incident pressure.

**Required fix, before relying on this procedure in production:** put a stable address in front of the single Phase 1 instance so replacement never requires a caller-side config change:

- Simplest: a private Route 53 record (e.g. `signing.internal.therock.aws`) pointed at the current instance's private IP, updated as part of step 4 below. Callers are configured with the DNS name, never a raw IP.
- Alternative: reuse the same Elastic Network Interface (ENI) across replacements — detach it from the old instance, attach to the new one — so the private IP itself never changes. Works for Phase 1's single-instance model; stop using this once Phase 2's ALB makes it moot.

Either is a small, one-time addition. Do not skip this — without it, this replacement procedure just relocates the "update every client" problem instead of solving it.

### 8.3 Procedure

```
1. Launch a new EC2 instance
   Same AMI base, same role-signing-server instance profile, same
   Security Group, same subnet as the instance being replaced

2. Bootstrap it (§1.5) pinned to the new release's tag/commit SHA
   sudo git clone https://github.com/ROCm/TheRock.git /opt/therock-signing
   cd /opt/therock-signing && sudo git checkout <new-tag-or-sha>
   sudo bash build_tools/packaging/linux/signing_infrastructure/tools/setup-server.sh \
     --secret signing/gpg/therock-release --region $AWS_REGION
   sudo systemctl start signing-server

3. Verify before cutover
   curl -k https://<new-instance-private-ip>/health
   Expect: {"status": "ok"} — confirms keyring loaded, service healthy,
   BEFORE any caller traffic is pointed at it

4. Cut over
   Update the Route 53 record (or re-attach the shared ENI) to point at
   the new instance. Nothing on the caller side changes.

5. Confirm traffic is flowing to the new instance
   Check the new instance's audit log / CloudWatch for incoming SIGNED
   entries; check the old instance has stopped receiving new requests

6. Terminate the old instance
   Only after step 5 confirms the new instance is serving successfully.
   Keep it stopped-but-not-terminated for a short window if you want a
   fast manual rollback option instead of terminating immediately.
```

### 8.4 Rollback

If step 3 or step 5 fails: do not cut over. Terminate the failed new instance, fix the issue, and retry from step 1. If a problem is only discovered *after* cutover (step 4): repeat this procedure targeting the previous known-good tag/commit — this is why pinning to a specific tag/SHA (not a branch) at every step matters; it's what makes "go back to the last thing that worked" an unambiguous action rather than a guess.
