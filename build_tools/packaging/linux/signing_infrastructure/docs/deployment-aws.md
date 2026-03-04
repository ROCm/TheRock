# AWS Deployment Guide

This guide covers deploying the GPG remote signing server on AWS infrastructure with production-grade security and reliability.

## Architecture Overview

```
GitHub Actions (Azure build servers)
        |
        | HTTPS (443)
        v
AWS Application Load Balancer (ALB)
  - TLS termination
  - ACM certificate
  - Health checks
        |
        | HTTP (8080)
        v
EC2 Instance (signing server)
  - Python 3 application
  - systemd service
  - GPG keyring
  - Audit logging
```

## Prerequisites

- AWS account with EC2, ALB, ACM access
- GPG private key(s) for signing
- Domain name for the signing server (e.g., `signing.company.com`)
- GitHub Actions or Azure build server IPs to whitelist

## EC2 Instance Setup

### 1. Launch EC2 Instance

**Recommended Instance Type:**
- **t3.small** for light usage (<100 signatures/day)
- **t3.medium** for moderate usage (100-1000 signatures/day)
- **c5.large** for heavy usage (>1000 signatures/day)

**AMI Selection:**
- Amazon Linux 2023 (Python 3.9+)
- Ubuntu 22.04 LTS (Python 3.10+)
- RHEL 8/9 (Python 3.6+)

**Storage:**
- 20 GB root volume (gp3 SSD)
- Enable encryption at rest

**Key Pair:**
- Create new key pair for SSH access
- Store private key securely (e.g., AWS Secrets Manager)

### 2. Security Group Configuration

Create security group `signing-server-sg`:

**Inbound Rules:**
```
Type        Protocol  Port    Source              Description
HTTP        TCP       8080    ALB security group  Application traffic from ALB
SSH         TCP       22      Admin VPN IP        Administrative access only
```

**Outbound Rules:**
```
Type        Protocol  Port    Destination         Description
HTTPS       TCP       443     0.0.0.0/0          Package updates
HTTP        TCP       80      0.0.0.0/0          Package updates (HTTP redirect)
```

### 3. Install Dependencies

SSH to instance and install required packages:

```bash
# Amazon Linux 2023 / RHEL
sudo dnf install -y python3 python3-pip gnupg2 git

# Ubuntu
sudo apt-get update
sudo apt-get install -y python3 python3-pip gnupg git
```

### 4. Create Service User

```bash
# Create dedicated user (no login shell)
sudo useradd -r -s /bin/false signing-server

# Create directories
sudo mkdir -p /opt/signing-server
sudo mkdir -p /var/log/gpg-signing
sudo mkdir -p /etc/signing-server

# Set ownership
sudo chown -R signing-server:signing-server /opt/signing-server
sudo chown -R signing-server:signing-server /var/log/gpg-signing
sudo chown -R signing-server:signing-server /etc/signing-server
```

### 5. Deploy Application

```bash
# Clone repository (or download release tarball)
cd /opt/signing-server
sudo -u signing-server git clone https://github.com/yourorg/gpg-signing-server.git .

# Or extract tarball
# sudo -u signing-server tar xzf signing-server-v1.0.tar.gz -C /opt/signing-server
```

### 6. Import GPG Keys

**Option A: Import from encrypted backup**

```bash
# Copy encrypted keyring backup to server
scp keyring-backup.tar.gz.gpg ec2-user@signing-server:/tmp/

# On server, decrypt and import
sudo -u signing-server gpg --decrypt /tmp/keyring-backup.tar.gpg | \
    tar xz -C /etc/signing-server/.gnupg

# Set correct permissions
sudo chmod 700 /etc/signing-server/.gnupg
sudo chmod 600 /etc/signing-server/.gnupg/*
```

**Option B: Generate new key on server**

```bash
# Generate key as signing-server user
sudo -u signing-server gpg --homedir /etc/signing-server/.gnupg \
    --batch --gen-key << EOF
Key-Type: RSA
Key-Length: 4096
Name-Real: Production Signing Key
Name-Email: signing@company.com
Expire-Date: 2y
%no-protection
%commit
EOF

# Export public key for verification
sudo -u signing-server gpg --homedir /etc/signing-server/.gnupg \
    --armor --export signing@company.com > /tmp/signing-key-pub.asc
```

**Option C: Import from AWS Secrets Manager**

```bash
# Install AWS CLI if not present
pip3 install awscli

# Retrieve key from Secrets Manager
aws secretsmanager get-secret-value \
    --secret-id prod/signing-key \
    --query SecretString \
    --output text > /tmp/private-key.asc

# Import key
sudo -u signing-server gpg --homedir /etc/signing-server/.gnupg \
    --import /tmp/private-key.asc

# Securely delete temporary file
shred -u /tmp/private-key.asc
```

### 7. Configure Application

Create `/etc/signing-server/secrets.json`:

```bash
# Retrieve from AWS Secrets Manager
aws secretsmanager get-secret-value \
    --secret-id prod/signing-secrets \
    --query SecretString \
    --output text | \
    sudo -u signing-server tee /etc/signing-server/secrets.json

sudo chmod 600 /etc/signing-server/secrets.json
```

Create `/etc/signing-server/authorization.json`:

```bash
# Copy from version control (not secret)
sudo cp /opt/signing-server/config/authorization.json.example \
    /etc/signing-server/authorization.json

# Edit with actual key IDs
sudo -u signing-server vim /etc/signing-server/authorization.json
```

### 8. Create systemd Service

Create `/etc/systemd/system/signing-server.service`:

```ini
[Unit]
Description=GPG Remote Signing Server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=signing-server
Group=signing-server
WorkingDirectory=/opt/signing-server

# Environment
Environment=GNUPGHOME=/etc/signing-server/.gnupg
Environment=PYTHONUNBUFFERED=1

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/gpg-signing

# Resource limits
LimitNOFILE=4096
TasksMax=20

# Start server
ExecStart=/usr/bin/python3 /opt/signing-server/signing-server.py \
    --port 8080 \
    --keyring /etc/signing-server/.gnupg \
    --enable-auth \
    --secrets-file /etc/signing-server/secrets.json \
    --authz-config /etc/signing-server/authorization.json \
    --audit-log /var/log/gpg-signing/audit.log \
    --max-threads 10 \
    --max-request-size 10240

# Restart policy
Restart=always
RestartSec=10s

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=signing-server

[Install]
WantedBy=multi-user.target
```

### 9. Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable signing-server

# Start service
sudo systemctl start signing-server

# Check status
sudo systemctl status signing-server

# View logs
sudo journalctl -u signing-server -f
```

### 10. Verify Service

```bash
# Check if listening on port 8080
sudo netstat -tlnp | grep 8080

# Test health endpoint
curl http://localhost:8080/health

# Expected response:
# {"status": "healthy", "version": "1.0"}
```

## Application Load Balancer Setup

### 1. Request ACM Certificate

```bash
# Via AWS Console or CLI
aws acm request-certificate \
    --domain-name signing.company.com \
    --validation-method DNS \
    --region us-east-1
```

Add DNS validation records to your domain:
- Check ACM console for CNAME records
- Add to Route 53 or external DNS provider
- Wait for validation (usually <30 minutes)

### 2. Create Target Group

```bash
aws elbv2 create-target-group \
    --name signing-server-tg \
    --protocol HTTP \
    --port 8080 \
    --vpc-id vpc-xxxxxx \
    --health-check-enabled \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3
```

Register EC2 instance:
```bash
aws elbv2 register-targets \
    --target-group-arn arn:aws:elasticloadbalancing:... \
    --targets Id=i-1234567890abcdef0
```

### 3. Create ALB Security Group

Create `alb-sg`:

**Inbound Rules:**
```
Type        Protocol  Port    Source              Description
HTTPS       TCP       443     GitHub Actions IPs  Client access
HTTPS       TCP       443     Azure build IPs     Client access
```

Find GitHub Actions IP ranges:
```bash
curl https://api.github.com/meta | jq -r '.actions[]'
```

**Outbound Rules:**
```
Type        Protocol  Port    Destination         Description
HTTP        TCP       8080    signing-server-sg   Backend traffic
```

### 4. Create Application Load Balancer

```bash
aws elbv2 create-load-balancer \
    --name signing-server-alb \
    --subnets subnet-xxxxxx subnet-yyyyyy \
    --security-groups sg-alb-xxxxxxxx \
    --scheme internet-facing \
    --type application \
    --ip-address-type ipv4
```

### 5. Create HTTPS Listener

```bash
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:... \
    --protocol HTTPS \
    --port 443 \
    --certificates CertificateArn=arn:aws:acm:... \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
```

### 6. Update DNS

Create Route 53 record (or update external DNS):

```
Type: A (Alias)
Name: signing.company.com
Alias Target: signing-server-alb-xxxxxxxxx.us-east-1.elb.amazonaws.com
```

### 7. Test ALB

```bash
# Test from external client
curl -H "Authorization: Bearer $TOKEN" \
    https://signing.company.com/health

# Should return:
# {"status": "healthy", "version": "1.0"}
```

## Certificate Management

### Option 1: AWS Certificate Manager (Recommended)

**Advantages:**
- Automatic renewal
- Free
- Integrated with ALB
- Managed by AWS

**Setup:**
- Request certificate via ACM
- Validate via DNS (CNAME records)
- Attach to ALB listener
- No action needed for renewal

### Option 2: Let's Encrypt

**Advantages:**
- Free
- Works outside AWS
- Widely trusted

**Setup:**

```bash
# Install certbot
sudo dnf install -y certbot

# Request certificate (requires port 80 open)
sudo certbot certonly --standalone \
    -d signing.company.com \
    --agree-tos \
    --email admin@company.com

# Certificate stored in:
# /etc/letsencrypt/live/signing.company.com/fullchain.pem
# /etc/letsencrypt/live/signing.company.com/privkey.pem
```

Configure server to use TLS directly (without ALB):

```bash
# Update systemd service
sudo systemctl edit signing-server

# Add:
Environment=TLS_CERT_FILE=/etc/letsencrypt/live/signing.company.com/fullchain.pem
Environment=TLS_KEY_FILE=/etc/letsencrypt/live/signing.company.com/privkey.pem

# Update ExecStart to enable TLS
ExecStart=/usr/bin/python3 /opt/signing-server/signing-server.py \
    --port 8443 \
    --enable-tls \
    --tls-cert $TLS_CERT_FILE \
    --tls-key $TLS_KEY_FILE \
    ...
```

Setup auto-renewal:
```bash
# Test renewal
sudo certbot renew --dry-run

# Enable auto-renewal (cron already configured)
sudo systemctl enable certbot-renew.timer
```

### Option 3: Corporate CA

**Use Case:** Internal deployments only

```bash
# Copy certificates to server
scp corporate-ca.crt ec2-user@signing-server:/tmp/
scp signing-server.crt ec2-user@signing-server:/tmp/
scp signing-server.key ec2-user@signing-server:/tmp/

# Move to proper location
sudo mkdir -p /etc/signing-server/tls
sudo mv /tmp/signing-server.crt /etc/signing-server/tls/
sudo mv /tmp/signing-server.key /etc/signing-server/tls/
sudo chmod 600 /etc/signing-server/tls/signing-server.key
sudo chown signing-server:signing-server /etc/signing-server/tls/*
```

Update server configuration as in Option 2.

## Monitoring and Alerting

### CloudWatch Metrics

**EC2 Metrics:**
- `CPUUtilization` - Alert if >80% for 5 minutes
- `NetworkIn/NetworkOut` - Track request volume
- `StatusCheckFailed` - Alert on instance health issues

**ALB Metrics:**
- `TargetResponseTime` - Track signing latency
- `HTTPCode_Target_2XX_Count` - Successful requests
- `HTTPCode_Target_4XX_Count` - Client errors (auth failures)
- `HTTPCode_Target_5XX_Count` - Server errors
- `RequestCount` - Total traffic

**Custom Application Metrics:**

Install CloudWatch agent:
```bash
sudo dnf install -y amazon-cloudwatch-agent
```

Configure metrics:
```json
{
  "metrics": {
    "namespace": "SigningServer",
    "metrics_collected": {
      "log_files": {
        "collect_list": [
          {
            "file_path": "/var/log/gpg-signing/audit.log",
            "log_group_name": "/aws/signing-server/audit",
            "log_stream_name": "{instance_id}"
          }
        ]
      }
    }
  }
}
```

### CloudWatch Alarms

**High Error Rate:**
```bash
aws cloudwatch put-metric-alarm \
    --alarm-name signing-server-high-error-rate \
    --alarm-description "Alert on high 5xx error rate" \
    --metric-name HTTPCode_Target_5XX_Count \
    --namespace AWS/ApplicationELB \
    --statistic Sum \
    --period 300 \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:ops-alerts
```

**High CPU:**
```bash
aws cloudwatch put-metric-alarm \
    --alarm-name signing-server-high-cpu \
    --alarm-description "Alert on sustained high CPU" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:ops-alerts
```

### Audit Log Analysis

Ship logs to CloudWatch Logs for analysis:

```bash
# Install and configure CloudWatch Logs agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -s \
    -c file:/etc/signing-server/cloudwatch-config.json
```

Query logs with CloudWatch Insights:

```sql
# Failed authentication attempts
fields timestamp, client_ip, client_id, action
| filter action = "AUTH_FAILED"
| stats count() by client_ip
| sort count desc

# Rate limited clients
fields timestamp, client_id, role
| filter action = "RATE_LIMITED"
| stats count() by client_id

# Successful signatures by role
fields timestamp, role, key_id
| filter action = "SIGNED"
| stats count() by role
```

### Health Checks

**ALB Health Check:**
- Path: `/health`
- Interval: 30 seconds
- Timeout: 5 seconds
- Healthy threshold: 2
- Unhealthy threshold: 3

**Custom Health Check Script:**

```bash
#!/bin/bash
# /usr/local/bin/check-signing-server.sh

# Check if process is running
if ! systemctl is-active --quiet signing-server; then
    echo "CRITICAL: Service not running"
    exit 2
fi

# Check if responding
if ! curl -sf http://localhost:8080/health > /dev/null; then
    echo "CRITICAL: Service not responding"
    exit 2
fi

# Check GPG keyring
if ! sudo -u signing-server gpg --homedir /etc/signing-server/.gnupg --list-keys > /dev/null 2>&1; then
    echo "WARNING: GPG keyring issue"
    exit 1
fi

echo "OK: All checks passed"
exit 0
```

Run via cron every 5 minutes:
```cron
*/5 * * * * /usr/local/bin/check-signing-server.sh || logger -t signing-check "Health check failed"
```

## Backup and Disaster Recovery

### Backup GPG Keyring

**Daily Automated Backup:**

```bash
#!/bin/bash
# /usr/local/bin/backup-gpg-keyring.sh

DATE=$(date +%Y%m%d)
BACKUP_FILE="/tmp/gpg-keyring-$DATE.tar.gz"
S3_BUCKET="s3://company-signing-backups/keyring/"

# Create encrypted backup
sudo -u signing-server tar czf - -C /etc/signing-server .gnupg | \
    gpg --encrypt --recipient backup@company.com > "$BACKUP_FILE.gpg"

# Upload to S3
aws s3 cp "$BACKUP_FILE.gpg" "$S3_BUCKET"

# Remove local copy
rm "$BACKUP_FILE.gpg"

# Rotate old backups (keep 30 days)
aws s3 ls "$S3_BUCKET" | awk '{print $4}' | \
    sort -r | tail -n +31 | \
    xargs -I {} aws s3 rm "$S3_BUCKET{}"
```

Add to cron:
```cron
0 2 * * * /usr/local/bin/backup-gpg-keyring.sh
```

### Disaster Recovery Procedure

**Scenario: EC2 instance lost**

1. **Launch new EC2 instance** (same setup as above)

2. **Restore GPG keyring from S3:**
   ```bash
   aws s3 cp s3://company-signing-backups/keyring/gpg-keyring-20260302.tar.gz.gpg /tmp/

   gpg --decrypt /tmp/gpg-keyring-20260302.tar.gz.gpg | \
       sudo -u signing-server tar xz -C /etc/signing-server/
   ```

3. **Restore configuration from Secrets Manager:**
   ```bash
   aws secretsmanager get-secret-value --secret-id prod/signing-secrets \
       --query SecretString --output text | \
       sudo -u signing-server tee /etc/signing-server/secrets.json
   ```

4. **Start service:**
   ```bash
   sudo systemctl start signing-server
   ```

5. **Register with ALB:**
   ```bash
   aws elbv2 register-targets \
       --target-group-arn arn:aws:elasticloadbalancing:... \
       --targets Id=i-<new-instance-id>
   ```

6. **Verify health:**
   ```bash
   curl http://localhost:8080/health
   ```

**Recovery Time Objective (RTO):** 15-30 minutes
**Recovery Point Objective (RPO):** 24 hours (daily backups)

## Security Hardening

### OS-Level Security

**Enable automatic security updates:**
```bash
# Amazon Linux 2023
sudo dnf install -y dnf-automatic
sudo systemctl enable --now dnf-automatic.timer

# Ubuntu
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

**Configure firewall:**
```bash
# Use firewalld (RHEL/Amazon Linux)
sudo systemctl enable --now firewalld
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# Or use ufw (Ubuntu)
sudo ufw allow from <ALB-IP> to any port 8080
sudo ufw enable
```

**Enable audit logging:**
```bash
sudo systemctl enable --now auditd

# Monitor GPG keyring access
sudo auditctl -w /etc/signing-server/.gnupg -p rwa -k gpg-keyring
```

### IAM Role

Attach IAM role to EC2 instance for AWS API access:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:us-east-1:123456789012:secret:prod/signing-*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::company-signing-backups/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:123456789012:log-group:/aws/signing-server/*"
    }
  ]
}
```

## Cost Optimization

**Estimated Monthly Cost (us-east-1):**

| Resource | Type | Cost |
|----------|------|------|
| EC2 Instance | t3.small | $15 |
| EBS Storage | 20 GB gp3 | $2 |
| ALB | 730 hours | $22 |
| Data Transfer | 1 GB/month | $0.09 |
| **Total** | | **~$40/month** |

**Cost Reduction Strategies:**

1. **Use Savings Plans** - Save 30-50% on EC2
2. **Reserved Instances** - 1-year commitment saves 30%
3. **Use ALB only if needed** - For single client, TLS on EC2 is cheaper
4. **Right-size instance** - Monitor CPU, downsize if underutilized
5. **Use gp3 instead of gp2** - Same performance, lower cost

## Troubleshooting

**Service won't start:**
```bash
# Check logs
sudo journalctl -u signing-server -n 50

# Common issues:
# - GPG keyring permissions (should be 700)
# - Missing secrets.json (should exist and be readable)
# - Port already in use (check with: sudo netstat -tlnp | grep 8080)
```

**Health check failing:**
```bash
# Test locally
curl -v http://localhost:8080/health

# Check if process is listening
sudo netstat -tlnp | grep 8080

# Check security group allows ALB traffic
```

**High latency:**
```bash
# Check CPU usage
top

# Check GPG performance
time echo "test" | sudo -u signing-server gpg --homedir /etc/signing-server/.gnupg --clearsign

# Check disk I/O
iostat -x 1 10
```

**Authentication failures:**
```bash
# Check audit log
sudo tail -f /var/log/gpg-signing/audit.log | grep AUTH_FAILED

# Verify secrets file
sudo -u signing-server cat /etc/signing-server/secrets.json
```
