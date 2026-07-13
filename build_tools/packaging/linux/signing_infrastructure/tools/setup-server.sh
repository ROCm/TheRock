#!/usr/bin/env bash
# setup-server.sh — One-time EC2 signing server provisioning script.
#
# Run this once after launching a fresh EC2 instance (Amazon Linux 2023
# or Ubuntu 22.04 LTS) with the role-signing-server IAM instance profile.
#
# What this script does:
#   1. Installs system dependencies (gpg, python3, pip)
#   2. Installs Python packages (boto3, PyJWT)
#   3. Creates the tmpfs keyring mount (persisted in /etc/fstab)
#   4. Copies signing-server.py to /opt/signing-server/
#   5. Installs and enables the systemd service
#
# Usage:
#   sudo bash setup-server.sh [--secret signing/gpg/therock-release] [--region us-east-1]
#
# After running, start the server:
#   sudo systemctl start signing-server
#   curl -k https://localhost/health

set -euo pipefail

# --- Configuration ---
INSTALL_DIR="/opt/signing-server"
KEYRING_DIR="/var/gpg-keyring"
SERVICE_NAME="signing-server"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SERVER_PORT=443
SM_SECRET=""
AWS_REGION=""

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --secret)   SM_SECRET="$2";  shift 2 ;;
        --region)   AWS_REGION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --- Detect OS ---
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID="${ID:-unknown}"
else
    OS_ID="unknown"
fi

echo "============================================"
echo "  GPG Signing Server — Setup Script"
echo "============================================"
echo "OS:          ${OS_ID}"
echo "Install dir: ${INSTALL_DIR}"
echo "Keyring:     ${KEYRING_DIR} (tmpfs)"
echo "Port:        ${SERVER_PORT}"
if [ -n "$SM_SECRET" ]; then
    echo "SM Secret:   ${SM_SECRET}"
fi
echo "============================================"

# --- 1. Install system dependencies ---
echo ""
echo "[1/5] Installing system dependencies..."

if [[ "$OS_ID" == "amzn" || "$OS_ID" == "rhel" || "$OS_ID" == "centos" ]]; then
    yum install -y gnupg2 python3 python3-pip openssl
elif [[ "$OS_ID" == "ubuntu" || "$OS_ID" == "debian" ]]; then
    apt-get update -q
    apt-get install -y gnupg2 python3 python3-pip openssl
else
    echo "WARNING: Unknown OS '${OS_ID}'. Install gnupg2 python3 python3-pip manually."
fi

gpg --version | head -1
python3 --version

# --- 2. Install Python packages into a virtual environment ---
echo ""
echo "[2/5] Installing Python packages..."

VENV_DIR="/opt/signing-server-venv"

# Create venv (Python 3.12+ on Debian/Ubuntu prohibits system-wide pip installs)
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet "boto3>=1.26.0" "PyJWT[crypto]>=2.8.0"

echo "  Virtual environment: ${VENV_DIR}"
echo "  boto3 and PyJWT installed"

# Use venv Python for the server process
PYTHON_BIN="${VENV_DIR}/bin/python3"

# --- 3. Create tmpfs keyring mount ---
echo ""
echo "[3/5] Setting up tmpfs keyring at ${KEYRING_DIR}..."

mkdir -p "${KEYRING_DIR}"
chmod 700 "${KEYRING_DIR}"

# Mount now (if not already mounted)
if ! mountpoint -q "${KEYRING_DIR}"; then
    mount -t tmpfs -o size=64m,mode=0700 tmpfs "${KEYRING_DIR}"
    echo "  Mounted tmpfs at ${KEYRING_DIR}"
else
    echo "  Already mounted"
fi

# Persist in /etc/fstab so it survives reboots
FSTAB_ENTRY="tmpfs  ${KEYRING_DIR}  tmpfs  size=64m,mode=0700  0  0"
if ! grep -qF "${KEYRING_DIR}" /etc/fstab; then
    echo "${FSTAB_ENTRY}" >> /etc/fstab
    echo "  Added to /etc/fstab"
else
    echo "  Already in /etc/fstab"
fi

# --- 4. Install server files ---
echo ""
echo "[4/5] Installing server to ${INSTALL_DIR}..."

mkdir -p "${INSTALL_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SRC="${SCRIPT_DIR}/../server"

cp "${SERVER_SRC}/signing-server.py" "${INSTALL_DIR}/"
cp "${SERVER_SRC}/auth.py"           "${INSTALL_DIR}/"

# Copy config if present
if [ -f "${SCRIPT_DIR}/../config/authorization.json" ]; then
    mkdir -p "${INSTALL_DIR}/config"
    cp "${SCRIPT_DIR}/../config/authorization.json" "${INSTALL_DIR}/config/"
fi

chmod 755 "${INSTALL_DIR}/signing-server.py"
echo "  Files installed to ${INSTALL_DIR}"

# --- 4b. Generate self-signed TLS cert (Phase 1) ---
CERT_DIR="${INSTALL_DIR}/certs"
mkdir -p "${CERT_DIR}"

if [ ! -f "${CERT_DIR}/server.crt" ]; then
    echo "  Generating self-signed TLS certificate..."
    openssl req -x509 -newkey rsa:4096 -nodes \
        -keyout "${CERT_DIR}/server.key" \
        -out    "${CERT_DIR}/server.crt" \
        -days   3650 \
        -subj   "/CN=signing-server" \
        -addext "subjectAltName=IP:127.0.0.1" \
        2>/dev/null
    chmod 600 "${CERT_DIR}/server.key"
    echo "  Certificate: ${CERT_DIR}/server.crt"
else
    echo "  TLS certificate already exists — skipping"
fi

# --- 5. Install systemd service ---
echo ""
echo "[5/5] Installing systemd service..."

# Build ExecStart command — use venv Python, not system Python
EXEC_START="${PYTHON_BIN} ${INSTALL_DIR}/signing-server.py"
EXEC_START="${EXEC_START} --host 0.0.0.0"
EXEC_START="${EXEC_START} --port ${SERVER_PORT}"
EXEC_START="${EXEC_START} --keyring ${KEYRING_DIR}"
EXEC_START="${EXEC_START} --enable-tls"
EXEC_START="${EXEC_START} --cert-file ${CERT_DIR}/server.crt"
EXEC_START="${EXEC_START} --key-file  ${CERT_DIR}/server.key"
EXEC_START="${EXEC_START} --authz-config ${INSTALL_DIR}/config/authorization.json"
if [ -n "$SM_SECRET" ]; then
    EXEC_START="${EXEC_START} --secrets-manager-secret ${SM_SECRET}"
fi
if [ -n "$AWS_REGION" ]; then
    EXEC_START="${EXEC_START} --region ${AWS_REGION}"
fi

cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=GPG Remote Signing Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
ExecStart=${EXEC_START}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=${KEYRING_DIR} /var/log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
echo "  Service installed and enabled: ${SERVICE_NAME}"

# --- Done ---
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Start the server:"
echo "       sudo systemctl start ${SERVICE_NAME}"
echo ""
echo "  2. Check status:"
echo "       sudo systemctl status ${SERVICE_NAME}"
echo "       sudo journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "  3. Verify health:"
echo "       curl -k https://localhost/health"
echo ""
if [ -z "$SM_SECRET" ]; then
    echo "  NOTE: No --secret provided. The server will start but /health"
    echo "  will return 503 until GPG keys are loaded. To load keys, restart"
    echo "  the service with --secrets-manager-secret <name>."
    echo "  Or re-run this script with --secret <secret-name>"
fi
