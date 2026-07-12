#!/usr/bin/env python3
"""
Production signing server for GPG package signing.

Accepts POST /sign requests containing data to sign, invokes gpg, and
returns the detached or clearsigned signature.

Phase 1: No application-layer auth — access controlled by VPC Security Groups.
Phase 2: Enable --enable-auth for pre-shared token validation.

Key loading:
  Pass --secrets-manager-secret <name> (repeatable) to fetch GPG private keys
  from AWS Secrets Manager at startup into the --keyring tmpfs directory.
  Without this flag, the keyring must be pre-populated manually (local testing).

Architecture:
- Multi-threaded server using ThreadingMixIn
- Configurable max concurrent threads (default: 10)
- Thread semaphore prevents resource exhaustion
"""

import json
import sys
import os
import time
import tempfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from base64 import b64decode, b64encode
from subprocess import run, PIPE, TimeoutExpired

# Global semaphore for limiting concurrent signing operations
_signing_semaphore = None
# Flag set to True once GPG keyring is loaded and verified
_keyring_ready = False


def get_signing_semaphore():
    global _signing_semaphore
    if _signing_semaphore is None:
        max_threads = int(os.environ.get('MAX_THREADS', '10'))
        _signing_semaphore = threading.Semaphore(max_threads)
    return _signing_semaphore


# ---------------------------------------------------------------------------
# Secrets Manager key loading (Phase 1)
# ---------------------------------------------------------------------------

def load_keys_from_secrets_manager(secret_names, keyring_dir, region=None, gpg_binary='gpg'):
    """
    Fetch GPG private keys from AWS Secrets Manager and import into keyring_dir.

    Each secret must contain a PEM-armored GPG private key block
    (-----BEGIN PGP PRIVATE KEY BLOCK-----).  Secrets Manager decrypts via
    KMS transparently — no kms:Decrypt call needed here.

    Args:
        secret_names: List of Secrets Manager secret names/ARNs
        keyring_dir:  Path to the tmpfs GNUPGHOME directory
        region:       AWS region (optional, uses boto3 default if None)
        gpg_binary:   Path to gpg binary

    Returns:
        Number of keys successfully imported

    Raises:
        SystemExit if boto3 is unavailable or any secret fetch fails
    """
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 is required for --secrets-manager-secret. "
              "Install with: pip install boto3", file=sys.stderr)
        sys.exit(1)

    kwargs = {}
    if region:
        kwargs['region_name'] = region
    client = boto3.client('secretsmanager', **kwargs)

    imported = 0
    env = os.environ.copy()
    env['GNUPGHOME'] = keyring_dir

    for secret_name in secret_names:
        print(f"Fetching GPG key from Secrets Manager: {secret_name}")
        try:
            response = client.get_secret_value(SecretId=secret_name)
        except Exception as e:
            print(f"ERROR: Failed to fetch secret '{secret_name}': {e}", file=sys.stderr)
            sys.exit(1)

        # SecretString contains the PEM-armored private key
        key_material = response.get('SecretString', '')
        if not key_material:
            print(f"ERROR: Secret '{secret_name}' is empty or binary", file=sys.stderr)
            sys.exit(1)

        if '-----BEGIN PGP PRIVATE KEY BLOCK-----' not in key_material:
            print(f"ERROR: Secret '{secret_name}' does not look like a GPG private key "
                  f"(missing PGP header). Check the secret was stored correctly.",
                  file=sys.stderr)
            sys.exit(1)

        # Pipe key directly into gpg --import — never write to disk
        result = run(
            [gpg_binary, '--batch', '--import'],
            input=key_material.encode('utf-8'),
            stdout=PIPE,
            stderr=PIPE,
            env=env
        )

        # Clear key material from memory
        key_material = None

        if result.returncode != 0:
            err = result.stderr.decode('utf-8', errors='ignore')
            print(f"ERROR: gpg --import failed for '{secret_name}': {err}", file=sys.stderr)
            sys.exit(1)

        print(f"  Imported key from '{secret_name}'")
        imported += 1

    return imported


def verify_keyring(keyring_dir, gpg_binary='gpg'):
    """
    Confirm at least one secret key is present in the keyring.

    Returns:
        True if at least one key found, False otherwise
    """
    env = os.environ.copy()
    env['GNUPGHOME'] = keyring_dir

    result = run(
        [gpg_binary, '--list-secret-keys', '--keyid-format', 'LONG'],
        stdout=PIPE,
        stderr=PIPE,
        env=env
    )
    output = result.stdout.decode('utf-8', errors='ignore')
    count = output.count('sec ')
    print(f"Keyring verified: {count} secret key(s) available")
    return count > 0


# ---------------------------------------------------------------------------
# Import authentication module
# ---------------------------------------------------------------------------

try:
    from auth import (
        validate_jwt_token, validate_github_oidc_token,
        validate_app_token,
        load_secrets, load_authorization_config, load_tokens_config,
        authorize_request, authorize_oidc_request,
        check_rate_limit, audit_log,
        OIDC_AVAILABLE
    )
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    OIDC_AVAILABLE = False


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class SigningHandler(BaseHTTPRequestHandler):

    # Class-level caches shared across all handler instances
    _secrets_cache = None
    _authz_cache = None
    _tokens_cache = None
    _rate_limits = {}  # client_id -> deque of timestamps

    @property
    def GPG_BINARY(self):
        return os.environ.get('GPG_BINARY', 'gpg')

    @property
    def GPG_HOME(self):
        return os.environ.get('GNUPGHOME', '')

    @property
    def SIGNING_TIMEOUT(self):
        return int(os.environ.get('GPG_TIMEOUT', '30'))

    @property
    def MAX_REQUEST_SIZE(self):
        return int(os.environ.get('MAX_REQUEST_SIZE', '10240'))

    @property
    def READ_TIMEOUT(self):
        return int(os.environ.get('READ_TIMEOUT', '10'))

    @property
    def AUTH_ENABLED(self):
        return os.environ.get('AUTH_ENABLED', 'false').lower() == 'true'

    @property
    def SECRETS_FILE(self):
        return os.environ.get('SECRETS_FILE', '')

    @property
    def AUTHZ_CONFIG_FILE(self):
        return os.environ.get('AUTHZ_CONFIG_FILE', '')

    @property
    def TOKENS_FILE(self):
        return os.environ.get('TOKENS_FILE', '')

    @property
    def AUDIT_LOG_FILE(self):
        return os.environ.get('AUDIT_LOG_FILE', '')

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    def do_GET(self):
        if self.path != '/health':
            self.send_error(404, 'Not Found')
            return

        if _keyring_ready:
            self.send_json_response(200, {'status': 'ok'})
        else:
            self.send_json_response(503, {'status': 'unavailable',
                                          'reason': 'keyring not loaded'})

    # ------------------------------------------------------------------
    # POST /sign and POST /quit
    # ------------------------------------------------------------------

    def do_POST(self):
        if self.path == '/quit':
            self.send_json_response(200, {'message': 'Server shutting down'})
            self.log_message("Shutdown requested via /quit endpoint")
            def shutdown():
                time.sleep(0.5)
                self.server.shutdown()
            threading.Thread(target=shutdown).start()
            return

        if self.path != '/sign':
            self.send_error(404, 'Not Found')
            return

        request_start = time.time()

        try:
            # --- Read and size-check request body ---
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > self.MAX_REQUEST_SIZE:
                self.log_message("Request rejected: %d bytes exceeds %d byte limit",
                                 content_length, self.MAX_REQUEST_SIZE)
                self.send_json_error(413,
                    f"Request too large: {content_length} bytes "
                    f"(max {self.MAX_REQUEST_SIZE} bytes)")
                return

            original_timeout = self.connection.gettimeout()
            try:
                self.connection.settimeout(self.READ_TIMEOUT)
                body = self.rfile.read(content_length)
            except Exception as e:
                self.log_message("Read timeout or error: %s", str(e))
                self.send_json_error(408, "Request timeout")
                return
            finally:
                if original_timeout is not None:
                    self.connection.settimeout(original_timeout)

            request = json.loads(body.decode('utf-8'))

            # --- Authentication (Phase 1: disabled; Phase 2: token/JWT/OIDC) ---
            payload = None
            auth_type = 'none'
            client_id = self.client_address[0]  # source IP as default identifier

            if self.AUTH_ENABLED:
                if not AUTH_AVAILABLE:
                    self.send_json_error(500,
                        "Authentication enabled but auth module not available")
                    return
                payload, auth_type = self.authenticate_request()
                if payload is None:
                    self.send_json_error(401, "Unauthorized: Invalid or missing token")
                    audit_log('AUTH_FAILED', client_id, 'none', '', '',
                              self.client_address[0], False,
                              self.AUDIT_LOG_FILE, None, 'none',
                              int((time.time() - request_start) * 1000))
                    return
                # Use token-based client_id if available
                if auth_type in ('jwt', 'token'):
                    client_id = payload.get('client_id', client_id)
                elif auth_type == 'oidc':
                    client_id = f"oidc:{payload.get('repository', 'unknown')}"

            # --- Validate request fields ---
            data_b64 = request.get('data')
            if not data_b64:
                self.send_json_error(400, "Missing 'data' field")
                return

            key_id = request.get('key_id', '')
            digest_algo = request.get('digest_algo', 'SHA256').upper()
            armor = request.get('armor', False)
            clearsign = request.get('clearsign', False)

            if not key_id:
                self.send_json_error(400, "key_id is required")
                return

            if not self.validate_key_id(key_id):
                self.send_json_error(400,
                    "Invalid key_id format: must be alphanumeric, @, ., -, or _")
                return

            try:
                data = b64decode(data_b64)
            except Exception as e:
                self.send_json_error(400, f"Invalid base64 in 'data': {str(e)}")
                return

            if len(data) == 0:
                self.send_json_error(400, "Data cannot be empty")
                return

            self.log_message("Signing request: key=%s, algo=%s, armor=%s, "
                             "clearsign=%s, data_len=%d, client=%s",
                             key_id, digest_algo, armor, clearsign,
                             len(data), client_id)

            # --- Authorization (Phase 1: only key_id in-keyring check) ---
            if self.AUTH_ENABLED and payload:
                if self._authz_cache is None:
                    self.__class__._authz_cache = load_authorization_config(
                        self.AUTHZ_CONFIG_FILE)

                if auth_type == 'oidc':
                    role, authorized, reason = authorize_oidc_request(
                        payload, key_id, digest_algo, self._authz_cache)
                    if not authorized:
                        self.log_message("Authorization denied: %s", reason)
                        self.send_json_error(403, f"Forbidden: {reason}")
                        audit_log('DENIED', client_id, role or 'unknown',
                                  key_id, digest_algo,
                                  self.client_address[0], False,
                                  self.AUDIT_LOG_FILE,
                                  self._oidc_context(payload), auth_type,
                                  int((time.time() - request_start) * 1000))
                        return
                elif auth_type in ('jwt', 'token'):
                    role = payload.get('role', '')
                    authorized, reason = authorize_request(
                        role, key_id, digest_algo, self._authz_cache)
                    if not authorized:
                        self.log_message("Authorization denied: %s", reason)
                        self.send_json_error(403, f"Forbidden: {reason}")
                        audit_log('DENIED', client_id, role, key_id,
                                  digest_algo, self.client_address[0], False,
                                  self.AUDIT_LOG_FILE, None, auth_type,
                                  int((time.time() - request_start) * 1000))
                        return

                # Rate limit keyed by client_id (token name or source IP)
                if self._authz_cache and not check_rate_limit(
                        client_id, payload.get('role', 'default'),
                        self._rate_limits, self._authz_cache):
                    self.log_message("Rate limit exceeded for client %s", client_id)
                    self.send_json_error(429, "Rate limit exceeded")
                    audit_log('RATE_LIMITED', client_id,
                              payload.get('role', 'unknown'),
                              key_id, digest_algo, self.client_address[0],
                              False, self.AUDIT_LOG_FILE, None, auth_type,
                              int((time.time() - request_start) * 1000))
                    return

            # Phase 1: source-IP rate limiting (no auth token, use IP as key)
            if not self.AUTH_ENABLED and AUTH_AVAILABLE and self.AUTHZ_CONFIG_FILE:
                if self._authz_cache is None:
                    self.__class__._authz_cache = load_authorization_config(
                        self.AUTHZ_CONFIG_FILE)
                check_rate_limit(client_id, 'default',
                                 self._rate_limits, self._authz_cache)

            # --- Acquire semaphore ---
            semaphore = get_signing_semaphore()
            acquired = semaphore.acquire(blocking=True, timeout=30)
            if not acquired:
                self.log_message("Failed to acquire signing semaphore (server busy)")
                self.send_json_error(503, "Server busy, try again later")
                return

            try:
                signature = self.sign_data(data, key_id, digest_algo,
                                           armor, clearsign)
            finally:
                semaphore.release()

            if signature is None:
                self.send_json_error(500, "Signing failed")
                return

            latency_ms = int((time.time() - request_start) * 1000)

            response = {
                'signature': b64encode(signature).decode('ascii'),
                'key_id': key_id,
                'digest_algo': digest_algo
            }
            self.send_json_response(200, response)
            self.log_message("Signing successful (%dms)", latency_ms)

            audit_log('SIGNED', client_id,
                      payload.get('role', 'none') if payload else 'none',
                      key_id, digest_algo, self.client_address[0], True,
                      self.AUDIT_LOG_FILE,
                      self._oidc_context(payload) if auth_type == 'oidc' else None,
                      auth_type, latency_ms)

        except json.JSONDecodeError as e:
            self.send_json_error(400, f"Invalid JSON: {str(e)}")
        except Exception as e:
            self.log_message("Error: %s", str(e))
            self.send_json_error(500, f"Internal error: {str(e)}")

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate_request(self):
        """
        Validate request token.

        Phase 2 — tries in order:
          1. Pre-shared app token (Bearer token, constant-time compare)
          2. GitHub OIDC token (RS256, requires PyJWT)
          3. JWT HMAC-SHA256 token (shared secret)

        Phase 1 — AUTH_ENABLED=false, this method is never called.

        Returns:
            (payload dict, auth_type str) or (None, None) on failure.
            auth_type is one of: 'token', 'oidc', 'jwt'
        """
        if not self.AUTH_ENABLED or not AUTH_AVAILABLE:
            return None, None

        auth_header = self.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return None, None

        token = auth_header[7:]

        # 1. Pre-shared app token (Phase 2 primary)
        if self.TOKENS_FILE:
            if self._tokens_cache is None:
                self.__class__._tokens_cache = load_tokens_config(self.TOKENS_FILE)
            token_payload = validate_app_token(token, self._tokens_cache)
            if token_payload:
                self.log_message("App token validated for client: %s",
                                 token_payload.get('client_id', 'unknown'))
                return token_payload, 'token'

        # 2. GitHub OIDC token
        if OIDC_AVAILABLE:
            oidc_audience = os.environ.get('OIDC_AUDIENCE', 'amd-signing-service')
            oidc_payload = validate_github_oidc_token(token, oidc_audience)
            if oidc_payload:
                self.log_message("OIDC token validated for repo: %s, ref: %s",
                                 oidc_payload.get('repository', 'unknown'),
                                 oidc_payload.get('ref', 'unknown'))
                return oidc_payload, 'oidc'

        # 3. JWT HMAC-SHA256 token (fallback)
        if self.SECRETS_FILE:
            if self._secrets_cache is None:
                self.__class__._secrets_cache = load_secrets(self.SECRETS_FILE)
            jwt_payload = validate_jwt_token(token, self._secrets_cache)
            if jwt_payload:
                self.log_message("JWT token validated for client: %s, role: %s",
                                 jwt_payload.get('client_id', 'unknown'),
                                 jwt_payload.get('role', 'unknown'))
                return jwt_payload, 'jwt'

        return None, None

    # ------------------------------------------------------------------
    # GPG signing
    # ------------------------------------------------------------------

    def sign_data(self, data, key_id, digest_algo, armor, clearsign=False):
        """
        Sign data using gpg.

        Supports:
          --detach-sign  (default) — binary or ASCII-armored detached signature
          --clearsign    — data + signature in one block (required for InRelease)
        """
        with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                         suffix='.dat') as f:
            data_file = f.name
            f.write(data)

        try:
            cmd = [
                self.GPG_BINARY,
                '--batch',
                '--no-tty',
                '--digest-algo', digest_algo,
                '--local-user', key_id,
            ]

            env = None
            if self.GPG_HOME:
                env = os.environ.copy()
                env['GNUPGHOME'] = self.GPG_HOME

            if clearsign:
                # Clearsigned output — data + signature combined
                # Always ASCII armored by nature; --armor flag is ignored
                cmd.extend(['--clearsign', '--output', '-', data_file])
            else:
                # Detached signature
                if armor:
                    cmd.append('--armor')
                cmd.extend(['--detach-sign', '--output', '-', data_file])

            result = run(
                cmd,
                stdout=PIPE,
                stderr=PIPE,
                timeout=self.SIGNING_TIMEOUT,
                env=env
            )

            if result.returncode != 0:
                err = result.stderr.decode('utf-8', errors='ignore')
                self.log_message("GPG error (exit %d): %s", result.returncode, err)
                return None

            if len(result.stdout) == 0:
                self.log_message("GPG returned empty output")
                return None

            return result.stdout

        except FileNotFoundError:
            self.log_message("GPG binary not found: %s", self.GPG_BINARY)
            return None
        except TimeoutExpired:
            self.log_message("GPG signing timeout after %d seconds",
                             self.SIGNING_TIMEOUT)
            return None
        except Exception as e:
            self.log_message("Signing error: %s", str(e))
            return None
        finally:
            try:
                os.unlink(data_file)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def validate_key_id(self, key_id):
        import re
        if not key_id or len(key_id) > 256:
            return False
        if not re.match(r'^[a-zA-Z0-9@.\-_ <>]+$', key_id):
            return False
        if '..' in key_id or '/' in key_id or '\\' in key_id:
            return False
        return True

    def _oidc_context(self, payload):
        if not payload:
            return None
        return {
            'repository': payload.get('repository'),
            'ref':        payload.get('ref'),
            'workflow':   payload.get('workflow'),
            'actor':      payload.get('actor'),
            'run_id':     payload.get('run_id'),
            'run_number': payload.get('run_number'),
            'event_name': payload.get('event_name'),
            'job_workflow_ref': payload.get('job_workflow_ref'),
        }

    def send_json_response(self, code, data):
        response = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def send_json_error(self, code, message):
        self.send_json_response(code, {'error': message})

    def log_message(self, format, *args):
        timestamp = self.log_date_time_string()
        sys.stderr.write(f"[{timestamp}] {format % args}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='GPG signing server for RPM/DEB package signing')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--gpg', default='gpg')
    parser.add_argument('--keyring', default='',
        help='Path to GPG keyring directory (GNUPGHOME). '
             'Must exist before starting if not using --secrets-manager-secret.')
    parser.add_argument('--max-request-size', type=int, default=10240)
    parser.add_argument('--read-timeout', type=int, default=10)
    parser.add_argument('--max-threads', type=int, default=10)

    # Phase 1: Secrets Manager key loading
    parser.add_argument('--secrets-manager-secret', action='append',
        dest='sm_secrets', metavar='SECRET_NAME', default=[],
        help='Secrets Manager secret name containing a GPG private key. '
             'Repeatable for multiple keys. Fetched at startup into --keyring.')
    parser.add_argument('--region', default='',
        help='AWS region for Secrets Manager (optional)')

    # Phase 2: Authentication
    parser.add_argument('--enable-auth', action='store_true',
        help='Enable application-layer token authentication (Phase 2)')
    parser.add_argument('--tokens-file', default='',
        help='Path to tokens.json with pre-shared tokens (Phase 2)')
    parser.add_argument('--secrets-file', default='',
        help='Path to secrets.json with JWT shared secrets (Phase 2 fallback)')
    parser.add_argument('--authz-config', default='',
        help='Path to authorization.json with role/key mappings')
    parser.add_argument('--audit-log', default='',
        help='Path to audit log file (optional; always logs to stdout)')

    # TLS
    parser.add_argument('--enable-tls', action='store_true')
    parser.add_argument('--cert-file', default='')
    parser.add_argument('--key-file', default='')

    args = parser.parse_args()

    # --- Validate keyring ---
    if args.keyring:
        if not os.path.isdir(args.keyring):
            print(f"Error: Keyring directory does not exist: {args.keyring}")
            sys.exit(1)
        os.environ['GNUPGHOME'] = os.path.abspath(args.keyring)
    elif args.sm_secrets:
        print("Error: --secrets-manager-secret requires --keyring to be set "
              "(the directory to import keys into)")
        sys.exit(1)

    # --- Validate auth config ---
    if args.enable_auth:
        if not AUTH_AVAILABLE:
            print("Error: auth.py module not found")
            sys.exit(1)
        if not args.authz_config:
            print("Error: --enable-auth requires --authz-config")
            sys.exit(1)

    # --- Validate TLS ---
    if args.enable_tls:
        if not args.cert_file or not args.key_file:
            print("Error: --enable-tls requires --cert-file and --key-file")
            sys.exit(1)
        for f in [args.cert_file, args.key_file]:
            if not os.path.exists(f):
                print(f"Error: TLS file not found: {f}")
                sys.exit(1)

    # --- Set env vars for handlers ---
    os.environ['GPG_BINARY'] = args.gpg
    os.environ['MAX_REQUEST_SIZE'] = str(args.max_request_size)
    os.environ['READ_TIMEOUT'] = str(args.read_timeout)
    os.environ['MAX_THREADS'] = str(args.max_threads)

    if args.enable_auth:
        os.environ['AUTH_ENABLED'] = 'true'
        if args.tokens_file:
            os.environ['TOKENS_FILE'] = os.path.abspath(args.tokens_file)
        if args.secrets_file:
            os.environ['SECRETS_FILE'] = os.path.abspath(args.secrets_file)
        if args.authz_config:
            os.environ['AUTHZ_CONFIG_FILE'] = os.path.abspath(args.authz_config)
    if args.authz_config:
        os.environ['AUTHZ_CONFIG_FILE'] = os.path.abspath(args.authz_config)
    if args.audit_log:
        os.environ['AUDIT_LOG_FILE'] = args.audit_log

    # --- Load keys from Secrets Manager (Phase 1) ---
    global _keyring_ready
    if args.sm_secrets:
        keyring_dir = os.environ.get('GNUPGHOME', '')
        n = load_keys_from_secrets_manager(
            args.sm_secrets, keyring_dir,
            region=args.region or None,
            gpg_binary=args.gpg
        )
        print(f"Loaded {n} key(s) from Secrets Manager")

    # --- Verify keyring ---
    keyring_dir = os.environ.get('GNUPGHOME', '')
    if keyring_dir:
        _keyring_ready = verify_keyring(keyring_dir, args.gpg)
        if not _keyring_ready:
            print("WARNING: No secret keys found in keyring. "
                  "/health will return 503 until keys are loaded.")
    else:
        # Using system default keyring — assume ready
        _keyring_ready = True

    # --- Initialize semaphore ---
    global _signing_semaphore
    _signing_semaphore = threading.Semaphore(args.max_threads)

    # --- Start server ---
    server = ThreadedHTTPServer((args.host, args.port), SigningHandler)

    if args.enable_tls:
        import ssl
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=args.cert_file,
                                    keyfile=args.key_file)
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        server.socket = ssl_context.wrap_socket(server.socket, server_side=True)

    protocol = "https" if args.enable_tls else "http"
    print("=" * 60)
    print("GPG Signing Server")
    print("=" * 60)
    print(f"Listening: {protocol}://{args.host}:{args.port}")
    print(f"Endpoints: POST /sign  GET /health")
    print(f"GPG:       {args.gpg}")
    print(f"Keyring:   {keyring_dir or '(system default)'}")
    print(f"Auth:      {'ENABLED' if args.enable_auth else 'DISABLED (Phase 1 — VPC Security Groups)'}")
    print(f"TLS:       {'ENABLED' if args.enable_tls else 'DISABLED'}")
    print(f"Threads:   {args.max_threads}")
    print(f"Max req:   {args.max_request_size} bytes")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
