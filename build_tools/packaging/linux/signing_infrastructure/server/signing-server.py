#!/usr/bin/env python3
"""
Production signing server for RPM signing.

This server receives digest (hash) values and signs them using GPG.
It never receives the full RPM file, only the pre-computed digest.

For a 1GB RPM file:
- Client sends: 32 bytes (SHA256 digest)
- Server returns: ~256 bytes (signature)

Total network transfer: ~300 bytes instead of 1GB+

Architecture:
- Multi-threaded server using ThreadingMixIn
- Configurable max concurrent threads (default: 10)
- Each signing request handled in separate thread
- Thread semaphore prevents resource exhaustion
"""

# Global semaphore for limiting concurrent signing operations
_signing_semaphore = None


def get_signing_semaphore():
    """Get or create the global signing semaphore."""
    global _signing_semaphore
    if _signing_semaphore is None:
        max_threads = int(os.environ.get('MAX_THREADS', '10'))
        _signing_semaphore = threading.Semaphore(max_threads)
    return _signing_semaphore

import json
import sys
import os
import tempfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from base64 import b64decode, b64encode
from subprocess import run, PIPE, TimeoutExpired
import hashlib

# Import authentication module (if auth is enabled)
try:
    from auth import (
        validate_jwt_token, load_secrets, load_authorization_config,
        authorize_request, check_rate_limit, audit_log
    )
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server with thread limits."""
    # Allow reuse of address to avoid "Address already in use" errors
    allow_reuse_address = True
    # Daemon threads so server can shut down cleanly
    daemon_threads = True


class SigningHandler(BaseHTTPRequestHandler):
    """HTTP request handler for signing operations."""

    # Class-level caches (shared across all handler instances)
    _secrets_cache = None
    _authz_cache = None
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
    def AUDIT_LOG_FILE(self):
        return os.environ.get('AUDIT_LOG_FILE', '/var/log/gpg-signing/audit.log')

    def authenticate_request(self):
        """
        Extract and validate JWT token from Authorization header.

        Returns:
            Decoded token payload dict if valid, None otherwise
        """
        if not self.AUTH_ENABLED or not AUTH_AVAILABLE:
            return None

        auth_header = self.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return None

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Load secrets (cached)
        if self._secrets_cache is None:
            self.__class__._secrets_cache = load_secrets(self.SECRETS_FILE)

        # Validate token
        payload = validate_jwt_token(token, self._secrets_cache)
        return payload

    def do_POST(self):
        """Handle POST requests."""
        # Handle shutdown endpoint
        if self.path == '/quit':
            self.send_json_response(200, {'message': 'Server shutting down'})
            self.log_message("Shutdown requested via /quit endpoint")
            # Schedule shutdown
            import threading
            def shutdown():
                import time
                time.sleep(0.5)
                self.server.shutdown()
            threading.Thread(target=shutdown).start()
            return

        # Handle signing endpoint
        if self.path != '/sign':
            self.send_error(404, "Not Found")
            return

        try:
            # Read and parse request with size and timeout limits
            content_length = int(self.headers.get('Content-Length', 0))

            # Log request size
            self.log_message("Request size: %d bytes (limit: %d bytes)",
                           content_length, self.MAX_REQUEST_SIZE)

            # Enforce size limit (defense against DoS)
            if content_length > self.MAX_REQUEST_SIZE:
                self.log_message("Request rejected: size %d exceeds limit %d",
                               content_length, self.MAX_REQUEST_SIZE)
                self.send_json_error(413,
                    f"Request too large: {content_length} bytes (max {self.MAX_REQUEST_SIZE} bytes)")
                return

            # Set socket timeout to prevent slow-read attacks
            original_timeout = self.connection.gettimeout()
            try:
                self.connection.settimeout(self.READ_TIMEOUT)
                body = self.rfile.read(content_length)
            except Exception as e:
                self.log_message("Read timeout or error: %s", str(e))
                self.send_json_error(408, "Request timeout")
                return
            finally:
                # Restore original timeout
                if original_timeout is not None:
                    self.connection.settimeout(original_timeout)
            request = json.loads(body.decode('utf-8'))

            # AUTHENTICATION: Validate token if auth is enabled
            payload = None
            if self.AUTH_ENABLED:
                if not AUTH_AVAILABLE:
                    self.send_json_error(500, "Authentication enabled but auth module not available")
                    return

                payload = self.authenticate_request()
                if not payload:
                    self.send_json_error(401, "Unauthorized: Invalid or missing token")
                    audit_log('AUTH_FAILED', 'unknown', 'none', '', '',
                             self.client_address[0], False, self.AUDIT_LOG_FILE)
                    return

            # Validate request
            data_b64 = request.get('data')
            if not data_b64:
                self.send_json_error(400, "Missing 'data' field")
                return

            # Extract parameters
            key_id = request.get('key_id', '')
            digest_algo = request.get('digest_algo', 'SHA256').upper()
            armor = request.get('armor', False)

            # Validate key_id is provided
            if not key_id:
                self.send_json_error(400, "key_id is required")
                return

            # Validate key_id to prevent injection attacks
            # Allow: alphanumeric, @, ., -, _, (email addresses and hex key IDs)
            if not self.validate_key_id(key_id):
                self.send_json_error(400,
                    "Invalid key_id format: must be alphanumeric, @, ., -, or _")
                return

            # Decode data
            try:
                data = b64decode(data_b64)
            except Exception as e:
                self.send_json_error(400, f"Invalid base64 in 'data': {str(e)}")
                return

            # Validate data is not empty
            if len(data) == 0:
                self.send_json_error(400, "Data cannot be empty")
                return

            # Log the signing request
            self.log_message("Signing request: key=%s, algo=%s, data_len=%d",
                           key_id or '(default)', digest_algo, len(data))

            # AUTHORIZATION: Check if client is authorized for this key/algo
            if self.AUTH_ENABLED and payload:
                # Load authorization config (cached)
                if self._authz_cache is None:
                    self.__class__._authz_cache = load_authorization_config(self.AUTHZ_CONFIG_FILE)

                role = payload.get('role', '')
                client_id = payload.get('client_id', '')

                # Check rate limit
                if not check_rate_limit(client_id, role, self._rate_limits, self._authz_cache):
                    self.log_message("Rate limit exceeded for client %s (role: %s)", client_id, role)
                    self.send_json_error(429, "Rate limit exceeded")
                    audit_log('RATE_LIMITED', client_id, role, key_id, digest_algo,
                             self.client_address[0], False, self.AUDIT_LOG_FILE)
                    return

                # Check authorization for key and digest algorithm
                authorized, reason = authorize_request(role, key_id, digest_algo, self._authz_cache)
                if not authorized:
                    self.log_message("Authorization denied for client %s (role: %s): %s",
                                   client_id, role, reason)
                    self.send_json_error(403, f"Forbidden: {reason}")
                    audit_log('DENIED', client_id, role, key_id, digest_algo,
                             self.client_address[0], False, self.AUDIT_LOG_FILE)
                    return

            # Acquire semaphore to limit concurrent signing operations
            semaphore = get_signing_semaphore()
            acquired = semaphore.acquire(blocking=True, timeout=30)
            if not acquired:
                self.log_message("Failed to acquire signing semaphore (server busy)")
                self.send_json_error(503, "Server busy, try again later")
                return

            try:
                # Perform signing
                signature = self.sign_data(data, key_id, digest_algo, armor)
            finally:
                # Always release semaphore
                semaphore.release()

            if signature is None:
                self.send_json_error(500, "Signing failed")
                return

            # Return response
            response = {
                'signature': b64encode(signature).decode('ascii'),
                'key_id': key_id,
                'digest_algo': digest_algo
            }

            self.send_json_response(200, response)
            self.log_message("Signing successful")

            # AUDIT LOG: Log successful signing if auth is enabled
            if self.AUTH_ENABLED and payload:
                audit_log('SIGNED', payload.get('client_id', 'unknown'),
                         payload.get('role', 'unknown'),
                         key_id, digest_algo, self.client_address[0], True,
                         self.AUDIT_LOG_FILE)

        except json.JSONDecodeError as e:
            self.send_json_error(400, f"Invalid JSON: {str(e)}")
        except Exception as e:
            self.log_message("Error: %s", str(e))
            self.send_json_error(500, f"Internal error: {str(e)}")

    def validate_key_id(self, key_id):
        """
        Validate key_id to prevent directory traversal and command injection.

        Valid key IDs:
        - Hex key IDs: 44D107C408145F7F
        - Email addresses: signer@example.com
        - Names with spaces: "Test Signer"

        Invalid:
        - Path traversal: ../../../etc/passwd
        - Command injection: key; rm -rf /
        """
        import re

        # Must not be empty or too long
        if not key_id or len(key_id) > 256:
            return False

        # Allow alphanumeric, @, ., -, _, space, <, > (for email format)
        # Disallow: / \ ; & | $ ` ' " ( ) [ ] { } and other shell metacharacters
        if not re.match(r'^[a-zA-Z0-9@.\-_ <>]+$', key_id):
            return False

        # Specifically block directory traversal attempts
        if '..' in key_id or '/' in key_id or '\\' in key_id:
            return False

        return True

    def sign_data(self, data, key_id, digest_algo, armor):
        """
        Sign data using GPG.

        GPG will compute the digest and sign it.
        """
        # Create a temporary file for the data
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.dat') as f:
            data_file = f.name
            f.write(data)

        try:
            # Build GPG command for signing data
            # GPG will compute the digest using the specified algorithm and sign it
            cmd = [
                self.GPG_BINARY,
                '--batch',
                '--no-tty',
                '--digest-algo', digest_algo,
            ]

            # Set GPG home if configured (for keyring isolation)
            env = None
            if self.GPG_HOME:
                env = os.environ.copy()
                env['GNUPGHOME'] = self.GPG_HOME

            # Always use the specified key_id (required parameter)
            cmd.extend(['--local-user', key_id])

            if armor:
                cmd.append('--armor')

            # Sign the data
            # We create a detached signature of the data file
            cmd.extend(['--detach-sign', '--output', '-', data_file])

            # Run GPG
            result = run(
                cmd,
                stdout=PIPE,
                stderr=PIPE,
                timeout=self.SIGNING_TIMEOUT,
                env=env
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                self.log_message("GPG error (exit %d): %s", result.returncode, error_msg)
                return None

            signature = result.stdout

            # Verify we got a signature
            if len(signature) == 0:
                self.log_message("GPG returned empty signature")
                return None

            return signature

        except FileNotFoundError:
            self.log_message("GPG binary not found: %s", self.GPG_BINARY)
            return None
        except TimeoutExpired:
            self.log_message("GPG signing timeout after %d seconds", self.SIGNING_TIMEOUT)
            return None
        except Exception as e:
            self.log_message("Signing error: %s", str(e))
            return None
        finally:
            # Clean up temporary file
            try:
                os.unlink(data_file)
            except:
                pass

    def send_json_response(self, code, data):
        """Send JSON response."""
        response = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def send_json_error(self, code, message):
        """Send JSON error response."""
        self.send_json_response(code, {'error': message})

    def log_message(self, format, *args):
        """Override to customize logging."""
        timestamp = self.log_date_time_string()
        message = format % args
        sys.stderr.write(f"[{timestamp}] {message}\n")


def main():
    """Run the signing server."""
    import argparse

    parser = argparse.ArgumentParser(description='GPG signing server for RPM signing')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port to listen on (default: 8080)')
    parser.add_argument('--host', default='localhost',
                       help='Host to bind to (default: localhost)')
    parser.add_argument('--gpg', default='gpg',
                       help='Path to GPG binary (default: gpg)')
    parser.add_argument('--keyring', default='',
                       help='Path to GPG keyring directory (GNUPGHOME). If not specified, uses system default.')
    parser.add_argument('--max-request-size', type=int, default=10240,
                       help='Maximum request size in bytes (default: 10240 = 10KB)')
    parser.add_argument('--read-timeout', type=int, default=10,
                       help='Socket read timeout in seconds (default: 10)')
    parser.add_argument('--max-threads', type=int, default=10,
                       help='Maximum concurrent signing threads (default: 10)')

    # Authentication arguments
    parser.add_argument('--enable-auth', action='store_true',
                       help='Enable authentication (requires --secrets-file and --authz-config)')
    parser.add_argument('--secrets-file', default='',
                       help='Path to secrets.json file containing client credentials')
    parser.add_argument('--authz-config', default='',
                       help='Path to authorization.json file with role definitions')
    parser.add_argument('--audit-log', default='/var/log/gpg-signing/audit.log',
                       help='Path to audit log file (default: /var/log/gpg-signing/audit.log)')

    # TLS arguments
    parser.add_argument('--enable-tls', action='store_true',
                       help='Enable TLS/HTTPS (requires --cert-file and --key-file)')
    parser.add_argument('--cert-file', default='',
                       help='Path to TLS certificate file (PEM format)')
    parser.add_argument('--key-file', default='',
                       help='Path to TLS private key file (PEM format)')

    args = parser.parse_args()

    # Validate authentication configuration
    if args.enable_auth:
        if not AUTH_AVAILABLE:
            print("Error: Authentication enabled but auth.py module not found")
            print("Make sure auth.py is in the same directory as this script")
            sys.exit(1)
        if not args.secrets_file:
            print("Error: --enable-auth requires --secrets-file")
            sys.exit(1)
        if not args.authz_config:
            print("Error: --enable-auth requires --authz-config")
            sys.exit(1)
        if not os.path.exists(args.secrets_file):
            print(f"Error: Secrets file not found: {args.secrets_file}")
            sys.exit(1)
        if not os.path.exists(args.authz_config):
            print(f"Error: Authorization config file not found: {args.authz_config}")
            sys.exit(1)

    # Validate TLS configuration
    if args.enable_tls:
        if not args.cert_file or not args.key_file:
            print("Error: --enable-tls requires both --cert-file and --key-file")
            sys.exit(1)
        if not os.path.exists(args.cert_file):
            print(f"Error: Certificate file not found: {args.cert_file}")
            sys.exit(1)
        if not os.path.exists(args.key_file):
            print(f"Error: Private key file not found: {args.key_file}")
            sys.exit(1)

    # Set configuration via environment for handler
    os.environ['GPG_BINARY'] = args.gpg
    if args.keyring:
        # Validate keyring directory exists
        if not os.path.isdir(args.keyring):
            print(f"Error: Keyring directory does not exist: {args.keyring}")
            sys.exit(1)
        os.environ['GNUPGHOME'] = os.path.abspath(args.keyring)
    os.environ['MAX_REQUEST_SIZE'] = str(args.max_request_size)
    os.environ['READ_TIMEOUT'] = str(args.read_timeout)
    os.environ['MAX_THREADS'] = str(args.max_threads)

    # Set authentication configuration
    if args.enable_auth:
        os.environ['AUTH_ENABLED'] = 'true'
        os.environ['SECRETS_FILE'] = os.path.abspath(args.secrets_file)
        os.environ['AUTHZ_CONFIG_FILE'] = os.path.abspath(args.authz_config)
        os.environ['AUDIT_LOG_FILE'] = args.audit_log

    # Initialize the signing semaphore
    global _signing_semaphore
    _signing_semaphore = threading.Semaphore(args.max_threads)

    server = ThreadedHTTPServer((args.host, args.port), SigningHandler)

    # Enable TLS if requested
    if args.enable_tls:
        import ssl
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=args.cert_file, keyfile=args.key_file)
        # Security: require TLS 1.2 or higher
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        server.socket = ssl_context.wrap_socket(server.socket, server_side=True)

    print("=" * 60)
    print("GPG Signing Server (Multi-threaded)")
    print("=" * 60)
    protocol = "https" if args.enable_tls else "http"
    print(f"Listening on: {protocol}://{args.host}:{args.port}")
    print("Endpoint: POST /sign")
    print(f"GPG binary: {args.gpg}")
    if args.keyring:
        print(f"Keyring: {args.keyring}")
        # List available keys
        try:
            list_result = run(
                [args.gpg, '--list-secret-keys', '--keyid-format', 'LONG'],
                stdout=PIPE,
                stderr=PIPE,
                env={'GNUPGHOME': os.environ.get('GNUPGHOME', '')} if args.keyring else None
            )
            if list_result.returncode == 0:
                output = list_result.stdout.decode('utf-8', errors='ignore')
                # Count keys
                key_count = output.count('sec ')
                print(f"Available keys: {key_count}")
            else:
                print("Available keys: (unable to list)")
        except:
            print("Available keys: (error listing)")
    else:
        print("Keyring: (using system default)")
    print("-" * 60)
    print("Security:")
    print(f"  Authentication: {'ENABLED' if args.enable_auth else 'DISABLED'}")
    if args.enable_auth:
        print(f"    Secrets file: {args.secrets_file}")
        print(f"    Authorization config: {args.authz_config}")
        print(f"    Audit log: {args.audit_log}")
    print(f"  TLS/HTTPS: {'ENABLED' if args.enable_tls else 'DISABLED'}")
    if args.enable_tls:
        print(f"    Certificate: {args.cert_file}")
        print("    Min TLS version: 1.2")
    print(f"  Max request size: {args.max_request_size} bytes")
    print(f"  Read timeout: {args.read_timeout} seconds")
    print(f"  Max concurrent threads: {args.max_threads}")
    print("-" * 60)
    print("Network optimization: Only digests are transferred, not full files")
    print("Example: 1GB RPM = 32 byte digest + ~256 byte signature")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
