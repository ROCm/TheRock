#!/usr/bin/env python3
"""
Authentication and authorization module for GPG signing server.

This module provides dual authentication support:
1. JWT (HMAC-SHA256) - Shared secret tokens stored in GitHub Secrets
2. OIDC (RS256) - GitHub-issued tokens (keyless, short-lived)

Token formats:
- JWT: HMAC-SHA256 signed tokens with role-based access
- OIDC: RS256 signed tokens from GitHub Actions OIDC provider

Authorization: Role-based key access control with workflow restriction
Rate limiting: Per-client request throttling
Audit logging: JSON-based request logging
"""

import json
import base64
import hmac
import hashlib
import time
import os
import urllib.request
import urllib.error
from collections import deque
from datetime import datetime, timedelta

# OIDC support - try to import PyJWT, fall back to JWT-only mode if unavailable
try:
    import jwt
    from jwt import PyJWKClient
    OIDC_AVAILABLE = True
except ImportError:
    OIDC_AVAILABLE = False
    # Provide stub for type hints
    jwt = None
    PyJWKClient = None


class AuthError(Exception):
    """Authentication/authorization error."""
    pass


def generate_jwt_token(client_id, role, secret, expires_hours=4):
    """
    Generate HMAC-SHA256 signed JWT token.

    Args:
        client_id: Unique identifier for the client
        role: Role name (e.g., 'production', 'development')
        secret: Shared secret for HMAC signature (string or bytes)
        expires_hours: Token validity period in hours (default: 4)

    Returns:
        JWT token string (base64url encoded)

    Example:
        token = generate_jwt_token('github-actions-prod', 'production', 'secret123', 24)
    """
    # JWT header (HS256 algorithm)
    header = {
        'alg': 'HS256',
        'typ': 'JWT'
    }

    # JWT payload
    now = int(time.time())
    expires = now + (expires_hours * 3600)

    payload = {
        'client_id': client_id,
        'role': role,
        'iat': now,  # issued at
        'exp': expires  # expiration
    }

    # Encode header and payload
    header_b64 = _base64url_encode(json.dumps(header).encode('utf-8'))
    payload_b64 = _base64url_encode(json.dumps(payload).encode('utf-8'))

    # Create signature
    message = f"{header_b64}.{payload_b64}"

    if isinstance(secret, str):
        secret = secret.encode('utf-8')

    signature = hmac.new(secret, message.encode('utf-8'), hashlib.sha256).digest()
    signature_b64 = _base64url_encode(signature)

    # Combine into JWT
    token = f"{header_b64}.{payload_b64}.{signature_b64}"
    return token


def validate_jwt_token(token, secrets_map):
    """
    Validate JWT token and return decoded payload.

    Args:
        token: JWT token string
        secrets_map: Dict mapping client_id -> secret (or dict with 'secret' key)

    Returns:
        Decoded payload dict if valid, None otherwise

    Example:
        secrets = {'github-actions-prod': {'secret': 'secret123'}}
        payload = validate_jwt_token(token, secrets)
        if payload:
            print("Role:", payload['role'])
    """
    if not token or not isinstance(token, str):
        return None

    # Split token into parts
    parts = token.split('.')
    if len(parts) != 3:
        return None

    header_b64, payload_b64, signature_b64 = parts

    try:
        # Decode payload to extract client_id
        payload_json = _base64url_decode(payload_b64)
        payload = json.loads(payload_json)

        client_id = payload.get('client_id')
        if not client_id or client_id not in secrets_map:
            return None

        # Get secret for this client
        client_secret = secrets_map[client_id]
        if isinstance(client_secret, dict):
            secret = client_secret.get('secret', '')
        else:
            secret = client_secret

        if isinstance(secret, str):
            secret = secret.encode('utf-8')

        # Verify signature
        message = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(secret, message.encode('utf-8'), hashlib.sha256).digest()
        expected_sig_b64 = _base64url_encode(expected_sig)

        if not hmac.compare_digest(signature_b64, expected_sig_b64):
            return None

        # Check expiration
        exp = payload.get('exp', 0)
        if exp < time.time():
            return None

        return payload

    except (json.JSONDecodeError, ValueError, KeyError):
        return None


# GitHub OIDC configuration
GITHUB_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
GITHUB_OIDC_JWKS_URL = f"{GITHUB_OIDC_ISSUER}/.well-known/jwks"

# Global JWKS client (cached)
_jwks_client = None


def validate_github_oidc_token(token, audience="amd-signing-service"):
    """
    Validate GitHub OIDC token.

    GitHub Actions OIDC tokens are RS256-signed JWT tokens that contain rich
    metadata about the workflow execution (repository, branch, actor, etc.).

    Args:
        token: OIDC token string from GitHub Actions
        audience: Expected audience claim (default: 'amd-signing-service')

    Returns:
        Decoded payload dict if valid, None otherwise

    Raises:
        RuntimeError: If OIDC support is not available (PyJWT not installed)

    Example:
        payload = validate_github_oidc_token(oidc_token)
        if payload:
            print("Repository:", payload['repository'])
            print("Branch:", payload['ref'])
            print("Workflow:", payload['workflow'])
    """
    if not OIDC_AVAILABLE:
        raise RuntimeError(
            "OIDC support requires PyJWT library. "
            "Install with: pip install PyJWT[crypto]"
        )

    if not token or not isinstance(token, str):
        return None

    global _jwks_client

    try:
        # Initialize JWKS client (lazy, cached)
        if _jwks_client is None:
            _jwks_client = PyJWKClient(GITHUB_OIDC_JWKS_URL, cache_keys=True)

        # Get signing key from GitHub's JWKS
        signing_key = _jwks_client.get_signing_key_from_jwt(token)

        # Verify and decode token
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=GITHUB_OIDC_ISSUER,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "verify_aud": True,
                "verify_iss": True,
            }
        )

        return payload

    except jwt.InvalidTokenError as e:
        # Token validation failed (expired, wrong signature, etc.)
        import sys
        sys.stderr.write(f"OIDC token validation failed: {str(e)}\n")
        return None
    except Exception as e:
        # Unexpected error (network issue fetching JWKS, etc.)
        import sys
        sys.stderr.write(f"OIDC validation error: {str(e)}\n")
        return None


def determine_role_from_oidc(payload, authz_config):
    """
    Determine role from OIDC token payload based on branch/ref.

    Maps GitHub Actions context to signing roles:
    - refs/heads/main or refs/heads/release/* → 'release' role
    - refs/heads/* (other branches) → 'development' role
    - refs/pull/* → 'development' role

    Args:
        payload: OIDC token payload (already validated)
        authz_config: Authorization configuration (optional, for custom mapping)

    Returns:
        Role name (string) or None if cannot be determined

    Example:
        role = determine_role_from_oidc(oidc_payload, config)
        # role = 'release' or 'development'
    """
    ref = payload.get('ref', '')

    # Check for custom role mapping in config
    role_mapping = authz_config.get('oidc_role_mapping', {})
    for pattern, role in role_mapping.items():
        if _match_ref_pattern(ref, pattern):
            return role

    # Default role mapping
    if ref == 'refs/heads/main':
        return 'release'
    elif ref.startswith('refs/heads/release/'):
        return 'release'
    elif ref.startswith('refs/heads/'):
        return 'development'
    elif ref.startswith('refs/pull/'):
        return 'development'
    else:
        return None


def authorize_oidc_request(payload, key_id, digest_algo, authz_config):
    """
    Authorize OIDC token request with workflow restriction checks.

    Validates:
    1. Repository is in allowed list
    2. Branch/ref matches allowed patterns
    3. Workflow is in allowed list
    4. Key and digest algorithm are authorized for the role

    Args:
        payload: OIDC token payload (already validated)
        key_id: Requested signing key ID
        digest_algo: Requested digest algorithm
        authz_config: Authorization configuration dict

    Returns:
        Tuple (role: str, authorized: bool, reason: str)

    Example:
        role, authorized, reason = authorize_oidc_request(payload, 'prod-key', 'SHA256', config)
        if not authorized:
            return 403, reason
    """
    # Extract OIDC claims
    repository = payload.get('repository')
    ref = payload.get('ref')
    workflow = payload.get('workflow')
    actor = payload.get('actor')

    # Determine role from ref
    role = determine_role_from_oidc(payload, authz_config)
    if not role:
        return None, False, f"Cannot determine role from ref: {ref}"

    roles = authz_config.get('roles', {})
    if role not in roles:
        return role, False, f"Unknown role: {role}"

    role_config = roles[role]

    # Check repository restriction
    allowed_repos = role_config.get('allowed_repositories', [])
    if allowed_repos and repository not in allowed_repos:
        return role, False, f"Repository '{repository}' not authorized for role '{role}'"

    # Check branch/ref restriction
    allowed_refs = role_config.get('allowed_refs', [])
    if allowed_refs:
        if not any(_match_ref_pattern(ref, pattern) for pattern in allowed_refs):
            return role, False, f"Branch '{ref}' not authorized for role '{role}'"

    # Check workflow restriction
    allowed_workflows = role_config.get('allowed_workflows', [])
    if allowed_workflows and workflow not in allowed_workflows:
        return role, False, f"Workflow '{workflow}' not authorized for role '{role}'"

    # Check allowed keys
    allowed_keys = role_config.get('allowed_keys', [])
    if allowed_keys and key_id not in allowed_keys:
        return role, False, f"Role '{role}' not authorized for key '{key_id}'"

    # Check allowed digest algorithms
    allowed_algos = role_config.get('allowed_digest_algos', [])
    if allowed_algos and digest_algo not in allowed_algos:
        return role, False, f"Role '{role}' not authorized for digest algorithm '{digest_algo}'"

    return role, True, "Authorized"


def _match_ref_pattern(ref, pattern):
    """
    Match git ref against pattern (supports wildcards).

    Examples:
        _match_ref_pattern('refs/heads/main', 'refs/heads/main') → True
        _match_ref_pattern('refs/heads/feature-1', 'refs/heads/*') → True
        _match_ref_pattern('refs/pull/123/merge', 'refs/pull/*') → True
    """
    import fnmatch
    return fnmatch.fnmatch(ref, pattern)


def load_secrets(secrets_file):
    """
    Load secrets from JSON file.

    Args:
        secrets_file: Path to secrets.json file

    Returns:
        Dict mapping client_id -> secret configuration

    Format:
        {
            "github-actions-prod": {
                "secret": "base64-encoded-secret",
                "description": "Production GitHub Actions",
                "created": "2026-03-02"
            }
        }
    """
    if not secrets_file or not os.path.exists(secrets_file):
        return {}

    try:
        with open(secrets_file, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {}


def load_authorization_config(authz_file):
    """
    Load authorization configuration from JSON file.

    Args:
        authz_file: Path to authorization.json file

    Returns:
        Authorization configuration dict

    Format:
        {
            "roles": {
                "production": {
                    "allowed_keys": ["prod-key@company.com"],
                    "allowed_digest_algos": ["SHA256", "SHA512"],
                    "max_requests_per_hour": 1000
                }
            }
        }
    """
    if not authz_file or not os.path.exists(authz_file):
        return {'roles': {}}

    try:
        with open(authz_file, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {'roles': {}}


def authorize_request(role, key_id, digest_algo, authz_config):
    """
    Check if role is authorized to use the specified key and digest algorithm.

    Args:
        role: Role name from JWT token
        key_id: Requested signing key ID
        digest_algo: Requested digest algorithm (e.g., 'SHA256')
        authz_config: Authorization configuration dict

    Returns:
        Tuple (authorized: bool, reason: str)

    Example:
        authorized, reason = authorize_request('production', 'prod-key', 'SHA256', config)
        if not authorized:
            return 403, reason
    """
    roles = authz_config.get('roles', {})

    if role not in roles:
        return False, f"Unknown role: {role}"

    role_config = roles[role]

    # Check allowed keys
    allowed_keys = role_config.get('allowed_keys', [])
    if allowed_keys and key_id not in allowed_keys:
        return False, f"Role '{role}' not authorized for key '{key_id}'"

    # Check allowed digest algorithms
    allowed_algos = role_config.get('allowed_digest_algos', [])
    if allowed_algos and digest_algo not in allowed_algos:
        return False, f"Role '{role}' not authorized for digest algorithm '{digest_algo}'"

    return True, "Authorized"


def check_rate_limit(client_id, role, rate_limits, authz_config):
    """
    Check if client has exceeded rate limit.

    Uses a rolling window counter with collections.deque to track requests.

    Args:
        client_id: Client identifier
        role: Role name (used to lookup limit)
        rate_limits: Dict mapping client_id -> deque of timestamps
        authz_config: Authorization config (contains max_requests_per_hour)

    Returns:
        True if under limit, False if exceeded

    Note:
        Modifies rate_limits dict in place (stores timestamps)
    """
    # Get rate limit for this role
    roles = authz_config.get('roles', {})
    if role not in roles:
        return True  # No limit if role not found

    max_requests = roles[role].get('max_requests_per_hour', 0)
    if max_requests <= 0:
        return True  # No limit configured

    # Initialize deque for this client if needed
    if client_id not in rate_limits:
        rate_limits[client_id] = deque()

    # Remove timestamps older than 1 hour
    now = time.time()
    one_hour_ago = now - 3600

    timestamps = rate_limits[client_id]
    while timestamps and timestamps[0] < one_hour_ago:
        timestamps.popleft()

    # Check if limit exceeded
    if len(timestamps) >= max_requests:
        return False

    # Add current timestamp
    timestamps.append(now)
    return True


def audit_log(action, client_id, role, key_id, digest_algo, client_ip, success, audit_file, oidc_context=None):
    """
    Write audit log entry.

    Logs are written as JSON lines (one JSON object per line) for easy parsing.

    Args:
        action: Action type (e.g., 'SIGNED', 'DENIED', 'AUTH_FAILED', 'RATE_LIMITED')
        client_id: Client identifier (or 'oidc' for OIDC tokens)
        role: Role name
        key_id: Signing key ID
        digest_algo: Digest algorithm used
        client_ip: Client IP address
        success: True if action succeeded
        audit_file: Path to audit log file
        oidc_context: Optional dict with OIDC token fields (repository, ref, workflow, actor, etc.)

    Note:
        Does not raise exceptions - logs failures to stderr instead
    """
    if not audit_file:
        return

    # Create log entry
    entry = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'action': action,
        'client_id': client_id,
        'role': role,
        'key_id': key_id,
        'digest_algo': digest_algo,
        'client_ip': client_ip,
        'success': success
    }

    # Add OIDC-specific context if available
    if oidc_context:
        entry['auth_type'] = 'oidc'
        entry['repository'] = oidc_context.get('repository')
        entry['ref'] = oidc_context.get('ref')
        entry['workflow'] = oidc_context.get('workflow')
        entry['actor'] = oidc_context.get('actor')
        entry['run_id'] = oidc_context.get('run_id')
        entry['run_number'] = oidc_context.get('run_number')
        entry['event_name'] = oidc_context.get('event_name')
        entry['job_workflow_ref'] = oidc_context.get('job_workflow_ref')
    else:
        entry['auth_type'] = 'jwt'

    try:
        # Create directory if needed
        log_dir = os.path.dirname(audit_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, mode=0o755)

        # Append to log file
        with open(audit_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except (IOError, OSError) as e:
        # Don't fail the request if logging fails
        import sys
        sys.stderr.write(f"Audit log error: {str(e)}\n")


# Internal helper functions

def _base64url_encode(data):
    """
    Base64url encode (URL-safe base64 without padding).

    JWT spec requires base64url encoding, which replaces + with - and / with _,
    and removes padding (=).
    """
    if isinstance(data, str):
        data = data.encode('utf-8')

    encoded = base64.urlsafe_b64encode(data).decode('ascii')
    # Remove padding
    return encoded.rstrip('=')


def _base64url_decode(data):
    """
    Base64url decode (URL-safe base64 without padding).
    """
    # Add padding if needed
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += '=' * padding

    decoded = base64.urlsafe_b64decode(data)
    return decoded.decode('utf-8')
