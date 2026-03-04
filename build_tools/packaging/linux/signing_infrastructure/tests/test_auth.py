#!/usr/bin/env python3
"""
Unit tests for auth.py module.

Run with: python3 test_auth.py
"""

import sys
import os
import json
import time
import tempfile
from auth import (
    generate_jwt_token,
    validate_jwt_token,
    authorize_request,
    check_rate_limit,
    load_secrets,
    load_authorization_config,
    audit_log
)


def test_token_generation_and_validation():
    """Test JWT token generation and validation."""
    print("Testing token generation and validation...")

    secret = "test-secret-123"
    client_id = "test-client"
    role = "production"

    # Generate token
    token = generate_jwt_token(client_id, role, secret, expires_hours=1)
    assert token is not None
    assert len(token.split('.')) == 3

    # Validate token
    secrets_map = {client_id: secret}
    payload = validate_jwt_token(token, secrets_map)

    assert payload is not None
    assert payload['client_id'] == client_id
    assert payload['role'] == role
    assert 'iat' in payload
    assert 'exp' in payload

    print("  ✓ Token generation and validation works")


def test_token_expiration():
    """Test that expired tokens are rejected."""
    print("Testing token expiration...")

    secret = "test-secret-123"
    client_id = "test-client"

    # Generate token that expires in -1 hours (already expired)
    token = generate_jwt_token(client_id, "production", secret, expires_hours=-1)

    secrets_map = {client_id: secret}
    payload = validate_jwt_token(token, secrets_map)

    assert payload is None, "Expired token should be rejected"

    print("  ✓ Expired tokens are rejected")


def test_token_invalid_signature():
    """Test that tokens with invalid signatures are rejected."""
    print("Testing invalid signature rejection...")

    secret = "test-secret-123"
    wrong_secret = "wrong-secret-456"
    client_id = "test-client"

    # Generate token with one secret
    token = generate_jwt_token(client_id, "production", secret)

    # Try to validate with different secret
    secrets_map = {client_id: wrong_secret}
    payload = validate_jwt_token(token, secrets_map)

    assert payload is None, "Token with invalid signature should be rejected"

    print("  ✓ Invalid signatures are rejected")


def test_authorization():
    """Test role-based authorization."""
    print("Testing authorization...")

    authz_config = {
        'roles': {
            'production': {
                'allowed_keys': ['prod-key@company.com'],
                'allowed_digest_algos': ['SHA256', 'SHA512']
            },
            'development': {
                'allowed_keys': ['dev-key@company.com'],
                'allowed_digest_algos': ['SHA256']
            }
        }
    }

    # Test authorized request
    authorized, reason = authorize_request('production', 'prod-key@company.com', 'SHA256', authz_config)
    assert authorized is True, "Should authorize production role with prod key"

    # Test unauthorized key
    authorized, reason = authorize_request('production', 'dev-key@company.com', 'SHA256', authz_config)
    assert authorized is False, "Should deny production role using dev key"
    assert 'not authorized for key' in reason

    # Test unauthorized algorithm
    authorized, reason = authorize_request('development', 'dev-key@company.com', 'SHA512', authz_config)
    assert authorized is False, "Should deny dev role using SHA512"
    assert 'digest algorithm' in reason

    # Test unknown role
    authorized, reason = authorize_request('unknown', 'any-key', 'SHA256', authz_config)
    assert authorized is False, "Should deny unknown role"
    assert 'Unknown role' in reason

    print("  ✓ Authorization checks work correctly")


def test_rate_limiting():
    """Test rate limiting functionality."""
    print("Testing rate limiting...")

    authz_config = {
        'roles': {
            'production': {
                'allowed_keys': ['prod-key'],
                'max_requests_per_hour': 5  # Very low limit for testing
            }
        }
    }

    rate_limits = {}
    client_id = "test-client"
    role = "production"

    # Make requests up to the limit
    for i in range(5):
        allowed = check_rate_limit(client_id, role, rate_limits, authz_config)
        assert allowed is True, "Requests under limit should be allowed"

    # Next request should be denied
    allowed = check_rate_limit(client_id, role, rate_limits, authz_config)
    assert allowed is False, "Request over limit should be denied"

    print("  ✓ Rate limiting works correctly")


def test_config_loading():
    """Test loading secrets and authorization configs."""
    print("Testing config file loading...")

    # Create temporary secrets file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        secrets_data = {
            'client1': {'secret': 'secret1', 'description': 'Test client'},
            'client2': {'secret': 'secret2'}
        }
        json.dump(secrets_data, f)
        secrets_file = f.name

    # Create temporary authz config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        authz_data = {
            'roles': {
                'production': {
                    'allowed_keys': ['prod-key']
                }
            }
        }
        json.dump(authz_data, f)
        authz_file = f.name

    try:
        # Load secrets
        secrets = load_secrets(secrets_file)
        assert 'client1' in secrets
        assert secrets['client1']['secret'] == 'secret1'

        # Load authz config
        authz = load_authorization_config(authz_file)
        assert 'roles' in authz
        assert 'production' in authz['roles']

        print("  ✓ Config file loading works")

    finally:
        # Cleanup
        os.unlink(secrets_file)
        os.unlink(authz_file)


def test_audit_logging():
    """Test audit logging functionality."""
    print("Testing audit logging...")

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        audit_file = f.name

    try:
        # Write some audit entries
        audit_log('SIGNED', 'client1', 'production', 'prod-key', 'SHA256', '192.168.1.1', True, audit_file)
        audit_log('DENIED', 'client2', 'development', 'prod-key', 'SHA256', '192.168.1.2', False, audit_file)

        # Read and verify
        with open(audit_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 2, "Should have 2 log entries"

        # Parse first entry
        entry1 = json.loads(lines[0])
        assert entry1['action'] == 'SIGNED'
        assert entry1['client_id'] == 'client1'
        assert entry1['success'] is True

        # Parse second entry
        entry2 = json.loads(lines[1])
        assert entry2['action'] == 'DENIED'
        assert entry2['success'] is False

        print("  ✓ Audit logging works correctly")

    finally:
        os.unlink(audit_file)


def run_all_tests():
    """Run all tests."""
    print("\n=== Running auth.py unit tests ===\n")

    tests = [
        test_token_generation_and_validation,
        test_token_expiration,
        test_token_invalid_signature,
        test_authorization,
        test_rate_limiting,
        test_config_loading,
        test_audit_logging
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print("  ✗ FAILED: {}".format(str(e)))
            failed += 1
        except Exception as e:
            print("  ✗ ERROR: {}".format(str(e)))
            failed += 1

    print("\n=== Test Results ===")
    print("Passed: {}/{}".format(len(tests) - failed, len(tests)))

    if failed > 0:
        print("FAILED: {} test(s) failed".format(failed))
        return 1
    else:
        print("SUCCESS: All tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
