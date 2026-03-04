#!/usr/bin/env python3
"""
Token management utility for GPG signing server authentication.

Generates, validates, and manages JWT tokens for client authentication.
"""

import sys
import json
import argparse
import secrets
import base64
from auth import generate_jwt_token, validate_jwt_token, load_secrets


def generate_secret(length=32):
    """Generate cryptographically secure random secret."""
    random_bytes = secrets.token_bytes(length)
    return base64.b64encode(random_bytes).decode('ascii')


def decode_token_payload(token):
    """Decode token payload without validation."""
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        padding = 4 - (len(payload_b64) % 4)
        if padding != 4:
            payload_b64 += '=' * padding
        payload_json = base64.urlsafe_b64decode(payload_b64).decode('utf-8')
        return json.loads(payload_json)
    except Exception:
        return None


def cmd_generate(args):
    """Generate a new token."""
    if not args.client_id or not args.role:
        print("Error: --client-id and --role are required", file=sys.stderr)
        return 1

    if args.secret_file:
        secrets_map = load_secrets(args.secret_file)
        if args.client_id not in secrets_map:
            print("Error: Client ID '{}' not found".format(args.client_id), file=sys.stderr)
            return 1
        client_secret = secrets_map[args.client_id]
        secret = client_secret.get('secret', '') if isinstance(client_secret, dict) else client_secret
    elif args.secret:
        secret = args.secret
    else:
        print("Error: --secret-file or --secret required", file=sys.stderr)
        return 1

    token = generate_jwt_token(args.client_id, args.role, secret, args.expires_hours)
    payload = decode_token_payload(token)

    print("Token generated!")
    print("Client: {}, Role: {}".format(args.client_id, args.role))
    if payload and 'exp' in payload:
        import datetime
        exp_dt = datetime.datetime.utcfromtimestamp(payload['exp'])
        print("Expires: {}".format(exp_dt.strftime('%Y-%m-%d %H:%M:%S UTC')))
    print("\nToken:\n{}".format(token))
    print("\nUsage:\n  export GPG_SERVER_TOKEN='{}'".format(token))
    return 0


def cmd_validate(args):
    """Validate a token."""
    if not args.secret_file:
        print("Error: --secret-file required", file=sys.stderr)
        return 1

    secrets_map = load_secrets(args.secret_file)
    payload = validate_jwt_token(args.validate, secrets_map)

    if payload:
        print("VALID - Client: {}, Role: {}".format(payload['client_id'], payload['role']))
        if 'exp' in payload:
            import datetime, time
            exp_dt = datetime.datetime.utcfromtimestamp(payload['exp'])
            remaining = (payload['exp'] - time.time()) / 3600
            print("Expires: {} ({:.1f}h remaining)".format(exp_dt.strftime('%Y-%m-%d %H:%M:%S'), remaining))
        return 0
    else:
        print("INVALID - Check signature/expiration")
        return 1


def cmd_decode(args):
    """Decode token without validation."""
    payload = decode_token_payload(args.decode)
    if not payload:
        print("Error: Invalid token format", file=sys.stderr)
        return 1

    print("Token contents (UNVERIFIED):")
    print(json.dumps(payload, indent=2))

    if 'exp' in payload:
        import datetime, time
        print("\nIssued: {}".format(datetime.datetime.utcfromtimestamp(payload['iat']).strftime('%Y-%m-%d %H:%M:%S')))
        print("Expires: {}".format(datetime.datetime.utcfromtimestamp(payload['exp']).strftime('%Y-%m-%d %H:%M:%S')))
        print("Status: {}".format("Valid" if payload['exp'] > time.time() else "EXPIRED"))
    return 0


def cmd_generate_secret(args):
    """Generate a new random secret."""
    if not args.client_id:
        print("Error: --client-id required", file=sys.stderr)
        return 1

    secret = generate_secret()
    print("New secret for '{}':\n".format(args.client_id))
    print(json.dumps({args.client_id: {"secret": secret, "description": "TODO", "created": "TODO"}}, indent=2))
    print("\nIMPORTANT: Store securely - cannot be recovered if lost")
    return 0


def main():
    parser = argparse.ArgumentParser(description='GPG signing server token management')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--generate', action='store_true', help='Generate token')
    mode.add_argument('--validate', help='Validate token')
    mode.add_argument('--decode', help='Decode token')
    mode.add_argument('--generate-secret', action='store_true', help='Generate secret')

    parser.add_argument('--client-id', help='Client ID')
    parser.add_argument('--role', help='Role name')
    parser.add_argument('--secret', help='Client secret')
    parser.add_argument('--secret-file', help='Secrets file path')
    parser.add_argument('--expires-hours', type=int, default=4, help='Token validity (hours)')

    args = parser.parse_args()

    if args.generate:
        return cmd_generate(args)
    elif args.validate:
        return cmd_validate(args)
    elif args.decode:
        return cmd_decode(args)
    elif args.generate_secret:
        return cmd_generate_secret(args)


if __name__ == '__main__':
    sys.exit(main())
