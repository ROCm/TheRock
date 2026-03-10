#!/usr/bin/env python3
"""
Sign repository metadata files (DEB Release, RPM repomd.xml) using remote signing server.

This script sends metadata files to the remote signing server for GPG signing,
then writes the signature to the output file.

Supports:
- DEB Release file signing (clearsigned InRelease or detached Release.gpg)
- RPM repomd.xml signing (detached signature)

Authentication:
- Uses JWT or OIDC token via --token parameter
- Token is sent as Bearer token in Authorization header
"""

import argparse
import sys
import json
import base64
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def sign_metadata(
    metadata_file: Path,
    output_file: Path,
    server_url: str,
    token: str,
    key_id: str = "",
    digest_algo: str = "SHA256",
    clearsign: bool = False,
    timeout: int = 60
):
    """
    Sign repository metadata file using remote signing server.

    Args:
        metadata_file: Path to metadata file (Release, repomd.xml, etc.)
        output_file: Path to write signature
        server_url: Signing server URL (e.g., https://signing.example.com/sign)
        token: Authentication token (JWT or OIDC)
        key_id: GPG key ID to use (optional, server determines from role)
        digest_algo: Digest algorithm (SHA256, SHA512)
        clearsign: Create clearsigned output (for InRelease files)
        timeout: Request timeout in seconds

    Raises:
        FileNotFoundError: If metadata_file doesn't exist
        HTTPError: If server returns error response
        URLError: If network error occurs
    """

    # Read metadata file
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'rb') as f:
        metadata_data = f.read()

    print(f"📝 Metadata file: {metadata_file} ({len(metadata_data)} bytes)")

    # Prepare signing request
    payload = {
        'data': base64.b64encode(metadata_data).decode('ascii'),
        'digest_algo': digest_algo.upper(),
        'armor': True,  # Metadata signatures are ASCII-armored
    }

    # Add optional parameters
    if key_id:
        payload['key_id'] = key_id

    if clearsign:
        payload['clearsign'] = True

    # Prepare HTTP request
    json_data = json.dumps(payload).encode('utf-8')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}',
        'User-Agent': 'sign_repo_metadata/1.0'
    }

    print(f"🔐 Sending signing request to: {server_url}")
    print(f"   Digest algorithm: {digest_algo}")
    print(f"   Clearsign: {clearsign}")

    # Send request to signing server
    try:
        request = Request(server_url, data=json_data, headers=headers)
        response = urlopen(request, timeout=timeout)

        # Parse response
        response_data = response.read()
        result = json.loads(response_data)

        # Extract signature
        if 'signature' in result:
            signature_b64 = result['signature']
            signature = base64.b64decode(signature_b64)
        else:
            raise ValueError("Server response missing 'signature' field")

        print(f"✅ Signature received ({len(signature)} bytes)")

        # Write signature to output file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'wb') as f:
            f.write(signature)

        print(f"✅ Signature written to: {output_file}")

        return True

    except HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"❌ HTTP Error {e.code}: {e.reason}", file=sys.stderr)
        print(f"   Server response: {error_body}", file=sys.stderr)

        try:
            error_json = json.loads(error_body)
            if 'error' in error_json:
                print(f"   Error detail: {error_json['error']}", file=sys.stderr)
        except json.JSONDecodeError:
            pass

        raise

    except URLError as e:
        print(f"❌ Network error: {e.reason}", file=sys.stderr)
        raise

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Sign repository metadata using remote signing server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sign DEB Release file (clearsigned InRelease)
  %(prog)s --metadata-file Release --output InRelease \\
    --server https://signing.example.com/sign --token $OIDC_TOKEN --clearsign

  # Sign DEB Release file (detached signature)
  %(prog)s --metadata-file Release --output Release.gpg \\
    --server https://signing.example.com/sign --token $OIDC_TOKEN

  # Sign RPM repomd.xml
  %(prog)s --metadata-file repomd.xml --output repomd.xml.asc \\
    --server https://signing.example.com/sign --token $OIDC_TOKEN
        """
    )

    parser.add_argument(
        '--metadata-file',
        type=Path,
        required=True,
        help='Path to metadata file to sign (Release, repomd.xml, etc.)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to write signature (InRelease, Release.gpg, repomd.xml.asc, etc.)'
    )

    parser.add_argument(
        '--server',
        type=str,
        required=True,
        help='Signing server URL (e.g., https://signing.example.com/sign)'
    )

    parser.add_argument(
        '--token',
        type=str,
        required=True,
        help='Authentication token (JWT or OIDC)'
    )

    parser.add_argument(
        '--key-id',
        type=str,
        default='',
        help='GPG key ID to use (optional, server determines from role/branch)'
    )

    parser.add_argument(
        '--digest-algo',
        type=str,
        default='SHA256',
        choices=['SHA256', 'SHA512'],
        help='Digest algorithm (default: SHA256)'
    )

    parser.add_argument(
        '--clearsign',
        action='store_true',
        help='Create clearsigned output (for InRelease files)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Request timeout in seconds (default: 60)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.metadata_file.exists():
        print(f"❌ Error: Metadata file not found: {args.metadata_file}", file=sys.stderr)
        return 1

    if not args.server:
        print("❌ Error: Signing server URL is required", file=sys.stderr)
        return 1

    if not args.token:
        print("❌ Error: Authentication token is required", file=sys.stderr)
        return 1

    # Sign metadata
    try:
        sign_metadata(
            metadata_file=args.metadata_file,
            output_file=args.output,
            server_url=args.server,
            token=args.token,
            key_id=args.key_id,
            digest_algo=args.digest_algo,
            clearsign=args.clearsign,
            timeout=args.timeout
        )
        return 0

    except Exception as e:
        print(f"❌ Failed to sign metadata: {str(e)}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
