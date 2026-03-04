#!/usr/bin/env python3
"""
Example signing server implementation.

This is a reference implementation showing the expected API.
In production, replace the signing logic with actual GPG/HSM/KMS calls.

Dependencies (not needed for the shim itself):
  - Flask or any WSGI framework for production
  - python-gnupg or direct GPG calls for actual signing

This example uses http.server for simplicity.
"""

import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from base64 import b64decode, b64encode
from subprocess import run, PIPE


class SigningHandler(BaseHTTPRequestHandler):
    """HTTP request handler for signing operations."""

    def do_POST(self):
        """Handle POST requests to /sign endpoint."""
        if self.path != '/sign':
            self.send_error(404, "Not Found")
            return

        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            # Parse JSON
            request = json.loads(body.decode('utf-8'))

            # Extract parameters
            data_b64 = request.get('data')
            key_id = request.get('key_id', '')
            digest_algo = request.get('digest_algo', 'SHA256')
            armor = request.get('armor', False)
            detach = request.get('detach', True)

            if not data_b64:
                self.send_json_error(400, "Missing 'data' field")
                return

            # Decode data
            data = b64decode(data_b64)

            # Perform signing (this is where you'd call your actual signing service)
            signature = self.sign_data(data, key_id, digest_algo, armor, detach)

            if signature is None:
                self.send_json_error(500, "Signing failed")
                return

            # Return response
            response = {
                'signature': b64encode(signature).decode('ascii'),
                'key_id': key_id
            }

            self.send_json_response(200, response)

        except json.JSONDecodeError as e:
            self.send_json_error(400, f"Invalid JSON: {str(e)}")
        except Exception as e:
            self.send_json_error(500, f"Internal error: {str(e)}")

    def sign_data(self, data, key_id, digest_algo, armor, detach):
        """
        Sign data using GPG.

        In production, replace this with:
        - HSM integration
        - Cloud KMS (AWS KMS, Google Cloud KMS, Azure Key Vault)
        - Hardware token
        - Remote GPG server
        """
        try:
            # Build GPG command
            cmd = ['gpg', '--batch', '--no-tty']

            if detach:
                cmd.append('--detach-sign')
            else:
                cmd.append('--sign')

            if armor:
                cmd.append('--armor')

            if key_id:
                cmd.extend(['--local-user', key_id])

            cmd.extend(['--digest-algo', digest_algo])

            # Run GPG
            result = run(cmd, input=data, stdout=PIPE, stderr=PIPE)

            if result.returncode != 0:
                print(f"GPG error: {result.stderr.decode('utf-8', errors='ignore')}", file=sys.stderr)
                return None

            return result.stdout

        except FileNotFoundError:
            print("GPG not found. Install gpg or implement alternative signing.", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Signing error: {str(e)}", file=sys.stderr)
            return None

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
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def main():
    """Run the signing server."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(('localhost', port), SigningHandler)

    print(f"Signing server listening on http://localhost:{port}")
    print("Endpoint: POST /sign")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
