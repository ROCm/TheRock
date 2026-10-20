#!/usr/bin/env python3
"""
Shared HTTP client for the remote GPG signing server.

Builds and sends a single POST /sign request, handling both the direct-HTTP
path (Phase 1/2 — GPG_SIGNING_SERVER, optional pre-shared bearer token) and
the AWS SigV4-via-API-Gateway path (Phase 1b — GPG_SIGNING_API_URL, cross-cloud
build runners). This is the one place the SigV4 signing algorithm is
implemented — both gpgshim (RPM package signing) and sign_package_repo.py
(repository metadata signing) import it, so there is exactly one copy to
keep correct, not two.

Stdlib only — no boto3/requests — matches gpgshim's zero-dependency,
Python 3.6-compatible design goal (signing_server-requirements.md P1b-SHIM-2).
Retry/backoff policy is deliberately NOT implemented here — callers decide
their own retry behavior; this module sends exactly one request per call
and raises on failure.
"""

import datetime
import hashlib
import hmac
import json
import os
from base64 import b64decode, b64encode
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


class SigningClientConfig:
    """Resolved configuration for signing requests.

    Construct via from_env() in the common case (reads the same environment
    variables gpgshim has always used, so existing CI configuration keeps
    working unchanged for any caller of this module).
    """

    def __init__(
        self,
        signing_server="",
        api_url="",
        auth_token="",
        aws_access_key="",
        aws_secret_key="",
        aws_session_token="",
        aws_region="us-east-1",
        ca_cert="",
        verify_ssl=True,
        timeout=30,
    ):
        self.signing_server = signing_server
        self.api_url = api_url
        self.auth_token = auth_token
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.aws_session_token = aws_session_token
        self.aws_region = aws_region
        self.ca_cert = ca_cert
        self.verify_ssl = verify_ssl
        self.timeout = timeout

    @classmethod
    def from_env(cls):
        return cls(
            signing_server=os.environ.get(
                "GPG_SIGNING_SERVER", "http://localhost:8080/sign"
            ),
            api_url=os.environ.get("GPG_SIGNING_API_URL", ""),
            auth_token=os.environ.get("GPG_SERVER_TOKEN", ""),
            aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN", ""),
            aws_region=os.environ.get(
                "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            ),
            ca_cert=os.environ.get("GPG_SERVER_CA_CERT", ""),
            verify_ssl=os.environ.get("GPG_VERIFY_SSL", "true").lower() != "false",
            timeout=int(os.environ.get("GPG_TIMEOUT", "30")),
        )

    @property
    def target_url(self):
        return self.api_url if self.api_url else self.signing_server


def sigv4_headers(method, url, body, access_key, secret_key, session_token, region):
    """Compute AWS Signature Version 4 headers for a request to API Gateway.

    Pure function — no dependency on SigningClientConfig — so it can be unit
    tested against AWS's published SigV4 test vectors independently of any
    caller's configuration plumbing.
    """
    parsed = urlsplit(url)
    host = parsed.netloc
    canonical_uri = quote(parsed.path if parsed.path else "/", safe="/-_.~")
    canonical_querystring = parsed.query or ""

    now = datetime.datetime.utcnow()
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    payload_hash = hashlib.sha256(body).hexdigest()

    headers_to_sign = {
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    if session_token:
        headers_to_sign["x-amz-security-token"] = session_token

    signed_header_names = sorted(headers_to_sign.keys())
    canonical_headers = "".join(
        "{}:{}\n".format(name, headers_to_sign[name]) for name in signed_header_names
    )
    signed_headers = ";".join(signed_header_names)

    canonical_request = "\n".join(
        [
            method,
            canonical_uri,
            canonical_querystring,
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )

    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = "{}/{}/execute-api/aws4_request".format(date_stamp, region)
    string_to_sign = "\n".join(
        [
            algorithm,
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )

    def _hmac_sha256(key, msg):
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    k_date = _hmac_sha256(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, "execute-api")
    k_signing = _hmac_sha256(k_service, "aws4_request")
    signature = hmac.new(
        k_signing, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    authorization_header = "{} Credential={}/{}, SignedHeaders={}, Signature={}".format(
        algorithm, access_key, credential_scope, signed_headers, signature
    )

    result = {
        "Host": host,
        "X-Amz-Date": amz_date,
        "X-Amz-Content-Sha256": payload_hash,
        "Authorization": authorization_header,
    }
    if session_token:
        result["X-Amz-Security-Token"] = session_token
    return result


def send_signing_request(config, payload, user_agent="therock-signing-client/1.0"):
    """Send one POST request to the signing server and return the parsed
    JSON response (a dict, typically containing 'signature' and 'key_id').

    Raises urllib.error.HTTPError / URLError on transport failure, or
    ValueError if required SigV4 credentials are missing — callers decide
    their own retry policy around this single-attempt call.
    """
    target_url = config.target_url
    body = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json", "User-Agent": user_agent}

    if config.api_url:
        # Cross-cloud path (Phase 1b): authenticate via AWS SigV4 using
        # temporary credentials already present in the caller's environment.
        # Fail fast with a clear error rather than sending an unsigned
        # request that API Gateway would reject with an opaque 403.
        if not (config.aws_access_key and config.aws_secret_key):
            raise ValueError(
                "GPG_SIGNING_API_URL is set but AWS_ACCESS_KEY_ID/"
                "AWS_SECRET_ACCESS_KEY are missing from the environment"
            )
        headers.update(
            sigv4_headers(
                "POST",
                target_url,
                body,
                config.aws_access_key,
                config.aws_secret_key,
                config.aws_session_token,
                config.aws_region,
            )
        )
    elif config.auth_token:
        # Direct/operator path: pre-shared bearer token (Phase 2), unchanged.
        headers["Authorization"] = "Bearer {}".format(config.auth_token)

    request = Request(target_url, data=body, headers=headers, method="POST")

    ssl_context = None
    if target_url.startswith("https://"):
        import ssl

        ssl_context = ssl.create_default_context()
        if config.ca_cert:
            ssl_context.load_verify_locations(config.ca_cert)
        if not config.verify_ssl:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

    with urlopen(request, timeout=config.timeout, context=ssl_context) as response:
        return json.loads(response.read().decode("utf-8"))


def sign_data(
    config,
    data,
    tier=None,
    artifact=None,
    key_id=None,
    digest_algo="SHA256",
    armor=False,
):
    """Convenience one-shot helper: sign `data` and return the raw signature
    bytes. No retry, no caching — intended for callers (like repo-metadata
    signing) that sign a handful of files once, not gpgshim's high-volume
    per-RPM-header path (which has its own retry/backoff and ppid cache).

    Either pass (tier, artifact) for the simplified API, or (key_id,
    digest_algo, armor) for the legacy explicit API — mirrors gpgshim's
    payload construction exactly.
    """
    if tier:
        payload = {
            "data": b64encode(data).decode("ascii"),
            "tier": tier,
            "artifact": artifact or "rpm",
        }
    else:
        payload = {
            "data": b64encode(data).decode("ascii"),
            "key_id": key_id,
            "digest_algo": digest_algo,
            "armor": armor,
        }

    result = send_signing_request(config, payload)
    if "signature" not in result:
        raise ValueError(
            "Invalid response from signing server: missing 'signature' field"
        )
    return b64decode(result["signature"])


# Maps a workflow `release_type` (ci/dev/nightly/prerelease/release) to the
# GPG_TIER value sent to the signing server, i.e. which key_alias in
# authorization.json is used. Two release types deliberately map to None
# (skip signing, no error):
#   - "ci": per signing-server-requirements.md P1-CI-1, signing is only
#     activated for dev/nightly/prerelease builds meant for distribution —
#     not routine CI/PR validation builds, to avoid unnecessary signing
#     server load.
#   - "release": per signing-server-design.md Section 4.1b, the
#     release-tier key is excluded from automated/cross-cloud signing and
#     stays a manual, operator-driven process (sign-file CLI over VPN) —
#     this function is how that boundary is enforced on the client/workflow
#     side, not just documented.
_RELEASE_TYPE_TO_TIER = {
    "ci": None,
    "dev": "dev",
    "nightly": "nightly",
    "prerelease": "nightly",
    "release": None,
}


def resolve_signing_tier(release_type):
    """Resolve a workflow `release_type` to a GPG_TIER, or None if this
    release type must not be signed automatically (see module comment)."""
    return _RELEASE_TYPE_TO_TIER.get(release_type)
