# Security & Communication Architecture Comparison

**Comparing:** POC stdlib HTTP Server vs. FastAPI REST API

## Summary

| Aspect | POC (stdlib) | FastAPI | Recommendation |
|--------|--------------|---------|----------------|
| **Security** | ✅ Equivalent | ✅ Equivalent | **Tie** |
| **Dependencies** | ✅ Zero deps | ⚠️ 10+ deps | **POC wins** |
| **Production Readiness** | ✅ Ready | ✅ Ready | **Tie** |
| **Maintainability** | ⚠️ Manual code | ✅ Framework patterns | **FastAPI wins** |
| **Performance** | ✅ Good enough | ✅ Excellent (async) | **FastAPI wins** |
| **Deployment** | ✅ Simpler | ⚠️ More complex | **POC wins** |
| **Documentation** | ⚠️ Manual | ✅ Auto OpenAPI | **FastAPI wins** |
| **Overall** | ✅ Good for AMD | ✅ Better long-term | **Context-dependent** |

---

## Detailed Security Analysis

### 1. Authentication

#### POC Implementation (stdlib)
```python
# Custom JWT implementation using stdlib
import hmac
import hashlib
import base64

def validate_jwt_token(token, secrets_map):
    # Manual JWT parsing
    # HMAC-SHA256 signature verification
    # Expiration checking
    # Constant-time comparison (hmac.compare_digest)
```

**Security features:**
- ✅ HMAC-SHA256 signed JWTs
- ✅ Token expiration validation
- ✅ Constant-time signature comparison (prevents timing attacks)
- ✅ Per-client secrets
- ✅ No external JWT library vulnerabilities

**Concerns:**
- ⚠️ Custom crypto code (though simple and well-understood HMAC)
- ⚠️ No JWT standard library validation (manual implementation)

#### FastAPI Implementation (would be)
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer
from jose import JWTError, jwt  # PyJWT or python-jose

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

**Security features:**
- ✅ Battle-tested JWT libraries (PyJWT or python-jose)
- ✅ Automatic token validation
- ✅ Framework-level security patterns
- ✅ OAuth2/OpenID Connect support (if needed)

**Concerns:**
- ⚠️ Dependency on external libraries (more attack surface)
- ⚠️ Library vulnerabilities (e.g., CVE-2022-29217 in PyJWT)

**Winner:** **Tie** - Both are secure if implemented correctly. POC has fewer dependencies, FastAPI has battle-tested libraries.

---

### 2. Authorization (RBAC)

#### POC Implementation
```python
def authorize_request(role, key_id, digest_algo, authz_config):
    """Role-based authorization."""
    roles = authz_config.get('roles', {})
    role_config = roles.get(role, {})

    allowed_keys = role_config.get('allowed_keys', [])
    if key_id not in allowed_keys:
        return False, f"Key '{key_id}' not allowed for role '{role}'"

    allowed_algos = role_config.get('allowed_digest_algos', [])
    if digest_algo not in allowed_algos:
        return False, f"Algorithm '{digest_algo}' not allowed for role '{role}'"

    return True, None
```

**Features:**
- ✅ Per-role key restrictions
- ✅ Per-role digest algorithm restrictions
- ✅ JSON configuration file
- ✅ Simple, easy to audit

#### FastAPI Implementation (would be)
```python
from fastapi import Depends, HTTPException
from typing import List

def check_permissions(
    required_role: str,
    token: dict = Depends(verify_token)
):
    if token['role'] != required_role:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return token

@app.post("/sign")
async def sign_package(
    request: SignRequest,
    user: dict = Depends(lambda: check_permissions("production"))
):
    # Signing logic
```

**Features:**
- ✅ Dependency injection for auth checks
- ✅ Cleaner code with decorators
- ✅ Automatic 403 responses
- ✅ Can use libraries like fastapi-permissions or casbin

**Winner:** **FastAPI** - More elegant patterns, better code organization

---

### 3. Rate Limiting

#### POC Implementation
```python
from collections import deque

# Class-level rate limit tracking
_rate_limits = {}  # client_id -> deque of timestamps

def check_rate_limit(client_id, role, rate_limits, authz_config):
    """Check if client has exceeded rate limit."""
    role_config = authz_config.get('roles', {}).get(role, {})
    max_requests = role_config.get('max_requests_per_hour', 0)

    if max_requests == 0:
        return True  # No limit

    now = time.time()
    hour_ago = now - 3600

    # Initialize or get request history
    if client_id not in rate_limits:
        rate_limits[client_id] = deque()

    requests = rate_limits[client_id]

    # Remove requests older than 1 hour
    while requests and requests[0] < hour_ago:
        requests.popleft()

    # Check if limit exceeded
    if len(requests) >= max_requests:
        return False

    # Add current request
    requests.append(now)
    return True
```

**Features:**
- ✅ Per-client rate limiting
- ✅ Sliding window (1 hour)
- ✅ Memory-efficient (deque cleanup)
- ✅ No external dependencies

**Concerns:**
- ⚠️ In-memory only (lost on restart)
- ⚠️ Not distributed (single server only)
- ⚠️ No Redis/persistent storage

#### FastAPI Implementation (would be)
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/sign")
@limiter.limit("100/hour")
async def sign_package(request: Request):
    # Signing logic
```

**Features:**
- ✅ Library-based (slowapi)
- ✅ Redis backend support (distributed)
- ✅ Multiple strategies (fixed window, sliding window, token bucket)
- ✅ Cleaner decorator syntax

**Winner:** **FastAPI** - Better for distributed deployments, but POC is sufficient for single-server

---

### 4. Input Validation

#### POC Implementation
```python
def validate_key_id(self, key_id):
    """Validate key_id to prevent injection attacks."""
    import re

    if not key_id or len(key_id) > 256:
        return False

    # Allow alphanumeric, @, ., -, _, space, <, >
    if not re.match(r'^[a-zA-Z0-9@.\-_ <>]+$', key_id):
        return False

    # Block directory traversal
    if '..' in key_id or '/' in key_id or '\\' in key_id:
        return False

    return True
```

**Features:**
- ✅ Regex validation
- ✅ Directory traversal prevention
- ✅ Command injection prevention
- ✅ Manual but explicit

**Concerns:**
- ⚠️ Manual validation (easy to miss edge cases)
- ⚠️ No type checking at request level

#### FastAPI Implementation (would be)
```python
from pydantic import BaseModel, Field, validator

class SignRequest(BaseModel):
    data: str = Field(..., description="Base64-encoded data")
    key_id: str = Field(..., min_length=1, max_length=256)
    digest_algo: str = Field(default="SHA256")
    armor: bool = Field(default=False)

    @validator('key_id')
    def validate_key_id(cls, v):
        import re
        if not re.match(r'^[a-zA-Z0-9@.\-_ <>]+$', v):
            raise ValueError('Invalid key_id format')
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError('Directory traversal attempt detected')
        return v

    @validator('digest_algo')
    def validate_algo(cls, v):
        allowed = ['SHA256', 'SHA512', 'SHA384']
        if v.upper() not in allowed:
            raise ValueError(f'Algorithm must be one of {allowed}')
        return v.upper()

@app.post("/sign")
async def sign_package(request: SignRequest):
    # request is already validated!
```

**Features:**
- ✅ Automatic type checking
- ✅ Request validation before handler
- ✅ Automatic 422 error responses
- ✅ OpenAPI schema generation
- ✅ Less chance of missed validation

**Winner:** **FastAPI** - Pydantic validation is superior

---

### 5. TLS/HTTPS

#### POC Implementation
```python
import ssl

# In main():
if args.enable_tls:
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile=args.cert_file, keyfile=args.key_file)
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
    server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
```

**Features:**
- ✅ TLS 1.2+ enforcement
- ✅ Custom certificate support
- ✅ Stdlib ssl module
- ✅ Works without reverse proxy

**Concerns:**
- ⚠️ Manual SSL context setup
- ⚠️ No automatic HSTS headers
- ⚠️ No automatic redirect HTTP → HTTPS

#### FastAPI Implementation (would be)
```python
# With uvicorn:
uvicorn.run(
    app,
    host="0.0.0.0",
    port=443,
    ssl_keyfile="/path/to/key.pem",
    ssl_certfile="/path/to/cert.pem",
    ssl_version=ssl.PROTOCOL_TLSv1_2
)

# Or with middleware for HSTS:
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["signing.example.com"])
```

**Features:**
- ✅ Uvicorn handles TLS
- ✅ Middleware for security headers
- ✅ Automatic HTTP → HTTPS redirect
- ✅ HSTS header support

**Winner:** **FastAPI** - More features, better defaults

---

### 6. Audit Logging

#### POC Implementation
```python
def audit_log(action, client_id, role, key_id, digest_algo, client_ip, success, audit_file):
    """Write audit log entry."""
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

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(audit_file), exist_ok=True)

        # Append to file (one JSON object per line)
        with open(audit_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except IOError:
        pass  # Don't fail request if audit logging fails
```

**Features:**
- ✅ JSON-based logging
- ✅ One entry per line (easy to parse)
- ✅ Includes all relevant fields
- ✅ Non-blocking (continues on error)

**Concerns:**
- ⚠️ File-based only (no structured logging)
- ⚠️ No rotation (file grows unbounded)
- ⚠️ No integration with logging services

#### FastAPI Implementation (would be)
```python
import logging
from pythonjsonlogger import jsonlogger

# Configure structured logging
logger = logging.getLogger("signing_service")
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        "request_processed",
        extra={
            "client_ip": request.client.host,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration": duration,
            "user_agent": request.headers.get("user-agent")
        }
    )
    return response
```

**Features:**
- ✅ Structured logging (python-json-logger)
- ✅ CloudWatch/Datadog integration
- ✅ Automatic log rotation (via logging config)
- ✅ Middleware-based (automatic for all requests)

**Winner:** **FastAPI** - Better logging infrastructure

---

### 7. DoS Protection

#### POC Implementation
```python
# Request size limit
MAX_REQUEST_SIZE = int(os.environ.get('MAX_REQUEST_SIZE', '10240'))

content_length = int(self.headers.get('Content-Length', 0))
if content_length > MAX_REQUEST_SIZE:
    self.send_json_error(413, f"Request too large: {content_length} bytes")
    return

# Read timeout
READ_TIMEOUT = int(os.environ.get('READ_TIMEOUT', '10'))
self.connection.settimeout(READ_TIMEOUT)

# Thread semaphore (max concurrent requests)
semaphore = threading.Semaphore(max_threads)
acquired = semaphore.acquire(blocking=True, timeout=30)
if not acquired:
    self.send_json_error(503, "Server busy")
    return
```

**Features:**
- ✅ Request size limits (default 10KB)
- ✅ Read timeouts (prevents slowloris)
- ✅ Thread limits (prevents exhaustion)
- ✅ 503 responses when busy

**Concerns:**
- ⚠️ No IP-based blocking
- ⚠️ No DDoS protection at application layer

#### FastAPI Implementation (would be)
```python
from fastapi import Request
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10240:
        return JSONResponse(
            status_code=413,
            content={"error": "Request too large"}
        )
    return await call_next(request)

# With uvicorn timeout
uvicorn.run(app, timeout_keep_alive=10, timeout_graceful_shutdown=30)
```

**Features:**
- ✅ Request size limits
- ✅ Configurable timeouts
- ✅ Can add IP-based rate limiting
- ✅ Integration with WAF (AWS WAF, Cloudflare)

**Winner:** **Tie** - Both handle basic DoS protection

---

## Communication Protocol Comparison

### POC: Plain HTTP/JSON

```python
# Request
POST /sign HTTP/1.1
Authorization: Bearer <JWT>
Content-Type: application/json

{
  "data": "base64...",
  "key_id": "therock-dev@amd.com",
  "digest_algo": "SHA256",
  "armor": false
}

# Response
HTTP/1.1 200 OK
Content-Type: application/json

{
  "signature": "base64...",
  "key_id": "therock-dev@amd.com",
  "digest_algo": "SHA256"
}
```

**Features:**
- ✅ Simple, standard HTTP
- ✅ JSON payloads
- ✅ Easy to debug (curl, httpie)
- ✅ Works with any HTTP client

### FastAPI: OpenAPI-documented REST

```python
# Same protocol, but with:
# - Automatic OpenAPI/Swagger docs at /docs
# - Automatic JSON schema validation
# - Request/response models
```

**Additional features:**
- ✅ Interactive API docs (Swagger UI)
- ✅ ReDoc documentation
- ✅ Schema export for client generation
- ✅ Better developer experience

**Winner:** **FastAPI** - Better documentation and developer experience

---

## Deployment Comparison

### POC Deployment

```bash
# Single command
python3 server/signing-server.py \
  --port 8443 \
  --keyring /etc/gpg-signing/keyring \
  --enable-auth \
  --secrets-file config/secrets.json \
  --authz-config config/authorization.json \
  --enable-tls \
  --cert-file server.crt \
  --key-file server.key

# Or systemd service
[Service]
ExecStart=/usr/bin/python3 /opt/gpg-signing/server/signing-server.py --port 8443 ...
```

**Pros:**
- ✅ No dependencies to install
- ✅ Simpler deployment
- ✅ Fewer moving parts
- ✅ Works on any Python 3.6+

**Cons:**
- ⚠️ No process manager built-in (need systemd)
- ⚠️ Manual restart on failure

### FastAPI Deployment

```bash
# Install dependencies
pip install fastapi uvicorn[standard] python-jose

# Run with uvicorn
uvicorn signing_service:app \
  --host 0.0.0.0 \
  --port 8443 \
  --ssl-keyfile server.key \
  --ssl-certfile server.crt \
  --workers 4

# Or with gunicorn (production)
gunicorn signing_service:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8443
```

**Pros:**
- ✅ Production-grade ASGI server (uvicorn)
- ✅ Worker process management (gunicorn)
- ✅ Auto-reload in development
- ✅ Better performance (async)

**Cons:**
- ⚠️ More dependencies
- ⚠️ More complex configuration
- ⚠️ Need to understand ASGI servers

**Winner:** **POC** for simplicity, **FastAPI** for production scale

---

## Performance Comparison

### POC (ThreadingHTTPServer)

**Architecture:**
- Multi-threaded (ThreadingMixIn)
- One thread per request
- Blocking I/O
- Thread pool via semaphore

**Performance characteristics:**
- ✅ Good for low-medium load (< 100 concurrent)
- ✅ Simple concurrency model
- ⚠️ Thread overhead for high concurrency
- ⚠️ GIL limitations

**Estimated throughput:** ~100-500 requests/sec

### FastAPI (uvicorn ASGI)

**Architecture:**
- Async/await (asyncio)
- Event loop
- Non-blocking I/O
- Multiple workers

**Performance characteristics:**
- ✅ Excellent for high load (1000+ concurrent)
- ✅ Low memory overhead
- ✅ No GIL issues (async)
- ⚠️ More complex async code

**Estimated throughput:** ~1000-5000 requests/sec

**Winner:** **FastAPI** - Significantly better performance at scale

---

## Recommendation

### For AMD's TheRock Project: **Use POC as-is**

**Reasons:**
1. ✅ **Already implemented and tested** - POC is production-ready
2. ✅ **Zero dependencies** - Critical for enterprise deployment
3. ✅ **Simple deployment** - Just copy files and run
4. ✅ **Sufficient performance** - Signing is CPU-bound (GPG), not I/O-bound
5. ✅ **All security features present** - JWT, RBAC, rate limiting, audit logging
6. ✅ **Well-documented** - Comprehensive docs already exist
7. ✅ **Proven architecture** - Successfully handles 250x data reduction

### When to Switch to FastAPI:

Switch if **any** of these become true:
- 🔴 **Scale requirements increase** - Need to handle 1000+ concurrent requests
- 🔴 **Need better monitoring** - Want Prometheus metrics, distributed tracing
- 🔴 **Team prefers modern frameworks** - Developers want FastAPI patterns
- 🔴 **Need OpenAPI docs** - External consumers require interactive API docs
- 🔴 **Distributed deployment** - Multiple signing servers with shared state (Redis)

### Migration Path (if needed later):

```
Phase 1: Keep POC architecture (2026 Q1-Q2)
├─ Deploy signing server with POC code
├─ Integrate with GitHub Actions
└─ Validate in production

Phase 2: Evaluate (2026 Q3)
├─ Monitor usage patterns
├─ Measure performance requirements
└─ Decide: Keep POC or migrate to FastAPI

Phase 3: Migrate (if needed, 2026 Q4)
├─ Port logic to FastAPI
├─ Add Prometheus metrics
├─ Deploy alongside POC (blue-green)
└─ Gradual traffic shift
```

---

## Conclusion

**Both approaches are secure and production-ready.**

The POC stdlib implementation is **better for AMD now** because:
- It's already done and tested
- Zero dependencies reduces risk
- Simpler deployment and maintenance
- Performance is sufficient for use case

FastAPI would be **better for future** if:
- Scale requirements increase significantly
- Team wants modern Python patterns
- Need advanced monitoring/observability
- Distributed deployment becomes necessary

**Recommendation: Start with POC, evaluate in 6 months, migrate only if needed.**
