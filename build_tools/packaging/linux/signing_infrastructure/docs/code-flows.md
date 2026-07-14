# Signing Server — Code Flows

**Component:** Remote GPG Signing Service  
**Related docs:** `signing-server-design.md`, `use-case-flows.md`

This document traces the execution path through the code for each significant operation. For the actor-level view of the same flows, see `use-case-flows.md`.

---

## Files and Responsibilities

| File | Responsibility |
|------|---------------|
| `server/signing-server.py` | HTTP server, request handling, GPG subprocess, startup key loading |
| `server/auth.py` | Token validation, authorization, rate limiting, audit logging |
| `build_tools/packaging/linux/gpgshim` | Drop-in gpg replacement for rpmsign; ppid cache; retry logic (canonical copy) |
| `tools/sign-file` | Operator CLI; reads file, calls POST /sign, writes output |
| `tools/setup-server.sh` | One-time EC2 provisioning; tmpfs, systemd, TLS cert |
| `config/authorization.json` | Role definitions, key allowlists, rate limits |

---

## CF-1: Server Startup

**Entry point:** `main()` in `signing-server.py`

```
main()
├── argparse → validates --keyring, --enable-tls, --enable-auth
│
├── if args.sm_secrets:
│     load_keys_from_secrets_manager(secret_names, keyring_dir)
│     ├── import boto3
│     ├── boto3.client('secretsmanager')
│     ├── for each secret_name:
│     │     client.get_secret_value(SecretId=secret_name)
│     │     → response['SecretString'] = PEM-armored GPG key
│     │     → run(['gpg', '--batch', '--import'], input=key_bytes, env={GNUPGHOME})
│     │     → key_material = None  ← clear from memory
│     └── returns count of imported keys
│
├── verify_keyring(keyring_dir)
│     └── run(['gpg', '--list-secret-keys'], env={GNUPGHOME})
│         counts 'sec ' occurrences → sets _keyring_ready = True/False
│
├── _signing_semaphore = threading.Semaphore(max_threads)
│
├── ThreadedHTTPServer((host, port), SigningHandler)
│
├── if args.enable_tls:
│     ssl.SSLContext → load_cert_chain → wrap_socket
│     minimum_version = TLSv1_2
│
└── server.serve_forever()
```

**Key globals set at startup:**
- `_keyring_ready` — controls `/health` response
- `_signing_semaphore` — limits concurrent signing operations
- `os.environ['GNUPGHOME']` — keyring path used by all handler instances

---

## CF-2: GET /health

**Entry point:** `SigningHandler.do_GET()`

```
do_GET()
├── if self.path != '/health': send_error(404); return
│
├── if _keyring_ready:
│     send_json_response(200, {'status': 'ok'})
└── else:
      send_json_response(503, {'status': 'unavailable', 'reason': '...'})
```

No auth check. No rate limiting. Lightweight — just reads the global flag.

---

## CF-3: POST /sign — Full Request Path

**Entry point:** `SigningHandler.do_POST()`

```
do_POST()
├── if path == '/quit': schedule shutdown; return
├── if path != '/sign': send_error(404); return
│
├── request_start = time.time()
│
├── [Size check]
│   content_length = int(headers['Content-Length'])
│   if content_length > MAX_REQUEST_SIZE (10240):
│       send_json_error(413, ...); return
│
├── [Read body with timeout]
│   connection.settimeout(READ_TIMEOUT)  ← 10 seconds
│   body = rfile.read(content_length)
│   request = json.loads(body)
│
├── [Authentication]  ← Phase 1: skipped (AUTH_ENABLED=false)
│   client_id = self.client_address[0]  ← source IP (Phase 1)
│   auth_type = 'none'
│   payload = None
│   │
│   if AUTH_ENABLED:                    ← Phase 2 path
│       authenticate_request()
│       ├── check Authorization: Bearer <token>
│       ├── try validate_app_token()   → auth_type='token'
│       ├── try validate_github_oidc_token() → auth_type='oidc'
│       └── try validate_jwt_token()   → auth_type='jwt'
│
├── [Field validation]
│   key_id = request['key_id']         ← required
│   digest_algo = request.get('digest_algo', 'SHA256').upper()
│   armor = request.get('armor', False)
│   clearsign = request.get('clearsign', False)
│   data = b64decode(request['data'])
│   validate_key_id(key_id)            ← regex allowlist, injection prevention
│
├── [Authorization]  ← Phase 1: skipped (AUTH_ENABLED=false)
│   if AUTH_ENABLED and payload:
│       if auth_type == 'oidc': authorize_oidc_request()
│       else:                   authorize_request()
│       check_rate_limit(client_id, role, _rate_limits, authz_config)
│
├── [Phase 1 IP-based rate limit]  ← optional, if authz_config present
│   check_rate_limit(source_ip, 'default', _rate_limits, authz_config)
│
├── [Concurrency control]
│   semaphore.acquire(blocking=True, timeout=30)
│   if not acquired: send_json_error(503, "Server busy"); return
│
├── sign_data(data, key_id, digest_algo, armor, clearsign)
│   ├── write data to tempfile
│   ├── build gpg command:
│   │   cmd = ['gpg', '--batch', '--no-tty',
│   │           '--digest-algo', digest_algo,
│   │           '--local-user', key_id]
│   │   if clearsign: cmd += ['--clearsign', '--output', '-', data_file]
│   │   else:
│   │       if armor: cmd += ['--armor']
│   │       cmd += ['--detach-sign', '--output', '-', data_file]
│   │   env = {GNUPGHOME: keyring_dir}
│   ├── run(cmd, stdout=PIPE, stderr=PIPE, timeout=30, env=env)
│   ├── if returncode != 0: log GPG error; return None
│   └── return stdout bytes (the signature)
│
├── semaphore.release()
│
├── latency_ms = int((time.time() - request_start) * 1000)
│
├── send_json_response(200, {
│       'signature': b64encode(signature),
│       'key_id': key_id,
│       'digest_algo': digest_algo
│   })
│
└── audit_log('SIGNED', client_id, role, key_id, digest_algo,
              source_ip, True, audit_file, oidc_context,
              auth_type, latency_ms)
    ├── builds JSON entry dict
    ├── sys.stdout.write(json_line)   ← CloudWatch via systemd journal
    └── if audit_file: open(audit_file, 'a').write(json_line)
```

---

## CF-4: auth.py — Rate Limiting

**Entry point:** `check_rate_limit(client_id, role, rate_limits, authz_config)`

```
check_rate_limit()
├── roles = authz_config['roles']
├── if role not in roles: role = 'default'
├── if role not in roles: return True  ← no limit configured
│
├── max_requests = roles[role]['max_requests_per_hour']
├── if max_requests <= 0: return True  ← unlimited
│
├── if client_id not in rate_limits:
│     rate_limits[client_id] = deque()
│
├── now = time.time()
├── one_hour_ago = now - 3600
│
├── [Sliding window cleanup]
│   while timestamps and timestamps[0] < one_hour_ago:
│       timestamps.popleft()
│
├── if len(timestamps) >= max_requests:
│     return False  ← rate limit exceeded → caller gets 429
│
├── timestamps.append(now)
└── return True  ← under limit
```

The `rate_limits` dict is a class-level variable on `SigningHandler` — shared across all request threads. Access is not explicitly locked; Python's GIL provides sufficient protection for deque operations.

---

## CF-5: auth.py — Phase 2 App Token Validation

**Entry point:** `validate_app_token(token, tokens_map)`

```
validate_app_token()
├── if not token or not tokens_map: return None
│
└── for each entry in tokens_map.values():
      stored = entry['token']
      if hmac.compare_digest(token, stored):  ← constant-time comparison
          return {
              'client_id': entry['client_id'],
              'role':      entry['role']
          }
      (continue to next entry if no match)
└── return None  ← no matching token
```

`hmac.compare_digest` prevents timing attacks — the comparison takes the same time regardless of where the strings differ.

---

## CF-6: gpgshim — Full Execution Path

**Entry point:** `main()` → `GPGShim().run(sys.argv[1:])`

```
run(args)
├── parse_args(args)
│   └── argparse extracts: --detach-sign, --local-user, --output,
│                          --digest-algo, --armor, etc.
│
├── [ppid cache check]
│   cache_file = '/tmp/gpgshim-cache-<ppid>.sig'
│   use_cache = os.path.exists(cache_file)
│
├── [Read stdin — always, regardless of cache]
│   data = sys.stdin.buffer.read()
│   (rpmsign pipes data via stdin; must be consumed even if cached)
│
├── if use_cache:                         ← Call 2 from rpmsign
│     signature = open(cache_file, 'rb').read()
│     os.unlink(cache_file)              ← delete after use
│
└── else:                                 ← Call 1 from rpmsign
      sign_remote(data, options)
      ├── data_hash = sha256(data).hexdigest()
      ├── if data_hash in self.data_cache: return cached  ← dedup
      │
      ├── payload = {
      │     'data': b64encode(data),
      │     'key_id': options['local_user'] or self.key_id,
      │     'digest_algo': options['digest_algo'],
      │     'armor': options['armor']
      │   }
      │
      ├── [Retry loop — exponential backoff]
      │   for attempt in range(max_retries):  ← default 5
      │     _make_signing_request(payload)
      │     ├── headers = {'Content-Type': 'application/json'}
      │     ├── if self.auth_token:           ← Phase 2 only
      │     │     headers['Authorization'] = 'Bearer ' + token
      │     ├── Request(url, data=json_payload, headers=headers)
      │     ├── ssl_context (verify or skip for self-signed)
      │     ├── urlopen(request, timeout=30, context=ssl_ctx)
      │     └── return b64decode(response['signature'])
      │
      │     on HTTPError 503: sleep(backoff + jitter); backoff *= 2
      │     on HTTPError 401/403/429/413: break (no retry)
      │     on URLError: sleep(backoff + jitter); backoff *= 2
      │
      └── self.data_cache[data_hash] = signature  ← session dedup cache
          write cache_file for ppid-based reuse
```

**Output:**
```
if output_file:
    open(output_file, 'wb').write(signature)
else:
    sys.stdout.buffer.write(signature)
```

---

## CF-7: sign-file — Operator CLI

**Entry point:** `main()` → `sign_file(args)`

```
sign_file(args)
├── open(args.file, 'rb') → data
│
├── payload = {
│     'data': b64encode(data),
│     'key_id': args.key_id,
│     'digest_algo': args.digest_algo,
│     'armor': args.armor or args.clearsign,
│     'clearsign': args.clearsign
│   }
│
├── headers = {'Content-Type': 'application/json', 'User-Agent': 'sign-file/1.0'}
├── token = args.token or os.environ.get('SIGNING_TOKEN', '')
├── if token: headers['Authorization'] = 'Bearer ' + token  ← Phase 2
│
├── ssl_ctx = ssl.create_default_context()
├── if args.no_verify_ssl:
│     ctx.check_hostname = False; ctx.verify_mode = CERT_NONE
│
├── urlopen(Request(server_url, body, headers), timeout=60, context=ssl_ctx)
│
├── on HTTPError: print human-readable error for 401/403/429/503/413; sys.exit(1)
├── on URLError: print "Network error — VPN connected?"; sys.exit(1)
│
├── signature = b64decode(response['signature'])
│
├── [Determine output path]
│   if args.output:    out = args.output
│   elif args.clearsign: out = splitext(file)[0] + '.clearsigned'
│   elif args.armor:   out = file + '.asc'
│   else:              out = file + '.sig'
│
└── open(out, 'wb').write(signature)
    print("Signature written to: <out>")
```

---

## CF-8: validate_key_id — Injection Prevention

**Entry point:** `SigningHandler.validate_key_id(key_id)`

```
validate_key_id(key_id)
├── if not key_id or len(key_id) > 256: return False
│
├── regex = r'^[a-zA-Z0-9@.\-_ <>]+$'
├── if not re.match(regex, key_id): return False
│   Blocks: ; & | $ ` ' " ( ) [ ] { } / \ and all other shell metacharacters
│
├── if '..' in key_id or '/' in key_id or '\\' in key_id: return False
│   Blocks: directory traversal attempts
│
└── return True
```

`key_id` is passed directly to the gpg `--local-user` argument. The validation prevents shell injection through GPG's argument handling.

---

## CF-9: audit_log — Structured Logging

**Entry point:** `audit_log(action, client_id, role, key_id, ...)` in `auth.py`

```
audit_log(action, client_id, role, key_id, digest_algo,
          client_ip, success, audit_file, oidc_context,
          auth_type, latency_ms)
├── entry = {
│     'timestamp':   datetime.utcnow().isoformat() + 'Z',
│     'action':      action,      ← 'SIGNED', 'DENIED', 'AUTH_FAILED', 'RATE_LIMITED'
│     'client_id':   client_id,   ← token name or source IP
│     'role':        role,
│     'key_id':      key_id,
│     'digest_algo': digest_algo,
│     'source_ip':   client_ip,
│     'auth_type':   auth_type,   ← 'none', 'token', 'jwt', 'oidc'
│     'success':     success,
│     'latency_ms':  latency_ms   ← only on SIGNED
│   }
│
├── if oidc_context:
│     entry.update({repository, ref, workflow, actor, run_id, ...})
│
├── line = json.dumps(entry) + '\n'
│
├── sys.stdout.write(line)        ← captured by systemd → CloudWatch Logs
│   sys.stdout.flush()
│
└── if audit_file:
      open(audit_file, 'a').write(line)  ← optional local file
```

**Sample output:**
```json
{"timestamp": "2026-07-11T10:23:41Z", "action": "SIGNED", "client_id": "10.0.1.12", "role": "none", "key_id": "therock-release@amd.com", "digest_algo": "SHA256", "source_ip": "10.0.1.12", "auth_type": "none", "success": true, "latency_ms": 87}
```
