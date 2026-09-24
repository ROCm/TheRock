# Local ASan Python Package Index

Phase 1 ASan packages can be staged and installed without publishing them to
S3 or any other network service. The local layout mirrors the proposed
per-family publication tree:

```text
<output-root>/
└── whl-asan/
    └── gfx942-all/
        ├── index.html
        ├── index-manifest.json
        ├── rocm-10.1.0+asan.<build-id>.tar.gz
        ├── rocm_sdk_*.whl
        └── simple/
            ├── index.html
            └── <normalized-project>/index.html
```

The top-level `index.html` is a flat `pip --find-links` index generated with
TheRock's existing local index utility. The `simple/` subtree is a PEP 503
repository with SHA-256 fragments. `index-manifest.json` records the same
hashes and package metadata for offline verification.

## Stage packages

First build ASan packages into a local package directory. Then stage them into
the isolated index:

```bash
python build_tools/packaging/python/stage_local_asan_index.py stage \
  --input-dir /path/to/packages \
  --output-root /path/to/phase1-local \
  --require-phase1-set
```

If `/path/to/packages/dist` exists, it is used automatically. Otherwise the
tool reads packages directly from `/path/to/packages`. By default, it accepts
only package metadata versions beginning with `10.1.0+asan.` and stages them
under `whl-asan/gfx942-all`.

`--require-phase1-set` verifies that the local index contains:

- the `rocm` selector sdist;
- `rocm-sdk-core`;
- `rocm-sdk-libraries`;
- `rocm-sdk-device-gfx942`;
- `rocm-sdk-devel`.

The tool refuses to overwrite an existing package with different contents. Use
a new output root for a different build ID rather than mixing builds.

## Verify and install offline

Verify every package against the manifest before installation:

```bash
python build_tools/packaging/python/stage_local_asan_index.py verify \
  --output-root /path/to/phase1-local \
  --require-phase1-set
```

Install from the flat local index with network package lookup disabled:

The single-target selector exposes the generic `device` extra, which is pinned
to `rocm-sdk-device-gfx942`. Per-target extras such as `device-gfx942` are only
generated for multi-target selectors.

```bash
INDEX=/path/to/phase1-local/whl-asan/gfx942-all
python -m venv /path/to/asan-venv
source /path/to/asan-venv/bin/activate
python -m pip install --no-index --find-links="$INDEX/index.html" --pre \
  "rocm[libraries,devel,device]"
```

Or exercise the PEP 503 tree directly:

```bash
python -m pip install --index-url="file://$INDEX/simple/" --pre \
  "rocm[libraries,devel,device]"
```

Run the installed SDK checks under the ASan environment described in
[`sanitizers.md`](../development/sanitizers.md):

```bash
export HSA_XNACK=1
export ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:print_stacktrace=1
python -m rocm_sdk path --root
rocm-sdk test
```

The staging command only uses local filesystem operations. It does not import
TheRock's storage backends, choose a bucket, or expose an upload mode.
