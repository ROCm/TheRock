# hipDNN Python Bindings — Installation and Packaging

## Component Ownership

### Build Phase

```
┌─ rocm-libraries (hipDNN) ─────────────────────────────────────────────┐
│                                                                       │
│  hipdnn/CMakeLists.txt                                                │
│    └── add_subdirectory(python)                                       │
│          └── python/CMakeLists.txt                                    │
│                ├── nanobind builds hipdnn_frontend_python.*.so        │
│                ├── if SKBUILD  → install to wheel (pip install .)     │
│                └── else        → install to python_bindings/          │
│                                                                       │
│  Output (stage tree):                                                 │
│    stage/lib/libhipdnn_backend.so          ← native library           │
│    stage/python_bindings/hipdnn_frontend/  ← Python extension + .py   │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ TheRock ─────────────────────────────────────────────────────────────┐
│                                                                       │
│  PACKAGING (two separate paths)                                       │
│                                                                       │
│  ┌─ tar.xz path ──────────────────────────────────────────────────┐   │
│  │  artifact-hipdnn.toml                                          │   │
│  │    components.lib  → libhipdnn_backend.so (python_bindings/*   │   │
│  │                      is EXCLUDED)                              │   │
│  │    components.test → test binaries, CTest files                │   │
│  │                                                                │   │
│  │  Consumed by: install_rocm_from_artifacts.py --hipdnn          │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─ wheel path ───────────────────────────────────────────────────┐   │
│  │  py_packaging.py                                               │   │
│  │    populate_runtime_files()    → picks up libhipdnn_backend.so │   │
│  │    populate_python_bindings()  → picks up python_bindings/*    │   │
│  │                                                                │   │
│  │  setup.py (rocm-sdk-libraries template)                        │   │
│  │    find_packages(where="./src") → discovers hipdnn_frontend    │   │
│  │                                                                │   │
│  │  build_python_packages.py → builds .whl                        │   │
│  │  upload_python_packages.py → uploads to find-links URL         │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  TESTING                                                              │
│                                                                       │
│  ┌─ tar.xz tests ───────────────────────────────────────────────-─┐   │
│  │  test_hipdnn.py → ctest (native test binaries)                 │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─ wheel tests ────────────────────────────────────────────────-─┐   │
│  │  test_rocm_wheels.yml                                          │   │
│  │    pip install rocm[libraries] --find-links=<url>              │   │
│  │    rocm-sdk test                                               │   │
│  │      └── libraries_test.py::testHipDNNFrontendImport           │   │
│  │            └── import hipdnn_frontend                          │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### End-User Installation

```
┌─ C/C++ consumer ──────────────────────┐   ┌─ Python consumer ─────────────────┐
│                                       │   │                                   │
│  install_rocm_from_artifacts.py       │   │  pip install rocm[libraries]      │
│    → extracts tar.xz                  │   │    → installs wheel               │
│    → /opt/rocm/lib/libhipdnn_*.so     │   │    → hipdnn_frontend importable   │
│    → NO python bindings               │   │    → libhipdnn_backend.so included│
│                                       │   │                                   │
└───────────────────────────────────────┘   └───────────────────────────────────┘
```

## Three Installation Paths

| Path | Command | Audience |
|---|---|---|
| Wheel (rocm-sdk-libraries) | `pip install rocm[libraries]` | End users, CI wheel tests |
| Standalone wheel | `pip install .` from `hipdnn/python/` with cmake defines | Developers building from source |
| cmake install | `ninja install` → `<prefix>/python_bindings/hipdnn_frontend/` | Not directly usable by Python without `PYTHONPATH` |

## rocm-sdk-libraries Python Wheel Build Flow

1. hipDNN's `add_subdirectory(python)` always builds the nanobind extension
2. `SKBUILD` is **not** set in the TheRock superbuild, so cmake installs to
   `python_bindings/hipdnn_frontend/` in the stage tree
3. `populate_python_bindings()` copies those files into the wheel's `src/`
   directory
4. `setup.py`'s `find_packages(where="./src")` discovers `hipdnn_frontend`
   and includes it in the wheel
5. The native `.so` extension (`hipdnn_frontend_python.*.so`) goes into the
   wheel alongside `__init__.py`
6. Runtime library dependencies (`libhipdnn_backend.so`, etc.) are separately
   included via `populate_runtime_files()`

## What's in the tar.xz vs the Wheel

- **tar.xz (`hipdnn_lib`)** — native `.so` libraries, headers, cmake
  configs. `python_bindings/**` is excluded.
- **wheel (`rocm-sdk-libraries`)** — native `.so` libraries (via
  `populate_runtime_files`) + Python bindings (via
  `populate_python_bindings`)
- No duplication of the Python binding files — they only go into the wheel.

## Standalone Wheel (`pip install .`)

Running `pip install .` from `rocm-libraries/projects/hipdnn/python/` builds
and installs the bindings directly into the active venv's site-packages:

1. pip sees `pyproject.toml` and invokes **scikit-build-core** as the build
   backend
2. scikit-build-core sets `SKBUILD=1` and runs cmake to build the nanobind
   extension
3. The `SKBUILD` branch in `CMakeLists.txt` installs to `.` (which
   scikit-build maps to `hipdnn_frontend/` via `wheel.install-dir` in
   `pyproject.toml`)
4. scikit-build-core packages everything into a `.whl`
5. pip installs the wheel into `site-packages/hipdnn_frontend/`

After that, `import hipdnn_frontend` just works. You need to pass cmake
defines so it can find hipDNN's dependencies:

```bash
cd rocm-libraries/projects/hipdnn/python
pip install . \
  -Ccmake.define.CMAKE_PREFIX_PATH=/opt/rocm
```

Or if using a TheRock build tree:

```bash
pip install . \
  -Ccmake.define.CMAKE_PREFIX_PATH=/path/to/TheRock/build/dist/rocm
```

## cmake Install Path

After `cmake --install --prefix /opt/rocm`, the bindings land at
`/opt/rocm/python_bindings/hipdnn_frontend/` which is **not** on Python's
search path. Options considered:

1. **PYTHONPATH** — `export PYTHONPATH=/opt/rocm/python_bindings:$PYTHONPATH`.
   Works but manual and fragile.
2. **`.pth` file in site-packages** — auto-adds the path, but installs
   outside the cmake prefix, ties to a specific Python version, and is hard
   to clean up.
3. **Just use the wheel** — cleanest approach; cmake install is for C/C++,
   pip is for Python.

The `python_bindings/` directory is an intermediate staging location for the
wheel builder, not a user-facing install path.

## CI Testing

- `rocm-sdk test` runs
  `libraries_test.py::testHipDNNFrontendImport` which imports
  `hipdnn_frontend`
- Skips gracefully if the package is not present in the build
- Already wired into the existing `test_rocm_wheels.yml` workflow — no CI
  changes needed
- Full CI chain: **build wheels** → **upload** → **pass find-links URL** →
  **test job pip installs** → **runs `rocm-sdk test`**
