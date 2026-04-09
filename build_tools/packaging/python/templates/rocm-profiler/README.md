# rocm-profiler

ROCm profiler applications package.

This Python wheel provides console entry points for the following ROCm
profiling tools:

- `rocprof-compute`
- `rocprof-sys-avail`
- `rocprof-sys-causal`
- `rocprof-sys-instrument`
- `rocprof-sys-python`
- `rocprof-sys-run`
- `rocprof-sys-sample`

## Purpose

This package exists to separate ROCm profiling tools from the core SDK
distribution. It provides Python console script wrappers that dispatch to
platform-specific binaries packaged within the wheel.

The actual profiler binaries are populated into the wheel during the ROCm
packaging pipeline.

## Runtime Behavior

Each console script resolves the packaged ROCm installation root within the
wheel and executes the corresponding binary under `bin/`.

If the wheel does not contain a populated runtime (e.g., template-only build),
invoking a console script will result in a `FileNotFoundError`.

## Packaging Notes

- No `install_requires` are declared.
- Dependencies are managed by the `rocm` meta package.
- Versioning is centrally managed via `rocm_sdk._dist_info`.

When integrated into the full ROCm packaging pipeline, this package will be
built via `build_python_packages.py`.
