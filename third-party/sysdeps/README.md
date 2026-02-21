# Bundled System Dependencies (sysdeps)

Sysdeps are libraries (zlib, zstd, elfutils, libdrm, numactl, etc.) that
we build from source with special packaging treatment — SONAME rewriting,
symbol versioning, and installation into `lib/rocm_sysdeps/` — so that ROCm can
ship private copies without conflicting with system-installed versions.

This is distinct from the other third-party libraries under `third-party/` (fmt,
spdlog, flatbuffers, etc.), which are build dependencies built as normal CMake
targets. See [Dependencies](/docs/development/dependencies.md) for
the full explanation of the two categories.
