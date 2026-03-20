# ROCm Media Libraries

This directory contains media decoding libraries for AMD GPUs.

- **rocDecode** -- high-performance video decoding using VA-API
- **rocJPEG** -- high-performance JPEG decoding using VA-API

Both libraries depend on AMD Mesa for VA-API support and are only available
on Linux.

## Dependencies

Media libraries require the `THEROCK_ENABLE_SYSDEPS_AMD_MESA` option to be
enabled, which provides the bundled Mesa VA-API driver. Each library can also
be individually controlled:

- `-DTHEROCK_ENABLE_ROCDECODE=ON`
- `-DTHEROCK_ENABLE_ROCJPEG=ON`

Or disabled as a group:

- `-DTHEROCK_ENABLE_MEDIA_LIBS=OFF`

## Source Layout

The source code for both libraries lives in the
[rocm-systems](https://github.com/ROCm/rocm-systems) repository:

- `rocm-systems/projects/rocdecode`
- `rocm-systems/projects/rocjpeg`
