# ROCm Python Packaging via TheRock

Given ROCm artifacts directories, performs surgery to re-layout them for
distribution as Python packages and builds sdists and wheels as appropriate.

This process involves both a re-organization of the sources and some surgical
alterations.

## General Design

We generate three types of packages:

- Selector package: The `rocm` package is built as an sdist so that it is
  evaluated at the point of install, allowing it to perform some selection logic
  to detect dependencies and needs. Most other packages are installed by
  asking to install extras of this one (i.e. `rocm[libraries]`, etc).
  - The `rocm` package provides the `rocm-sdk` tool.
  - The `rocm` package uses the `import rocm_sdk` Python namespace as we do
    not want a barename `rocm`.
- Runtime packages: Most packages are runtime packages. These are wheel files
  that are ready to be expanded directly into a site-lib directory. They contain
  only the minimal set of files needed to run. Critically, they do not contain
  symlinks (instead, only SONAME libraries are included, and any executable
  symlinks are emitted by dynamically compiling an executable that can execle
  a relative binary).
- Devel package: The `rocm-sdk-devel` package is the catch-all for everything.
  For any file already populated in a runtime package, it will include it as
  a relative symlink (also rewriting shared library soname links as needed).
  Since symlinks and non-standard attributes cannot be included in a wheel file,
  the platform contents are stored in a `_devel.tar` or `_devel.tar.xz` file.
  The installed package is extended in response to requesting a path to it
  via the `rocm-sdk` tool.

Runtime packages can either be target neutral or target specific. Target specific
packages are suffixed with their target family and the setup files have special logic
to determine what is correct to load based on the system. In the future, this
will be re-arranged to have more of a thin structure where the host code is
always emitted as target neutral and separate device code packages are loaded
as needed.

It is expected that all packages are installed in the same site-lib, as they use
relative symlinks and RPATHs that cross the top-level package boundary. The
built-in tests (via `rocm-sdk test`) verify these conditions.

## Building Packages

### Target Family Selection

The target GPU family used when building packages is determined in the following order:

1. `ROCM_SDK_TARGET_FAMILY` environment variable, if set.
1. Dynamically discovered target family on the system.
1. The fallback default defined in `dist_info.DEFAULT_TARGET_FAMILY`.

For CI builds on CPU-only machines, this will typically fall through to (3).
For developer machines with GPUs, the dynamic detection in (2) should suffice,
but can be overridden by setting (1).

In case of multiple GPU architectures present, the dynamically discovered target family
defaults to the first available family, in which case it might be necessary to set the
`ROCM_SDK_TARGET_FAMILY` environment variable.

### Example

```bash
./build_tools/build_python_packages.py \
    --artifact-dir ./output-linux-portable/build/artifacts \
    --dest-dir ${HOME}/tmp/packages
```

Note that this does do some dynamic compilation of files and it performs
patching via patchelf. On Linux, it is recommended to run this in the same
portable container as was used to build the SDK (so as to avoid the possibility
of accidentally referencing too-new glibc symbols).

To install locally built packages you can either

1. Directly install the Python packages by file name:

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install ${HOME}/tmp/packages/dist/rocm-7.0.0.dev0.tar.gz \
               ${HOME}/tmp/packages/dist/rocm_sdk_core-7.0.0.dev0-py3-none-linux_x86_64.whl
   # Optionally install rocm_sdk_devel and rocm_sdk_libraries wheels
   ```

1. Generate a local index using [piprepo](https://pypi.org/project/piprepo/) and
   install using more natural package names:

   ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install piprepo setuptools
   piprepo build ${HOME}/tmp/packages/dist
   pip install rocm[libraries,devel]==7.0.0.dev0 \
     --extra-index-url ${HOME}/tmp/packages/dist/simple \
     --force-reinstall --no-cache-dir
   ```

## Using Packages from Frameworks

### Building Python Based Projects

Generally, Python based framework users will build against installed ROCm
Python development packages. For frameworks that only depend on the core ROCm
subset (including runtime, HIP, system libraries and critical path math
libraries), this is a process of installing the development packages like:

```bash
# See RELEASES.md for exact arch specific incantations, --index-url
# combinations, etc.
pip install rocm[libraries,devel]
```

Then build your framework by setting appropriate CMake settings or environment
variables. Examples (exact settings needed will vary by project being built):

```bash
-DCMAKE_PREFIX_PATH=$(rocm-sdk path --cmake)
-DROCM_HOME=$(rocm-sdk path --root)
export PATH="$(rocm-sdk path --bin):$PATH"
```

### Configuring your Python Project Dependencies

This is typically sufficient to *build* ROCm based Python projects. However,
because built projects will typically do dynamic linking to ROCm host libraries
and this will be done on arbitrary systems, it is also necessary to change the
project's packaging and initialization.

Taking PyTorch as an example, you want to inject an install requirement on
the ROCm python packages for a specific version (all projects will have their
own way to do this):

```bash
export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="rocm[libraries]=={$(rocm-sdk version)}"
```

### Performing Initialization

Typical ROCm dependent Python projects will have an `__init__.py` which precedes
using any extensions or native libraries which depend on ROCm. Ultimately,
this initializer will need to call `rocm_sdk.initialize_process()` to initialize
needed libraries. We recommend the following common idiom for managing this:

When building from ROCm wheels, add a `_rocm_init.py` to the root of the
project such that it is included in built wheels. Then in your `__init__.py`
add code like this:

```python
try:
    from . import _rocm_init
except ModuleNotFoundError:
    pass
else:
    _rocm_init.initialize()
    del _rocm_init
```

Generate a `_rocm_init.py` file like this (using any suitable scripting):

```bash
echo "
def initialize():
  import rocm_sdk
  rocm_sdk.initialize_process(preload_shortnames=[
    'amd_comgr',
    'amdhip64',
    'roctx64',
    'hiprtc',
    'hipblas',
    'hipfft',
    'hiprand',
    'hipsparse',
    'hipsolver',
    'rccl',
    'hipblaslt',
    'miopen',
  ],
  check_version='$(rocm-sdk version)')
" > torch/_rocm_init.py
```

Note that you must preload any libraries that your native extensions depend on
so that when the operating system attempts to resolve linkage, they are already
in the namespace. The above is an example that, at the time of writing, was
suitable for PyTorch. Note that it version locks to a specific ROCm SDK version.
This is generally appropriate for development/nightly builds. For production
builds, you will want to lock to a major/minor version only and use a wildcard
(\*) to match the suffix.

By default, on version mismatch, a Warning will be raised. You can pass
`fail_on_version_mismatch=True` to make this an exception.

### Dynamic Library Resolution

The above procedure works for libraries that are hard dependencies of native
extensions or their deps. If code needs to dynamically load libraries, typically
preloading them in this way will work, and then a subsequent `dlopen()` or
equiv will find them. However, the `rocm_sdk.find_libraries(*shortnames)`
entrypoint is also provided and can be used to query an OS independent
absolute path to a given named library that is known to the distribution.

### Testing

The `rocm` distribution, if installed, bundles self tests which verify
API contracts and file/directory layout. Since the Python ecosystem is ever
evolving, and packages like this use several "adventurous" features, we
bundle the ability for detailed self checks. Run them with:

```bash
rocm-sdk test
```

If you are having any problem with your ROCm Python installation, it is
recommended to run this to verify integrity and basic functionality.

Providing the output of `rocm-sdk test` in any bug reports is greatly
appreciated.
