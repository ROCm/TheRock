# Building ROCm with Sanitizers

## Basic Usage

Sanitizers can be enabled via the `THEROCK_SANITIZER` variable. We will be extending this to support all sanitizers. Presently supported sanitizers are:

- `ASAN` : Enables ASAN, using a shared library ASAN runtime throughout (i.e. everything links to the appropriate support library).
- `OFF` : Explicitly disable sanitizers.

The sanitizer selection can be controlled per project by using a variable of the form `{subproject}_SANITIZER={VALUE}`. This is most commonly used to disable santiziers for specific projects once enabled globally.

Because ROCm includes a compiler and uses multiple toolchains to build, there are only certain configurations of the project that support ASAN: generally, we allow ASAN to be enabled for any component that is built with the ROCm version of LLVM. This ensures that we only link to a single ASAN support library (each process can have only one) and can setup RPATH entries and other settings so that anything so compiled will function without further flags, preloads, path settings, etc. We are in the process of switching more of the project to bootstrap off of the built-in compiler, which will make more of the project capable of being instrumented out of the box with a sanitizer.

### CMake Preset

In order to simplify use, the following presets are available for setting up specific sanitizer strategies:

- `--preset linux-release-asan`: Our default "ASAN enabled release" builds that we use for most build and test pipelines. This enables ASAN globally and then selectively disables it for the compiler and certain system libraries that are not yet ready for generic sanitizer builds.
- TODO: compiler-asan preset: We will enable a build mode such that the compiler and base libraries can also be instrumented. We will use this for qualifying compiler builds but not generally for *using* the compiler.

## Sanitizer Aware Project Development

### GFX Target Munging

Certain GFX targets have special hardware support for device-side ASAN. This is enabled transparently where available by changing the `GPU_TARGETS` variable propagated to sup-projects:

- `gfx942` -> `gfx942:xnack+`
- `gfx950` -> `gfx950:xnack+`

Some sub-projects have strict gfx target checks that do not allow these extends target specifiers. In general, sub-projects should take targets as given vs trying to guess whether some should be ASAN enabled.

### CMake Variables

When a project is configured for sanitizers, it will have certain variables injected into it. While it is often possible to not require projects to have any special knowledge of what sanitizer they were compiled for, some do need to know. For these cases, it can be necessary to use these special variables so injected.

- `THEROCK_SANITIZER={ASAN}` : Set if a sanitizer is active for the project and indicates which one.
- `THEROCK_SANITIZER_LAUNCHER` : If invoking certain tools at build time that dynamically link to a shared library compiled with a sanitizer, you need to prefix it with this value as `${THEROCK_SANITIZER_LAUNCHER}` (not surrounded in quotes so it can expand to multiple terms). This is most commonly needed for invoking system-python and importing native extensions that were built in the project with ASAN.

You are recommended to code defensively with patterns like:

```
if(NOT DEFINED THEROCK_SANITIZER)
  set(THEROCK_SANITIZER)
endif()
if(NOT DEFINED THEROCK_SANITIZER_LAUNCHER)
  set(THEROCK_SANITIZER_LAUNCHER)
endif()
```

Note that you may need to add additional environment variable special casing to support all launch modes. For example, when using system python, it is common to disable the leak sanitizer because non-debug Python (by design) does not deallocate on exit.

Example:

```
if(NOT DEFINED THEROCK_SANITIZER_LAUNCHER)
  set(THEROCK_SANITIZER_LAUNCHER)
endif()
set(PYTHON_LAUNCHER ${THEROCK_SANITIZER_LAUNCHER} "${Python3_EXECUTABLE}")
if(THEROCK_SANITIZER STREQUAL "ASAN")
  list(PREPEND PYTHON_LAUNCHER "${CMAKE_COMMAND}" -E ASAN_OPTIONS=detect_leaks=0 --)
endif()

# Then use ${PYTHON_LAUNCHER} (without quotes) in command lists.
```

We may add more built-in helpers if we see a lot of common usages like this. However, if history is a guide, projects that need to special case, often need to do so in bespoke ways that are hard to make common.

### Compile Time Checks

If needing to activate specific codepaths in ROCm libraries specifically for ASAN, CMake plumbing can be done to set ad-hoc defines, but it is highly recommended to instead use built-in compile definitions for this. For example:

```
#if defined(__has_feature)
# if __has_feature(address_sanitizer)
    // Code that builds only when AddressSanitizer is enabled
#  define ASAN_ENABLED
# endif
#endif

// Later in your code:
#ifdef ASAN_ENABLED
    // Perform AddressSanitizer-specific actions
#endif
```

Note that this only works for clang so must not be done in public/API headers that may be used by any consumer.

## Implementation Details

Handling sanitizer builds can be very complicated because of the need to fit all details together. We do it in several places:

- During toolchain selection, routines are called in `therock_sanitizers.cmake` to compute and set variables/options for the sub-project.
- In `therock_subproject_dep_provider.cmake` (TODO: rename to `therock_subproject_init.cmake` to signify what it has become), we do initialization within the project. This includes:
  - Handling the special `THEROCK_INCLUDE_CLANG_RESOURCE_DIR_RPATH` injected variable which tells us to query `clang` for its resource directory and setup both BUILD and INSTALL RPATH extensions so that we can always find the sanitizer runtime.
  - This augments the lists that are used in `therock_global_post_subproject.cmake` to walk through all targets and setup RPATHs.
    - `THEROCK_PRIVATE_INSTALL_RPATH_DIRS`
    - `THEROCK_PRIVATE_INSTALL_RPATH_DIRS`
  - Setup `THEROCK_SANITIZER_LAUNCHER`

## Troubleshooting

TODO: Add troubleshooting tips here.
