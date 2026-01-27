# TheRock

TheRock is a CMake super-project for building HIP and ROCm from source.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/ROCm/TheRock.git
cd TheRock
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 ./build_tools/fetch_sources.py

# Configure (adjust AMDGPU_FAMILIES for your GPU)
cmake -B build -GNinja -DTHEROCK_AMDGPU_FAMILIES=gfx1100

# Build
ninja -C build
```

See [README.md](README.md) for full setup and [docs/development/development_guide.md](docs/development/development_guide.md) for details.

## Development Workflows

### Build Directory Layout

Each component produces:

```
build/component/
├── build/    # CMake build tree
├── stage/    # Install tree (this component only)
├── dist/     # stage/ + runtime dependencies merged
└── stamp/    # Incremental build tracking
```

Final unified output: `build/dist/rocm/` - combined ROCm installation.

### Component Build Targets

Every component exposes these targets (replace `component` with actual name like `hipify`, `clr`, `rocblas`):

| Target                    | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| `ninja component`         | Full build (configure + build + stage + dist) |
| `ninja component+build`   | Rebuild after source changes                  |
| `ninja component+dist`    | Update artifacts without full rebuild         |
| `ninja component+expunge` | Clean slate - remove all intermediate files   |

### Common Development Patterns

**Iterate on a single component:**

```bash
# After making changes to component source
ninja -C build clr+build

# Force complete rebuild of one component
ninja -C build clr+expunge && ninja -C build clr
```

**Build subset of ROCm:**

```bash
cmake -B build -GNinja \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_HIPIFY=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100

ninja -C build
```

**Test a built component:**

```bash
# Run tests from unified distribution
LD_LIBRARY_PATH=build/dist/rocm/lib build/dist/rocm/bin/test_rocrand_basic

# Or use ctest
ctest --test-dir build
```

**Faster rebuilds with ccache:**

```bash
cmake -B build -GNinja \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100
```

**Debug build for specific component:**

```bash
cmake -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -Drocblas_BUILD_TYPE=RelWithDebInfo \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100
```

### Top-Level Targets

| Target                 | Purpose                                       |
| ---------------------- | --------------------------------------------- |
| `ninja` / `ninja dist` | Build everything, populate `build/dist/rocm/` |
| `ninja artifacts`      | Generate artifact directories and manifests   |
| `ninja archives`       | Create `.tar.xz` distribution archives        |
| `ninja expunge`        | Remove all build artifacts                    |

### Submodule Management

- Components are git submodules - use normal git within each
- `./build_tools/fetch_sources.py` resets all submodules and reapplies patches (**destructive** - commit first!)
- Recover lost work: check `git reflog` in affected submodule

### IDE Support

Generate combined compile_commands.json for IDE support:

```bash
cmake --build build --target therock_merged_compile_commands
```

## Code Quality

```bash
pip install pre-commit
pre-commit run              # staged files
pre-commit run --all-files  # all files
pre-commit install          # auto-run on commit
```

Hooks: Black (Python), clang-format (C++), mdformat (Markdown), actionlint (GitHub Actions).

## Style Guidelines

See the [docs/development/style_guides/](docs/development/style_guides/)
directory for each style guide:

- [README.md - General principles](docs/development/style_guides/README.md#general-principles)
- [bash_style_guide.md](docs/development/style_guides/bash_style_guide.md)
- [cmake_style_guide.md](docs/development/style_guides/cmake_style_guide.md)
- [github_actions_style_guide.md](docs/development/style_guides/github_actions_style_guide.md)
- [python_style_guide.md](docs/development/style_guides/python_style_guide.md)

**Python:**

- Use `pathlib.Path` for filesystem operations
- Add type hints to function signatures
- Use `argparse` for CLI with help text
- Don't assume cwd - use script-relative paths

**CMake:**

- Dependencies at super-project level ([docs/development/dependencies.md](docs/development/dependencies.md))
- Build phases: configure → build → stage → dist

## Git Workflow

**Branches:** `users/<username>/<description>` or `shared/<description>`

**PRs:** Target `main`, ensure workflows pass.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Project Structure

```
base/           # rocm-systems (driver, runtime foundations)
compiler/       # LLVM/Clang/LLD, device libraries
core/           # HIP, CLR, ROCr
math-libs/      # rocBLAS, rocFFT, etc.
ml-libs/        # MIOpen, composable_kernel
comm-libs/      # RCCL
profiler/       # rocprofiler, roctracer
build_tools/    # Python build scripts
cmake/          # CMake infrastructure
docs/           # Documentation
```

## Key Documentation

Reference the below for specialty tasks and deeper analysis, asking questions with subagents in order to avoid polluting context:

- [README.md](README.md) - Build setup, feature flags
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [docs/development/build_system.md](docs/development/build_system.md) - Build architecture
- [docs/development/development_guide.md](docs/development/development_guide.md) - Component development
- [docs/development/dependencies.md](docs/development/dependencies.md) - Dependency management

If development patterns become useful for certain development styles, prefer to document the salient details locally in this CLAUDE.md in addition to exhaustive documentation elsewhere.
