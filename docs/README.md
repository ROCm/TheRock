# TheRock Documentation

Welcome to TheRock documentation. This directory contains comprehensive guides, references, and technical documentation for building, developing, and using TheRock.

## Quick Links

- [Main README](../README.md) - Project overview and quick start
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [Roadmap](../ROADMAP.md) - Project roadmap and future plans
- [Releases](../RELEASES.md) - Release notes and changelog

## Documentation Structure

### 📚 User Guides

Essential guides for users and developers working with TheRock:

- **[Custom Build Guide](guides/custom-build.md)** - Building TheRock with custom configurations
- **[Environment Setup](guides/environment-setup.md)** - Setting up your development environment
- **[SELinux + ROCm Setup](guides/selinux-rocm-setup.md)** - Configuring SELinux for ROCm
- **[Package Updates Guide](guides/package-updates.md)** - Safely updating Python packages
- **[Open Interpreter Best Practices](guides/open-interpreter-best-practices.md)** - Working with Open Interpreter on TheRock

### 🔧 Development

Documentation for TheRock developers:

- **[Development Guide](development/development_guide.md)** - Getting started with TheRock development
- **[Build System](development/build_system.md)** - Understanding the build system
- **[Dependencies](development/dependencies.md)** - Managing dependencies
- **[Adding Tests](development/adding_tests.md)** - Writing and running tests
- **[Test Harness](development/therock_test_harness.md)** - Using the test harness
- **[Test Debugging](development/test_debugging.md)** - Debugging test failures
- **[Test Environment Reproduction](development/test_environment_reproduction.md)** - Reproducing test environments
- **[Style Guide](development/style_guide.md)** - Code style guidelines
- **[Sanitizers](development/sanitizers.md)** - Using sanitizers for debugging
- **[Git Chores](development/git_chores.md)** - Git workflows and maintenance
- **[GitHub Actions Debugging](development/github_actions_debugging.md)** - Debugging CI/CD
- **[Build Containers](development/build_containers.md)** - Using build containers
- **[Artifacts](development/artifacts.md)** - Managing build artifacts
- **[Installing Artifacts](development/installing_artifacts.md)** - Installing build artifacts
- **[Windows Support](development/windows_support.md)** - Windows development

### 📦 Packaging

Documentation for building and distributing packages:

- **[Native Packaging](packaging/native_packaging.md)** - Creating native packages
- **[Python Packaging](packaging/python_packaging.md)** - Python package creation
- **[Versioning](packaging/versioning.md)** - Version numbering and management

### 🏗️ Design & Architecture

Design documents and architectural decisions:

- **[Manylinux Builds](design/manylinux_builds.md)** - Manylinux compatibility

### 📋 RFCs (Requests for Comments)

Formal proposals for major changes:

- **[RFC Index](rfcs/README.md)** - Overview of all RFCs
- **[RFC0001](rfcs/RFC0001-BLAS-Stack-Build-Improvements.md)** - BLAS Stack Build Improvements
- **[RFC0002](rfcs/RFC0002-MonoRepo-Gardener-Rotations.md)** - MonoRepo Gardener Rotations
- **[RFC0003](rfcs/RFC0003-Build-Tree-Normalization.md)** - Build Tree Normalization
- **[RFC0004](rfcs/RFC0004-Fusilli-IREE-Kernel-Provider-hipDNN.md)** - Fusilli IREE Kernel Provider
- **[RFC0005](rfcs/RFC0005-hipDNN-Project-Integration.md)** - hipDNN Project Integration
- **[RFC0006](rfcs/RFC0006-libhipcxx-ROCm-Core-Inclusion.md)** - libhipcxx ROCm Core Inclusion
- **[RFC0007](rfcs/RFC0007-rdc-therock-integration.md)** - RDC TheRock Integration
- **[RFC0008](rfcs/RFC0008-Multi-Arch-Packaging.md)** - Multi-Arch Packaging

### 🔍 Troubleshooting

Solutions to common issues:

- **[Browser Search Fix](troubleshooting/browser-search-fix.md)** - Fixing Open Interpreter browser.search()

### ⚙️ Custom Configuration

Custom setup and configuration notes:

- **[Banner & Aliases Update](custom/banner-aliases-update.md)** - ROCm 7.11 banner and shell aliases

### 📊 Reference

Technical reference documentation:

- **[Supported GPUs](supported-gpus.md)** - List of supported AMD GPUs
- **[Optimization Plan](optimization-plan.md)** - Performance optimization roadmap

## Contributing to Documentation

When adding new documentation:

1. **Choose the right location:**

   - User-facing guides → `guides/`
   - Developer documentation → `development/`
   - Troubleshooting/fixes → `troubleshooting/`
   - Design documents → `design/`
   - Major proposals → `rfcs/`
   - Custom configurations → `custom/`

1. **Use clear filenames:**

   - Use kebab-case (lowercase with hyphens)
   - Be descriptive but concise
   - Example: `custom-build.md`, `browser-search-fix.md`

1. **Update this index:**

   - Add your new document to the appropriate section
   - Include a brief description

1. **Follow markdown conventions:**

   - Use proper heading hierarchy (# → ## → ###)
   - Include a table of contents for long documents
   - Add code examples where relevant
   - Link to related documentation

## Getting Help

If you can't find what you're looking for:

1. Check the [Main README](../README.md) for an overview
1. Search this directory: `grep -r "your search term" docs/`
1. Check the [CONTRIBUTING](../CONTRIBUTING.md) guide
1. Open an issue on GitHub

## Documentation Standards

- **Markdown format** - All docs use GitHub-flavored markdown
- **Code examples** - Include working code samples
- **Testing** - Test all commands and procedures
- **Updates** - Keep docs in sync with code changes
- **Clear language** - Write for the target audience
