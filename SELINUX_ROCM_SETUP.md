# SELinux Configuration for ROCm in /opt/rocm

## Current Issue

Your `/opt/rocm` has the wrong SELinux context:

```
Current:  unconfined_u:object_r:user_home_t:s0
Correct:  system_u:object_r:lib_t:s0
```

The `user_home_t` context is for home directories, not system libraries. This can cause SELinux denials when services try to access ROCm.

## Fix SELinux Context for /opt/rocm

### After Installing New ROCm 7.11 Build

Once your build completes and you install to `/opt/rocm`, run:

```bash
# Set correct SELinux context for /opt/rocm
sudo semanage fcontext -a -t lib_t "/opt/rocm(/.*)?"
sudo restorecon -Rv /opt/rocm

# Verify the context
ls -Z /opt/rocm
```

**Expected result:**

```
system_u:object_r:lib_t:s0 bin
system_u:object_r:lib_t:s0 lib
system_u:object_r:lib_t:s0 lib64
...
```

### Make Executables Executable

For binaries that need to run:

```bash
# Mark ROCm binaries as executable libraries
sudo semanage fcontext -a -t bin_t "/opt/rocm/bin(/.*)?"
sudo semanage fcontext -a -t bin_t "/opt/rocm/llvm/bin(/.*)?"
sudo restorecon -Rv /opt/rocm/bin /opt/rocm/llvm/bin
```

### Allow GPU Device Access

ROCm needs access to `/dev/kfd` and `/dev/dri/*`:

```bash
# Check current context
ls -Z /dev/kfd /dev/dri/

# These should already be correct, but verify:
# /dev/kfd should be: system_u:object_r:device_t:s0
# /dev/dri/* should be: system_u:object_r:xserver_misc_device_t:s0
```

## SELinux Policy for ROCm Services

### For Ollama Service

Create custom policy if needed:

```bash
# Check for denials
sudo ausearch -m avc -ts recent | grep ollama

# If denials exist, generate policy
sudo ausearch -m avc -ts recent | grep ollama | audit2allow -M ollama_rocm

# Review the policy
cat ollama_rocm.te

# Install if safe
sudo semodule -i ollama_rocm.pp
```

### For LMStudio Service

```bash
# Check for denials
sudo ausearch -m avc -ts recent | grep lm-studio

# Generate policy if needed
sudo ausearch -m avc -ts recent | grep lm-studio | audit2allow -M lmstudio_rocm
sudo semodule -i lmstudio_rocm.pp
```

### For llama-server

```bash
# Check for denials
sudo ausearch -m avc -ts recent | grep llama-server

# Generate policy if needed
sudo ausearch -m avc -ts recent | grep llama-server | audit2allow -M llamaserver_rocm
sudo semodule -i llamaserver_rocm.pp
```

## Container Strategy

Your plan makes perfect sense! Here's the breakdown:

### Scenario 1: Native ROCm 7.11 (Testing)

**Location:** `/opt/rocm` (your new build)
**Purpose:** Testing and development
**Services:** All (Ollama, LMStudio, llama-server)

**Installation after build:**

```bash
cd /home/hashcat/TheRock
cmake --build build --target install

# This should install to /opt/rocm via:
# cmake --install build --prefix /opt/rocm
# or
# pkexec cmake --install build --prefix /opt/rocm
```

**Fix SELinux:**

```bash
sudo semanage fcontext -a -t lib_t "/opt/rocm(/.*)?"
sudo semanage fcontext -a -t bin_t "/opt/rocm/bin(/.*)?"
sudo semanage fcontext -a -t bin_t "/opt/rocm/llvm/bin(/.*)?"
sudo restorecon -Rv /opt/rocm
```

### Scenario 2: Containerized ROCm 6.2 (Fallback/LMStudio)

**Location:** Container with `/opt/rocm-6.2` mounted
**Purpose:** Stable fallback for LMStudio
**Base:** Use official ROCm 6.2 container or build custom

## Container Configuration

### Option A: Official ROCm Container

```dockerfile
FROM rocm/dev-ubuntu-22.04:6.2

# Install LMStudio dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libstdc++6 \
    wget \
    ca-certificates

# Copy LMStudio
COPY lm_studio.appimage /opt/lmstudio/
RUN chmod +x /opt/lmstudio/lm_studio.appimage

# ROCm environment
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV HSA_ENABLE_SDMA=0
ENV HSA_XNACK=0

# Expose LMStudio port
EXPOSE 1234

CMD ["/opt/lmstudio/lm_studio.appimage", "--run-as-service"]
```

### Option B: Custom Fedora + ROCm 6.2

```dockerfile
FROM fedora:39

# Install ROCm 6.2 from AMD repo
RUN dnf install -y \
    'dnf-command(config-manager)' && \
    dnf config-manager --add-repo \
    https://repo.radeon.com/rocm/rhel9/6.2/main && \
    dnf install -y rocm-hip-sdk

# Add LMStudio, configure as above
```

## Docker/Podman SELinux Configuration

### For GPU Access in Container

```bash
# Allow containers to access GPU devices
sudo setsebool -P container_use_devices 1

# Run container with GPU access (Podman)
podman run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=disable \
  -v /opt/rocm-6.2:/opt/rocm:ro \
  -p 127.0.0.1:1234:1234 \
  lmstudio-rocm62:latest

# Or with Docker
docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt label=type:container_runtime_t \
  -v /opt/rocm-6.2:/opt/rocm:ro \
  -p 127.0.0.1:1234:1234 \
  lmstudio-rocm62:latest
```

### SELinux Context for Container Volumes

```bash
# If mounting host ROCm into container
sudo semanage fcontext -a -t container_file_t "/opt/rocm-6.2(/.*)?"
sudo restorecon -Rv /opt/rocm-6.2
```

## Recommended Deployment Strategy

### Phase 1: Build & Test Native ROCm 7.11

1. ✅ Complete current build (gfx103X family)
1. Install to `/opt/rocm`
1. Fix SELinux contexts
1. Test with Ollama, LMStudio, llama-server
1. Monitor for 1-2 weeks

### Phase 2: Create ROCm 6.2 Container (Fallback)

1. Build container with ROCm 6.2
1. Configure for LMStudio specifically
1. Test GPU access in container
1. Keep as stable fallback

### Phase 3: Optimized ROCm 7.11 (Future)

1. Clone to `/home/hashcat/TheRock-gfx103X-optimized`
1. Build with march=native + LTO
1. Install to `/opt/rocm-optimized`
1. A/B test performance

## Directory Structure Plan

```
/opt/
├── rocm/                    # ROCm 7.11 baseline (current build)
├── rocm-6.2/                # ROCm 6.2 for containers (fallback)
├── rocm-optimized/          # Future optimized 7.11 build
└── rocm.backup-*/           # Your existing backups

/home/hashcat/
├── TheRock/                 # Current 7.11 baseline build
├── TheRock-gfx103X-optimized/  # Future optimized build
└── containers/
    ├── lmstudio-rocm62/     # LMStudio container
    ├── ollama-rocm62/       # Ollama container (optional)
    └── llamacpp-rocm62/     # llama.cpp container (optional)
```

## Testing Checklist After ROCm 7.11 Installation

```bash
# 1. Check SELinux contexts
ls -Z /opt/rocm/{bin,lib,lib64}

# 2. Test ROCm tools
/opt/rocm/bin/rocminfo
/opt/rocm/bin/rocm-smi

# 3. Check for SELinux denials
sudo ausearch -m avc -ts today | grep rocm

# 4. Test Ollama
systemctl restart ollama
curl http://localhost:11434/api/tags

# 5. Test LMStudio
systemctl --user restart lmstudio-server
curl http://localhost:1234/v1/models

# 6. Check GPU detection
rocm-smi
```

## Troubleshooting SELinux Issues

### Check for Denials

```bash
# Real-time monitoring
sudo tail -f /var/log/audit/audit.log | grep denied

# Recent denials
sudo ausearch -m avc -ts recent
```

### Generate Policy from Denials

```bash
# Collect denials for specific service
sudo ausearch -m avc -ts today | grep <service-name> | audit2allow -M <policy-name>

# Review generated policy
cat <policy-name>.te

# Install if safe
sudo semodule -i <policy-name>.pp
```

### Temporary Permissive Mode (Testing Only!)

```bash
# Make specific domain permissive (safer)
sudo semanage permissive -a <domain_t>

# Restore to enforcing
sudo semanage permissive -d <domain_t>

# NEVER do this on production:
# sudo setenforce 0  # BAD - disables all SELinux!
```

______________________________________________________________________

**Created:** 2025-11-28
**Purpose:** SELinux configuration for ROCm 7.11 installation and container fallback strategy
