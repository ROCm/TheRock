# Git mirror setup for faster submodule checkouts

TheRock has 14+ submodules including large repositories like `llvm-project`,
`rocm-libraries`, and `rocm-systems`. The initial `git submodule update --init`
can take 15-25 minutes over the network. By maintaining local **bare mirror
repositories**, subsequent workspace setups across multiple checkouts become
near-instant.

## How it works

Git's `--reference` flag tells `git clone` to read objects from a local
repository before fetching from the network. When a mirror contains the
objects a submodule needs, the clone completes in seconds using local disk
reads instead of network downloads.

```text
Without mirrors:
  python fetch_sources.py
    -> fetches ~3GB+ for llvm-project from github.com       (10-20 min)
    -> repeats for each of the 14+ submodules

With mirrors (using fetch_sources.py --reference-dir):
  python fetch_sources.py --reference-dir ~/.rocm-git-mirrors
    -> selects the correct mirror for each submodule automatically
    -> reads most objects from local disk                    (~5 sec per submodule)
    -> fetches only missing objects (if any) from remote
```

The mirror directory layout follows the GitHub URL structure:

```text
~/.rocm-git-mirrors/
├── ROCm/
│   ├── llvm-project.git
│   ├── rocm-libraries.git
│   ├── rocm-systems.git
│   ├── half.git
│   ├── HIPIFY.git
│   ├── rocm-cmake.git
│   ├── SPIRV-LLVM-Translator.git
│   ├── rocprof-trace-decoder.git
│   ├── mesa-fork.git
│   ├── rocm-kpack.git
│   ├── libhipcxx.git
│   └── rocgdb.git
└── iree-org/
    ├── iree.git
    └── fusilli.git
```

## Quick start

```bash
cd TheRock

# Create mirrors for all submodules (one-time, takes 20-40 minutes)
python3 ./build_tools/setup_git_mirrors.py \
    --mirror-dir ~/.rocm-git-mirrors \
    --jobs 4

# Use mirrors when fetching sources
python3 ./build_tools/fetch_sources.py \
    --reference-dir ~/.rocm-git-mirrors
```

## Linux setup

### Initial mirror creation

```bash
# Choose a persistent location with enough disk space (~15-20 GB)
export THEROCK_GIT_MIRROR_DIR="$HOME/.rocm-git-mirrors"

# Create all mirrors (parallelized, with retry)
python3 ./build_tools/setup_git_mirrors.py \
    --mirror-dir "$THEROCK_GIT_MIRROR_DIR" \
    --jobs 4
```

### Keeping mirrors up to date

Mirrors should be updated periodically so they stay close to the remote HEAD.
The `--update` flag uses `git ls-remote` to check for new refs before fetching,
skipping mirrors that are already current.

```bash
# Manual update
python3 ./build_tools/setup_git_mirrors.py \
    --mirror-dir "$THEROCK_GIT_MIRROR_DIR" \
    --update --jobs 4
```

To automate updates, add a cron job:

```bash
# Update mirrors every 4 hours
crontab -e
```

Add the following line (adjust paths as needed):

```text
0 */4 * * * cd /path/to/TheRock && python3 ./build_tools/setup_git_mirrors.py --mirror-dir $HOME/.rocm-git-mirrors --update --jobs 4 >> /tmp/rocm-mirror-update.log 2>&1
```

### Setting the environment variable

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export THEROCK_GIT_MIRROR_DIR="$HOME/.rocm-git-mirrors"
```

With this set, `fetch_sources.py` will automatically use the mirrors without
needing `--reference-dir` on every invocation.

## Windows setup

### Initial mirror creation

Open a terminal (PowerShell or Git Bash) in the TheRock directory:

```powershell
# Choose a persistent location
$env:THEROCK_GIT_MIRROR_DIR = "C:\rocm-git-mirrors"

# Create all mirrors
python ./build_tools/setup_git_mirrors.py `
    --mirror-dir $env:THEROCK_GIT_MIRROR_DIR `
    --jobs 4
```

### Keeping mirrors up to date

```powershell
python ./build_tools/setup_git_mirrors.py `
    --mirror-dir $env:THEROCK_GIT_MIRROR_DIR `
    --update --jobs 4
```

To automate via Task Scheduler:

1. Open Task Scheduler
1. Create a new Basic Task named "ROCm Mirror Update"
1. Set the trigger to repeat every 4 hours
1. Set the action to run:
   - Program: `python`
   - Arguments: `C:\path\to\TheRock\build_tools\setup_git_mirrors.py --mirror-dir C:\rocm-git-mirrors --update --jobs 4`
   - Start in: `C:\path\to\TheRock`

### Setting the environment variable

Set it permanently via System Properties or PowerShell:

```powershell
[System.Environment]::SetEnvironmentVariable(
    "THEROCK_GIT_MIRROR_DIR",
    "C:\rocm-git-mirrors",
    "User"
)
```

## Usage with fetch_sources.py

Once mirrors are set up, there are two ways to use them:

### Option 1: Environment variable (recommended)

```bash
export THEROCK_GIT_MIRROR_DIR="$HOME/.rocm-git-mirrors"

# fetch_sources.py picks it up automatically
python3 ./build_tools/fetch_sources.py --jobs 12
```

### Option 2: Explicit flag

```bash
python3 ./build_tools/fetch_sources.py \
    --reference-dir ~/.rocm-git-mirrors \
    --jobs 12
```

Both options are fully backward compatible. If the mirror directory doesn't
exist or a specific submodule's mirror is missing, `fetch_sources.py`
falls back to normal network fetches automatically.

## Verifying mirrors

Check that all mirrors are healthy:

```bash
python3 ./build_tools/setup_git_mirrors.py \
    --mirror-dir ~/.rocm-git-mirrors \
    --verify
```

## Cleaning up stale mirrors

If submodules are removed from `.gitmodules`, prune orphaned mirrors:

```bash
python3 ./build_tools/setup_git_mirrors.py \
    --mirror-dir ~/.rocm-git-mirrors \
    --prune
```

## Working with multiple workspaces

The primary benefit of mirrors is that **multiple TheRock checkouts share the
same object store**. Once the mirrors are populated, creating a new workspace
is fast:

```bash
# First workspace (mirrors already populated)
git clone https://github.com/ROCm/TheRock.git workspace-1
cd workspace-1
python3 ./build_tools/fetch_sources.py --reference-dir ~/.rocm-git-mirrors

# Second workspace (same mirrors, near-instant submodule init)
cd ..
git clone https://github.com/ROCm/TheRock.git workspace-2
cd workspace-2
python3 ./build_tools/fetch_sources.py --reference-dir ~/.rocm-git-mirrors
```

## Troubleshooting

### Mirror creation fails with network errors

The script retries failed operations up to 3 times with exponential backoff.
To increase retries:

```bash
python3 ./build_tools/setup_git_mirrors.py \
    --mirror-dir ~/.rocm-git-mirrors \
    --retries 5
```

### Submodule update fails with reference

If a `--reference` clone fails (e.g., corrupted mirror), `fetch_sources.py`
automatically retries without `--reference`. You'll see a warning:

```text
WARNING: --reference clone failed for compiler/amd-llvm, retrying without reference...
```

To fix the mirror, re-create it:

```bash
# Remove the bad mirror
rm -rf ~/.rocm-git-mirrors/ROCm/llvm-project.git

# Re-create it
python3 ./build_tools/setup_git_mirrors.py \
    --mirror-dir ~/.rocm-git-mirrors
```

### Disk space

The full set of mirrors requires approximately 15-20 GB. The largest
repositories are:

- `llvm-project` (~3-4 GB)
- `rocm-libraries` (~2-3 GB)
- `rocm-systems` (~2-3 GB)
- `iree` (~1-2 GB)
- `rocgdb` (~1-2 GB)

### What happens if mirrors become stale?

Stale mirrors still provide value. Git reads available objects from the mirror
and fetches only the missing delta from the network. Even a mirror that's weeks
old will still speed up submodule init significantly because most of the git
history is already present locally.
