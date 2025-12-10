# Open Interpreter Best Practices

## 🔧 What Works & What Doesn't

### ✗ BROKEN - Never Use These

| Function                  | Status    | Reason                        |
| ------------------------- | --------- | ----------------------------- |
| `computer.files.search()` | ❌ BROKEN | `aifs` module not initialized |
| `computer.os` module      | ❌ BROKEN | Import fails                  |

### ⚠️ LIMITED - SSH/No GUI Environment

These require GUI/X11 which we don't have in SSH:

| Function                  | Status     | Alternative                                 |
| ------------------------- | ---------- | ------------------------------------------- |
| `computer.display`        | ⚠️ Limited | Use `subprocess` for screenshots if needed  |
| `computer.clipboard`      | ⚠️ Limited | Use file I/O instead                        |
| `computer.browser`        | ⚠️ Limited | Use `requests`, `curl`, or headless browser |
| `computer.keyboard/mouse` | ⚠️ Limited | Not applicable in SSH                       |
| `computer.vision`         | ⚠️ Limited | May work but needs models                   |

### ✓ ALWAYS USE - Reliable Alternatives

#### File Operations

```python
# List files
import os

files = os.listdir(".")

# Using pathlib (recommended)
from pathlib import Path

files = list(Path(".").iterdir())

# Find files by pattern
from pathlib import Path

python_files = list(Path(".").glob("**/*.py"))

# Read file
with open("file.txt", "r") as f:
    content = f.read()

# Or with pathlib
from pathlib import Path

content = Path("file.txt").read_text()
```

#### Shell Commands

```python
# Simple command
import subprocess

result = subprocess.run(["ls", "-la"], capture_output=True, text=True)
print(result.stdout)

# With shell=True (be careful with user input!)
result = subprocess.run("ls -la | grep py", shell=True, capture_output=True, text=True)
print(result.stdout)

# Quick and dirty
import os

os.system("ls -la")
```

#### Directory Operations

```python
import os
from pathlib import Path

# Create directory
Path("/tmp/test").mkdir(parents=True, exist_ok=True)

# Walk directory tree
for root, dirs, files in os.walk("."):
    for file in files:
        print(os.path.join(root, file))

# Check if file/dir exists
Path("file.txt").exists()
Path("dir").is_dir()
```

## 🖥️ VM Management

Use the `vm-helper` command:

```bash
# List VMs
vm-helper list

# Create VM: name, RAM(MB), vCPUs, Disk(GB)
vm-helper create test-vm 4096 4 50

# Start/stop
vm-helper start test-vm
vm-helper stop test-vm

# Destroy (delete)
vm-helper destroy test-vm

# Info
vm-helper info test-vm
vm-helper status
```

## 💡 Pro Tips

1. **Prefer Python stdlib** over `computer.*` functions
1. **Use `subprocess.run()`** for shell commands (more reliable than `os.system()`)
1. **Use `pathlib`** instead of `os.path` (more Pythonic)
1. **Check file existence** before operations to avoid errors
1. **Use `with` statements** for file operations (auto-closes files)

## 🎯 Common Tasks

### Find all Python files

```python
from pathlib import Path

py_files = list(Path(".").rglob("*.py"))
for f in py_files:
    print(f)
```

### Search file content

```python
import subprocess

result = subprocess.run(["grep", "-r", "pattern", "."], capture_output=True, text=True)
print(result.stdout)
```

### Create and write file

```python
from pathlib import Path

Path("/tmp/test.txt").write_text("Hello World\n")
```

### Get file info

```python
from pathlib import Path
import os

p = Path("file.txt")
print(f"Size: {p.stat().st_size} bytes")
print(f"Modified: {os.path.getmtime(p)}")
print(f"Exists: {p.exists()}")
```

## 🚀 System Info

- **GPU:** AMD RX 6700 XT (gfx1031/RDNA2) with 12GB VRAM
- **ROCm:** 7.11 (custom build)
- **Environment:** SSH (no GUI)
- **Shell:** bash/zsh
- **Python:** 3.14 (in venv)

## 📝 Notes

- These guidelines are in your OI profile: `~/.config/open-interpreter/profiles/llama-server.yaml`
- OI will automatically follow these best practices
- No need to tell OI about these - it knows!
