# TH3R0CK Banner & Aliases Update - ROCm 7.11 Edition

## Summary

Updated .bashrc and .zshrc with enhanced F3D0R4-themed banners and comprehensive aliases for ROCm 7.11 + llama-server infrastructure.

## Changes Made

### 1. Enhanced Banners (Both .bashrc and .zshrc)

**Updated Title:**

- Changed from: `F3D0R4 43 // R0Cm D3V ST4T10N`
- Changed to: `F3D0R4 43 // TH3R0CK R0Cm 7.11 D3V ST4T10N`

**Real-time Service Status:**

- llama-server: Shows ✓ RUNNING or ✗ stopped with service URL
- Ollama: Shows ✓ RUNNING or ✗ stopped with service URL

**Updated Info Sections:**

- ROCm version now shows "7.11 TheRock"
- VRAM usage added to top banner section
- Service status shows actual systemd state instead of cached values

**Updated Command Reference Table:**

- Added `llm-bench` for benchmarking both servers
- Added `test-llama` for quick llama-server testing
- Added OI Lab Container section with Docker commands
- Reorganized commands for better clarity

### 2. Updated Aliases

#### llama-server Aliases (Now Primary Backend)

```bash
alias llama='systemctl status llama-server.service --no-pager'
alias llama-status='systemctl status llama-server.service --no-pager | head -15'
alias llama-start='sudo systemctl start llama-server'
alias llama-stop='sudo systemctl stop llama-server'
alias llama-restart='sudo systemctl restart llama-server'
alias llama-logs='journalctl -u llama-server -f'
alias llama-health='curl -s http://127.0.0.1:8080/health | jq .'
alias llama-chat='curl -s http://127.0.0.1:8080/v1/chat/completions ...'
alias llama-test='time curl -s http://127.0.0.1:8080/v1/chat/completions ...'
```

#### Enhanced GPU & ROCm 7.11 Aliases

```bash
alias vram='rocm-smi --showmeminfo vram 2>/dev/null | grep -E "VRAM|GPU"'
alias vram-watch='watch -n 1 "rocm-smi --showmeminfo vram | grep -E \"VRAM|Used|Total\""'
alias gpu-mon='watch -n 1 "rocm-smi && echo && echo VRAM: && rocm-smi --showmeminfo vram | grep -E \"Used|Total\" | head -2"'
alias gpu-full='echo -e "═══ ROCm 7.11 Info ═══" && rocminfo | grep -E "(Name|Agent|Marketing)" | head -10 && echo && echo -e "═══ GPU Status ═══" && rocm-smi'
alias rocm-status='echo -e "ROCm: 7.11 (TheRock gfx103X)" && echo -e "HIP: $(hipcc --version 2>/dev/null | head -1 | awk "{print \$2}")"'
alias rocm-native='echo "ROCm 7.11 - Native gfx1031 support ✓"'
```

#### Service Management

```bash
alias services='echo -e "═══ LLM Services ═══" && echo "llama-server: $(systemctl is-active llama-server.service)" && echo "Ollama: $(systemctl is-active ollama.service)"'
alias restart-llm='sudo systemctl restart llama-server && sudo systemctl restart ollama'
alias start-llm='sudo systemctl start llama-server && sudo systemctl start ollama'
alias stop-llm='sudo systemctl stop llama-server && sudo systemctl stop ollama'
```

#### Quick Info & Status

```bash
alias show-stack='echo -e "═══ AI/ML Stack ═══\nROCm: 7.11 (TheRock gfx103X native)\nllama-server: $(systemctl is-active llama-server.service) @ http://127.0.0.1:8080\nOllama: $(systemctl is-active ollama.service) @ http://localhost:11434\nOI Backend: llama-server (95 tok/sec)"'
alias show-vram='vram'
alias show-services='services'
```

#### Testing & Benchmarking

```bash
alias test-all='echo -e "═══ Testing ROCm 7.11 Stack ═══" && echo "GPU:" && rocminfo | grep -E "gfx103" | head -1 && echo && echo "llama-server:" && curl -s http://127.0.0.1:8080/health && echo && echo "Ollama:" && curl -s http://localhost:11434/api/tags 2>/dev/null | jq -r ".models | length"'
alias test-gpu-native='rocminfo | grep -E "gfx1031" && echo "✓ Native gfx1031 detected"'
alias test-rocm-711='rocminfo | grep -E "(gfx1031|ROCm)" | head -5'
alias test-llama='llama-test'
alias llm-bench='echo "llama-server:" && time llama-test && echo && echo "Ollama:" && time ollama run qwen2.5:0.5b "2+2=?" 2>/dev/null'
```

#### OI Lab Container (Docker)

```bash
alias lab-build='cd ~/TheRock/oi-lab-container && make build'
alias lab-start='cd ~/TheRock/oi-lab-container && make start'
alias lab-stop='cd ~/TheRock/oi-lab-container && make stop'
alias lab-shell='cd ~/TheRock/oi-lab-container && make shell'
alias lab-test='cd ~/TheRock/oi-lab-container && make test'
alias lab-logs='cd ~/TheRock/oi-lab-container && make logs'
alias lab-clean='cd ~/TheRock/oi-lab-container && make clean'
alias lab-ship='cd ~/TheRock/oi-lab-container && make ship'
```

#### Cheatsheets

```bash
alias cheat-llm='echo -e "═══ LLM Quick Reference ═══\noi → Open Interpreter (llama-server)\nllama-chat → Chat with llama-server\nollama-chat → Chat with Ollama\nllama-status → llama-server service\nollama-status → Ollama service\nvram → VRAM usage\nservices → All LLM services\ntest-all → Test full stack"'

alias cheat-rocm='echo -e "═══ ROCm 7.11 Commands ═══\ngpu / rocm-smi → GPU status\nvram → VRAM usage\nrocminfo → Full ROCm info\nrocm-status → ROCm version\ngpu-full → Complete diagnostic\ntest-gpu-native → Verify gfx1031"'
```

### 3. Open Interpreter Configuration

**Primary Backend Changed:**

- `oi` alias now uses llama-server by default (95 tok/sec)
- `oi-ollama` available for Ollama backend
- `oi-lms` available for LM Studio backend

### 4. External Aliases File

The comprehensive alias collection is also available in:

```
~/.config/therock_aliases_rocm711.sh
```

Both .bashrc and .zshrc source this file automatically.

## Key Features

1. **Performance-Focused**: llama-server now primary backend (2-20x faster than Ollama)
1. **ROCm 7.11 Native**: All commands highlight native gfx1031 support
1. **Real-time Status**: Service status queries systemd directly
1. **Comprehensive Testing**: Quick tests for GPU, llama-server, and Ollama
1. **F3D0R4 Theme**: Maintains Fedora blue color scheme throughout

## Usage

To see the new banner, open a new terminal or run:

```bash
banner
```

To see available commands:

```bash
cheat-llm    # LLM-related commands
cheat-rocm   # ROCm-related commands
```

To test the full stack:

```bash
test-all     # Tests GPU, llama-server, and Ollama
```

## Files Modified

1. `/home/hashcat/.bashrc`

   - Updated banner (lines 222-263)
   - Updated llama-server aliases (lines 291-300)
   - Enhanced GPU/ROCm aliases (lines 327-344)
   - Added service management (lines 364-368)
   - Added quick info (lines 370-373)
   - Enhanced testing (lines 375-386)
   - Added cheatsheets (lines 388-390)

1. `/home/hashcat/.zshrc`

   - Updated banner (lines 206-246)
   - Updated llama-server aliases (lines 276-285)
   - Enhanced GPU/ROCm aliases (lines 312-329)
   - Added service management (lines 349-353)
   - Added quick info (lines 355-358)
   - Enhanced testing (lines 360-371)
   - Added cheatsheets (lines 373-375)

1. `/home/hashcat/.config/therock_aliases_rocm711.sh`

   - Comprehensive standalone alias collection
   - Sourced by both .bashrc and .zshrc

## Next Steps

1. Open a new terminal to see the updated banner
1. Try the new aliases:
   - `vram` - Check VRAM usage
   - `services` - Check LLM service status
   - `test-all` - Test the full stack
   - `llm-bench` - Benchmark both servers
1. Use `cheat-llm` or `cheat-rocm` for command reference
