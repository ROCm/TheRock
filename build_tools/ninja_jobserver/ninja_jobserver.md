# Ninja jobserver support for TheRock

This folder contains a small, **Python-only** build entrypoint that makes Ninja correctly participate in the **GNU Make jobserver** when you choose to drive TheRock builds through `make -jN`.

## Why jobserver matters

When we have a “top-level” orchestrator (eg. GNU `make`) running with a global parallelism limit (e.g. `make -j32`), we want *all* nested build tools to cooperate with that same limit. GNU make provides this via the **jobserver** mechanism and exports the necessary details through `MAKEFLAGS` (e.g. `--jobserver-auth=...`).

Ninja supports joining the GNU make jobserver **automatically**, but only if:
- `MAKEFLAGS` contains `--jobserver-auth=...` (meaning an outer `make -jN` started a jobserver), and
- Ninja is **not** forced to use a separate job count via `-j`.

If Ninja is invoked with `-j`, it will run with its own parallelism and will not coordinate with the outer jobserver, which can lead to oversubscription (too many concurrent compile/link steps across nested builds).

## Currently in the TheRock

- TheRock’s standard build flow (`cmake --build ...`) typically does **not** require any core repo changes for jobserver itself.
- The biggest jobserver requirement is **wrapper scripts / CI steps** that always pass `-j$(nproc)` (or `ninja -j ...`). That disables jobserver coordination when you *do* run under `make -jN`.
- GitHub Actions does **not** create a jobserver by default. so we need to explicitly run the build under `make -jN`

## What these scripts do ? 

### `build_the_rock.py`
A jobserver-aware driver that runs:

1) `cmake -S <repo_root> -B <build_dir> -GNinja ...`  
2) `cmake --build <build_dir> -- [ninja args]`

It decides whether to pass `-j` to Ninja:

- **If jobserver is present** (`MAKEFLAGS` contains `--jobserver-auth=`):  
  it **does not** pass `-j`, so Ninja can join the jobserver pool.

- **If jobserver is not present** ( may be in CI shell step):  
  it passes `-j<cpu_count>` (or `-j$JOBS`) for fast standalone builds.

### ` build_therock_withJobserver.py`
A simple Python wrapper that starts jobserver mode by running:

- `make -jN therock`

We still need a top-level `Makefile` with a `therock` target that calls `build_the_rock.py`.

---
#### Local usage

- Standalone (no jobserver):  
  ```bash
  python3 build_tools/ninja_jobserver/build_the_rock.py all
  ```
  or
  ```bash
  make therock
  ```

- Jobserver-coordinated build:  
  ```bash
  make -j32 therock
  ```
  Here GNU make creates the jobserver and exports `MAKEFLAGS=...--jobserver-auth=...`.
  The Python driver detects that and avoids `-j`, allowing Ninja to join the pool.

This gives us:

- `build_with_jobserver.py` runs `make -jN therock`, which creates the jobserver and sets `MAKEFLAGS`.
- The Makefile runs `build_the_rock.py all`.
- `build_the_rock.py` sees the jobserver and does **not** pass `-j` to Ninja, so Ninja participates in the same job pool.

---

## Guidance and pitfalls

### ✅ Do
- Use **one** top-level place to set parallelism:
  - `make -jN therock` (jobserver mode), OR
  - run `build_the_rock.py` directly and let it pass `-j` itself.

### ❌ Don’t (when using jobserver)
- Don’t pass `-j` to Ninja explicitly in jobserver mode:
  - avoid `cmake --build ... -- -j...`
  - avoid `ninja -j...`

Those override the jobserver and can oversubscribe CPU.

---

## Quick sanity test

### Local jobserver detection test

```bash
make -j8 therock
```

In output from `build_the_rock.py`, you should see a line similar to:

- `Jobserver detected -> NOT passing -j to Ninja ...`


