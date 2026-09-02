# WSL ROCDXG CI Stage

This document describes the multi-arch CI stage that builds the WSL-only
`rocdxg` bridge library from `rocm-systems`.

> [!NOTE]
> This page is about the WSL ROCDXG CI stage. For native Windows source builds,
> see [Windows Support](windows_support.md).

## Execution Model

The WSL ROCDXG job starts on a Windows runner but runs TheRock's Linux-side
build steps inside a WSL Ubuntu shell:

```mermaid
flowchart TD
  runner["Windows runner<br/>checkout + MSVC/SDK discovery"]
  wsl["Enter WSL Ubuntu shell"]
  deps["Linux base + compiler-runtime artifacts"]
  cfg["Inside WSL Ubuntu<br/>configure wsl-rocdxg<br/>amd-hip toolchain"]
  build["Inside WSL Ubuntu<br/>build stage-wsl-rocdxg<br/>publish ROCDXG artifact"]

  runner --> wsl
  deps --> cfg
  wsl --> cfg --> build
```

The Windows host is used for checkout, MSVC setup, and Windows SDK discovery.
The WSL shell handles artifact fetch, CMake configure, build, and artifact
upload. The workflow bridges the Windows SDK shared include path into WSL with
`WSLENV`.

## Build Topology

`wsl-rocdxg` is modeled as its own multi-arch stage and artifact group in
[`BUILD_TOPOLOGY.toml`](../../BUILD_TOPOLOGY.toml). The stage produces a
target-neutral artifact from the `rocm-systems` source set.

The relevant feature controls are:

- `THEROCK_ENABLE_WSL`: enables the WSL feature group.
- `THEROCK_ENABLE_WSL_ROCDXG`: enables the ROCDXG component.

The `wsl-rocdxg` artifact group depends on `base`, and the artifact depends on
`core-hip`. This lets the WSL stage bootstrap from the same inbound Linux
artifact pattern used by other downstream multi-arch stages and use TheRock's
`amd-hip` compiler toolchain for the nested ROCDXG build.

## Linux Inbound Artifacts

Although the job starts on a Windows runner, CMake runs inside WSL Ubuntu. That
means the configure and build environment is Linux userspace, so the stage
consumes Linux prebuilt artifacts, including the `base` artifact and compiler
runtime artifacts used by the `amd-hip` toolchain.

The portable Linux multi-arch workflow gives the WSL job a dependency on
`compiler-runtime` so those inbound artifacts exist before the WSL stage fetches
them.

## Configure Behavior

For the WSL stage, [`configure_stage.py`](../../build_tools/configure_stage.py)
is invoked with `--stage=wsl-rocdxg` and `--platform=linux`. The generated
CMake arguments include:

```text
-DTHEROCK_ENABLE_ALL=OFF
-DTHEROCK_ENABLE_WSL_ROCDXG=ON
```

Topology-derived inbound dependency features are also enabled for the stage,
including the `core-hip` dependency used for the `amd-hip` compiler toolchain.

## Workflow Differences

Compared with the portable Linux artifact workflow, the WSL workflow:

- Runs on the WSL-capable Windows runner.
- Sets up Ubuntu under WSL.
- Configures with `--platform=linux`.
- Builds `stage-${stage_name}` for the WSL stage.

Compared with the native Windows artifact workflow, the WSL workflow:

- Uses host Windows for checkout, MSVC setup, and Windows SDK discovery.
- Uses WSL-local Python and CMake tooling for artifact fetch, configure, build,
  and upload.
- Bridges the Windows SDK shared include path into the WSL shell with `WSLENV`.

## ROCDXG Build

TheRock does not build `rocdxg` directly. The wrapper project invokes the
`libhsakmt` CMake path in `rocm-systems` that produces the WSL `rocdxg`
library, then installs the WSL-built outputs back into the normal TheRock
artifact flow.

## Troubleshooting

- If the WSL build cannot find Windows SDK shared headers, check the Windows SDK
  discovery step and the `WSLENV` bridge into WSL.
- If `rocdxg` outputs are missing, check the `libhsakmt` WSL CMake path in
  `rocm-systems` and the wrapper install step.
