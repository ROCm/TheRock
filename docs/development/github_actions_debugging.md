# GitHub Actions Debugging

## Connecting to Kubernetes runners for interactive debugging

While we don't have anything as sophisticated as
https://github.com/pytorch/pytorch/wiki/Debugging-using-with-ssh-for-Github-Actions
yet, we do have the basic ability to SSH to some of our self-hosted GitHub
Actions runners while they are online. Once connected to a machine you can debug
by inspecting files, running commands, etc.

> [!NOTE]
> This procedure only works for authorized users (AMD employees with access
> to the cloud projects).

1. Install `az` and `kubectl` following the installation instructions:

   - https://learn.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest
   - https://kubernetes.io/docs/tasks/tools/#kubectl

1. Authenticate with Azure and get aks credentials:

   ```
   az login
   az account set --subscription <subscription_id>
   az aks get-credentials --resource-group <resource_group_name> --name <aks_name>
   ```

   (Ask around if you are unsure of which subscription, resource group, and
   name to use)

1. Optionally edit the workflow file you want to debug to include a pause so you
   won't be kicked off while still debugging:

   ```yml
   - name: Suspend for interactive debugging
     if: ${{ !cancelled() }}
     run: sleep 21600
   ```

1. Trigger the workflow you want to test, if not already running

1. Look for the runner name in the `Set up job` step:

   ```
   Current runner version: '2.324.0'
   Runner name: 'azure-windows-scale-rocm-2jjjw-runner-7htbh'
   Machine name: 'AZURE-WINDOWS-S'
   ```

1. Connect to the runner, choosing the appropriate shell for the operating
   system:

   ```
   kubectl exec -it azure-windows-scale-rocm-2jjjw-runner-7htbh  -n arc-runners -- powershell
   ```

### Tips for debugging on Windows runners

Relevant directories:

| Directory                                    | Description                                                                           |
| -------------------------------------------- | ------------------------------------------------------------------------------------- |
| `C:\home\runner\_work\`                      | Files related to the current job                                                      |
| `C:\home\runner\_work\TheRock\TheRock\`      | Source checkout                                                                       |
| `C:\home\runner\_work\_tool\Python\3.12.10\` | Python installs for [`actions/setup-python`](https://github.com/actions/setup-python) |
| `B:\build\`                                  | CMake build directory                                                                 |

To monitor CPU usage a tool like
[btop4win](https://github.com/aristocratos/btop4win) can be installed and run:

```powershell
$progresspreference="SilentlyContinue"; Invoke-WebRequest https://github.com/aristocratos/btop4win/releases/download/v1.0.4/btop4win-x64.zip -OutFile btop4win-x64.zip; Expand-Archive btop4win-x64.zip -Force; $env:PATH="$env:PATH;$pwd\btop4win-x64\btop4win\"; btop4win.exe
```

To re-run CMake build commands:

> [!IMPORTANT]
> The choices of shell (bash/powershell/cmd) are very important here. If commands
> are run from the wrong shell, CMake configure may fail in confusing ways.

```bash
# Assumed starting under powershell

# Setup MSVC under cmd
cmd
"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"

# Switch to bash, move into the source directory
bash
cd _work/TheRock/TheRock

# Optionally set environment variables
export TEATIME_FORCE_INTERACTIVE=1

# Copy the configure command from the "Configure Projects" step
cmake -B "B:/build" -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx110X-dgpu -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DTHEROCK_VERBOSE=ON -DBUILD_TESTING=ON -DCMAKE_C_COMPILER="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" -DCMAKE_CXX_COMPILER="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" -DCMAKE_LINKER="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/link.exe" -DTHEROCK_BACKGROUND_BUILD_JOBS=4

# Build CMake targets
# You could also run buildctl.py here to enable/disable specific subprojects
cmake --build "B:\build" --target MIOpen+expunge
cmake --build "B:\build" --target MIOpen+dist
```

### Issues with debugging notes

- https://github.com/ROCm/TheRock/issues/840: Builds hitting 6 hour timeouts
- https://github.com/ROCm/TheRock/issues/1407: Flaky compiler crashes during builds

## Working effectively from forks

The structure outlined at
[Overall build architecture](./development_guide.md#overall-build-architecture)
shows how [artifacts](./artifacts.md) produced by source builds can be leveraged
for package builds such as those for [Python packaging](./../packaging/python_packaging.md).

This modular and pipelined build architecture is particularly useful when
developing and debugging packaging workflows, since the complete build/release
pipeline can take several hours sharded across multiple types of build and test
machines.

The [`.github/workflows/build_windows_python_packages.yml`](/.github/workflows/build_windows_python_packages.yml)
and [`.github/workflows/build_portable_linux_python_packages.yml`](.github/workflows/build_portable_linux_python_packages.yml)
workflows are both runnable from personal repository forks. By default they
download artifacts from a recent workflow run in the https://github.com/ROCm/TheRock
repository. You can customize where artifacts are downloaded from by setting
the `artifact_github_repo` and `artifact_run_id` workflow inputs.

Eventually we would like for all ROCm CI and CD workflow runs to produce and
upload artifacts in a compatible schema so that more workflows (e.g. producing
native Linux or Windows packages, running framework tests, etc.) can extend this
foundation.
