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

Once connected, you'll find files related to the current job at
`C:\home\runner\_work`. If using
[`actions/setup-python`](https://github.com/actions/setup-python), you can see
installed Python packages under paths like
`C:\home\runner\_work\_tool\Python\3.12.10\x64\Lib\site-packages`.

To monitor CPU usage a tool like
[btop4win](https://github.com/aristocratos/btop4win) can be installed and run:

```powershell
$progresspreference="SilentlyContinue"; Invoke-WebRequest https://github.com/aristocratos/btop4win/releases/download/v1.0.4/btop4win-x64.zip -OutFile btop4win-x64.zip; Expand-Archive btop4win-x64.zip -Force; $env:PATH="$env:PATH;$pwd\btop4win-x64\btop4win\"; btop4win.exe
```

See also https://github.com/ROCm/TheRock/issues/840.
