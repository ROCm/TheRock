## Linux CI runner setup

This directory contains documentation and scripts about setting up a Linux CI Runner for [`ROCm`](https://github.com/ROCm) organization and used by [`TheRock`](https://github.com/ROCm/TheRock) repository.

Note: you must have sufficient permissions to access [ROCm runner page](https://github.com/organizations/ROCm/settings/actions/runners)

### Setup

For brand new machines that do not that ROCm or Docker installed, please follow these steps. Otherwise, please skip to step 3.

1. Install ROCm to the machine using `sudo ./rocm_install.sh`. This script will install ROCm 6.4 and AMD drivers for Ubuntu24, then it will reboot the system.

   - Rebooting the system is required to load ROCm.
   - If you have a different Linux distribution, follow [ROCm installation quick start guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
   - <b>After reboot, please try `rocminfo` and `rocm-smi` to make sure ROCm is loaded and drivers are installed.</b> If there are issues, please try each command in `rocm_install.sh` instead.

1. If docker is not installed, please run `sudo ./docker_install.sh`. This script will download docker for Ubuntu.

1. After ROCm and Docker are installed, please run `sudo ./runner_setup_1.sh {IDENTIFIER}`. There may be multiple GPUs per system, so please add an identifier to make this runner unique and easily understood. Examples: gfx1201 GPU -> `sudo ./runner_setup_1.sh gfx1201-gpu-1`

1. After the runner packages are there, please follow these steps and run the commands:

   - Please retrieve token from [ROCm GitHub runner page](https://github.com/organizations/ROCm/settings/actions/runners/new?arch=x64&os=linux) in the `Configure` tab.
   - Please add an unique identifying label for this CI runner. Example: Linux gfx1201 -> label `linux-gfx1201-gpu-rocm`. This is the label that will be used in workflows and will be shared amongst other identical machines.

   ```
   cd {IDENTIFIER_FROM_STEP_3}
   ./config.sh --url https://github.com/ROCm --token {TOKEN} --no-default-labels --labels {LABEL}
   ```

   - During the config.sh setup step:
     - `Default` is fine for runner group
     - For "name of runner," please include an unique identifier for this runner. Example: for runner gfx1201, `linux-gfx1201-gpu-rocm-1`. A good practice is to have `{LABEL}-{ID}`. Remember, label != name of runner, there may be many gfx1201 machines sharing the label `linux-gfx1201-gpu-rocm`.
     - `_work` is fine for work folder.

1. After ./config.sh script has been completed, please follow these steps and run the commands:

   - For your CI runner to run on a specific GPU, you will need to obtain the correct `{ROCR_VISIBLE_DEVICE}`.
   - To get this, please run `rocminfo` and figure out which `Node` your GPU is running on. Example:

   ```
   *******
   Agent 10
   *******
   Name:                    gfx1201
   Marketing Name:          AMD Instinct machine
   Vendor Name:             AMD
   Node:                    9
   ```

   - After getting the `Node`, please run `rocm-smi` and determine which `Device` corresponds with your `Node`. From this example, `ROCR_VISIBLE_DEVICES` is 5:

   ```
   ============================================ ROCm System Management Interface ============================================
   ====================================================== Concise Info ======================================================
   Device  Node  IDs              Temp        Power     Partitions          SCLK     MCLK    Fan  Perf  PwrCap  VRAM%  GPU%
               (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)
   ==========================================================================================================================
   5       9     0x0000,   00000 00.0°C      000.0W    0000, 000, 0        000Mhz   000Mhz  0%   0000  000.0W  0%     0%
   ==========================================================================================================================
   ================================================== End of ROCm SMI Log ===================================================
   ```

   - Then run these commands with your correct `ROCR_VISIBLE_DEVICES`

   ```
   cd {IDENTIFIER_FROM_STEP_3}
   sudo ./runner_setup_2.sh {ROCR_VISIBLE_DEVICE}
   ```

You are <b>done!</b>. You can use your CI runner using `runs-on: {LABEL}` in GitHub workflows and you'll be able to see your runner in your organization runners page as "Idle"

Appendix:

- [Requirements for self hosted runners](https://github.com/shivammathur/setup-php/wiki/Requirements-for-self-hosted-runners)
- [Configuring the self-hosted runner application as a service](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/configuring-the-self-hosted-runner-application-as-a-service)
- [ROCm quick start installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
- [Docker install Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
