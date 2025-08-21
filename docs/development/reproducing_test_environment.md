# Reproducing tests environment

## Linux

For reproducing the test environment for a particular CI run, follow the instructions below:

```bash
# This docker container ensures that ROCm is sourced from TheRock
$ docker run -i \
    --ipc host \
    --group-add video \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add 992 \
    -t ghcr.io/rocm/no_rocm_image_ubuntu24_04@sha256:405945a40deaff9db90b9839c0f41d4cba4a383c1a7459b28627047bf6302a26 /bin/bash
$ git clone https://github.com/ROCm/TheRock.git
$ cd TheRock
# The CI_RUN_ID is sourced from the CI run (ex: https://github.com/ROCm/TheRock/actions/runs/16948046392 -> CI_RUN_ID = 16948046392)
# The GPU_FAMILY is the LLVM target name (ex: gfx94X-dcgpu, gfx1151, gfx110X-dgpu)
$ python build_tools/install_rocm_from_artifacts.py --run-id {CI_RUN_ID} --amdgpu-family {GPU_FAMILY} --tests
# In the bin, you can find the test executables
$ cd therock-build/bin
```
