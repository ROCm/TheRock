# Test Environment Reproduction

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
$ GITHUB_REPOSITORY={GITHUB_REPO} python build_tools/install_rocm_from_artifacts.py --run-id {CI_RUN_ID} --amdgpu-family {GPU_FAMILY} --tests
$ export THEROCK_BIN_DIR=./therock-build/bin
# The python test scripts are in directory "build_tools/github_actions/test_executable_scripts/"
# Below is an example on how to run "test_rocblas.py"
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements-test.txt
$ python build_tools/github_actions/test_executable_scripts/test_rocblas.py
```

`install_rocm_from_artifacts.py` parameters

- CI_RUN_ID is sourced from the CI run (ex: https://github.com/ROCm/TheRock/actions/runs/16948046392 -> CI_RUN_ID = 16948046392)
- GPU_FAMILY is the LLVM target name (ex: gfx94X-dcgpu, gfx1151, gfx110X-dgpu)
- GITHUB_REPO is the GitHub repository that this CI run was executed. (ex: ROCm/rocm-libraries, ROCm/rccl)

To view which python test wrappers we have, please checkout [`test_executable_scripts/`](https://github.com/ROCm/TheRock/tree/main/build_tools/github_actions/test_executable_scripts)
