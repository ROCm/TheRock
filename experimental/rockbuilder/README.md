# RockBuilder

RockBuilder provides a configuration file based way of building one or multiple external projects on top of the existing ROCM core installation.

Project can be application, library or some other buildable or installable entity.

RockBuilder supports both the Linux and Windows but some of the applications build may be operating system-specific.

## Applications Build

RockBuilder will build the list of applications that are listed in the in the project list configuration file:

```
projects/core_apps.pcfg
```

For each application build there is a application specific configuration file which will then specify the application name, version, source code checkout address and required configure, build and install commands.

At the moment RockBuilder will build by default a following list of applications in Linux and Windows:

- pytorch (projects/pytorch.cfg)
- pytorch vision (projects/pytorch_vision.cfg)
- pytorch audio (projects/pytorch_audio.cfg)
- torch migraphx (projects/torch_migraphx.cfg)

Configuration file format is specified in [CONFIG.md](CONFIG.md) document.

# Usage

Below are described the steps required for setting up the RockBuilder environment
and how to use it either to build all projects or to use it for just to execute some smaller task.

## Build everything by using TheRock ROCm build

First build the TheRock base system by following the instructions in
[README.md](../../README.md#building-from-source).
Then build the RockBuilder projects from the
experimental/rockbuilder directory. Example:

Example commands to build and test on Linux:

```bash
cd TheRock
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ./build_tools/fetch_sources.py
cmake -B build -GNinja . -DTHEROCK_AMDGPU_TARGETS=gfx1201
cmake --build build
cd experimental/rockbuilder
python rockbuilder.py
cd examples
export ROCM_HOME=TheRock/build/dist/rocm
export LD_LIBRARY_PATH=${ROCM_HOME}/lib:${ROCM_HOME}/lib/llvm/lib
python torch_gpu_hello_world.py
python torch_vision_hello_world.py
python torch_audio_hello_world.py
```

Example commands to build and test on Windows by using the 'x64 Native MSVC command prompt':

```bash
cd c:\TheRock
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python ./build_tools/fetch_sources.py
cmake -B build -GNinja . -DTHEROCK_AMDGPU_TARGETS=gfx1201
cmake --build build
cd experimental\rockbuilder
set PYTORCH_ROCM_ARCH=gfx1201
python rockbuilder.py
cd examples
set PATH=c:\TheRock\build\dist\rocm\bin;c:\TheRock\build\dist\rocm\lib;%PATH%
python torch_gpu_hello_world.py
python torch_vision_hello_world.py
python torch_audio_hello_world.py

```

Example output from test apps in Windows when using the AMD Radeon W7900 GPU:

```bash
(.venv) D:\rock\TheRock\experimental\rockbuilder\examples>python torch_gpu_hello_world.py
Pytorch version: 2.7.0
ROCM HIP version: 6.5.25222-1f8e4aaca
cuda device count: 1
default cuda device name: AMD Radeon PRO W7900 Dual Slot
device type: cuda
Tensor training running on cuda: True
Running simple model training test
    tensor([0., 1., 2.], device='cuda:0')
Hello World, test executed succesfully

(.venv) D:\rock\TheRock\experimental\rockbuilder\examples>python torch_vision_hello_world.py
pytorch version: 2.7.0
pytorch vision version: 0.22.0

(.venv) D:\rock\TheRock\experimental\rockbuilder\examples>python torch_audio_hello_world.py
pytorch version: 2.7.0
pytorch audio version: 2.7.0

```

## Build everything by using TheRock ROCm install

1. First build the TheRock base system by following the instructions in
   [README.md](../../README.md#building-from-source).
   Then build the RockBuilder projects from the
   experimental/rockbuilder directory. Example:

```bash
export ROCM_HOME=/opt/rocm
source ${ROCM_HOME}/.venv/bin/activate
cd TheRock/experimental/rockbuilder
python rockbuilder.py
cd examples
python torch_gpu_hello_world.py
python torch_vision_hello_world.py
python torch_audio_hello_world.py
```

Wheels that have been build can be found from the packages/wheels directory.

## Checkout all pytorch_28_amd projects (without build and install)

```bash
python rockbuilder.py --checkout --project_list projects/pytorch_28_amd.pcfg
```

Source code would be checked out to directory `src_projects`

## Checkout all projects to custom directory

```bash
python rockbuilder.py --checkout --src-base-dir src_prj
```

Source code for each project would be checked out under the directory `src_prj`

## Checkout and build only pytorch_audio

In this example we build and install only the pytorch audio and
copy the produced pytorch audio wheel to directory "test" instead of using default "packages/wheels"
Note that pytorch audio requires that pytorch has been built and installed first.

```bash
python rockbuilder.py --project pytorch_audio --output-dir test
```

or

```bash
python rockbuilder.py --project projects/pytorch_audio.cfg --output-dir test
```

By default this checks out pytorch audio source to directory `src_projects/pytorch_audio`:

```bash
$ ls src_projects/
pytorch_audio/
```

## Checkout pytorch_audio sources to custom directory

```bash
python rockbuilder.py --checkout --project pytorch_audio --src-dir src_prj/py_audio
```

Source code would be checked out to directory `src_prj/py_audio`:

```bash
$ ls src_prj/
py_audio/
```

## Checkout custom pytorch_audio version

This would checkout the v2.6.0 version instead of the version specified in the pytorch_audio.cfg file

```bash
python rockbuilder.py --checkout --project pytorch_audio --pytorch_audio-version=v2.6.0
```

## Build only pytorch audio

Note that pytorch audio requires that pytorch has been built and installed first.

```bash
python rockbuilder.py --build --project pytorch_audio
```

## Install only pytorch audio

Note that pytorch audio requires that pytorch has been built and installed first.

```bash
python rockbuilder.py --install --project pytorch_audio
```

# Environment setup

Rockbuilder requires the existing ROCM environment and Python installation.

## ROCM Environment

If ROCM_HOME environment variable is defined, then the ROCM environment is
used from that directory.

If ROCM_HOME is not defined, RockBuilder will try to find it from the directory

```
  therock/build/dist/rocm
```

## Python Environment

Rockbuilder expects by default that Python venv is activated as it is the
recommended way to use and install python applications that are required by the
RockBuilder. Applications that are build by the RockBuilder will also be installed
to the python environment that is used as they may be required by other applications
that are later build by the rockbuilder.

Recommended python version should be same than what is used to build TheRock and
can be for example python 3.11, 3.12 or 3.13.

You can either create a new python venv or use the one already done and used for TheRock build.

```bash
cd TheRock
source .venv/bin/activate
```

By default the RockBuilder will refuse to run if you are trying to use a real
python virtual environment instead of using a virtual env. You can change that behaviour
by setting the ROCK_PYTHON_PATH environment variable that will point to python directory.
In Linux this can be set for example in a following way:

```bash
export ROCK_PYTHON_PATH=/usr/bin
```

# Adding new projects to RockBuilder

There exist two types of configuration files that are stored in the applications directory.

## Project list configuration files

projects/core_apps.pcfg is an example of project list configuration file.

Project list configuration files are used to define a list of projects
that are build by the RockBuilder. For now the RockBuilder is hardcoded
to always use the projects/core_apps.pcfg to check the appliation list but
in the future the functionality can be extended to allow having multiple
different project lists.

core_apps.pcfg example:

```bash
[projects]
project_list=
    pytorch
    pytorch_vision
    pytorch_audio
    torch_migraphx
```

## Project specific configuration files

projects/pytorch.cfg is an example from the project configuration file.

Project configuration file specifies actions that RockBuilder executes for the project:

- checkout
- clean
- configure
- build
- install

By default the RockBuilder executes checkout, configure, build and install actions for the project but user
can override this for example by specifying single command. For example:

```bash
python rockbuilder.py --checkout
```

There can be separate action commands for posix based systems(Linux) and windows-based systems.

If optional action parameter is specified (for example --build), RockBuilder does not yet
check whether other actions would be needed to be executed before that.
This is expected to be changed in the future.
