# RockBuilder

RockBuilder provides a configuration file based way of building
one or multiple projects external projects on top of the existing
ROCM installation.

Project can be application, library or some other buildable or installable entity.

## List of applications build

At the moment RockBuilder will build following application:
- pytorch
- pytorch vision
- pytorch audio
- torch migraphx

Full list of applications build is defined the project list configuration file
projects/core_apps.pcfg.

# Usage

Below are described the steps required for setting up the RockBuilder environment
and how to use it either to build all projects or to use it for just to execute some smaller task.

## Environment setup

Rockbuilder requires the existing ROCM environment and Python installation.

### ROCM Environment

If ROCM_HOME environment variable is defined, then the ROCM environment is
used from that directory.

If ROCM_HOME is not defined, RockBuilder will try to find it from the directory

```
  therock/build/dist/rocm
```

### Python Environment

Rockbuilder expects by default that Python venv is activated as it is the
recommended way to use and install python applications that are required by the
RockBuilder. Applications that are build by the RockBuilder will also be installed
to the python environment that is used.

Recommended python version should be same than what is used to build TheRock and
can be for example python 3.11, 3.12 or 3.13.

You can either create a new python venv or use the one already done and used for TheRock build.

```bash
cd TheRock
source .venv/bin/activate
```

If you want to use instead real python environment instead of venv,
you must force that by defining ROCK_PYTHON_PATH environment variable.
For example on Linux:


```bash
export ROCK_PYTHON_PATH=/usr/bin
```

## Build and install all RockBuilder projects

### Building and using ROCM from TheRock build environment

Build firth the Rock base system by following the instructions in
the TheRock/README.md and then build the RockBuilder projects from the external folder.

Example for building on Linux.

```bash
cd TheRock
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ./build_tools/fetch_sources.py
cmake -B build -GNinja . -DTHEROCK_AMDGPU_TARGETS=gfx1201
cmake --build build
cd external-projects
python rockbuilder.py
```

### Using existing ROCM Installation

Build firth the Rock base system by following the instructions in
the TheRock/README.md and then build the RockBuilder projects from the
external folder. For example:

```bash
SET ROCM_HOME=/opt/rocm
source .venv/bin/activate
cd TheRock/external-projects
python rockbuilder.py
```
    
## Checkout all projects (without build and install)

```bash
python rockbuilder.py --checkout
```
    
## Checkout only the pytorch_audio sources

```bash
python rockbuilder.py --checkout --project pytorch_audio
```
    
## Build only pytorch audio

```bash
python rockbuilder.py --build --project pytorch_audio
```
    
## Install only pytorch audio

```bash
python rockbuilder.py --install --project pytorch_audio
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

## Project specific configuration file

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

There can be separate action commands for posix based systems(Linux) and dos-based systems(Windows).

If specific action is specified (for example --build), RockBuilder does not yet
check whether other actions needed by the build action are already done. Instead it expected that
the user knows that before that the checkout and configure steps are needed. This may change in the future.
