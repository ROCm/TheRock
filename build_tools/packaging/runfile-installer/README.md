# ROCm Runfile Installer

## Overview

The ROCm Runfile installer is a method of installing ROCm and/or the AMDGPU driver without using a native Linux package manager.
If all required dependencies are met, the ROCm Runfile installer can be used to install ROCm offline without network or internet access.


## Prerequisites

>[Required]

1. Network/internet connection for installer build.

>[Optional]

1. Network/internet connection for installer runtime if installing 3rd party dependency using the installer.

## Linux Distributions

The ROCm Runfile installer is designed to support the follow list of Linux Distros:

* Ubuntu        : `22.04, 24.04`
* RHEL          : `8.10, 9.4, 9.6, 9.7, 10.0, 10.1`
* Oracle Linux  : `8.10, 9.6, 10.0`
* Rocky Linux   : `9.6`
* SLES          : `15.7`
* Debian        : `12, 13`

## Building

### Prerequisites

The following packages require installation prior to building the ROCm Runfile installer.

#### Ubuntu / Debian

``` shell
    sudo apt install -y binutils xz-utils zstd wget curl sudo rsync cmake
```

#### RHEL / OL

Install the RHEL version-specific prerequisites:

Install the following for RHEL 8.x / OL 8.x:

``` shell
    sudo dnf install wget
    wget https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
    sudo rpm -ivh epel-release-latest-8.noarch.rpm
    sudo crb enable
```

Install the following for RHEL 9.x / OL 9.x / Rocky 9.x:

``` shell
    sudo dnf install wget
    wget https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
    sudo rpm -ivh epel-release-latest-9.noarch.rpm
    sudo crb enable
```

Install the following for RHEL 10.x / OL 10.x:

``` shell
    sudo dnf install wget
    wget https://dl.fedoraproject.org/pub/epel/epel-release-latest-10.noarch.rpm
    sudo rpm -ivh epel-release-latest-10.noarch.rpm
    sudo crb enable
```

Install the generic RHEL x.x prerequisites

``` shell
    sudo wget curl sudo rsync cmake
```

#### SLES

Install the following for SLES 15.7:

``` shell
    sudo SUSEConnect -p PackageHub/15.7/x86_64
    sudo zypper install awk cmake gcc gcc-c++ sudo wget rsync
```

### Clone the source

``` shell
    git clone git@github.com:ROCm/rocm-installer-runfile-internal.git
```

### Building the Installer

The process of building the Runfile installer is a two stage process: Setup and Build.  For the Setup stage, Debian or RPM packages are "pulled" from source ROCm and amdgpu repos.  The Build stage then uses the pulled packages from the Setup stage extracts and creates the self-extracting .run file for the installer as well as the GUI.

#### Setup

The setup and pull of packages uses the **setup-installer.sh** script.  The script will pull packages based on the Linux distribution currently running using a repo configuration file located in the **offline-installer-contained/package-puller/config** directory.  As packages are pulled (downloaded) from the configured repositories, the setup-installer.sh script will then move the packages into the package-extractor directory in preparation of the Build stage.

#### Build

The building of the installer uses the **build-installer.sh** script.  Following setup, the Build stage will extract the contents of all downloaded packages and prepare the installer content directory, build the GUI, and create the self-extracting .run file for the installer.

The following shows the sequence to setup and build the complete Runfile installer:

``` shell
cd offline-installer-contained
./setup-installer.sh
./build-installer.sh
```

Once the build-install.sh completes, the installer .run file will be located in the **offline-installer-contained/build** directory.

## Install - GUI

The ROCm Runfile installer can be used to install ROCm and/or the AMDGPU driver using the provided GUI.  To install using the GUI, run the installer from the terminal command line as follows:

``` shell
bash rocm-installer.run
```

For more information on how to use the GUI refer to the documentation here: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/rocm-runfile-installer.html

## Install - Command line

The ROCm Runfile installer can be used to install ROCm and/or the AMDGPU driver using the command line.
Run the installer from the terminal command line as follows:

``` shell
bash rocm-installer.run <options>
```

The <options> parameter can be set to these options:

* User help/information
  * help: Displays information on how to use the ROCm Runfile Installer.
  * version: Displays the current version of the ROCm Runfile Installer.

* Runfile options
  * noexec: Disable all installer execution. Extract the .run file content only.
  * noexec-cleanup: Disable cleanup after installer execution. Keep all .run extracted and runtime files.
  * untar [directory]: Extract only ROCm installation components from the .run file tarball to [directory].

* Dependencies
  * deps=[arg] [compo]:
    * [arg]:
    * list [compo]: Lists the required dependencies for the install [compo].
    * validate [compo]: Validates which required dependencies are installed or not installed for [compo].
    * install-only [compo]: Installs the required dependencies only for [compo].
    * install [compo]: Installs with the required dependencies for [compo].

      file [file-path]: Installs with the dependencies from a dependency configuration file with path [file_path].
 file-only [file-path]: Install the dependencies from a dependency configuration file with path [file_path] only.
 
    * [compo]: Install component (rocm/amdgpu/rocm amdgpu).

* Install
  * rocm: Enable ROCm components install.
  * amdgpu: Enable AMDGPU driver install.
  * force: Force the ROCm and AMDGPU driver install.
  * target=[directory]: The target directory path for the ROCm components install.

* Post-install
  * postrocm: Run the post-installation ROCm configuration (for instance, script execution and symbolic link creation).
  * amdgpu-start: Start the AMDGPU driver after the install.
  * gpu-access=[access_type]
    * [access_type]:
    * user: Adds the current user to the render,video group for GPU access.
    * all: Grants GPU access to all users on the system using udev rules.
    
* Uninstall
  * uninstall-rocm (target=[directory]): Uninstall ROCm.
    * (target=[directory]): Optional target directory for the ROCm uninstall.
  * uninstall-amdgpu: Uninstall the AMDGPU driver.

* Information/Debug
  * findrocm: Search for a ROCm installation.
  * complist: List the version of ROCm components included in the installer.
  * prompt: Run the installer with user prompts.
  * verbose: Run the installer with verbose logging.

### Runfile options

The ROCm Runfile Installer is a self-extracting .run file that verifies a checksum and extracts files to the current working directory.  The content is extracted to a new directory named rocm-installer.

When extraction completes, execution begins automatically by either starting up the GUI (if no arguments are provided) or executing the rocm-installer.sh script with all command line arguments. The rocm-installer.sh script is in the extracted rocm-installer directory.

When the GUI exits or the command line completes execution, the Runfile installer will automatically clean up the rocm-installer directory and delete all content except for the log files.

In some cases, a user may choose to run multiple commands from the same extraction and save the time required to verify and extract the package contents of the .run file. There are two command line options that let you disable the .run cleanup process: noexec and noexec-cleanup.

* noexec

The noexec option is a single command line argument that lets the checksum and extraction process complete and then exits without starting the GUI or executing the rocm-installer.sh script. All content will be maintained after the exit. You can then use the rocm-installer.sh script directly from the command line without specifying the .run file name.

For example, extract the .run file and then use rocm-installer.sh instead of rocm-installer.run to install ROCm and the AMDGPU driver separately:

``` shell
bash rocm-installer.run noexec
cd rocm-installer
bash rocm-installer.sh rocm
bash rocm-installer.sh amdgpu
```

* noexec-cleanup

The noexec-cleanup option disables the cleanup process after the GUI or command line interface exits. Unlike the noexec option, all command line arguments are processed as normal, but no content is deleted upon exit or completion. At this point, you can switch to using the rocm-installer.sh script within the rocm-installer directory to avoid re-extracting the contents.

* untar [directory]

The ROCm Runfile Installer provides an extraction-only mode of installing ROCm without installation processing.  In this extraction-only mode, the Runfile Installer will untar the contents of tarball archive within the runfile to a target directory.  The output will be the extraction of the rocm-x.y.z directory included in the build of the installer as well as setup scripts for simple rocm post-install.

For example, to extract-only to a directory called **myrocm** in the $USER directory, the command line is as follows:

``` shell
bash rocm-installer.run untar "/home/amd/myrocm"
```

Once the extraction completes, the **untar** command will generate a setup script called **setup-modules**.  The **setup-modules** script will be specific for the ROCm version based on the runfile build and found in the rocm-x.y.z output directory of extraction.  Users may run the **setup-modules-x.y.z.sh** script to perform a basic ROCm setup using environment modules. 

For example, to set up an untar extraction of ROCm 7.1 into output directory  /home/amd/myrocm, run the following command line:

``` shell
cd /home/amd/myrocm/rocm-7.1.0
./setup-modules-7.1.0.sh
source /etc/profile.d/modules.sh
module load rocm/7.1.0
```

Note, that the setup-modules script will install the environment-modules package as part of the setup process to enable environment modules support.  Also, the setup-modules script will not setup any required ROCm dependencies.  Dependencies must be installed manually or using the Runfile installer dependency installation via the **deps=install-only rocm** command line argument or installation using the Runfile GUI.  In addition to dependencies, manualy setup or installation via the Runfile installer may be required for access to GPU resource for ROCm operation (**gpu-access=all/user**).

### Dependency options

ROCm or AMDGPU driver installation by the Runfile installer requires the pre-installation of non-AMD libraries and frameworks for ROCm and the driver to function correctly.  These dependencies are packages that may be installed seperately or via the Runfile installer.  The following provides options for listing, validating and installing the required non-AMD package dependencies.

* deps=list <rocm/amdgpu>

This dependency option lists all the dependencies required for ROCm and/or AMDGPU installation. It lists all required (Debian or RPM) packages that require pre-installation on the system. The additional **rocm** or **amdgpu** parameters is a requirement for the deps=list option that instructs the installer to list only the dependencies required by ROCm, the AMDGPU driver, or both.

Running deps=list causes the installer to quit after listing the dependencies.

Use deps=list <rocm/amdgpu> as a single <options> parameter:

``` shell
bash rocm-installer.run deps=list rocm
bash rocm-installer.run deps=list amdgpu
bash rocm-installer.run deps=list rocm amdgpu
```

** Note: The list of required packages can be installed separately from the Runfile Installer.

* deps=validate <rocm/amdgpu>

This dependency option verifies whether any of the required ROCm or AMDGPU driver packages in the dependency list are already installed on the system running the command. The output is a list of any missing dependency packages that require installation.

Running **deps=validate** causes the installer to quit after listing the missing dependencies.

Use **deps=validate <rocm/amdgpu>** as a single <options> parameter:

``` shell
bash rocm-installer.run deps=validate rocm
bash rocm-installer.run deps=validate amdgpu
bash rocm-installer.run deps=validate rocm amdgpu
```

** Note: The list of missing packages can be installed separately from the Runfile Installer.

* deps=install

This dependency option validates and installs any required packages in the dependency list for ROCm, the AMDGPU driver, or both that are missing on the system running the command. This dependency option is not a single <options> parameter and can be added to a list of other options for the installer. The deps=install option expects at least one of the rocm or amdgpu <options> parameters to also be present in the list to enable the pre-installation of required dependencies before the Runfile Installer installs ROCm, the AMDGPU driver, or both.

For example, to install the dependencies and ROCm, the command line is as follows:

``` shell
bash rocm-installer.run deps=install rocm
```

To install the dependencies and the AMDGPU driver, the command line is as follows:

``` shell
bash rocm-installer.run deps=install amdgpu
```

To install the dependencies and both ROCm and the AMDGPU driver, the command line is as follows:

``` shell
bash rocm-installer.run deps=install rocm amdgpu
```

** Note: The installer can be set to only install the ROCm dependencies, AMDGPU dependencies, or both and then quit. In this case, add -only to the deps=install option:

``` shell
bash rocm-installer.run deps=install-only rocm
bash rocm-installer.run deps=install-only amdgpu
bash rocm-installer.run deps=install-only rocm amdgpu
```

* deps=file <file-path>

This dependency option specifies the name of a input file for the installer. This file contains a custom list of dependency packages to install. The list must have as the same format as the output of the deps=list option. Specify each (Debian) package by name, one package per line. <file-path> is the second parameter, which indicates the absolute path to the dependency file.

For example, to install the dependencies listed in a file named **mydeps.txt** as part of a ROCm install, the command line is as follows:

``` shell
bash rocm-installer.run deps=file /home/amd/mydeps.txt
```

** Note: The installer can be set only to install the dependencies file and then quit. In this case, add -only to deps=file.

``` shell
bash rocm-installer.run deps=file-only /home/amd/mydeps.txt
```

### Install options

Once dependencies are pre-installed (manually or via the installer) the ROCm Runfile Installer can be configured to install ROCm and/or the AMDGPU driver to a specified location.

* rocm

At the command line, add the **rocm** option to enable ROCm installation.

** Note: This option must be in the list of .run <options> for ROCm component installation.

For example, to install ROCm with no other options, the command line is as follows:

``` shell
bash rocm-installer.run rocm
```

* target=<directory>

This install option is used for setting the target directory where ROCm will be installed. The target=<directory> is only used as an option for ROCm installation and is not required for an AMDGPU driver install.

When target=<directory> is not specified in the <options> list, the installer uses a default installation path for ROCm. The default install directory is **$PWD** where you launched the installer. In this configuration, the installer creates a new directory inside $PWD named **rocm** and installs all ROCm components to this location.

The user can change the default location and set the ROCm component installation directory using the target=<directory> option. The <directory> argument must be a valid and absolute path to a directory on the system executing the ROCm Runfile Installer.

For example, to install ROCm to the usual **/opt/rocm** location, the command line is as follows:

``` shell
bash rocm-installer.run target="/" rocm
```

To install ROCm to a directory called **myrocm** in the $USER directory, the command line is as follows:

``` shell
bash rocm-installer.run target="/home/amd/myrocm rocm"
```

* amdgpu

At the command line, add the **amdgpu** option to enable AMDGPU driver installation.

** Note: This option must be in the list of .run <options> for AMDGPU driver installation.

For example, to install the AMDGPU driver with no other options, the command line is as follows:

``` shell
bash rocm-installer.run amdgpu
```

** Note: Both rocm and amdgpu can be combined in the <options> list to install both components. For this case, the command line is as follows:

``` shell
bash rocm-installer.run rocm amdgpu
```

* force

The force install option can be added to the <options> list for the case of multiple pre-existing Runfile installs of ROCm. Add this option to disable any installer prompts that ask for confirmation to continue with the ROCm install if there are currently one or more Runfile ROCm installations already on the system.

### Post-install options

The post-install options can configure the ROCm Runfile Installer to apply additional post-installation options after completing the installation. At the command line, add one or more of the post-installation options to the <options> list for the .run file.

* postrocm

This post-install option applies any post-installation configuration settings for ROCm following installation on the target system. The post-installation configuration includes any symbolic link creation, library configuration, and script execution required to use the ROCm runtime.

To enable the ROCm post-install configuration, add postrocm to the <options> list at the command line.

For example, to enable ROCm post-installation configuration for a ROCm installation to the **/** directory, the command line is as follows:

``` shell
bash rocm-installer.run target="/" rocm postrocm
```

In cases where the **postrocm** is not included in the install command for ROCm, the post-install can still be run separately after install.  To run post installation of ROCm from the command-line, use the **postrocm** argument in conjunction with **target=rocm-install-path** where **rocm-install-path** is the location of the version-specific ROCm installation for the used runfile installer.  For example, if the current runfile installer is for ROCm 7.1.0, then the **rocm-install-path** must be the path to a ROCm 7.1.0 runfile installation.

To use the **postrocm** argument separately from the initial install of ROCm 7.1.0 to **/home/amd/myrocm** the command-line is as follows:

``` shell
bash rocm-installer.run target="/home/amd/myrocm/rocm-7.1.0" postrocm
```

** Note: Adding the postrocm option to the <options> list is highly recommended to guarantee proper functioning of the ROCm components and applications.

* gpu-access=<access_type>

This post-install option sets the GPU resource access permissions. ROCm runtime libraries and applications might need access to the GPU. This requires setting the access permission to the video and render groups using the <access_type>.

If the ROCm installation is for a single user, then set the <access_type> for the gpu-access option to **user**.

For example, to add the current user ($USER) to the video,render group for GPU access for a ROCm installation, the command line is as follows:

``` shell
bash rocm-installer.run rocm gpu-access=user
```

In cases where a system administrator is installing ROCm for multiple users, they might want to enable GPU access permission for all users. For this case, set the <access_type> for the gpu-access option to **all**:

``` shell
bash rocm-installer.run rocm gpu-access=all
```

** Note: Adding the gpu-access option to the <options> list is recommended for using ROCm. A typical ROCm installation includes both the postrocm option and one of the gpu-access types.

### Uninstall options

These options configure the ROCm Runfile Installer to uninstall a previous ROCm or AMDGPU driver installation.

* uninstall-rocm (target=<directory>)

This option configures the ROCm Runfile Installer to uninstall a previous ROCm installation.
The parameter target=<directory> is optional for **uninstall-rocm**. If set, the uninstall looks for a pre-existing ROCm installation at the specified directory path and attempts to remove it.

To uninstall ROCm from the default location **$PWD/rocm**, use the following command line:

``` shell
bash rocm-installer.run uninstall-rocm
```

To uninstall ROCm from a specific location, append **target=<directory>**. For example, if ROCm was previously installed to **/home/amd/myrocm**, use the following command line:

``` shell
bash rocm-installer.run uninstall-rocm target="/home/amd/myrocm"
```

** Note: The **uninstall-rocm** option can only remove ROCm if it was installed using the ROCm Runfile Installer. Traditional package-based ROCm installs will not be removed.

* uninstall-amdgpu

This option configures the ROCm Runfile Installer to uninstall a previous AMDGPU driver installation.

To uninstall the AMDGPU driver from the system, use the following command line:

``` shell
bash rocm-installer.run uninstall-amdgpu
```

** Note: The uninstall-amdgpu option can only remove the AMDGPU driver if it was installed using the ROCm Runfile Installer. Traditional package-based AMDGPU driver installs will not be removed.

### Information and debug options

The ROCm Runfile Installer includes options for information output or debugging.

* findrocm

This option searches the install system for any existing installations of ROCm. Any install locations are output to the terminal.

Use **findrocm** as a standalone <options> parameter:

``` shell
bash rocm-installer.run findrocm
```

* complist

This option lists the version of each ROCm component included in the installer. The **complist** option causes the Runfile Installer to quit after listing the ROCm components.

Use **complist** as a standalone <options> parameter:

``` shell
bash rocm-installer.run complist
```

* prompt

This debug option enables user prompts in the installer. At specific, critical points of the installation process, the installer halts execution and prompts the user to either continue with the install or exit.

* verbose

This debug option enables verbose logging during ROCm Runfile Installer execution.

For example, to install ROCm with user prompts and verbose logging, the command line is as follows:

``` shell
bash rocm-installer.run target="/" rocm prompt verbose
```

## Log Files

### Setup Logs

When setting up for building the installer using the **setup-installer.sh** script, the process of downloading the packages from the configured repos will be logged to the folowing location: **offline-installer-contained/package-puller/logs**

### Build Logs

Once the setup is completed, the **build-installer.sh** script is run to extract package contents, create the installer components, build the GUI and finally package the **rocm-installer** directory into the self-extracting .run file.  The logs for the package extraction process are logged to the following location:  **offline-installer-contained/package-extractor/logs**

### Install Logs

During installation, the ROCm Runfile Installer will output execution logs into the following location: **rocm-installer/logs**

The rocm-installer directory is created when the ROCm Runfile Installer self-extracts to the current working directory where it is being executed.

## Testing

Test scripts for testing basic ROCm functionality following installation are in the following location: **offline-installer-contained/test**

* amd-smi-test.sh

* rocdecode-test.sh

* rocm-examples-test.sh


