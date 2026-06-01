# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import re
import socket
import subprocess
import tempfile


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout.decode("ascii")


def search(pattern, string):
    m = re.search(pattern, string, re.MULTILINE)
    if m is not None:
        return m.group(1)
    return None


def _subprocess_helper(cmd: list) -> tuple:
    """
    This is a helper method which runs a command line argument
    and returns the result in text format.

    Args
    ----
    cmd : A list of commands with relevant options

    Returns
    -------
    success : A boolean to indicate whether the command was executed succcesfully
    info : The result of the command in text format

    """

    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    success = False
    info = ""

    try:
        process = subprocess.Popen(cmd, stdout=fout, stderr=ferr)
        process.wait()
        fout.seek(0)
        info = fout.read()
        success = True
    except subprocess.CalledProcessError as process_error:
        info = "None"
    except FileNotFoundError as not_found_err:
        info = "None"
    finally:
        fout.close()
        ferr.close()

    return success, info


def get_cuda_info() -> str:
    """
    This function gets the cuda version if available. Else, returns "None".

    Args
    ----
    None

    Returns
    -------
    info : A string containing the CUDA version information

    """

    path = Path("/usr/local/cuda/version.txt")
    if path.exists():
        info = path.read_text()

    else:
        success, info = _subprocess_helper(["nvidia-smi"])

        if success:
            info = re.split(r"\{2,}", info.split("\n")[2])[-2]
            return info

        else:
            success, info = _subprocess_helper(["nvcc", "--version"])

            if success:
                info = "\n".join(info.split("\n")[-3:-1])
                return info

    return info


def get_gpu_info(devicenum: int = 0) -> str:
    """
    This function gets the NVIDIA GPU infomration if available. Else, returns "None".

    Args
    ----
    devicenum : A integer denoting the device number

    Returns
    -------
    info : A string containing the NVIDIA GPU information

    """

    success, info = _subprocess_helper(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
            "-i",
            str(devicenum),
        ]
    )

    return info


def get_device_info() -> str:

    """

    This function returns architecture_name and internal_product_name from rocminfo.
    If not found returns blank output.

    """

    success, info = _subprocess_helper("rocminfo")
    architecture_name = ""
    internal_product_name = ""

    if info != "None":

        output = info.split("\n")
        start_num = 0
        for i, row in enumerate(output):
            if "Agent 2" in row:
                start_num = i

        if start_num != 0:

            for i in range(start_num, len(output)):
                if "Name" in output[i]:
                    s = search(r"Name:\s*(.*?)$", output[i])
                    if s.startswith("gfx"):
                        architecture_name = s
                        break

            for i in range(start_num, len(output)):
                if "Marketing Name" in output[i]:
                    internal_product_name = search(
                        r"Marketing Name:\s*(.*?)$", output[i]
                    )
                    break

    return architecture_name, internal_product_name



def get_sbios_info() -> str:

    """

    This function returns architecture_name and internal_product_name from rocminfo.
    If not found returns blank output.

    """

    sbios_info = 'None'
    try:
        sbios_info = Path('/sys/class/dmi/id/bios_vendor').read_text().strip() + Path(
        '/sys/class/dmi/id/bios_version').read_text().strip()
    except Exception:
        sbios_info = 'None'
    return sbios_info


def get_machine_specs(filename: str, devicenum: int = 0):
    """
    This method is used to get the machine specifications and
    store it in the specified location as a text file.

    Args
    ----------
    filename : Absolute filepath where the file is to be created
    devicenum : Device number { optional }

    Returns
    -------
    filename : Absolute filepath of the created specs file

    Raises
    ------
    IOError (Is a directory error) : If the specified filename ( input ) should be a file
                                     and not a directory

    Usage
    -----
    CLI syntax
    ----------
    linux : python3 -m pts_amd create_specs_info < filename >
    windows : python -m pts_amd create_specs_info < filename >

    """

    try:
        # Getting the specs information of current device
        # If the specified information is not found , then the value is stored as None
        cpu_path = Path("/proc/cpuinfo")
        meminfo_path = Path("/proc/meminfo")
        version_path = Path("/proc/version")
        osrelease_path = Path("/etc/os-release")
        rocm_path = Path("/opt/rocm/.info/version-utils")
        if cpu_path.exists():
            cpuinfo = cpu_path.read_text()
            cpu_info = search(r"^model name\s*: (.*?)$", cpuinfo)
        else:
            cpuinfo = None
        if meminfo_path.exists():
            meminfo = meminfo_path.read_text()
            ram = search(r"MemTotal:\s*(\S*)", meminfo)
        else:
            meminfo = None
        if version_path.exists():
            version = version_path.read_text()
            kernel_version = search(r"version (\S*)", version)
        else:
            version = None
        if osrelease_path.exists():
            os_release = osrelease_path.read_text()
            distro = search(r'PRETTY_NAME="(.*?)"', os_release)
        else:
            os_release = None
        if rocm_path.exists():
            rocm_info = rocm_path.read_text()
        else:
            rocm_info = None
        try:
            rocm_smi = run(
                [
                    "rocm-smi",
                    "--showvbios",
                    "--showid",
                    "--showproductname",
                    "--showperflevel",
                    "--showclocks",
                    "--showmeminfo",
                    "vram",
                    "lshw",
                ]
            )
        except FileNotFoundError as e:
            rocm_smi = None

        try:
            rocminfo = run(["rocminfo"])
        except FileNotFoundError as e:
            rocminfo = None

        device = rf"^GPU\[{devicenum}\]\s*: "
        hostname = socket.gethostname()

        if rocm_info is None:
            rocm_version = None
        else:
            rocm_version = rocm_info.strip()

        if rocm_smi is None:
            vbios_version = (
                gpuid
            ) = deviceinfo = vram = performance_level = memory_clk = system_clk = None
        else:
            if device is None:
                device = ""
            vbios_version = search(device + r"VBIOS version: (.*?)$", rocm_smi)
            gpuid = search(device + r"Device ID: (.*?)$", rocm_smi)
            if not gpuid:
                gpuid = search(device + r"GPU ID: (.*?)$", rocm_smi)
            vram = search(device + r".... Total Memory .B.: (\d+)$", rocm_smi)
            performance_level = search(device + r"Performance Level: (.*?)$", rocm_smi)
            memory_clk = search(device + r"mclk.*\((.*?)\)$", rocm_smi)
            system_clk = search(device + r"sclk.*\((.*?)\)$", rocm_smi)

        architecture_name, internal_product_name = get_device_info()

        if architecture_name and gpuid:
            deviceinfo = f"{architecture_name.strip()}_{gpuid.strip()}"
        else:
            deviceinfo = None

        if ram is not None:
            ram = "{:.2f} GiB".format(float(ram) / 1024**2)
        if vram is not None:
            vram = "{:.2f} GiB".format(float(vram) / 1024**3)

        gpu_info = get_gpu_info(devicenum)
        cuda_info = get_cuda_info()

        specs = [
            "hostinfo :\n",
            f"hostname : {hostname}",
            f"cpu_info : {cpu_info}",
            f"sbios_info: {get_sbios_info()}",
            f"ram : {ram}",
            f"distro : {distro}",
            f"kernel_version : {kernel_version}",
            f"rocm_version : {rocm_version}",
            "\ndevice info:\n",
            f"device : {deviceinfo}",
            f"vbios_version : {vbios_version}",
            f"performance_level : {performance_level}",
            f"memory_clk : {memory_clk}",
            f"system_clk : {system_clk}",
            f"vram : {vram}",
            f"NVIDIA GPU info: {gpu_info}",
            f"CUDA info: {cuda_info}",
        ]

        if filename == "":
            specs_dict = {
                "hostname": hostname,
                "cpu_info": cpu_info,
                "sbios_info": get_sbios_info(),
                "ram": ram,
                "distro": distro,
                "kernel_version": kernel_version,
                "rocm_version": rocm_version,
                "device": deviceinfo,
                "vbios_version": vbios_version,
                "performance_level": performance_level,
                "memory_clk": memory_clk,
                "system_clk": system_clk,
                "vram": vram,
                "NVIDIA GPU info": gpu_info,
                "CUDA info": cuda_info,
            }
            return specs_dict

        filename = writing_to_file(filename, specs)

    except Exception as error:
        print(error)
        return filename

    return filename


def writing_to_file(filename: str, specs: list) -> str:

    """
    This method is used to store the specs info in the specified location as a text file.

    Args
    ----------
    filename : Absolute filepath where the file is to be created

    Returns
    -------
    filename : Absolute filepath of the created specs file

    Raises
    ------
    IOError (Is a directory error) : If the specified filename ( input ) should be a file
                                     and not a directory


    """

    try:

        # After getting all the values , then those are written in a file.
        with open(filename, "w") as file:
            for info in specs:
                file.write(info + "\n")

    except Exception as error:
        print(error)
        return filename

    return filename
