"""Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights reserved.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""

def _subprocess_helper(cmd, *args, **kwargs):
    import subprocess
    import tempfile
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    cout = ""
    success = False
    try:
        p = subprocess.Popen(cmd, stdout=fout, stderr=ferr, *args, **kwargs)
        p.wait()
        fout.seek(0)
        cout = fout.read()
        success = True
    except subprocess.CalledProcessError:
        pass
    except FileNotFoundError:
        pass
    return success, cout

def get_smi_exec(cuda):
    if cuda:
        return "nvidia-smi"
    else:
        return "/opt/rocm/bin/rocm-smi"

def getgfx(devicenum, cuda):
    if cuda:
        return "N/A"
    else:
        cmd = ["/opt/rocm/bin/rocm_agent_enumerator"]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        # Add 1 to devicenum since rocm-agent-enum always prints gfx900 first
        return cout.splitlines()[devicenum+1]

# Get the hostname
def gethostname():
    import socket
    hostname = socket.gethostname()
    return hostname

# Get the host cpu information
def getcpu():
    cmd = ["lscpu"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    cpulist = ""
    searchstr = "Model name:"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            cpulist += line[len(searchstr):].strip()
    return cpulist

# Get the kernel version
def getkernel():
    import subprocess
    cmd = ["uname", "-r"]
    import tempfile
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    return cout.strip()

# Get the host ram size
def getram():
    import re
    cmd = ["lshw", "-class",  "memory"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "size:"
    for line in cout.split("\n"):
        m = re.search(searchstr, line)
        if not m == None:
            return line.strip()[len(searchstr):].strip()

# Get the Linux distro information
def getdistro():
    cmd = ["lsb_release", "-a"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "Description:"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            return line[len(searchstr):].strip()

# Get the version number for rocm
def getrocmversion():
    cmd = ["apt", "show", "rocm-libs"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "Version:"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            return line[len(searchstr):].strip()


# Get the vbios version for the specified device
def getvbios(devicenum, cuda):
    cmd = [get_smi_exec(cuda), "-v", "-d", str(devicenum)]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=vbios_version", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    if cuda:
        return cout

    searchstr = "GPU["+str(devicenum)+"]"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            tmp = line[len(searchstr):].strip()[1:]
            pos = tmp.find(":")
            return tmp[pos+1:].strip()
    return ""

def getgpuid(devicenum, cuda):
    import re
    name = ""
    cmd = [get_smi_exec(cuda), "-i", "-d", str(devicenum)]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=name", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    if cuda:
        return cout

    searchstr = "GPU["+str(devicenum)+"]"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            line = line[len(searchstr):].strip()
            line = re.sub(":", "", line)
            line = re.sub("GPU ID", "", line)
            name += " " + line.strip()
            name = name.replace(" ", "")
    return name

# Get the name of the device from lshw which has index devicenum
def getdeviceinfo(devicenum, cuda):
    import re
    cmd = ["lshw", "-C", "video"]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    searchstr = "-display"
    indices = []
    name = ""
    for idx, line in enumerate(cout.split("\n")):
        if re.search(searchstr, line) != None:
            indices.append(idx)
    for idx, line in enumerate(cout.split("\n")):
        if idx >= indices[devicenum]:
            searchstr = "product:"
            if re.search(searchstr, line) != None:
                pos = line.find(":")
                name += line[pos+1:].strip()
    name += " " + getgpuid(devicenum, cuda)
    return name

# Get the vram for the specified device
def getvram(devicenum, cuda):
    import re
    cmd = [get_smi_exec(cuda), "--showmeminfo", "vram", "-d", str(devicenum)]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=memory.total", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)

    if not success:
        return "N/A"
    if cuda:
        return cout

    searchstr = "GPU["+str(devicenum)+"]"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            line = line[len(searchstr):].strip()
            prestring = "vram :: total:"
            line = re.sub(":", "", line)
            line = re.sub("vram", "", line)
            line = re.sub("total", "", line)
            pos = line.find("used")
            print(line[:pos].strip())
            return line[:pos].strip()

# Get the performance level for the specified device
def getperflevel(devicenum, cuda):
    import re
    cmd = [get_smi_exec(cuda), "-p", "-d", str(devicenum)]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=pstate", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    if cuda:
        return cout

    searchstr = "GPU["+str(devicenum)+"]"
    for line in cout.split("\n"):
        if line.startswith(searchstr):
            line = line[len(searchstr):].strip()
            skipstr = "Current Performance Level "
            line = re.sub(":", "", line)[len(skipstr):].strip()
            return line

# Get the memory clock for the specified device
def getmclk(devicenum, cuda):
    import re
    cmd = [get_smi_exec(cuda), "--showclocks", "-d", str(devicenum)]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=clocks.mem", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    if cuda:
        return cout

    searchstr = "mclk"
    for line in cout.split("\n"):
        m = re.search(searchstr, line)
        if m != None:
            p0 = line.find("(")
            p1 = line.find(")")
            return line[p0+1:p1]

# Get the system clock for the specified device
def getsclk(devicenum, cuda):
    import re
    cmd = [get_smi_exec(cuda), "--showclocks", "-d", str(devicenum)]
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=clocks.sm", "--format=csv,noheader", "-i", str(devicenum)]
    success, cout = _subprocess_helper(cmd)
    if not success:
        return "N/A"
    if cuda:
        return cout

    searchstr = "sclk"
    for line in cout.split("\n"):
        m = re.search(searchstr, line)
        if m != None:
            p0 = line.find("(")
            p1 = line.find(")")
            return line[p0+1:p1]

def getbandwidth(devicenum, cuda):
    gpuid = getgpuid(devicenum, cuda)
    if gpuid == "0x66af":
        # radeon7: float: 13.8 TFLOPs, double: 3.46 TFLOPs, 1024 GB/s
        return (13.8, 3.46, 1024)

def listdevices(cuda, smi=None):
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=count", "--format=csv,noheader", "-i", '0']
        success, cout = _subprocess_helper(cmd)
        if not success:
            return []
        return list(range(0, int(cout)))
        # something
    elif smi is not None:
        return smi.listDevices()

def getbus(devicenum, cuda, smi=None):
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", str(devicenum)]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        return cout
    else:
        if smi is not None:
            return smi.getBus(devicenum)

def getprofile(devicenum, cuda):
    import re
    if cuda:
        return "N/A"
    else:
        cmd = [get_smi_exec(cuda), "-l", "-d", str(devicenum)]
        success, cout = _subprocess_helper(cmd)

        if not success:
            return "N/A"

        searchstr = "GPU["+str(devicenum)+"]"
        for line in cout.split("\n"):
            if line.startswith(searchstr) and "*" in line:
                line = line[len(searchstr):].strip()
                line = re.sub(":", "", line).strip()
                return line[0]

        return "N/A"

def getfanspeedpercent(devicenum, cuda, smi=None):
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=fan.speed", "--format=csv,noheader", "-i", str(devicenum)]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        return str(cout)
    elif smi is not None:
        return str(smi.getFanSpeed(devicenum)[1])

def validclocknames(cuda, smi=None):
    if cuda:
        return ["graphics", "sm", "memory", "video"]
    elif smi is not None:
        return smi.validClockNames

def getcurrentclockfreq(devicenum, clock, cuda, smi=None):
    import re
    if cuda:
        cmd = [get_smi_exec(cuda), "--query-gpu=clocks.current." + clock, "--format=csv,noheader", "-i", str(devicenum)]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"
        return cout
    else:
        cmd = [get_smi_exec(cuda), "--showclocks", "-d", str(devicenum)]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"

        for line in cout.split("\n"):
            m = re.search(clock, line)
            if m != None:
                p0 = line.find("(")
                p1 = line.find(")")
                return line[p0+1:p1]

    return "N/A"

def getcurrentclocklevel(devicenum, clock, cuda):
    import re
    if cuda:
        return "N/A"
    else:
        cmd = [get_smi_exec(cuda), "--showclocks", "-d", str(devicenum)]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"

        searchstr = clock + " clock level: "
        for line in cout.split("\n"):
            m = re.search(searchstr, line)
            if m != None:
                p0 = line.find(searchstr)
                line = line[p0 + len(searchstr):]
                p1 = line.find(":")
                line = line[:p1]
                return line

    return "N/A"

def getmaxlevel(devicenum, clock, cuda):
    import re
    if cuda:
        return "N/A"
    else:
        cmd = [get_smi_exec(cuda), "-s", "-d", str(devicenum)]
        success, cout = _subprocess_helper(cmd)
        if not success:
            return "N/A"

        maxlevel = -1
        searchstr = "Supported " + clock + " frequencies on GPU" + str(devicenum)
        idstr = "GPU["+str(devicenum)+"]"
        p0 = cout.find(searchstr)
        if p0 != -1:
            cout = cout[p0:]
            for line in cout.split("\n"):
                line=line[len(idstr):].strip()
                line=re.sub(":","",line).strip()
                if line:
                    maxlevel = line[0]
                else:
                    break
            return maxlevel
    return "N/A"


def validmemtypes(cuda, smi=None):
    if cuda:
        return ["vram"]
    elif smi is not None:
        # Hardcoded in /opt/rocm/rocm_smi/bindings/rsmiBindings.py
        return ["VRAM", "VIS_VRAM", "GTT"]

def getmeminfo(devicenum, mem_type, cuda, smi=None):
    if cuda:
        if mem_type == 'vram':
            cmd = [get_smi_exec(cuda), "--query-gpu=memory.total", "--format=csv,noheader", "-i", str(devicenum)]
            success, cout_total = _subprocess_helper(cmd)
            if not success:
                return "N/A"
            cmd = [get_smi_exec(cuda), "--query-gpu=memory.used", "--format=csv,noheader", "-i", str(devicenum)]
            success, cout_used = _subprocess_helper(cmd)
            if not success:
                return "N/A"
            return cout_used, cout_total
        else:
            return "N/A"
    elif smi is not None:
        return smi.getMemInfo(devicenum, mem_type)

def validversioncomponents(cuda, smi=None):
    # currently only driver according to /opt/rocm/bin/rocm_smi.py
    # driver corresponds to 0 in /opt/rocm/bin/rocm_smi.py
    if cuda:
        return ['driver']
    else:
        return [0]

def getversion(devicenum, component, cuda, smi=None):
    if cuda:
        if component == 'driver':
            cmd = [get_smi_exec(cuda), "--query-gpu=driver_version", "--format=csv,noheader", "-i", str(devicenum)]
            success, cout = _subprocess_helper(cmd)
            if not success:
                return "N/A"
            return cout
        else:
            return "N/A"
    elif smi is not None:
        return smi.getVersion([devicenum], component)
