# build_tools/packaging/linux/utils.py
import re
import logging

# Create a common logger
logger = logging.getLogger("rocm_installer")
logger.setLevel(logging.INFO)  # default level

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add handler if not already added
if not logger.hasHandlers():
    logger.addHandler(ch)


def get_os_id(os_release_path="/etc/os-release"):
    """Detect OS family (debian/redhat/suse/unknown)."""
    os_release = {}
    try:
        with open("/etc/os-release", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    os_release[k] = v.strip('"')
    except FileNotFoundError:
        system_name = platform.system().lower()
        return "linux" if "linux" in system_name else "unknown"

    os_id = os_release.get("ID", "").lower()
    os_like = os_release.get("ID_LIKE", "").lower()

    if "ubuntu" in os_id or "debian" in os_like:
        print("returning debian")
        return "debian"
    elif any(x in os_id for x in ["rhel", "centos"]) or "redhat" in os_like:
        return "redhat"
    elif "suse" in os_id or "sles" in os_id:
        return "suse"
    else:
        return "unknown"

