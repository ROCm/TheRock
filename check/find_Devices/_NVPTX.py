from utils import *


_nvptx = {
    # --- Pascal (sm_60: HPC / sm_61: Consumer) ---
    "sm_60": ["NVIDIA Tesla P100"],
    "sm_61": [
        "NVIDIA TITAN X",
        "NVIDIA GeForce GTX 1080 Ti",
        "NVIDIA GeForce GTX 1080",
        "NVIDIA GeForce GTX 1070 Ti",
        "NVIDIA GeForce GTX 1070",
        "NVIDIA GeForce GTX 1060",
        "NVIDIA GeForce GTX 1050 Ti",
        "NVIDIA GeForce GTX 1050",
        "NVIDIA Tesla P40",
        "NVIDIA Tesla P4",
    ],
    # --- Volta (sm_70) ---
    "sm_70": ["NVIDIA Tesla V100", "NVIDIA TITAN V", "NVIDIA Quadro GV100"],
    # --- Turing (sm_75) ---
    "sm_75": [
        "NVIDIA TITAN RTX",
        "NVIDIA GeForce RTX 2080 Ti",
        "NVIDIA GeForce RTX 2080",
        "NVIDIA GeForce RTX 2070",
        "NVIDIA GeForce RTX 2060",
        "NVIDIA GeForce GTX 1660 Ti",
        "NVIDIA GeForce GTX 1660",
        "NVIDIA GeForce GTX 1650",
        "NVIDIA Tesla T4",
        "NVIDIA Quadro RTX 8000",
        "NVIDIA Quadro RTX 6000",
        "NVIDIA Quadro RTX 5000",
        "NVIDIA Quadro RTX 4000",
    ],
    # --- Ampere (sm_80: HPC A100 Only) ---
    "sm_80": ["NVIDIA A100", "NVIDIA A800"],
    # --- Ampere (sm_86: Consumer & Cloud Graphics) ---
    "sm_86": [
        "NVIDIA GeForce RTX 3090 Ti",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 3080 Ti",
        "NVIDIA GeForce RTX 3080",
        "NVIDIA GeForce RTX 3070 Ti",
        "NVIDIA GeForce RTX 3070",
        "NVIDIA GeForce RTX 3060 Ti",
        "NVIDIA GeForce RTX 3060",
        "NVIDIA GeForce RTX 3050",
        "NVIDIA RTX A6000",
        "NVIDIA RTX A5000",
        "NVIDIA RTX A4000",
        "NVIDIA A40",
        "NVIDIA A16",
        "NVIDIA A10",
        "NVIDIA A2",
    ],
    # --- Ada Lovelace (sm_89) ---
    "sm_89": [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 4080",
        "NVIDIA GeForce RTX 4070",
        "NVIDIA GeForce RTX 4060",
        "NVIDIA RTX 6000 Ada",
        "NVIDIA L40S",
        "NVIDIA L40",
        "NVIDIA L4",
    ],
    # --- Hopper (sm_90) ---
    "sm_90": ["NVIDIA H100", "NVIDIA H200", "NVIDIA H800", "NVIDIA GH200"],
    # --- Blackwell (sm_100) ---
    "sm_100": [
        # Data Center & HPC
        "NVIDIA B200",
        "NVIDIA B100",
        "NVIDIA GB200",
        "NVIDIA GB10",  # DGX Spark
        "NVIDIA DGX Spark",
        "NVIDIA RTX 6000 Blackwell",
        "NVIDIA RTX Pro 6000 Blackwell",
        "NVIDIA GeForce RTX 5090",
        "NVIDIA GeForce RTX 5080",
        "NVIDIA GeForce RTX 5070",
    ],
}


def nvptx_sm_name(GPU: str):

    query_name = GPU.lower().strip()

    for arch, models in _nvptx.items():
        for model in models:
            if model.lower() in query_name:
                return arch

    return NOTDEFINED
