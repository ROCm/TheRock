# AMDGPU_LLVM_TARGET
# Information from: [
#     https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
#     https://www.techpowerup.com/gpu-specs/
#   ]
# Note: New devices should be added to this list as they become available.

# Add list: From Radeon -> Radeon Pro -> Radeon MI / CDNA.


_amdgpu = {
    # RDNA 3
    "gfx1100": [
        "AMD Radeon RX 7900 XTX",
        "AMD Radeon RX 7900 XT",
        "AMD Radeon RX 7900 GRE",
        "AMD Radeon PRO W7900 Dual Slot",
        "AMD Radeon PRO W7900",
        "AMD Radeon PRO W7800 48GB",
        "AMD Radeon PRO W7800",
    ],
    "gfx1101": [
        "AMD Radeon RX 7800 XT",
        "AMD Radeon RX 7800",
        "AMD Radeon RX 7700 XT",
        "AMD Radeon RX 7700",
        "AMD Radeon PRO W7700",
        "AMD Radeon PRO V710",
    ],
    "gfx1102": [
        "AMD Radeon RX 7700S",
        "AMD Radeon RX 7650 GRE",
        "AMD Radeon RX 7600 XT",
        "AMD Radeon RX 7600",
        "AMD Radeon RX 7400",
    ],
    "gfx1103": [
        "AMD Radeon 780M",
    ],
    # RDNA 3.5
    "gfx1151": [
        "AMD Strix Halo",
    ],
    "gfx1150": [
        "AMD Strix Point",
    ],
    "gfx1152": [
        " AMD Krackan Point",
    ],
    # RDNA 4
    "gfx1201": [
        "AMD Radeon RX 9070 XT",
        "AMD Radeon RX 9070 GRE",
        "AMD Radeon RX 9070",
        "AMD Radeon AI PRO R9700",
    ],
    "gfx1200": [
        "AMD Radeon RX 9060 XT",
        "AMD Radeon RX 9060 XT 8GB",
        "AMD Radeon RX 9060",
    ],
    # RDNA 2
    "gfx1030": [
        "AMD Radeon RX 6950 XT",
        "AMD Radeon RX 6900 XT",
        "AMD Radeon RX 6800 XT",
        "AMD Radeon PRO W6800",
        "AMD Radeon PRO V620",
    ],
    "gfx1031": [
        "AMD Radeon RX 6750 XT",
        "AMD Radeon RX 6700 XT",
        "AMD Radeon RX 6700",
    ],
    "gfx1032": [
        "AMD Radeon RX 6650 XT",
        "AMD Radeon RX 6600 XT",
        "AMD Radeon RX 6600",
        "AMD Radeon PRO W6600",
    ],
    "gfx1034": [
        "AMD Radeon RX 6500 XT",
        "AMD Radeon RX 6500",
        "AMD Radeon RX 6500 4GB",
        "AMD Radeon PRO W6400",
    ],
    # RDNA 1
    "gfx1010": [
        "AMD Radeon RX 5700 XT 50th Anniversary",
        "AMD Radeon RX 5700 XT",
        "AMD Radeon RX 5700",
        "AMD Radeon RX 5600 XT",
        "AMD Radeon RX 5600",
    ],
    "gfx1011": [
        "AMD Radeon RX 5500 XT",
        "AMD Radeon RX 5500",
    ],
    "gfx1012": ["AMD Radeon RX 5300 XT", "AMD Radeon Pro W5500"],
    # Radeon VEGA
    "gfx906": [
        "AMD Radeon VII",
    ],
}
