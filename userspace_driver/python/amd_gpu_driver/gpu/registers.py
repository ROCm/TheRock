"""Compute register offsets for AMD GPUs.

These are offsets from the SH_REG base (0x2C00 typically).
Used with PACKET3_SET_SH_REG to program compute dispatch state.

Register definitions from:
- AMD GCN3 ISA Architecture Manual
- AMDGPU kernel driver headers (gfx_v9_0.h, gfx_v10_0.h, gfx_v11_0.h)
"""

# Compute program address (low/high 32 bits of code pointer >> 8)
COMPUTE_PGM_LO = 0x2E0C
COMPUTE_PGM_HI = 0x2E0D

# Compute program resource registers
COMPUTE_PGM_RSRC1 = 0x2E12
COMPUTE_PGM_RSRC2 = 0x2E13
COMPUTE_PGM_RSRC3 = 0x2E2D  # GFX942 (gc_9_4_3) only

# User data SGPRs (for passing kernarg pointer, etc.)
COMPUTE_USER_DATA_0 = 0x2E40
COMPUTE_USER_DATA_1 = 0x2E41
COMPUTE_USER_DATA_2 = 0x2E42
COMPUTE_USER_DATA_3 = 0x2E43

# Thread/workgroup dimensions
COMPUTE_START_X = 0x2E04
COMPUTE_START_Y = 0x2E05
COMPUTE_START_Z = 0x2E06
COMPUTE_NUM_THREAD_X = 0x2E07
COMPUTE_NUM_THREAD_Y = 0x2E08
COMPUTE_NUM_THREAD_Z = 0x2E09

# Scratch/temporary ring
COMPUTE_TMPRING_SIZE = 0x2E18

# Resource limits
COMPUTE_RESOURCE_LIMITS = 0x2E15

# Dispatch initiator
COMPUTE_DISPATCH_INITIATOR = 0x2E00

# Restart coordinates
COMPUTE_RESTART_X = 0x2E1B
COMPUTE_RESTART_Y = 0x2E1C
COMPUTE_RESTART_Z = 0x2E1D

# Shader control
COMPUTE_SHADER_CHKSUM = 0x2E2C  # GFX942 (gc_9_4_3) only
COMPUTE_MISC_RESERVED = 0x2E1F


def sh_reg_offset(reg: int) -> int:
    """Convert absolute register address to offset from SH_REG_BASE (0x2C00).

    For use with PACKET3_SET_SH_REG, which takes an offset relative to
    the SH_REG_BASE.
    """
    return reg - 0x2C00
